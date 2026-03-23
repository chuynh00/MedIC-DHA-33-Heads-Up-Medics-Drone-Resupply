from pathlib import Path

from fastapi.testclient import TestClient

from resupply_engine.api import create_app
from resupply_engine.config import Settings
from resupply_engine.ingest import DispatchIngestCoordinator, FileDropJsonIngestAdapter
from resupply_engine.models import PlanningRequest
from resupply_engine.models import SupplyNeed
from resupply_engine.packing import pack_supply_needs
from resupply_engine.redundancy import calculate_redundancy_multiplier
from resupply_engine.service import build_service


def make_settings(tmp_path: Path) -> Settings:
    base_dir = Path(__file__).resolve().parent.parent
    return Settings(
        base_dir=base_dir,
        data_dir=base_dir / "data",
        db_path=tmp_path / "resupply-test.db",
    )


def make_client(tmp_path: Path) -> TestClient:
    settings = make_settings(tmp_path)
    app = create_app(settings=settings)
    return TestClient(app)


def sample_payload(notes: str | None = None) -> dict:
    return {
        "shootdown_rate": 25,
        "target_arrival_probability": 0.95,
        "payload": {
            "burst_id": "burst-001",
            "patient_id": "patient-123",
            "case_id": "case-123",
            "medic_id": "medic-alpha",
            "occurred_at": "2026-03-20T20:00:00Z",
            "evacuation_eta_hours": 36,
            "symptoms": ["severe_tbi", "hemorrhage", "elevated_icp"],
            "risk_score": 82,
            "notes": notes,
            "vitals": {"gcs": 7, "spo2": 92, "systolic_bp": 96},
            "source": {"source_system": "medic-ai", "received_via": "local-rest"},
        },
    }


def test_create_plan_and_idempotent_retransmission(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    payload = sample_payload()

    first = client.post("/v1/plans", json=payload)
    second = client.post("/v1/plans", json=payload)

    assert first.status_code == 200
    assert second.status_code == 200
    first_json = first.json()
    second_json = second.json()
    assert first_json["plan_id"] == second_json["plan_id"]
    assert first_json["total_drones"] == 9
    assert first_json["manual_review_required"] is False
    assert first_json["canonical_case"]["source"]["received_via"] == "http_json"
    assert first_json["canonical_case"]["extra"]["ingest_context"]["source_locator"] == "/v1/plans"


def test_packing_example_uses_two_drones() -> None:
    needs = [
        SupplyNeed(item_id="item-a", name="Item A", quantity=1, unit_weight_lb=8.0),
        SupplyNeed(item_id="item-b", name="Item B", quantity=1, unit_weight_lb=6.0),
    ]

    manifests = pack_supply_needs(needs, 10.0)

    assert len(manifests) == 2
    assert sorted(manifest.total_weight_lb for manifest in manifests) == [6.0, 8.0]


def test_redundancy_supported_buckets() -> None:
    expected = {
        0: 1,
        10: 2,
        25: 3,
        50: 5,
        75: 11,
        90: 29,
    }
    for rate, expected_multiplier in expected.items():
        multiplier, probability = calculate_redundancy_multiplier(rate, 0.95)
        assert multiplier == expected_multiplier
        assert probability >= 0.95 or rate == 0


def test_high_risk_case_requires_manual_review_when_drone_cap_exceeded(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    payload = sample_payload()
    payload["shootdown_rate"] = 90

    response = client.post("/v1/plans", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["manual_review_required"] is True
    assert body["total_drones"] > 24
    assert any(flag["code"] == "exceeds_auto_dispatch_drone_cap" for flag in body["review_flags"])


def test_low_confidence_note_extraction_triggers_review(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    payload = sample_payload(notes="Patient deteriorating with unclear field shorthand zeta-9.")

    response = client.post("/v1/plans", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["manual_review_required"] is True
    assert any(flag["code"] == "low_confidence_note_extraction" for flag in body["review_flags"])


def test_operator_recalculate_with_manual_manifest(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    create_response = client.post("/v1/plans", json=sample_payload())
    plan_id = create_response.json()["plan_id"]

    recalc_response = client.post(
        f"/v1/plans/{plan_id}/recalculate",
        json={
            "base_manifests": [
                {
                    "manifest_id": 1,
                    "items": [{"item_id": "tbi_monitoring_kit", "quantity": 1}],
                },
                {
                    "manifest_id": 2,
                    "items": [
                        {"item_id": "iv_fluid_bundle", "quantity": 1},
                        {"item_id": "hemostatic_dressing_bundle", "quantity": 1},
                    ],
                },
                {
                    "manifest_id": 3,
                    "items": [
                        {"item_id": "hemostatic_dressing_bundle", "quantity": 1},
                        {"item_id": "hypertonic_saline_kit", "quantity": 1},
                    ],
                },
                {
                    "manifest_id": 4,
                    "items": [{"item_id": "analgesia_pack", "quantity": 1}],
                },
            ],
            "operator_notes": "Grouped hemorrhage control around fluid support.",
        },
    )

    assert recalc_response.status_code == 200
    body = recalc_response.json()
    assert body["plan_id"] == plan_id
    assert len(body["base_manifests"]) == 4
    assert any(flag["code"] == "operator_notes" for flag in body["review_flags"])


def test_decision_endpoint_freezes_status(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    create_response = client.post("/v1/plans", json=sample_payload())
    plan_id = create_response.json()["plan_id"]

    decision_response = client.post(
        f"/v1/plans/{plan_id}/decision",
        json={"decision": "approved", "operator_id": "operator-1", "notes": "Dispatch now."},
    )

    assert decision_response.status_code == 200
    assert decision_response.json()["status"] == "approved"


def test_file_drop_adapter_feeds_same_planner_core(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    service = build_service(settings)
    coordinator = DispatchIngestCoordinator(service)
    adapter = FileDropJsonIngestAdapter()
    planning_request = PlanningRequest.model_validate(sample_payload())

    plan = coordinator.submit(
        adapter.ingest(
            planning_request,
            source_path="/var/forward-base/inbox/burst-001.json",
        )
    )

    assert plan.canonical_case.source.received_via == "file_drop_json"
    assert plan.canonical_case.extra["ingest_context"]["source_locator"] == "/var/forward-base/inbox/burst-001.json"
