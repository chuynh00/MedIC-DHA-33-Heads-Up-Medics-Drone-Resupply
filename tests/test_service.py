import json
from pathlib import Path

from fastapi.testclient import TestClient

from resupply_engine.api import create_app
from resupply_engine.config import Settings
from resupply_engine.ingest import DispatchIngestCoordinator, FileDropJsonIngestAdapter
from resupply_engine.models import PlanningRequest, SupplyNeed, TbiBurstPayload
from resupply_engine.normalize import normalize_payload
from resupply_engine.packing import pack_supply_needs
from resupply_engine.redundancy import calculate_redundancy_multiplier
from resupply_engine.service import build_service


def make_settings(tmp_path: Path) -> Settings:
    base_dir = Path(__file__).resolve().parent.parent
    return Settings(
        base_dir=base_dir,
        data_dir=base_dir / "data",
        db_path=tmp_path / "resupply-test.db",
        exports_dir=tmp_path / "exports",
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
            "burst_id": "BST-20340316-0001",
            "patient_id": "PT-2847",
            "case_id": "CASE-9182",
            "mission_id": "MSN-0042",
            "medic_id": "PJ-14",
            "timestamp_utc": "2034-03-16T14:23:00Z",
            "time_since_injury_min": 45,
            "casualty_count": 1,
            "priority_flag": "URGENT_SURGICAL",
            "evacuation_eta_hours": 36,
            "risk_score": 82,
            "notes": notes,
            "vitals": {
                "blood_pressure_systolic": 162,
                "blood_pressure_diastolic": 94,
                "heart_rate_bpm": 52,
                "respiratory_rate_rpm": 8,
                "spo2_pct": 91,
                "temperature_f": 98.6,
            },
            "neuro_exam": {
                "gcs_total": 7,
                "gcs_eye": 2,
                "gcs_verbal": 2,
                "gcs_motor": 3,
                "loc_duration": ">30min",
                "pupil_response": "UNEQUAL",
                "vomiting": True,
                "seizure": False,
                "suspected_icp": "YES",
            },
            "injury": {"mechanism": "BLAST", "tbi_severity": "SEVERE", "hemorrhage": "YES"},
            "airway_status": "COMPROMISED",
            "treatment_given": [
                "airway_repositioned",
                "supplemental_oxygen",
                "iv_access_established",
                "head_elevated_30_degrees",
            ],
            "requested_supplies": ["hpt_saline", "whole_blood", "auto_burr_hole_kit"],
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
    assert first_json["total_drones"] == 12
    assert first_json["manual_review_required"] is False
    assert first_json["canonical_case"]["source"]["received_via"] == "http_json"
    assert first_json["canonical_case"]["extra"]["ingest_context"]["source_locator"] == "/v1/plans"
    assert first_json["canonical_case"]["mission_id"] == "MSN-0042"
    assert first_json["canonical_case"]["priority_flag"] == "URGENT_SURGICAL"
    assert first_json["export"]["export_format"] == "txt"
    assert first_json["export"]["export_revision"] == 1
    assert first_json["export"]["export_path"].endswith("__rev-001.txt")
    assert Path(first_json["export"]["export_path"]).exists()


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
                    "items": [{"item_id": "ketamine_analgesia_kit", "quantity": 1}],
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
    assert body["export"]["export_revision"] == 2
    assert body["export"]["export_path"].endswith("__rev-002.txt")
    assert Path(body["export"]["export_path"]).exists()


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
    assert plan.export.export_path is not None


def test_example_payload_is_accepted_as_is() -> None:
    planning_request = json.loads((Path(__file__).resolve().parent.parent / "data" / "example_payload.json").read_text())
    validated = TbiBurstPayload.model_validate(planning_request["payload"])
    assert validated.burst_id == "BST-20340316-0001"
    assert validated.case_id == "CASE-9182"
    assert validated.injury.tbi_severity == "SEVERE"


def test_invalid_tbi_enum_values_are_rejected() -> None:
    payload = sample_payload()["payload"]
    payload["neuro_exam"]["pupil_response"] = "BROKEN"
    try:
        TbiBurstPayload.model_validate(payload)
    except Exception as exc:
        assert "pupil_response" in str(exc)
    else:
        raise AssertionError("Expected invalid enum value to be rejected.")


def test_normalization_derives_symptoms_and_preserves_context() -> None:
    payload = TbiBurstPayload.model_validate(sample_payload()["payload"])
    canonical_case = normalize_payload(payload)

    assert canonical_case.occurred_at.isoformat() == "2034-03-16T14:23:00+00:00"
    assert canonical_case.vitals.temperature_c == 37.0
    assert canonical_case.vitals.gcs == 7
    assert canonical_case.vitals.gcs_eye == 2
    assert canonical_case.vitals.diastolic_bp == 94
    assert canonical_case.mission_id == "MSN-0042"
    assert canonical_case.time_since_injury_min == 45
    assert canonical_case.casualty_count == 1
    assert canonical_case.requested_supplies == ["hpt_saline", "whole_blood", "auto_burr_hole_kit"]
    assert canonical_case.treatment_given == [
        "airway_repositioned",
        "supplemental_oxygen",
        "iv_access_established",
        "head_elevated_30_degrees",
    ]
    assert canonical_case.extra["raw_assessment"]["injury"]["tbi_severity"] == "SEVERE"
    assert canonical_case.extra["raw_assessment"]["neuro_exam"]["suspected_icp"] == "YES"
    assert canonical_case.symptoms == ["airway_compromise", "elevated_icp", "hemorrhage", "severe_tbi", "vomiting"]


def test_text_export_contains_operator_fields(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    response = client.post("/v1/plans", json=sample_payload())

    assert response.status_code == 200
    body = response.json()
    export_text = Path(body["export"]["export_path"]).read_text(encoding="utf-8")

    assert "Patient ID: PT-2847" in export_text
    assert "Mission ID: MSN-0042" in export_text
    assert "Medic ID: PJ-14" in export_text
    assert "This plan is the response to burst ID: BST-20340316-0001" in export_text
    assert "Shootdown rate: 25%" in export_text
    assert "Target arrival probability: 95.00%" in export_text
    assert "Burst timestamp_utc: 2034-03-16T14:23:00+00:00" in export_text
    assert "Required Supplies:" in export_text
    assert "Base Manifest:" in export_text
    assert "Redundancy multiplier:" in export_text
    assert "Per item arrival probability:" in export_text
    assert "Total drones to be sent:" in export_text
    assert '"patient_id"' not in export_text
