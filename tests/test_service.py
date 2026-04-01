import json
from pathlib import Path

from fastapi.testclient import TestClient

from resupply_engine.api import create_app
from resupply_engine.llm_recommender import FakeJSONBackend, LocalLLMRecommender, OllamaBackend
from resupply_engine.config import Settings
from resupply_engine.ingest import DispatchIngestCoordinator, FileDropJsonIngestAdapter
from resupply_engine.models import PlanningRequest, SoftwareDecisionSupportPayload, SupplyNeed
from resupply_engine.normalize import normalize_payload
from resupply_engine.packing import pack_supply_needs
from resupply_engine.redundancy import calculate_redundancy_multiplier
from resupply_engine.service import DispatchPlanningService, build_service
from resupply_engine.workbook_policy import compile_clinical_workbook


def make_settings(tmp_path: Path) -> Settings:
    base_dir = Path(__file__).resolve().parent.parent
    return Settings(
        base_dir=base_dir,
        data_dir=base_dir / "data",
        db_path=tmp_path / "resupply-test.db",
        exports_dir=tmp_path / "exports",
    )


def make_client(tmp_path: Path, service: DispatchPlanningService | None = None) -> TestClient:
    settings = make_settings(tmp_path)
    app = create_app(settings=settings, service=service)
    return TestClient(app)


def sample_patient_payload(notes: str | None = None, march_flags: list[str] | None = None) -> dict:
    return {
        "patient_id": "PT-2847",
        "mission_id": "MSN-0042",
        "medic_id": "PJ-14",
        "gcs_total": 7,
        "gcs_eye": 2,
        "gcs_verbal": 2,
        "gcs_motor": 3,
        "bp_systolic": 162,
        "bp_diastolic": 94,
        "heart_rate": 52,
        "oxygen_saturation": 91,
        "temp_c": 37.0,
        "seizure": False,
        "vomiting": True,
        "head_external_hemorrhage": True,
        "suspected_icp": True,
        "location": "FRONT-LEFT",
        "march_flags": march_flags if march_flags is not None else ["airway_compromised", "circulatory_compromise"],
        "notes": notes,
        "timestamp": "2034-03-16T14:23:00Z",
    }


def sample_payload(notes: str | None = None, march_flags: list[str] | None = None) -> dict:
    return {
        "shootdown_rate": 25,
        "target_arrival_probability": 0.95,
        "burst_id": "BST-20340316-0001",
        "software_decision_support_payload": sample_patient_payload(notes=notes, march_flags=march_flags),
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
    assert first_json["burst_id"] == "BST-20340316-0001"
    assert first_json["canonical_case"]["source"]["received_via"] == "http_json"
    assert first_json["canonical_case"]["extra"]["ingest_context"]["source_locator"] == "/v1/plans"
    assert first_json["canonical_case"]["mission_id"] == "MSN-0042"
    assert first_json["canonical_case"]["march_flags"] == ["airway_compromised", "circulatory_compromise"]
    assert {item["item_id"] for item in first_json["required_supplies"]} == {
        "tbi_detection_monitoring_kit",
        "airway_support_kit",
        "hemostatic_dressing_bundle",
        "hypertonic_saline_kit",
        "infrared_intracranial_hemorrhage_device",
        "iv_fluid_bundle",
        "ketamine_analgesia_kit",
    }
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
                    "items": [{"item_id": "tbi_detection_monitoring_kit", "quantity": 1}],
                },
                {
                    "manifest_id": 2,
                    "items": [{"item_id": "iv_fluid_bundle", "quantity": 1}],
                },
                {
                    "manifest_id": 3,
                    "items": [
                        {"item_id": "airway_support_kit", "quantity": 1},
                        {"item_id": "hemostatic_dressing_bundle", "quantity": 2},
                    ],
                },
                {
                    "manifest_id": 4,
                    "items": [{"item_id": "hypertonic_saline_kit", "quantity": 1}],
                },
            ],
            "operator_notes": "Grouped hemorrhage control separately from fluids.",
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


def test_workbook_compiles_against_catalog(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    service = build_service(settings)

    compiled = compile_clinical_workbook(settings.clinical_workbook_path, service.catalog)

    assert len(compiled.item_rules) >= 10
    assert "tbi_detection_monitoring_kit" in compiled.priority_ranks
    assert any(rule.logic_type == "ALWAYS" for rule in compiled.item_rules)


def test_new_payload_shape_is_accepted_and_old_shape_is_rejected(tmp_path: Path) -> None:
    client = make_client(tmp_path)

    accepted = client.post("/v1/plans", json=sample_payload())
    assert accepted.status_code == 200

    legacy_request = {
        "shootdown_rate": 25,
        "target_arrival_probability": 0.95,
        "payload": {
            "burst_id": "BST-OLD-1",
            "patient_id": "PT-OLD",
            "case_id": "CASE-OLD",
            "mission_id": "MSN-OLD",
            "medic_id": "PJ-OLD",
            "timestamp_utc": "2034-03-16T14:23:00Z",
            "time_since_injury_min": 45,
            "casualty_count": 1,
            "priority_flag": "URGENT_SURGICAL",
            "vitals": {"blood_pressure_systolic": 120},
            "injury": {"tbi_severity": "SEVERE"},
        },
    }
    rejected = client.post("/v1/plans", json=legacy_request)
    assert rejected.status_code == 422


def test_medic_side_shootdown_is_ignored_and_top_level_value_drives_plan(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    payload = sample_payload()
    payload["software_decision_support_payload"]["shootdown"] = 90

    response = client.post("/v1/plans", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["shootdown_rate"] == 25


def test_example_and_incoming_payload_files_validate() -> None:
    base_dir = Path(__file__).resolve().parent.parent / "data"

    incoming = json.loads((base_dir / "incoming_payload.json").read_text())
    validated_incoming = SoftwareDecisionSupportPayload.model_validate(incoming)
    assert validated_incoming.patient_id == "PT-2847"
    assert validated_incoming.gcs_total == 7

    planning_request = json.loads((base_dir / "example_payload.json").read_text())
    validated_request = PlanningRequest.model_validate(planning_request)
    assert validated_request.burst_id == "BST-20340401-0002"
    assert validated_request.software_decision_support_payload.patient_id == "PT-9990"


def test_normalization_maps_flat_payload_and_preserves_context() -> None:
    payload = SoftwareDecisionSupportPayload.model_validate(sample_patient_payload())
    canonical_case = normalize_payload(payload, burst_id="BST-20340316-0001")

    assert canonical_case.reported_at.isoformat() == "2034-03-16T14:23:00+00:00"
    assert canonical_case.vitals.temperature_c == 37.0
    assert canonical_case.vitals.gcs == 7
    assert canonical_case.vitals.gcs_eye == 2
    assert canonical_case.vitals.diastolic_bp == 94
    assert canonical_case.mission_id == "MSN-0042"
    assert canonical_case.location == "FRONT-LEFT"
    assert canonical_case.march_flags == ["airway_compromised", "circulatory_compromise"]
    assert canonical_case.extra["raw_assessment"]["gcs_total"] == 7
    assert canonical_case.extra["raw_assessment"]["suspected_icp"] is True
    assert canonical_case.symptoms == [
        "airway_compromise",
        "circulatory_compromise",
        "elevated_icp",
        "hemorrhage",
        "severe_tbi",
        "vomiting",
    ]


def test_supported_march_flags_trigger_direct_rules(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    scenarios = {
        "airway_compromised": "airway_support_kit",
        "respiratory_compromise": "airway_support_kit",
        "massive_hemorrhage": "hemostatic_dressing_bundle",
        "circulatory_compromise": "iv_fluid_bundle",
        "hypothermia": "iv_fluid_bundle",
    }

    for flag, expected_item in scenarios.items():
        payload = sample_payload(march_flags=[flag])
        profile = payload["software_decision_support_payload"]
        profile["gcs_total"] = 15
        profile["gcs_eye"] = 4
        profile["gcs_verbal"] = 5
        profile["gcs_motor"] = 6
        profile["seizure"] = False
        profile["vomiting"] = False
        profile["head_external_hemorrhage"] = False
        profile["suspected_icp"] = False
        profile["notes"] = None

        response = client.post("/v1/plans", json=payload)
        assert response.status_code == 200
        item_ids = {item["item_id"] for item in response.json()["required_supplies"]}
        assert expected_item in item_ids


def test_workbook_always_rule_includes_monitoring_kit(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    payload = sample_payload(march_flags=[])
    profile = payload["software_decision_support_payload"]
    profile["gcs_total"] = 15
    profile["gcs_eye"] = 4
    profile["gcs_verbal"] = 5
    profile["gcs_motor"] = 6
    profile["seizure"] = False
    profile["vomiting"] = False
    profile["head_external_hemorrhage"] = False
    profile["suspected_icp"] = False
    profile["notes"] = None

    response = client.post("/v1/plans", json=payload)

    assert response.status_code == 200
    body = response.json()
    item_ids = {item["item_id"] for item in body["required_supplies"]}
    assert "tbi_detection_monitoring_kit" in item_ids


def test_llm_malformed_json_falls_back_to_workbook_policy(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    settings = Settings(
        base_dir=settings.base_dir,
        data_dir=settings.data_dir,
        db_path=settings.db_path,
        exports_dir=settings.exports_dir,
        clinical_workbook_path=settings.clinical_workbook_path,
        llm_enabled=True,
    )
    service = DispatchPlanningService(
        settings=settings,
        recommender=LocalLLMRecommender(FakeJSONBackend("not-json")),
    )
    client = make_client(tmp_path, service=service)

    response = client.post("/v1/plans", json=sample_payload())

    assert response.status_code == 200
    body = response.json()
    assert any(flag["code"] == "llm_unavailable_fallback" for flag in body["review_flags"])
    assert any(item["item_id"] == "tbi_detection_monitoring_kit" for item in body["required_supplies"])


def test_llm_output_is_repaired_and_unknown_items_removed(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    settings = Settings(
        base_dir=settings.base_dir,
        data_dir=settings.data_dir,
        db_path=settings.db_path,
        exports_dir=settings.exports_dir,
        clinical_workbook_path=settings.clinical_workbook_path,
        llm_enabled=True,
    )
    llm_json = json.dumps(
        {
            "recommended_items": [
                {
                    "item_id": "airway_support_kit",
                    "quantity": 1,
                    "rationale": "LLM recommended airway support.",
                    "policy_refs": ["workbook-rule-2-airway_support_kit"],
                    "confidence": 0.82,
                },
                {
                    "item_id": "unknown_kit",
                    "quantity": 1,
                    "rationale": "Invalid item.",
                    "policy_refs": [],
                    "confidence": 0.1,
                },
            ],
            "overall_confidence": 0.82,
            "manual_review_notes": "Review for omitted critical items.",
        }
    )
    service = DispatchPlanningService(
        settings=settings,
        recommender=LocalLLMRecommender(FakeJSONBackend(llm_json)),
    )
    client = make_client(tmp_path, service=service)

    response = client.post("/v1/plans", json=sample_payload())

    assert response.status_code == 200
    body = response.json()
    item_ids = {item["item_id"] for item in body["required_supplies"]}
    assert "unknown_kit" not in item_ids
    assert "airway_support_kit" in item_ids
    assert "tbi_detection_monitoring_kit" in item_ids
    assert any(flag["code"] == "llm_item_removed" for flag in body["review_flags"])
    assert any(flag["code"] == "llm_missing_mandatory_item" for flag in body["review_flags"])


def test_ollama_backend_parses_local_generate_response(monkeypatch) -> None:
    backend = OllamaBackend(base_url="http://127.0.0.1:11434/api", model="qwen2.5:1.5b")

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self) -> bytes:
            return b'{"response":"{\\"recommended_items\\":[],\\"overall_confidence\\":0.7,\\"manual_review_notes\\":\\"\\"}"}'

    monkeypatch.setattr("resupply_engine.llm_recommender.request.urlopen", lambda req, timeout: FakeResponse())

    raw = backend.generate("hello")

    assert '"recommended_items"' in raw


def test_service_builds_ollama_recommender_by_default_when_llm_enabled(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    settings = Settings(
        base_dir=settings.base_dir,
        data_dir=settings.data_dir,
        db_path=settings.db_path,
        exports_dir=settings.exports_dir,
        clinical_workbook_path=settings.clinical_workbook_path,
        llm_enabled=True,
    )

    service = build_service(settings)

    assert isinstance(service.recommender.backend, OllamaBackend)


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
    assert "GCS Total: 7" in export_text
    assert "MARCH Flags: airway_compromised, circulatory_compromise" in export_text
    assert "Required Supplies:" in export_text
    assert "Base Manifest:" in export_text
    assert "Redundancy multiplier:" in export_text
    assert "Per item arrival probability:" in export_text
    assert "Total drones to be sent:" in export_text
    assert '"patient_id"' not in export_text
