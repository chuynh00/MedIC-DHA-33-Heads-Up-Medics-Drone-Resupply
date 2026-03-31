from __future__ import annotations

from resupply_engine.models import CanonicalCase, SoftwareDecisionSupportPayload, VitalSigns


def _canonicalize_token(value: str) -> str:
    return value.strip().lower().replace(" ", "_").replace("-", "_")


def _derive_symptoms(payload: SoftwareDecisionSupportPayload) -> list[str]:
    symptoms: set[str] = set()

    if payload.gcs_total <= 8:
        symptoms.add("severe_tbi")
    if payload.seizure:
        symptoms.add("seizure")
    if payload.vomiting:
        symptoms.add("vomiting")
    if payload.head_external_hemorrhage:
        symptoms.add("hemorrhage")
    if payload.suspected_icp:
        symptoms.add("elevated_icp")

    for flag in payload.march_flags:
        normalized = _canonicalize_token(flag)
        if normalized == "airway_compromised":
            symptoms.add("airway_compromise")
        elif normalized == "massive_hemorrhage":
            symptoms.add("hemorrhage")
        else:
            symptoms.add(normalized)

    return sorted(symptoms)


def _normalize_vitals(payload: SoftwareDecisionSupportPayload) -> VitalSigns:
    return VitalSigns(
        gcs=payload.gcs_total,
        gcs_eye=payload.gcs_eye,
        gcs_verbal=payload.gcs_verbal,
        gcs_motor=payload.gcs_motor,
        heart_rate=payload.heart_rate,
        spo2=payload.oxygen_saturation,
        systolic_bp=payload.bp_systolic,
        diastolic_bp=payload.bp_diastolic,
        temperature_c=payload.temp_c,
    )


def normalize_payload(payload: SoftwareDecisionSupportPayload, burst_id: str) -> CanonicalCase:
    return CanonicalCase(
        burst_id=burst_id,
        patient_id=payload.patient_id,
        mission_id=payload.mission_id,
        medic_id=payload.medic_id,
        reported_at=payload.timestamp,
        seizure=payload.seizure,
        vomiting=payload.vomiting,
        head_external_hemorrhage=payload.head_external_hemorrhage,
        suspected_icp=payload.suspected_icp,
        location=payload.location.strip() if payload.location else None,
        march_flags=payload.march_flags,
        symptoms=_derive_symptoms(payload),
        vitals=_normalize_vitals(payload),
        notes=payload.notes.strip() if payload.notes else None,
        source=payload.source,
        extra={
            **payload.extra,
            "raw_assessment": payload.model_dump(mode="json", exclude={"source", "extra"}),
        },
    )
