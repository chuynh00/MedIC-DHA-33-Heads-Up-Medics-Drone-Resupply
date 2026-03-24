from __future__ import annotations

from resupply_engine.models import CanonicalCase, TbiBurstPayload, VitalSigns


def _canonicalize_symptom(symptom: str) -> str:
    return symptom.strip().lower().replace(" ", "_").replace("-", "_")


def _fahrenheit_to_celsius(temperature_f: float | None) -> float | None:
    if temperature_f is None:
        return None
    return round((temperature_f - 32) * 5 / 9, 2)


def _derive_symptoms(payload: TbiBurstPayload) -> list[str]:
    symptoms: set[str] = set()
    if payload.injury.tbi_severity in {"SEVERE", "PENETRATING"}:
        symptoms.add("severe_tbi")
    if payload.airway_status and payload.airway_status.strip().upper() == "COMPROMISED":
        symptoms.add("airway_compromise")
    if payload.neuro_exam is not None:
        if payload.neuro_exam.seizure is True:
            symptoms.add("seizure")
        if payload.neuro_exam.vomiting is True:
            symptoms.add("vomiting")
        if payload.neuro_exam.suspected_icp == "YES":
            symptoms.add("elevated_icp")
        if payload.neuro_exam.pupil_response in {"UNEQUAL", "NON_REACTIVE"}:
            symptoms.add("severe_tbi")
        if payload.neuro_exam.gcs_total is not None and payload.neuro_exam.gcs_total <= 8:
            symptoms.add("severe_tbi")
        if payload.injury.hemorrhage == "YES":
            symptoms.add("hemorrhage")
    return sorted(_canonicalize_symptom(symptom) for symptom in symptoms)


def _normalize_vitals(payload: TbiBurstPayload) -> VitalSigns:
    neuro_exam = payload.neuro_exam
    return VitalSigns(
        gcs=neuro_exam.gcs_total if neuro_exam is not None else None,
        gcs_eye=neuro_exam.gcs_eye if neuro_exam is not None else None,
        gcs_verbal=neuro_exam.gcs_verbal if neuro_exam is not None else None,
        gcs_motor=neuro_exam.gcs_motor if neuro_exam is not None else None,
        heart_rate=payload.vitals.heart_rate_bpm,
        respiratory_rate=payload.vitals.respiratory_rate_rpm,
        spo2=payload.vitals.spo2_pct,
        systolic_bp=payload.vitals.blood_pressure_systolic,
        diastolic_bp=payload.vitals.blood_pressure_diastolic,
        temperature_c=_fahrenheit_to_celsius(payload.vitals.temperature_f),
    )


def normalize_payload(payload: TbiBurstPayload) -> CanonicalCase:
    normalized_symptoms = _derive_symptoms(payload)
    return CanonicalCase(
        burst_id=payload.burst_id,
        patient_id=payload.patient_id,
        case_id=payload.case_id,
        mission_id=payload.mission_id,
        medic_id=payload.medic_id,
        occurred_at=payload.timestamp_utc,
        time_since_injury_min=payload.time_since_injury_min,
        casualty_count=payload.casualty_count,
        priority_flag=payload.priority_flag,
        airway_status=payload.airway_status,
        treatment_given=payload.treatment_given,
        requested_supplies=payload.requested_supplies,
        evacuation_eta_hours=payload.evacuation_eta_hours,
        symptoms=normalized_symptoms,
        vitals=_normalize_vitals(payload),
        risk_score=payload.risk_score,
        notes=payload.notes.strip() if payload.notes else None,
        source=payload.source,
        extra={
            **payload.extra,
            "raw_assessment": {
                "vitals": payload.vitals.model_dump(mode="json"),
                "neuro_exam": payload.neuro_exam.model_dump(mode="json") if payload.neuro_exam else None,
                "injury": payload.injury.model_dump(mode="json"),
            },
        },
    )
