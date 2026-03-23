from __future__ import annotations

from resupply_engine.models import BurstPayload, CanonicalCase


def _canonicalize_symptom(symptom: str) -> str:
    return symptom.strip().lower().replace(" ", "_").replace("-", "_")


def normalize_payload(payload: BurstPayload) -> CanonicalCase:
    normalized_symptoms = sorted(
        {_canonicalize_symptom(symptom) for symptom in payload.symptoms if symptom.strip()}
    )
    return CanonicalCase(
        burst_id=payload.burst_id,
        patient_id=payload.patient_id,
        case_id=payload.case_id,
        medic_id=payload.medic_id,
        occurred_at=payload.occurred_at,
        evacuation_eta_hours=payload.evacuation_eta_hours,
        symptoms=normalized_symptoms,
        vitals=payload.vitals,
        risk_score=payload.risk_score,
        notes=payload.notes.strip() if payload.notes else None,
        source=payload.source,
        extra=payload.extra,
    )
