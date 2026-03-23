from __future__ import annotations

from resupply_engine.models import ExtractionResult


KEYWORD_TO_SYMPTOM = {
    "seizure": "seizure",
    "convulsion": "seizure",
    "airway": "airway_compromise",
    "hemorrhage": "hemorrhage",
    "bleeding": "hemorrhage",
    "tbi": "severe_tbi",
    "head injury": "severe_tbi",
    "anisocoria": "severe_tbi",
    "brain swelling": "elevated_icp",
    "unequal pupils": "severe_tbi",
}


def extract_symptoms_from_notes(notes: str | None) -> ExtractionResult:
    if not notes:
        return ExtractionResult()

    lowered = notes.lower()
    matches = sorted({symptom for keyword, symptom in KEYWORD_TO_SYMPTOM.items() if keyword in lowered})
    if not matches:
        return ExtractionResult(
            extracted_symptoms=[],
            confidence=0.35,
            explanation="Free-text note did not map cleanly to the bounded symptom schema.",
        )

    confidence = min(0.95, 0.55 + 0.12 * len(matches))
    return ExtractionResult(
        extracted_symptoms=matches,
        confidence=confidence,
        explanation="Keyword-based local extraction populated bounded symptoms from free text.",
    )
