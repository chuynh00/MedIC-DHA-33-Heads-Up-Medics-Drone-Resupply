from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class VitalSigns(BaseModel):
    """Structured vital signs captured for a single patient case."""

    gcs: int | None = None
    gcs_eye: int | None = None
    gcs_verbal: int | None = None
    gcs_motor: int | None = None
    heart_rate: int | None = None
    respiratory_rate: int | None = None
    spo2: int | None = None
    systolic_bp: int | None = None
    diastolic_bp: int | None = None
    temperature_c: float | None = None


class SourceMetadata(BaseModel):
    """Metadata describing which upstream system or operator sent the case."""

    source_system: str = "unknown"
    received_via: str = "local-rest"
    operator_id: str | None = None
    source_message_type: str | None = None


class IncomingVitals(BaseModel):
    """Vitals as transmitted in the external TBI burst payload."""

    blood_pressure_systolic: int | None = None
    blood_pressure_diastolic: int | None = None
    heart_rate_bpm: int | None = None
    respiratory_rate_rpm: int | None = None
    spo2_pct: int | None = None
    temperature_f: float | None = None


class IncomingNeuroExam(BaseModel):
    """Structured neuro exam fields sent by the upstream assessment system."""

    gcs_total: int | None = None
    gcs_eye: int | None = None
    gcs_verbal: int | None = None
    gcs_motor: int | None = None
    loc_duration: Literal["NONE", "<30MIN", ">30MIN"] | None = None
    pupil_response: Literal["NORMAL", "UNEQUAL", "NON_REACTIVE"] | None = None
    vomiting: bool | None = None
    seizure: bool | None = None
    suspected_icp: Literal["YES", "NO", "UNKNOWN"] | None = None

    @field_validator("loc_duration", "pupil_response", "suspected_icp", mode="before")
    @classmethod
    def normalize_enum_fields(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip().upper()
        if normalized == "NONE":
            return "NONE"
        if normalized == "<30MIN":
            return "<30MIN"
        if normalized == ">30MIN":
            return ">30MIN"
        if normalized == "NON-REACTIVE":
            return "NON_REACTIVE"
        return normalized.replace(" ", "_")


class IncomingInjury(BaseModel):
    """Injury characterization fields sent in the external burst payload."""

    mechanism: str | None = None
    tbi_severity: Literal["MILD", "MODERATE", "SEVERE", "PENETRATING"]
    hemorrhage: Literal["YES", "NO", "UNKNOWN"] | None = None



class TbiBurstPayload(BaseModel):
    """Raw inbound TBI burst message before the app normalizes it."""

    burst_id: str
    patient_id: str
    case_id: str
    mission_id: str
    medic_id: str
    timestamp_utc: datetime
    time_since_injury_min: int
    casualty_count: int
    priority_flag: str
    vitals: IncomingVitals = Field(default_factory=IncomingVitals)
    neuro_exam: IncomingNeuroExam | None = None
    injury: IncomingInjury
    airway_status: str | None = None
    treatment_given: list[str] = Field(default_factory=list)
    requested_supplies: list[str] = Field(default_factory=list)
    evacuation_eta_hours: int | None = None
    risk_score: float | None = None
    notes: str | None = None
    source: SourceMetadata = Field(default_factory=SourceMetadata)
    extra: dict[str, Any] = Field(default_factory=dict)


class PlanningRequest(BaseModel):
    """Top-level request combining a patient burst with dispatch planning parameters."""

    shootdown_rate: int
    target_arrival_probability: float = 0.95
    payload: TbiBurstPayload

    @field_validator("target_arrival_probability")
    @classmethod
    def validate_probability(cls, value: float) -> float:
        if not 0 < value <= 1:
            raise ValueError("target_arrival_probability must be between 0 and 1")
        return value


class CanonicalCase(BaseModel):
    """Normalized internal patient case used by the recommendation engine."""

    burst_id: str
    patient_id: str
    case_id: str
    mission_id: str
    medic_id: str
    occurred_at: datetime
    time_since_injury_min: int
    casualty_count: int
    priority_flag: str
    airway_status: str | None = None
    treatment_given: list[str] = Field(default_factory=list)
    requested_supplies: list[str] = Field(default_factory=list)
    evacuation_eta_hours: int | None = None
    symptoms: list[str] = Field(default_factory=list)
    vitals: VitalSigns = Field(default_factory=VitalSigns)
    risk_score: float | None = None
    notes: str | None = None
    source: SourceMetadata = Field(default_factory=SourceMetadata)
    extra: dict[str, Any] = Field(default_factory=dict)


class SupplyCatalogItem(BaseModel):
    """One sendable supply item or bundle from the local catalog."""

    item_id: str
    name: str
    unit_weight_lb: float
    unit_of_issue: str = "each"
    is_bundle: bool = False
    notes: str | None = None


class SupplyRule(BaseModel):
    """Clinician-authored rule that maps case conditions to a required supply item."""

    rule_id: str
    symptom: str | None = None
    min_risk_score: float | None = None
    max_risk_score: float | None = None
    min_evacuation_eta_hours: int | None = None
    max_evacuation_eta_hours: int | None = None
    item_id: str
    quantity: int = 1
    rationale: str
    requires_manual_review_on_missing: bool = False


class RuleMatch(BaseModel):
    """Audit record showing that a specific supply rule fired for this case."""

    rule_id: str
    item_id: str
    quantity: int
    rationale: str
    triggered_by: list[str] = Field(default_factory=list)


class SupplyNeed(BaseModel):
    """Aggregated item requirement after all matching rules are combined."""

    item_id: str
    name: str
    quantity: int
    unit_weight_lb: float
    rationale: list[str] = Field(default_factory=list)
    source_rule_ids: list[str] = Field(default_factory=list)


class DroneItemAllocation(BaseModel):
    """Quantity of one catalog item assigned to a particular drone."""

    item_id: str
    name: str
    quantity: int
    unit_weight_lb: float


class DroneManifest(BaseModel):
    """One drone loadout with its total weight and packed items."""

    manifest_id: int
    total_weight_lb: float
    items: list[DroneItemAllocation] = Field(default_factory=list)


class ReviewFlag(BaseModel):
    """Human-review warning or blocking issue attached to a dispatch plan. - how the system marks something about a plan that an operator should notice before dispatch."""

    code: str
    severity: Literal["info", "warning", "critical"]
    message: str


class ExtractionResult(BaseModel):
    """Structured output from the bounded note-extraction step."""

    extracted_symptoms: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    explanation: str = ""


class DispatchPlan(BaseModel):
    """Full dispatch recommendation including supplies, manifests, redundancy, and review state."""

    plan_id: str
    request_signature: str
    burst_id: str
    raw_payload_hash: str
    canonical_case: CanonicalCase
    matched_rules: list[RuleMatch] = Field(default_factory=list)
    required_supplies: list[SupplyNeed] = Field(default_factory=list)
    base_manifests: list[DroneManifest] = Field(default_factory=list)
    shootdown_rate: int
    target_arrival_probability: float
    redundancy_multiplier: int
    per_item_arrival_probability: float
    total_drones: int
    manual_review_required: bool
    review_flags: list[ReviewFlag] = Field(default_factory=list)
    summary_text: str
    status: Literal["draft", "approved", "rejected"] = "draft"
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class EditedSupplyNeed(BaseModel):
    """Operator-provided override for the quantity of a required supply item."""

    item_id: str
    quantity: int

    @field_validator("quantity")
    @classmethod
    def validate_quantity(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("quantity must be positive")
        return value


class EditedManifestItem(BaseModel):
    """Operator-edited item assignment inside a single drone manifest."""

    item_id: str
    quantity: int

    @field_validator("quantity")
    @classmethod
    def validate_quantity(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("quantity must be positive")
        return value


class EditedManifest(BaseModel):
    """Operator-edited version of one drone manifest."""

    manifest_id: int
    items: list[EditedManifestItem]


class RecalculatePlanRequest(BaseModel):
    """Operator request to change supplies, manifests, or risk parameters and replan."""

    required_supplies: list[EditedSupplyNeed] | None = None
    base_manifests: list[EditedManifest] | None = None
    shootdown_rate: int | None = None
    target_arrival_probability: float | None = None
    operator_notes: str | None = None

    @model_validator(mode="after")
    def ensure_changes_present(self) -> "RecalculatePlanRequest":
        if self.required_supplies is None and self.base_manifests is None and self.shootdown_rate is None and self.target_arrival_probability is None:
            raise ValueError("at least one editable field must be provided")
        return self


class PlanDecisionRequest(BaseModel):
    """Operator approval or rejection submitted for a generated plan."""

    decision: Literal["approved", "rejected"]
    operator_id: str
    notes: str | None = None


class PlanDecisionRecord(BaseModel):
    """Persisted audit record of the operator's final plan decision."""

    plan_id: str
    decision: Literal["approved", "rejected"]
    operator_id: str
    notes: str | None = None
    decided_at: datetime = Field(default_factory=utc_now)
