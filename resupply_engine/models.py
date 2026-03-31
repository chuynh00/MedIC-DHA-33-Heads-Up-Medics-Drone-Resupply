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


class SoftwareDecisionSupportPayload(BaseModel):
    """Flat medic-system patient profile sent from the field decision support tool."""

    patient_id: str
    mission_id: str
    medic_id: str
    gcs_total: int
    gcs_eye: int
    gcs_verbal: int
    gcs_motor: int
    bp_systolic: int
    bp_diastolic: int
    heart_rate: int
    oxygen_saturation: int
    temp_c: float
    seizure: bool
    vomiting: bool
    head_external_hemorrhage: bool
    suspected_icp: bool
    location: str | None = None
    march_flags: list[str] = Field(default_factory=list)
    notes: str | None = None
    timestamp: datetime
    source: SourceMetadata = Field(default_factory=SourceMetadata)
    extra: dict[str, Any] = Field(default_factory=dict)

    @field_validator("march_flags", mode="before")
    @classmethod
    def normalize_march_flags(cls, value: list[str] | None) -> list[str]:
        if value is None:
            return []
        return [
            str(flag).strip().lower().replace(" ", "_").replace("-", "_")
            for flag in value
            if str(flag).strip()
        ]


class PlanningRequest(BaseModel):
    """Top-level request combining a patient burst with dispatch planning parameters."""

    shootdown_rate: int
    target_arrival_probability: float = 0.95
    burst_id: str | None = None
    software_decision_support_payload: SoftwareDecisionSupportPayload

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
    mission_id: str
    medic_id: str
    reported_at: datetime
    seizure: bool
    vomiting: bool
    head_external_hemorrhage: bool
    suspected_icp: bool
    location: str | None = None
    march_flags: list[str] = Field(default_factory=list)
    symptoms: list[str] = Field(default_factory=list)
    vitals: VitalSigns = Field(default_factory=VitalSigns)
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
    min_gcs_total: int | None = None
    max_gcs_total: int | None = None
    seizure: bool | None = None
    vomiting: bool | None = None
    head_external_hemorrhage: bool | None = None
    suspected_icp: bool | None = None
    location_contains: str | None = None
    required_march_flag: str | None = None
    min_systolic_bp: int | None = None
    max_systolic_bp: int | None = None
    min_spo2: int | None = None
    max_spo2: int | None = None
    min_heart_rate: int | None = None
    max_heart_rate: int | None = None
    min_temp_c: float | None = None
    max_temp_c: float | None = None
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
    source_type: Literal["rule", "llm", "repaired_llm", "operator_override"] = "rule"
    policy_refs: list[str] = Field(default_factory=list)
    llm_confidence: float | None = None
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


class WorkbookTriggerClause(BaseModel):
    """One machine-readable condition compiled from the clinical workbook."""

    field_name: str
    operator: Literal["eq", "contains", "lte", "gte", "lt", "gt"]
    value: str | int | float | bool


class CompiledItemRule(BaseModel):
    """Compiled workbook item trigger rule used by policy evaluation and LLM prompting."""

    rule_id: str
    item_id: str
    item_name: str
    default_quantity: int = 1
    logic_type: str
    priority_rank: int
    clinical_rationale: str
    trigger_text: str
    any_clauses: list[WorkbookTriggerClause] = Field(default_factory=list)
    all_clauses: list[WorkbookTriggerClause] = Field(default_factory=list)
    unsupported_clauses: list[str] = Field(default_factory=list)


class CompiledPriorityRank(BaseModel):
    """Priority rank and notes for one supply item from the workbook."""

    item_id: str
    item_name: str
    priority_rank: int
    workbook_weight_lb: float | None = None
    notes: str | None = None


class CompiledClinicalWorkbook(BaseModel):
    """Structured workbook snapshot compiled from the Excel clinical policy source."""

    workbook_path: str
    item_rules: list[CompiledItemRule] = Field(default_factory=list)
    priority_ranks: dict[str, CompiledPriorityRank] = Field(default_factory=dict)


class LLMRecommendedItem(BaseModel):
    """One structured supply recommendation emitted by the LLM."""

    item_id: str
    quantity: int
    rationale: str = ""
    policy_refs: list[str] = Field(default_factory=list)
    confidence: float | None = None

    @field_validator("quantity")
    @classmethod
    def validate_quantity(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("quantity must be positive")
        return value


class LLMRecommendationResponse(BaseModel):
    """Structured top-level LLM output used before deterministic validation and repair."""

    recommended_items: list[LLMRecommendedItem] = Field(default_factory=list)
    overall_confidence: float | None = None
    manual_review_notes: str = ""


class PolicyEvaluationResult(BaseModel):
    """Deterministic workbook policy evaluation output for one canonical case."""

    must_include_items: list[SupplyNeed] = Field(default_factory=list)
    allowed_items: list[str] = Field(default_factory=list)
    blocked_items: list[str] = Field(default_factory=list)
    policy_matches: list[RuleMatch] = Field(default_factory=list)
    priority_rank_map: dict[str, int] = Field(default_factory=dict)


class ExportMetadata(BaseModel):
    """Metadata describing a generated operator-facing text export."""

    export_path: str | None = None
    export_format: Literal["txt"] | None = None
    export_revision: int | None = None


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
    export: ExportMetadata = Field(default_factory=ExportMetadata)
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
