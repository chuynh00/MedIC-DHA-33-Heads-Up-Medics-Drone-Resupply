from __future__ import annotations

import hashlib
import json
import uuid
from copy import deepcopy

from resupply_engine.catalog import load_supply_catalog, load_supply_rules
from resupply_engine.clinical_rules import ClinicalRuleEngine
from resupply_engine.config import Settings
from resupply_engine.exporter import OperatorTextExporter, add_export_failure_flag
from resupply_engine.ingest import IngestContext
from resupply_engine.llm_extractor import extract_symptoms_from_notes
from resupply_engine.models import (
    CanonicalCase,
    DispatchPlan,
    EditedSupplyNeed,
    PlanDecisionRecord,
    PlanDecisionRequest,
    PlanningRequest,
    RecalculatePlanRequest,
    ReviewFlag,
    SupplyNeed,
    utc_now,
)
from resupply_engine.normalize import normalize_payload
from resupply_engine.packing import PackingError, manifests_from_operator_input, pack_supply_needs
from resupply_engine.redundancy import calculate_redundancy_multiplier
from resupply_engine.storage import SQLitePlanStore


class DispatchPlanningService:
    def __init__(self, settings: Settings, store: SQLitePlanStore | None = None) -> None:
        # Load the static rule/catelog inputs once so every request uses the same local policy snapshot.
        self.settings = settings
        self.catalog = load_supply_catalog(settings.data_dir / "supply_catalog.csv")
        self.rules = load_supply_rules(settings.data_dir / "supply_rules.csv")
        self.rule_engine = ClinicalRuleEngine(self.catalog, self.rules)
        self.store = store or SQLitePlanStore(settings.db_path)
        self.text_exporter = OperatorTextExporter(settings.exports_dir)

    def create_plan(
        self,
        request: PlanningRequest,
        ingest_context: IngestContext | None = None,
    ) -> DispatchPlan:
        # Reject unsupported risk parameters before we touch persistence or planning logic.
        self._validate_operating_bounds(request.shootdown_rate, request.target_arrival_probability)
        raw_payload = request.payload.model_dump(mode="json")
        raw_payload_hash = self._hash_payload(raw_payload)

        # Use only the clinical payload plus planning knobs for idempotency so the same case
        # does not generate duplicate plans just because it arrived through a different adapter.
        request_signature = self._hash_payload(
            {
                "raw_payload_hash": raw_payload_hash,
                "shootdown_rate": request.shootdown_rate,
                "target_arrival_probability": request.target_arrival_probability,
            }
        )

        # Return an existing plan for retransmitted requests and append an audit event if a
        # transport adapter provided additional ingest context for this duplicate arrival.
        existing = self.store.get_plan_by_signature(request_signature)
        if existing is not None:
            if ingest_context is not None:
                self.store.append_event(
                    existing.plan_id,
                    "duplicate_ingest",
                    {
                        "request_signature": request_signature,
                        "ingest_context": ingest_context.model_dump(mode="json"),
                    },
                    utc_now().isoformat(),
                )
            return existing

        # Enrich the inbound payload with transport metadata, normalize it, and derive the
        # rule-based supply requirements before composing the full dispatch plan.
        effective_request = self._apply_ingest_context(request, ingest_context)
        canonical_case = normalize_payload(effective_request.payload)
        matched_rules, required_supplies, review_flags = self._recommend_supplies(canonical_case)
        plan = self._compose_plan(
            canonical_case=canonical_case,
            request_signature=request_signature,
            raw_payload_hash=raw_payload_hash,
            shootdown_rate=request.shootdown_rate,
            target_arrival_probability=request.target_arrival_probability,
            matched_rules=matched_rules,
            review_flags=review_flags,
            required_supplies=required_supplies,
        )
        plan = self._attach_text_export(plan, revision=1)

        # Persist both the rendered plan and a creation event so the operator workflow is auditable.
        self.store.save_new_plan(plan, raw_payload)
        created_event = {"request_signature": request_signature}
        if ingest_context is not None:
            created_event["ingest_context"] = ingest_context.model_dump(mode="json")
        self.store.update_plan(plan, "created", created_event)
        return plan

    def get_plan(self, plan_id: str) -> DispatchPlan | None:
        return self.store.get_plan(plan_id)

    def recalculate_plan(self, plan_id: str, request: RecalculatePlanRequest) -> DispatchPlan:
        existing = self.store.get_plan(plan_id)
        if existing is None:
            raise KeyError(f"Unknown plan_id: {plan_id}")

        # Merge only the operator-provided overrides; everything else should inherit from the
        # previously generated plan so recalculation stays incremental.
        shootdown_rate = request.shootdown_rate if request.shootdown_rate is not None else existing.shootdown_rate
        target = (
            request.target_arrival_probability
            if request.target_arrival_probability is not None
            else existing.target_arrival_probability
        )
        self._validate_operating_bounds(shootdown_rate, target)

        # Start from the prior review state but drop transient capacity warnings since those
        # need to be recomputed against the updated manifests and redundancy math.
        review_flags = self._copy_flags_without_transient(existing.review_flags)
        if request.operator_notes:
            review_flags.append(
                ReviewFlag(
                    code="operator_notes",
                    severity="info",
                    message=f"Operator update: {request.operator_notes}",
                )
            )

        if request.base_manifests is not None:
            # When the operator edits manifests directly, trust that explicit packing layout and
            # derive the required supplies back from those manifests.
            base_manifests, required_supplies = manifests_from_operator_input(
                request.base_manifests,
                catalog=self.catalog,
                max_payload_lb=self.settings.max_drone_payload_lb,
            )
            matched_rules = deepcopy(existing.matched_rules)
            if request.required_supplies is not None:
                # Guard against mismatches where the overall edited totals disagree with the
                # quantities implied by the manually edited manifests.
                self._validate_supply_totals(request.required_supplies, required_supplies)
        else:
            # Otherwise update the total requested supplies and let the packing engine compute
            # a fresh optimal base drone set for those items.
            required_supplies = (
                self._needs_from_operator_request(request.required_supplies)
                if request.required_supplies is not None
                else deepcopy(existing.required_supplies)
            )
            matched_rules = deepcopy(existing.matched_rules)
            base_manifests = pack_supply_needs(required_supplies, self.settings.max_drone_payload_lb)

        plan = self._compose_plan(
            canonical_case=existing.canonical_case,
            request_signature=existing.request_signature,
            raw_payload_hash=existing.raw_payload_hash,
            shootdown_rate=shootdown_rate,
            target_arrival_probability=target,
            matched_rules=matched_rules,
            review_flags=review_flags,
            required_supplies=required_supplies,
            provided_base_manifests=base_manifests,
            plan_id=existing.plan_id,
            status="draft",
        )
        next_revision = (existing.export.export_revision or 0) + 1
        plan = self._attach_text_export(plan, revision=next_revision)

        # Store the recalculated version under the same plan id so the latest draft remains the
        # canonical operator-facing record, while the event log preserves the edit history.
        self.store.update_plan(
            plan,
            "recalculated",
            request.model_dump(mode="json"),
        )
        return plan

    def record_decision(self, plan_id: str, request: PlanDecisionRequest) -> DispatchPlan:
        existing = self.store.get_plan(plan_id)
        if existing is None:
            raise KeyError(f"Unknown plan_id: {plan_id}")

        # Freeze the latest plan state with the operator's final decision and record that
        # decision separately for audit/history purposes.
        status = "approved" if request.decision == "approved" else "rejected"
        plan = existing.model_copy(update={"status": status, "updated_at": utc_now()})
        self.store.record_decision(
            PlanDecisionRecord(
                plan_id=plan.plan_id,
                decision=request.decision,
                operator_id=request.operator_id,
                notes=request.notes,
            )
        )
        self.store.update_plan(
            plan,
            "decision_recorded",
            request.model_dump(mode="json"),
        )
        return plan

    def _recommend_supplies(
        self,
        canonical_case: CanonicalCase,
    ) -> tuple[list, list[SupplyNeed], list[ReviewFlag]]:
        review_flags: list[ReviewFlag] = []

        # Use the bounded extractor only to enrich structured symptoms from free text; the
        # actual medical supply recommendation remains rule-based.
        extraction = extract_symptoms_from_notes(canonical_case.notes)
        symptoms = set(canonical_case.symptoms)
        if extraction.extracted_symptoms:
            symptoms.update(extraction.extracted_symptoms)
            canonical_case.symptoms = sorted(symptoms)

        # Low-confidence extraction does not block planning, but it does force a human to review
        # the result because the case context may have been incompletely interpreted.
        if canonical_case.notes and extraction.confidence < self.settings.llm_extraction_confidence_threshold:
            review_flags.append(
                ReviewFlag(
                    code="low_confidence_note_extraction",
                    severity="warning",
                    message=extraction.explanation or "Free-text note extraction confidence is low.",
                )
            )

        # Convert the normalized case into concrete required supplies via clinician-authored rules.
        matched_rules, required_supplies = self.rule_engine.recommend(canonical_case)
        if not required_supplies:
            # A plan with no matched protocol is still returned, but only as a manual-review case.
            review_flags.append(
                ReviewFlag(
                    code="no_rules_matched",
                    severity="critical",
                    message="No clinician-authored supply rules matched this case. Manual review required.",
                )
            )
        return matched_rules, required_supplies, review_flags

    def _compose_plan(
        self,
        canonical_case: CanonicalCase,
        request_signature: str,
        raw_payload_hash: str,
        shootdown_rate: int,
        target_arrival_probability: float,
        matched_rules: list,
        review_flags: list[ReviewFlag],
        required_supplies: list[SupplyNeed],
        provided_base_manifests: list | None = None,
        plan_id: str | None = None,
        status: str = "draft",
    ) -> DispatchPlan:
        # Either respect an operator-provided manifest layout or compute a fresh packed base set.
        # Base manifests is one complete set of drones that would need to be dispatched to get all required supplies to the patient
        base_manifests = (
            provided_base_manifests
            if provided_base_manifests is not None
            else pack_supply_needs(required_supplies, self.settings.max_drone_payload_lb)
        )

        # Redundancy is applied at the level of the whole packed base set, not per-item
        # independently, so the output can be handed directly to a dispatch operator.
        redundancy_multiplier, item_arrival_probability = calculate_redundancy_multiplier(
            shootdown_rate,
            target_arrival_probability,
        )
        total_drones = len(base_manifests) * redundancy_multiplier

        # Recompute blocking review conditions from the latest packing and redundancy results.
        finalized_flags = deepcopy(review_flags)
        if total_drones > self.settings.max_total_drones:
            finalized_flags.append(
                ReviewFlag(
                    code="exceeds_auto_dispatch_drone_cap",
                    severity="critical",
                    message=(
                        f"Computed plan requires {total_drones} drones which exceeds the auto-dispatch cap "
                        f"of {self.settings.max_total_drones}."
                    ),
                )
            )

        # Any warning or critical flag pushes the plan into a human-review-required state.
        manual_review_required = any(flag.severity in {"warning", "critical"} for flag in finalized_flags)
        plan = DispatchPlan(
            plan_id=plan_id or str(uuid.uuid4()),
            request_signature=request_signature,
            burst_id=canonical_case.burst_id,
            raw_payload_hash=raw_payload_hash,
            canonical_case=canonical_case,
            matched_rules=matched_rules,
            required_supplies=required_supplies,
            base_manifests=base_manifests,
            shootdown_rate=shootdown_rate,
            target_arrival_probability=target_arrival_probability,
            redundancy_multiplier=redundancy_multiplier,
            per_item_arrival_probability=item_arrival_probability,
            total_drones=total_drones,
            manual_review_required=manual_review_required,
            review_flags=finalized_flags,
            summary_text=self._render_summary(
                canonical_case,
                required_supplies,
                base_manifests,
                redundancy_multiplier,
                item_arrival_probability,
                shootdown_rate,
                total_drones,
                finalized_flags,
            ),
            status=status,
        )
        return plan

    def _render_summary(
        self,
        canonical_case: CanonicalCase,
        required_supplies: list[SupplyNeed],
        base_manifests: list,
        redundancy_multiplier: int,
        item_arrival_probability: float,
        shootdown_rate: int,
        total_drones: int,
        review_flags: list[ReviewFlag],
    ) -> str:
        # Render a compact operator-facing summary that mirrors the structured plan payload.
        supply_summary = ", ".join(
            f"{need.quantity}x {need.name} ({need.unit_weight_lb} lb each)" for need in required_supplies
        ) or "No supplies matched"
        manifest_lines = [
            f"Drone {manifest.manifest_id}: {manifest.total_weight_lb} lb -> "
            + ", ".join(f"{item.quantity}x {item.name}" for item in manifest.items)
            for manifest in base_manifests
        ]
        flags_text = "; ".join(flag.message for flag in review_flags) or "No review flags."
        medic_requested_text = ", ".join(canonical_case.requested_supplies) or "none"
        return (
            f"Patient {canonical_case.patient_id} / case {canonical_case.case_id} / mission {canonical_case.mission_id}\n"
            f"Priority: {canonical_case.priority_flag} | Time since injury: {canonical_case.time_since_injury_min} min | "
            f"Casualty count: {canonical_case.casualty_count}\n"
            f"Symptoms: {', '.join(canonical_case.symptoms) or 'none reported'}\n"
            f"Medic requested supplies: {medic_requested_text}\n"
            f"Requested supplies: {supply_summary}\n"
            f"Base drone set ({len(base_manifests)} drones): {' | '.join(manifest_lines) if manifest_lines else 'none'}\n"
            f"Shootdown rate: {shootdown_rate}% | Replication count: {redundancy_multiplier} | "
            f"Per-item arrival probability: {item_arrival_probability:.2%}\n"
            f"Total drones to dispatch: {total_drones}\n"
            f"Review flags: {flags_text}"
        )

    def _attach_text_export(self, plan: DispatchPlan, revision: int) -> DispatchPlan:
        try:
            export_metadata = self.text_exporter.export(plan, revision=revision)
            return plan.model_copy(update={"export": export_metadata})
        except OSError as exc:
            updated_flags = add_export_failure_flag(
                plan.review_flags,
                f"Failed to write operator text export: {exc}",
            )
            return plan.model_copy(
                update={
                    "review_flags": updated_flags,
                    "manual_review_required": True,
                }
            )

    def _needs_from_operator_request(self, required_supplies: list[EditedSupplyNeed] | None) -> list[SupplyNeed]:
        if required_supplies is None:
            return []

        # Translate operator-edited totals back into the standard SupplyNeed shape so the rest
        # of the planning pipeline can treat overrides and rule-generated needs uniformly.
        needs: list[SupplyNeed] = []
        for edited_need in required_supplies:
            if edited_need.item_id not in self.catalog:
                raise KeyError(f"Unknown catalog item: {edited_need.item_id}")
            catalog_item = self.catalog[edited_need.item_id]
            needs.append(
                SupplyNeed(
                    item_id=edited_need.item_id,
                    name=catalog_item.name,
                    quantity=edited_need.quantity,
                    unit_weight_lb=catalog_item.unit_weight_lb,
                    rationale=["Operator override"],
                    source_rule_ids=["operator_override"],
                )
            )
        return sorted(needs, key=lambda need: (need.name, need.item_id))

    @staticmethod
    def _validate_supply_totals(edited_supplies: list[EditedSupplyNeed], derived_supplies: list[SupplyNeed]) -> None:
        # When both totals and manifests are edited together, require them to describe the same plan.
        edited = {item.item_id: item.quantity for item in edited_supplies}
        derived = {item.item_id: item.quantity for item in derived_supplies}
        if edited != derived:
            raise ValueError("Edited manifests do not match the supplied required_supplies totals.")

    @staticmethod
    def _copy_flags_without_transient(flags: list[ReviewFlag]) -> list[ReviewFlag]:
        # Capacity-related warnings are recalculated on every replan, so avoid carrying them forward.
        return [
            flag
            for flag in deepcopy(flags)
            if flag.code != "exceeds_auto_dispatch_drone_cap"
        ]

    def _validate_operating_bounds(self, shootdown_rate: int, target_arrival_probability: float) -> None:
        # Keep risk inputs within the operating assumptions baked into the current MVP.
        if shootdown_rate not in self.settings.allowed_shootdown_rates:
            raise ValueError(
                f"shootdown_rate must be one of {self.settings.allowed_shootdown_rates}"
            )
        if not 0 < target_arrival_probability <= 1:
            raise ValueError("target_arrival_probability must be between 0 and 1")

    @staticmethod
    def _apply_ingest_context(
        request: PlanningRequest,
        ingest_context: IngestContext | None,
    ) -> PlanningRequest:
        if ingest_context is None:
            return request

        # Preserve the original clinical payload, but enrich it with transport metadata for
        # downstream auditability and future adapter-specific behavior.
        payload = request.payload.model_copy(deep=True)
        payload.source = payload.source.model_copy(
            update={
                "received_via": ingest_context.transport,
                "source_message_type": payload.source.source_message_type or ingest_context.content_type,
            }
        )
        payload.extra = {
            **payload.extra,
            "ingest_context": ingest_context.model_dump(mode="json"),
        }
        return request.model_copy(update={"payload": payload})

    @staticmethod
    def _hash_payload(payload: dict) -> str:
        # Stable hashing underpins idempotent plan creation for retried or repeated messages.
        normalized = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def build_service(settings: Settings | None = None) -> DispatchPlanningService:
    resolved_settings = settings or Settings()
    return DispatchPlanningService(resolved_settings)
