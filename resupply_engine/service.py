from __future__ import annotations

import hashlib
import json
import uuid
from copy import deepcopy

from resupply_engine.catalog import load_supply_catalog
from resupply_engine.config import Settings
from resupply_engine.exporter import OperatorTextExporter, add_export_failure_flag
from resupply_engine.ingest import IngestContext
from resupply_engine.llm_extractor import extract_symptoms_from_notes
from resupply_engine.llm_recommender import (
    LLMRecommendationError,
    LocalHuggingFaceBackend,
    LocalLLMRecommender,
    OllamaBackend,
)
from resupply_engine.models import (
    CanonicalCase,
    DispatchPlan,
    EditedSupplyNeed,
    LLMRecommendationResponse,
    LLMRecommendedItem,
    PlanDecisionRecord,
    PlanDecisionRequest,
    PlanningRequest,
    PolicyEvaluationResult,
    RecalculatePlanRequest,
    ReviewFlag,
    SupplyNeed,
    utc_now,
)
from resupply_engine.normalize import normalize_payload
from resupply_engine.packing import PackingError, manifests_from_operator_input, pack_supply_needs
from resupply_engine.redundancy import calculate_redundancy_multiplier
from resupply_engine.storage import SQLitePlanStore
from resupply_engine.workbook_policy import compile_clinical_workbook, evaluate_clinical_workbook


class DispatchPlanningService:
    def __init__(
        self,
        settings: Settings,
        store: SQLitePlanStore | None = None,
        recommender: LocalLLMRecommender | None = None,
    ) -> None:
        # Load the static catalog and compiled workbook policy once so every request uses the
        # same local clinical snapshot.
        self.settings = settings
        self.catalog = load_supply_catalog(settings.data_dir / "supply_catalog.csv")
        self.workbook = compile_clinical_workbook(settings.clinical_workbook_path, self.catalog)
        self.store = store or SQLitePlanStore(settings.db_path)
        self.text_exporter = OperatorTextExporter(settings.exports_dir)
        self.recommender = recommender or self._build_recommender()

    def create_plan(
        self,
        request: PlanningRequest,
        ingest_context: IngestContext | None = None,
    ) -> DispatchPlan:
        # Reject unsupported risk parameters before we touch persistence or planning logic.
        self._validate_operating_bounds(request.shootdown_rate, request.target_arrival_probability)
        raw_payload = request.software_decision_support_payload.model_dump(mode="json")
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
        # workbook-gated supply requirements before composing the full dispatch plan.
        effective_request = self._apply_ingest_context(request, ingest_context)
        burst_id = request.burst_id or self._derive_burst_id(raw_payload_hash)
        canonical_case = normalize_payload(effective_request.software_decision_support_payload, burst_id=burst_id)
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

        # Use the bounded extractor only to enrich structured findings from free text; the
        # actual supply recommendation remains workbook- and validator-gated.
        extraction = extract_symptoms_from_notes(canonical_case.notes)
        findings = set(canonical_case.symptoms)
        if extraction.extracted_symptoms:
            findings.update(extraction.extracted_symptoms)
            canonical_case.symptoms = sorted(findings)

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

        policy_result = evaluate_clinical_workbook(self.workbook, canonical_case, self.catalog)
        matched_rules = policy_result.policy_matches

        llm_response: LLMRecommendationResponse | None = None
        if self.settings.llm_enabled:
            try:
                llm_response = self.recommender.recommend(canonical_case, self.workbook, self.catalog)
            except LLMRecommendationError as exc:
                review_flags.append(
                    ReviewFlag(
                        code="llm_unavailable_fallback",
                        severity="warning",
                        message=f"Local LLM recommender unavailable; using workbook-only fallback. {exc}",
                    )
                )

        required_supplies, llm_flags = self._finalize_recommended_supplies(policy_result, llm_response)
        review_flags.extend(llm_flags)

        if not required_supplies:
            review_flags.append(
                ReviewFlag(
                    code="no_policy_matches",
                    severity="critical",
                    message="No clinical workbook policy matched this case. Manual review required.",
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
        base_manifests = (
            provided_base_manifests
            if provided_base_manifests is not None
            else pack_supply_needs(required_supplies, self.settings.max_drone_payload_lb)
        )

        # Redundancy is still fully deterministic and based on operator-provided shootdown rate.
        redundancy_multiplier, item_arrival_probability = calculate_redundancy_multiplier(
            shootdown_rate,
            target_arrival_probability,
        )
        total_drones = len(base_manifests) * redundancy_multiplier

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

        manual_review_required = any(flag.severity in {"warning", "critical"} for flag in finalized_flags)
        return DispatchPlan(
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
        supply_summary = ", ".join(
            f"{need.quantity}x {need.name} [{need.source_type}] ({need.unit_weight_lb} lb each)"
            for need in required_supplies
        ) or "No supplies matched"
        manifest_lines = [
            f"Drone {manifest.manifest_id}: {manifest.total_weight_lb} lb -> "
            + ", ".join(f"{item.quantity}x {item.name}" for item in manifest.items)
            for manifest in base_manifests
        ]
        flags_text = "; ".join(flag.message for flag in review_flags) or "No review flags."
        march_flags_text = ", ".join(canonical_case.march_flags) or "none"
        findings_text = ", ".join(canonical_case.symptoms) or "none reported"
        return (
            f"Patient {canonical_case.patient_id} / mission {canonical_case.mission_id} / medic {canonical_case.medic_id}\n"
            f"GCS: {canonical_case.vitals.gcs} (E{canonical_case.vitals.gcs_eye}/V{canonical_case.vitals.gcs_verbal}/M{canonical_case.vitals.gcs_motor}) | "
            f"SBP: {canonical_case.vitals.systolic_bp} | SpO2: {canonical_case.vitals.spo2}% | Temp: {canonical_case.vitals.temperature_c} C\n"
            f"Direct findings: seizure={str(canonical_case.seizure).lower()}, vomiting={str(canonical_case.vomiting).lower()}, "
            f"head_external_hemorrhage={str(canonical_case.head_external_hemorrhage).lower()}, "
            f"suspected_icp={str(canonical_case.suspected_icp).lower()}, location={canonical_case.location or 'none'}\n"
            f"MARCH flags: {march_flags_text}\n"
            f"Derived findings: {findings_text}\n"
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
                    source_type="operator_override",
                    policy_refs=["operator_override"],
                    llm_confidence=None,
                    rationale=["Operator override"],
                    source_rule_ids=["operator_override"],
                )
            )
        return sorted(needs, key=lambda need: (need.name, need.item_id))

    @staticmethod
    def _validate_supply_totals(edited_supplies: list[EditedSupplyNeed], derived_supplies: list[SupplyNeed]) -> None:
        edited = {item.item_id: item.quantity for item in edited_supplies}
        derived = {item.item_id: item.quantity for item in derived_supplies}
        if edited != derived:
            raise ValueError("Edited manifests do not match the supplied required_supplies totals.")

    @staticmethod
    def _copy_flags_without_transient(flags: list[ReviewFlag]) -> list[ReviewFlag]:
        return [
            flag
            for flag in deepcopy(flags)
            if flag.code != "exceeds_auto_dispatch_drone_cap"
        ]

    def _validate_operating_bounds(self, shootdown_rate: int, target_arrival_probability: float) -> None:
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

        payload = request.software_decision_support_payload.model_copy(deep=True)
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
        return request.model_copy(update={"software_decision_support_payload": payload})

    @staticmethod
    def _derive_burst_id(raw_payload_hash: str) -> str:
        return f"BST-{raw_payload_hash[:12].upper()}"

    @staticmethod
    def _hash_payload(payload: dict) -> str:
        normalized = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def _build_recommender(self) -> LocalLLMRecommender:
        if not self.settings.llm_enabled:
            return LocalLLMRecommender()
        backend_name = self.settings.llm_backend.strip().lower()
        if backend_name == "ollama":
            backend = OllamaBackend(
                base_url=self.settings.ollama_base_url,
                model=self.settings.ollama_model,
                temperature=self.settings.llm_temperature,
                max_tokens=self.settings.llm_max_tokens,
                timeout_seconds=self.settings.llm_timeout_seconds,
            )
            return LocalLLMRecommender(backend=backend)
        if backend_name == "huggingface":
            if self.settings.llm_model_path is None:
                return LocalLLMRecommender()
            backend = LocalHuggingFaceBackend(
                self.settings.llm_model_path,
                temperature=self.settings.llm_temperature,
                max_tokens=self.settings.llm_max_tokens,
            )
            return LocalLLMRecommender(backend=backend)
        raise ValueError(f"Unsupported llm_backend: {self.settings.llm_backend}")

    def _finalize_recommended_supplies(
        self,
        policy_result: PolicyEvaluationResult,
        llm_response: LLMRecommendationResponse | None,
    ) -> tuple[list[SupplyNeed], list[ReviewFlag]]:
        review_flags: list[ReviewFlag] = []
        ranked_items = policy_result.priority_rank_map

        if llm_response is None:
            return policy_result.must_include_items, review_flags

        if llm_response.manual_review_notes:
            review_flags.append(
                ReviewFlag(
                    code="llm_manual_review_notes",
                    severity="info",
                    message=llm_response.manual_review_notes,
                )
            )

        aggregated: dict[str, SupplyNeed] = {}
        for item in llm_response.recommended_items:
            candidate = self._validated_llm_supply_need(item, policy_result)
            if candidate is None:
                review_flags.append(
                    ReviewFlag(
                        code="llm_item_removed",
                        severity="warning",
                        message=f"Removed unsupported LLM recommendation for item {item.item_id}.",
                    )
                )
                continue
            if candidate.item_id in aggregated:
                existing = aggregated[candidate.item_id]
                aggregated[candidate.item_id] = existing.model_copy(
                    update={
                        "quantity": existing.quantity + candidate.quantity,
                        "rationale": sorted(set([*existing.rationale, *candidate.rationale])),
                        "policy_refs": sorted(set([*existing.policy_refs, *candidate.policy_refs])),
                        "source_rule_ids": sorted(set([*existing.source_rule_ids, *candidate.source_rule_ids])),
                        "llm_confidence": max(existing.llm_confidence or 0.0, candidate.llm_confidence or 0.0),
                    }
                )
            else:
                aggregated[candidate.item_id] = candidate

        for must_include in policy_result.must_include_items:
            if must_include.item_id not in aggregated:
                aggregated[must_include.item_id] = must_include.model_copy(
                    update={
                        "source_type": "repaired_llm",
                        "policy_refs": sorted(set([*must_include.policy_refs, "deterministic_policy_fallback"])),
                    }
                )
                review_flags.append(
                    ReviewFlag(
                        code="llm_missing_mandatory_item",
                        severity="warning",
                        message=f"Added mandatory workbook item {must_include.item_id} after LLM omission.",
                    )
                )
                continue

            current = aggregated[must_include.item_id]
            if current.quantity < must_include.quantity:
                aggregated[must_include.item_id] = current.model_copy(
                    update={
                        "quantity": must_include.quantity,
                        "source_type": "repaired_llm",
                        "policy_refs": sorted(set([*current.policy_refs, *must_include.policy_refs])),
                        "rationale": sorted(set([*current.rationale, *must_include.rationale])),
                        "source_rule_ids": sorted(set([*current.source_rule_ids, *must_include.source_rule_ids])),
                    }
                )
                review_flags.append(
                    ReviewFlag(
                        code="llm_quantity_repaired",
                        severity="warning",
                        message=f"Increased quantity for mandatory workbook item {must_include.item_id} to policy minimum.",
                    )
                )

        finalized = list(aggregated.values())
        finalized.sort(key=lambda need: (ranked_items.get(need.item_id, 999), need.name, need.item_id))
        return finalized, review_flags

    def _validated_llm_supply_need(
        self,
        item: LLMRecommendedItem,
        policy_result: PolicyEvaluationResult,
    ) -> SupplyNeed | None:
        if item.item_id not in self.catalog:
            return None
        if item.item_id in policy_result.blocked_items:
            return None
        if item.item_id not in policy_result.allowed_items:
            return None

        catalog_item = self.catalog[item.item_id]
        return SupplyNeed(
            item_id=item.item_id,
            name=catalog_item.name,
            quantity=item.quantity,
            unit_weight_lb=catalog_item.unit_weight_lb,
            source_type="llm",
            policy_refs=sorted(set(item.policy_refs)),
            llm_confidence=item.confidence,
            rationale=[item.rationale] if item.rationale else [],
            source_rule_ids=sorted(set(item.policy_refs)),
        )


def build_service(settings: Settings | None = None) -> DispatchPlanningService:
    resolved_settings = settings or Settings()
    return DispatchPlanningService(resolved_settings)
