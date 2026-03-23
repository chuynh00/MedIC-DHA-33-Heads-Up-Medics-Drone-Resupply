from __future__ import annotations

from collections import defaultdict

from resupply_engine.models import CanonicalCase, RuleMatch, SupplyCatalogItem, SupplyNeed, SupplyRule


class ClinicalRuleEngine:
    def __init__(self, catalog: dict[str, SupplyCatalogItem], rules: list[SupplyRule]) -> None:
        self.catalog = catalog
        self.rules = rules

    def recommend(self, case: CanonicalCase) -> tuple[list[RuleMatch], list[SupplyNeed]]:
        matches: list[RuleMatch] = []
        aggregated: dict[str, dict[str, object]] = defaultdict(
            lambda: {"quantity": 0, "rationale": [], "source_rule_ids": []}
        )

        case_symptoms = set(case.symptoms)
        for rule in self.rules:
            if not self._rule_matches(rule, case, case_symptoms):
                continue

            catalog_item = self.catalog[rule.item_id]
            triggered_by = [token for token in [rule.symptom, self._risk_trigger(rule, case), self._eta_trigger(rule, case)] if token]
            matches.append(
                RuleMatch(
                    rule_id=rule.rule_id,
                    item_id=rule.item_id,
                    quantity=rule.quantity,
                    rationale=rule.rationale,
                    triggered_by=triggered_by,
                )
            )
            entry = aggregated[rule.item_id]
            entry["quantity"] = int(entry["quantity"]) + rule.quantity
            rationale = list(entry["rationale"])
            rationale.append(rule.rationale)
            entry["rationale"] = rationale
            source_rule_ids = list(entry["source_rule_ids"])
            source_rule_ids.append(rule.rule_id)
            entry["source_rule_ids"] = source_rule_ids
            entry["name"] = catalog_item.name
            entry["unit_weight_lb"] = catalog_item.unit_weight_lb

        needs = [
            SupplyNeed(
                item_id=item_id,
                name=str(details["name"]),
                quantity=int(details["quantity"]),
                unit_weight_lb=float(details["unit_weight_lb"]),
                rationale=sorted(set(details["rationale"])),
                source_rule_ids=sorted(set(details["source_rule_ids"])),
            )
            for item_id, details in aggregated.items()
        ]
        needs.sort(key=lambda need: (need.name, need.item_id))
        return matches, needs

    @staticmethod
    def _rule_matches(rule: SupplyRule, case: CanonicalCase, case_symptoms: set[str]) -> bool:
        if rule.symptom and rule.symptom not in case_symptoms:
            return False
        if rule.min_risk_score is not None and (case.risk_score is None or case.risk_score < rule.min_risk_score):
            return False
        if rule.max_risk_score is not None and (case.risk_score is None or case.risk_score > rule.max_risk_score):
            return False
        if rule.min_evacuation_eta_hours is not None and (
            case.evacuation_eta_hours is None or case.evacuation_eta_hours < rule.min_evacuation_eta_hours
        ):
            return False
        if rule.max_evacuation_eta_hours is not None and (
            case.evacuation_eta_hours is None or case.evacuation_eta_hours > rule.max_evacuation_eta_hours
        ):
            return False
        return True

    @staticmethod
    def _risk_trigger(rule: SupplyRule, case: CanonicalCase) -> str | None:
        if case.risk_score is None:
            return None
        if rule.min_risk_score is not None or rule.max_risk_score is not None:
            return f"risk_score={case.risk_score}"
        return None

    @staticmethod
    def _eta_trigger(rule: SupplyRule, case: CanonicalCase) -> str | None:
        if case.evacuation_eta_hours is None:
            return None
        if rule.min_evacuation_eta_hours is not None or rule.max_evacuation_eta_hours is not None:
            return f"evac_eta_hours={case.evacuation_eta_hours}"
        return None
