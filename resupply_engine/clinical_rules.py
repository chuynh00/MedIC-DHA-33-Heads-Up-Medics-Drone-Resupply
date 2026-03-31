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

        for rule in self.rules:
            triggered_by = self._triggered_by(rule, case)
            if triggered_by is None:
                continue

            catalog_item = self.catalog[rule.item_id]
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

    @classmethod
    def _triggered_by(cls, rule: SupplyRule, case: CanonicalCase) -> list[str] | None:
        triggered_by: list[str] = []

        if rule.min_gcs_total is not None:
            if case.vitals.gcs is None or case.vitals.gcs < rule.min_gcs_total:
                return None
            triggered_by.append(f"gcs_total={case.vitals.gcs}")

        if rule.max_gcs_total is not None:
            if case.vitals.gcs is None or case.vitals.gcs > rule.max_gcs_total:
                return None
            if f"gcs_total={case.vitals.gcs}" not in triggered_by:
                triggered_by.append(f"gcs_total={case.vitals.gcs}")

        for field_name in ("seizure", "vomiting", "head_external_hemorrhage", "suspected_icp"):
            rule_value = getattr(rule, field_name)
            if rule_value is None:
                continue
            case_value = getattr(case, field_name)
            if case_value is not rule_value:
                return None
            triggered_by.append(f"{field_name}={str(case_value).lower()}")

        if rule.location_contains is not None:
            location = (case.location or "").lower()
            required = rule.location_contains.lower()
            if required not in location:
                return None
            triggered_by.append(f"location~{rule.location_contains}")

        if rule.required_march_flag is not None:
            if rule.required_march_flag not in set(case.march_flags):
                return None
            triggered_by.append(f"march_flag={rule.required_march_flag}")

        numeric_checks = (
            ("min_systolic_bp", "max_systolic_bp", case.vitals.systolic_bp, "systolic_bp"),
            ("min_spo2", "max_spo2", case.vitals.spo2, "spo2"),
            ("min_heart_rate", "max_heart_rate", case.vitals.heart_rate, "heart_rate"),
            ("min_temp_c", "max_temp_c", case.vitals.temperature_c, "temp_c"),
        )
        for min_field, max_field, case_value, label in numeric_checks:
            min_rule = getattr(rule, min_field)
            max_rule = getattr(rule, max_field)
            if min_rule is None and max_rule is None:
                continue
            if case_value is None:
                return None
            if min_rule is not None and case_value < min_rule:
                return None
            if max_rule is not None and case_value > max_rule:
                return None
            triggered_by.append(f"{label}={case_value}")

        return triggered_by
