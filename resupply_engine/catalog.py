from __future__ import annotations

import csv
from pathlib import Path

from resupply_engine.models import SupplyCatalogItem, SupplyRule


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in {"true", "1", "yes", "y"}


def _parse_optional_bool(value: str) -> bool | None:
    value = value.strip()
    if not value:
        return None
    return _parse_bool(value)


def _parse_optional_float(value: str) -> float | None:
    value = value.strip()
    return float(value) if value else None


def _parse_optional_int(value: str) -> int | None:
    value = value.strip()
    return int(value) if value else None


def load_supply_catalog(path: Path) -> dict[str, SupplyCatalogItem]:
    items: dict[str, SupplyCatalogItem] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            item = SupplyCatalogItem(
                item_id=row["item_id"].strip(),
                name=row["name"].strip(),
                unit_weight_lb=float(row["unit_weight_lb"]),
                unit_of_issue=row.get("unit_of_issue", "each").strip() or "each",
                is_bundle=_parse_bool(row.get("is_bundle", "")),
                notes=row.get("notes", "").strip() or None,
            )
            items[item.item_id] = item
    return items


def load_supply_rules(path: Path) -> list[SupplyRule]:
    rules: list[SupplyRule] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rules.append(
                SupplyRule(
                    rule_id=row["rule_id"].strip(),
                    min_gcs_total=_parse_optional_int(row.get("min_gcs_total", "")),
                    max_gcs_total=_parse_optional_int(row.get("max_gcs_total", "")),
                    seizure=_parse_optional_bool(row.get("seizure", "")),
                    vomiting=_parse_optional_bool(row.get("vomiting", "")),
                    head_external_hemorrhage=_parse_optional_bool(row.get("head_external_hemorrhage", "")),
                    suspected_icp=_parse_optional_bool(row.get("suspected_icp", "")),
                    location_contains=row.get("location_contains", "").strip() or None,
                    required_march_flag=row.get("required_march_flag", "").strip().lower().replace("-", "_").replace(" ", "_") or None,
                    min_systolic_bp=_parse_optional_int(row.get("min_systolic_bp", "")),
                    max_systolic_bp=_parse_optional_int(row.get("max_systolic_bp", "")),
                    min_spo2=_parse_optional_int(row.get("min_spo2", "")),
                    max_spo2=_parse_optional_int(row.get("max_spo2", "")),
                    min_heart_rate=_parse_optional_int(row.get("min_heart_rate", "")),
                    max_heart_rate=_parse_optional_int(row.get("max_heart_rate", "")),
                    min_temp_c=_parse_optional_float(row.get("min_temp_c", "")),
                    max_temp_c=_parse_optional_float(row.get("max_temp_c", "")),
                    item_id=row["item_id"].strip(),
                    quantity=int(row.get("quantity", "1")),
                    rationale=row["rationale"].strip(),
                    requires_manual_review_on_missing=_parse_bool(
                        row.get("requires_manual_review_on_missing", "")
                    ),
                )
            )
    return rules
