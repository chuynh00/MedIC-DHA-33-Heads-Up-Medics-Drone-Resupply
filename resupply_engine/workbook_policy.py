from __future__ import annotations

import re
import zipfile
from collections import defaultdict
from html import unescape
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

from resupply_engine.models import (
    CanonicalCase,
    CompiledClinicalWorkbook,
    CompiledItemRule,
    CompiledPriorityRank,
    RuleMatch,
    SupplyCatalogItem,
    SupplyNeed,
    WorkbookTriggerClause,
    PolicyEvaluationResult,
)




class WorkbookPolicyError(ValueError):
    pass


def compile_clinical_workbook(
    path: Path,
    catalog: dict[str, SupplyCatalogItem],
    supply_rules_path: Path | None = None,
) -> CompiledClinicalWorkbook:
    workbook_rows = _read_workbook_rows(path)
    if "Item Trigger Rules" not in workbook_rows:
        raise WorkbookPolicyError("Workbook is missing required sheet: Item Trigger Rules")
    if "Priority Ranking and Weight" not in workbook_rows:
        raise WorkbookPolicyError("Workbook is missing required sheet: Priority Ranking and Weight")

    csv_quantities = _load_csv_quantities(supply_rules_path) if supply_rules_path else {}
    priority_ranks = _compile_priority_sheet(workbook_rows["Priority Ranking and Weight"], catalog)
    item_rules = _compile_item_rules_sheet(workbook_rows["Item Trigger Rules"], catalog, priority_ranks, csv_quantities)
    return CompiledClinicalWorkbook(
        workbook_path=str(path),
        item_rules=item_rules,
        priority_ranks=priority_ranks,
    )


def evaluate_clinical_workbook(
    workbook: CompiledClinicalWorkbook,
    case: CanonicalCase,
    catalog: dict[str, SupplyCatalogItem],
) -> PolicyEvaluationResult:
    matches: list[RuleMatch] = []
    aggregated: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "quantity": 0,
            "rationale": [],
            "source_rule_ids": [],
            "policy_refs": [],
            "source_type": "rule",
            "llm_confidence": None,
        }
    )
    blocked_items: set[str] = set()
    allowed_items = {rule.item_id for rule in workbook.item_rules}
    priority_rank_map = {item_id: rank.priority_rank for item_id, rank in workbook.priority_ranks.items()}

    for rule in workbook.item_rules:
        if rule.unsupported_clauses:
            blocked_items.add(rule.item_id)
            allowed_items.discard(rule.item_id)
            continue

        triggered_by = _evaluate_item_rule(rule, case)
        if triggered_by is None:
            continue

        catalog_item = catalog[rule.item_id]
        matches.append(
            RuleMatch(
                rule_id=rule.rule_id,
                item_id=rule.item_id,
                quantity=rule.default_quantity,
                rationale=rule.clinical_rationale,
                triggered_by=triggered_by,
            )
        )
        entry = aggregated[rule.item_id]
        entry["quantity"] = int(entry["quantity"]) + rule.default_quantity
        entry["rationale"] = sorted(set([*entry["rationale"], rule.clinical_rationale]))
        entry["source_rule_ids"] = sorted(set([*entry["source_rule_ids"], rule.rule_id]))
        entry["policy_refs"] = sorted(set([*entry["policy_refs"], rule.rule_id]))
        entry["name"] = catalog_item.name
        entry["unit_weight_lb"] = catalog_item.unit_weight_lb

    must_include_items = [
        SupplyNeed(
            item_id=item_id,
            name=str(details["name"]),
            quantity=int(details["quantity"]),
            unit_weight_lb=float(details["unit_weight_lb"]),
            source_type="rule",
            policy_refs=list(details["policy_refs"]),
            llm_confidence=None,
            rationale=list(details["rationale"]),
            source_rule_ids=list(details["source_rule_ids"]),
        )
        for item_id, details in aggregated.items()
    ]
    must_include_items.sort(key=lambda need: (priority_rank_map.get(need.item_id, 999), need.name, need.item_id))

    return PolicyEvaluationResult(
        must_include_items=must_include_items,
        allowed_items=sorted(allowed_items),
        blocked_items=sorted(blocked_items),
        policy_matches=matches,
        priority_rank_map=priority_rank_map,
    )


def _read_workbook_rows(path: Path) -> dict[str, list[list[str]]]:
    if not path.exists():
        raise WorkbookPolicyError(f"Clinical workbook not found: {path}")

    with zipfile.ZipFile(path) as workbook_zip:
        workbook = ET.fromstring(workbook_zip.read("xl/workbook.xml"))
        ns = {
            "x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
            "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
        }
        rels = ET.fromstring(workbook_zip.read("xl/_rels/workbook.xml.rels"))
        rel_map = {rel.attrib["Id"]: rel.attrib["Target"] for rel in rels}
        shared_strings = _read_shared_strings(workbook_zip)

        sheet_rows: dict[str, list[list[str]]] = {}
        for sheet in workbook.find("x:sheets", ns):
            sheet_name = _canonical_sheet_name(sheet.attrib["name"])
            target = rel_map[sheet.attrib["{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"]]
            root = ET.fromstring(workbook_zip.read(f"xl/{target}"))
            rows = root.find("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}sheetData")
            sheet_rows[sheet_name] = [
                _read_row(row, shared_strings)
                for row in rows
            ]
        return sheet_rows


def _read_shared_strings(workbook_zip: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in workbook_zip.namelist():
        return []
    shared_root = ET.fromstring(workbook_zip.read("xl/sharedStrings.xml"))
    strings: list[str] = []
    for si in shared_root:
        texts = [t.text or "" for t in si.iter("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t")]
        strings.append("".join(texts))
    return strings


def _read_row(row: ET.Element, shared_strings: list[str]) -> list[str]:
    values: list[str] = []
    for cell in row:
        value = cell.find("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}v")
        if value is None:
            values.append("")
        elif cell.attrib.get("t") == "s":
            values.append(shared_strings[int(value.text)])
        else:
            values.append(value.text or "")
    return values


def _compile_priority_sheet(
    rows: list[list[str]],
    catalog: dict[str, SupplyCatalogItem],
) -> dict[str, CompiledPriorityRank]:
    compiled: dict[str, CompiledPriorityRank] = {}
    for row in rows[3:]:
        if len(row) < 4 or not row[0].strip():
            continue
        item_id = _canonical_item_id(row[2].strip())
        if item_id not in catalog:
            raise WorkbookPolicyError(f"Priority sheet item_id not found in catalog: {item_id}")
        compiled[item_id] = CompiledPriorityRank(
            item_id=item_id,
            item_name=row[1].strip(),
            priority_rank=int(float(row[0])),
            workbook_weight_lb=float(row[3]) if row[3].strip() else None,
            notes=row[4].strip() or None if len(row) > 4 else None,
        )
    return compiled


def _load_csv_quantities(path: Path) -> dict[str, int]:
    """Return a mapping of item_id -> max quantity from supply_rules.csv."""
    import csv
    quantities: dict[str, int] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            item_id = row["item_id"].strip()
            qty = int(row.get("quantity", "1"))
            quantities[item_id] = max(quantities.get(item_id, 0), qty)
    return quantities


def _compile_item_rules_sheet(
    rows: list[list[str]],
    catalog: dict[str, SupplyCatalogItem],
    priority_ranks: dict[str, CompiledPriorityRank],
    csv_quantities: dict[str, int] | None = None,
) -> list[CompiledItemRule]:
    compiled: list[CompiledItemRule] = []
    for index, row in enumerate(rows[3:], start=1):
        if len(row) < 8 or not row[0].strip():
            continue
        item_id = _canonical_item_id(row[2].strip())
        if item_id not in catalog:
            raise WorkbookPolicyError(f"Item trigger sheet item_id not found in catalog: {item_id}")
        logic_type = row[5].strip() or "ANY (OR)"
        any_clauses, all_clauses, unsupported_clauses = _compile_trigger_expression(row[4], logic_type)
        priority_rank = int(float(row[7])) if row[7].strip() else priority_ranks.get(item_id, CompiledPriorityRank(item_id=item_id, item_name=row[1].strip(), priority_rank=999)).priority_rank
        default_quantity = (csv_quantities or {}).get(item_id, 1)
        compiled.append(
            CompiledItemRule(
                rule_id=f"workbook-rule-{index}-{item_id}",
                item_id=item_id,
                item_name=row[1].strip(),
                default_quantity=default_quantity,
                logic_type=logic_type,
                priority_rank=priority_rank,
                clinical_rationale=row[6].strip(),
                trigger_text=row[4].strip(),
                any_clauses=any_clauses,
                all_clauses=all_clauses,
                unsupported_clauses=unsupported_clauses,
            )
        )
    return compiled


def _compile_trigger_expression(trigger_text: str, logic_type: str) -> tuple[list[WorkbookTriggerClause], list[WorkbookTriggerClause], list[str]]:
    normalized_logic = logic_type.strip().upper()
    if normalized_logic == "ALWAYS":
        return [], [], []

    lines = [line.strip(" ()") for line in trigger_text.splitlines() if line.strip()]
    any_clauses: list[WorkbookTriggerClause] = []
    all_clauses: list[WorkbookTriggerClause] = []
    unsupported_clauses: list[str] = []

    parsed_lines: list[WorkbookTriggerClause | None] = []
    for line in lines:
        cleaned = line.replace("AND ", "").replace("OR ", "").strip(" ()")
        clause = _parse_trigger_clause(cleaned)
        parsed_lines.append(clause)
        if clause is None:
            unsupported_clauses.append(cleaned)

    parsed_supported = [clause for clause in parsed_lines if clause is not None]
    if normalized_logic.startswith("ANY"):
        any_clauses.extend(parsed_supported)
        return any_clauses, all_clauses, unsupported_clauses
    if normalized_logic.startswith("ALL") and "NESTED" not in normalized_logic:
        all_clauses.extend(parsed_supported)
        return any_clauses, all_clauses, unsupported_clauses
    if "NESTED" in normalized_logic:
        if parsed_supported:
            all_clauses.append(parsed_supported[0])
            any_clauses.extend(parsed_supported[1:])
        return any_clauses, all_clauses, unsupported_clauses

    any_clauses.extend(parsed_supported)
    return any_clauses, all_clauses, unsupported_clauses


def _parse_trigger_clause(text: str) -> WorkbookTriggerClause | None:
    normalized = " ".join(text.replace("≤", "<=").replace("≥", ">=").split())

    match = re.fullmatch(r"march_flags includes '([^']+)'", normalized, flags=re.IGNORECASE)
    if match:
        return WorkbookTriggerClause(field_name="march_flags", operator="contains", value=_canonical_flag(match.group(1)))

    match = re.fullmatch(r"(seizure|vomiting|head_external_hemorrhage|suspected_icp)\s*==\s*(true|false)", normalized, flags=re.IGNORECASE)
    if match:
        return WorkbookTriggerClause(field_name=match.group(1), operator="eq", value=match.group(2).lower() == "true")

    numeric_patterns = (
        (r"(gcs_total|oxygen_saturation|systolic_bp|heart_rate|temp_c)\s*<=\s*(-?\d+(?:\.\d+)?)", "lte"),
        (r"(gcs_total|oxygen_saturation|systolic_bp|heart_rate|temp_c)\s*>=\s*(-?\d+(?:\.\d+)?)", "gte"),
        (r"(gcs_total|oxygen_saturation|systolic_bp|heart_rate|temp_c)\s*<\s*(-?\d+(?:\.\d+)?)", "lt"),
        (r"(gcs_total|oxygen_saturation|systolic_bp|heart_rate|temp_c)\s*>\s*(-?\d+(?:\.\d+)?)", "gt"),
    )
    for pattern, operator in numeric_patterns:
        match = re.fullmatch(pattern, normalized, flags=re.IGNORECASE)
        if match:
            raw_value = float(match.group(2))
            value: int | float = int(raw_value) if raw_value.is_integer() else raw_value
            return WorkbookTriggerClause(field_name=match.group(1), operator=operator, value=value)

    if normalized.lower() == "include for all tbi cases":
        return None
    return None


def _evaluate_item_rule(rule: CompiledItemRule, case: CanonicalCase) -> list[str] | None:
    if rule.logic_type.strip().upper() == "ALWAYS":
        return ["include_mode=ALWAYS"]

    triggered_all = [_evaluate_clause(clause, case) for clause in rule.all_clauses]
    if any(result is None for result in triggered_all):
        return None

    triggered_any: list[str] = []
    if rule.any_clauses:
        evaluated_any = [_evaluate_clause(clause, case) for clause in rule.any_clauses]
        triggered_any = [result for result in evaluated_any if result is not None]
        if not triggered_any:
            return None

    return [*triggered_all, *triggered_any]


def _evaluate_clause(clause: WorkbookTriggerClause, case: CanonicalCase) -> str | None:
    case_value = _get_case_value(case, clause.field_name)
    if case_value is None:
        return None

    if clause.operator == "contains":
        expected = str(clause.value)
        if isinstance(case_value, list) and expected in case_value:
            return f"{clause.field_name}={expected}"
        if isinstance(case_value, str) and expected.lower() in case_value.lower():
            return f"{clause.field_name}~{expected}"
        return None

    if clause.operator == "eq":
        if case_value == clause.value:
            return f"{clause.field_name}={str(case_value).lower() if isinstance(case_value, bool) else case_value}"
        return None

    numeric_value = float(case_value)
    expected = float(clause.value)
    comparisons = {
        "lte": numeric_value <= expected,
        "gte": numeric_value >= expected,
        "lt": numeric_value < expected,
        "gt": numeric_value > expected,
    }
    if comparisons.get(clause.operator):
        return f"{clause.field_name}={case_value}"
    return None


def _get_case_value(case: CanonicalCase, field_name: str) -> Any:
    if field_name == "gcs_total":
        return case.vitals.gcs
    if field_name == "oxygen_saturation":
        return case.vitals.spo2
    if field_name == "systolic_bp":
        return case.vitals.systolic_bp
    if field_name == "heart_rate":
        return case.vitals.heart_rate
    if field_name == "temp_c":
        return case.vitals.temperature_c
    if field_name == "march_flags":
        return case.march_flags
    return getattr(case, field_name, None)


def _canonical_item_id(item_id: str) -> str:
    return item_id.strip()


def _canonical_sheet_name(sheet_name: str) -> str:
    normalized = sheet_name
    while True:
        unescaped = unescape(normalized)
        if unescaped == normalized:
            break
        normalized = unescaped

    normalized = normalized.strip()
    if normalized == "Priority Ranking & Weight":
        return "Priority Ranking and Weight"
    return normalized


def _canonical_flag(flag: str) -> str:
    return flag.strip().lower().replace(" ", "_").replace("-", "_")
