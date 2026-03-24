from __future__ import annotations

import re
from pathlib import Path

from resupply_engine.models import DispatchPlan, ExportMetadata, ReviewFlag


def _sanitize_identifier(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    return sanitized.strip("-") or "unknown"


def render_operator_text(plan: DispatchPlan) -> str:
    required_supplies_lines = [
        f"- {need.quantity}x {need.name} ({need.unit_weight_lb} lb each)"
        for need in plan.required_supplies
    ] or ["- none"]

    base_manifest_lines = []
    for manifest in plan.base_manifests:
        item_text = ", ".join(f"{item.quantity}x {item.name}" for item in manifest.items) or "no items"
        base_manifest_lines.append(
            f"- Drone {manifest.manifest_id}: {manifest.total_weight_lb} lb | {item_text}"
        )
    if not base_manifest_lines:
        base_manifest_lines = ["- none"]

    review_flag_lines = [f"- [{flag.severity.upper()}] {flag.message}" for flag in plan.review_flags] or ["- none"]

    return "\n".join(
        [
            "MEDIC DRONE RESUPPLY DISPATCH BRIEF",
            "",
            f"Patient ID: {plan.canonical_case.patient_id}",
            f"Mission ID: {plan.canonical_case.mission_id}",
            f"Medic ID: {plan.canonical_case.medic_id}",
            f"This plan is the response to burst ID: {plan.burst_id}",
            f"Burst timestamp_utc: {plan.canonical_case.occurred_at.isoformat()}",
            f"Shootdown rate: {plan.shootdown_rate}%",
            f"Target arrival probability: {plan.target_arrival_probability:.2%}",
            "",
            "Required Supplies:",
            *required_supplies_lines,
            "",
            "Base Manifest:",
            *base_manifest_lines,
            "",
            f"Redundancy multiplier: {plan.redundancy_multiplier}",
            f"Per item arrival probability: {plan.per_item_arrival_probability:.2%}",
            f"Total drones to be sent: {plan.total_drones}",
            "",
            "Review Flags:",
            *review_flag_lines,
        ]
    )


class OperatorTextExporter:
    def __init__(self, exports_dir: Path) -> None:
        self.exports_dir = exports_dir

    def export(self, plan: DispatchPlan, revision: int) -> ExportMetadata:
        plan_dir = self.exports_dir / _sanitize_identifier(plan.plan_id)
        plan_dir.mkdir(parents=True, exist_ok=True)
        mission_id = _sanitize_identifier(plan.canonical_case.mission_id)
        burst_id = _sanitize_identifier(plan.burst_id)
        filename = f"mission-{mission_id}__burst-{burst_id}__rev-{revision:03d}.txt"
        export_path = plan_dir / filename
        export_path.write_text(render_operator_text(plan), encoding="utf-8")
        return ExportMetadata(
            export_path=str(export_path),
            export_format="txt",
            export_revision=revision,
        )


def add_export_failure_flag(review_flags: list[ReviewFlag], message: str) -> list[ReviewFlag]:
    updated = list(review_flags)
    updated.append(
        ReviewFlag(
            code="text_export_write_failed",
            severity="warning",
            message=message,
        )
    )
    return updated
