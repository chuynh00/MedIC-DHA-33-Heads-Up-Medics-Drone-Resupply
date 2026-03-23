from __future__ import annotations

import math
from dataclasses import dataclass

from resupply_engine.models import DroneItemAllocation, DroneManifest, SupplyCatalogItem, SupplyNeed


@dataclass(frozen=True)
class UnitItem:
    unit_id: str
    item_id: str
    name: str
    weight: float


class PackingError(ValueError):
    pass


def pack_supply_needs(
    needs: list[SupplyNeed],
    max_payload_lb: float,
) -> list[DroneManifest]:
    units: list[UnitItem] = []
    for need in needs:
        if need.unit_weight_lb > max_payload_lb:
            raise PackingError(
                f"Item {need.item_id} weighs {need.unit_weight_lb} lb and cannot fit in a single drone."
            )
        for index in range(need.quantity):
            units.append(
                UnitItem(
                    unit_id=f"{need.item_id}-{index}",
                    item_id=need.item_id,
                    name=need.name,
                    weight=need.unit_weight_lb,
                )
            )

    if not units:
        return []

    units.sort(key=lambda item: item.weight, reverse=True)
    lower_bound = max(1, math.ceil(sum(unit.weight for unit in units) / max_payload_lb))
    for bin_count in range(lower_bound, len(units) + 1):
        manifests = _search_exact_pack(units, bin_count, max_payload_lb)
        if manifests is not None:
            return manifests
    raise PackingError("Unable to compute a valid drone packing solution.")


def manifests_from_operator_input(
    edited_manifests: list,
    catalog: dict[str, SupplyCatalogItem],
    max_payload_lb: float,
) -> tuple[list[DroneManifest], list[SupplyNeed]]:
    manifests: list[DroneManifest] = []
    aggregated: dict[str, SupplyNeed] = {}
    for manifest in edited_manifests:
        items: list[DroneItemAllocation] = []
        total_weight = 0.0
        for entry in manifest.items:
            if entry.item_id not in catalog:
                raise PackingError(f"Unknown catalog item: {entry.item_id}")
            catalog_item = catalog[entry.item_id]
            total_weight += catalog_item.unit_weight_lb * entry.quantity
            items.append(
                DroneItemAllocation(
                    item_id=entry.item_id,
                    name=catalog_item.name,
                    quantity=entry.quantity,
                    unit_weight_lb=catalog_item.unit_weight_lb,
                )
            )
            if entry.item_id not in aggregated:
                aggregated[entry.item_id] = SupplyNeed(
                    item_id=entry.item_id,
                    name=catalog_item.name,
                    quantity=0,
                    unit_weight_lb=catalog_item.unit_weight_lb,
                    rationale=["Operator override"],
                    source_rule_ids=["operator_override"],
                )
            aggregated[entry.item_id].quantity += entry.quantity
        if total_weight > max_payload_lb + 1e-9:
            raise PackingError(
                f"Drone manifest {manifest.manifest_id} exceeds max payload {max_payload_lb} lb."
            )
        manifests.append(
            DroneManifest(
                manifest_id=manifest.manifest_id,
                total_weight_lb=round(total_weight, 2),
                items=items,
            )
        )
    needs = sorted(aggregated.values(), key=lambda need: (need.name, need.item_id))
    return manifests, needs


def _search_exact_pack(
    units: list[UnitItem],
    bin_count: int,
    max_payload_lb: float,
) -> list[DroneManifest] | None:
    remaining = [max_payload_lb] * bin_count
    contents: list[list[UnitItem]] = [[] for _ in range(bin_count)]

    def place(index: int) -> bool:
        if index == len(units):
            return True

        unit = units[index]
        seen_capacities: set[float] = set()
        for bin_index in range(bin_count):
            capacity = remaining[bin_index]
            rounded_capacity = round(capacity, 5)
            if rounded_capacity in seen_capacities:
                continue
            if capacity + 1e-9 < unit.weight:
                continue

            seen_capacities.add(rounded_capacity)
            remaining[bin_index] -= unit.weight
            contents[bin_index].append(unit)
            if place(index + 1):
                return True
            contents[bin_index].pop()
            remaining[bin_index] += unit.weight

            if abs(capacity - max_payload_lb) < 1e-9:
                break
        return False

    if not place(0):
        return None

    manifests: list[DroneManifest] = []
    manifest_index = 1
    for units_in_bin in contents:
        if not units_in_bin:
            continue
        grouped: dict[str, DroneItemAllocation] = {}
        total_weight = 0.0
        for unit in units_in_bin:
            total_weight += unit.weight
            if unit.item_id not in grouped:
                grouped[unit.item_id] = DroneItemAllocation(
                    item_id=unit.item_id,
                    name=unit.name,
                    quantity=0,
                    unit_weight_lb=unit.weight,
                )
            grouped[unit.item_id].quantity += 1
        manifests.append(
            DroneManifest(
                manifest_id=manifest_index,
                total_weight_lb=round(total_weight, 2),
                items=sorted(grouped.values(), key=lambda item: (item.name, item.item_id)),
            )
        )
        manifest_index += 1
    return manifests
