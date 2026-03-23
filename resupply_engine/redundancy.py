from __future__ import annotations

import math


def calculate_redundancy_multiplier(shootdown_rate: int, target_arrival_probability: float) -> tuple[int, float]:
    survival_probability = 1 - (shootdown_rate / 100)
    if survival_probability <= 0:
        raise ValueError("Shootdown rate leaves zero survival probability.")
    if survival_probability >= 1:
        return 1, 1.0
    if target_arrival_probability <= survival_probability:
        return 1, survival_probability

    multiplier = math.ceil(
        math.log(1 - target_arrival_probability) / math.log(1 - survival_probability)
    )
    item_arrival_probability = 1 - ((1 - survival_probability) ** multiplier)
    return multiplier, round(item_arrival_probability, 6)
