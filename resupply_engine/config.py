from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    base_dir: Path = Path(__file__).resolve().parent.parent
    data_dir: Path = Path(__file__).resolve().parent.parent / "data"
    db_path: Path = Path(__file__).resolve().parent.parent / "resupply.db"
    max_drone_payload_lb: float = 10.0
    allowed_shootdown_rates: tuple[int, ...] = (0, 10, 25, 50, 75, 90)
    default_target_arrival_probability: float = 0.95
    max_total_drones: int = 24
    llm_extraction_confidence_threshold: float = 0.65
