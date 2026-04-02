from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    base_dir: Path = Path(__file__).resolve().parent.parent
    data_dir: Path = Path(__file__).resolve().parent.parent / "data"
    db_path: Path = Path(__file__).resolve().parent.parent / "resupply.db"
    exports_dir: Path = Path(__file__).resolve().parent.parent / "exports"
    clinical_workbook_path: Path = Path(__file__).resolve().parent.parent / "data" / "TBI_Clinical_Rules.xlsx"
    max_drone_payload_lb: float = 10.0
    allowed_shootdown_rates: tuple[int, ...] = (0, 10, 25, 50, 75, 90)
    default_target_arrival_probability: float = 0.95
    max_total_drones: int = 24
    llm_extraction_confidence_threshold: float = 0.65
    llm_enabled: bool = True
    llm_backend: str = "ollama"
    llm_model_path: Path | None = None
    ollama_base_url: str = "http://127.0.0.1:11434/api"
    ollama_model: str = "qwen2.5:1.5b"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 600
    llm_timeout_seconds: int = 30
