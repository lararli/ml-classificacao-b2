"""Loads and validates YAML configs using dataclasses."""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p.resolve()}")
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@dataclass
class DataConfig:
    source: str
    target: str
    numerical: list[str]
    categorical: list[str]
    file_pattern: str = "*.csv"

    @classmethod
    def load(cls, path: str = "config/data.yaml") -> "DataConfig":
        raw = load_yaml(path)
        raw["numerical"] = raw.pop("numericas", raw.get("numerical", []))
        raw["categorical"] = raw.pop("categoricas", raw.get("categorical", []))
        return cls(**raw)

    @property
    def all_features(self) -> list[str]:
        return self.numerical + self.categorical

    @property
    def all_columns(self) -> list[str]:
        return self.all_features + [self.target]


@dataclass
class PathsConfig:
    raw_data_dir: str
    processed_data_dir: str


@dataclass
class DriftConfig:
    historical_fraction: float = 0.7
    threshold_pct: float = 10.0


@dataclass
class PipelineConfig:
    paths: PathsConfig
    random_state: int
    test_size: float = 0.2
    stratify: bool = True
    drift: DriftConfig = field(default_factory=DriftConfig)

    @classmethod
    def load(cls, path: str = "config/pipeline.yaml") -> "PipelineConfig":
        raw = load_yaml(path)
        raw["paths"] = PathsConfig(**raw["paths"])
        if "drift" in raw:
            raw["drift"] = DriftConfig(**raw["drift"])
        return cls(**raw)


@dataclass
class TableExpectation:
    min_rows: int
    max_rows: int


@dataclass
class ColumnExpectation:
    min_value: float | None = None
    max_value: float | None = None
    nullable: bool = True
    allowed_values: list | None = None


@dataclass
class QualityConfig:
    table: TableExpectation
    columns: dict[str, ColumnExpectation] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str = "config/quality.yaml") -> "QualityConfig":
        raw = load_yaml(path)
        return cls(
            table=TableExpectation(**raw["table"]),
            columns={col: ColumnExpectation(**exp) for col, exp in raw.get("columns", {}).items()},
        )


@dataclass
class ModelSpec:
    name: str
    model_class: str
    params: dict[str, Any]
    search_params: dict[str, Any] | None = None
    reduction: str | None = None


@dataclass
class ExperimentsConfig:
    models: list[ModelSpec]

    @classmethod
    def load(cls, mode: str = "production", config_dir: str = "config") -> "ExperimentsConfig":
        filename = f"experiments_{'prod' if mode == 'production' else 'test'}.yaml"
        raw = load_yaml(f"{config_dir}/{filename}")
        return cls(models=[ModelSpec(**m) for m in raw["models"]])


def resolve_execution_id(execution_id: str | None = None) -> str:
    return execution_id if execution_id else datetime.now().strftime("%m%Y")
