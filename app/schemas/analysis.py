from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


class DescriptiveAnalysisRequest(BaseModel):
    columns: list[str] = Field(min_length=1, max_length=100)
    group_by: str | None = None
    confidence_level: float = Field(default=0.95, gt=0.5, lt=1.0)
    percentiles: list[float] = Field(default_factory=lambda: [0.25, 0.5, 0.75], max_length=20)
    histogram_bins: int = Field(default=10, ge=3, le=100)
    frequency_limit: int = Field(default=25, ge=1, le=100)

    @field_validator("columns")
    @classmethod
    def unique_columns(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("Cada columna se debe seleccionar una sola vez")
        return value

    @field_validator("percentiles")
    @classmethod
    def valid_percentiles(cls, value: list[float]) -> list[float]:
        if any(item <= 0 or item >= 1 for item in value):
            raise ValueError("Los percentiles deben estar entre 0 y 1")
        return sorted(set(value))

    @model_validator(mode="after")
    def group_is_not_analyzed_column(self) -> "DescriptiveAnalysisRequest":
        if self.group_by and self.group_by in self.columns:
            raise ValueError("La variable de agrupación no puede ser una variable analizada")
        return self


class AnalysisRunRead(BaseModel):
    id: int
    dataset_id: int
    dataset_revision: int
    dataset_sha256: str
    analysis_type: str
    status: str
    engine_version: str
    parameters: dict[str, Any]
    result: dict[str, Any]
    warnings: list[str]
    created_at: datetime
    is_stale: bool


class AnalysisRunSummary(BaseModel):
    id: int
    dataset_id: int
    dataset_revision: int
    analysis_type: str
    status: str
    created_at: datetime
    is_stale: bool
