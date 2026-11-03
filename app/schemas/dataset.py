from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class DatasetColumn(BaseModel):
    key: str = Field(min_length=1, max_length=80, pattern=r"^[A-Za-z0-9_-]+$")
    name: str = Field(min_length=1, max_length=120)
    data_type: Literal["number", "text", "date"] = "number"


class DatasetPayload(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    columns: list[DatasetColumn] = Field(min_length=1, max_length=500)
    rows: list[dict[str, Any]] = Field(default_factory=list, max_length=100_000)

    @model_validator(mode="after")
    def validate_tabular_shape(self) -> "DatasetPayload":
        keys = [column.key for column in self.columns]
        if len(keys) != len(set(keys)):
            raise ValueError("Las claves de columna deben ser unicas")

        allowed_keys = set(keys)
        if any(set(row) - allowed_keys for row in self.rows):
            raise ValueError("Los datos contienen columnas no declaradas")
        return self


class DatasetCreate(DatasetPayload):
    pass


class DatasetUpdate(DatasetPayload):
    pass


class DatasetRead(DatasetPayload):
    id: int
    owner_email: str
    row_count: int
    revision: int
    content_sha256: str
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class DatasetSummary(BaseModel):
    id: int
    name: str
    row_count: int
    column_count: int
    updated_at: datetime
