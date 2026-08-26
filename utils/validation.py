from typing import Any

import pandas as pd
from flask import request

from modules.dataset_store import Dataset
from utils.errors import APIError


def get_json_object() -> dict:
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        raise APIError("El cuerpo debe ser un objeto JSON.", code="invalid_json")
    return payload


def require_fields(payload: dict, *fields: str) -> None:
    missing = [field for field in fields if field not in payload]
    if missing:
        raise APIError("Faltan campos requeridos.", details={"fields": missing})


def validate_columns(dataset: Dataset, columns: list[Any]) -> None:
    invalid = [column for column in columns if not isinstance(column, str) or column not in dataset.frame.columns]
    if invalid:
        raise APIError("Una o más columnas no existen.", details={"columns": invalid})


def numeric_series(dataset: Dataset, column: str, minimum: int = 1) -> pd.Series:
    validate_columns(dataset, [column])
    original = dataset.frame[column].dropna()
    series = pd.to_numeric(original, errors="coerce").dropna().astype(float)
    if len(series) < minimum:
        raise APIError(
            f"La columna '{column}' requiere al menos {minimum} valores numéricos.",
            details={"column": column, "minimum": minimum},
        )
    return series


def alpha_value(payload: dict) -> float:
    alpha = payload.get("alpha", 0.05)
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not 0 < alpha < 1:
        raise APIError("'alpha' debe ser un número entre 0 y 1.", details={"field": "alpha"})
    return float(alpha)


def choice(payload: dict, field: str, allowed: set[str], default: str | None = None) -> str:
    value = payload.get(field, default)
    if not isinstance(value, str) or value not in allowed:
        raise APIError(
            f"'{field}' debe ser uno de: {', '.join(sorted(allowed))}.",
            details={"field": field, "allowed": sorted(allowed)},
        )
    return value


def number_value(
    payload: dict, field: str, *, default=None, minimum: float | None = None,
    maximum: float | None = None, integer: bool = False,
):
    value = payload.get(field, default)
    valid_type = isinstance(value, int) if integer else isinstance(value, (int, float))
    if isinstance(value, bool) or not valid_type:
        raise APIError(f"'{field}' debe ser {'entero' if integer else 'numérico'}.", details={"field": field})
    if minimum is not None and value < minimum or maximum is not None and value > maximum:
        raise APIError(f"'{field}' está fuera del rango permitido.", details={"field": field})
    return int(value) if integer else float(value)


def column_list(dataset: Dataset, payload: dict, field: str = "columns", *, minimum: int = 1, maximum: int = 20) -> list[str]:
    columns = payload.get(field)
    if not isinstance(columns, list) or not minimum <= len(columns) <= maximum:
        raise APIError(f"'{field}' debe contener entre {minimum} y {maximum} columnas.", details={"field": field})
    if len(set(columns)) != len(columns):
        raise APIError(f"'{field}' no debe contener duplicados.", details={"field": field})
    validate_columns(dataset, columns)
    return columns
