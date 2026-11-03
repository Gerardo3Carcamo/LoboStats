from __future__ import annotations

import math
from collections import Counter
from datetime import date, datetime
from typing import Any

import numpy as np
from scipy import stats

from app.models.dataset import Dataset
from app.schemas.analysis import DescriptiveAnalysisRequest

ENGINE_VERSION = "descriptive-1.0"
MAX_QQ_POINTS = 200


class DescriptiveAnalysisError(ValueError):
    pass


def analyze_descriptive(
    dataset: Dataset, request: DescriptiveAnalysisRequest
) -> tuple[dict[str, Any], list[str]]:
    columns = {column["key"]: column for column in dataset.columns_data}
    requested_keys = [*request.columns, *([request.group_by] if request.group_by else [])]
    missing = [key for key in requested_keys if key not in columns]
    if missing:
        raise DescriptiveAnalysisError(
            f"Columnas no encontradas: {', '.join(sorted(set(missing)))}"
        )

    groups = _group_rows(dataset.rows_data, request.group_by)
    warnings: list[str] = []
    group_results = []
    for group_value, rows in groups:
        analyzed_columns = []
        for key in request.columns:
            column = columns[key]
            result, column_warnings = _analyze_column(
                column=column,
                rows=rows,
                confidence_level=request.confidence_level,
                percentiles=request.percentiles,
                histogram_bins=request.histogram_bins,
                frequency_limit=request.frequency_limit,
            )
            analyzed_columns.append(result)
            warnings.extend(
                f"{column['name']}{_group_suffix(group_value, request.group_by)}: {message}"
                for message in column_warnings
            )
        group_results.append(
            {
                "value": _json_scalar(group_value),
                "label": _group_label(group_value, request.group_by),
                "row_count": len(rows),
                "columns": analyzed_columns,
            }
        )

    return (
        {
            "dataset": {
                "id": dataset.id,
                "name": dataset.name,
                "revision": dataset.revision,
                "content_sha256": dataset.content_sha256,
                "row_count": dataset.row_count,
            },
            "selection": {
                "columns": request.columns,
                "group_by": request.group_by,
                "confidence_level": request.confidence_level,
                "percentiles": request.percentiles,
                "histogram_bins": request.histogram_bins,
                "frequency_limit": request.frequency_limit,
            },
            "groups": group_results,
            "guidance": [
                "Los cálculos numéricos excluyen valores faltantes y reportan cuántas "
                "observaciones se utilizaron.",
                "La desviación estándar y la varianza usan el estimador muestral (n - 1).",
                "Una descripción resume la muestra; no demuestra causalidad ni "
                "significancia estadística.",
            ],
        },
        list(dict.fromkeys(warnings)),
    )


def _group_rows(
    rows: list[dict[str, Any]], group_by: str | None
) -> list[tuple[Any, list[dict[str, Any]]]]:
    if not group_by:
        return [(None, rows)]
    grouped: dict[tuple[str, str], tuple[Any, list[dict[str, Any]]]] = {}
    for row in rows:
        value = row.get(group_by)
        identity = (type(value).__name__, repr(value))
        if identity not in grouped:
            grouped[identity] = (value, [])
        grouped[identity][1].append(row)
    return sorted(grouped.values(), key=lambda item: (item[0] is None, str(item[0]).casefold()))


def _analyze_column(
    column: dict[str, Any],
    rows: list[dict[str, Any]],
    confidence_level: float,
    percentiles: list[float],
    histogram_bins: int,
    frequency_limit: int,
) -> tuple[dict[str, Any], list[str]]:
    values = [row.get(column["key"]) for row in rows]
    present = [value for value in values if not _is_missing(value)]
    base = {
        "key": column["key"],
        "name": column["name"],
        "data_type": column["data_type"],
        "total_count": len(values),
        "valid_count": len(present),
        "missing_count": len(values) - len(present),
    }
    if column["data_type"] == "number":
        numeric, rejected = _numeric_values(present)
        summary, warnings = _numeric_summary(numeric, confidence_level, percentiles, histogram_bins)
        if rejected:
            warnings.append(f"se excluyeron {rejected} valores no numéricos o no finitos")
        base["valid_count"] = len(numeric)
        base["missing_count"] = len(values) - len(numeric)
        return {**base, "numeric": summary, "categorical": None, "date": None}, warnings
    if column["data_type"] == "date":
        return {
            **base,
            "numeric": None,
            "categorical": _frequency_summary(present, frequency_limit),
            "date": _date_summary(present),
        }, []
    return {
        **base,
        "numeric": None,
        "categorical": _frequency_summary(present, frequency_limit),
        "date": None,
    }, []


def _numeric_values(values: list[Any]) -> tuple[list[float], int]:
    numeric: list[float] = []
    rejected = 0
    for value in values:
        if isinstance(value, bool):
            rejected += 1
            continue
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            rejected += 1
            continue
        if not math.isfinite(parsed):
            rejected += 1
            continue
        numeric.append(parsed)
    return numeric, rejected


def _numeric_summary(
    values: list[float],
    confidence_level: float,
    percentiles: list[float],
    histogram_bins: int,
) -> tuple[dict[str, Any] | None, list[str]]:
    if not values:
        return None, ["no hay observaciones numéricas disponibles"]
    ordered = sorted(values)
    array = np.asarray(ordered, dtype=float)
    count = len(ordered)
    mean = float(np.mean(array))
    median = float(np.median(array))
    variance = float(np.var(array, ddof=1)) if count > 1 else None
    std_dev = math.sqrt(variance) if variance is not None else None
    std_error = std_dev / math.sqrt(count) if std_dev is not None else None
    q1 = _interpolated_quantile(ordered, 0.25)
    q3 = _interpolated_quantile(ordered, 0.75)
    iqr = q3 - q1
    frequencies = Counter(ordered)
    mode_count = max(frequencies.values())
    modes = sorted(value for value, frequency in frequencies.items() if frequency == mode_count)
    if mode_count == 1:
        modes = []
    confidence_interval = _mean_confidence_interval(mean, std_error, count, confidence_level)
    skewness = _finite_or_none(stats.skew(array, bias=False)) if count >= 3 else None
    kurtosis = (
        _finite_or_none(stats.kurtosis(array, fisher=True, bias=False)) if count >= 4 else None
    )
    counts, edges = np.histogram(array, bins=histogram_bins)
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr
    inside = [value for value in ordered if lower_fence <= value <= upper_fence]
    qq = _qq_points(array)
    warnings = []
    if count == 1:
        warnings.append("la varianza, la desviación y el intervalo requieren al menos 2 valores")
    if np.ptp(array) == 0:
        warnings.append("la columna es constante; asimetría y curtosis no están definidas")
        skewness = None
        kurtosis = None
    return {
        "mean": mean,
        "median": median,
        "modes": modes,
        "mode_frequency": mode_count if modes else None,
        "variance": variance,
        "std_dev": std_dev,
        "std_error": std_error,
        "coefficient_of_variation": (
            abs(std_dev / mean) * 100 if std_dev is not None and mean != 0 else None
        ),
        "minimum": ordered[0],
        "maximum": ordered[-1],
        "range": ordered[-1] - ordered[0],
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "sum": float(np.sum(array)),
        "sum_squares": float(np.sum(np.square(array))),
        "skewness": skewness,
        "kurtosis_excess": kurtosis,
        "percentiles": {
            _percentile_key(percentile): _interpolated_quantile(ordered, percentile)
            for percentile in percentiles
        },
        "confidence_interval_mean": confidence_interval,
        "histogram": {
            "counts": [int(value) for value in counts],
            "edges": [float(value) for value in edges],
        },
        "boxplot": {
            "lower_whisker": inside[0] if inside else ordered[0],
            "q1": q1,
            "median": median,
            "q3": q3,
            "upper_whisker": inside[-1] if inside else ordered[-1],
            "outliers": [value for value in ordered if value < lower_fence or value > upper_fence][
                :200
            ],
        },
        "qq_plot": qq,
    }, warnings


def _mean_confidence_interval(
    mean: float, std_error: float | None, count: int, confidence_level: float
) -> dict[str, float] | None:
    if std_error is None or count < 2:
        return None
    if std_error == 0:
        return {"level": confidence_level, "lower": mean, "upper": mean}
    critical = float(stats.t.ppf((1 + confidence_level) / 2, count - 1))
    margin = critical * std_error
    return {"level": confidence_level, "lower": mean - margin, "upper": mean + margin}


def _interpolated_quantile(ordered: list[float], probability: float) -> float:
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) + 1)
    if position <= 1:
        return ordered[0]
    if position >= len(ordered):
        return ordered[-1]
    lower_index = math.floor(position) - 1
    fraction = position - math.floor(position)
    return ordered[lower_index] + fraction * (ordered[lower_index + 1] - ordered[lower_index])


def _qq_points(array: np.ndarray) -> dict[str, list[float]]:
    theoretical, observed = stats.probplot(array, dist="norm", fit=False)
    theoretical_array = np.asarray(theoretical, dtype=float)
    observed_array = np.asarray(observed, dtype=float)
    if len(theoretical_array) > MAX_QQ_POINTS:
        indices = np.linspace(0, len(theoretical_array) - 1, MAX_QQ_POINTS, dtype=int)
        theoretical_array = theoretical_array[indices]
        observed_array = observed_array[indices]
    return {
        "theoretical": [float(value) for value in theoretical_array],
        "observed": [float(value) for value in observed_array],
    }


def _frequency_summary(values: list[Any], limit: int) -> dict[str, Any]:
    normalized = [_display_value(value) for value in values]
    counts = Counter(normalized)
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0].casefold()))
    total = len(normalized)
    rows = [
        {
            "value": value,
            "count": count,
            "percentage": (count / total * 100) if total else 0,
        }
        for value, count in ordered[:limit]
    ]
    return {
        "unique_count": len(counts),
        "mode": ordered[0][0] if ordered else None,
        "mode_frequency": ordered[0][1] if ordered else None,
        "frequencies": rows,
        "truncated": len(ordered) > limit,
    }


def _date_summary(values: list[Any]) -> dict[str, str | None]:
    rendered = sorted(_display_value(value) for value in values)
    return {
        "minimum": rendered[0] if rendered else None,
        "maximum": rendered[-1] if rendered else None,
    }


def _is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def _display_value(value: Any) -> str:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return str(value)


def _json_scalar(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return str(value)


def _group_label(value: Any, group_by: str | None) -> str:
    if not group_by:
        return "Todos los datos"
    if _is_missing(value):
        return "Valores faltantes"
    return _display_value(value)


def _group_suffix(value: Any, group_by: str | None) -> str:
    return f" — {_group_label(value, group_by)}" if group_by else ""


def _finite_or_none(value: Any) -> float | None:
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


def _percentile_key(value: float) -> str:
    return f"p{value * 100:g}"
