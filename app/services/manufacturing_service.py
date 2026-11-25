from __future__ import annotations

import math
from collections import defaultdict
from datetime import datetime
from typing import Any

import numpy as np

from app.models.dataset import Dataset
from app.schemas.manufacturing import ManufacturingAnalysisRequest

ENGINE_VERSION = "manufacturing-1.0"


class ManufacturingAnalysisError(ValueError):
    pass


METRIC_CATALOG = [
    {"key": "cycle", "label": "Tiempos de ciclo", "requires": ["cycle_time"]},
    {
        "key": "downtime",
        "label": "Tiempos muertos",
        "requires": ["event_state", "duration | start+end"],
    },
    {
        "key": "quality",
        "label": "Calidad",
        "requires": ["total_count", "good_count | quality_result"],
    },
    {
        "key": "oee",
        "label": "OEE",
        "requires": [
            "planned_time",
            "operating_time",
            "ideal_cycle_time",
            "total_count",
            "good_count",
        ],
    },
    {
        "key": "capability",
        "label": "Capacidad de proceso",
        "requires": ["measurement", "LSL", "USL"],
    },
]


def manufacturing_catalog() -> list[dict[str, Any]]:
    return METRIC_CATALOG


def analyze_manufacturing(
    dataset: Dataset, request: ManufacturingAnalysisRequest
) -> tuple[dict[str, Any], list[str]]:
    columns = {column["key"] for column in dataset.columns_data}
    mapped = {
        value for value in request.mapping.model_dump().values() if isinstance(value, str) and value
    }
    filter_columns = set(request.filters)
    missing = sorted((mapped | filter_columns) - columns)
    if missing:
        raise ManufacturingAnalysisError(f"Columnas no encontradas: {', '.join(missing)}")

    rows = _filtered_rows(dataset.rows_data, request.filters)
    if not rows:
        raise ManufacturingAnalysisError("Los filtros no dejaron filas para analizar")
    warnings: list[str] = []

    if request.mode == "summary":
        sections, mode_warnings = _analyze_summary(rows, request)
    elif request.mode == "event_log":
        sections, mode_warnings = _analyze_events(rows, request)
    else:
        sections, mode_warnings = _analyze_cycles(rows, request)
    warnings.extend(mode_warnings)

    cycle = sections.get("cycle") or _cycle_from_available_rows(rows, request)
    if cycle:
        sections["cycle"] = cycle
    quality = sections.get("quality") or _quality_from_available_rows(rows, request)
    if quality:
        sections["quality"] = quality
    capability = _capability_from_rows(rows, request)
    if capability:
        sections["capability"] = capability

    sections.setdefault("cycle", None)
    sections.setdefault("downtime", None)
    sections.setdefault("quality", None)
    sections.setdefault("oee", None)
    sections.setdefault("capability", None)

    availability = {
        key: value is not None
        for key, value in sections.items()
        if key in {"cycle", "downtime", "quality", "oee", "capability"}
    }
    missing_metrics = [key for key, available in availability.items() if not available]
    if missing_metrics:
        warnings.append(
            "No se calcularon métricas sin datos suficientes: " + ", ".join(missing_metrics) + "."
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
            "scope": {
                "mode": request.mode,
                "analyzed_rows": len(rows),
                "filters": request.filters,
                "time_unit": request.time_unit,
                "cycle_time_unit": request.cycle_time_unit,
            },
            "availability": availability,
            **sections,
            "guidance": [
                "OEE requiere tiempo programado, tiempo operativo, ciclo ideal "
                "y conteos de calidad.",
                "Los paros superpuestos deben corregirse en el origen para evitar doble conteo.",
                "MTBF y MTTR describen el historial observado; no predicen por sí "
                "solos la próxima falla.",
                "Capacidad de proceso requiere un sistema de medición confiable y "
                "un proceso estable.",
            ],
        },
        list(dict.fromkeys(warnings)),
    )


def _analyze_summary(
    rows: list[dict[str, Any]], request: ManufacturingAnalysisRequest
) -> tuple[dict[str, Any], list[str]]:
    mapping = request.mapping
    factor = _time_factor(request.time_unit)
    warnings: list[str] = []
    planned = _sum_numeric(rows, mapping.planned_time) * factor
    planned_stop = _sum_numeric(rows, mapping.planned_downtime) * factor
    unplanned_stop = _sum_numeric(rows, mapping.unplanned_downtime) * factor
    operating_explicit = _sum_numeric(rows, mapping.operating_time) * factor
    if operating_explicit > 0:
        operating = operating_explicit
    else:
        operating = max(0.0, planned - planned_stop - unplanned_stop)
    unclassified_stop = max(0.0, planned - operating - planned_stop - unplanned_stop)
    if operating > planned and planned > 0:
        warnings.append(
            "El tiempo operativo supera al programado; revise unidades o solapamientos."
        )

    downtime = None
    total_stop = planned_stop + unplanned_stop + unclassified_stop
    if total_stop > 0 or planned > 0:
        downtime = {
            "planned_minutes": planned_stop,
            "unplanned_minutes": unplanned_stop,
            "unclassified_minutes": unclassified_stop,
            "total_minutes": total_stop,
            "events": None,
            "mtbf_minutes": None,
            "mttr_minutes": None,
            "pareto": [],
            "timeline": [],
        }

    counts = _aggregate_counts(rows, request)
    quality = _quality_from_counts(counts, request)
    oee = _oee(
        planned_minutes=planned,
        operating_minutes=operating,
        planned_stop_minutes=planned_stop,
        unplanned_stop_minutes=unplanned_stop,
        counts=counts,
        rows=rows,
        request=request,
    )

    return {
        "time_accounting": _time_accounting(
            planned,
            operating,
            planned_stop,
            unplanned_stop,
            unclassified_stop,
            oee,
        ),
        "downtime": downtime,
        "quality": quality,
        "oee": oee,
    }, warnings


def _analyze_events(
    rows: list[dict[str, Any]], request: ManufacturingAnalysisRequest
) -> tuple[dict[str, Any], list[str]]:
    mapping = request.mapping
    events = []
    warnings: list[str] = []
    for source_row, row in enumerate(rows, start=1):
        duration = _event_duration_minutes(row, request)
        state = row.get(mapping.event_state)
        if duration is None or duration < 0 or _is_missing(state):
            warnings.append(f"Fila {source_row}: evento excluido por duración o estado inválido.")
            continue
        category = _state_category(state, request)
        start = _parse_datetime(row.get(mapping.period_start)) if mapping.period_start else None
        end = _parse_datetime(row.get(mapping.period_end)) if mapping.period_end else None
        events.append(
            {
                "row": source_row,
                "state": str(state),
                "category": category,
                "reason": str(row.get(mapping.reason) or "Sin causa"),
                "duration_minutes": duration,
                "start": start,
                "end": end,
                "line": _mapped_text(row, mapping.line),
                "machine": _mapped_text(row, mapping.machine),
                "shift": _mapped_text(row, mapping.shift),
            }
        )
    if not events:
        raise ManufacturingAnalysisError("No hay eventos válidos para analizar")

    timed = [event for event in events if event["start"] and event["end"]]
    overlaps = _count_overlaps(timed)
    if overlaps:
        warnings.append(
            f"Se detectaron {overlaps} solapamientos entre eventos; las sumas "
            "pueden duplicar tiempo."
        )

    operating = sum(event["duration_minutes"] for event in events if event["category"] == "running")
    planned_stop = sum(
        event["duration_minutes"] for event in events if event["category"] == "planned_stop"
    )
    unplanned_events = [event for event in events if event["category"] == "unplanned_stop"]
    unplanned_stop = sum(event["duration_minutes"] for event in unplanned_events)
    planned = operating + planned_stop + unplanned_stop
    if timed:
        span = (
            max(event["end"] for event in timed) - min(event["start"] for event in timed)
        ).total_seconds() / 60
        if span > planned:
            planned = span
    unclassified_stop = max(0.0, planned - operating - planned_stop - unplanned_stop)
    failure_count = len(unplanned_events)
    mttr = unplanned_stop / failure_count if failure_count else None
    mtbf = operating / failure_count if failure_count else None

    reason_totals: dict[str, dict[str, float]] = defaultdict(lambda: {"minutes": 0.0, "events": 0})
    for event in unplanned_events:
        reason_totals[event["reason"]]["minutes"] += event["duration_minutes"]
        reason_totals[event["reason"]]["events"] += 1
    pareto = _pareto(reason_totals)
    timeline = [
        {
            "row": event["row"],
            "start": event["start"].isoformat() if event["start"] else None,
            "end": event["end"].isoformat() if event["end"] else None,
            "duration_minutes": event["duration_minutes"],
            "category": event["category"],
            "reason": event["reason"],
        }
        for event in events[:1000]
    ]

    counts = _aggregate_counts(rows, request)
    quality = _quality_from_counts(counts, request)
    oee = _oee(
        planned,
        operating,
        planned_stop,
        unplanned_stop,
        counts,
        rows,
        request,
    )
    return {
        "time_accounting": _time_accounting(
            planned,
            operating,
            planned_stop,
            unplanned_stop,
            unclassified_stop,
            oee,
        ),
        "downtime": {
            "planned_minutes": planned_stop,
            "unplanned_minutes": unplanned_stop,
            "unclassified_minutes": unclassified_stop,
            "total_minutes": planned_stop + unplanned_stop + unclassified_stop,
            "events": failure_count,
            "mtbf_minutes": mtbf,
            "mttr_minutes": mttr,
            "pareto": pareto,
            "timeline": timeline,
            "overlaps": overlaps,
        },
        "quality": quality,
        "oee": oee,
    }, warnings


def _analyze_cycles(
    rows: list[dict[str, Any]], request: ManufacturingAnalysisRequest
) -> tuple[dict[str, Any], list[str]]:
    cycle = _cycle_from_available_rows(rows, request)
    if cycle is None:
        raise ManufacturingAnalysisError("No hay tiempos de ciclo válidos")
    quality = _quality_from_available_rows(rows, request)
    return {"cycle": cycle, "quality": quality}, []


def _cycle_from_available_rows(
    rows: list[dict[str, Any]], request: ManufacturingAnalysisRequest
) -> dict[str, Any] | None:
    key = request.mapping.cycle_time
    if not key:
        return None
    factor = _time_factor(request.cycle_time_unit)
    values = np.asarray(
        [value * factor for row in rows if (value := _finite_number(row.get(key))) is not None],
        dtype=float,
    )
    values = values[values >= 0]
    if not len(values):
        return None
    standard_values = _numeric_values(rows, request.mapping.standard_cycle_time)
    standard = (
        float(np.mean(standard_values)) * factor
        if standard_values
        else (
            request.targets.standard_cycle_time * factor
            if request.targets.standard_cycle_time is not None
            else None
        )
    )
    planned = _sum_numeric(rows, request.mapping.planned_time) * _time_factor(request.time_unit)
    demand = _sum_numeric(rows, request.mapping.demand) or request.targets.demand
    takt = planned / demand if planned > 0 and demand and demand > 0 else None
    histogram_counts, histogram_edges = np.histogram(
        values, bins=min(20, max(5, round(math.sqrt(len(values)))))
    )
    return {
        "n": len(values),
        "mean_minutes": float(np.mean(values)),
        "median_minutes": float(np.median(values)),
        "p90_minutes": float(np.percentile(values, 90)),
        "p95_minutes": float(np.percentile(values, 95)),
        "std_dev_minutes": float(np.std(values, ddof=1)) if len(values) > 1 else None,
        "coefficient_of_variation": (
            float(np.std(values, ddof=1) / np.mean(values))
            if len(values) > 1 and np.mean(values)
            else None
        ),
        "standard_minutes": standard,
        "above_standard_count": int(np.sum(values > standard)) if standard is not None else None,
        "above_standard_percentage": (
            float(np.mean(values > standard) * 100) if standard is not None else None
        ),
        "takt_minutes": takt,
        "cycle_to_takt_ratio": float(np.mean(values) / takt) if takt else None,
        "throughput_per_hour": 60 / float(np.mean(values)) if np.mean(values) > 0 else None,
        "histogram": {"counts": histogram_counts.tolist(), "edges": histogram_edges.tolist()},
        "series": _sample_series(values),
    }


def _quality_from_available_rows(
    rows: list[dict[str, Any]], request: ManufacturingAnalysisRequest
) -> dict[str, Any] | None:
    counts = _aggregate_counts(rows, request)
    quality = _quality_from_counts(counts, request)
    if quality:
        return quality
    key = request.mapping.quality_result
    if not key:
        return None
    values = [row.get(key) for row in rows if not _is_missing(row.get(key))]
    if not values:
        return None
    good = sum(_matches(value, request.good_values) for value in values)
    counts = {
        "total": float(len(values)),
        "good": float(good),
        "first_pass_good": float(good),
        "scrap": float(len(values) - good),
        "rework": 0.0,
        "defects": float(len(values) - good),
        "opportunities": None,
    }
    return _quality_from_counts(counts, request)


def _aggregate_counts(
    rows: list[dict[str, Any]], request: ManufacturingAnalysisRequest
) -> dict[str, float | None]:
    mapping = request.mapping
    total = _sum_numeric(rows, mapping.total_count)
    good = _sum_numeric(rows, mapping.good_count)
    first_pass = _sum_numeric(rows, mapping.first_pass_good_count)
    scrap = _sum_numeric(rows, mapping.scrap_count)
    rework = _sum_numeric(rows, mapping.rework_count)
    defects = _sum_numeric(rows, mapping.defect_count)
    opportunities = _sum_numeric(rows, mapping.opportunities)
    if total <= 0 and mapping.quality_result:
        values = [
            row.get(mapping.quality_result)
            for row in rows
            if not _is_missing(row.get(mapping.quality_result))
        ]
        total = float(len(values))
        good = float(sum(_matches(value, request.good_values) for value in values))
        scrap = total - good
        defects = scrap
    return {
        "total": total,
        "good": good,
        "first_pass_good": first_pass,
        "scrap": scrap,
        "rework": rework,
        "defects": defects,
        "opportunities": opportunities or None,
    }


def _quality_from_counts(
    counts: dict[str, float | None], request: ManufacturingAnalysisRequest
) -> dict[str, Any] | None:
    total = float(counts["total"] or 0)
    if total <= 0:
        return None
    good = float(counts["good"] or 0)
    first_pass = float(counts["first_pass_good"] or good)
    scrap = float(counts["scrap"] or max(0, total - good))
    rework = float(counts["rework"] or 0)
    defects = float(counts["defects"] or scrap)
    opportunities = counts["opportunities"]
    if opportunities is None and request.targets.opportunities_per_unit:
        opportunities = total * request.targets.opportunities_per_unit
    return {
        "total_count": total,
        "good_count": good,
        "first_pass_good_count": first_pass,
        "scrap_count": scrap,
        "rework_count": rework,
        "defect_count": defects,
        "yield": _ratio(good, total),
        "first_pass_yield": _ratio(first_pass, total),
        "scrap_rate": _ratio(scrap, total),
        "rework_rate": _ratio(rework, total),
        "defects_per_unit": _ratio(defects, total),
        "dpmo": defects / float(opportunities) * 1_000_000 if opportunities else None,
    }


def _oee(
    planned_minutes: float,
    operating_minutes: float,
    planned_stop_minutes: float,
    unplanned_stop_minutes: float,
    counts: dict[str, float | None],
    rows: list[dict[str, Any]],
    request: ManufacturingAnalysisRequest,
) -> dict[str, Any] | None:
    total = float(counts["total"] or 0)
    good = float(counts["good"] or 0)
    ideal_values = _numeric_values(rows, request.mapping.ideal_cycle_time)
    factor = _time_factor(request.cycle_time_unit)
    if ideal_values and total > 0:
        ideal_cycle = float(np.mean(ideal_values)) * factor
    elif request.targets.ideal_cycle_time is not None:
        ideal_cycle = request.targets.ideal_cycle_time * factor
    else:
        ideal_cycle = None
    if planned_minutes <= 0 or operating_minutes <= 0 or total <= 0 or ideal_cycle is None:
        return None
    availability = _bounded_ratio(operating_minutes, planned_minutes)
    performance = _bounded_ratio(ideal_cycle * total, operating_minutes)
    quality = _bounded_ratio(good, total)
    oee = availability * performance * quality
    productive = planned_minutes * oee
    return {
        "availability": availability,
        "performance": performance,
        "quality": quality,
        "oee": oee,
        "planned_minutes": planned_minutes,
        "operating_minutes": operating_minutes,
        "ideal_cycle_minutes": ideal_cycle,
        "productive_equivalent_minutes": productive,
        "losses": {
            "planned_stop_minutes": planned_stop_minutes,
            "unplanned_stop_minutes": unplanned_stop_minutes,
            "speed_loss_minutes": max(0.0, operating_minutes - ideal_cycle * total),
            "quality_loss_minutes": max(0.0, ideal_cycle * (total - good)),
        },
    }


def _capability_from_rows(
    rows: list[dict[str, Any]], request: ManufacturingAnalysisRequest
) -> dict[str, Any] | None:
    key = request.mapping.measurement
    lsl = request.targets.lower_specification
    usl = request.targets.upper_specification
    if not key or lsl is None or usl is None:
        return None
    values = np.asarray(
        [value for row in rows if (value := _finite_number(row.get(key))) is not None],
        dtype=float,
    )
    if len(values) < 2 or np.ptp(values) == 0:
        return None
    mean = float(np.mean(values))
    overall_std = float(np.std(values, ddof=1))
    within_std = _within_subgroup_std(rows, request) or overall_std
    pp = (usl - lsl) / (6 * overall_std)
    ppk = min((usl - mean) / (3 * overall_std), (mean - lsl) / (3 * overall_std))
    cp = (usl - lsl) / (6 * within_std)
    cpk = min((usl - mean) / (3 * within_std), (mean - lsl) / (3 * within_std))
    moving_ranges = np.abs(np.diff(values))
    mr_bar = float(np.mean(moving_ranges)) if len(moving_ranges) else 0
    sigma_individual = mr_bar / 1.128 if mr_bar else overall_std
    return {
        "n": len(values),
        "mean": mean,
        "std_dev_overall": overall_std,
        "std_dev_within": within_std,
        "lsl": lsl,
        "target": request.targets.target,
        "usl": usl,
        "cp": cp,
        "cpk": cpk,
        "pp": pp,
        "ppk": ppk,
        "individual_chart": {
            "center": mean,
            "lower_control_limit": mean - 3 * sigma_individual,
            "upper_control_limit": mean + 3 * sigma_individual,
            "values": _sample_series(values),
        },
        "moving_range_chart": {
            "center": mr_bar,
            "lower_control_limit": 0,
            "upper_control_limit": 3.267 * mr_bar,
            "values": _sample_series(moving_ranges),
        },
    }


def _within_subgroup_std(
    rows: list[dict[str, Any]], request: ManufacturingAnalysisRequest
) -> float | None:
    subgroup_key = request.mapping.subgroup
    measurement_key = request.mapping.measurement
    if not subgroup_key or not measurement_key:
        return None
    groups: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = _finite_number(row.get(measurement_key))
        subgroup = row.get(subgroup_key)
        if value is not None and not _is_missing(subgroup):
            groups[str(subgroup)].append(value)
    numerator = 0.0
    denominator = 0
    for values in groups.values():
        if len(values) > 1:
            numerator += (len(values) - 1) * float(np.var(values, ddof=1))
            denominator += len(values) - 1
    return math.sqrt(numerator / denominator) if denominator else None


def _time_accounting(
    planned: float,
    operating: float,
    planned_stop: float,
    unplanned_stop: float,
    unclassified_stop: float,
    oee: dict[str, Any] | None,
) -> dict[str, Any]:
    losses = (oee or {}).get("losses") or {}
    productive = (oee or {}).get("productive_equivalent_minutes")
    return {
        "planned_minutes": planned,
        "planned_stop_minutes": planned_stop,
        "unplanned_stop_minutes": unplanned_stop,
        "unclassified_stop_minutes": unclassified_stop,
        "operating_minutes": operating,
        "speed_loss_minutes": losses.get("speed_loss_minutes"),
        "quality_loss_minutes": losses.get("quality_loss_minutes"),
        "productive_equivalent_minutes": productive,
    }


def _event_duration_minutes(
    row: dict[str, Any], request: ManufacturingAnalysisRequest
) -> float | None:
    mapping = request.mapping
    if mapping.duration:
        value = _finite_number(row.get(mapping.duration))
        return value * _time_factor(request.time_unit) if value is not None else None
    start = _parse_datetime(row.get(mapping.period_start))
    end = _parse_datetime(row.get(mapping.period_end))
    if start and end:
        return (end - start).total_seconds() / 60
    return None


def _parse_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip().replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _state_category(value: Any, request: ManufacturingAnalysisRequest) -> str:
    if _matches(value, request.running_values):
        return "running"
    if _matches(value, request.planned_stop_values):
        return "planned_stop"
    if _matches(value, request.unplanned_stop_values):
        return "unplanned_stop"
    return "other"


def _count_overlaps(events: list[dict[str, Any]]) -> int:
    ordered = sorted(events, key=lambda event: event["start"])
    overlaps = 0
    previous_end = None
    for event in ordered:
        if previous_end is not None and event["start"] < previous_end:
            overlaps += 1
        previous_end = max(previous_end, event["end"]) if previous_end else event["end"]
    return overlaps


def _pareto(totals: dict[str, dict[str, float]]) -> list[dict[str, Any]]:
    ordered = sorted(totals.items(), key=lambda item: item[1]["minutes"], reverse=True)
    total = sum(values["minutes"] for _, values in ordered)
    cumulative = 0.0
    result = []
    for reason, values in ordered:
        cumulative += values["minutes"]
        result.append(
            {
                "reason": reason,
                "minutes": values["minutes"],
                "events": int(values["events"]),
                "percentage": values["minutes"] / total * 100 if total else 0,
                "cumulative_percentage": cumulative / total * 100 if total else 0,
            }
        )
    return result


def _filtered_rows(
    rows: list[dict[str, Any]], filters: dict[str, list[Any]]
) -> list[dict[str, Any]]:
    if not filters:
        return rows
    normalized = {key: {str(value) for value in values} for key, values in filters.items()}
    return [
        row
        for row in rows
        if all(str(row.get(key)) in values for key, values in normalized.items())
    ]


def _sum_numeric(rows: list[dict[str, Any]], key: str | None) -> float:
    if not key:
        return 0.0
    return sum(value for row in rows if (value := _finite_number(row.get(key))) is not None)


def _numeric_values(rows: list[dict[str, Any]], key: str | None) -> list[float]:
    if not key:
        return []
    return [value for row in rows if (value := _finite_number(row.get(key))) is not None]


def _finite_number(value: Any) -> float | None:
    if value is None or value == "" or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _time_factor(unit: str) -> float:
    return {"seconds": 1 / 60, "minutes": 1, "hours": 60}[unit]


def _ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator > 0 else None


def _bounded_ratio(numerator: float, denominator: float) -> float:
    return min(1.0, max(0.0, numerator / denominator)) if denominator > 0 else 0.0


def _matches(value: Any, candidates: list[Any]) -> bool:
    normalized = str(value).strip().casefold()
    return any(normalized == str(candidate).strip().casefold() for candidate in candidates)


def _is_missing(value: Any) -> bool:
    return value is None or value == ""


def _mapped_text(row: dict[str, Any], key: str | None) -> str | None:
    if not key or _is_missing(row.get(key)):
        return None
    return str(row.get(key))


def _sample_series(values: np.ndarray, maximum: int = 500) -> list[float]:
    if len(values) <= maximum:
        return values.astype(float).tolist()
    indices = np.linspace(0, len(values) - 1, maximum, dtype=int)
    return values[indices].astype(float).tolist()
