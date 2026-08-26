from flask import Blueprint, current_app, jsonify

import numpy as np
import pandas as pd
from scipy import stats

from modules.charts import (boxplot_data, grouped_chart, histogram_data, numeric_points,
                            sampled_frame, scatter_data)
from utils.errors import APIError
from utils.serialization import json_safe
from utils.validation import (alpha_value, choice, column_list, get_json_object, number_value,
                              numeric_series, require_fields, validate_columns)

charts_bp = Blueprint("charts", __name__, url_prefix="/api/charts")


def _dataset(payload: dict):
    require_fields(payload, "dataset_id")
    return current_app.extensions["dataset_store"].get(payload["dataset_id"])


@charts_bp.post("/histogram")
def histogram():
    payload = get_json_object()
    require_fields(payload, "dataset_id", "column")
    dataset = _dataset(payload)
    bins = payload.get("bins", 10)
    if isinstance(bins, bool) or not isinstance(bins, int) or not 1 <= bins <= 200:
        raise APIError("'bins' debe ser un entero entre 1 y 200.", details={"field": "bins"})
    values = numeric_series(dataset, payload["column"])
    density = payload.get("density", False); kde = payload.get("kde", False)
    if not isinstance(density, bool) or not isinstance(kde, bool):
        raise APIError("'density' y 'kde' deben ser booleanos.")
    return jsonify(json_safe({
        "dataset_id": dataset.id,
        "column": payload["column"],
        "results": histogram_data(values, bins, density, kde),
        "interpretation": "Las cuentas corresponden a los intervalos definidos por bin_edges.",
    }))


@charts_bp.post("/boxplot")
def boxplot():
    payload = get_json_object()
    require_fields(payload, "dataset_id", "column")
    dataset = _dataset(payload)
    group_by = payload.get("group_by")
    validate_columns(dataset, [payload["column"]] + ([group_by] if group_by is not None else []))
    numeric_series(dataset, payload["column"])
    results = boxplot_data(dataset, payload["column"], group_by)
    if not results["groups"]:
        raise APIError("No hay grupos con valores numéricos para generar el boxplot.")
    return jsonify(json_safe({
        "dataset_id": dataset.id,
        "column": payload["column"],
        "group_by": group_by,
        "results": results,
        "interpretation": "Los atípicos se identifican con el criterio de 1.5 veces el rango intercuartílico.",
    }))


@charts_bp.post("/scatter")
def scatter():
    payload = get_json_object()
    require_fields(payload, "dataset_id", "x", "y")
    dataset = _dataset(payload)
    numeric_series(dataset, payload["x"])
    numeric_series(dataset, payload["y"])
    results = scatter_data(dataset, payload["x"], payload["y"])
    if results["count"] == 0:
        raise APIError("No existen pares numéricos completos para el gráfico.")
    return jsonify({
        "dataset_id": dataset.id,
        "x": payload["x"],
        "y": payload["y"],
        "results": results,
        "interpretation": "Cada punto representa una observación con valores válidos en ambos ejes.",
    })


def _chart_response(dataset, chart, results, **metadata):
    return jsonify(json_safe({"dataset_id": dataset.id, "chart": chart, **metadata, "results": results,
                              "limits": {"max_points": 5000, "max_categories": 100}}))


def _grouped(chart, default_aggregation="sum"):
    payload = get_json_object(); require_fields(payload, "category"); dataset = _dataset(payload)
    value = payload.get("value"); validate_columns(dataset, [payload["category"]] + ([value] if value else []))
    aggregation = choice(payload, "aggregation", {"sum", "mean", "count", "median"}, default_aggregation)
    if value is None: aggregation = "count"
    results = grouped_chart(dataset.frame, payload["category"], value, aggregation)
    if chart == "pareto":
        total = sum(item["value"] for item in results); running = 0
        for item in results:
            running += item["value"]; item["cumulative_percentage"] = running / total * 100 if total else 0
    return _chart_response(dataset, chart, {"series": results, "aggregation": aggregation})


@charts_bp.post("/bar")
def bar(): return _grouped("bar", "sum")


@charts_bp.post("/pie")
def pie(): return _grouped("pie", "sum")


@charts_bp.post("/pareto")
def pareto(): return _grouped("pareto", "count")


@charts_bp.post("/qq")
def qq():
    payload = get_json_object(); require_fields(payload, "column"); dataset = _dataset(payload)
    values = numeric_series(dataset, payload["column"], 2).sort_values()
    if len(values) > 5000: values = values.iloc[np.linspace(0, len(values) - 1, 5000, dtype=int)]
    theoretical, ordered = stats.probplot(values, dist="norm", fit=False)
    fit = stats.linregress(theoretical, ordered)
    return _chart_response(dataset, "qq", {"points": [{"theoretical": x, "observed": y} for x, y in zip(theoretical, ordered)],
                                            "line": {"slope": fit.slope, "intercept": fit.intercept}, "count": len(values)})


@charts_bp.post("/violin")
def violin():
    payload = get_json_object(); require_fields(payload, "column"); dataset = _dataset(payload); group = payload.get("group_by")
    validate_columns(dataset, [payload["column"]] + ([group] if group else [])); groups = []
    iterator = dataset.frame.groupby(group, dropna=False) if group else [(None, dataset.frame)]
    for name, frame in iterator:
        values = pd.to_numeric(frame[payload["column"]], errors="coerce").dropna()
        if len(values) > 1000: values = values.iloc[np.linspace(0, len(values) - 1, 1000, dtype=int)]
        if len(values): groups.append({"group": name, "values": values.tolist(), "count": len(values)})
    return _chart_response(dataset, "violin", {"groups": groups})


@charts_bp.post("/dot")
def dot():
    payload = get_json_object(); require_fields(payload, "column"); dataset = _dataset(payload); values = numeric_series(dataset, payload["column"])
    counts = values.value_counts().sort_index().iloc[:1000]
    return _chart_response(dataset, "dot", {"points": [{"value": x, "count": n} for x, n in counts.items()]})


def _xy_chart(chart):
    payload = get_json_object(); require_fields(payload, "x", "y"); dataset = _dataset(payload); validate_columns(dataset, [payload["x"], payload["y"]])
    data = dataset.frame[[payload["x"], payload["y"]]].copy(); data[payload["y"]] = pd.to_numeric(data[payload["y"]], errors="coerce"); data = data.dropna()
    data, sampled = sampled_frame(data, [payload["x"], payload["y"]], 5000); data = data.sort_values(payload["x"])
    points = [{"x": x, "y": y} for x, y in data.itertuples(index=False, name=None)]
    return _chart_response(dataset, chart, {"points": points, "count": len(points), "sampled": sampled})


@charts_bp.post("/line")
def line(): return _xy_chart("line")


@charts_bp.post("/area")
def area(): return _xy_chart("area")


@charts_bp.post("/interval")
def interval():
    payload = get_json_object(); require_fields(payload, "value", "group"); dataset = _dataset(payload); validate_columns(dataset, [payload["value"], payload["group"]])
    confidence = number_value(payload, "confidence", default=.95, minimum=.5, maximum=.999); rows = []
    for name, frame in dataset.frame.groupby(payload["group"], dropna=False):
        values = pd.to_numeric(frame[payload["value"]], errors="coerce").dropna()
        if len(values) < 2: continue
        mean, margin = values.mean(), stats.t.ppf((1 + confidence) / 2, len(values)-1) * stats.sem(values)
        rows.append({"group": name, "mean": mean, "lower": mean-margin, "upper": mean+margin, "count": len(values)})
    if not rows: raise APIError("No hay grupos con al menos dos valores.")
    return _chart_response(dataset, "interval", {"intervals": rows, "confidence": confidence})


@charts_bp.post("/pairplot")
def pairplot():
    payload = get_json_object(); dataset = _dataset(payload); columns = column_list(dataset, payload, minimum=2, maximum=10)
    data = dataset.frame[columns].apply(pd.to_numeric, errors="coerce").dropna(); data, sampled = sampled_frame(data, columns, 1000)
    return _chart_response(dataset, "pairplot", {"columns": columns, "rows": data.to_dict("records"), "sampled": sampled,
                                                  "correlation": data.corr().to_numpy()}, max_points=1000)


@charts_bp.post("/bubble")
def bubble():
    payload = get_json_object(); require_fields(payload, "x", "y", "size"); dataset = _dataset(payload)
    columns = [payload["x"], payload["y"], payload["size"]] + ([payload["color"]] if payload.get("color") else [])
    validate_columns(dataset, columns)
    data = dataset.frame[columns].copy(); data[columns[:3]] = data[columns[:3]].apply(pd.to_numeric, errors="coerce"); data = data.dropna(subset=columns[:3]); data, sampled = sampled_frame(data, columns, 5000)
    return _chart_response(dataset, "bubble", {"points": data.to_dict("records"), "sampled": sampled, "count": len(data)})


@charts_bp.post("/heatmap")
def heatmap():
    payload = get_json_object(); dataset = _dataset(payload); columns = column_list(dataset, payload, minimum=2)
    method = choice(payload, "method", {"pearson", "spearman", "kendall"}, "pearson"); data = dataset.frame[columns].apply(pd.to_numeric, errors="coerce")
    return _chart_response(dataset, "heatmap", {"columns": columns, "matrix": data.corr(method=method).to_numpy(), "method": method})


@charts_bp.post("/scatter-3d")
def scatter_3d():
    payload = get_json_object(); require_fields(payload, "x", "y", "z"); dataset = _dataset(payload); columns = [payload["x"], payload["y"], payload["z"]]
    validate_columns(dataset, columns)
    return _chart_response(dataset, "scatter-3d", numeric_points(dataset.frame, columns))
