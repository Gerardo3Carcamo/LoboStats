from flask import Blueprint, current_app, jsonify

from modules.descriptive import frequencies, summarize
from utils.serialization import json_safe
from utils.validation import alpha_value, get_json_object, require_fields, validate_columns

descriptive_bp = Blueprint("descriptive", __name__, url_prefix="/api/descriptive")


def _dataset(payload: dict):
    require_fields(payload, "dataset_id")
    return current_app.extensions["dataset_store"].get(payload["dataset_id"])


@descriptive_bp.post("/summary")
def summary():
    payload = get_json_object()
    dataset = _dataset(payload)
    columns = payload.get("columns", list(dataset.frame.columns))
    if not isinstance(columns, list) or not columns:
        from utils.errors import APIError
        raise APIError("'columns' debe ser una lista no vacía.", details={"field": "columns"})
    validate_columns(dataset, columns)
    alpha = alpha_value(payload)
    return jsonify(json_safe({
        "dataset_id": dataset.id,
        "alpha": alpha,
        "results": summarize(dataset, columns, alpha),
        "interpretation": "Resumen descriptivo calculado; el CV se expresa como porcentaje.",
    }))


@descriptive_bp.post("/frequencies")
def frequency_table():
    payload = get_json_object()
    require_fields(payload, "dataset_id", "column")
    dataset = _dataset(payload)
    validate_columns(dataset, [payload["column"]])
    return jsonify(json_safe({
        "dataset_id": dataset.id,
        "column": payload["column"],
        "results": frequencies(dataset, payload["column"]),
        "interpretation": "Frecuencias absolutas y porcentuales de cada valor, incluidos los ausentes.",
    }))
