from io import BytesIO
from pathlib import Path

import pandas as pd
from flask import Blueprint, current_app, jsonify, request

from utils.errors import APIError
from utils.serialization import dataframe_records, json_safe
from utils.validation import get_json_object, require_fields

data_bp = Blueprint("data", __name__, url_prefix="/api/data")
ALLOWED_EXTENSIONS = {".csv", ".tsv", ".xls", ".xlsx", ".json"}


def _store():
    return current_app.extensions["dataset_store"]


def _read_upload(filename: str, content: bytes) -> pd.DataFrame:
    extension = Path(filename).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise APIError("Formato no soportado. Use CSV, TSV, XLS, XLSX o JSON.", 415, "unsupported_format")
    try:
        stream = BytesIO(content)
        if extension == ".csv":
            return pd.read_csv(stream)
        if extension == ".tsv":
            return pd.read_csv(stream, sep="\t")
        if extension in {".xls", ".xlsx"}:
            return pd.read_excel(stream)
        return pd.read_json(stream)
    except Exception as exc:
        raise APIError("No se pudo interpretar el archivo.", 400, "invalid_file", str(exc)) from exc


@data_bp.post("/upload")
def upload():
    file = request.files.get("file")
    if file is None or not file.filename:
        raise APIError("Se requiere un archivo en el campo 'file'.", details={"field": "file"})
    frame = _read_upload(file.filename, file.read())
    dataset = _store().create(file.filename, frame)
    return jsonify(dataset), 201


@data_bp.post("/manual")
def manual():
    payload = get_json_object()
    require_fields(payload, "name", "columns", "rows")
    name, columns, rows = payload["name"], payload["columns"], payload["rows"]
    if not isinstance(name, str) or not name.strip():
        raise APIError("'name' debe ser un texto no vacío.", details={"field": "name"})
    if not isinstance(columns, list) or not columns or not all(isinstance(item, str) and item for item in columns):
        raise APIError("'columns' debe ser una lista no vacía de textos.", details={"field": "columns"})
    if len(set(columns)) != len(columns):
        raise APIError("Los nombres de columnas deben ser únicos.", details={"field": "columns"})
    if not isinstance(rows, list) or not all(isinstance(row, list) and len(row) == len(columns) for row in rows):
        raise APIError("Cada fila debe ser una lista con el mismo tamaño que 'columns'.", details={"field": "rows"})
    frame = pd.DataFrame(rows, columns=columns)
    dataset = _store().create(name.strip(), frame)
    return jsonify(dataset), 201


@data_bp.get("/<dataset_id>")
def get_dataset(dataset_id: str):
    dataset = _store().get(dataset_id)
    page = request.args.get("page", 1, type=int)
    page_size = request.args.get("page_size", 50, type=int)
    if page is None or page < 1 or page_size is None or not 1 <= page_size <= 500:
        raise APIError("'page' debe ser >= 1 y 'page_size' debe estar entre 1 y 500.", details={"fields": ["page", "page_size"]})
    start = (page - 1) * page_size
    frame = dataset.frame.iloc[start : start + page_size]
    return jsonify(json_safe({
        **dataset.metadata(),
        "data": dataframe_records(frame),
        "pagination": {"page": page, "page_size": page_size, "total": len(dataset.frame)},
    }))


@data_bp.delete("/<dataset_id>")
def delete_dataset(dataset_id: str):
    _store().delete(dataset_id)
    return "", 204
