import pandas as pd
from flask import Blueprint, current_app, jsonify

from modules.anova import one_way, post_hoc, repeated, two_way
from utils.errors import APIError
from utils.serialization import json_safe
from utils.validation import alpha_value, choice, get_json_object, number_value, require_fields, validate_columns

anova_bp = Blueprint("anova", __name__, url_prefix="/api/anova")


def _dataset(payload):
    require_fields(payload, "dataset_id")
    return current_app.extensions["dataset_store"].get(payload["dataset_id"])


def _long_data(dataset, value, columns, internal):
    validate_columns(dataset, [value, *columns])
    data = dataset.frame[[value, *columns]].copy(); data[value] = pd.to_numeric(data[value], errors="coerce")
    data = data.dropna().rename(columns={value: "value", **dict(zip(columns, internal))})
    if data.empty or data["value"].nunique() < 2: raise APIError("ANOVA requiere valores numéricos completos con variación.")
    return data


def _result(dataset, test, results, alpha):
    p_values = [row.get("p_value") for row in results.get("table", []) if row.get("p_value") is not None]
    if test == "anova_post_hoc":
        significant = sum(bool(row.get("reject")) for row in results.get("comparisons", []))
        interpretation = f"Se identificaron {significant} comparaciones significativas con el método {results.get('method')}."
    elif p_values:
        interpretation = ("Existe al menos un efecto estadísticamente significativo."
                          if any(value < alpha for value in p_values)
                          else "No se detectaron efectos estadísticamente significativos.")
    else:
        interpretation = "El análisis se completó; revise la tabla de resultados y sus supuestos."
    return jsonify(json_safe({"dataset_id": dataset.id, "test": test, "alpha": alpha,
                              "results": results, "interpretation": interpretation}))


@anova_bp.post("/one-way")
def one_way_route():
    payload = get_json_object(); require_fields(payload, "value_column", "group_column"); dataset = _dataset(payload); alpha = alpha_value(payload)
    data = _long_data(dataset, payload["value_column"], [payload["group_column"]], ["group"])
    if data["group"].nunique() < 2 or (data.groupby("group").size() < 2).any(): raise APIError("Se requieren al menos dos grupos con dos observaciones.")
    return _result(dataset, "anova_one_way", one_way(data, alpha), alpha)


def _two_way_response(test="anova_two_way"):
    payload = get_json_object(); require_fields(payload, "value_column", "factor_a", "factor_b"); dataset = _dataset(payload); alpha = alpha_value(payload)
    interaction = payload.get("interaction", True)
    if not isinstance(interaction, bool): raise APIError("'interaction' debe ser booleano.")
    typ = number_value(payload, "typ", default=2, minimum=1, maximum=3, integer=True)
    data = _long_data(dataset, payload["value_column"], [payload["factor_a"], payload["factor_b"]], ["factor_a", "factor_b"])
    if data["factor_a"].nunique() < 2 or data["factor_b"].nunique() < 2: raise APIError("Cada factor requiere al menos dos niveles.")
    results = two_way(data, interaction, typ, alpha)
    if test == "anova_main_effects": results["main_effects"] = [row for row in results["table"] if row["term"] in {"C(factor_a)", "C(factor_b)"}]
    if test == "anova_interaction": results["interaction_effect"] = next((row for row in results["table"] if ":" in row["term"]), None)
    return _result(dataset, test, results, alpha)


@anova_bp.post("/two-way")
def two_way_route(): return _two_way_response()


@anova_bp.post("/main-effects")
def main_effects(): return _two_way_response("anova_main_effects")


@anova_bp.post("/interaction")
def interaction(): return _two_way_response("anova_interaction")


@anova_bp.post("/post-hoc")
def post_hoc_route():
    payload = get_json_object(); require_fields(payload, "value_column", "group_column"); dataset = _dataset(payload); alpha = alpha_value(payload)
    method = choice(payload, "method", {"tukey", "bonferroni", "scheffe"}, "tukey")
    data = _long_data(dataset, payload["value_column"], [payload["group_column"]], ["group"])
    if data["group"].nunique() < 2: raise APIError("Post-hoc requiere al menos dos grupos.")
    return _result(dataset, "anova_post_hoc", post_hoc(data, method, alpha), alpha)


@anova_bp.post("/repeated-measures")
def repeated_route():
    payload = get_json_object(); require_fields(payload, "subject", "within", "value"); dataset = _dataset(payload)
    data = _long_data(dataset, payload["value"], [payload["subject"], payload["within"]], ["subject", "within"]); alpha = alpha_value(payload)
    if data.duplicated(["subject", "within"]).any(): raise APIError("Debe existir una sola observación por sujeto y nivel within.")
    counts = data.groupby("subject")["within"].nunique()
    if data["subject"].nunique() < 2 or data["within"].nunique() < 2 or counts.nunique() != 1 or counts.iloc[0] != data["within"].nunique(): raise APIError("AnovaRM requiere al menos dos sujetos y niveles en un diseño completo y balanceado.")
    return _result(dataset, "anova_repeated_measures", repeated(data), alpha)
