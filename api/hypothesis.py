import pandas as pd
from flask import Blueprint, current_app, jsonify

from modules.hypothesis import (
    anova_test,
    chi_square_test,
    correlation_test,
    decision,
    independent_t,
    one_sample_t,
    paired_t,
)
from utils.errors import APIError
from utils.serialization import json_safe
from utils.validation import alpha_value, get_json_object, numeric_series, require_fields, validate_columns

hypothesis_bp = Blueprint("hypothesis", __name__, url_prefix="/api/hypothesis")


def _dataset(payload: dict):
    require_fields(payload, "dataset_id")
    return current_app.extensions["dataset_store"].get(payload["dataset_id"])


def _paired_values(dataset, first_column: str, second_column: str, minimum: int = 2):
    validate_columns(dataset, [first_column, second_column])
    pairs = dataset.frame[[first_column, second_column]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(pairs) < minimum:
        raise APIError(f"Se requieren al menos {minimum} pares numéricos completos.")
    return pairs[first_column].astype(float), pairs[second_column].astype(float)


@hypothesis_bp.post("/t-test")
def t_test():
    payload = get_json_object()
    require_fields(payload, "dataset_id", "type")
    dataset = _dataset(payload)
    alpha = alpha_value(payload)
    test_type = payload["type"]

    if test_type == "one-sample":
        require_fields(payload, "column", "popmean")
        if isinstance(payload["popmean"], bool) or not isinstance(payload["popmean"], (int, float)):
            raise APIError("'popmean' debe ser numérico.", details={"field": "popmean"})
        values = numeric_series(dataset, payload["column"], 2)
        if values.nunique() < 2:
            raise APIError("La prueba t requiere variación en la muestra.")
        results = one_sample_t(values, float(payload["popmean"]))
        context = "de que la media difiere del valor de referencia"
    elif test_type == "independent":
        require_fields(payload, "value_column", "group_column")
        validate_columns(dataset, [payload["value_column"], payload["group_column"]])
        available = dataset.frame[payload["group_column"]].dropna().unique().tolist()
        selected = payload.get("groups", available[:2])
        if not isinstance(selected, list) or len(selected) != 2 or selected[0] == selected[1]:
            raise APIError("'groups' debe contener exactamente dos grupos distintos.", details={"field": "groups"})
        samples = []
        for group in selected:
            values = pd.to_numeric(
                dataset.frame.loc[dataset.frame[payload["group_column"]] == group, payload["value_column"]],
                errors="coerce",
            ).dropna().astype(float)
            if len(values) < 2:
                raise APIError("Cada grupo requiere al menos dos valores numéricos.", details={"group": json_safe(group)})
            if values.nunique() < 2:
                raise APIError("Cada grupo requiere variación en sus valores.", details={"group": json_safe(group)})
            samples.append(values)
        results = independent_t(*samples)
        results["groups"] = json_safe(selected)
        context = "de una diferencia entre las medias de los grupos"
    elif test_type == "paired":
        require_fields(payload, "column1", "column2")
        first, second = _paired_values(dataset, payload["column1"], payload["column2"])
        if (first - second).nunique() < 2:
            raise APIError("La prueba t pareada requiere variación en las diferencias.")
        results = paired_t(first, second)
        context = "de una diferencia media entre las mediciones pareadas"
    else:
        raise APIError("'type' debe ser one-sample, independent o paired.", details={"field": "type"})

    return jsonify(json_safe({
        "dataset_id": dataset.id,
        "test": "t-test",
        "type": test_type,
        "alpha": alpha,
        "results": results,
        "interpretation": decision(results["p_value"], alpha, context),
    }))


@hypothesis_bp.post("/anova")
def anova():
    payload = get_json_object()
    require_fields(payload, "dataset_id", "value_column", "group_column")
    dataset = _dataset(payload)
    alpha = alpha_value(payload)
    validate_columns(dataset, [payload["value_column"], payload["group_column"]])
    groups = []
    for name, frame in dataset.frame.groupby(payload["group_column"], dropna=False):
        values = pd.to_numeric(frame[payload["value_column"]], errors="coerce").dropna().astype(float)
        if len(values) >= 2:
            groups.append((json_safe(name), values))
    if len(groups) < 2:
        raise APIError("ANOVA requiere al menos dos grupos con dos valores numéricos cada uno.")
    if pd.concat([values for _, values in groups]).nunique() < 2:
        raise APIError("ANOVA requiere variación en los valores.")
    results = anova_test(groups)
    return jsonify(json_safe({
        "dataset_id": dataset.id,
        "test": "anova_one_way",
        "alpha": alpha,
        "results": results,
        "interpretation": decision(results["p_value"], alpha, "de diferencias entre al menos dos medias de grupo"),
    }))


@hypothesis_bp.post("/chi-square")
def chi_square():
    payload = get_json_object()
    require_fields(payload, "dataset_id", "row_column", "column_column")
    dataset = _dataset(payload)
    alpha = alpha_value(payload)
    validate_columns(dataset, [payload["row_column"], payload["column_column"]])
    complete = dataset.frame[[payload["row_column"], payload["column_column"]]].dropna()
    table = pd.crosstab(complete[payload["row_column"]], complete[payload["column_column"]])
    if table.shape[0] < 2 or table.shape[1] < 2:
        raise APIError("Chi-cuadrado requiere una tabla de contingencia de al menos 2x2.")
    results = chi_square_test(table)
    return jsonify(json_safe({
        "dataset_id": dataset.id,
        "test": "chi_square_independence",
        "alpha": alpha,
        "results": results,
        "interpretation": decision(results["p_value"], alpha, "de asociación entre las variables categóricas"),
    }))


@hypothesis_bp.post("/correlation")
def correlation():
    payload = get_json_object()
    require_fields(payload, "dataset_id", "x", "y")
    dataset = _dataset(payload)
    alpha = alpha_value(payload)
    x, y = _paired_values(dataset, payload["x"], payload["y"], 3)
    if x.nunique() < 2 or y.nunique() < 2:
        raise APIError("La correlación requiere variación en ambas columnas.")
    results = correlation_test(x, y)
    strength = abs(results["coefficient"])
    qualifier = "fuerte" if strength >= 0.7 else "moderada" if strength >= 0.4 else "débil"
    significance = decision(results["p_value"], alpha, "de correlación lineal")
    return jsonify(json_safe({
        "dataset_id": dataset.id,
        "test": "pearson_correlation",
        "alpha": alpha,
        "results": results,
        "interpretation": f"La correlación lineal es {qualifier} (r={results['coefficient']:.3f}). {significance}",
    }))
