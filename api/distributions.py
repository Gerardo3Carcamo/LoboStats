import math
from typing import Any

import numpy as np
from flask import Blueprint, current_app, jsonify
from scipy import stats

from modules.distributions import (DEFAULT_CANDIDATES, FIT_SPECS, fit_distribution,
                                   identify_distributions, probability_plot)
from utils.errors import APIError
from utils.serialization import json_safe
from utils.validation import get_json_object, numeric_series, require_fields


distributions_bp = Blueprint("distributions", __name__, url_prefix="/api/distributions")
MAX_INPUT_VALUES = 10000
MAX_RANDOM_SIZE = 10000
MAX_CURVE_POINTS = 2000


DIRECT_SPECS = {
    "normal": (stats.norm, {"mean": (0.0, None), "std": (1.0, "positive")}, ("mean", "std"), False),
    "t": (stats.t, {"df": (None, "positive"), "loc": (0.0, None), "scale": (1.0, "positive")}, ("df", "loc", "scale"), False),
    "chi2": (stats.chi2, {"df": (None, "positive"), "loc": (0.0, None), "scale": (1.0, "positive")}, ("df", "loc", "scale"), False),
    "f": (stats.f, {"dfn": (None, "positive"), "dfd": (None, "positive"), "loc": (0.0, None), "scale": (1.0, "positive")}, ("dfn", "dfd", "loc", "scale"), False),
    "binomial": (stats.binom, {"n": (None, "nonnegative_int"), "p": (None, "probability"), "loc": (0, "integer")}, ("n", "p", "loc"), True),
    "poisson": (stats.poisson, {"mu": (None, "nonnegative"), "loc": (0, "integer")}, ("mu", "loc"), True),
    "hypergeom": (stats.hypergeom, {"M": (None, "positive_int"), "n": (None, "nonnegative_int"), "N": (None, "nonnegative_int"), "loc": (0, "integer")}, ("M", "n", "N", "loc"), True),
    "exponential": (stats.expon, {"loc": (0.0, None), "scale": (1.0, "positive")}, ("loc", "scale"), False),
    "gamma": (stats.gamma, {"shape": (None, "positive"), "loc": (0.0, None), "scale": (1.0, "positive")}, ("shape", "loc", "scale"), False),
    "beta": (stats.beta, {"a": (None, "positive"), "b": (None, "positive"), "loc": (0.0, None), "scale": (1.0, "positive")}, ("a", "b", "loc", "scale"), False),
    "weibull": (stats.weibull_min, {"shape": (None, "positive"), "loc": (0.0, None), "scale": (1.0, "positive")}, ("shape", "loc", "scale"), False),
    "lognormal": (stats.lognorm, {"shape": (None, "positive"), "loc": (0.0, None), "scale": (1.0, "positive")}, ("shape", "loc", "scale"), False),
    "uniform": (stats.uniform, {"loc": (0.0, None), "scale": (1.0, "positive")}, ("loc", "scale"), False),
    "logistic": (stats.logistic, {"loc": (0.0, None), "scale": (1.0, "positive")}, ("loc", "scale"), False),
    "gumbel_l": (stats.gumbel_l, {"loc": (0.0, None), "scale": (1.0, "positive")}, ("loc", "scale"), False),
    "gumbel_r": (stats.gumbel_r, {"loc": (0.0, None), "scale": (1.0, "positive")}, ("loc", "scale"), False),
}


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise APIError(f"'{field}' debe ser un número finito.", details={"field": field})
    return float(value)


def _parameters(payload: dict):
    require_fields(payload, "distribution", "params")
    name = payload["distribution"]
    if not isinstance(name, str) or name not in DIRECT_SPECS:
        raise APIError("Distribución no soportada.", details={"field": "distribution", "allowed": sorted(DIRECT_SPECS)})
    supplied = payload["params"]
    if not isinstance(supplied, dict):
        raise APIError("'params' debe ser un objeto JSON.", details={"field": "params"})
    scipy_dist, definitions, order, discrete = DIRECT_SPECS[name]
    unknown = sorted(set(supplied) - set(definitions))
    if unknown:
        raise APIError("'params' contiene parámetros desconocidos.", details={"fields": unknown})
    parameters = {}
    for key, (default, rule) in definitions.items():
        if key not in supplied and default is None:
            raise APIError("Faltan parámetros de distribución.", details={"fields": [key]})
        raw = supplied.get(key, default)
        if rule in {"integer", "positive_int", "nonnegative_int"}:
            if isinstance(raw, bool) or not isinstance(raw, int):
                raise APIError(f"'{key}' debe ser entero.", details={"field": key})
            value = int(raw)
        else:
            value = _finite_number(raw, key)
        if rule in {"positive", "positive_int"} and value <= 0:
            raise APIError(f"'{key}' debe ser mayor que cero.", details={"field": key})
        if rule in {"nonnegative", "nonnegative_int"} and value < 0:
            raise APIError(f"'{key}' no puede ser negativo.", details={"field": key})
        if rule == "probability" and not 0 <= value <= 1:
            raise APIError(f"'{key}' debe estar entre 0 y 1.", details={"field": key})
        parameters[key] = value
    if name == "hypergeom" and (parameters["n"] > parameters["M"] or parameters["N"] > parameters["M"]):
        raise APIError("En hypergeom, 'n' y 'N' no pueden superar 'M'.")
    args = tuple(parameters[key] for key in order)
    return name, scipy_dist, parameters, args, discrete


def _numeric_input(payload: dict, field: str, *, probability: bool = False):
    require_fields(payload, field)
    raw = payload[field]
    scalar = not isinstance(raw, list)
    items = [raw] if scalar else raw
    if not items or len(items) > MAX_INPUT_VALUES:
        raise APIError(f"'{field}' debe contener entre 1 y {MAX_INPUT_VALUES} valores.", details={"field": field})
    values = [_finite_number(item, field) for item in items]
    if probability and any(not 0 < value < 1 for value in values):
        raise APIError(f"'{field}' debe estar estrictamente entre 0 y 1.", details={"field": field})
    return np.asarray(values, dtype=float), scalar


def _response(name: str, operation: str, parameters: dict, results: dict, interpretation: str):
    return jsonify(json_safe({
        "distribution": name,
        "operation": operation,
        "parameters": parameters,
        "results": results,
        "interpretation": interpretation,
    }))


def _evaluate(operation: str):
    payload = get_json_object()
    name, scipy_dist, parameters, args, discrete = _parameters(payload)
    field = "q" if operation == "ppf" else "x"
    values, scalar = _numeric_input(payload, field, probability=operation == "ppf")
    if operation == "pdf" and discrete and np.any(values != np.floor(values)):
        raise APIError("'x' debe contener enteros para una PMF discreta.", details={"field": "x"})
    method = "pmf" if operation == "pdf" and discrete else operation
    calculated = np.asarray(getattr(scipy_dist, method)(values, *args), dtype=float)
    output = float(calculated[0]) if scalar else calculated.tolist()
    key = "value" if scalar else "values"
    results = {
        key: output,
        "metadata": {
            "function": method.upper(),
            "requested_operation": operation,
            "discrete": discrete,
            "count": int(len(values)),
        },
    }
    interpretation = (
        "Para esta distribución discreta, /pdf evalúa la función de masa de probabilidad (PMF)."
        if operation == "pdf" and discrete
        else f"Se evaluó {method.upper()} con los parámetros indicados."
    )
    return _response(name, operation, parameters, results, interpretation)


@distributions_bp.post("/pdf")
def pdf():
    return _evaluate("pdf")


@distributions_bp.post("/cdf")
def cdf():
    return _evaluate("cdf")


@distributions_bp.post("/sf")
def sf():
    return _evaluate("sf")


@distributions_bp.post("/ppf")
def ppf():
    return _evaluate("ppf")


@distributions_bp.post("/interval")
def interval():
    payload = get_json_object()
    name, scipy_dist, parameters, args, discrete = _parameters(payload)
    require_fields(payload, "confidence")
    confidence = _finite_number(payload["confidence"], "confidence")
    if not 0 < confidence < 1:
        raise APIError("'confidence' debe estar estrictamente entre 0 y 1.", details={"field": "confidence"})
    lower, upper = scipy_dist.interval(confidence, *args)
    results = {
        "values": [lower, upper],
        "lower": lower,
        "upper": upper,
        "metadata": {"confidence": confidence, "discrete": discrete},
    }
    return _response(name, "interval", parameters, results, "Intervalo central de probabilidad para la distribución.")


@distributions_bp.post("/random")
def random_values():
    payload = get_json_object()
    name, scipy_dist, parameters, args, discrete = _parameters(payload)
    require_fields(payload, "size", "seed")
    size, seed = payload["size"], payload["seed"]
    if isinstance(size, bool) or not isinstance(size, int) or not 1 <= size <= MAX_RANDOM_SIZE:
        raise APIError(f"'size' debe ser un entero entre 1 y {MAX_RANDOM_SIZE}.", details={"field": "size"})
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**32 - 1:
        raise APIError("'seed' debe ser un entero entre 0 y 2^32-1.", details={"field": "seed"})
    values = scipy_dist.rvs(*args, size=size, random_state=np.random.default_rng(seed))
    results = {"values": np.asarray(values).tolist(), "metadata": {"size": size, "seed": seed, "discrete": discrete}}
    return _response(name, "random", parameters, results, "Muestra pseudoaleatoria reproducible con la semilla indicada.")


@distributions_bp.post("/curve")
def curve():
    payload = get_json_object()
    name, scipy_dist, parameters, args, discrete = _parameters(payload)
    points = payload.get("points", 200)
    if isinstance(points, bool) or not isinstance(points, int) or not 2 <= points <= MAX_CURVE_POINTS:
        raise APIError(f"'points' debe ser un entero entre 2 y {MAX_CURVE_POINTS}.", details={"field": "points"})
    lower = payload.get("x_min", scipy_dist.ppf(0.001, *args))
    upper = payload.get("x_max", scipy_dist.ppf(0.999, *args))
    lower, upper = _finite_number(lower, "x_min"), _finite_number(upper, "x_max")
    if lower >= upper:
        raise APIError("'x_min' debe ser menor que 'x_max'.")
    if discrete:
        grid = np.arange(math.ceil(lower), math.floor(upper) + 1, dtype=float)
        if len(grid) > points:
            grid = grid[np.linspace(0, len(grid) - 1, points, dtype=int)]
        density = scipy_dist.pmf(grid, *args)
        function = "PMF"
    else:
        grid = np.linspace(lower, upper, points)
        density = scipy_dist.pdf(grid, *args)
        function = "PDF"
    cdf_values = scipy_dist.cdf(grid, *args)
    values = [
        {"x": float(x), "density": float(y), "cdf": float(cumulative)}
        for x, y, cumulative in zip(grid, density, cdf_values)
    ]
    results = {
        "values": values,
        "metadata": {"function": function, "points": len(values), "x_min": lower, "x_max": upper, "discrete": discrete},
    }
    return _response(name, "curve", parameters, results, f"Curva {function} y CDF listas para representación.")


def _dataset_values(payload: dict):
    require_fields(payload, "dataset_id", "column")
    dataset = current_app.extensions["dataset_store"].get(payload["dataset_id"])
    if not isinstance(payload["column"], str):
        raise APIError("'column' debe ser texto.", details={"field": "column"})
    values = numeric_series(dataset, payload["column"], 8).to_numpy(dtype=float)
    if not np.all(np.isfinite(values)):
        raise APIError("La columna contiene valores numéricos no finitos.", details={"field": "column"})
    if np.unique(values).size < 2:
        raise APIError("El ajuste requiere variación en los datos.")
    return dataset, values


def _fit_name(payload: dict) -> str:
    require_fields(payload, "distribution")
    name = payload["distribution"]
    if not isinstance(name, str) or name not in FIT_SPECS:
        raise APIError("La distribución no es una candidata continua soportada.", details={"allowed": sorted(FIT_SPECS)})
    return name


def _bootstrap_reps(payload: dict) -> int:
    value = payload.get("bootstrap_reps", 200)
    if isinstance(value, bool) or not isinstance(value, int) or not 20 <= value <= 1000:
        raise APIError("'bootstrap_reps' debe ser un entero entre 20 y 1000.", details={"field": "bootstrap_reps"})
    return value


def _fit_route(operation: str):
    payload = get_json_object()
    dataset, values = _dataset_values(payload)
    name = _fit_name(payload)
    reps = _bootstrap_reps(payload)
    try:
        results = fit_distribution(values, name, reps)
    except (ValueError, TypeError, FloatingPointError, RuntimeError) as error:
        raise APIError("No fue posible ajustar la distribución.", details={"reason": str(error)}) from error
    results.pop("_fitted_tuple", None)
    parameters = results["fitted_parameters"]
    results["value"] = results["anderson_darling"]["statistic"]
    results["metadata"] = {"dataset_id": dataset.id, "column": payload["column"]}
    return _response(name, operation, parameters, results, "Ajuste MLE con pruebas de bondad de ajuste; revise las advertencias de cada prueba.")


@distributions_bp.post("/fit")
def fit():
    return _fit_route("fit")


@distributions_bp.post("/goodness-of-fit")
def goodness_of_fit():
    return _fit_route("goodness-of-fit")


@distributions_bp.post("/identify")
def identify():
    payload = get_json_object()
    dataset, values = _dataset_values(payload)
    candidates = payload.get("candidates", list(DEFAULT_CANDIDATES))
    if not isinstance(candidates, list) or not 1 <= len(candidates) <= len(FIT_SPECS):
        raise APIError("'candidates' debe ser una lista no vacía de candidatas continuas.", details={"field": "candidates"})
    if any(not isinstance(item, str) or item not in FIT_SPECS for item in candidates):
        raise APIError("'candidates' contiene una distribución no soportada.", details={"allowed": sorted(FIT_SPECS)})
    if len(set(candidates)) != len(candidates):
        raise APIError("'candidates' no debe contener duplicados.", details={"field": "candidates"})
    reps = _bootstrap_reps(payload)
    results = identify_distributions(values, candidates, reps)
    results["values"] = results["candidates"]
    results["metadata"] = {"dataset_id": dataset.id, "column": payload["column"], "bootstrap_reps": reps}
    return _response("multiple", "identify", {}, results, "Las candidatas se ordenan por Anderson-Darling y luego AIC; los fallos individuales no cancelan el análisis.")


@distributions_bp.post("/probability-plot")
def probability_plot_route():
    payload = get_json_object()
    dataset, values = _dataset_values(payload)
    name = _fit_name(payload)
    confidence = _finite_number(payload.get("confidence", 0.95), "confidence")
    if not 0 < confidence < 1:
        raise APIError("'confidence' debe estar estrictamente entre 0 y 1.", details={"field": "confidence"})
    try:
        results = probability_plot(values, name, confidence)
    except (ValueError, TypeError, FloatingPointError, RuntimeError) as error:
        raise APIError("No fue posible construir el gráfico de probabilidad.", details={"reason": str(error)}) from error
    parameters = results["fitted_parameters"]
    results["values"] = results["points"]
    results["metadata"] = {"dataset_id": dataset.id, "column": payload["column"]}
    return _response(name, "probability-plot", parameters, results, "Los puntos próximos a la línea de referencia son compatibles visualmente con el modelo ajustado.")
