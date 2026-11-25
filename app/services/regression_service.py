from __future__ import annotations

import itertools
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import optimize, special, stats

from app.models.dataset import Dataset
from app.schemas.regression import RegressionAnalysisRequest

ENGINE_VERSION = "regression-1.0"
MAX_PLOT_POINTS = 500


class RegressionAnalysisError(ValueError):
    pass


MODEL_CATALOG: list[dict[str, Any]] = [
    {
        "key": "linear",
        "label": "Regresión lineal",
        "category": "Respuesta continua",
        "description": "Ajusta una respuesta continua con uno o más predictores.",
    },
    {
        "key": "fitted_line",
        "label": "Línea ajustada",
        "category": "Respuesta continua",
        "description": "Compara ajustes lineal, cuadrático o cúbico para un predictor.",
    },
    {
        "key": "stepwise",
        "label": "Regresión escalonada",
        "category": "Selección de modelo",
        "description": "Agrega o elimina predictores usando umbrales de significancia.",
    },
    {
        "key": "best_subsets",
        "label": "Mejores subconjuntos",
        "category": "Selección de modelo",
        "description": "Compara los mejores modelos de cada tamaño.",
    },
    {
        "key": "pls",
        "label": "Mínimos cuadrados parciales",
        "category": "Selección de modelo",
        "description": "Extrae componentes cuando los predictores son numerosos o colineales.",
    },
    {
        "key": "binary_logistic",
        "label": "Logística binaria",
        "category": "Respuesta categórica",
        "description": "Modela la probabilidad de uno de dos resultados.",
    },
    {
        "key": "ordinal_logistic",
        "label": "Logística ordinal",
        "category": "Respuesta categórica",
        "description": "Modela categorías que tienen un orden natural.",
    },
    {
        "key": "nominal_logistic",
        "label": "Logística nominal",
        "category": "Respuesta categórica",
        "description": "Modela tres o más categorías sin orden natural.",
    },
]


@dataclass
class PreparedData:
    x: np.ndarray
    y: np.ndarray
    feature_names: list[str]
    row_numbers: list[int]
    response_name: str
    predictor_names: list[str]
    raw_response: list[Any]
    missing_rows: int
    numeric_sources: dict[str, np.ndarray]


@dataclass
class OLSFit:
    beta: np.ndarray
    fitted: np.ndarray
    residuals: np.ndarray
    covariance: np.ndarray
    standard_errors: np.ndarray
    t_values: np.ndarray
    p_values: np.ndarray
    df_error: int
    rank: int
    mse: float
    sse: float
    sst: float
    r_squared: float
    adjusted_r_squared: float
    predicted_r_squared: float | None
    leverage: np.ndarray
    studentized: np.ndarray
    cook_distance: np.ndarray
    dffits: np.ndarray
    aic: float
    aicc: float | None
    bic: float


def regression_catalog() -> list[dict[str, Any]]:
    return MODEL_CATALOG


def analyze_regression(
    dataset: Dataset, request: RegressionAnalysisRequest
) -> tuple[dict[str, Any], list[str]]:
    columns = {column["key"]: column for column in dataset.columns_data}
    requested = [request.response, *request.predictors]
    missing = [key for key in requested if key not in columns]
    if missing:
        raise RegressionAnalysisError(f"Columnas no encontradas: {', '.join(sorted(set(missing)))}")

    handlers: dict[
        str, Callable[[Dataset, RegressionAnalysisRequest], tuple[dict[str, Any], list[str]]]
    ] = {
        "linear": _linear_analysis,
        "fitted_line": _linear_analysis,
        "stepwise": _stepwise_analysis,
        "best_subsets": _best_subsets_analysis,
        "pls": _pls_analysis,
        "binary_logistic": _binary_logistic_analysis,
        "ordinal_logistic": _ordinal_logistic_analysis,
        "nominal_logistic": _nominal_logistic_analysis,
    }
    try:
        result, warnings = handlers[request.model](dataset, request)
    except RegressionAnalysisError:
        raise
    except (ValueError, TypeError, np.linalg.LinAlgError, FloatingPointError) as error:
        raise RegressionAnalysisError(
            "Los datos no permiten ajustar el modelo con la configuración indicada"
        ) from error

    result = _json_value(result)
    spec = next(item for item in MODEL_CATALOG if item["key"] == request.model)
    return (
        {
            "dataset": {
                "id": dataset.id,
                "name": dataset.name,
                "revision": dataset.revision,
                "content_sha256": dataset.content_sha256,
                "row_count": dataset.row_count,
            },
            "model": {
                "key": request.model,
                "label": spec["label"],
                "category": spec["category"],
                "response": request.response,
                "predictors": request.predictors,
            },
            **result,
            "guidance": [
                "Una asociación del modelo no demuestra causalidad.",
                "Revise los residuos y las observaciones influyentes antes de "
                "interpretar coeficientes.",
                "La selección automática propone modelos candidatos; debe "
                "confirmarlos con conocimiento del problema.",
                "Las predicciones son más confiables dentro del rango observado "
                "de los predictores.",
            ],
        },
        list(dict.fromkeys(warnings)),
    )


def _linear_analysis(
    dataset: Dataset, request: RegressionAnalysisRequest
) -> tuple[dict[str, Any], list[str]]:
    prepared = _prepare_continuous_data(dataset, request)
    fit = _fit_ols(prepared.x, prepared.y)
    result = _ols_result(prepared, fit, request)
    if request.model == "fitted_line":
        result["plots"]["fitted_line"] = _fitted_line_plot(prepared, fit, request)
    return result, _regression_warnings(prepared, fit)


def _stepwise_analysis(
    dataset: Dataset, request: RegressionAnalysisRequest
) -> tuple[dict[str, Any], list[str]]:
    if request.categorical_predictors or request.interactions:
        raise RegressionAnalysisError(
            "La selección escalonada inicial admite predictores numéricos sin interacciones"
        )
    selected, history = _select_predictors(dataset, request)
    if not selected:
        raise RegressionAnalysisError("Ningún predictor cumplió los criterios de entrada")
    selected_request = request.model_copy(update={"model": "linear", "predictors": selected})
    prepared = _prepare_continuous_data(dataset, selected_request)
    fit = _fit_ols(prepared.x, prepared.y)
    result = _ols_result(prepared, fit, selected_request)
    result["selection"] = {
        "method": request.selection_method,
        "selected_predictors": selected,
        "history": history,
        "alpha_to_enter": request.alpha_to_enter,
        "alpha_to_remove": request.alpha_to_remove,
    }
    return result, [
        *_regression_warnings(prepared, fit),
        "La selección escalonada puede producir coeficientes y p-values optimistas.",
    ]


def _best_subsets_analysis(
    dataset: Dataset, request: RegressionAnalysisRequest
) -> tuple[dict[str, Any], list[str]]:
    if request.categorical_predictors or request.interactions:
        raise RegressionAnalysisError("Mejores subconjuntos requiere predictores numéricos")
    full_request = request.model_copy(update={"model": "linear"})
    full_data = _prepare_continuous_data(dataset, full_request)
    full_fit = _fit_ols(full_data.x, full_data.y)
    candidates: list[dict[str, Any]] = []
    best_by_size: list[dict[str, Any]] = []
    for size in range(1, len(request.predictors) + 1):
        models = []
        for predictors in itertools.combinations(request.predictors, size):
            candidate_request = request.model_copy(
                update={"model": "linear", "predictors": list(predictors)}
            )
            data = _prepare_continuous_data(dataset, candidate_request)
            fit = _fit_ols(data.x, data.y)
            parameter_count = len(fit.beta)
            cp = fit.sse / full_fit.mse - (len(data.y) - 2 * parameter_count)
            models.append(
                {
                    "predictors": list(predictors),
                    "r_squared": fit.r_squared,
                    "adjusted_r_squared": fit.adjusted_r_squared,
                    "predicted_r_squared": fit.predicted_r_squared,
                    "s": math.sqrt(fit.mse),
                    "mallows_cp": cp,
                    "aicc": fit.aicc,
                    "bic": fit.bic,
                }
            )
        models.sort(key=lambda item: (item["bic"], -(item["predicted_r_squared"] or -math.inf)))
        chosen = models[: request.best_models_per_size]
        best_by_size.extend(chosen)
        candidates.extend(models)
    best = min(candidates, key=lambda item: item["bic"])
    best_request = request.model_copy(update={"model": "linear", "predictors": best["predictors"]})
    prepared = _prepare_continuous_data(dataset, best_request)
    fit = _fit_ols(prepared.x, prepared.y)
    result = _ols_result(prepared, fit, best_request)
    result["selection"] = {
        "method": "best_subsets",
        "selected_predictors": best["predictors"],
        "models": best_by_size,
        "criterion": "BIC mínimo",
    }
    return result, [
        *_regression_warnings(prepared, fit),
        "Explore los modelos candidatos con diagnósticos; el criterio mínimo no garantiza validez.",
    ]


def _pls_analysis(
    dataset: Dataset, request: RegressionAnalysisRequest
) -> tuple[dict[str, Any], list[str]]:
    prepared = _prepare_continuous_data(dataset, request, force_numeric=True, intercept=False)
    maximum = min(request.components, prepared.x.shape[1], prepared.x.shape[0] - 1)
    if maximum < 1:
        raise RegressionAnalysisError("PLS requiere más observaciones y predictores válidos")
    cv = _pls_cross_validation(prepared.x, prepared.y, maximum, request.scale)
    components = min(request.components, maximum)
    model = _fit_pls(prepared.x, prepared.y, components, request.scale)
    fitted = model["fitted"]
    residuals = prepared.y - fitted
    sse = float(residuals @ residuals)
    sst = float(np.sum((prepared.y - np.mean(prepared.y)) ** 2))
    r_squared = 1 - sse / sst if sst > 0 else 0
    predictions = [
        {
            "row": row,
            "observed": float(observed),
            "fitted": float(predicted),
            "residual": float(observed - predicted),
        }
        for row, observed, predicted in zip(prepared.row_numbers, prepared.y, fitted, strict=True)
    ]
    return (
        {
            "equation": _equation(
                prepared.response_name,
                ["Constante", *prepared.feature_names],
                np.r_[model["intercept"], model["coefficients"]],
            ),
            "fit": {
                "n": len(prepared.y),
                "components": components,
                "r_squared": r_squared,
                "rmse": math.sqrt(sse / len(prepared.y)),
                "cross_validation": cv,
            },
            "coefficients": [
                {"term": "Constante", "coefficient": model["intercept"]},
                *[
                    {"term": name, "coefficient": coefficient}
                    for name, coefficient in zip(
                        prepared.feature_names, model["coefficients"], strict=True
                    )
                ],
            ],
            "anova": None,
            "diagnostics": {
                "loadings": model["loadings"],
                "weights": model["weights"],
                "scores": _sample_rows(model["scores"]),
            },
            "predictions": predictions,
            "plots": {
                "observed_fitted": _observed_fitted_plot(prepared.y, fitted),
                "residuals": _residual_plots(prepared, fitted, residuals),
            },
        },
        _missing_warnings(prepared),
    )


def _binary_logistic_analysis(
    dataset: Dataset, request: RegressionAnalysisRequest
) -> tuple[dict[str, Any], list[str]]:
    prepared = _prepare_categorical_response(dataset, request)
    categories = sorted({str(value) for value in prepared.raw_response})
    event = str(request.event_value)
    if event not in categories or len(categories) != 2:
        raise RegressionAnalysisError(
            "La respuesta binaria debe contener dos categorías y el evento indicado"
        )
    y = np.asarray([1.0 if str(value) == event else 0.0 for value in prepared.raw_response])
    if y.min() == y.max():
        raise RegressionAnalysisError("La respuesta requiere eventos y no eventos")
    beta, covariance, converged, iterations, log_likelihood = _fit_binary_logit(prepared.x, y)
    standard_errors = np.sqrt(np.clip(np.diag(covariance), 0, math.inf))
    z_values = np.divide(beta, standard_errors, out=np.zeros_like(beta), where=standard_errors > 0)
    p_values = 2 * stats.norm.sf(np.abs(z_values))
    critical = stats.norm.ppf((1 + request.confidence_level) / 2)
    probabilities = special.expit(prepared.x @ beta)
    null_probability = np.mean(y)
    null_log_likelihood = float(
        np.sum(y * math.log(null_probability) + (1 - y) * math.log(1 - null_probability))
    )
    deviance = -2 * log_likelihood
    null_deviance = -2 * null_log_likelihood
    roc = _roc_curve(y, probabilities)
    predicted = probabilities >= 0.5
    coefficients = []
    for name, coefficient, error, z_value, p_value in zip(
        prepared.feature_names, beta, standard_errors, z_values, p_values, strict=True
    ):
        coefficients.append(
            {
                "term": name,
                "coefficient": float(coefficient),
                "standard_error": float(error),
                "statistic": float(z_value),
                "p_value": float(p_value),
                "lower": float(coefficient - critical * error),
                "upper": float(coefficient + critical * error),
                "odds_ratio": float(math.exp(np.clip(coefficient, -700, 700))),
            }
        )
    predictions = [
        {
            "row": row,
            "observed": str(observed),
            "probability": float(probability),
            "predicted": event if is_event else next(item for item in categories if item != event),
        }
        for row, observed, probability, is_event in zip(
            prepared.row_numbers, prepared.raw_response, probabilities, predicted, strict=True
        )
    ]
    warnings = _missing_warnings(prepared)
    if not converged:
        warnings.append("El optimizador no convergió; no interprete coeficientes ni intervalos.")
    if np.max(np.abs(beta)) > 20:
        warnings.append("Coeficientes extremos sugieren separación completa o casi completa.")
    return (
        {
            "equation": f"logit(P({prepared.response_name} = {event})) = "
            + _linear_expression(prepared.feature_names, beta),
            "fit": {
                "n": len(y),
                "events": int(np.sum(y)),
                "event": event,
                "log_likelihood": log_likelihood,
                "deviance": deviance,
                "null_deviance": null_deviance,
                "deviance_r_squared": 1 - deviance / null_deviance if null_deviance else None,
                "aic": deviance + 2 * len(beta),
                "bic": deviance + len(beta) * math.log(len(y)),
                "auc": roc["auc"],
                "accuracy": float(np.mean(predicted == y)),
                "converged": converged,
                "iterations": iterations,
            },
            "coefficients": coefficients,
            "anova": None,
            "diagnostics": {
                "roc": roc,
                "classification": _classification(y, predicted),
                "residual_deviance": _binary_deviance_residuals(y, probabilities).tolist(),
            },
            "predictions": predictions,
            "plots": {
                "probability": _probability_plot(prepared, probabilities),
                "roc": roc["points"],
            },
        },
        warnings,
    )


def _ordinal_logistic_analysis(
    dataset: Dataset, request: RegressionAnalysisRequest
) -> tuple[dict[str, Any], list[str]]:
    prepared = _prepare_categorical_response(dataset, request, intercept=False)
    levels = [str(value) for value in request.category_order or []]
    observed_levels = {str(value) for value in prepared.raw_response}
    if set(levels) != observed_levels:
        raise RegressionAnalysisError(
            "El orden debe incluir exactamente todas las categorías observadas"
        )
    y = np.asarray([levels.index(str(value)) for value in prepared.raw_response])
    beta, thresholds, covariance, converged, iterations, log_likelihood = _fit_ordinal_logit(
        prepared.x, y, len(levels)
    )
    probabilities = _ordinal_probabilities(prepared.x, beta, thresholds)
    predicted_index = np.argmax(probabilities, axis=1)
    parameter_names = [
        *prepared.feature_names,
        *[f"Umbral {levels[index]} | {levels[index + 1]}" for index in range(len(levels) - 1)],
    ]
    parameters = np.r_[beta, thresholds]
    errors = np.sqrt(np.clip(np.diag(covariance), 0, math.inf))
    coefficients = [
        {
            "term": name,
            "coefficient": float(value),
            "standard_error": float(error),
            "statistic": float(value / error) if error else None,
            "p_value": float(2 * stats.norm.sf(abs(value / error))) if error else None,
        }
        for name, value, error in zip(parameter_names, parameters, errors, strict=True)
    ]
    predictions = [
        {
            "row": row,
            "observed": str(observed),
            "predicted": levels[int(predicted)],
            "probabilities": {
                level: float(probability)
                for level, probability in zip(levels, row_probabilities, strict=True)
            },
        }
        for row, observed, predicted, row_probabilities in zip(
            prepared.row_numbers, prepared.raw_response, predicted_index, probabilities, strict=True
        )
    ]
    warnings = _missing_warnings(prepared)
    if not converged:
        warnings.append("El modelo ordinal no convergió; revise categorías y predictores.")
    return (
        {
            "equation": "logit(P(Y ≤ categoría)) = umbral − ("
            + _linear_expression(prepared.feature_names, beta)
            + ")",
            "fit": {
                "n": len(y),
                "categories": levels,
                "log_likelihood": log_likelihood,
                "deviance": -2 * log_likelihood,
                "aic": -2 * log_likelihood + 2 * len(parameters),
                "bic": -2 * log_likelihood + len(parameters) * math.log(len(y)),
                "accuracy": float(np.mean(predicted_index == y)),
                "converged": converged,
                "iterations": iterations,
            },
            "coefficients": coefficients,
            "anova": None,
            "diagnostics": {"confusion_matrix": _multiclass_confusion(y, predicted_index, levels)},
            "predictions": predictions,
            "plots": {"probability": _multiclass_probability_plot(probabilities, levels)},
        },
        warnings,
    )


def _nominal_logistic_analysis(
    dataset: Dataset, request: RegressionAnalysisRequest
) -> tuple[dict[str, Any], list[str]]:
    prepared = _prepare_categorical_response(dataset, request)
    levels = sorted({str(value) for value in prepared.raw_response})
    if len(levels) < 3:
        raise RegressionAnalysisError("La regresión nominal requiere al menos tres categorías")
    reference = (
        str(request.reference_category) if request.reference_category is not None else levels[0]
    )
    if reference not in levels:
        raise RegressionAnalysisError("La categoría de referencia no existe en la respuesta")
    levels = [reference, *[level for level in levels if level != reference]]
    y = np.asarray([levels.index(str(value)) for value in prepared.raw_response])
    beta, covariance, converged, iterations, log_likelihood = _fit_nominal_logit(
        prepared.x, y, len(levels)
    )
    probabilities = _softmax_with_reference(prepared.x, beta)
    predicted_index = np.argmax(probabilities, axis=1)
    errors = np.sqrt(np.clip(np.diag(covariance), 0, math.inf)).reshape(beta.shape)
    coefficients = []
    for category_index, category in enumerate(levels[1:]):
        for feature_index, feature in enumerate(prepared.feature_names):
            value = beta[category_index, feature_index]
            error = errors[category_index, feature_index]
            coefficients.append(
                {
                    "category": category,
                    "reference": reference,
                    "term": feature,
                    "coefficient": float(value),
                    "standard_error": float(error),
                    "statistic": float(value / error) if error else None,
                    "p_value": float(2 * stats.norm.sf(abs(value / error))) if error else None,
                    "relative_risk_ratio": float(math.exp(np.clip(value, -700, 700))),
                }
            )
    predictions = [
        {
            "row": row,
            "observed": str(observed),
            "predicted": levels[int(predicted)],
            "probabilities": {
                level: float(probability)
                for level, probability in zip(levels, row_probabilities, strict=True)
            },
        }
        for row, observed, predicted, row_probabilities in zip(
            prepared.row_numbers, prepared.raw_response, predicted_index, probabilities, strict=True
        )
    ]
    parameter_count = beta.size
    warnings = _missing_warnings(prepared)
    if not converged:
        warnings.append("El modelo nominal no convergió; revise categorías y predictores.")
    return (
        {
            "equation": f"log(P(categoría) / P({reference})) = β₀ + ΣβX",
            "fit": {
                "n": len(y),
                "categories": levels,
                "reference_category": reference,
                "log_likelihood": log_likelihood,
                "deviance": -2 * log_likelihood,
                "aic": -2 * log_likelihood + 2 * parameter_count,
                "bic": -2 * log_likelihood + parameter_count * math.log(len(y)),
                "accuracy": float(np.mean(predicted_index == y)),
                "converged": converged,
                "iterations": iterations,
            },
            "coefficients": coefficients,
            "anova": None,
            "diagnostics": {"confusion_matrix": _multiclass_confusion(y, predicted_index, levels)},
            "predictions": predictions,
            "plots": {"probability": _multiclass_probability_plot(probabilities, levels)},
        },
        warnings,
    )


def _prepare_continuous_data(
    dataset: Dataset,
    request: RegressionAnalysisRequest,
    force_numeric: bool = False,
    intercept: bool | None = None,
) -> PreparedData:
    rows, response, row_numbers, missing = _complete_rows(dataset, request)
    y_values = [_finite_number(value) for value in response]
    if any(value is None for value in y_values):
        raise RegressionAnalysisError("La respuesta debe ser numérica y finita")
    use_intercept = request.include_intercept if intercept is None else intercept
    x, feature_names, numeric_sources = _design_matrix(rows, request, force_numeric, use_intercept)
    y = np.asarray(y_values, dtype=float)
    if len(y) <= x.shape[1]:
        raise RegressionAnalysisError("Se requieren más observaciones que parámetros del modelo")
    return PreparedData(
        x=x,
        y=y,
        feature_names=_feature_labels(dataset, feature_names),
        row_numbers=row_numbers,
        response_name=_column_name(dataset, request.response),
        predictor_names=[_column_name(dataset, key) for key in request.predictors],
        raw_response=response,
        missing_rows=missing,
        numeric_sources=numeric_sources,
    )


def _prepare_categorical_response(
    dataset: Dataset, request: RegressionAnalysisRequest, intercept: bool = True
) -> PreparedData:
    rows, response, row_numbers, missing = _complete_rows(dataset, request)
    if len(response) < 3:
        raise RegressionAnalysisError("Se requieren al menos tres respuestas completas")
    x, feature_names, numeric_sources = _design_matrix(rows, request, False, intercept)
    if len(response) <= x.shape[1]:
        raise RegressionAnalysisError("Se requieren más observaciones que parámetros del modelo")
    return PreparedData(
        x=x,
        y=np.zeros(len(response)),
        feature_names=_feature_labels(dataset, feature_names),
        row_numbers=row_numbers,
        response_name=_column_name(dataset, request.response),
        predictor_names=[_column_name(dataset, key) for key in request.predictors],
        raw_response=response,
        missing_rows=missing,
        numeric_sources=numeric_sources,
    )


def _complete_rows(
    dataset: Dataset, request: RegressionAnalysisRequest
) -> tuple[list[dict[str, Any]], list[Any], list[int], int]:
    complete = []
    response = []
    row_numbers = []
    missing = 0
    for row_number, row in enumerate(dataset.rows_data, start=1):
        values = [row.get(request.response), *[row.get(key) for key in request.predictors]]
        if any(value is None or value == "" for value in values):
            missing += 1
            continue
        complete.append(row)
        response.append(row.get(request.response))
        row_numbers.append(row_number)
    if not complete:
        raise RegressionAnalysisError("No hay filas completas para ajustar el modelo")
    return complete, response, row_numbers, missing


def _design_matrix(
    rows: list[dict[str, Any]],
    request: RegressionAnalysisRequest,
    force_numeric: bool,
    intercept: bool,
) -> tuple[np.ndarray, list[str], dict[str, np.ndarray]]:
    columns = []
    names = []
    numeric_sources: dict[str, np.ndarray] = {}
    if intercept:
        columns.append(np.ones(len(rows)))
        names.append("Constante")
    for key in request.predictors:
        values = [row.get(key) for row in rows]
        if key in request.categorical_predictors and not force_numeric:
            levels = sorted({str(value) for value in values})
            reference = levels[0]
            for level in levels[1:]:
                columns.append(
                    np.asarray([1.0 if str(value) == level else 0.0 for value in values])
                )
                names.append(f"{key}[{level}] (ref. {reference})")
        else:
            numeric = np.asarray([_required_number(value, key) for value in values])
            numeric_sources[key] = numeric
            columns.append(numeric)
            names.append(key)
            if request.model == "fitted_line":
                for degree in range(2, request.degree + 1):
                    columns.append(numeric**degree)
                    names.append(f"{key}^{degree}")
    for first, second in request.interactions:
        if first not in numeric_sources or second not in numeric_sources:
            raise RegressionAnalysisError("Las interacciones requieren predictores numéricos")
        columns.append(numeric_sources[first] * numeric_sources[second])
        names.append(f"{first} × {second}")
    return np.column_stack(columns), names, numeric_sources


def _fit_ols(x: np.ndarray, y: np.ndarray) -> OLSFit:
    beta, _, rank, _ = np.linalg.lstsq(x, y, rcond=None)
    if rank < x.shape[1]:
        raise RegressionAnalysisError(
            "La matriz del modelo no tiene rango completo; elimine términos redundantes"
        )
    fitted = x @ beta
    residuals = y - fitted
    n, p = x.shape
    df_error = n - rank
    if df_error <= 0:
        raise RegressionAnalysisError("El modelo no deja grados de libertad para estimar el error")
    sse = float(residuals @ residuals)
    mse = sse / df_error
    inverse = np.linalg.pinv(x.T @ x)
    covariance = inverse * mse
    standard_errors = np.sqrt(np.clip(np.diag(covariance), 0, math.inf))
    t_values = np.divide(beta, standard_errors, out=np.zeros_like(beta), where=standard_errors > 0)
    p_values = 2 * stats.t.sf(np.abs(t_values), df_error)
    centered = y - np.mean(y)
    sst = float(centered @ centered)
    r_squared = 1 - sse / sst if sst > 0 else 0
    adjusted = 1 - (1 - r_squared) * (n - 1) / df_error if df_error else 0
    leverage = np.sum((x @ inverse) * x, axis=1)
    press_residuals = np.divide(
        residuals,
        1 - leverage,
        out=np.full_like(residuals, np.nan),
        where=np.abs(1 - leverage) > 1e-12,
    )
    press = float(np.nansum(press_residuals**2))
    predicted_r_squared = 1 - press / sst if sst > 0 and np.isfinite(press) else None
    scale = math.sqrt(mse) if mse > 0 else 0
    studentized = np.divide(
        residuals,
        scale * np.sqrt(np.clip(1 - leverage, 1e-12, math.inf)),
        out=np.zeros_like(residuals),
        where=scale > 0,
    )
    cook = studentized**2 * leverage / (rank * np.clip(1 - leverage, 1e-12, math.inf))
    dffits = studentized * np.sqrt(leverage / np.clip(1 - leverage, 1e-12, math.inf))
    safe_sse = max(sse, np.finfo(float).tiny)
    aic = n * math.log(safe_sse / n) + 2 * p
    bic = n * math.log(safe_sse / n) + p * math.log(n)
    aicc = aic + 2 * p * (p + 1) / (n - p - 1) if n > p + 1 else None
    return OLSFit(
        beta,
        fitted,
        residuals,
        covariance,
        standard_errors,
        t_values,
        p_values,
        df_error,
        rank,
        mse,
        sse,
        sst,
        r_squared,
        adjusted,
        predicted_r_squared,
        leverage,
        studentized,
        cook,
        dffits,
        aic,
        aicc,
        bic,
    )


def _ols_result(
    prepared: PreparedData, fit: OLSFit, request: RegressionAnalysisRequest
) -> dict[str, Any]:
    critical = stats.t.ppf((1 + request.confidence_level) / 2, fit.df_error)
    coefficients = [
        {
            "term": name,
            "coefficient": float(coefficient),
            "standard_error": float(error),
            "statistic": float(statistic),
            "p_value": float(p_value),
            "lower": float(coefficient - critical * error),
            "upper": float(coefficient + critical * error),
            "vif": None,
        }
        for name, coefficient, error, statistic, p_value in zip(
            prepared.feature_names,
            fit.beta,
            fit.standard_errors,
            fit.t_values,
            fit.p_values,
            strict=True,
        )
    ]
    vifs = _vif(prepared.x, prepared.feature_names)
    for row in coefficients:
        row["vif"] = vifs.get(row["term"])
    df_model = fit.rank - (1 if request.include_intercept else 0)
    ssr = max(0.0, fit.sst - fit.sse)
    msr = ssr / df_model if df_model else None
    f_value = msr / fit.mse if msr is not None and fit.mse else None
    f_p = float(stats.f.sf(f_value, df_model, fit.df_error)) if f_value is not None else None
    predictions = _ols_predictions(prepared, fit, request.confidence_level)
    return {
        "equation": _equation(prepared.response_name, prepared.feature_names, fit.beta),
        "fit": {
            "n": len(prepared.y),
            "parameters": fit.rank,
            "degrees_of_freedom_error": fit.df_error,
            "s": math.sqrt(fit.mse),
            "r_squared": fit.r_squared,
            "adjusted_r_squared": fit.adjusted_r_squared,
            "predicted_r_squared": fit.predicted_r_squared,
            "press": None
            if fit.predicted_r_squared is None
            else fit.sst * (1 - fit.predicted_r_squared),
            "aic": fit.aic,
            "aicc": fit.aicc,
            "bic": fit.bic,
        },
        "coefficients": coefficients,
        "anova": [
            {
                "source": "Regresión",
                "df": df_model,
                "sum_squares": ssr,
                "mean_square": msr,
                "statistic": f_value,
                "p_value": f_p,
            },
            {
                "source": "Error",
                "df": fit.df_error,
                "sum_squares": fit.sse,
                "mean_square": fit.mse,
                "statistic": None,
                "p_value": None,
            },
            {
                "source": "Total",
                "df": len(prepared.y) - 1,
                "sum_squares": fit.sst,
                "mean_square": None,
                "statistic": None,
                "p_value": None,
            },
        ],
        "diagnostics": _ols_diagnostics(prepared, fit, vifs),
        "predictions": predictions,
        "plots": {
            "observed_fitted": _observed_fitted_plot(prepared.y, fit.fitted),
            "residuals": _residual_plots(prepared, fit.fitted, fit.residuals),
        },
    }


def _select_predictors(
    dataset: Dataset, request: RegressionAnalysisRequest
) -> tuple[list[str], list[dict[str, Any]]]:
    selected = list(request.predictors) if request.selection_method == "backward" else []
    remaining = [key for key in request.predictors if key not in selected]
    history = []
    step = 0
    while True:
        step += 1
        changed = False
        if request.selection_method in {"forward", "stepwise"} and remaining:
            choices = []
            for candidate in remaining:
                candidate_request = request.model_copy(
                    update={"model": "linear", "predictors": [*selected, candidate]}
                )
                data = _prepare_continuous_data(dataset, candidate_request)
                fit = _fit_ols(data.x, data.y)
                choices.append((float(fit.p_values[-1]), candidate))
            p_value, candidate = min(choices)
            if p_value <= request.alpha_to_enter:
                selected.append(candidate)
                remaining.remove(candidate)
                changed = True
                history.append(
                    {
                        "step": step,
                        "action": "enter",
                        "predictor": candidate,
                        "p_value": p_value,
                        "model": selected[:],
                    }
                )
        if request.selection_method in {"backward", "stepwise"} and len(selected) > 1:
            data = _prepare_continuous_data(
                dataset, request.model_copy(update={"model": "linear", "predictors": selected})
            )
            fit = _fit_ols(data.x, data.y)
            start = 1 if request.include_intercept else 0
            term_p = list(zip(selected, fit.p_values[start : start + len(selected)], strict=True))
            candidate, p_value = max(term_p, key=lambda item: item[1])
            if p_value > request.alpha_to_remove:
                selected.remove(candidate)
                remaining.append(candidate)
                changed = True
                history.append(
                    {
                        "step": step,
                        "action": "remove",
                        "predictor": candidate,
                        "p_value": float(p_value),
                        "model": selected[:],
                    }
                )
        if not changed or step > len(request.predictors) * 4:
            break
    return selected, history


def _fit_pls(x: np.ndarray, y: np.ndarray, components: int, scale: bool) -> dict[str, Any]:
    x_mean = np.mean(x, axis=0)
    y_mean = float(np.mean(y))
    x_scale = np.std(x, axis=0, ddof=1) if scale else np.ones(x.shape[1])
    y_scale = float(np.std(y, ddof=1)) if scale else 1.0
    x_scale[x_scale == 0] = 1
    y_scale = y_scale or 1.0
    x_residual = (x - x_mean) / x_scale
    y_residual = (y - y_mean) / y_scale
    weights = []
    loadings = []
    q_values = []
    scores = []
    for _ in range(components):
        weight = x_residual.T @ y_residual
        norm = np.linalg.norm(weight)
        if norm <= 1e-12:
            break
        weight /= norm
        score = x_residual @ weight
        denominator = float(score @ score)
        loading = x_residual.T @ score / denominator
        q_value = float(y_residual @ score / denominator)
        x_residual -= np.outer(score, loading)
        y_residual -= score * q_value
        weights.append(weight)
        loadings.append(loading)
        q_values.append(q_value)
        scores.append(score)
    if not weights:
        raise RegressionAnalysisError("PLS no pudo extraer componentes")
    w = np.column_stack(weights)
    p = np.column_stack(loadings)
    q = np.asarray(q_values)
    scaled_coef = w @ np.linalg.pinv(p.T @ w) @ q
    coefficients = scaled_coef * y_scale / x_scale
    intercept = y_mean - float(x_mean @ coefficients)
    fitted = intercept + x @ coefficients
    return {
        "coefficients": coefficients,
        "intercept": intercept,
        "fitted": fitted,
        "weights": w.T.tolist(),
        "loadings": p.T.tolist(),
        "scores": np.column_stack(scores),
    }


def _pls_cross_validation(
    x: np.ndarray, y: np.ndarray, maximum: int, scale: bool
) -> list[dict[str, Any]]:
    folds = min(5, len(y))
    assignments = np.arange(len(y)) % folds
    results = []
    for components in range(1, maximum + 1):
        errors = []
        for fold in range(folds):
            train = assignments != fold
            test = ~train
            model = _fit_pls(x[train], y[train], min(components, np.sum(train) - 1), scale)
            predicted = model["intercept"] + x[test] @ model["coefficients"]
            errors.extend((y[test] - predicted).tolist())
        results.append(
            {
                "components": components,
                "rmse": math.sqrt(float(np.mean(np.asarray(errors) ** 2))),
                "press": float(np.sum(np.asarray(errors) ** 2)),
            }
        )
    return results


def _fit_binary_logit(
    x: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, bool, int, float]:
    def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        linear = x @ beta
        value = float(np.sum(np.logaddexp(0, linear) - y * linear))
        gradient = x.T @ (special.expit(linear) - y)
        return value, gradient

    result = optimize.minimize(
        objective, np.zeros(x.shape[1]), jac=True, method="BFGS", options={"maxiter": 1000}
    )
    beta = result.x
    probabilities = special.expit(x @ beta)
    weights = probabilities * (1 - probabilities)
    covariance = np.linalg.pinv(x.T @ (x * weights[:, None]))
    log_likelihood = float(
        np.sum(
            y * np.log(np.clip(probabilities, 1e-15, 1))
            + (1 - y) * np.log(np.clip(1 - probabilities, 1e-15, 1))
        )
    )
    return beta, covariance, bool(result.success), int(result.nit), log_likelihood


def _fit_ordinal_logit(
    x: np.ndarray, y: np.ndarray, categories: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool, int, float]:
    predictor_count = x.shape[1]
    initial_thresholds = np.linspace(-1, 1, categories - 1)
    raw_thresholds = np.r_[initial_thresholds[0], np.log(np.diff(initial_thresholds))]
    initial = np.r_[np.zeros(predictor_count), raw_thresholds]

    def unpack(parameters: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        beta = parameters[:predictor_count]
        raw = parameters[predictor_count:]
        thresholds = np.r_[raw[0], raw[0] + np.cumsum(np.exp(raw[1:]))]
        return beta, thresholds

    def objective(parameters: np.ndarray) -> float:
        beta, thresholds = unpack(parameters)
        probabilities = _ordinal_probabilities(x, beta, thresholds)
        return float(-np.sum(np.log(np.clip(probabilities[np.arange(len(y)), y], 1e-15, 1))))

    result = optimize.minimize(objective, initial, method="BFGS", options={"maxiter": 1500})
    beta, thresholds = unpack(result.x)
    covariance_raw = np.asarray(result.hess_inv)
    jacobian = np.zeros((len(result.x), len(result.x)))
    jacobian[:predictor_count, :predictor_count] = np.eye(predictor_count)
    for row in range(categories - 1):
        jacobian[predictor_count + row, predictor_count] = 1
        for column in range(1, row + 1):
            jacobian[predictor_count + row, predictor_count + column] = math.exp(
                result.x[predictor_count + column]
            )
    covariance = jacobian @ covariance_raw @ jacobian.T
    return beta, thresholds, covariance, bool(result.success), int(result.nit), -float(result.fun)


def _ordinal_probabilities(x: np.ndarray, beta: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    cumulative = special.expit(thresholds[None, :] - (x @ beta)[:, None])
    return np.column_stack([cumulative[:, 0], np.diff(cumulative, axis=1), 1 - cumulative[:, -1]])


def _fit_nominal_logit(
    x: np.ndarray, y: np.ndarray, categories: int
) -> tuple[np.ndarray, np.ndarray, bool, int, float]:
    shape = (categories - 1, x.shape[1])

    def objective(flat: np.ndarray) -> tuple[float, np.ndarray]:
        beta = flat.reshape(shape)
        probabilities = _softmax_with_reference(x, beta)
        value = -float(np.sum(np.log(np.clip(probabilities[np.arange(len(y)), y], 1e-15, 1))))
        indicators = np.eye(categories)[y][:, 1:]
        gradient = (probabilities[:, 1:] - indicators).T @ x
        return value, gradient.ravel()

    result = optimize.minimize(
        objective, np.zeros(np.prod(shape)), jac=True, method="BFGS", options={"maxiter": 1500}
    )
    beta = result.x.reshape(shape)
    covariance = np.asarray(result.hess_inv)
    return beta, covariance, bool(result.success), int(result.nit), -float(result.fun)


def _softmax_with_reference(x: np.ndarray, beta: np.ndarray) -> np.ndarray:
    logits = np.column_stack([np.zeros(len(x)), x @ beta.T])
    logits -= np.max(logits, axis=1, keepdims=True)
    exponentials = np.exp(logits)
    return exponentials / np.sum(exponentials, axis=1, keepdims=True)


def _ols_predictions(
    prepared: PreparedData, fit: OLSFit, confidence: float
) -> list[dict[str, Any]]:
    critical = stats.t.ppf((1 + confidence) / 2, fit.df_error)
    inverse = np.linalg.pinv(prepared.x.T @ prepared.x)
    mean_se = np.sqrt(
        np.clip(np.sum((prepared.x @ inverse) * prepared.x, axis=1) * fit.mse, 0, math.inf)
    )
    prediction_se = np.sqrt(mean_se**2 + fit.mse)
    rows = []
    for index, (row, observed, fitted, residual) in enumerate(
        zip(prepared.row_numbers, prepared.y, fit.fitted, fit.residuals, strict=True)
    ):
        rows.append(
            {
                "row": row,
                "observed": float(observed),
                "fitted": float(fitted),
                "residual": float(residual),
                "mean_lower": float(fitted - critical * mean_se[index]),
                "mean_upper": float(fitted + critical * mean_se[index]),
                "prediction_lower": float(fitted - critical * prediction_se[index]),
                "prediction_upper": float(fitted + critical * prediction_se[index]),
                "leverage": float(fit.leverage[index]),
                "studentized_residual": float(fit.studentized[index]),
                "cook_distance": float(fit.cook_distance[index]),
                "dffits": float(fit.dffits[index]),
            }
        )
    return rows


def _ols_diagnostics(
    prepared: PreparedData, fit: OLSFit, vifs: dict[str, float | None]
) -> dict[str, Any]:
    normality = (
        stats.shapiro(fit.residuals)
        if 3 <= len(fit.residuals) <= 5000 and np.ptp(fit.residuals) > 0
        else None
    )
    differences = np.diff(fit.residuals)
    durbin_watson = float(differences @ differences / fit.sse) if fit.sse else None
    influential = []
    for index, row in enumerate(prepared.row_numbers):
        if (
            fit.cook_distance[index] > 4 / len(prepared.y)
            or fit.leverage[index] > 2 * fit.rank / len(prepared.y)
            or abs(fit.studentized[index]) > 2
        ):
            influential.append(
                {
                    "row": row,
                    "residual": float(fit.residuals[index]),
                    "studentized_residual": float(fit.studentized[index]),
                    "leverage": float(fit.leverage[index]),
                    "cook_distance": float(fit.cook_distance[index]),
                    "dffits": float(fit.dffits[index]),
                }
            )
    theoretical, ordered = stats.probplot(fit.residuals, dist="norm", fit=False)
    return {
        "durbin_watson": durbin_watson,
        "normality": None
        if normality is None
        else {
            "test": "Shapiro–Wilk",
            "statistic": float(normality.statistic),
            "p_value": float(normality.pvalue),
        },
        "vif": vifs,
        "influential_observations": influential,
        "qq_plot": {
            "theoretical": _sample_array(np.asarray(theoretical)),
            "observed": _sample_array(np.asarray(ordered)),
        },
    }


def _vif(x: np.ndarray, names: list[str]) -> dict[str, float | None]:
    result = {}
    for index, name in enumerate(names):
        if name == "Constante":
            result[name] = None
            continue
        target = x[:, index]
        others = np.delete(x, index, axis=1)
        if others.shape[1] == 0:
            result[name] = 1.0
            continue
        fitted = others @ np.linalg.lstsq(others, target, rcond=None)[0]
        sst = float(np.sum((target - np.mean(target)) ** 2))
        sse = float(np.sum((target - fitted) ** 2))
        r_squared = 1 - sse / sst if sst else 1
        result[name] = None if r_squared >= 1 - 1e-12 else 1 / (1 - r_squared)
    return result


def _regression_warnings(prepared: PreparedData, fit: OLSFit) -> list[str]:
    warnings = _missing_warnings(prepared)
    high_vif = [
        name
        for name, value in _vif(prepared.x, prepared.feature_names).items()
        if value is not None and value > 5
    ]
    if high_vif:
        warnings.append(f"Multicolinealidad elevada (VIF > 5): {', '.join(high_vif)}.")
    influential = int(np.sum(fit.cook_distance > 4 / len(prepared.y)))
    if influential:
        warnings.append(f"{influential} observaciones superan el umbral orientativo de Cook (4/n).")
    if (
        fit.predicted_r_squared is not None
        and fit.adjusted_r_squared - fit.predicted_r_squared > 0.2
    ):
        warnings.append("R² predicho es sustancialmente menor que R² ajustado; revise sobreajuste.")
    return warnings


def _fitted_line_plot(
    prepared: PreparedData, fit: OLSFit, request: RegressionAnalysisRequest
) -> dict[str, Any]:
    predictor = prepared.numeric_sources[request.predictors[0]]
    order = np.argsort(predictor)
    grid = np.linspace(float(np.min(predictor)), float(np.max(predictor)), 120)
    columns = [np.ones(len(grid))] if request.include_intercept else []
    for degree in range(1, request.degree + 1):
        columns.append(grid**degree)
    matrix = np.column_stack(columns)
    fitted = matrix @ fit.beta
    inverse = np.linalg.pinv(prepared.x.T @ prepared.x)
    mean_se = np.sqrt(np.clip(np.sum((matrix @ inverse) * matrix, axis=1) * fit.mse, 0, math.inf))
    critical = stats.t.ppf((1 + request.confidence_level) / 2, fit.df_error)
    prediction_se = np.sqrt(mean_se**2 + fit.mse)
    return {
        "x": grid.tolist(),
        "fitted": fitted.tolist(),
        "mean_lower": (fitted - critical * mean_se).tolist(),
        "mean_upper": (fitted + critical * mean_se).tolist(),
        "prediction_lower": (fitted - critical * prediction_se).tolist(),
        "prediction_upper": (fitted + critical * prediction_se).tolist(),
        "observed_x": _sample_array(predictor[order]),
        "observed_y": _sample_array(prepared.y[order]),
    }


def _observed_fitted_plot(observed: np.ndarray, fitted: np.ndarray) -> dict[str, list[float]]:
    indices = _sample_indices(len(observed))
    return {"observed": observed[indices].tolist(), "fitted": fitted[indices].tolist()}


def _residual_plots(
    prepared: PreparedData, fitted: np.ndarray, residuals: np.ndarray
) -> dict[str, list[float]]:
    indices = _sample_indices(len(residuals))
    return {
        "row": [prepared.row_numbers[index] for index in indices],
        "fitted": fitted[indices].tolist(),
        "residual": residuals[indices].tolist(),
    }


def _probability_plot(prepared: PreparedData, probabilities: np.ndarray) -> dict[str, Any]:
    first_numeric = next(iter(prepared.numeric_sources.values()), np.arange(len(probabilities)))
    order = np.argsort(first_numeric)
    indices = order[_sample_indices(len(order))]
    return {"x": first_numeric[indices].tolist(), "probability": probabilities[indices].tolist()}


def _multiclass_probability_plot(probabilities: np.ndarray, levels: list[str]) -> dict[str, Any]:
    indices = _sample_indices(len(probabilities))
    return {
        "row": (indices + 1).tolist(),
        "levels": levels,
        "probabilities": probabilities[indices].tolist(),
    }


def _roc_curve(y: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    thresholds = np.r_[math.inf, np.unique(probability)[::-1], -math.inf]
    points = []
    positives = max(1, int(np.sum(y == 1)))
    negatives = max(1, int(np.sum(y == 0)))
    for threshold in thresholds:
        predicted = probability >= threshold
        points.append(
            {
                "threshold": None if not math.isfinite(threshold) else float(threshold),
                "false_positive_rate": float(np.sum(predicted & (y == 0)) / negatives),
                "true_positive_rate": float(np.sum(predicted & (y == 1)) / positives),
            }
        )
    points.sort(key=lambda item: (item["false_positive_rate"], item["true_positive_rate"]))
    auc = float(
        np.trapezoid(
            [item["true_positive_rate"] for item in points],
            [item["false_positive_rate"] for item in points],
        )
    )
    return {"auc": auc, "points": points}


def _classification(y: np.ndarray, predicted: np.ndarray) -> dict[str, int]:
    return {
        "true_positive": int(np.sum((predicted == 1) & (y == 1))),
        "true_negative": int(np.sum((predicted == 0) & (y == 0))),
        "false_positive": int(np.sum((predicted == 1) & (y == 0))),
        "false_negative": int(np.sum((predicted == 0) & (y == 1))),
    }


def _multiclass_confusion(
    observed: np.ndarray, predicted: np.ndarray, levels: list[str]
) -> dict[str, Any]:
    matrix = np.zeros((len(levels), len(levels)), dtype=int)
    for actual, estimate in zip(observed, predicted, strict=True):
        matrix[int(actual), int(estimate)] += 1
    return {"levels": levels, "matrix": matrix.tolist()}


def _binary_deviance_residuals(y: np.ndarray, probability: np.ndarray) -> np.ndarray:
    probability = np.clip(probability, 1e-15, 1 - 1e-15)
    contribution = np.where(y == 1, -2 * np.log(probability), -2 * np.log(1 - probability))
    return np.sign(y - probability) * np.sqrt(contribution)


def _equation(response: str, names: list[str], coefficients: np.ndarray) -> str:
    return f"{response} = " + _linear_expression(names, coefficients)


def _linear_expression(names: list[str], coefficients: np.ndarray) -> str:
    parts = []
    for index, (name, coefficient) in enumerate(zip(names, coefficients, strict=True)):
        magnitude = abs(float(coefficient))
        term = f"{magnitude:.6g}" if name == "Constante" else f"{magnitude:.6g}·{name}"
        if index == 0:
            parts.append(("−" if coefficient < 0 else "") + term)
        else:
            parts.append((" − " if coefficient < 0 else " + ") + term)
    return "".join(parts)


def _column_name(dataset: Dataset, key: str) -> str:
    return next((column["name"] for column in dataset.columns_data if column["key"] == key), key)


def _required_number(value: Any, key: str) -> float:
    parsed = _finite_number(value)
    if parsed is None:
        raise RegressionAnalysisError(f"{key} contiene valores no numéricos")
    return parsed


def _finite_number(value: Any) -> float | None:
    if value is None or value == "" or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _missing_warnings(prepared: PreparedData) -> list[str]:
    return (
        []
        if prepared.missing_rows == 0
        else [
            f"Se excluyeron {prepared.missing_rows} filas incompletas mediante "
            "eliminación por lista."
        ]
    )


def _feature_labels(dataset: Dataset, features: list[str]) -> list[str]:
    labels = {column["key"]: column["name"] for column in dataset.columns_data}
    output = []
    for feature in features:
        if feature == "Constante":
            output.append(feature)
            continue
        rendered = feature
        for key, label in sorted(labels.items(), key=lambda item: len(item[0]), reverse=True):
            if rendered == key or rendered.startswith(f"{key}^") or rendered.startswith(f"{key}["):
                rendered = f"{label}{rendered[len(key) :]}"
                break
            rendered = rendered.replace(f"{key} × ", f"{label} × ").replace(
                f" × {key}", f" × {label}"
            )
        output.append(rendered)
    return output


def _sample_indices(count: int) -> np.ndarray:
    return (
        np.arange(count)
        if count <= MAX_PLOT_POINTS
        else np.linspace(0, count - 1, MAX_PLOT_POINTS, dtype=int)
    )


def _sample_array(values: np.ndarray) -> list[float]:
    return values[_sample_indices(len(values))].astype(float).tolist()


def _sample_rows(values: np.ndarray) -> list[list[float]]:
    return values[_sample_indices(len(values))].astype(float).tolist()


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_json_value(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        parsed = float(value)
        return parsed if math.isfinite(parsed) else None
    return value
