import numpy as np
import pandas as pd
from flask import Blueprint, current_app, jsonify
from scipy import stats
from statsmodels.stats.descriptivestats import sign_test

from modules.advanced_stats import (confidence_interval, matrix, normality, partial_correlation,
                                    one_sample_z, power_analysis, proportion_one, proportion_two,
                                    rank_correlation)
from modules.hypothesis import decision
from utils.errors import APIError
from utils.serialization import json_safe
from utils.validation import (alpha_value, choice, column_list, get_json_object, number_value,
                              numeric_series, require_fields, validate_columns)

basic_stats_bp = Blueprint("basic_stats", __name__, url_prefix="/api/basic-stats")
variance_bp = Blueprint("variance", __name__, url_prefix="/api/variance")
nonparametric_bp = Blueprint("nonparametric", __name__, url_prefix="/api/nonparametric")
proportion_bp = Blueprint("proportion", __name__, url_prefix="/api/proportion")
correlation_bp = Blueprint("correlation", __name__, url_prefix="/api/correlation")
power_bp = Blueprint("power", __name__, url_prefix="/api/power")


def _dataset(payload):
    require_fields(payload, "dataset_id")
    return current_app.extensions["dataset_store"].get(payload["dataset_id"])


def _response(dataset, test, results, alpha=None):
    body = {"dataset_id": dataset.id if dataset else None, "test": test, "results": results}
    if alpha is not None:
        body["alpha"] = alpha
        if results.get("p_value") is not None:
            body["interpretation"] = decision(results["p_value"], alpha, f"en la prueba {test}")
        elif results.get("reject_normality") is not None:
            body["interpretation"] = ("Se rechaza el supuesto de normalidad."
                                      if results["reject_normality"] else "No se rechaza el supuesto de normalidad.")
    if "interpretation" not in body:
        descriptions = {
            "confidence_interval": "El intervalo muestra el rango compatible con la estimación y el nivel de confianza elegido.",
            "basic_stats_by_group": "Compare centro y dispersión entre los grupos antes de aplicar pruebas inferenciales.",
            "correlation_matrix": "Los valores cercanos a -1 o 1 indican asociaciones más intensas.",
            "covariance_matrix": "El signo indica la dirección de la variación conjunta; su magnitud depende de las unidades.",
        }
        body["interpretation"] = descriptions.get(test, "El cálculo se completó con los parámetros indicados.")
    return jsonify(json_safe(body))


def _paired(dataset, columns, minimum=2):
    validate_columns(dataset, columns)
    data = dataset.frame[columns].apply(pd.to_numeric, errors="coerce").dropna().astype(float)
    if len(data) < minimum:
        raise APIError(f"Se requieren al menos {minimum} observaciones numéricas completas.")
    return data


def _groups(payload, minimum_groups=2, minimum_size=2):
    require_fields(payload, "value_column", "group_column")
    dataset = _dataset(payload)
    validate_columns(dataset, [payload["value_column"], payload["group_column"]])
    groups = [(name, pd.to_numeric(frame[payload["value_column"]], errors="coerce").dropna().astype(float))
              for name, frame in dataset.frame.groupby(payload["group_column"], dropna=False)]
    groups = [(name, values) for name, values in groups if len(values) >= minimum_size]
    if len(groups) < minimum_groups:
        raise APIError(f"Se requieren al menos {minimum_groups} grupos con {minimum_size} observaciones cada uno.")
    return dataset, groups


@basic_stats_bp.post("/confidence-interval")
def confidence_interval_route():
    payload = get_json_object(); require_fields(payload, "column")
    dataset = _dataset(payload)
    statistic = choice(payload, "statistic", {"mean", "median"}, "mean")
    confidence = number_value(payload, "confidence", default=0.95, minimum=0.5, maximum=0.999)
    resamples = number_value(payload, "resamples", default=5000, minimum=100, maximum=50000, integer=True)
    seed = number_value(payload, "seed", default=0, minimum=0, maximum=2**32 - 1, integer=True)
    values = numeric_series(dataset, payload["column"], 2)
    return _response(dataset, "confidence_interval", confidence_interval(values, confidence, statistic, resamples, seed))


@basic_stats_bp.post("/z-test")
def z_test_route():
    payload = get_json_object(); require_fields(payload, "column", "null_mean", "population_std")
    dataset = _dataset(payload); alpha = alpha_value(payload)
    values = numeric_series(dataset, payload["column"], 2)
    null_mean = number_value(payload, "null_mean")
    population_std = number_value(payload, "population_std", minimum=0.000001)
    alternative = choice(payload, "alternative", {"two-sided", "smaller", "larger"}, "two-sided")
    return _response(dataset, "one_sample_z", one_sample_z(values, null_mean, population_std, alpha, alternative), alpha)


@basic_stats_bp.post("/by-group")
def by_group():
    payload = get_json_object(); dataset, groups = _groups(payload, 1, 1)
    results = [{"group": name, "count": len(v), "mean": v.mean(), "median": v.median(),
                "std": v.std(ddof=1), "min": v.min(), "max": v.max()} for name, v in groups]
    return _response(dataset, "basic_stats_by_group", {"groups": results})


@basic_stats_bp.post("/correlation-matrix")
def correlation_matrix():
    payload = get_json_object(); dataset = _dataset(payload)
    columns = column_list(dataset, payload, minimum=2)
    for column in columns: numeric_series(dataset, column, 2)
    method = choice(payload, "method", {"pearson", "spearman", "kendall"}, "pearson")
    return _response(dataset, "correlation_matrix", matrix(dataset.frame, columns, method))


@basic_stats_bp.post("/covariance-matrix")
def covariance_matrix():
    payload = get_json_object(); dataset = _dataset(payload)
    columns = column_list(dataset, payload, minimum=2)
    for column in columns: numeric_series(dataset, column, 2)
    return _response(dataset, "covariance_matrix", matrix(dataset.frame, columns, "pearson", True))


@variance_bp.post("/normality")
def normality_route():
    payload = get_json_object(); require_fields(payload, "column")
    dataset = _dataset(payload); alpha = alpha_value(payload)
    method = choice(payload, "method", {"shapiro", "anderson", "kolmogorov-smirnov", "dagostino"}, "shapiro")
    minimum = 8 if method == "dagostino" else 3
    values = numeric_series(dataset, payload["column"], minimum)
    if values.nunique() < 2: raise APIError("La prueba de normalidad requiere variación.")
    if method == "shapiro" and len(values) > 5000:
        values = values.sample(5000, random_state=0)
    return _response(dataset, "normality", normality(values, method, alpha), alpha)


def _variance_test(method):
    payload = get_json_object(); dataset, groups = _groups(payload)
    alpha = alpha_value(payload)
    if pd.concat([values for _, values in groups]).nunique() < 2:
        raise APIError("La prueba de varianzas requiere variación.")
    if method == "bartlett" and any(values.nunique() < 2 for _, values in groups):
        raise APIError("Bartlett requiere variación dentro de cada grupo.")
    result = stats.levene(*(v for _, v in groups), center=choice(payload, "center", {"mean", "median", "trimmed"}, "median")) if method == "levene" else stats.bartlett(*(v for _, v in groups))
    return _response(dataset, method, {"statistic": result.statistic, "p_value": result.pvalue,
                                      "groups": [{"name": n, "count": len(v), "variance": v.var(ddof=1)} for n, v in groups]}, alpha)


@variance_bp.post("/levene")
def levene(): return _variance_test("levene")


@variance_bp.post("/bartlett")
def bartlett(): return _variance_test("bartlett")


def _two_column_rank(method, paired=False):
    payload = get_json_object(); require_fields(payload, "column1", "column2")
    dataset = _dataset(payload); alpha = alpha_value(payload)
    if paired:
        data = _paired(dataset, [payload["column1"], payload["column2"]])
        if np.allclose(data.iloc[:, 0], data.iloc[:, 1]): raise APIError("Las diferencias no pueden ser todas cero.")
        result = stats.wilcoxon(data.iloc[:, 0], data.iloc[:, 1])
    else:
        first = numeric_series(dataset, payload["column1"], 1); second = numeric_series(dataset, payload["column2"], 1)
        result = stats.mannwhitneyu(first, second, alternative=choice(payload, "alternative", {"two-sided", "less", "greater"}, "two-sided"))
    return _response(dataset, method, {"statistic": result.statistic, "p_value": result.pvalue}, alpha)


@nonparametric_bp.post("/mann-whitney")
def mann_whitney(): return _two_column_rank("mann_whitney")


@nonparametric_bp.post("/wilcoxon")
def wilcoxon(): return _two_column_rank("wilcoxon", True)


@nonparametric_bp.post("/kruskal-wallis")
def kruskal():
    payload = get_json_object(); dataset, groups = _groups(payload); alpha = alpha_value(payload)
    result = stats.kruskal(*(v for _, v in groups))
    return _response(dataset, "kruskal_wallis", {"statistic": result.statistic, "p_value": result.pvalue,
                     "groups": [{"name": n, "count": len(v)} for n, v in groups]}, alpha)


@nonparametric_bp.post("/friedman")
def friedman():
    payload = get_json_object(); dataset = _dataset(payload); columns = column_list(dataset, payload, minimum=3)
    data = _paired(dataset, columns, 2); alpha = alpha_value(payload); result = stats.friedmanchisquare(*(data[c] for c in columns))
    return _response(dataset, "friedman", {"statistic": result.statistic, "p_value": result.pvalue,
                     "blocks": len(data), "columns": columns}, alpha)


@nonparametric_bp.post("/sign-test")
def sign_test_route():
    payload = get_json_object(); require_fields(payload, "column")
    dataset = _dataset(payload); values = numeric_series(dataset, payload["column"], 1); alpha = alpha_value(payload)
    mu0 = number_value(payload, "median", default=0); statistic, p_value = sign_test(values, mu0)
    return _response(dataset, "sign_test", {"statistic": statistic, "p_value": p_value, "null_median": mu0}, alpha)


def _count(payload, field): return number_value(payload, field, minimum=0, integer=True)


@proportion_bp.post("/one-sample")
def one_proportion():
    payload = get_json_object(); require_fields(payload, "successes", "nobs", "null_proportion")
    successes, nobs = _count(payload, "successes"), _count(payload, "nobs")
    if nobs < 1 or successes > nobs: raise APIError("'successes' debe estar entre 0 y 'nobs'.")
    null = number_value(payload, "null_proportion", minimum=0.000001, maximum=0.999999); alpha = alpha_value(payload)
    alternative = choice(payload, "alternative", {"two-sided", "smaller", "larger"}, "two-sided")
    return _response(None, "one_sample_proportion", proportion_one(successes, nobs, null, alpha, alternative), alpha)


@proportion_bp.post("/two-sample")
def two_proportion():
    payload = get_json_object(); require_fields(payload, "successes", "nobs")
    successes, nobs = payload["successes"], payload["nobs"]
    if not all(isinstance(v, list) and len(v) == 2 and all(isinstance(x, int) and not isinstance(x, bool) for x in v) for v in [successes, nobs]) or any(n < 1 or s < 0 or s > n for s, n in zip(successes, nobs)):
        raise APIError("'successes' y 'nobs' deben ser listas válidas de dos enteros.")
    if sum(successes) in {0, sum(nobs)}: raise APIError("La prueba z requiere éxitos y fracasos observados.")
    alpha = alpha_value(payload); alternative = choice(payload, "alternative", {"two-sided", "smaller", "larger"}, "two-sided")
    return _response(None, "two_sample_proportion", proportion_two(successes, nobs, alpha, alternative), alpha)


def _correlation(method):
    payload = get_json_object(); require_fields(payload, "x", "y"); dataset = _dataset(payload); alpha = alpha_value(payload)
    data = _paired(dataset, [payload["x"], payload["y"]], 3)
    if data.iloc[:, 0].nunique() < 2 or data.iloc[:, 1].nunique() < 2: raise APIError("Ambas columnas requieren variación.")
    return _response(dataset, method, rank_correlation(data.iloc[:, 0], data.iloc[:, 1], method), alpha)


@correlation_bp.post("/spearman")
def spearman(): return _correlation("spearman")


@correlation_bp.post("/kendall")
def kendall(): return _correlation("kendall")


@correlation_bp.post("/partial")
def partial():
    payload = get_json_object(); require_fields(payload, "x", "y"); dataset = _dataset(payload)
    controls = column_list(dataset, payload, "controls", minimum=1, maximum=10); validate_columns(dataset, [payload["x"], payload["y"]])
    if payload["x"] in controls or payload["y"] in controls or payload["x"] == payload["y"]: raise APIError("x, y y controls deben ser columnas distintas.")
    complete = dataset.frame[[payload["x"], payload["y"], *controls]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(complete) <= len(controls) + 2: raise APIError("No hay suficientes observaciones para los controles indicados.")
    results = partial_correlation(dataset.frame, payload["x"], payload["y"], controls); alpha = alpha_value(payload)
    return _response(dataset, "partial_correlation", results, alpha)


def _power(kind, payload=None):
    payload = payload or get_json_object(); require_fields(payload, "effect_size")
    effect = number_value(payload, "effect_size", minimum=0.000001); alpha = alpha_value(payload)
    target = payload.get("power"); nobs = payload.get("sample_size")
    if (target is None) == (nobs is None): raise APIError("Indique exactamente uno de 'power' o 'sample_size'.")
    power = number_value(payload, "power", minimum=0.01, maximum=0.999) if target is not None else None
    nobs = number_value(payload, "sample_size", minimum=2) if nobs is not None else None
    groups = number_value(payload, "groups", default=2, minimum=2, maximum=100, integer=True)
    alternative = choice(payload, "alternative", {"two-sided", "larger", "smaller"}, "two-sided")
    try:
        results = power_analysis(kind, effect, alpha, power, nobs, alternative, groups)
    except (ValueError, TypeError) as exc:
        raise APIError("No se pudo resolver el análisis de potencia con esos parámetros.", details=str(exc)) from exc
    return _response(None, f"power_{kind}", results)


@power_bp.post("/t-test")
def t_power():
    payload = get_json_object()
    return _power(choice(payload, "design", {"t-test", "t-test-independent"}, "t-test-independent"), payload)


@power_bp.post("/anova")
def anova_power(): return _power("anova")


@power_bp.post("/sample-size")
def sample_size():
    payload = get_json_object(); kind = choice(payload, "test", {"t-test", "t-test-independent", "anova"})
    if "power" not in payload: payload["power"] = 0.8
    return _power(kind, payload)
