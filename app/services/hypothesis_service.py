from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable
from typing import Any

import numpy as np
from scipy import stats

from app.models.dataset import Dataset
from app.schemas.hypothesis import HypothesisAnalysisRequest

ENGINE_VERSION = "hypothesis-1.0"


class HypothesisAnalysisError(ValueError):
    pass


TEST_CATALOG: list[dict[str, Any]] = [
    {
        "key": "one_sample_z",
        "label": "Z de una muestra",
        "category": "Medias",
        "design": "one_numeric",
        "description": "Compara una media con un valor cuando σ es conocida.",
    },
    {
        "key": "one_sample_t",
        "label": "t de una muestra",
        "category": "Medias",
        "design": "one_numeric",
        "description": "Compara una media con un valor de referencia.",
    },
    {
        "key": "two_sample_t",
        "label": "t de dos muestras",
        "category": "Medias",
        "design": "grouped_numeric",
        "description": "Compara las medias de dos grupos independientes.",
    },
    {
        "key": "paired_t",
        "label": "t pareada",
        "category": "Medias",
        "design": "paired_numeric",
        "description": "Compara dos mediciones realizadas sobre las mismas unidades.",
    },
    {
        "key": "one_proportion",
        "label": "Una proporción",
        "category": "Proporciones y tasas",
        "design": "one_categorical",
        "description": "Compara la proporción de un evento con un valor de referencia.",
    },
    {
        "key": "two_proportions",
        "label": "Dos proporciones",
        "category": "Proporciones y tasas",
        "design": "grouped_categorical",
        "description": "Compara la proporción de un evento en dos grupos.",
    },
    {
        "key": "one_sample_poisson_rate",
        "label": "Tasa Poisson de una muestra",
        "category": "Proporciones y tasas",
        "design": "one_numeric",
        "description": "Compara una tasa de ocurrencia con un valor de referencia.",
    },
    {
        "key": "two_sample_poisson_rate",
        "label": "Tasas Poisson de dos muestras",
        "category": "Proporciones y tasas",
        "design": "grouped_numeric",
        "description": "Compara tasas de ocurrencia de dos grupos.",
    },
    {
        "key": "one_variance",
        "label": "Una varianza",
        "category": "Dispersión",
        "design": "one_numeric",
        "description": "Compara una varianza con un valor de referencia.",
    },
    {
        "key": "two_variances",
        "label": "Dos varianzas",
        "category": "Dispersión",
        "design": "grouped_numeric",
        "description": "Compara la variabilidad de dos grupos.",
    },
    {
        "key": "correlation",
        "label": "Correlación",
        "category": "Asociación",
        "design": "paired_numeric",
        "description": "Contrasta una correlación lineal o monótona.",
    },
    {
        "key": "chi_square_independence",
        "label": "Chi-cuadrado de independencia",
        "category": "Asociación",
        "design": "paired_categorical",
        "description": "Evalúa la asociación entre dos variables categóricas.",
    },
    {
        "key": "normality",
        "label": "Normalidad",
        "category": "Ajuste y aleatoriedad",
        "design": "one_numeric",
        "description": "Evalúa si una muestra es compatible con una distribución normal.",
    },
    {
        "key": "poisson_goodness_of_fit",
        "label": "Bondad de ajuste Poisson",
        "category": "Ajuste y aleatoriedad",
        "design": "one_numeric",
        "description": "Evalúa si conteos siguen una distribución Poisson.",
    },
    {
        "key": "one_sample_sign",
        "label": "Signos de una muestra",
        "category": "No paramétricas",
        "design": "one_numeric",
        "description": "Compara una mediana sin asumir simetría.",
    },
    {
        "key": "one_sample_wilcoxon",
        "label": "Wilcoxon de una muestra",
        "category": "No paramétricas",
        "design": "one_numeric",
        "description": "Compara el centro de una distribución simétrica.",
    },
    {
        "key": "mann_whitney",
        "label": "Mann–Whitney",
        "category": "No paramétricas",
        "design": "grouped_numeric",
        "description": "Compara dos grupos independientes mediante rangos.",
    },
    {
        "key": "kruskal_wallis",
        "label": "Kruskal–Wallis",
        "category": "No paramétricas",
        "design": "grouped_numeric",
        "description": "Compara dos o más grupos mediante rangos.",
    },
    {
        "key": "mood_median",
        "label": "Mediana de Mood",
        "category": "No paramétricas",
        "design": "grouped_numeric",
        "description": "Compara medianas y es resistente a valores extremos.",
    },
    {
        "key": "friedman",
        "label": "Friedman",
        "category": "No paramétricas",
        "design": "blocked_numeric",
        "description": "Compara tratamientos dentro de bloques relacionados.",
    },
    {
        "key": "runs",
        "label": "Prueba de corridas",
        "category": "Ajuste y aleatoriedad",
        "design": "one_numeric",
        "description": "Evalúa la aleatoriedad del orden de una secuencia.",
    },
]


def hypothesis_catalog() -> list[dict[str, Any]]:
    return TEST_CATALOG


def analyze_hypothesis(
    dataset: Dataset, request: HypothesisAnalysisRequest
) -> tuple[dict[str, Any], list[str]]:
    columns = {column["key"]: column for column in dataset.columns_data}
    requested = [request.column, request.second_column, request.group_by, request.block_by]
    missing = [key for key in requested if key and key not in columns]
    if missing:
        raise HypothesisAnalysisError(f"Columnas no encontradas: {', '.join(sorted(set(missing)))}")

    handlers: dict[
        str, Callable[[Dataset, HypothesisAnalysisRequest], tuple[dict[str, Any], list[str]]]
    ] = {
        "one_sample_z": _one_sample_z,
        "one_sample_t": _one_sample_t,
        "two_sample_t": _two_sample_t,
        "paired_t": _paired_t,
        "one_proportion": _one_proportion,
        "two_proportions": _two_proportions,
        "one_sample_poisson_rate": _one_poisson_rate,
        "two_sample_poisson_rate": _two_poisson_rates,
        "one_variance": _one_variance,
        "two_variances": _two_variances,
        "correlation": _correlation,
        "normality": _normality,
        "chi_square_independence": _chi_square,
        "poisson_goodness_of_fit": _poisson_goodness_of_fit,
        "one_sample_sign": _one_sample_sign,
        "one_sample_wilcoxon": _one_sample_wilcoxon,
        "mann_whitney": _mann_whitney,
        "kruskal_wallis": _kruskal_wallis,
        "mood_median": _mood_median,
        "friedman": _friedman,
        "runs": _runs,
    }
    try:
        result, warnings = handlers[request.test](dataset, request)
    except HypothesisAnalysisError:
        raise
    except (TypeError, ValueError, FloatingPointError) as error:
        raise HypothesisAnalysisError(
            "Los datos no permiten calcular esta prueba con la configuración indicada"
        ) from error
    p_value = result.get("p_value")
    alpha = 1 - request.confidence_level
    decision = None if p_value is None else ("reject" if p_value <= alpha else "fail_to_reject")
    interpretation = _interpretation(decision, request.test, alpha)
    spec = next(item for item in TEST_CATALOG if item["key"] == request.test)
    return (
        {
            "dataset": {
                "id": dataset.id,
                "name": dataset.name,
                "revision": dataset.revision,
                "content_sha256": dataset.content_sha256,
                "row_count": dataset.row_count,
            },
            "test": {"key": request.test, "label": spec["label"], "category": spec["category"]},
            "hypotheses": _hypotheses(request),
            "alpha": alpha,
            "confidence_level": request.confidence_level,
            "alternative": request.alternative,
            "method": result.pop("method"),
            "result": result,
            "decision": decision,
            "interpretation": interpretation,
            "guidance": [
                "Un resultado significativo no demuestra importancia práctica ni causalidad.",
                "No rechazar H₀ no equivale a demostrar que H₀ sea verdadera.",
                "Revise los supuestos, el diseño de muestreo y el tamaño del efecto "
                "junto con el p-value.",
            ],
        },
        list(dict.fromkeys(warnings)),
    )


def _one_sample_z(
    dataset: Dataset, request: HypothesisAnalysisRequest
) -> tuple[dict[str, Any], list[str]]:
    values, missing = _numeric_column(dataset, request.column)
    _require_n(values, 2)
    sigma = float(request.known_std_dev)
    mean = float(np.mean(values))
    se = sigma / math.sqrt(len(values))
    statistic = (mean - request.hypothesized_value) / se
    p_value = _normal_p_value(statistic, request.alternative)
    return _result(
        "Z de una muestra (σ conocida)",
        statistic,
        p_value,
        None,
        mean,
        request.hypothesized_value,
        _normal_interval(mean, se, request),
        {
            "n": len(values),
            "mean": mean,
            "population_std_dev": sigma,
            "standard_error": se,
            "effect_size": (mean - request.hypothesized_value) / sigma,
        },
    ), _missing_warning(missing)


def _one_sample_t(
    dataset: Dataset, request: HypothesisAnalysisRequest
) -> tuple[dict[str, Any], list[str]]:
    values, missing = _numeric_column(dataset, request.column)
    _require_variable(values, 2)
    scipy_alt = _scipy_alternative(request.alternative)
    test = stats.ttest_1samp(values, request.hypothesized_value, alternative=scipy_alt)
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1))
    se = std / math.sqrt(len(values))
    return _result(
        "t de una muestra",
        test.statistic,
        test.pvalue,
        len(values) - 1,
        mean,
        request.hypothesized_value,
        _t_interval(mean, se, len(values) - 1, request),
        {
            "n": len(values),
            "mean": mean,
            "std_dev": std,
            "standard_error": se,
            "effect_size": (mean - request.hypothesized_value) / std,
        },
    ), _missing_warning(missing)


def _two_sample_t(
    dataset: Dataset, request: HypothesisAnalysisRequest
) -> tuple[dict[str, Any], list[str]]:
    groups, missing = _numeric_groups(dataset, request, exactly=2)
    (name1, first), (name2, second) = groups
    _require_variable(first, 2)
    _require_variable(second, 2)
    equal = request.equal_variances
    test = stats.ttest_ind(
        first, second, equal_var=equal, alternative=_scipy_alternative(request.alternative)
    )
    n1, n2 = len(first), len(second)
    var1 = float(np.var(first, ddof=1))
    var2 = float(np.var(second, ddof=1))
    if equal:
        df = n1 + n2 - 2
        pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / df
        se = math.sqrt(pooled_var * (1 / n1 + 1 / n2))
        pooled_std = math.sqrt(pooled_var)
    else:
        se2 = var1 / n1 + var2 / n2
        se = math.sqrt(se2)
        df = se2**2 / ((var1 / n1) ** 2 / (n1 - 1) + (var2 / n2) ** 2 / (n2 - 1))
        pooled_std = math.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    difference = float(np.mean(first) - np.mean(second))
    return _result(
        "t agrupada" if equal else "t de Welch",
        test.statistic,
        test.pvalue,
        df,
        difference,
        request.hypothesized_value,
        _t_interval(difference, se, df, request),
        {
            "groups": [_sample_summary(name1, first), _sample_summary(name2, second)],
            "standard_error": se,
            "effect_size": difference / pooled_std,
        },
    ), _missing_warning(missing)


def _paired_t(
    dataset: Dataset, request: HypothesisAnalysisRequest
) -> tuple[dict[str, Any], list[str]]:
    first, second, missing = _paired_numeric(dataset, request.column, request.second_column)
    differences = first - second
    _require_variable(differences, 2)
    test = stats.ttest_rel(first, second, alternative=_scipy_alternative(request.alternative))
    mean = float(np.mean(differences))
    std = float(np.std(differences, ddof=1))
    se = std / math.sqrt(len(differences))
    return _result(
        "t pareada",
        test.statistic,
        test.pvalue,
        len(differences) - 1,
        mean,
        request.hypothesized_value,
        _t_interval(mean, se, len(differences) - 1, request),
        {
            "pairs": len(differences),
            "mean_first": float(np.mean(first)),
            "mean_second": float(np.mean(second)),
            "std_dev_difference": std,
            "standard_error": se,
            "effect_size": (mean - request.hypothesized_value) / std,
        },
    ), _missing_warning(missing)


def _one_proportion(
    dataset: Dataset, request: HypothesisAnalysisRequest
) -> tuple[dict[str, Any], list[str]]:
    values, missing = _categorical_column(dataset, request.column)
    if not values:
        raise HypothesisAnalysisError("Se requiere al menos una observación válida")
    events = sum(_same_category(value, request.success_value) for value in values)
    n = len(values)
    p0 = request.hypothesized_value
    if not 0 <= p0 <= 1:
        raise HypothesisAnalysisError("La proporción hipotética debe estar entre 0 y 1")
    if request.method in {"auto", "exact"}:
        test = stats.binomtest(events, n, p0, alternative=_scipy_alternative(request.alternative))
        p_value = test.pvalue
        statistic = events / n
        method = "Binomial exacta"
    else:
        se0 = math.sqrt(p0 * (1 - p0) / n)
        if se0 == 0:
            raise HypothesisAnalysisError("La aproximación normal requiere 0 < p₀ < 1")
        statistic = (events / n - p0) / se0
        p_value = _normal_p_value(statistic, request.alternative)
        method = "Aproximación normal"
    estimate = events / n
    interval = _clopper_pearson(events, n, request.confidence_level, request.alternative)
    return _result(
        method,
        statistic,
        p_value,
        None,
        estimate,
        p0,
        interval,
        {
            "n": n,
            "events": events,
            "event": request.success_value,
            "sample_proportion": estimate,
            "effect_size": estimate - p0,
        },
    ), _missing_warning(missing)


def _two_proportions(
    dataset: Dataset, request: HypothesisAnalysisRequest
) -> tuple[dict[str, Any], list[str]]:
    groups, missing = _categorical_groups(dataset, request, exactly=2)
    (name1, first), (name2, second) = groups
    x1 = sum(_same_category(value, request.success_value) for value in first)
    x2 = sum(_same_category(value, request.success_value) for value in second)
    n1, n2 = len(first), len(second)
    p1, p2 = x1 / n1, x2 / n2
    difference = p1 - p2
    if request.method == "exact":
        table = [[x1, n1 - x1], [x2, n2 - x2]]
        test = stats.fisher_exact(table, alternative=_scipy_alternative(request.alternative))
        statistic, p_value, method = test.statistic, test.pvalue, "Fisher exacta"
    else:
        if request.hypothesized_value != 0:
            raise HypothesisAnalysisError(
                "La aproximación agrupada requiere una diferencia hipotética de 0"
            )
        pooled = (x1 + x2) / (n1 + n2)
        se0 = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
        if se0 == 0:
            raise HypothesisAnalysisError("No hay variación binomial suficiente para la prueba")
        statistic = difference / se0
        p_value = _normal_p_value(statistic, request.alternative)
        method = "Aproximación normal agrupada"
    se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    return _result(
        method,
        statistic,
        p_value,
        None,
        difference,
        request.hypothesized_value,
        _normal_interval(difference, se, request),
        {
            "groups": [
                {"name": name1, "n": n1, "events": x1, "proportion": p1},
                {"name": name2, "n": n2, "events": x2, "proportion": p2},
            ],
            "event": request.success_value,
            "standard_error": se,
        },
    ), _missing_warning(missing)


def _one_poisson_rate(
    dataset: Dataset, request: HypothesisAnalysisRequest
) -> tuple[dict[str, Any], list[str]]:
    values, missing = _numeric_column(dataset, request.column)
    _require_counts(values)
    total = int(np.sum(values))
    exposure = len(values) * request.observation_length
    expected = request.hypothesized_value * exposure
    if expected < 0:
        raise HypothesisAnalysisError("La tasa hipotética no puede ser negativa")
    p_value = _poisson_p_value(total, expected, request.alternative)
    rate = total / exposure
    interval = _poisson_interval(total, exposure, request.confidence_level, request.alternative)
    return _result(
        "Poisson exacta",
        rate,
        p_value,
        None,
        rate,
        request.hypothesized_value,
        interval,
        {
            "n": len(values),
            "total_occurrences": total,
            "exposure": exposure,
            "observation_length": request.observation_length,
        },
    ), _missing_warning(missing)


def _two_poisson_rates(
    dataset: Dataset, request: HypothesisAnalysisRequest
) -> tuple[dict[str, Any], list[str]]:
    groups, missing = _numeric_groups(dataset, request, exactly=2)
    (name1, first), (name2, second) = groups
    _require_counts(first)
    _require_counts(second)
    x1, x2 = int(np.sum(first)), int(np.sum(second))
    e1 = len(first) * request.observation_length
    e2 = len(second) * request.second_observation_length
    r1, r2 = x1 / e1, x2 / e2
    difference = r1 - r2
    if request.hypothesized_value != 0:
        raise HypothesisAnalysisError(
            "La prueba de tasas implementada contrasta una diferencia de 0"
        )
    if request.method == "exact":
        test = stats.binomtest(
            x1, x1 + x2, e1 / (e1 + e2), alternative=_scipy_alternative(request.alternative)
        )
        statistic, p_value, method = (
            r1 / r2 if r2 else None,
            test.pvalue,
            "Poisson exacta condicional",
        )
    else:
        se = math.sqrt(x1 / e1**2 + x2 / e2**2)
        if se == 0:
            raise HypothesisAnalysisError("No hay ocurrencias suficientes para comparar tasas")
        statistic = difference / se
        p_value = _normal_p_value(statistic, request.alternative)
        method = "Aproximación normal"
    se = math.sqrt(x1 / e1**2 + x2 / e2**2)
    return _result(
        method,
        statistic,
        p_value,
        None,
        difference,
        0,
        _normal_interval(difference, se, request),
        {
            "groups": [
                {"name": name1, "occurrences": x1, "exposure": e1, "rate": r1},
                {"name": name2, "occurrences": x2, "exposure": e2, "rate": r2},
            ],
            "standard_error": se,
        },
    ), _missing_warning(missing)


def _one_variance(
    dataset: Dataset, request: HypothesisAnalysisRequest
) -> tuple[dict[str, Any], list[str]]:
    values, missing = _numeric_column(dataset, request.column)
    _require_variable(values, 2)
    variance = float(np.var(values, ddof=1))
    null = request.hypothesized_value
    if null <= 0:
        raise HypothesisAnalysisError("La varianza hipotética debe ser mayor que 0")
    df = len(values) - 1
    statistic = df * variance / null
    lower_tail = stats.chi2.cdf(statistic, df)
    upper_tail = stats.chi2.sf(statistic, df)
    p_value = _tail_p_value(lower_tail, upper_tail, request.alternative)
    alpha = 1 - request.confidence_level
    interval = {
        "level": request.confidence_level,
        "lower": df * variance / stats.chi2.ppf(1 - alpha / 2, df),
        "upper": df * variance / stats.chi2.ppf(alpha / 2, df),
    }
    return _result(
        "Chi-cuadrado para una varianza",
        statistic,
        p_value,
        df,
        variance,
        null,
        interval,
        {
            "n": len(values),
            "std_dev": math.sqrt(variance),
            "variance": variance,
            "ratio": variance / null,
        },
    ), _missing_warning(missing)


def _two_variances(
    dataset: Dataset, request: HypothesisAnalysisRequest
) -> tuple[dict[str, Any], list[str]]:
    groups, missing = _numeric_groups(dataset, request, exactly=2)
    (name1, first), (name2, second) = groups
    _require_variable(first, 2)
    _require_variable(second, 2)
    var1, var2 = float(np.var(first, ddof=1)), float(np.var(second, ddof=1))
    ratio = var1 / var2
    null = request.hypothesized_value or 1.0
    statistic = ratio / null
    df1, df2 = len(first) - 1, len(second) - 1
    lower_tail = stats.f.cdf(statistic, df1, df2)
    upper_tail = stats.f.sf(statistic, df1, df2)
    p_value = _tail_p_value(lower_tail, upper_tail, request.alternative)
    alpha = 1 - request.confidence_level
    interval = {
        "level": request.confidence_level,
        "lower": ratio / stats.f.ppf(1 - alpha / 2, df1, df2),
        "upper": ratio / stats.f.ppf(alpha / 2, df1, df2),
    }
    return _result(
        "F para dos varianzas",
        statistic,
        p_value,
        [df1, df2],
        ratio,
        null,
        interval,
        {"groups": [_sample_summary(name1, first), _sample_summary(name2, second)]},
    ), _missing_warning(missing)


def _correlation(
    dataset: Dataset, request: HypothesisAnalysisRequest
) -> tuple[dict[str, Any], list[str]]:
    first, second, missing = _paired_numeric(dataset, request.column, request.second_column)
    _require_variable(first, 3)
    _require_variable(second, 3)
    method = request.method if request.method in {"pearson", "spearman"} else "pearson"
    function = stats.spearmanr if method == "spearman" else stats.pearsonr
    test = function(first, second, alternative=_scipy_alternative(request.alternative))
    coefficient = float(test.statistic)
    interval = _correlation_interval(coefficient, len(first), request.confidence_level)
    return _result(
        "Spearman" if method == "spearman" else "Pearson",
        coefficient,
        test.pvalue,
        len(first) - 2,
        coefficient,
        request.hypothesized_value,
        interval,
        {"n": len(first), "coefficient": coefficient, "effect_size": coefficient},
    ), _missing_warning(missing)


def _normality(
    dataset: Dataset, request: HypothesisAnalysisRequest
) -> tuple[dict[str, Any], list[str]]:
    values, missing = _numeric_column(dataset, request.column)
    _require_variable(values, 3)
    method = (
        request.method
        if request.method in {"anderson_darling", "ryan_joiner", "kolmogorov_smirnov"}
        else "anderson_darling"
    )
    warnings = _missing_warning(missing)
    if method == "anderson_darling":
        statistic, p_value = _anderson_darling(values)
        label = "Anderson–Darling"
    elif method == "ryan_joiner":
        statistic, p_value = _ryan_joiner(values)
        label = "Ryan–Joiner (p-value por simulación)"
        warnings.append(
            "El p-value de Ryan–Joiner se estima con una simulación determinista; "
            "puede diferir de Minitab 15."
        )
    else:
        standardized = (values - np.mean(values)) / np.std(values, ddof=1)
        test = stats.kstest(standardized, "norm")
        statistic, p_value, label = test.statistic, test.pvalue, "Kolmogórov–Smirnov"
        warnings.append(
            "El p-value no incorpora la corrección propietaria usada por algunas "
            "versiones de Minitab."
        )
    return _result(
        label,
        statistic,
        p_value,
        None,
        None,
        None,
        None,
        {
            "n": len(values),
            "mean": float(np.mean(values)),
            "std_dev": float(np.std(values, ddof=1)),
        },
    ), warnings


def _chi_square(
    dataset: Dataset, request: HypothesisAnalysisRequest
) -> tuple[dict[str, Any], list[str]]:
    pairs, missing = _paired_categories(dataset, request.column, request.second_column)
    if not pairs:
        raise HypothesisAnalysisError("Se requieren observaciones categóricas completas")
    rows = sorted({str(first) for first, _ in pairs})
    columns = sorted({str(second) for _, second in pairs})
    if len(rows) < 2 or len(columns) < 2:
        raise HypothesisAnalysisError("Se requiere una tabla de al menos 2 × 2")
    observed = np.zeros((len(rows), len(columns)), dtype=int)
    row_index = {value: index for index, value in enumerate(rows)}
    column_index = {value: index for index, value in enumerate(columns)}
    for first, second in pairs:
        observed[row_index[str(first)], column_index[str(second)]] += 1
    if request.method == "exact":
        if observed.shape != (2, 2):
            raise HypothesisAnalysisError("Fisher exacta requiere una tabla 2 × 2")
        test = stats.fisher_exact(observed, alternative=_scipy_alternative(request.alternative))
        statistic, p_value, method = test.statistic, test.pvalue, "Fisher exacta"
        expected = stats.contingency.expected_freq(observed)
        df = 1
    else:
        test = stats.chi2_contingency(observed, correction=False)
        statistic, p_value, df, expected = test.statistic, test.pvalue, test.dof, test.expected_freq
        method = "Chi-cuadrado de Pearson"
    residuals = np.divide(
        observed - expected,
        np.sqrt(expected),
        out=np.zeros_like(expected, dtype=float),
        where=expected > 0,
    )
    low_expected = int(np.sum(expected < 5))
    warnings = _missing_warning(missing)
    if low_expected:
        warnings.append(
            f"{low_expected} celdas tienen frecuencia esperada menor que 5; "
            "considere Fisher o reagrupar categorías."
        )
    return _result(
        method,
        statistic,
        p_value,
        df,
        None,
        None,
        None,
        {
            "row_labels": rows,
            "column_labels": columns,
            "observed": observed.tolist(),
            "expected": expected.tolist(),
            "standardized_residuals": residuals.tolist(),
            "cramers_v": math.sqrt(
                float(
                    test.statistic
                    if request.method != "exact"
                    else stats.chi2_contingency(observed, correction=False).statistic
                )
                / (observed.sum() * min(observed.shape[0] - 1, observed.shape[1] - 1))
            ),
        },
    ), warnings


def _poisson_goodness_of_fit(
    dataset: Dataset, request: HypothesisAnalysisRequest
) -> tuple[dict[str, Any], list[str]]:
    values, missing = _numeric_column(dataset, request.column)
    _require_counts(values)
    integer_values = values.astype(int)
    rate = float(np.mean(integer_values))
    frequencies = Counter(integer_values.tolist())
    maximum = max(frequencies)
    observed = np.array(
        [frequencies.get(value, 0) for value in range(maximum + 1)] + [0], dtype=float
    )
    probabilities = np.array(
        [stats.poisson.pmf(value, rate) for value in range(maximum + 1)]
        + [stats.poisson.sf(maximum, rate)]
    )
    expected = probabilities * len(integer_values)
    observed, expected, labels = _pool_expected(
        observed, expected, [str(value) for value in range(maximum + 1)] + [f"> {maximum}"]
    )
    if len(observed) < 3:
        raise HypothesisAnalysisError(
            "No hay suficientes categorías para evaluar el ajuste Poisson"
        )
    statistic = float(np.sum((observed - expected) ** 2 / expected))
    df = len(observed) - 2
    p_value = float(stats.chi2.sf(statistic, df))
    return _result(
        "Chi-cuadrado de bondad de ajuste Poisson",
        statistic,
        p_value,
        df,
        rate,
        None,
        None,
        {
            "n": len(values),
            "estimated_rate": rate,
            "categories": labels,
            "observed": observed.tolist(),
            "expected": expected.tolist(),
        },
    ), _missing_warning(missing)


def _one_sample_sign(
    dataset: Dataset, request: HypothesisAnalysisRequest
) -> tuple[dict[str, Any], list[str]]:
    values, missing = _numeric_column(dataset, request.column)
    _require_n(values, 1)
    differences = values - request.hypothesized_value
    positives = int(np.sum(differences > 0))
    negatives = int(np.sum(differences < 0))
    n = positives + negatives
    if n == 0:
        raise HypothesisAnalysisError("Todos los valores son iguales a la mediana hipotética")
    test = stats.binomtest(positives, n, 0.5, alternative=_scipy_alternative(request.alternative))
    return _result(
        "Signos exacta",
        positives,
        test.pvalue,
        None,
        float(np.median(values)),
        request.hypothesized_value,
        None,
        {
            "n": len(values),
            "n_for_test": n,
            "positive": positives,
            "negative": negatives,
            "ties": len(values) - n,
            "median": float(np.median(values)),
        },
    ), _missing_warning(missing)


def _one_sample_wilcoxon(
    dataset: Dataset, request: HypothesisAnalysisRequest
) -> tuple[dict[str, Any], list[str]]:
    values, missing = _numeric_column(dataset, request.column)
    _require_n(values, 2)
    differences = values - request.hypothesized_value
    if np.all(differences == 0):
        raise HypothesisAnalysisError("Wilcoxon requiere al menos una diferencia distinta de cero")
    test = stats.wilcoxon(
        differences, alternative=_scipy_alternative(request.alternative), method="auto"
    )
    return _result(
        "Wilcoxon de rangos con signo",
        test.statistic,
        test.pvalue,
        None,
        float(np.median(values)),
        request.hypothesized_value,
        None,
        {
            "n": len(values),
            "n_for_test": int(np.sum(differences != 0)),
            "median": float(np.median(values)),
        },
    ), _missing_warning(missing)


def _mann_whitney(
    dataset: Dataset, request: HypothesisAnalysisRequest
) -> tuple[dict[str, Any], list[str]]:
    groups, missing = _numeric_groups(dataset, request, exactly=2)
    (name1, first), (name2, second) = groups
    _require_n(first, 1)
    _require_n(second, 1)
    test = stats.mannwhitneyu(
        first, second, alternative=_scipy_alternative(request.alternative), method="auto"
    )
    effect = 2 * float(test.statistic) / (len(first) * len(second)) - 1
    return _result(
        "Mann–Whitney",
        test.statistic,
        test.pvalue,
        None,
        float(np.median(first) - np.median(second)),
        0,
        None,
        {
            "groups": [_sample_summary(name1, first), _sample_summary(name2, second)],
            "rank_biserial": effect,
        },
    ), _missing_warning(missing)


def _kruskal_wallis(
    dataset: Dataset, request: HypothesisAnalysisRequest
) -> tuple[dict[str, Any], list[str]]:
    groups, missing = _numeric_groups(dataset, request, minimum=2)
    arrays = [values for _, values in groups]
    for values in arrays:
        _require_n(values, 1)
    test = stats.kruskal(*arrays)
    n = sum(map(len, arrays))
    effect = max(0.0, (float(test.statistic) - len(arrays) + 1) / (n - len(arrays)))
    return _result(
        "Kruskal–Wallis",
        test.statistic,
        test.pvalue,
        len(arrays) - 1,
        None,
        None,
        None,
        {
            "groups": [_sample_summary(name, values) for name, values in groups],
            "epsilon_squared": effect,
        },
    ), _missing_warning(missing)


def _mood_median(
    dataset: Dataset, request: HypothesisAnalysisRequest
) -> tuple[dict[str, Any], list[str]]:
    groups, missing = _numeric_groups(dataset, request, minimum=2)
    arrays = [values for _, values in groups]
    for values in arrays:
        _require_n(values, 1)
    test = stats.median_test(*arrays, ties="below", correction=False)
    statistic, p_value, grand_median, table = test
    return _result(
        "Mediana de Mood",
        statistic,
        p_value,
        len(arrays) - 1,
        grand_median,
        None,
        None,
        {
            "groups": [_sample_summary(name, values) for name, values in groups],
            "grand_median": grand_median,
            "contingency_table": table.tolist(),
        },
    ), _missing_warning(missing)


def _friedman(
    dataset: Dataset, request: HypothesisAnalysisRequest
) -> tuple[dict[str, Any], list[str]]:
    rows = dataset.rows_data
    treatment_key, block_key, value_key = request.group_by, request.block_by, request.column
    treatments = sorted(
        {str(row.get(treatment_key)) for row in rows if row.get(treatment_key) is not None}
    )
    blocks = sorted({str(row.get(block_key)) for row in rows if row.get(block_key) is not None})
    matrix: dict[tuple[str, str], float] = {}
    missing = 0
    for row in rows:
        treatment, block = row.get(treatment_key), row.get(block_key)
        value = _finite_number(row.get(value_key))
        if treatment is None or block is None or value is None:
            missing += 1
            continue
        key = (str(treatment), str(block))
        if key in matrix:
            raise HypothesisAnalysisError(
                "Friedman requiere una sola observación por tratamiento y bloque"
            )
        matrix[key] = value
    complete_blocks = [
        block for block in blocks if all((treatment, block) in matrix for treatment in treatments)
    ]
    if len(treatments) < 3 or len(complete_blocks) < 2:
        raise HypothesisAnalysisError(
            "Friedman requiere al menos tres tratamientos y dos bloques completos"
        )
    arrays = [
        np.array([matrix[(treatment, block)] for block in complete_blocks])
        for treatment in treatments
    ]
    test = stats.friedmanchisquare(*arrays)
    ranks_by_block = np.apply_along_axis(stats.rankdata, 0, np.vstack(arrays))
    return _result(
        "Friedman",
        test.statistic,
        test.pvalue,
        len(treatments) - 1,
        None,
        None,
        None,
        {
            "treatments": [
                {
                    "name": name,
                    "median": float(np.median(values)),
                    "rank_sum": float(np.sum(ranks_by_block[index])),
                }
                for index, (name, values) in enumerate(zip(treatments, arrays, strict=True))
            ],
            "blocks": len(complete_blocks),
        },
    ), _missing_warning(missing + (len(blocks) - len(complete_blocks)) * len(treatments))


def _runs(dataset: Dataset, request: HypothesisAnalysisRequest) -> tuple[dict[str, Any], list[str]]:
    values, missing = _numeric_column(dataset, request.column)
    _require_n(values, 2)
    cutoff = (
        request.sequence_cutoff if request.sequence_cutoff is not None else float(np.median(values))
    )
    sequence = values >= cutoff
    n1, n2 = int(np.sum(sequence)), int(np.sum(~sequence))
    if n1 == 0 or n2 == 0:
        raise HypothesisAnalysisError(
            "La prueba de corridas requiere valores a ambos lados del punto de corte"
        )
    runs = 1 + int(np.sum(sequence[1:] != sequence[:-1]))
    mean = 1 + 2 * n1 * n2 / (n1 + n2)
    variance = 2 * n1 * n2 * (2 * n1 * n2 - n1 - n2) / (((n1 + n2) ** 2) * (n1 + n2 - 1))
    statistic = (runs - mean) / math.sqrt(variance)
    p_value = _normal_p_value(statistic, request.alternative)
    return _result(
        "Corridas (aproximación normal)",
        statistic,
        p_value,
        None,
        runs,
        mean,
        None,
        {
            "n": len(values),
            "cutoff": cutoff,
            "runs": runs,
            "above_or_equal": n1,
            "below": n2,
            "expected_runs": mean,
        },
    ), _missing_warning(missing)


def _result(
    method: str,
    statistic: Any,
    p_value: Any,
    degrees: Any,
    estimate: Any,
    null: Any,
    interval: dict[str, Any] | None,
    details: dict[str, Any],
) -> dict[str, Any]:
    return {
        "method": method,
        "statistic": _number(statistic),
        "p_value": _number(p_value),
        "degrees_of_freedom": _json_degrees(degrees),
        "estimate": _number(estimate),
        "null_value": _number(null),
        "confidence_interval": _json_value(interval),
        "details": _json_value(details),
    }


def _numeric_column(dataset: Dataset, key: str | None) -> tuple[np.ndarray, int]:
    values = []
    missing = 0
    for row in dataset.rows_data:
        value = _finite_number(row.get(key))
        if value is None:
            missing += 1
        else:
            values.append(value)
    return np.asarray(values, dtype=float), missing


def _categorical_column(dataset: Dataset, key: str | None) -> tuple[list[Any], int]:
    values = []
    missing = 0
    for row in dataset.rows_data:
        value = row.get(key)
        if _is_missing(value):
            missing += 1
        else:
            values.append(value)
    return values, missing


def _paired_numeric(
    dataset: Dataset, first_key: str | None, second_key: str | None
) -> tuple[np.ndarray, np.ndarray, int]:
    first = []
    second = []
    missing = 0
    for row in dataset.rows_data:
        left, right = _finite_number(row.get(first_key)), _finite_number(row.get(second_key))
        if left is None or right is None:
            missing += 1
        else:
            first.append(left)
            second.append(right)
    return np.asarray(first), np.asarray(second), missing


def _paired_categories(
    dataset: Dataset, first_key: str | None, second_key: str | None
) -> tuple[list[tuple[Any, Any]], int]:
    pairs = []
    missing = 0
    for row in dataset.rows_data:
        left, right = row.get(first_key), row.get(second_key)
        if _is_missing(left) or _is_missing(right):
            missing += 1
        else:
            pairs.append((left, right))
    return pairs, missing


def _numeric_groups(
    dataset: Dataset,
    request: HypothesisAnalysisRequest,
    exactly: int | None = None,
    minimum: int | None = None,
) -> tuple[list[tuple[str, np.ndarray]], int]:
    selected = None if request.groups is None else {str(value) for value in request.groups}
    grouped: dict[str, list[float]] = {}
    missing = 0
    for row in dataset.rows_data:
        group = row.get(request.group_by)
        value = _finite_number(row.get(request.column))
        if _is_missing(group) or value is None:
            missing += 1
            continue
        name = str(group)
        if selected is not None and name not in selected:
            continue
        grouped.setdefault(name, []).append(value)
    groups = [(name, np.asarray(values)) for name, values in sorted(grouped.items())]
    if exactly is not None and len(groups) != exactly:
        raise HypothesisAnalysisError(f"La prueba requiere exactamente {exactly} grupos")
    if minimum is not None and len(groups) < minimum:
        raise HypothesisAnalysisError(f"La prueba requiere al menos {minimum} grupos")
    return groups, missing


def _categorical_groups(
    dataset: Dataset, request: HypothesisAnalysisRequest, exactly: int
) -> tuple[list[tuple[str, list[Any]]], int]:
    selected = None if request.groups is None else {str(value) for value in request.groups}
    grouped: dict[str, list[Any]] = {}
    missing = 0
    for row in dataset.rows_data:
        group, value = row.get(request.group_by), row.get(request.column)
        if _is_missing(group) or _is_missing(value):
            missing += 1
            continue
        name = str(group)
        if selected is not None and name not in selected:
            continue
        grouped.setdefault(name, []).append(value)
    groups = sorted(grouped.items())
    if len(groups) != exactly:
        raise HypothesisAnalysisError(f"La prueba requiere exactamente {exactly} grupos")
    return groups, missing


def _require_n(values: np.ndarray, minimum: int) -> None:
    if len(values) < minimum:
        raise HypothesisAnalysisError(f"Se requieren al menos {minimum} observaciones válidas")


def _require_variable(values: np.ndarray, minimum: int) -> None:
    _require_n(values, minimum)
    if np.ptp(values) == 0:
        raise HypothesisAnalysisError("La prueba requiere variación en los datos")


def _require_counts(values: np.ndarray) -> None:
    _require_n(values, 1)
    if np.any(values < 0) or np.any(values != np.floor(values)):
        raise HypothesisAnalysisError("La prueba Poisson requiere conteos enteros no negativos")


def _finite_number(value: Any) -> float | None:
    if value is None or value == "" or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _is_missing(value: Any) -> bool:
    return value is None or value == ""


def _same_category(left: Any, right: Any) -> bool:
    return left == right or str(left) == str(right)


def _sample_summary(name: str, values: np.ndarray) -> dict[str, Any]:
    return {
        "name": name,
        "n": len(values),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std_dev": float(np.std(values, ddof=1)) if len(values) > 1 else None,
        "variance": float(np.var(values, ddof=1)) if len(values) > 1 else None,
    }


def _scipy_alternative(value: str) -> str:
    return "two-sided" if value == "two_sided" else value


def _normal_p_value(statistic: float, alternative: str) -> float:
    if alternative == "less":
        return float(stats.norm.cdf(statistic))
    if alternative == "greater":
        return float(stats.norm.sf(statistic))
    return float(2 * stats.norm.sf(abs(statistic)))


def _tail_p_value(lower: float, upper: float, alternative: str) -> float:
    if alternative == "less":
        return float(lower)
    if alternative == "greater":
        return float(upper)
    return float(min(1.0, 2 * min(lower, upper)))


def _normal_interval(
    estimate: float, se: float, request: HypothesisAnalysisRequest
) -> dict[str, Any]:
    critical = stats.norm.ppf(
        (1 + request.confidence_level) / 2
        if request.alternative == "two_sided"
        else request.confidence_level
    )
    return _bounded_interval(estimate, se, critical, request)


def _t_interval(
    estimate: float, se: float, df: float, request: HypothesisAnalysisRequest
) -> dict[str, Any]:
    critical = stats.t.ppf(
        (1 + request.confidence_level) / 2
        if request.alternative == "two_sided"
        else request.confidence_level,
        df,
    )
    return _bounded_interval(estimate, se, critical, request)


def _bounded_interval(
    estimate: float, se: float, critical: float, request: HypothesisAnalysisRequest
) -> dict[str, Any]:
    lower = None if request.alternative == "less" else estimate - critical * se
    upper = None if request.alternative == "greater" else estimate + critical * se
    return {"level": request.confidence_level, "lower": _number(lower), "upper": _number(upper)}


def _clopper_pearson(events: int, n: int, level: float, alternative: str) -> dict[str, Any]:
    alpha = 1 - level
    if alternative == "two_sided":
        alpha /= 2
    lower = 0.0 if events == 0 else float(stats.beta.ppf(alpha, events, n - events + 1))
    upper = 1.0 if events == n else float(stats.beta.ppf(1 - alpha, events + 1, n - events))
    if alternative == "less":
        lower = None
    if alternative == "greater":
        upper = None
    return {"level": level, "lower": lower, "upper": upper}


def _poisson_p_value(observed: int, expected: float, alternative: str) -> float:
    if alternative == "less":
        return float(stats.poisson.cdf(observed, expected))
    if alternative == "greater":
        return float(stats.poisson.sf(observed - 1, expected))
    probability = stats.poisson.pmf(observed, expected)
    maximum = int(max(observed, expected + 12 * math.sqrt(expected + 1)))
    values = np.arange(0, maximum + 1)
    probabilities = stats.poisson.pmf(values, expected)
    return float(min(1.0, probabilities[probabilities <= probability + 1e-15].sum()))


def _poisson_interval(
    total: int, exposure: float, level: float, alternative: str
) -> dict[str, Any]:
    alpha = 1 - level
    if alternative == "two_sided":
        alpha /= 2
    lower = 0.0 if total == 0 else float(stats.chi2.ppf(alpha, 2 * total) / (2 * exposure))
    upper = float(stats.chi2.ppf(1 - alpha, 2 * (total + 1)) / (2 * exposure))
    if alternative == "less":
        lower = None
    if alternative == "greater":
        upper = None
    return {"level": level, "lower": lower, "upper": upper}


def _correlation_interval(coefficient: float, n: int, level: float) -> dict[str, Any] | None:
    if n <= 3 or abs(coefficient) >= 1:
        return None
    z = np.arctanh(coefficient)
    margin = stats.norm.ppf((1 + level) / 2) / math.sqrt(n - 3)
    return {
        "level": level,
        "lower": float(np.tanh(z - margin)),
        "upper": float(np.tanh(z + margin)),
    }


def _anderson_darling(values: np.ndarray) -> tuple[float, float]:
    count = len(values)
    standardized = (np.sort(values) - np.mean(values)) / np.std(values, ddof=1)
    cumulative = np.clip(stats.norm.cdf(standardized), 1e-15, 1 - 1e-15)
    weights = 2 * np.arange(1, count + 1) - 1
    statistic = float(
        -count - np.sum(weights * (np.log(cumulative) + np.log1p(-cumulative[::-1]))) / count
    )
    adjusted = statistic * (1 + 0.75 / count + 2.25 / count**2)
    if adjusted >= 0.6:
        p = math.exp(1.2937 - 5.709 * adjusted + 0.0186 * adjusted**2)
    elif adjusted >= 0.34:
        p = math.exp(0.9177 - 4.279 * adjusted - 1.38 * adjusted**2)
    elif adjusted > 0.2:
        p = 1 - math.exp(-8.318 + 42.796 * adjusted - 59.938 * adjusted**2)
    else:
        p = 1 - math.exp(-13.436 + 101.14 * adjusted - 223.73 * adjusted**2)
    return adjusted, min(max(p, 0.0), 1.0)


def _ryan_joiner(values: np.ndarray) -> tuple[float, float]:
    ordered = np.sort(values)
    n = len(values)
    scores = stats.norm.ppf((np.arange(1, n + 1) - 0.375) / (n + 0.25))
    statistic = float(np.corrcoef(ordered, scores)[0, 1])
    rng = np.random.default_rng(15012007 + n)
    simulations = np.sort(rng.normal(size=(2500, n)), axis=1)
    simulated = np.array([np.corrcoef(row, scores)[0, 1] for row in simulations])
    p_value = (np.sum(simulated <= statistic) + 1) / (len(simulated) + 1)
    return statistic, float(p_value)


def _pool_expected(
    observed: np.ndarray, expected: np.ndarray, labels: list[str]
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    observed_list = observed.tolist()
    expected_list = expected.tolist()
    label_list = labels[:]
    while len(expected_list) > 2 and min(expected_list) < 5:
        index = expected_list.index(min(expected_list))
        neighbor = 1 if index == 0 else index - 1
        observed_list[neighbor] += observed_list[index]
        expected_list[neighbor] += expected_list[index]
        label_list[neighbor] = f"{label_list[neighbor]}, {label_list[index]}"
        for collection in (observed_list, expected_list, label_list):
            collection.pop(index)
    return np.asarray(observed_list), np.asarray(expected_list), label_list


def _hypotheses(request: HypothesisAnalysisRequest) -> dict[str, str]:
    relation = {"two_sided": "≠", "less": "<", "greater": ">"}[request.alternative]
    labels = {
        "one_sample_z": "μ",
        "one_sample_t": "μ",
        "paired_t": "μd",
        "one_proportion": "p",
        "one_sample_poisson_rate": "λ",
        "one_variance": "σ²",
        "correlation": "ρ",
        "one_sample_sign": "η",
        "one_sample_wilcoxon": "η",
    }
    parameter = labels.get(request.test, "parámetro")
    if request.test in {"normality", "poisson_goodness_of_fit", "chi_square_independence", "runs"}:
        text = {
            "normality": (
                "Los datos siguen una distribución normal",
                "Los datos no siguen una distribución normal",
            ),
            "poisson_goodness_of_fit": (
                "Los datos siguen una distribución Poisson",
                "Los datos no siguen una distribución Poisson",
            ),
            "chi_square_independence": (
                "Las variables son independientes",
                "Las variables están asociadas",
            ),
            "runs": ("La secuencia es aleatoria", "La secuencia no es aleatoria"),
        }[request.test]
        return {"null": text[0], "alternative": text[1]}
    if request.test in {"kruskal_wallis", "mood_median", "friedman"}:
        return {
            "null": "Todos los grupos tienen el mismo centro",
            "alternative": "Al menos un grupo difiere",
        }
    return {
        "null": f"H₀: {parameter} = {request.hypothesized_value:g}",
        "alternative": f"H₁: {parameter} {relation} {request.hypothesized_value:g}",
    }


def _interpretation(decision: str | None, test: str, alpha: float) -> str:
    if decision is None:
        return "La prueba no produjo una decisión basada en p-value."
    subject = {
        "normality": "el ajuste normal",
        "poisson_goodness_of_fit": "el ajuste Poisson",
        "chi_square_independence": "la independencia",
        "runs": "la aleatoriedad",
    }.get(test, "la hipótesis nula")
    if decision == "reject":
        return (
            f"Con α = {alpha:g}, se rechaza {subject}; la muestra aporta evidencia "
            "estadísticamente significativa."
        )
    return f"Con α = {alpha:g}, no se rechaza {subject}; la evidencia disponible no es suficiente."


def _missing_warning(count: int) -> list[str]:
    return (
        []
        if count == 0
        else [f"Se excluyeron {count} filas con valores faltantes, no numéricos o no finitos."]
    )


def _number(value: Any) -> float | int | None:
    if value is None:
        return None
    if isinstance(value, (np.integer, int)):
        return int(value)
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _json_degrees(value: Any) -> Any:
    if isinstance(value, list):
        return [_number(item) for item in value]
    return _number(value)


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_json_value(item) for item in value]
    if isinstance(value, (np.floating, float, np.integer, int)):
        return _number(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value
