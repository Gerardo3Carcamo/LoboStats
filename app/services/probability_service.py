import math
from typing import Any

import numpy as np
from scipy import stats

from app.schemas.analysis import ProbabilityAnalysisRequest
from app.services.probability_catalog import PROBABILITY_SPECS

ENGINE_VERSION = "probability-4.0"


class ProbabilityAnalysisError(ValueError):
    pass


def analyze_probability(
    request: ProbabilityAnalysisRequest,
) -> tuple[dict[str, Any], list[str]]:
    distribution = _distribution(request)
    mean, variance = distribution.stats(moments="mv")
    mean_value = _finite_or_none(mean)
    variance_value = _finite_or_none(variance)
    std_dev = math.sqrt(variance_value) if variance_value is not None else None
    warnings: list[str] = []

    if request.distribution == "binomial" and request.success_probability in {0, 1}:
        warnings.append("La probabilidad de éxito produce un modelo determinista sin variación")

    if request.mode == "event":
        result_value = min(1.0, max(0.0, _event_probability(distribution, request)))
        calculations = [
            {
                "index": 0,
                "input": request.lower,
                "result": result_value,
                "status": "valid",
                "error": None,
            }
        ]
        primary_calculation = calculations[0]
        highlight = float(request.lower)
        if result_value < 0.001:
            warnings.append("El evento tiene una probabilidad menor a 0.1 % bajo este modelo")
    else:
        calculations = _calculate_inputs(distribution, request)
        valid = [row for row in calculations if row["status"] == "valid"]
        if not valid:
            raise ProbabilityAnalysisError("Ningún valor de entrada es válido para este cálculo")
        result_value = float(valid[0]["result"])
        primary_calculation = valid[0]
        highlight = (
            result_value if request.mode == "inverse" else float(primary_calculation["input"])
        )
        invalid_count = len(calculations) - len(valid)
        if invalid_count:
            warnings.append(
                f"{invalid_count} valores no se calcularon; revisa los errores en la tabla"
            )

    points = _chart_points(distribution, request, highlight)
    if not any(point["selected"] for point in points):
        warnings.append("El valor principal queda fuera del tramo central mostrado en la gráfica")

    result: dict[str, Any] = {
        "mode": request.mode,
        "calculation_label": _calculation_label(request),
        "result_value": result_value,
        "calculations": calculations,
        "input": None if request.input is None else request.input.model_dump(),
        "distribution": {
            "key": request.distribution,
            "label": _distribution_label(request.distribution),
            "kind": _spec(request)["kind"],
            "parameters": _parameters(request),
            "mean": mean_value,
            "variance": variance_value,
            "std_dev": std_dev,
        },
        "model_formula": _model_formula(request),
        "calculation_formula": _calculation_formula(request, primary_calculation),
        "interpretation": _interpretation(request, result_value, calculations),
        "points": points,
        "guidance": _guidance(request),
    }

    if request.mode == "event":
        result.update(
            {
                "event": {
                    "operation": request.operation,
                    "label": _event_label(request),
                    "lower": request.lower,
                    "upper": request.upper,
                },
                "probability": result_value,
                "percentage": result_value * 100,
                "complement_probability": 1 - result_value,
                "event_formula": _event_formula(request),
            }
        )

    return result, warnings


def _distribution(request: ProbabilityAnalysisRequest):
    if request.distribution == "binomial":
        return stats.binom(n=request.trials, p=request.success_probability)
    if request.distribution == "poisson":
        return stats.poisson(mu=request.rate)
    if request.distribution == "normal":
        return stats.norm(loc=request.mean, scale=request.std_dev)
    if request.distribution == "geometric":
        return stats.geom(p=request.success_probability)
    if request.distribution == "negative_binomial":
        return stats.nbinom(n=request.required_successes, p=request.success_probability)
    if request.distribution == "hypergeometric":
        return stats.hypergeom(
            M=request.population_size,
            n=request.population_successes,
            N=request.draws,
        )
    if request.distribution == "exponential":
        return stats.expon(loc=request.threshold or 0, scale=request.scale)
    if request.distribution == "student_t":
        return stats.t(df=request.degrees_freedom)
    if request.distribution == "chi_square":
        if request.noncentrality:
            return stats.ncx2(df=request.degrees_freedom, nc=request.noncentrality)
        return stats.chi2(df=request.degrees_freedom)
    if request.distribution == "f":
        if request.noncentrality:
            return stats.ncf(
                dfn=request.numerator_df,
                dfd=request.denominator_df,
                nc=request.noncentrality,
            )
        return stats.f(dfn=request.numerator_df, dfd=request.denominator_df)
    if request.distribution == "uniform":
        return stats.uniform(
            loc=request.support_min,
            scale=float(request.support_max) - float(request.support_min),
        )
    if request.distribution == "integer_uniform":
        return stats.randint(low=request.integer_min, high=int(request.integer_max) + 1)
    if request.distribution == "custom_discrete":
        return stats.rv_discrete(
            name="lobostats_custom",
            values=(request.custom_values, request.custom_probabilities),
        )
    if request.distribution == "beta":
        return stats.beta(a=request.shape_alpha, b=request.shape_beta)
    if request.distribution == "cauchy":
        return stats.cauchy(loc=request.location, scale=request.scale)
    if request.distribution == "gamma":
        return stats.gamma(a=request.shape, loc=request.threshold or 0, scale=request.scale)
    if request.distribution == "laplace":
        return stats.laplace(loc=request.location, scale=request.scale)
    if request.distribution == "logistic":
        return stats.logistic(loc=request.location, scale=request.scale)
    if request.distribution == "loglogistic":
        return stats.fisk(
            c=1 / float(request.scale),
            loc=request.threshold or 0,
            scale=math.exp(float(request.location)),
        )
    if request.distribution == "lognormal":
        return stats.lognorm(
            s=request.scale,
            loc=request.threshold or 0,
            scale=math.exp(float(request.location)),
        )
    if request.distribution == "triangular":
        width = float(request.support_max) - float(request.support_min)
        return stats.triang(
            c=(float(request.mode_value) - float(request.support_min)) / width,
            loc=request.support_min,
            scale=width,
        )
    if request.distribution == "weibull":
        return stats.weibull_min(
            c=request.shape,
            loc=request.threshold or 0,
            scale=request.scale,
        )
    if request.distribution == "extreme_min":
        return stats.gumbel_l(loc=request.location, scale=request.scale)
    if request.distribution == "extreme_max":
        return stats.gumbel_r(loc=request.location, scale=request.scale)
    raise ProbabilityAnalysisError("Distribución de probabilidad no compatible")


def _event_probability(distribution: Any, request: ProbabilityAnalysisRequest) -> float:
    if _spec(request)["kind"] == "continuous":
        if request.operation == "at_most":
            return float(distribution.cdf(request.lower))
        if request.operation == "at_least":
            return float(distribution.sf(request.lower))
        if request.operation == "between" and request.upper is not None:
            return float(distribution.cdf(request.upper) - distribution.cdf(request.lower))
        raise ProbabilityAnalysisError("Operación no válida para una distribución continua")

    lower = (
        float(request.lower) if request.distribution == "custom_discrete" else int(request.lower)
    )
    before_lower = (
        float(np.nextafter(lower, -math.inf))
        if request.distribution == "custom_discrete"
        else lower - 1
    )
    if request.operation == "exact":
        return float(distribution.pmf(lower))
    if request.operation == "at_most":
        return float(distribution.cdf(lower))
    if request.operation == "at_least":
        return float(distribution.sf(before_lower))
    if request.operation == "between" and request.upper is not None:
        upper = (
            float(request.upper)
            if request.distribution == "custom_discrete"
            else int(request.upper)
        )
        return float(distribution.cdf(upper) - distribution.cdf(before_lower))
    raise ProbabilityAnalysisError("Operación de probabilidad no válida")


def _calculate_inputs(
    distribution: Any, request: ProbabilityAnalysisRequest
) -> list[dict[str, Any]]:
    assert request.input is not None
    return [
        _calculate_input(distribution, request, index, value)
        for index, value in enumerate(request.input.values)
    ]


def _calculate_input(
    distribution: Any,
    request: ProbabilityAnalysisRequest,
    index: int,
    value: float | None,
) -> dict[str, Any]:
    if value is None:
        return _invalid_row(index, value, "Valor faltante")
    if (
        request.mode in {"pdf", "cdf"}
        and _spec(request)["kind"] == "discrete"
        and request.distribution != "custom_discrete"
    ):
        if not float(value).is_integer():
            return _invalid_row(index, value, "Se requiere un valor entero")
        if request.distribution != "integer_uniform" and value < 0:
            return _invalid_row(index, value, "Se requiere un conteo no negativo")
    if request.mode == "inverse" and not 0 < value < 1:
        return _invalid_row(index, value, "La probabilidad debe ser mayor que 0 y menor que 1")

    if request.mode == "pdf":
        result = (
            distribution.pdf(value)
            if _spec(request)["kind"] == "continuous"
            else distribution.pmf(value)
        )
    elif request.mode == "cdf":
        result = distribution.cdf(value)
    else:
        result = distribution.ppf(value)
    result_value = float(result)
    if not math.isfinite(result_value):
        return _invalid_row(index, value, "El resultado no es finito para este valor")
    return {
        "index": index,
        "input": value,
        "result": result_value,
        "status": "valid",
        "error": None,
    }


def _invalid_row(index: int, value: float | None, message: str) -> dict[str, Any]:
    return {"index": index, "input": value, "result": None, "status": "invalid", "error": message}


def _chart_points(
    distribution: Any, request: ProbabilityAnalysisRequest, highlight: float
) -> list[dict[str, Any]]:
    if _spec(request)["kind"] == "continuous":
        mean, variance = distribution.stats(moments="mv")
        mean = float(mean) if math.isfinite(float(mean)) else float(distribution.median())
        std_dev = math.sqrt(float(variance)) if math.isfinite(float(variance)) else 1.0
        quantile_lower = float(distribution.ppf(0.001))
        quantile_upper = float(distribution.ppf(0.999))
        base_lower = quantile_lower if math.isfinite(quantile_lower) else mean - 4 * std_dev
        base_upper = quantile_upper if math.isfinite(quantile_upper) else mean + 4 * std_dev
        visible_bounds = [
            value
            for value in [request.lower, request.upper, highlight]
            if value is not None and base_lower - 4 * std_dev <= value <= base_upper + 4 * std_dev
        ]
        chart_lower = min([base_lower, *visible_bounds])
        chart_upper = max([base_upper, *visible_bounds])
        values = np.linspace(chart_lower, chart_upper, 161)
        values = np.unique(np.append(values, visible_bounds))
        return [
            {
                "x": float(value),
                "value": float(distribution.pdf(value)),
                "selected": _selected(float(value), request, highlight),
            }
            for value in values
        ]

    if request.distribution == "custom_discrete":
        return [
            {
                "x": float(value),
                "value": float(probability),
                "selected": _selected(float(value), request, highlight),
            }
            for value, probability in sorted(
                zip(
                    request.custom_values or [],
                    request.custom_probabilities or [],
                    strict=True,
                ),
                key=lambda item: item[0],
            )
        ]

    lower_quantile = float(distribution.ppf(0.0005))
    upper_quantile = float(distribution.ppf(0.9995))
    chart_lower = max(0, int(math.floor(lower_quantile))) if math.isfinite(lower_quantile) else 0
    support_lower, support_upper = distribution.support()
    if math.isfinite(float(support_lower)):
        chart_lower = max(chart_lower, int(math.floor(float(support_lower))))
    chart_upper = int(math.ceil(upper_quantile))
    if math.isfinite(float(support_upper)):
        chart_upper = min(chart_upper, int(math.ceil(float(support_upper))))
    chart_upper = max(chart_lower, chart_upper)
    return [
        {
            "x": value,
            "value": float(distribution.pmf(value)),
            "selected": _selected(value, request, highlight),
        }
        for value in range(chart_lower, chart_upper + 1)
    ]


def _selected(value: float, request: ProbabilityAnalysisRequest, highlight: float) -> bool:
    if request.mode == "pdf":
        return math.isclose(value, highlight, rel_tol=1e-9, abs_tol=1e-9)
    if request.mode in {"cdf", "inverse"}:
        return value <= highlight
    if request.operation == "exact":
        return value == request.lower
    if request.operation == "at_most":
        return value <= request.lower
    if request.operation == "at_least":
        return value >= request.lower
    return request.upper is not None and request.lower <= value <= request.upper


def _parameters(request: ProbabilityAnalysisRequest) -> dict[str, Any]:
    return {
        parameter["key"]: getattr(request, parameter["key"])
        for parameter in _spec(request)["parameters"]
        if getattr(request, parameter["key"]) is not None
    }


def _spec(request: ProbabilityAnalysisRequest) -> dict[str, Any]:
    return PROBABILITY_SPECS[request.distribution]


def _distribution_label(distribution: str) -> str:
    return str(PROBABILITY_SPECS[distribution]["label"])


def _calculation_label(request: ProbabilityAnalysisRequest) -> str:
    if request.mode == "pdf":
        return (
            "Densidad de probabilidad (PDF)"
            if _spec(request)["kind"] == "continuous"
            else "Masa de probabilidad (PMF)"
        )
    if request.mode == "cdf":
        return "Probabilidad acumulada (CDF)"
    if request.mode == "inverse":
        return "Probabilidad acumulada inversa"
    return "Probabilidad del evento"


def _event_label(request: ProbabilityAnalysisRequest) -> str:
    value = _number(float(request.lower))
    if request.operation == "exact":
        return f"Exactamente {value}"
    if request.operation == "at_most":
        return f"Como máximo {value}"
    if request.operation == "at_least":
        return f"Como mínimo {value}"
    return f"Entre {value} y {_number(float(request.upper))}"


def _model_formula(request: ProbabilityAnalysisRequest) -> str:
    if request.distribution == "binomial":
        return f"X ~ Binomial(n={request.trials}, p={_number(float(request.success_probability))})"
    if request.distribution == "poisson":
        return f"X ~ Poisson(λ={_number(float(request.rate))})"
    if request.distribution == "normal":
        return f"X ~ Normal(μ={_number(float(request.mean))}, σ={_number(float(request.std_dev))})"
    parameters = ", ".join(
        f"{parameter['key']}={_parameter_text(getattr(request, parameter['key']))}"
        for parameter in _spec(request)["parameters"]
    )
    return f"X ~ {_distribution_label(request.distribution)}({parameters})"


def _event_formula(request: ProbabilityAnalysisRequest) -> str:
    lower = _number(float(request.lower))
    if request.operation == "exact":
        return f"P(X = {lower})"
    if request.operation == "at_most":
        return f"P(X ≤ {lower})"
    if request.operation == "at_least":
        return f"P(X ≥ {lower})"
    return f"P({lower} ≤ X ≤ {_number(float(request.upper))})"


def _calculation_formula(request: ProbabilityAnalysisRequest, calculation: dict[str, Any]) -> str:
    if request.mode == "event":
        return _event_formula(request)
    input_value = calculation["input"]
    if request.mode == "pdf":
        function = "f" if _spec(request)["kind"] == "continuous" else "P"
        return f"{function}(X = {_number(float(input_value))})"
    if request.mode == "cdf":
        return f"F({_number(float(input_value))}) = P(X ≤ {_number(float(input_value))})"
    return f"F⁻¹({_number(float(input_value))})"


def _interpretation(
    request: ProbabilityAnalysisRequest,
    result_value: float,
    calculations: list[dict[str, Any]],
) -> str:
    if request.mode != "event":
        valid_count = sum(row["status"] == "valid" for row in calculations)
        if len(calculations) > 1:
            return (
                f"Se calcularon {valid_count} de {len(calculations)} valores con el modelo "
                f"{_distribution_label(request.distribution)}."
            )
        input_value = float(calculations[0]["input"])
        if request.mode == "pdf" and _spec(request)["kind"] == "continuous":
            return (
                f"La densidad en X = {_number(input_value)} es {_number(result_value)}; "
                "la altura de la curva no es una probabilidad puntual."
            )
        if request.mode == "pdf":
            return f"La probabilidad de X = {_number(input_value)} es {result_value * 100:.2f} %."
        if request.mode == "cdf":
            return (
                f"La probabilidad acumulada hasta {_number(input_value)} es "
                f"{result_value * 100:.2f} %."
            )
        return (
            f"La probabilidad acumulada {_number(input_value)} corresponde al cuantil "
            f"{_number(result_value)}."
        )

    event = _event_label(request).lower()
    percentage = f"{result_value * 100:.2f} %"
    context = f"bajo el modelo {_distribution_label(request.distribution)} configurado"
    return f"La probabilidad de observar {event} es {percentage} {context}."


def _guidance(request: ProbabilityAnalysisRequest) -> list[str]:
    assumptions = {
        "binomial": ["Ensayos independientes", "Probabilidad de éxito constante"],
        "poisson": ["Eventos independientes", "Tasa constante por intervalo"],
        "normal": ["Variable continua", "Forma aproximadamente simétrica"],
        "geometric": ["Ensayos independientes", "Se cuenta hasta el primer éxito"],
        "negative_binomial": ["Ensayos independientes", "Se cuentan fracasos antes de r éxitos"],
        "hypergeometric": ["Muestreo sin reemplazo", "Población finita"],
        "exponential": ["Tasa de ocurrencia constante", "Propiedad sin memoria"],
        "student_t": [
            "Variable continua y simétrica",
            "Las colas dependen de los grados de libertad",
        ],
        "chi_square": ["Soporte no negativo", "La forma depende de los grados de libertad"],
        "f": ["Soporte no negativo", "Modela razones de varianzas"],
        "uniform": [
            "Densidad constante dentro del intervalo",
            "Probabilidad cero fuera de los límites",
        ],
        "integer_uniform": [
            "Cada entero del intervalo tiene la misma probabilidad",
            "Los límites son inclusivos",
        ],
        "custom_discrete": ["Las probabilidades deben sumar 1", "Cada valor debe ser único"],
        "beta": ["Soporte entre 0 y 1", "Las formas controlan asimetría y concentración"],
        "cauchy": [
            "Media y varianza no están definidas",
            "Las colas son más pesadas que en la normal",
        ],
        "gamma": ["Soporte por encima del umbral", "Forma y escala deben ser positivas"],
        "laplace": ["Distribución simétrica", "Presenta un pico central más agudo que la normal"],
        "logistic": ["Distribución simétrica", "Sus colas son más pesadas que las de la normal"],
        "loglogistic": [
            "Soporte por encima del umbral",
            "Adecuada para tiempos positivos con colas pesadas",
        ],
        "lognormal": [
            "Los valores sobre el umbral son positivos",
            "El logaritmo sigue una distribución normal",
        ],
        "triangular": ["Requiere límites finitos", "La moda debe quedar dentro del intervalo"],
        "weibull": [
            "Soporte por encima del umbral",
            "La forma controla el comportamiento de la tasa de falla",
        ],
        "extreme_min": ["Modela mínimos por bloque", "Ubicación y escala describen el extremo"],
        "extreme_max": ["Modela máximos por bloque", "Ubicación y escala describen el extremo"],
    }
    return assumptions[request.distribution]


def _number(value: float) -> str:
    return f"{value:.6g}"


def _parameter_text(value: Any) -> str:
    if isinstance(value, list):
        return f"{len(value)} valores"
    return _number(float(value or 0))


def _finite_or_none(value: Any) -> float | None:
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None
