from typing import Any


def _parameter(
    key: str,
    label: str,
    default: float | None,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    step: float | None = None,
    integer: bool = False,
    required: bool = True,
    kind: str = "number",
) -> dict[str, Any]:
    return {
        "key": key,
        "label": label,
        "default": default,
        "minimum": minimum,
        "maximum": maximum,
        "step": step,
        "integer": integer,
        "required": required,
        "kind": kind,
    }


PROBABILITY_SPECS: dict[str, dict[str, Any]] = {
    "binomial": {
        "key": "binomial",
        "label": "Binomial",
        "kind": "discrete",
        "category": "Frecuentes",
        "description": "Éxitos en un número fijo de ensayos independientes",
        "parameters": [
            _parameter(
                "trials", "Ensayos (n)", 10, minimum=1, maximum=10_000, step=1, integer=True
            ),
            _parameter(
                "success_probability",
                "Probabilidad de éxito (p)",
                0.3,
                minimum=0,
                maximum=1,
                step=0.01,
            ),
        ],
    },
    "poisson": {
        "key": "poisson",
        "label": "Poisson",
        "kind": "discrete",
        "category": "Frecuentes",
        "description": "Eventos contados dentro de un intervalo fijo",
        "parameters": [
            _parameter("rate", "Tasa media (λ)", 4, minimum=0.000001, maximum=1_000, step=0.1)
        ],
    },
    "normal": {
        "key": "normal",
        "label": "Normal",
        "kind": "continuous",
        "category": "Frecuentes",
        "description": "Mediciones continuas alrededor de una media",
        "parameters": [
            _parameter("mean", "Media (μ)", 100, step=0.1),
            _parameter(
                "std_dev",
                "Desviación estándar (σ)",
                15,
                minimum=0.000001,
                maximum=1_000_000,
                step=0.1,
            ),
        ],
    },
    "geometric": {
        "key": "geometric",
        "label": "Geométrica",
        "kind": "discrete",
        "category": "Discretas",
        "description": "Ensayos necesarios hasta obtener el primer éxito",
        "parameters": [
            _parameter(
                "success_probability",
                "Probabilidad de éxito (p)",
                0.25,
                minimum=0.000001,
                maximum=1,
                step=0.01,
            )
        ],
    },
    "negative_binomial": {
        "key": "negative_binomial",
        "label": "Binomial negativa",
        "kind": "discrete",
        "category": "Discretas",
        "description": "Fracasos antes de alcanzar un número de éxitos",
        "parameters": [
            _parameter(
                "required_successes",
                "Éxitos requeridos (r)",
                3,
                minimum=1,
                maximum=10_000,
                step=1,
                integer=True,
            ),
            _parameter(
                "success_probability",
                "Probabilidad de éxito (p)",
                0.4,
                minimum=0.000001,
                maximum=1,
                step=0.01,
            ),
        ],
    },
    "hypergeometric": {
        "key": "hypergeometric",
        "label": "Hipergeométrica",
        "kind": "discrete",
        "category": "Discretas",
        "description": "Éxitos en una muestra tomada sin reemplazo",
        "parameters": [
            _parameter(
                "population_size",
                "Tamaño de población (N)",
                50,
                minimum=1,
                maximum=1_000_000,
                step=1,
                integer=True,
            ),
            _parameter(
                "population_successes",
                "Éxitos en la población (K)",
                10,
                minimum=0,
                maximum=1_000_000,
                step=1,
                integer=True,
            ),
            _parameter(
                "draws",
                "Tamaño de muestra (n)",
                5,
                minimum=1,
                maximum=1_000_000,
                step=1,
                integer=True,
            ),
        ],
    },
    "exponential": {
        "key": "exponential",
        "label": "Exponencial",
        "kind": "continuous",
        "category": "Continuas",
        "description": "Tiempo entre eventos con una tasa constante",
        "parameters": [
            _parameter("scale", "Escala o media", 5, minimum=0.000001, maximum=1_000_000, step=0.1),
            _parameter("threshold", "Umbral", 0, step=0.1, required=False),
        ],
    },
    "student_t": {
        "key": "student_t",
        "label": "t de Student",
        "kind": "continuous",
        "category": "Inferencia",
        "description": "Distribución para medias con varianza poblacional desconocida",
        "parameters": [
            _parameter(
                "degrees_freedom",
                "Grados de libertad",
                10,
                minimum=0.000001,
                maximum=1_000_000,
                step=1,
            )
        ],
    },
    "chi_square": {
        "key": "chi_square",
        "label": "Chi-cuadrada",
        "kind": "continuous",
        "category": "Inferencia",
        "description": "Distribución para varianzas y pruebas de independencia",
        "parameters": [
            _parameter(
                "degrees_freedom",
                "Grados de libertad",
                5,
                minimum=0.000001,
                maximum=1_000_000,
                step=1,
            ),
            _parameter(
                "noncentrality",
                "No centralidad",
                0,
                minimum=0,
                maximum=1_000_000,
                step=0.1,
                required=False,
            ),
        ],
    },
    "f": {
        "key": "f",
        "label": "F",
        "kind": "continuous",
        "category": "Inferencia",
        "description": "Razón de varianzas utilizada en ANOVA",
        "parameters": [
            _parameter(
                "numerator_df", "GL del numerador", 5, minimum=0.000001, maximum=1_000_000, step=1
            ),
            _parameter(
                "denominator_df",
                "GL del denominador",
                10,
                minimum=0.000001,
                maximum=1_000_000,
                step=1,
            ),
            _parameter(
                "noncentrality",
                "No centralidad",
                0,
                minimum=0,
                maximum=1_000_000,
                step=0.1,
                required=False,
            ),
        ],
    },
    "uniform": {
        "key": "uniform",
        "label": "Uniforme",
        "kind": "continuous",
        "category": "Continuas",
        "description": "Todos los valores del intervalo tienen la misma densidad",
        "parameters": [
            _parameter("support_min", "Límite inferior", 0, step=0.1),
            _parameter("support_max", "Límite superior", 10, step=0.1),
        ],
    },
    "integer_uniform": {
        "key": "integer_uniform",
        "label": "Uniforme entera",
        "kind": "discrete",
        "category": "Discretas",
        "description": "Enteros equiprobables entre un mínimo y un máximo",
        "parameters": [
            _parameter("integer_min", "Entero mínimo", 1, step=1, integer=True),
            _parameter("integer_max", "Entero máximo", 6, step=1, integer=True),
        ],
    },
    "custom_discrete": {
        "key": "custom_discrete",
        "label": "Discreta personalizada",
        "kind": "discrete",
        "category": "Discretas",
        "description": "Valores y probabilidades definidos en dos columnas de la hoja",
        "parameters": [
            _parameter("custom_values", "Columna de valores", None, kind="column"),
            _parameter("custom_probabilities", "Columna de probabilidades", None, kind="column"),
        ],
    },
    "beta": {
        "key": "beta",
        "label": "Beta",
        "kind": "continuous",
        "category": "Continuas",
        "description": "Proporciones continuas acotadas entre 0 y 1",
        "parameters": [
            _parameter("shape_alpha", "Forma α", 2, minimum=0.000001, step=0.1),
            _parameter("shape_beta", "Forma β", 2, minimum=0.000001, step=0.1),
        ],
    },
    "cauchy": {
        "key": "cauchy",
        "label": "Cauchy",
        "kind": "continuous",
        "category": "Robustas",
        "description": "Distribución simétrica con colas muy pesadas",
        "parameters": [
            _parameter("location", "Ubicación", 0, step=0.1),
            _parameter("scale", "Escala", 1, minimum=0.000001, step=0.1),
        ],
    },
    "gamma": {
        "key": "gamma",
        "label": "Gamma",
        "kind": "continuous",
        "category": "Modelado y confiabilidad",
        "description": "Tiempos y magnitudes positivas con asimetría",
        "parameters": [
            _parameter("shape", "Forma", 2, minimum=0.000001, step=0.1),
            _parameter("scale", "Escala", 3, minimum=0.000001, step=0.1),
            _parameter("threshold", "Umbral", 0, step=0.1, required=False),
        ],
    },
    "laplace": {
        "key": "laplace",
        "label": "Laplace",
        "kind": "continuous",
        "category": "Robustas",
        "description": "Distribución simétrica con pico central y colas pesadas",
        "parameters": [
            _parameter("location", "Ubicación", 0, step=0.1),
            _parameter("scale", "Escala", 1, minimum=0.000001, step=0.1),
        ],
    },
    "logistic": {
        "key": "logistic",
        "label": "Logística",
        "kind": "continuous",
        "category": "Robustas",
        "description": "Curva simétrica similar a la normal con colas más pesadas",
        "parameters": [
            _parameter("location", "Ubicación", 0, step=0.1),
            _parameter("scale", "Escala", 1, minimum=0.000001, step=0.1),
        ],
    },
    "loglogistic": {
        "key": "loglogistic",
        "label": "Loglogística",
        "kind": "continuous",
        "category": "Modelado y confiabilidad",
        "description": "Tiempos positivos con colas más pesadas que Weibull",
        "parameters": [
            _parameter("location", "Ubicación logarítmica", 0, step=0.1),
            _parameter("scale", "Escala logarítmica", 1, minimum=0.000001, step=0.1),
            _parameter("threshold", "Umbral", 0, step=0.1, required=False),
        ],
    },
    "lognormal": {
        "key": "lognormal",
        "label": "Lognormal",
        "kind": "continuous",
        "category": "Modelado y confiabilidad",
        "description": "Variable positiva cuyo logaritmo sigue una distribución normal",
        "parameters": [
            _parameter("location", "Media logarítmica (μ)", 0, step=0.1),
            _parameter("scale", "Desviación logarítmica (σ)", 1, minimum=0.000001, step=0.1),
            _parameter("threshold", "Umbral", 0, step=0.1, required=False),
        ],
    },
    "triangular": {
        "key": "triangular",
        "label": "Triangular",
        "kind": "continuous",
        "category": "Continuas",
        "description": "Estimaciones acotadas mediante mínimo, moda y máximo",
        "parameters": [
            _parameter("support_min", "Límite inferior", 0, step=0.1),
            _parameter("mode_value", "Moda", 5, step=0.1),
            _parameter("support_max", "Límite superior", 10, step=0.1),
        ],
    },
    "weibull": {
        "key": "weibull",
        "label": "Weibull",
        "kind": "continuous",
        "category": "Modelado y confiabilidad",
        "description": "Tiempos de falla con tasa de riesgo flexible",
        "parameters": [
            _parameter("shape", "Forma", 2, minimum=0.000001, step=0.1),
            _parameter("scale", "Escala", 100, minimum=0.000001, step=0.1),
            _parameter("threshold", "Umbral", 0, step=0.1, required=False),
        ],
    },
    "extreme_min": {
        "key": "extreme_min",
        "label": "Valor extremo mínimo",
        "kind": "continuous",
        "category": "Valores extremos",
        "description": "Modela mínimos observados en bloques o periodos",
        "parameters": [
            _parameter("location", "Ubicación", 0, step=0.1),
            _parameter("scale", "Escala", 1, minimum=0.000001, step=0.1),
        ],
    },
    "extreme_max": {
        "key": "extreme_max",
        "label": "Valor extremo máximo",
        "kind": "continuous",
        "category": "Valores extremos",
        "description": "Modela máximos observados en bloques o periodos",
        "parameters": [
            _parameter("location", "Ubicación", 0, step=0.1),
            _parameter("scale", "Escala", 1, minimum=0.000001, step=0.1),
        ],
    },
}


def probability_catalog() -> list[dict[str, Any]]:
    return [dict(spec) for spec in PROBABILITY_SPECS.values()]
