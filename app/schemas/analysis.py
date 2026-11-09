import math
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.services.probability_catalog import PROBABILITY_SPECS


class DescriptiveAnalysisRequest(BaseModel):
    columns: list[str] = Field(min_length=1, max_length=100)
    group_by: str | None = None
    confidence_level: float = Field(default=0.95, gt=0.5, lt=1.0)
    percentiles: list[float] = Field(default_factory=lambda: [0.25, 0.5, 0.75], max_length=20)
    histogram_bins: int = Field(default=10, ge=3, le=100)
    frequency_limit: int = Field(default=25, ge=1, le=100)

    @field_validator("columns")
    @classmethod
    def unique_columns(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("Cada columna se debe seleccionar una sola vez")
        return value

    @field_validator("percentiles")
    @classmethod
    def valid_percentiles(cls, value: list[float]) -> list[float]:
        if any(item <= 0 or item >= 1 for item in value):
            raise ValueError("Los percentiles deben estar entre 0 y 1")
        return sorted(set(value))

    @model_validator(mode="after")
    def group_is_not_analyzed_column(self) -> "DescriptiveAnalysisRequest":
        if self.group_by and self.group_by in self.columns:
            raise ValueError("La variable de agrupación no puede ser una variable analizada")
        return self


class ProbabilityAnalysisRequest(BaseModel):
    distribution: Literal[
        "binomial",
        "poisson",
        "normal",
        "geometric",
        "negative_binomial",
        "hypergeometric",
        "exponential",
        "student_t",
        "chi_square",
        "f",
        "uniform",
        "integer_uniform",
        "custom_discrete",
        "beta",
        "cauchy",
        "gamma",
        "laplace",
        "logistic",
        "loglogistic",
        "lognormal",
        "triangular",
        "weibull",
        "extreme_min",
        "extreme_max",
    ]
    mode: Literal["pdf", "cdf", "inverse", "event"] = "event"
    operation: Literal["exact", "at_most", "at_least", "between"] | None = None
    input: "ProbabilityInput | None" = None
    trials: int | None = Field(default=None, ge=1, le=10_000)
    success_probability: float | None = Field(default=None, ge=0, le=1)
    rate: float | None = Field(default=None, gt=0, le=1_000)
    mean: float | None = None
    std_dev: float | None = Field(default=None, gt=0, le=1_000_000)
    required_successes: int | None = Field(default=None, ge=1, le=10_000)
    population_size: int | None = Field(default=None, ge=1, le=1_000_000)
    population_successes: int | None = Field(default=None, ge=0, le=1_000_000)
    draws: int | None = Field(default=None, ge=1, le=1_000_000)
    scale: float | None = Field(default=None, gt=0, le=1_000_000)
    threshold: float | None = None
    degrees_freedom: float | None = Field(default=None, gt=0, le=1_000_000)
    numerator_df: float | None = Field(default=None, gt=0, le=1_000_000)
    denominator_df: float | None = Field(default=None, gt=0, le=1_000_000)
    noncentrality: float | None = Field(default=None, ge=0, le=1_000_000)
    support_min: float | None = None
    support_max: float | None = None
    integer_min: int | None = None
    integer_max: int | None = None
    custom_values: list[float] | None = Field(default=None, min_length=1, max_length=1_000)
    custom_probabilities: list[float] | None = Field(default=None, min_length=1, max_length=1_000)
    shape_alpha: float | None = Field(default=None, gt=0, le=1_000_000)
    shape_beta: float | None = Field(default=None, gt=0, le=1_000_000)
    location: float | None = None
    shape: float | None = Field(default=None, gt=0, le=1_000_000)
    mode_value: float | None = None
    lower: float | None = None
    upper: float | None = None

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_distribution_and_event(self) -> "ProbabilityAnalysisRequest":
        spec = PROBABILITY_SPECS[self.distribution]
        allowed = {parameter["key"] for parameter in spec["parameters"]}
        required = {
            parameter["key"] for parameter in spec["parameters"] if parameter.get("required", True)
        }
        supplied = {
            key
            for key in {
                "trials",
                "success_probability",
                "rate",
                "mean",
                "std_dev",
                "required_successes",
                "population_size",
                "population_successes",
                "draws",
                "scale",
                "threshold",
                "degrees_freedom",
                "numerator_df",
                "denominator_df",
                "noncentrality",
                "support_min",
                "support_max",
                "integer_min",
                "integer_max",
                "custom_values",
                "custom_probabilities",
                "shape_alpha",
                "shape_beta",
                "location",
                "shape",
                "mode_value",
            }
            if getattr(self, key) is not None
        }
        missing = required - supplied
        unexpected = supplied - allowed
        if missing:
            raise ValueError(f"Faltan parámetros requeridos: {', '.join(sorted(missing))}")
        if unexpected:
            raise ValueError(f"Parámetros no utilizados: {', '.join(sorted(unexpected))}")
        if spec["kind"] == "continuous" and self.operation == "exact":
            raise ValueError(
                "En una distribución continua, la probabilidad de un valor exacto es cero"
            )
        if self.distribution in {"geometric", "negative_binomial"} and not (
            0 < float(self.success_probability) <= 1
        ):
            raise ValueError("La probabilidad de éxito debe ser mayor que 0 y menor o igual que 1")
        if self.distribution == "hypergeometric":
            if int(self.population_successes) > int(self.population_size):
                raise ValueError("Los éxitos no pueden superar el tamaño de la población")
            if int(self.draws) > int(self.population_size):
                raise ValueError("La muestra no puede superar el tamaño de la población")
        if self.distribution in {"uniform", "triangular"} and not (
            float(self.support_min) < float(self.support_max)
        ):
            raise ValueError("El límite inferior debe ser menor que el límite superior")
        if self.distribution == "triangular" and not (
            float(self.support_min) <= float(self.mode_value) <= float(self.support_max)
        ):
            raise ValueError("La moda debe estar entre los límites de la distribución")
        if self.distribution == "integer_uniform" and int(self.integer_min) > int(self.integer_max):
            raise ValueError("El entero mínimo no puede superar al máximo")
        if self.distribution == "custom_discrete":
            values = self.custom_values or []
            probabilities = self.custom_probabilities or []
            if len(values) != len(probabilities):
                raise ValueError("Valores y probabilidades deben tener la misma longitud")
            if len(values) != len(set(values)):
                raise ValueError("Los valores de una distribución discreta deben ser únicos")
            if any(probability < 0 for probability in probabilities):
                raise ValueError("Las probabilidades no pueden ser negativas")
            if not math.isclose(sum(probabilities), 1.0, rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError("Las probabilidades deben sumar 1")

        if self.mode == "event" and self.operation is None:
            raise ValueError("El cálculo de un evento requiere una operación")
        if self.mode != "event" and self.operation is not None:
            raise ValueError("La operación solo se utiliza en el modo de evento")
        if self.mode == "event" and self.lower is None:
            raise ValueError("El cálculo de un evento requiere un valor")
        if self.mode != "event" and self.input is None:
            raise ValueError("El modo seleccionado requiere valores de entrada")
        if self.mode == "event" and self.input is not None:
            raise ValueError("La entrada por valores no se utiliza en el modo de evento")

        if self.mode == "event" and self.operation == "between":
            if self.upper is None:
                raise ValueError("El intervalo requiere un límite superior")
            if float(self.lower) > self.upper:
                raise ValueError("El límite inferior no puede superar al límite superior")
        elif self.upper is not None:
            raise ValueError(
                "El límite superior solo se utiliza en una probabilidad entre dos valores"
            )

        if (
            self.mode == "event"
            and self.distribution != "custom_discrete"
            and spec["kind"] == "discrete"
        ):
            values = [float(self.lower), *([] if self.upper is None else [self.upper])]
            if any(not float(value).is_integer() for value in values):
                raise ValueError("Esta distribución requiere valores enteros")
            if self.distribution != "integer_uniform" and any(value < 0 for value in values):
                raise ValueError("Esta distribución requiere conteos no negativos")
        if self.mean is not None and not math.isfinite(self.mean):
            raise ValueError("La media debe ser un número finito")
        return self


class ProbabilityInput(BaseModel):
    kind: Literal["constant", "column"]
    values: list[float | None] = Field(min_length=1, max_length=10_000)
    column_key: str | None = Field(default=None, max_length=80, pattern=r"^[A-Za-z0-9_-]+$")

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_source(self) -> "ProbabilityInput":
        if self.kind == "column" and not self.column_key:
            raise ValueError("La entrada por columna requiere la clave de la variable")
        if self.kind == "constant" and self.column_key is not None:
            raise ValueError("Una constante no debe indicar una columna")
        if self.kind == "constant" and len(self.values) != 1:
            raise ValueError("La entrada constante requiere exactamente un valor")
        return self


ProbabilityAnalysisRequest.model_rebuild()


class AnalysisRunRead(BaseModel):
    id: int
    dataset_id: int
    dataset_revision: int
    dataset_sha256: str
    analysis_type: str
    status: str
    engine_version: str
    parameters: dict[str, Any]
    result: dict[str, Any]
    warnings: list[str]
    created_at: datetime
    is_stale: bool


class AnalysisRunSummary(BaseModel):
    id: int
    dataset_id: int
    dataset_revision: int
    analysis_type: str
    status: str
    created_at: datetime
    is_stale: bool
    engine_version: str
    distribution: str | None = None
    mode: str | None = None
    formula: str | None = None
    result_preview: float | None = None
    input_kind: str | None = None
    presentation: "AnalysisRunPresentation"


class AnalysisRunPresentation(BaseModel):
    module: str
    title: str
    detail: str
    result: str
    filter_key: str
    filter_label: str
    tags: list[str] = Field(default_factory=list)


AnalysisRunSummary.model_rebuild()
