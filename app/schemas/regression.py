from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

RegressionModelKey = Literal[
    "linear",
    "fitted_line",
    "stepwise",
    "best_subsets",
    "pls",
    "binary_logistic",
    "ordinal_logistic",
    "nominal_logistic",
]


class RegressionAnalysisRequest(BaseModel):
    model: RegressionModelKey
    response: str = Field(max_length=80, pattern=r"^[A-Za-z0-9_-]+$")
    predictors: list[str] = Field(min_length=1, max_length=50)
    categorical_predictors: list[str] = Field(default_factory=list, max_length=20)
    interactions: list[list[str]] = Field(default_factory=list, max_length=50)
    degree: int = Field(default=1, ge=1, le=3)
    include_intercept: bool = True
    confidence_level: float = Field(default=0.95, gt=0.5, lt=1.0)
    selection_method: Literal["forward", "backward", "stepwise"] = "stepwise"
    alpha_to_enter: float = Field(default=0.15, gt=0, lt=1)
    alpha_to_remove: float = Field(default=0.15, gt=0, lt=1)
    best_models_per_size: int = Field(default=2, ge=1, le=5)
    components: int = Field(default=2, ge=1, le=20)
    scale: bool = True
    event_value: Any | None = None
    reference_category: Any | None = None
    category_order: list[Any] | None = Field(default=None, min_length=3, max_length=20)

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    @field_validator("predictors")
    @classmethod
    def unique_predictors(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("Cada predictor se debe seleccionar una sola vez")
        return value

    @model_validator(mode="after")
    def validate_model_configuration(self) -> "RegressionAnalysisRequest":
        if self.response in self.predictors:
            raise ValueError("La respuesta no puede utilizarse también como predictor")
        unknown_categorical = set(self.categorical_predictors) - set(self.predictors)
        if unknown_categorical:
            raise ValueError("Los predictores categóricos deben estar incluidos en predictores")
        if self.model == "fitted_line" and len(self.predictors) != 1:
            raise ValueError("La gráfica de línea ajustada requiere exactamente un predictor")
        if self.model in {"stepwise", "best_subsets"} and len(self.predictors) < 2:
            raise ValueError("La selección de modelos requiere al menos dos predictores")
        if self.model == "best_subsets" and len(self.predictors) > 15:
            raise ValueError("Mejores subconjuntos admite hasta 15 predictores en esta versión")
        if self.model == "pls" and self.categorical_predictors:
            raise ValueError("PLS requiere predictores numéricos")
        if self.model == "binary_logistic" and self.event_value is None:
            raise ValueError("La regresión logística binaria requiere definir el evento")
        if self.model == "ordinal_logistic" and not self.category_order:
            raise ValueError("La regresión ordinal requiere el orden de las categorías")
        for interaction in self.interactions:
            if len(interaction) != 2 or any(key not in self.predictors for key in interaction):
                raise ValueError("Cada interacción requiere dos predictores seleccionados")
            if any(key in self.categorical_predictors for key in interaction):
                raise ValueError("Las interacciones categóricas todavía no están disponibles")
        return self
