from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

HypothesisTestKey = Literal[
    "one_sample_z",
    "one_sample_t",
    "two_sample_t",
    "paired_t",
    "one_proportion",
    "two_proportions",
    "one_sample_poisson_rate",
    "two_sample_poisson_rate",
    "one_variance",
    "two_variances",
    "correlation",
    "normality",
    "chi_square_independence",
    "poisson_goodness_of_fit",
    "one_sample_sign",
    "one_sample_wilcoxon",
    "mann_whitney",
    "kruskal_wallis",
    "mood_median",
    "friedman",
    "runs",
]


class HypothesisAnalysisRequest(BaseModel):
    test: HypothesisTestKey
    column: str | None = Field(default=None, max_length=80, pattern=r"^[A-Za-z0-9_-]+$")
    second_column: str | None = Field(default=None, max_length=80, pattern=r"^[A-Za-z0-9_-]+$")
    group_by: str | None = Field(default=None, max_length=80, pattern=r"^[A-Za-z0-9_-]+$")
    block_by: str | None = Field(default=None, max_length=80, pattern=r"^[A-Za-z0-9_-]+$")
    groups: list[Any] | None = Field(default=None, min_length=2, max_length=100)
    hypothesized_value: float = 0.0
    known_std_dev: float | None = Field(default=None, gt=0)
    alternative: Literal["two_sided", "less", "greater"] = "two_sided"
    confidence_level: float = Field(default=0.95, gt=0.5, lt=1.0)
    equal_variances: bool = False
    method: Literal[
        "auto",
        "exact",
        "normal",
        "pearson",
        "spearman",
        "anderson_darling",
        "ryan_joiner",
        "kolmogorov_smirnov",
    ] = "auto"
    success_value: Any | None = None
    observation_length: float = Field(default=1.0, gt=0)
    second_observation_length: float = Field(default=1.0, gt=0)
    expected_probabilities: dict[str, float] | None = None
    sequence_cutoff: float | None = None

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_required_sources(self) -> "HypothesisAnalysisRequest":
        if self.column is None:
            raise ValueError("La prueba requiere una columna principal")
        if self.test in {"paired_t", "correlation", "chi_square_independence"}:
            if self.second_column is None:
                raise ValueError("La prueba requiere una segunda columna")
        if (
            self.test
            in {
                "two_sample_t",
                "two_proportions",
                "two_sample_poisson_rate",
                "two_variances",
                "mann_whitney",
                "kruskal_wallis",
                "mood_median",
                "friedman",
            }
            and self.group_by is None
        ):
            raise ValueError("La prueba requiere una variable de agrupación")
        if self.test == "friedman" and self.block_by is None:
            raise ValueError("Friedman requiere una variable de bloque")
        if self.test == "one_sample_z" and self.known_std_dev is None:
            raise ValueError("La prueba Z requiere la desviación estándar poblacional")
        if self.test in {"one_proportion", "two_proportions"} and self.success_value is None:
            raise ValueError("La prueba de proporciones requiere definir el evento de interés")
        if self.expected_probabilities:
            total = sum(self.expected_probabilities.values())
            if any(value <= 0 for value in self.expected_probabilities.values()):
                raise ValueError("Las probabilidades esperadas deben ser positivas")
            if abs(total - 1.0) > 1e-9:
                raise ValueError("Las probabilidades esperadas deben sumar 1")
        return self
