from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ManufacturingColumnMapping(BaseModel):
    line: str | None = None
    machine: str | None = None
    shift: str | None = None
    product: str | None = None
    process: str | None = None
    subgroup: str | None = None

    period_start: str | None = None
    period_end: str | None = None
    duration: str | None = None
    event_state: str | None = None
    reason: str | None = None

    planned_time: str | None = None
    operating_time: str | None = None
    planned_downtime: str | None = None
    unplanned_downtime: str | None = None

    cycle_time: str | None = None
    ideal_cycle_time: str | None = None
    standard_cycle_time: str | None = None
    demand: str | None = None

    total_count: str | None = None
    good_count: str | None = None
    first_pass_good_count: str | None = None
    scrap_count: str | None = None
    rework_count: str | None = None
    defect_count: str | None = None
    opportunities: str | None = None
    quality_result: str | None = None
    measurement: str | None = None

    model_config = ConfigDict(extra="forbid")


class ManufacturingTargets(BaseModel):
    ideal_cycle_time: float | None = Field(default=None, gt=0)
    standard_cycle_time: float | None = Field(default=None, gt=0)
    demand: float | None = Field(default=None, gt=0)
    opportunities_per_unit: float | None = Field(default=None, gt=0)
    lower_specification: float | None = None
    target: float | None = None
    upper_specification: float | None = None

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_specifications(self) -> "ManufacturingTargets":
        if (
            self.lower_specification is not None
            and self.upper_specification is not None
            and self.lower_specification >= self.upper_specification
        ):
            raise ValueError("LSL debe ser menor que USL")
        return self


class ManufacturingAnalysisRequest(BaseModel):
    mode: Literal["summary", "event_log", "cycle_log"]
    mapping: ManufacturingColumnMapping
    targets: ManufacturingTargets = Field(default_factory=ManufacturingTargets)
    time_unit: Literal["seconds", "minutes", "hours"] = "minutes"
    cycle_time_unit: Literal["seconds", "minutes", "hours"] = "seconds"
    running_values: list[Any] = Field(default_factory=lambda: ["running", "operando"])
    planned_stop_values: list[Any] = Field(
        default_factory=lambda: ["planned_stop", "paro planeado", "setup"]
    )
    unplanned_stop_values: list[Any] = Field(
        default_factory=lambda: ["unplanned_stop", "paro no planeado", "falla"]
    )
    good_values: list[Any] = Field(default_factory=lambda: ["good", "bueno", "ok"])
    filters: dict[str, list[Any]] = Field(default_factory=dict, max_length=20)

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_mode_mapping(self) -> "ManufacturingAnalysisRequest":
        if self.mode == "summary" and self.mapping.planned_time is None:
            raise ValueError("El resumen por turno requiere tiempo programado")
        if self.mode == "event_log" and not (
            self.mapping.duration or (self.mapping.period_start and self.mapping.period_end)
        ):
            raise ValueError("La bitácora requiere duración o columnas de inicio y fin")
        if self.mode == "event_log" and self.mapping.event_state is None:
            raise ValueError("La bitácora requiere una columna de estado")
        if self.mode == "cycle_log" and self.mapping.cycle_time is None:
            raise ValueError("El registro de ciclos requiere tiempo de ciclo")
        return self
