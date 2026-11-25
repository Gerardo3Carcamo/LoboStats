from types import SimpleNamespace

import pytest

from app.schemas.manufacturing import ManufacturingAnalysisRequest
from app.services.manufacturing_service import analyze_manufacturing


def dataset(rows, columns):
    return SimpleNamespace(
        id=1,
        name="Planta Norte",
        revision=1,
        content_sha256="d" * 64,
        row_count=len(rows),
        columns_data=[
            {"key": key, "name": name, "data_type": data_type} for key, name, data_type in columns
        ],
        rows_data=rows,
    )


def test_summary_calculates_oee_and_quality_losses() -> None:
    sample = dataset(
        [
            {
                "shift": "A",
                "planned": 480,
                "planned_stop": 30,
                "unplanned_stop": 60,
                "operating": 390,
                "ideal": 60,
                "total": 350,
                "good": 330,
                "first_pass": 320,
                "scrap": 20,
                "rework": 10,
                "defects": 25,
            },
            {
                "shift": "B",
                "planned": 480,
                "planned_stop": 30,
                "unplanned_stop": 60,
                "operating": 390,
                "ideal": 60,
                "total": 350,
                "good": 330,
                "first_pass": 320,
                "scrap": 20,
                "rework": 10,
                "defects": 25,
            },
        ],
        [
            ("shift", "Turno", "text"),
            ("planned", "Programado", "number"),
            ("planned_stop", "Paro planeado", "number"),
            ("unplanned_stop", "Paro no planeado", "number"),
            ("operating", "Operación", "number"),
            ("ideal", "Ciclo ideal", "number"),
            ("total", "Unidades", "number"),
            ("good", "Buenas", "number"),
            ("first_pass", "Buenas primera", "number"),
            ("scrap", "Scrap", "number"),
            ("rework", "Retrabajo", "number"),
            ("defects", "Defectos", "number"),
        ],
    )
    request = ManufacturingAnalysisRequest.model_validate(
        {
            "mode": "summary",
            "mapping": {
                "shift": "shift",
                "planned_time": "planned",
                "planned_downtime": "planned_stop",
                "unplanned_downtime": "unplanned_stop",
                "operating_time": "operating",
                "ideal_cycle_time": "ideal",
                "total_count": "total",
                "good_count": "good",
                "first_pass_good_count": "first_pass",
                "scrap_count": "scrap",
                "rework_count": "rework",
                "defect_count": "defects",
            },
            "time_unit": "minutes",
            "cycle_time_unit": "seconds",
            "targets": {"opportunities_per_unit": 5},
        }
    )

    result, warnings = analyze_manufacturing(sample, request)

    assert result["oee"]["availability"] == pytest.approx(780 / 960)
    assert result["oee"]["performance"] == pytest.approx(700 / 780)
    assert result["oee"]["quality"] == pytest.approx(660 / 700)
    assert result["oee"]["oee"] == pytest.approx((780 / 960) * (700 / 780) * (660 / 700))
    assert result["quality"]["first_pass_yield"] == pytest.approx(640 / 700)
    assert result["quality"]["dpmo"] == pytest.approx(50 / (700 * 5) * 1_000_000)
    assert any("cycle" in warning for warning in warnings)


def test_event_log_reports_pareto_mtbf_and_mttr() -> None:
    sample = dataset(
        [
            {"state": "Operando", "duration": 300, "reason": "Producción"},
            {"state": "Falla", "duration": 30, "reason": "Sensor"},
            {"state": "Operando", "duration": 120, "reason": "Producción"},
            {"state": "Falla", "duration": 20, "reason": "Sensor"},
            {"state": "Setup", "duration": 10, "reason": "Cambio"},
        ],
        [
            ("state", "Estado", "text"),
            ("duration", "Duración", "number"),
            ("reason", "Causa", "text"),
        ],
    )
    request = ManufacturingAnalysisRequest.model_validate(
        {
            "mode": "event_log",
            "mapping": {
                "event_state": "state",
                "duration": "duration",
                "reason": "reason",
            },
            "running_values": ["Operando"],
            "planned_stop_values": ["Setup"],
            "unplanned_stop_values": ["Falla"],
        }
    )

    result, _ = analyze_manufacturing(sample, request)

    assert result["downtime"]["unplanned_minutes"] == 50
    assert result["downtime"]["mttr_minutes"] == 25
    assert result["downtime"]["mtbf_minutes"] == 210
    assert result["downtime"]["pareto"][0]["reason"] == "Sensor"


def test_cycle_log_reports_distribution_quality_and_capability() -> None:
    rows = []
    for index, cycle in enumerate([54, 58, 61, 57, 65, 59, 62, 56, 60, 63], start=1):
        rows.append(
            {
                "cycle": cycle,
                "result": "OK" if index not in {5, 10} else "Scrap",
                "measure": 10 + (index - 5) * 0.02,
                "subgroup": f"S{(index - 1) // 2 + 1}",
            }
        )
    sample = dataset(
        rows,
        [
            ("cycle", "Ciclo", "number"),
            ("result", "Resultado", "text"),
            ("measure", "Medición", "number"),
            ("subgroup", "Subgrupo", "text"),
        ],
    )
    request = ManufacturingAnalysisRequest.model_validate(
        {
            "mode": "cycle_log",
            "mapping": {
                "cycle_time": "cycle",
                "quality_result": "result",
                "measurement": "measure",
                "subgroup": "subgroup",
            },
            "cycle_time_unit": "seconds",
            "good_values": ["OK"],
            "targets": {
                "standard_cycle_time": 60,
                "lower_specification": 9.8,
                "target": 10,
                "upper_specification": 10.2,
            },
        }
    )

    result, _ = analyze_manufacturing(sample, request)

    assert result["cycle"]["mean_minutes"] == pytest.approx(59.5 / 60)
    assert result["cycle"]["above_standard_count"] == 4
    assert result["quality"]["yield"] == pytest.approx(0.8)
    assert result["capability"]["cp"] > 0
    assert result["capability"]["ppk"] > 0
