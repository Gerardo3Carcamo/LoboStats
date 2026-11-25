from types import SimpleNamespace

import numpy as np
import pytest

from app.schemas.regression import RegressionAnalysisRequest
from app.services.regression_service import analyze_regression


def dataset():
    rows = []
    for index in range(1, 61):
        x1 = index / 5
        x2 = float(np.sin(index / 4) * 3 + index % 3)
        y = 3 + 2 * x1 - 1.5 * x2 + ((index % 5) - 2) * 0.2
        score = -1 + 0.22 * x1 - 0.12 * x2
        probability = 1 / (1 + np.exp(-score))
        binary = "Sí" if ((index * 37) % 100) / 100 < probability else "No"
        ordinal = "Bajo" if y < 10 else "Medio" if y < 20 else "Alto"
        nominal = ["A", "B", "C"][index % 3]
        rows.append(
            {
                "y": y,
                "x1": x1,
                "x2": x2,
                "group": "Norte" if index % 2 else "Sur",
                "binary": binary,
                "ordinal": ordinal,
                "nominal": nominal,
            }
        )
    return SimpleNamespace(
        id=1,
        name="Modelos",
        revision=1,
        content_sha256="b" * 64,
        row_count=len(rows),
        columns_data=[
            {"key": "y", "name": "Respuesta", "data_type": "number"},
            {"key": "x1", "name": "Predictor 1", "data_type": "number"},
            {"key": "x2", "name": "Predictor 2", "data_type": "number"},
            {"key": "group", "name": "Región", "data_type": "text"},
            {"key": "binary", "name": "Compra", "data_type": "text"},
            {"key": "ordinal", "name": "Nivel", "data_type": "text"},
            {"key": "nominal", "name": "Preferencia", "data_type": "text"},
        ],
        rows_data=rows,
    )


def test_multiple_linear_regression_reports_fit_and_diagnostics() -> None:
    result, warnings = analyze_regression(
        dataset(),
        RegressionAnalysisRequest(model="linear", response="y", predictors=["x1", "x2"]),
    )

    assert result["fit"]["r_squared"] > 0.99
    assert result["coefficients"][1]["coefficient"] == pytest.approx(2, abs=0.02)
    assert result["coefficients"][2]["coefficient"] == pytest.approx(-1.5, abs=0.02)
    assert len(result["predictions"]) == 60
    assert result["diagnostics"]["durbin_watson"] is not None
    assert all("Cook" in warning for warning in warnings)


def test_fitted_line_produces_confidence_and_prediction_bands() -> None:
    result, _ = analyze_regression(
        dataset(),
        RegressionAnalysisRequest(
            model="fitted_line",
            response="y",
            predictors=["x1"],
            degree=2,
        ),
    )

    plot = result["plots"]["fitted_line"]
    assert len(plot["x"]) == 120
    assert len(plot["prediction_lower"]) == 120


@pytest.mark.parametrize("model", ["stepwise", "best_subsets"])
def test_model_selection_returns_candidate_history(model: str) -> None:
    result, warnings = analyze_regression(
        dataset(),
        RegressionAnalysisRequest(model=model, response="y", predictors=["x1", "x2"]),
    )

    assert set(result["selection"]["selected_predictors"]) == {"x1", "x2"}
    assert any(
        "selección" in warning.lower() or "modelos" in warning.lower() for warning in warnings
    )


def test_pls_returns_components_and_cross_validation() -> None:
    result, _ = analyze_regression(
        dataset(),
        RegressionAnalysisRequest(
            model="pls",
            response="y",
            predictors=["x1", "x2"],
            components=2,
        ),
    )

    assert result["fit"]["components"] == 2
    assert result["fit"]["r_squared"] > 0.98
    assert len(result["fit"]["cross_validation"]) == 2


def test_binary_logistic_reports_probabilities_and_auc() -> None:
    result, _ = analyze_regression(
        dataset(),
        RegressionAnalysisRequest(
            model="binary_logistic",
            response="binary",
            predictors=["x1", "x2"],
            event_value="Sí",
        ),
    )

    assert 0.5 <= result["fit"]["auc"] <= 1
    assert all(0 <= row["probability"] <= 1 for row in result["predictions"])


def test_ordinal_logistic_preserves_category_order() -> None:
    result, _ = analyze_regression(
        dataset(),
        RegressionAnalysisRequest(
            model="ordinal_logistic",
            response="ordinal",
            predictors=["x1", "x2"],
            category_order=["Bajo", "Medio", "Alto"],
        ),
    )

    assert result["fit"]["categories"] == ["Bajo", "Medio", "Alto"]
    assert len(result["predictions"]) == 60


def test_nominal_logistic_uses_reference_category() -> None:
    result, _ = analyze_regression(
        dataset(),
        RegressionAnalysisRequest(
            model="nominal_logistic",
            response="nominal",
            predictors=["x1", "x2"],
            reference_category="A",
        ),
    )

    assert result["fit"]["reference_category"] == "A"
    assert len(result["coefficients"]) == 6
