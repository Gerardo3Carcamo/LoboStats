from types import SimpleNamespace

import pytest

from app.schemas.analysis import DescriptiveAnalysisRequest
from app.services.descriptive_service import analyze_descriptive


def dataset(rows):
    return SimpleNamespace(
        id=1,
        name="Ejemplo",
        revision=1,
        content_sha256="a" * 64,
        row_count=len(rows),
        columns_data=[
            {"key": "x", "name": "Medición", "data_type": "number"},
            {"key": "category", "name": "Categoría", "data_type": "text"},
        ],
        rows_data=rows,
    )


def test_numeric_summary_uses_sample_statistics_and_reports_graphs() -> None:
    result, warnings = analyze_descriptive(
        dataset(
            [
                {"x": 1, "category": "A"},
                {"x": 2, "category": "A"},
                {"x": 3, "category": "B"},
                {"x": 4, "category": "B"},
                {"x": None, "category": "B"},
            ]
        ),
        DescriptiveAnalysisRequest(columns=["x"], histogram_bins=4),
    )

    column = result["groups"][0]["columns"][0]
    numeric = column["numeric"]
    assert column["valid_count"] == 4
    assert column["missing_count"] == 1
    assert numeric["mean"] == 2.5
    assert numeric["variance"] == pytest.approx(5 / 3)
    assert numeric["q1"] == 1.25
    assert numeric["q3"] == 3.75
    assert sum(numeric["histogram"]["counts"]) == 4
    assert len(numeric["qq_plot"]["observed"]) == 4
    assert warnings == []


def test_categorical_summary_is_frequency_sorted() -> None:
    result, _ = analyze_descriptive(
        dataset(
            [
                {"x": 1, "category": "B"},
                {"x": 2, "category": "A"},
                {"x": 3, "category": "B"},
                {"x": 4, "category": None},
            ]
        ),
        DescriptiveAnalysisRequest(columns=["category"]),
    )

    column = result["groups"][0]["columns"][0]
    assert column["valid_count"] == 3
    assert column["missing_count"] == 1
    assert column["categorical"]["mode"] == "B"
    assert column["categorical"]["frequencies"][0] == {
        "value": "B",
        "count": 2,
        "percentage": pytest.approx(200 / 3),
    }


def test_single_numeric_value_returns_explicit_warning() -> None:
    result, warnings = analyze_descriptive(
        dataset([{"x": 7, "category": "A"}]),
        DescriptiveAnalysisRequest(columns=["x"]),
    )

    numeric = result["groups"][0]["columns"][0]["numeric"]
    assert numeric["variance"] is None
    assert numeric["confidence_interval_mean"] is None
    assert any("al menos 2 valores" in warning for warning in warnings)
