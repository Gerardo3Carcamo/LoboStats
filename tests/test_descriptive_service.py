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


def test_advanced_summary_reports_robust_measures_normality_and_missingness() -> None:
    result, _ = analyze_descriptive(
        dataset(
            [{"x": value, "category": "A"} for value in [1, 2, 3, 4, 5, 6, 7, 8, 9, 30]]
            + [{"x": None, "category": "A"}]
        ),
        DescriptiveAnalysisRequest(columns=["x"], percentiles=[0.1, 0.9]),
    )

    numeric = result["groups"][0]["columns"][0]["numeric"]
    assert numeric["trimmed_mean"] == pytest.approx(5.5)
    assert numeric["mad"] == pytest.approx(2.5)
    assert numeric["geometric_mean"] is not None
    assert numeric["percentiles"] == {"p10": 1.1, "p90": pytest.approx(27.9)}
    assert numeric["confidence_interval_median"]["method"] == "binomial exacto"
    assert numeric["confidence_interval_std_dev"]["lower"] > 0
    assert 0 <= numeric["normality"]["p_value"] <= 1
    assert numeric["ecdf"]["probabilities"][-1] == 1
    assert result["missingness"]["complete_rows"] == 10
    assert result["missingness"]["incomplete_rows"] == 1


def test_grouped_outlier_keeps_original_worksheet_row_number() -> None:
    rows = [
        {"x": 900, "category": "A"},
        {"x": 1, "category": "B"},
        {"x": 901, "category": "A"},
        {"x": 2, "category": "B"},
        {"x": 3, "category": "B"},
        {"x": 4, "category": "B"},
        {"x": 5, "category": "B"},
        {"x": 6, "category": "B"},
        {"x": 7, "category": "B"},
        {"x": 100, "category": "B"},
    ]
    result, _ = analyze_descriptive(
        dataset(rows),
        DescriptiveAnalysisRequest(columns=["x"], group_by="category"),
    )

    group_b = next(group for group in result["groups"] if group["label"] == "B")
    assert group_b["columns"][0]["numeric"]["boxplot"]["outlier_details"] == [
        {"row": 10, "value": 100.0}
    ]
