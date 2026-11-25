from types import SimpleNamespace

import pytest

from app.schemas.hypothesis import HypothesisAnalysisRequest
from app.services.hypothesis_service import analyze_hypothesis


def dataset():
    rows = []
    for index in range(1, 19):
        group = "A" if index <= 9 else "B"
        rows.append(
            {
                "x": index,
                "y": index * 2 + (index % 2),
                "before": 20 + index,
                "after": 18 + index + index % 2,
                "group": group,
                "event": "Sí" if index % 3 else "No",
                "category": "Alta" if index > 6 else "Baja",
                "count": index % 4,
                "block": f"B{(index - 1) % 6 + 1}",
                "treatment": "T1" if index <= 6 else "T2" if index <= 12 else "T3",
            }
        )
    return SimpleNamespace(
        id=1,
        name="Ejemplo",
        revision=1,
        content_sha256="a" * 64,
        row_count=len(rows),
        columns_data=[
            {"key": "x", "name": "X", "data_type": "number"},
            {"key": "y", "name": "Y", "data_type": "number"},
            {"key": "before", "name": "Antes", "data_type": "number"},
            {"key": "after", "name": "Después", "data_type": "number"},
            {"key": "group", "name": "Grupo", "data_type": "text"},
            {"key": "event", "name": "Evento", "data_type": "text"},
            {"key": "category", "name": "Categoría", "data_type": "text"},
            {"key": "count", "name": "Conteo", "data_type": "number"},
            {"key": "block", "name": "Bloque", "data_type": "text"},
            {"key": "treatment", "name": "Tratamiento", "data_type": "text"},
        ],
        rows_data=rows,
    )


@pytest.mark.parametrize(
    ("analysis_request", "method"),
    [
        (
            HypothesisAnalysisRequest(
                test="one_sample_z", column="x", hypothesized_value=5, known_std_dev=4
            ),
            "Z de una muestra",
        ),
        (
            HypothesisAnalysisRequest(test="one_sample_t", column="x", hypothesized_value=5),
            "t de una muestra",
        ),
        (
            HypothesisAnalysisRequest(test="two_sample_t", column="x", group_by="group"),
            "t de Welch",
        ),
        (
            HypothesisAnalysisRequest(test="paired_t", column="before", second_column="after"),
            "t pareada",
        ),
        (
            HypothesisAnalysisRequest(test="one_variance", column="x", hypothesized_value=10),
            "Chi-cuadrado",
        ),
        (
            HypothesisAnalysisRequest(
                test="two_variances", column="x", group_by="group", hypothesized_value=1
            ),
            "F para",
        ),
        (HypothesisAnalysisRequest(test="correlation", column="x", second_column="y"), "Pearson"),
        (HypothesisAnalysisRequest(test="normality", column="x"), "Anderson"),
        (
            HypothesisAnalysisRequest(test="one_sample_sign", column="x", hypothesized_value=5),
            "Signos",
        ),
        (
            HypothesisAnalysisRequest(test="one_sample_wilcoxon", column="x", hypothesized_value=5),
            "Wilcoxon",
        ),
        (HypothesisAnalysisRequest(test="mann_whitney", column="x", group_by="group"), "Mann"),
        (HypothesisAnalysisRequest(test="kruskal_wallis", column="x", group_by="group"), "Kruskal"),
        (HypothesisAnalysisRequest(test="mood_median", column="x", group_by="group"), "Mood"),
        (HypothesisAnalysisRequest(test="runs", column="x", sequence_cutoff=6.5), "Corridas"),
    ],
)
def test_supported_numeric_tests_return_a_decision(analysis_request, method) -> None:
    result, warnings = analyze_hypothesis(dataset(), analysis_request)

    assert method in result["method"]
    assert 0 <= result["result"]["p_value"] <= 1
    assert result["decision"] in {"reject", "fail_to_reject"}
    assert result["interpretation"]
    assert warnings == []


def test_proportion_poisson_and_categorical_tests() -> None:
    requests = [
        HypothesisAnalysisRequest(
            test="one_proportion",
            column="event",
            success_value="Sí",
            hypothesized_value=0.5,
        ),
        HypothesisAnalysisRequest(
            test="two_proportions",
            column="event",
            group_by="group",
            success_value="Sí",
        ),
        HypothesisAnalysisRequest(
            test="one_sample_poisson_rate",
            column="count",
            hypothesized_value=1.5,
        ),
        HypothesisAnalysisRequest(
            test="two_sample_poisson_rate",
            column="count",
            group_by="group",
        ),
        HypothesisAnalysisRequest(
            test="chi_square_independence",
            column="event",
            second_column="category",
        ),
    ]

    for request in requests:
        result, _ = analyze_hypothesis(dataset(), request)
        assert result["result"]["p_value"] is not None
        assert 0 <= result["result"]["p_value"] <= 1


def test_poisson_goodness_of_fit() -> None:
    sample = dataset()
    sample.rows_data = [
        {"count": value}
        for value in ([0] * 22 + [1] * 34 + [2] * 25 + [3] * 13 + [4] * 5 + [5])
    ]
    sample.row_count = len(sample.rows_data)

    result, _ = analyze_hypothesis(
        sample,
        HypothesisAnalysisRequest(test="poisson_goodness_of_fit", column="count"),
    )

    assert 0 <= result["result"]["p_value"] <= 1


def test_friedman_uses_complete_blocks() -> None:
    result, _ = analyze_hypothesis(
        dataset(),
        HypothesisAnalysisRequest(
            test="friedman",
            column="x",
            group_by="treatment",
            block_by="block",
        ),
    )

    assert result["result"]["details"]["blocks"] == 6


def test_missing_values_are_reported() -> None:
    sample = dataset()
    sample.rows_data.append({"x": None})
    sample.row_count += 1

    _, warnings = analyze_hypothesis(
        sample,
        HypothesisAnalysisRequest(test="one_sample_t", column="x"),
    )

    assert "Se excluyeron 1 filas" in warnings[0]
