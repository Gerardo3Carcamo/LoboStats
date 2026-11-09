import pytest
from pydantic import ValidationError

from app.schemas.analysis import ProbabilityAnalysisRequest
from app.services.probability_service import analyze_probability


def test_binomial_exact_probability() -> None:
    request = ProbabilityAnalysisRequest(
        distribution="binomial",
        operation="exact",
        trials=10,
        success_probability=0.3,
        lower=3,
    )

    result, warnings = analyze_probability(request)

    assert result["probability"] == pytest.approx(0.266827932)
    assert result["distribution"]["mean"] == pytest.approx(3)
    assert result["event_formula"] == "P(X = 3)"
    assert warnings == []


def test_poisson_interval_probability() -> None:
    request = ProbabilityAnalysisRequest(
        distribution="poisson",
        operation="between",
        rate=4,
        lower=2,
        upper=5,
    )

    result, _ = analyze_probability(request)

    assert result["probability"] == pytest.approx(0.6935521926)
    assert any(point["selected"] for point in result["points"])


def test_normal_central_probability() -> None:
    request = ProbabilityAnalysisRequest(
        distribution="normal",
        operation="between",
        mean=100,
        std_dev=15,
        lower=85,
        upper=115,
    )

    result, _ = analyze_probability(request)

    assert result["probability"] == pytest.approx(0.682689492, rel=1e-7)
    assert result["percentage"] == pytest.approx(68.2689492, rel=1e-7)


def test_normal_cdf_and_inverse_probability() -> None:
    cdf_request = ProbabilityAnalysisRequest(
        distribution="normal",
        mode="cdf",
        mean=100,
        std_dev=15,
        input={"kind": "constant", "values": [115]},
    )
    inverse_request = ProbabilityAnalysisRequest(
        distribution="normal",
        mode="inverse",
        mean=100,
        std_dev=15,
        input={"kind": "constant", "values": [0.975]},
    )

    cdf_result, _ = analyze_probability(cdf_request)
    inverse_result, _ = analyze_probability(inverse_request)

    assert cdf_result["result_value"] == pytest.approx(0.841344746, rel=1e-7)
    assert inverse_result["result_value"] == pytest.approx(129.39946, rel=1e-6)
    assert inverse_result["calculation_formula"] == "F⁻¹(0.975)"


def test_batch_calculation_keeps_row_errors() -> None:
    request = ProbabilityAnalysisRequest(
        distribution="binomial",
        mode="pdf",
        trials=10,
        success_probability=0.3,
        input={"kind": "column", "column_key": "c1", "values": [2, None, 2.5, 3]},
    )

    result, warnings = analyze_probability(request)

    assert [row["status"] for row in result["calculations"]] == [
        "valid",
        "invalid",
        "invalid",
        "valid",
    ]
    assert result["calculations"][3]["result"] == pytest.approx(0.266827932)
    assert warnings == ["2 valores no se calcularon; revisa los errores en la tabla"]


@pytest.mark.parametrize(
    ("distribution", "parameters", "value", "expected"),
    [
        ("geometric", {"success_probability": 0.25}, 3, 0.578125),
        (
            "negative_binomial",
            {"required_successes": 3, "success_probability": 0.4},
            2,
            0.31744,
        ),
        (
            "hypergeometric",
            {"population_size": 50, "population_successes": 10, "draws": 5},
            1,
            0.7418999792,
        ),
        ("exponential", {"scale": 5, "threshold": 0}, 5, 0.6321205588),
        ("student_t", {"degrees_freedom": 10}, 0, 0.5),
        ("chi_square", {"degrees_freedom": 5, "noncentrality": 0}, 5, 0.584119813),
        (
            "f",
            {"numerator_df": 5, "denominator_df": 10, "noncentrality": 0},
            1,
            0.5348805735,
        ),
    ],
)
def test_additional_distribution_cdfs(
    distribution: str,
    parameters: dict[str, float | int],
    value: float,
    expected: float,
) -> None:
    request = ProbabilityAnalysisRequest.model_validate(
        {
            "distribution": distribution,
            "mode": "cdf",
            "input": {"kind": "constant", "values": [value]},
            **parameters,
        }
    )

    result, _ = analyze_probability(request)

    assert result["result_value"] == pytest.approx(expected, rel=1e-7)


@pytest.mark.parametrize(
    ("distribution", "parameters", "value", "expected"),
    [
        ("uniform", {"support_min": 0, "support_max": 10}, 5, 0.5),
        ("integer_uniform", {"integer_min": 1, "integer_max": 6}, 3, 0.5),
        (
            "custom_discrete",
            {"custom_values": [1, 2, 3], "custom_probabilities": [0.2, 0.3, 0.5]},
            2,
            0.5,
        ),
        ("beta", {"shape_alpha": 2, "shape_beta": 2}, 0.5, 0.5),
        ("cauchy", {"location": 0, "scale": 1}, 0, 0.5),
        ("gamma", {"shape": 2, "scale": 3, "threshold": 0}, 6, 0.5939941503),
        ("laplace", {"location": 0, "scale": 1}, 0, 0.5),
        ("logistic", {"location": 0, "scale": 1}, 0, 0.5),
        (
            "loglogistic",
            {"location": 0, "scale": 1, "threshold": 0},
            1,
            0.5,
        ),
        ("lognormal", {"location": 0, "scale": 1, "threshold": 0}, 1, 0.5),
        (
            "triangular",
            {"support_min": 0, "mode_value": 5, "support_max": 10},
            5,
            0.5,
        ),
        ("weibull", {"shape": 2, "scale": 1, "threshold": 0}, 1, 0.6321205588),
        ("extreme_min", {"location": 0, "scale": 1}, 0, 0.6321205588),
        ("extreme_max", {"location": 0, "scale": 1}, 0, 0.3678794412),
    ],
)
def test_minitab_catalog_distribution_cdfs(
    distribution: str,
    parameters: dict[str, object],
    value: float,
    expected: float,
) -> None:
    request = ProbabilityAnalysisRequest.model_validate(
        {
            "distribution": distribution,
            "mode": "cdf",
            "input": {"kind": "constant", "values": [value]},
            **parameters,
        }
    )

    result, _ = analyze_probability(request)

    assert result["result_value"] == pytest.approx(expected, rel=1e-7)


@pytest.mark.parametrize(
    "payload",
    [
        {"distribution": "normal", "operation": "exact", "mean": 0, "std_dev": 1, "lower": 0},
        {
            "distribution": "binomial",
            "operation": "between",
            "trials": 10,
            "success_probability": 0.5,
            "lower": 6,
            "upper": 3,
        },
        {
            "distribution": "poisson",
            "operation": "at_most",
            "rate": 3,
            "lower": 1.5,
        },
        {
            "distribution": "triangular",
            "mode": "cdf",
            "support_min": 0,
            "mode_value": 12,
            "support_max": 10,
            "input": {"kind": "constant", "values": [5]},
        },
        {
            "distribution": "custom_discrete",
            "mode": "cdf",
            "custom_values": [1, 2],
            "custom_probabilities": [0.2, 0.2],
            "input": {"kind": "constant", "values": [1]},
        },
    ],
)
def test_rejects_invalid_probability_requests(payload: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        ProbabilityAnalysisRequest.model_validate(payload)
