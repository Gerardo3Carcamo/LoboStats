import math

import numpy as np
import pytest
from scipy import stats


def post(client, operation, distribution, params, **payload):
    return client.post(
        f"/api/distributions/{operation}",
        json={"distribution": distribution, "params": params, **payload},
    )


def test_normal_operations_and_json_contract(client):
    params = {"mean": 2, "std": 3}
    pdf = post(client, "pdf", "normal", params, x=[2, 5])
    assert pdf.status_code == 200
    body = pdf.get_json()
    assert set(body) == {"distribution", "operation", "parameters", "results", "interpretation"}
    assert body["results"]["values"] == pytest.approx(stats.norm.pdf([2, 5], 2, 3))

    cdf = post(client, "cdf", "normal", params, x=2).get_json()
    assert cdf["results"]["value"] == pytest.approx(0.5)
    ppf = post(client, "ppf", "normal", params, q=0.975).get_json()
    assert ppf["results"]["value"] == pytest.approx(stats.norm.ppf(0.975, 2, 3))
    interval = post(client, "interval", "normal", params, confidence=0.95).get_json()
    assert interval["results"]["lower"] < 2 < interval["results"]["upper"]


@pytest.mark.parametrize(
    ("distribution", "params", "x", "expected"),
    [
        ("binomial", {"n": 10, "p": 0.3, "loc": 0}, 3, stats.binom.pmf(3, 10, 0.3)),
        ("t", {"df": 8, "loc": 1, "scale": 2}, 1, stats.t.pdf(1, 8, 1, 2)),
        ("chi2", {"df": 4, "loc": 0, "scale": 1}, 3, stats.chi2.pdf(3, 4)),
        ("hypergeom", {"M": 30, "n": 12, "N": 8, "loc": 0}, 4, stats.hypergeom.pmf(4, 30, 12, 8)),
    ],
)
def test_distribution_values(client, distribution, params, x, expected):
    body = post(client, "pdf", distribution, params, x=x).get_json()
    assert body["results"]["value"] == pytest.approx(expected)
    if distribution in {"binomial", "hypergeom"}:
        assert body["results"]["metadata"]["function"] == "PMF"
        assert "PMF" in body["interpretation"]


def test_random_is_reproducible_and_curve_is_bounded(client):
    payload = {"size": 25, "seed": 12}
    first = post(client, "random", "poisson", {"mu": 2, "loc": 0}, **payload).get_json()
    second = post(client, "random", "poisson", {"mu": 2, "loc": 0}, **payload).get_json()
    assert first["results"]["values"] == second["results"]["values"]
    curve = post(client, "curve", "binomial", {"n": 10, "p": 0.5}, points=5).get_json()
    assert len(curve["results"]["values"]) <= 5
    assert curve["results"]["metadata"]["function"] == "PMF"


@pytest.mark.parametrize(
    "payload",
    [
        {"distribution": "normal", "params": {"mean": 0, "std": 0}, "x": 0},
        {"distribution": "binomial", "params": {"n": 2.5, "p": 0.5}, "x": 1},
        {"distribution": "hypergeom", "params": {"M": 10, "n": 11, "N": 2}, "x": 1},
        {"distribution": "normal", "params": {}, "q": 1},
        {"distribution": "normal", "params": {}, "x": math.inf},
    ],
)
def test_invalid_distribution_inputs(client, payload):
    operation = "ppf" if "q" in payload else "pdf"
    assert client.post(f"/api/distributions/{operation}", json=payload).status_code == 400


@pytest.fixture()
def distribution_dataset(client):
    rng = np.random.default_rng(41)
    values = rng.lognormal(mean=0.3, sigma=0.35, size=90)
    response = client.post(
        "/api/data/manual",
        json={"name": "distribution-id", "columns": ["positive"], "rows": [[float(x)] for x in values]},
    )
    assert response.status_code == 201
    return response.get_json()["dataset_id"]


def test_fit_and_goodness_of_fit(client, distribution_dataset):
    payload = {
        "dataset_id": distribution_dataset,
        "column": "positive",
        "distribution": "lognormal",
        "bootstrap_reps": 30,
    }
    fit = client.post("/api/distributions/fit", json=payload)
    assert fit.status_code == 200
    results = fit.get_json()["results"]
    assert results["fitted_parameters"]["loc"] == 0
    assert results["parameter_count"] == 2
    assert results["anderson_darling"]["bootstrap_reps_successful"] == 30
    assert math.isfinite(results["aic"])
    gof = client.post("/api/distributions/goodness-of-fit", json=payload)
    assert gof.status_code == 200
    assert gof.get_json()["operation"] == "goodness-of-fit"


def test_identify_ranks_and_probability_bands(client, distribution_dataset):
    common = {"dataset_id": distribution_dataset, "column": "positive"}
    identified = client.post(
        "/api/distributions/identify",
        json={**common, "candidates": ["normal", "lognormal"], "bootstrap_reps": 30},
    )
    assert identified.status_code == 200
    results = identified.get_json()["results"]
    assert results["best_distribution"] in {"normal", "lognormal"}
    assert [item["rank"] for item in results["candidates"]] == [1, 2]
    assert results["failures"] == []

    plot = client.post(
        "/api/distributions/probability-plot",
        json={**common, "distribution": "lognormal", "confidence": 0.9},
    )
    assert plot.status_code == 200
    plot_results = plot.get_json()["results"]
    assert len(plot_results["points"]) == len(plot_results["confidence_band"]) == 90
    assert all(item["lower"] < item["upper"] for item in plot_results["confidence_band"])
    assert plot_results["reference_line"] == {"intercept": 0.0, "slope": 1.0}
    assert "puntual" in plot_results["warning"]
