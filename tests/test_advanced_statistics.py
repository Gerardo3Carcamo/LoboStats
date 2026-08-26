import math

import pytest


def post(client, route, dataset_id, **payload):
    return client.post(route, json={"dataset_id": dataset_id, **payload})


def test_basic_stats_family(client, dataset_id):
    mean = post(client, "/api/basic-stats/confidence-interval", dataset_id, column="value")
    median1 = post(client, "/api/basic-stats/confidence-interval", dataset_id, column="value", statistic="median", seed=7)
    median2 = post(client, "/api/basic-stats/confidence-interval", dataset_id, column="value", statistic="median", seed=7)
    assert mean.get_json()["results"]["estimate"] == 15.25
    assert median1.get_json()["results"] == median2.get_json()["results"]
    grouped = post(client, "/api/basic-stats/by-group", dataset_id, value_column="value", group_column="group").get_json()["results"]
    assert [g["mean"] for g in grouped["groups"]] == [10.0, 20.5]
    corr = post(client, "/api/basic-stats/correlation-matrix", dataset_id, columns=["x", "y", "value"], method="spearman").get_json()["results"]
    cov = post(client, "/api/basic-stats/covariance-matrix", dataset_id, columns=["x", "y"]).get_json()["results"]
    assert corr["matrix"][0][1] > .99 and cov["matrix"][0][0] == 6.0


def test_one_sample_z(client, dataset_id):
    response = post(client, "/api/basic-stats/z-test", dataset_id, column="value", null_mean=10,
                    population_std=5, alternative="two-sided")
    assert response.status_code == 200
    results = response.get_json()["results"]
    assert results["z_statistic"] > 0
    assert results["confidence_interval"][0] < results["sample_mean"] < results["confidence_interval"][1]


@pytest.mark.parametrize("method", ["shapiro", "anderson", "kolmogorov-smirnov", "dagostino"])
def test_normality_methods(client, dataset_id, method):
    result = post(client, "/api/variance/normality", dataset_id, column="value", method=method)
    assert result.status_code == 200
    assert result.get_json()["results"]["statistic"] is not None


@pytest.mark.parametrize("route", ["levene", "bartlett"])
def test_variance_tests(client, dataset_id, route):
    body = post(client, f"/api/variance/{route}", dataset_id, value_column="value", group_column="group").get_json()
    assert math.isfinite(body["results"]["statistic"])
    assert len(body["results"]["groups"]) == 2


def test_nonparametric_family(client, dataset_id):
    tests = [
        ("mann-whitney", {"column1": "before", "column2": "after"}),
        ("wilcoxon", {"column1": "before", "column2": "after"}),
        ("kruskal-wallis", {"value_column": "value", "group_column": "group"}),
        ("friedman", {"columns": ["before", "after", "value"]}),
        ("sign-test", {"column": "value", "median": 15}),
    ]
    for route, payload in tests:
        response = post(client, f"/api/nonparametric/{route}", dataset_id, **payload)
        assert response.status_code == 200, response.get_json()
        assert response.get_json()["results"]["p_value"] is not None


def test_proportion_family_and_validation(client):
    one = client.post("/api/proportion/one-sample", json={"successes": 60, "nobs": 100, "null_proportion": .5})
    two = client.post("/api/proportion/two-sample", json={"successes": [60, 45], "nobs": [100, 100]})
    assert one.get_json()["results"]["proportion"] == .6
    assert two.get_json()["results"]["difference"] == pytest.approx(.15)
    invalid = client.post("/api/proportion/one-sample", json={"successes": 11, "nobs": 10, "null_proportion": .5})
    assert invalid.status_code == 400 and invalid.get_json()["error"]["code"] == "validation_error"


def test_correlation_family(client, dataset_id):
    for method in ["spearman", "kendall"]:
        result = post(client, f"/api/correlation/{method}", dataset_id, x="x", y="y").get_json()["results"]
        assert result["coefficient"] > .99
    partial = post(client, "/api/correlation/partial", dataset_id, x="x", y="y", controls=["value"])
    assert partial.status_code == 200
    assert partial.get_json()["results"]["degrees_of_freedom"] == 5


def test_power_family(client):
    t = client.post("/api/power/t-test", json={"effect_size": .5, "power": .8})
    anova = client.post("/api/power/anova", json={"effect_size": .25, "power": .8, "groups": 3})
    size = client.post("/api/power/sample-size", json={"test": "t-test", "effect_size": .5, "power": .8})
    assert t.get_json()["results"]["sample_size_per_group"] > 60
    assert anova.get_json()["results"]["total_sample_size"] > 100
    assert size.get_json()["results"]["sample_size"] > 30
    assert client.post("/api/power/t-test", json={"effect_size": .5, "power": .8, "sample_size": 20}).status_code == 400
