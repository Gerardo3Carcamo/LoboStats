SUMMARY_FIELDS = {
    "count", "missing", "mean", "median", "mode", "std", "variance", "min", "q1", "q3",
    "max", "range", "iqr", "cv", "sem", "skewness", "kurtosis", "shapiro",
}


def test_summary_and_frequencies(client, dataset_id):
    response = client.post("/api/descriptive/summary", json={"dataset_id": dataset_id, "columns": ["value", "group"]})
    assert response.status_code == 200
    results = response.get_json()["results"]
    assert SUMMARY_FIELDS == set(results["value"])
    assert results["value"]["count"] == 8
    assert results["value"]["mean"] == 15.25
    assert results["group"]["mean"] is None
    assert results["value"]["shapiro"]["p_value"] is not None

    response = client.post("/api/descriptive/frequencies", json={"dataset_id": dataset_id, "column": "group"})
    frequencies = response.get_json()["results"]
    assert {item["value"]: item["count"] for item in frequencies} == {"A": 4, "B": 4}


def test_chart_data(client, dataset_id):
    histogram = client.post("/api/charts/histogram", json={"dataset_id": dataset_id, "column": "value", "bins": 4})
    assert histogram.status_code == 200
    assert sum(histogram.get_json()["results"]["counts"]) == 8

    boxplot = client.post("/api/charts/boxplot", json={"dataset_id": dataset_id, "column": "value", "group_by": "group"})
    assert boxplot.status_code == 200
    assert len(boxplot.get_json()["results"]["groups"]) == 2

    scatter = client.post("/api/charts/scatter", json={"dataset_id": dataset_id, "x": "x", "y": "y"})
    assert scatter.status_code == 200
    assert scatter.get_json()["results"]["count"] == 8
    assert len(scatter.get_json()["results"]["regression_line"]) == 2
