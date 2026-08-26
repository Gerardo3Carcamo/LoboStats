import pytest


@pytest.mark.parametrize("route,payload,key", [
    ("bar", {"category": "group", "value": "value"}, "series"),
    ("pie", {"category": "group"}, "series"),
    ("pareto", {"category": "group"}, "series"),
    ("qq", {"column": "value"}, "points"),
    ("violin", {"column": "value", "group_by": "group"}, "groups"),
    ("dot", {"column": "value"}, "points"),
    ("line", {"x": "x", "y": "y"}, "points"),
    ("area", {"x": "x", "y": "y"}, "points"),
    ("interval", {"value": "value", "group": "group"}, "intervals"),
    ("pairplot", {"columns": ["x", "y", "value"]}, "rows"),
    ("bubble", {"x": "x", "y": "y", "size": "value", "color": "group"}, "points"),
    ("heatmap", {"columns": ["x", "y", "value"]}, "matrix"),
    ("scatter-3d", {"x": "x", "y": "y", "z": "value"}, "points"),
])
def test_advanced_chart(client, dataset_id, route, payload, key):
    response = client.post(f"/api/charts/{route}", json={"dataset_id": dataset_id, **payload})
    assert response.status_code == 200, response.get_json()
    assert response.get_json()["results"][key]


def test_histogram_density_kde_and_invalid_chart(client, dataset_id):
    response = client.post("/api/charts/histogram", json={"dataset_id": dataset_id, "column": "value", "density": True, "kde": True})
    assert response.status_code == 200 and len(response.get_json()["results"]["kde"]) >= 50
    invalid = client.post("/api/charts/heatmap", json={"dataset_id": dataset_id, "columns": ["x"], "method": "wrong"})
    assert invalid.status_code == 400
