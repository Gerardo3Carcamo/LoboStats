import pytest

from api import create_app


@pytest.fixture()
def app():
    return create_app({"TESTING": True})


@pytest.fixture()
def client(app):
    return app.test_client()


@pytest.fixture()
def dataset_id(client):
    payload = {
        "name": "experimento",
        "columns": ["value", "group", "before", "after", "x", "y", "row_cat", "col_cat", "subject", "time"],
        "rows": [
            [10, "A", 8, 10, 1, 2.1, "R1", "C1", "S1", "pre"],
            [11, "A", 9, 10, 2, 4.0, "R1", "C1", "S1", "post"],
            [9, "A", 7, 9, 3, 6.2, "R1", "C2", "S2", "pre"],
            [10, "A", 10, 11, 4, 7.9, "R2", "C1", "S2", "post"],
            [20, "B", 17, 19, 5, 10.1, "R2", "C2", "S3", "pre"],
            [22, "B", 18, 21, 6, 12.2, "R2", "C2", "S3", "post"],
            [19, "B", 17, 20, 7, 13.8, "R2", "C1", "S4", "pre"],
            [21, "B", 19, 21, 8, 16.0, "R1", "C2", "S4", "post"],
        ],
    }
    response = client.post("/api/data/manual", json=payload)
    assert response.status_code == 201
    return response.get_json()["dataset_id"]
