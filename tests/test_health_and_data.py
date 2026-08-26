from io import BytesIO


def test_health(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.get_json()["status"] == "ok"


def test_manual_pagination_and_delete(client):
    created = client.post("/api/data/manual", json={
        "name": "manual", "columns": ["a", "b"], "rows": [[1, None], [2, "x"], [3, "y"]]
    })
    assert created.status_code == 201
    dataset_id = created.get_json()["dataset_id"]

    response = client.get(f"/api/data/{dataset_id}?page=2&page_size=2")
    body = response.get_json()
    assert response.status_code == 200
    assert body["data"] == [{"a": 3, "b": "y"}]
    assert body["pagination"]["total"] == 3

    assert client.delete(f"/api/data/{dataset_id}").status_code == 204
    missing = client.get(f"/api/data/{dataset_id}")
    assert missing.status_code == 404
    assert missing.get_json()["error"]["code"] == "dataset_not_found"


def test_csv_upload(client):
    response = client.post(
        "/api/data/upload",
        data={"file": (BytesIO(b"name,value\nA,1\nB,2\n"), "sample.csv")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 201
    body = response.get_json()
    assert body["row_count"] == 2
    assert body["id"] == body["dataset_id"]
    assert [column["name"] for column in body["columns"]] == ["name", "value"]
    assert body["columns"][1]["type"] == "numérico"
    assert body["preview"] == [{"name": "A", "value": 1}, {"name": "B", "value": 2}]


def test_uniform_validation_error(client):
    response = client.post("/api/data/manual", json={"name": "bad"})
    assert response.status_code == 400
    assert set(response.get_json()["error"]) == {"code", "message", "details"}
