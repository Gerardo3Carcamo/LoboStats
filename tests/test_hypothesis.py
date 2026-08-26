def assert_valid_test(response):
    assert response.status_code == 200, response.get_json()
    body = response.get_json()
    assert body["results"]["p_value"] is not None
    assert body["interpretation"]
    assert body["alpha"] == 0.05
    return body


def test_all_t_test_variants(client, dataset_id):
    one_sample = client.post("/api/hypothesis/t-test", json={
        "dataset_id": dataset_id, "type": "one-sample", "column": "value", "popmean": 10,
    })
    assert assert_valid_test(one_sample)["results"]["degrees_of_freedom"] == 7

    independent = client.post("/api/hypothesis/t-test", json={
        "dataset_id": dataset_id, "type": "independent", "value_column": "value",
        "group_column": "group", "groups": ["A", "B"],
    })
    assert assert_valid_test(independent)["results"]["equal_variance"] is False

    paired = client.post("/api/hypothesis/t-test", json={
        "dataset_id": dataset_id, "type": "paired", "column1": "before", "column2": "after",
    })
    assert assert_valid_test(paired)["results"]["pair_count"] == 8


def test_anova(client, dataset_id):
    response = client.post("/api/hypothesis/anova", json={
        "dataset_id": dataset_id, "value_column": "value", "group_column": "group",
    })
    body = assert_valid_test(response)
    assert body["test"] == "anova_one_way"
    assert len(body["results"]["groups"]) == 2


def test_chi_square(client, dataset_id):
    response = client.post("/api/hypothesis/chi-square", json={
        "dataset_id": dataset_id, "row_column": "row_cat", "column_column": "col_cat",
    })
    body = assert_valid_test(response)
    assert body["results"]["degrees_of_freedom"] == 1


def test_correlation(client, dataset_id):
    response = client.post("/api/hypothesis/correlation", json={
        "dataset_id": dataset_id, "x": "x", "y": "y", "alpha": 0.05,
    })
    body = assert_valid_test(response)
    assert body["results"]["coefficient"] > 0.99


def test_t_test_rejects_degenerate_sample(client):
    created = client.post("/api/data/manual", json={
        "name": "constant", "columns": ["value"], "rows": [[1], [1], [1]],
    })
    response = client.post("/api/hypothesis/t-test", json={
        "dataset_id": created.get_json()["dataset_id"], "type": "one-sample", "column": "value", "popmean": 1,
    })
    assert response.status_code == 400
    assert response.get_json()["error"]["code"] == "validation_error"
