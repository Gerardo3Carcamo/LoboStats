import pytest


def request(client, route, dataset_id, **extra):
    return client.post(f"/api/anova/{route}", json={"dataset_id": dataset_id, **extra})


def test_one_and_two_way_anova(client, dataset_id):
    one = request(client, "one-way", dataset_id, value_column="value", group_column="group").get_json()["results"]
    assert 0 < one["eta_squared"] < 1
    assert {row["term"] for row in one["table"]} == {"C(group)", "Residual"}
    two = request(client, "two-way", dataset_id, value_column="value", factor_a="row_cat", factor_b="col_cat", typ=2)
    assert two.status_code == 200, two.get_json()
    assert any(":" in row["term"] for row in two.get_json()["results"]["table"])


@pytest.mark.parametrize("method", ["tukey", "bonferroni", "scheffe"])
def test_post_hoc_methods(client, dataset_id, method):
    response = request(client, "post-hoc", dataset_id, value_column="value", group_column="row_cat", method=method)
    assert response.status_code == 200, response.get_json()
    comparison = response.get_json()["results"]["comparisons"][0]
    assert comparison["p_value"] is not None
    if method == "scheffe": assert comparison["lower"] < comparison["upper"]


def test_repeated_main_and_interaction(client, dataset_id):
    repeated = request(client, "repeated-measures", dataset_id, subject="subject", within="time", value="value")
    assert repeated.status_code == 200, repeated.get_json()
    assert repeated.get_json()["results"]["subjects"] == 4
    payload = {"value_column": "value", "factor_a": "row_cat", "factor_b": "col_cat"}
    main = request(client, "main-effects", dataset_id, **payload).get_json()["results"]
    interaction = request(client, "interaction", dataset_id, **payload).get_json()["results"]
    assert len(main["main_effects"]) == 2
    assert interaction["interaction_effect"] is not None


def test_anova_rejects_invalid_type(client, dataset_id):
    response = request(client, "two-way", dataset_id, value_column="value", factor_a="row_cat", factor_b="col_cat", typ=4)
    assert response.status_code == 400
