import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.core.database import Base, get_db
from app.main import app


@pytest.fixture
async def client(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'test.db'}")
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)

    async def override_db():
        async with session_factory() as session:
            yield session

    app.dependency_overrides[get_db] = override_db
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as http:
        yield http
    app.dependency_overrides.clear()
    await engine.dispose()


async def test_health(client: AsyncClient) -> None:
    response = await client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


async def test_probability_catalog(client: AsyncClient) -> None:
    response = await client.get("/api/v1/probability/distributions")

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 24
    assert {item["key"] for item in body} >= {
        "geometric",
        "negative_binomial",
        "hypergeometric",
        "exponential",
        "student_t",
        "chi_square",
        "f",
        "uniform",
        "integer_uniform",
        "custom_discrete",
        "beta",
        "gamma",
        "lognormal",
        "weibull",
        "triangular",
    }


async def test_dataset_lifecycle(client: AsyncClient) -> None:
    payload = {
        "name": "Calificaciones grupo A",
        "columns": [{"key": "c1", "name": "Calificacion", "data_type": "number"}],
        "rows": [{"c1": 86}, {"c1": 92}],
    }
    created = await client.post("/api/v1/datasets", json=payload)
    assert created.status_code == 201
    assert created.json()["row_count"] == 2
    assert created.json()["revision"] == 1
    assert len(created.json()["content_sha256"]) == 64

    listed = await client.get("/api/v1/datasets")
    assert listed.status_code == 200
    assert listed.json()[0]["column_count"] == 1

    dataset_id = created.json()["id"]
    deleted = await client.delete(f"/api/v1/datasets/{dataset_id}")
    assert deleted.status_code == 204


async def test_descriptive_analysis_lifecycle(client: AsyncClient) -> None:
    dataset = await client.post(
        "/api/v1/datasets",
        json={
            "name": "Mediciones",
            "columns": [
                {"key": "value", "name": "Valor", "data_type": "number"},
                {"key": "group", "name": "Grupo", "data_type": "text"},
            ],
            "rows": [
                {"value": 1, "group": "A"},
                {"value": 2, "group": "A"},
                {"value": 3, "group": "B"},
                {"value": 4, "group": "B"},
                {"value": None, "group": "B"},
            ],
        },
    )
    dataset_id = dataset.json()["id"]

    response = await client.post(
        f"/api/v1/datasets/{dataset_id}/analyses/descriptive",
        json={
            "columns": ["value"],
            "group_by": "group",
            "confidence_level": 0.95,
            "percentiles": [0.25, 0.5, 0.75],
            "histogram_bins": 4,
        },
    )
    assert response.status_code == 201
    body = response.json()
    assert body["analysis_type"] == "descriptive"
    assert body["is_stale"] is False
    assert [group["label"] for group in body["result"]["groups"]] == ["A", "B"]
    group_b = body["result"]["groups"][1]["columns"][0]
    assert group_b["valid_count"] == 2
    assert group_b["missing_count"] == 1
    assert group_b["numeric"]["mean"] == 3.5

    listed = await client.get(f"/api/v1/datasets/{dataset_id}/analyses")
    assert listed.status_code == 200
    assert listed.json()[0]["id"] == body["id"]
    assert listed.json()[0]["presentation"]["module"] == "Estadística descriptiva"
    assert listed.json()[0]["presentation"]["result"] == "1 variables · 5 observaciones"

    updated_payload = {
        "name": "Mediciones actualizadas",
        "columns": dataset.json()["columns"],
        "rows": [*dataset.json()["rows"], {"value": 5, "group": "B"}],
    }
    updated = await client.put(f"/api/v1/datasets/{dataset_id}", json=updated_payload)
    assert updated.status_code == 200
    assert updated.json()["revision"] == 2

    previous = await client.get(f"/api/v1/analyses/{body['id']}")
    assert previous.status_code == 200
    assert previous.json()["is_stale"] is True


async def test_probability_analysis_lifecycle(client: AsyncClient) -> None:
    dataset = await client.post(
        "/api/v1/datasets",
        json={
            "name": "Espacio de probabilidad",
            "columns": [{"key": "value", "name": "Valor", "data_type": "number"}],
            "rows": [],
        },
    )
    dataset_id = dataset.json()["id"]

    response = await client.post(
        f"/api/v1/datasets/{dataset_id}/analyses/probability",
        json={
            "distribution": "binomial",
            "operation": "at_least",
            "trials": 12,
            "success_probability": 0.4,
            "lower": 6,
        },
    )

    assert response.status_code == 201
    body = response.json()
    assert body["analysis_type"] == "probability"
    assert body["result"]["probability"] == pytest.approx(0.3347914424)
    assert body["result"]["distribution"]["label"] == "Binomial"
    assert body["is_stale"] is False

    listed = await client.get(
        f"/api/v1/datasets/{dataset_id}/analyses",
        params={"analysis_type": "probability", "limit": 1},
    )
    assert listed.status_code == 200
    summary = listed.json()[0]
    assert summary["distribution"] == "Binomial"
    assert summary["mode"] == "event"
    assert summary["formula"] == "P(X ≥ 6)"
    assert summary["result_preview"] == pytest.approx(body["result"]["probability"])
    assert summary["engine_version"] == "probability-4.0"
    assert summary["presentation"]["title"] == "Binomial · Evento"
    assert summary["presentation"]["result"] == f"{body['result']['probability'] * 100:.2f} %"

    updated = await client.put(
        f"/api/v1/datasets/{dataset_id}",
        json={
            "name": "Espacio actualizado",
            "columns": dataset.json()["columns"],
            "rows": [{"value": 1}],
        },
    )
    assert updated.status_code == 200

    previous = await client.get(f"/api/v1/analyses/{body['id']}")
    assert previous.status_code == 200
    assert previous.json()["is_stale"] is False


async def test_hypothesis_analysis_lifecycle(client: AsyncClient) -> None:
    dataset = await client.post(
        "/api/v1/datasets",
        json={
            "name": "Contraste de medias",
            "columns": [{"key": "value", "name": "Valor", "data_type": "number"}],
            "rows": [{"value": value} for value in [18, 19, 20, 21, 22, 24]],
        },
    )
    dataset_id = dataset.json()["id"]

    catalog = await client.get("/api/v1/hypothesis/tests")
    assert catalog.status_code == 200
    assert {item["key"] for item in catalog.json()} >= {"one_sample_t", "mann_whitney"}

    response = await client.post(
        f"/api/v1/datasets/{dataset_id}/analyses/hypothesis",
        json={
            "test": "one_sample_t",
            "column": "value",
            "hypothesized_value": 20,
            "alternative": "two_sided",
            "confidence_level": 0.95,
        },
    )
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["analysis_type"] == "hypothesis"
    assert body["result"]["test"]["label"] == "t de una muestra"
    assert body["result"]["result"]["p_value"] == pytest.approx(0.48376, rel=1e-3)

    listed = await client.get(
        f"/api/v1/datasets/{dataset_id}/analyses",
        params={"analysis_type": "hypothesis"},
    )
    assert listed.status_code == 200
    assert listed.json()[0]["presentation"]["module"] == "Pruebas de hipótesis"
    assert listed.json()[0]["presentation"]["filter_key"] == "one_sample_t"


async def test_rejects_non_uadec_email(client: AsyncClient) -> None:
    response = await client.get("/api/v1/datasets", headers={"X-User-Email": "user@example.com"})
    assert response.status_code == 403
