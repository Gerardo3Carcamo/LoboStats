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


async def test_rejects_non_uadec_email(client: AsyncClient) -> None:
    response = await client.get("/api/v1/datasets", headers={"X-User-Email": "user@example.com"})
    assert response.status_code == 403
