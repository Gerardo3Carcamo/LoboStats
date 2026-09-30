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

    listed = await client.get("/api/v1/datasets")
    assert listed.status_code == 200
    assert listed.json()[0]["column_count"] == 1

    dataset_id = created.json()["id"]
    deleted = await client.delete(f"/api/v1/datasets/{dataset_id}")
    assert deleted.status_code == 204


async def test_rejects_non_uadec_email(client: AsyncClient) -> None:
    response = await client.get("/api/v1/datasets", headers={"X-User-Email": "user@example.com"})
    assert response.status_code == 403
