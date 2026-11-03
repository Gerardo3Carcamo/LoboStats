from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.security import get_current_email
from app.models.dataset import Dataset
from app.schemas.dataset import DatasetCreate, DatasetRead, DatasetSummary, DatasetUpdate
from app.services.dataset_content import dataset_content_sha256

router = APIRouter(prefix="/datasets", tags=["datasets"])
Database = Annotated[AsyncSession, Depends(get_db)]
CurrentEmail = Annotated[str, Depends(get_current_email)]


def serialize_dataset(dataset: Dataset) -> DatasetRead:
    content_sha256 = dataset.content_sha256 or dataset_content_sha256(
        dataset.columns_data, dataset.rows_data
    )
    return DatasetRead(
        id=dataset.id,
        owner_email=dataset.owner_email,
        name=dataset.name,
        columns=dataset.columns_data,
        rows=dataset.rows_data,
        row_count=dataset.row_count,
        revision=dataset.revision,
        content_sha256=content_sha256,
        created_at=dataset.created_at,
        updated_at=dataset.updated_at,
    )


async def owned_dataset(db: AsyncSession, dataset_id: int, email: str) -> Dataset:
    dataset = await db.scalar(
        select(Dataset).where(Dataset.id == dataset_id, Dataset.owner_email == email)
    )
    if dataset is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset no encontrado")
    return dataset


@router.get("", response_model=list[DatasetSummary])
async def list_datasets(db: Database, email: CurrentEmail) -> list[DatasetSummary]:
    datasets = (
        await db.scalars(
            select(Dataset).where(Dataset.owner_email == email).order_by(Dataset.updated_at.desc())
        )
    ).all()
    return [
        DatasetSummary(
            id=item.id,
            name=item.name,
            row_count=item.row_count,
            column_count=len(item.columns_data),
            updated_at=item.updated_at,
        )
        for item in datasets
    ]


@router.post("", response_model=DatasetRead, status_code=status.HTTP_201_CREATED)
async def create_dataset(payload: DatasetCreate, db: Database, email: CurrentEmail) -> DatasetRead:
    columns = [column.model_dump() for column in payload.columns]
    dataset = Dataset(
        owner_email=email,
        name=payload.name,
        columns_data=columns,
        rows_data=payload.rows,
        row_count=len(payload.rows),
        revision=1,
        content_sha256=dataset_content_sha256(columns, payload.rows),
    )
    db.add(dataset)
    await db.commit()
    await db.refresh(dataset)
    return serialize_dataset(dataset)


@router.get("/{dataset_id}", response_model=DatasetRead)
async def get_dataset(dataset_id: int, db: Database, email: CurrentEmail) -> DatasetRead:
    return serialize_dataset(await owned_dataset(db, dataset_id, email))


@router.put("/{dataset_id}", response_model=DatasetRead)
async def update_dataset(
    dataset_id: int, payload: DatasetUpdate, db: Database, email: CurrentEmail
) -> DatasetRead:
    dataset = await owned_dataset(db, dataset_id, email)
    dataset.name = payload.name
    columns = [column.model_dump() for column in payload.columns]
    dataset.columns_data = columns
    dataset.rows_data = payload.rows
    dataset.row_count = len(payload.rows)
    dataset.revision += 1
    dataset.content_sha256 = dataset_content_sha256(columns, payload.rows)
    await db.commit()
    await db.refresh(dataset)
    return serialize_dataset(dataset)


@router.delete("/{dataset_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_dataset(dataset_id: int, db: Database, email: CurrentEmail) -> Response:
    dataset = await owned_dataset(db, dataset_id, email)
    await db.delete(dataset)
    await db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)
