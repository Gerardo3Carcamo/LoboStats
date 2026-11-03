from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.routes.datasets import owned_dataset
from app.core.database import get_db
from app.core.security import get_current_email
from app.models.analysis_run import AnalysisRun
from app.models.dataset import Dataset
from app.schemas.analysis import (
    AnalysisRunRead,
    AnalysisRunSummary,
    DescriptiveAnalysisRequest,
)
from app.services.dataset_content import dataset_content_sha256
from app.services.descriptive_service import (
    ENGINE_VERSION,
    DescriptiveAnalysisError,
    analyze_descriptive,
)

router = APIRouter(tags=["analyses"])
Database = Annotated[AsyncSession, Depends(get_db)]
CurrentEmail = Annotated[str, Depends(get_current_email)]


def serialize_run(run: AnalysisRun, dataset: Dataset) -> AnalysisRunRead:
    return AnalysisRunRead(
        id=run.id,
        dataset_id=run.dataset_id,
        dataset_revision=run.dataset_revision,
        dataset_sha256=run.dataset_sha256,
        analysis_type=run.analysis_type,
        status=run.status,
        engine_version=run.engine_version,
        parameters=run.parameters_data,
        result=run.result_data,
        warnings=run.warnings_data,
        created_at=run.created_at,
        is_stale=(
            run.dataset_revision != dataset.revision or run.dataset_sha256 != dataset.content_sha256
        ),
    )


@router.post(
    "/datasets/{dataset_id}/analyses/descriptive",
    response_model=AnalysisRunRead,
    status_code=status.HTTP_201_CREATED,
)
async def create_descriptive_analysis(
    dataset_id: int,
    payload: DescriptiveAnalysisRequest,
    db: Database,
    email: CurrentEmail,
) -> AnalysisRunRead:
    dataset = await owned_dataset(db, dataset_id, email)
    if not dataset.content_sha256:
        dataset.content_sha256 = dataset_content_sha256(dataset.columns_data, dataset.rows_data)
    try:
        result, warnings = analyze_descriptive(dataset, payload)
    except DescriptiveAnalysisError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(error)
        ) from error

    run = AnalysisRun(
        dataset_id=dataset.id,
        owner_email=email,
        dataset_revision=dataset.revision,
        dataset_sha256=dataset.content_sha256,
        analysis_type="descriptive",
        status="completed",
        engine_version=ENGINE_VERSION,
        parameters_data=payload.model_dump(),
        result_data=result,
        warnings_data=warnings,
    )
    db.add(run)
    await db.commit()
    await db.refresh(run)
    return serialize_run(run, dataset)


@router.get("/datasets/{dataset_id}/analyses", response_model=list[AnalysisRunSummary])
async def list_dataset_analyses(
    dataset_id: int, db: Database, email: CurrentEmail
) -> list[AnalysisRunSummary]:
    dataset = await owned_dataset(db, dataset_id, email)
    runs = (
        await db.scalars(
            select(AnalysisRun)
            .where(
                AnalysisRun.dataset_id == dataset_id,
                AnalysisRun.owner_email == email,
            )
            .order_by(AnalysisRun.created_at.desc())
        )
    ).all()
    return [
        AnalysisRunSummary(
            id=run.id,
            dataset_id=run.dataset_id,
            dataset_revision=run.dataset_revision,
            analysis_type=run.analysis_type,
            status=run.status,
            created_at=run.created_at,
            is_stale=(
                run.dataset_revision != dataset.revision
                or run.dataset_sha256 != dataset.content_sha256
            ),
        )
        for run in runs
    ]


@router.get("/analyses/{analysis_id}", response_model=AnalysisRunRead)
async def get_analysis(analysis_id: int, db: Database, email: CurrentEmail) -> AnalysisRunRead:
    run = await db.scalar(
        select(AnalysisRun).where(AnalysisRun.id == analysis_id, AnalysisRun.owner_email == email)
    )
    if run is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Análisis no encontrado")
    dataset = await owned_dataset(db, run.dataset_id, email)
    return serialize_run(run, dataset)
