from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.routes.analyses import serialize_run
from app.api.routes.datasets import owned_dataset
from app.core.database import get_db
from app.core.security import get_current_email
from app.models.analysis_run import AnalysisRun
from app.schemas.analysis import AnalysisRunRead
from app.schemas.regression import RegressionAnalysisRequest
from app.services.dataset_content import dataset_content_sha256
from app.services.regression_service import (
    ENGINE_VERSION,
    RegressionAnalysisError,
    analyze_regression,
    regression_catalog,
)

router = APIRouter(tags=["regression"])
Database = Annotated[AsyncSession, Depends(get_db)]
CurrentEmail = Annotated[str, Depends(get_current_email)]


@router.get("/regression/models", response_model=list[dict[str, Any]])
async def list_regression_models() -> list[dict[str, Any]]:
    return regression_catalog()


@router.post(
    "/datasets/{dataset_id}/analyses/regression",
    response_model=AnalysisRunRead,
    status_code=status.HTTP_201_CREATED,
)
async def create_regression_analysis(
    dataset_id: int,
    payload: RegressionAnalysisRequest,
    db: Database,
    email: CurrentEmail,
) -> AnalysisRunRead:
    dataset = await owned_dataset(db, dataset_id, email)
    if not dataset.content_sha256:
        dataset.content_sha256 = dataset_content_sha256(dataset.columns_data, dataset.rows_data)
    try:
        result, warnings = analyze_regression(dataset, payload)
    except RegressionAnalysisError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(error),
        ) from error

    run = AnalysisRun(
        dataset_id=dataset.id,
        owner_email=email,
        dataset_revision=dataset.revision,
        dataset_sha256=dataset.content_sha256,
        analysis_type="regression",
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
