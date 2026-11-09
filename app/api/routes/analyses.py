from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.routes.datasets import owned_dataset
from app.core.database import get_db
from app.core.security import get_current_email
from app.models.analysis_run import AnalysisRun
from app.models.dataset import Dataset
from app.schemas.analysis import (
    AnalysisRunPresentation,
    AnalysisRunRead,
    AnalysisRunSummary,
    DescriptiveAnalysisRequest,
    ProbabilityAnalysisRequest,
)
from app.services.dataset_content import dataset_content_sha256
from app.services.descriptive_service import (
    ENGINE_VERSION,
    DescriptiveAnalysisError,
    analyze_descriptive,
)
from app.services.probability_catalog import probability_catalog
from app.services.probability_service import (
    ENGINE_VERSION as PROBABILITY_ENGINE_VERSION,
)
from app.services.probability_service import (
    ProbabilityAnalysisError,
    analyze_probability,
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
        is_stale=run_is_stale(run, dataset),
    )


def run_is_stale(run: AnalysisRun, dataset: Dataset) -> bool:
    if run.analysis_type == "probability":
        return False
    return run.dataset_revision != dataset.revision or run.dataset_sha256 != dataset.content_sha256


@router.get("/probability/distributions", response_model=list[dict[str, Any]])
async def list_probability_distributions() -> list[dict[str, Any]]:
    return probability_catalog()


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


@router.post(
    "/datasets/{dataset_id}/analyses/probability",
    response_model=AnalysisRunRead,
    status_code=status.HTTP_201_CREATED,
)
async def create_probability_analysis(
    dataset_id: int,
    payload: ProbabilityAnalysisRequest,
    db: Database,
    email: CurrentEmail,
) -> AnalysisRunRead:
    dataset = await owned_dataset(db, dataset_id, email)
    if not dataset.content_sha256:
        dataset.content_sha256 = dataset_content_sha256(dataset.columns_data, dataset.rows_data)
    try:
        result, warnings = analyze_probability(payload)
    except ProbabilityAnalysisError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(error)
        ) from error

    run = AnalysisRun(
        dataset_id=dataset.id,
        owner_email=email,
        dataset_revision=dataset.revision,
        dataset_sha256=dataset.content_sha256,
        analysis_type="probability",
        status="completed",
        engine_version=PROBABILITY_ENGINE_VERSION,
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
    dataset_id: int,
    db: Database,
    email: CurrentEmail,
    analysis_type: str | None = Query(default=None, max_length=64),
    limit: int = Query(default=50, ge=1, le=100),
    before_id: int | None = Query(default=None, ge=1),
) -> list[AnalysisRunSummary]:
    dataset = await owned_dataset(db, dataset_id, email)
    statement = select(AnalysisRun).where(
        AnalysisRun.dataset_id == dataset_id,
        AnalysisRun.owner_email == email,
    )
    if analysis_type:
        statement = statement.where(AnalysisRun.analysis_type == analysis_type)
    if before_id:
        statement = statement.where(AnalysisRun.id < before_id)
    runs = (
        await db.scalars(
            statement.order_by(AnalysisRun.created_at.desc(), AnalysisRun.id.desc()).limit(limit)
        )
    ).all()
    return [serialize_summary(run, dataset) for run in runs]


def serialize_summary(run: AnalysisRun, dataset: Dataset) -> AnalysisRunSummary:
    result = run.result_data or {}
    distribution_data = result.get("distribution") or {}
    input_data = result.get("input") or {}
    preview = result.get("result_value", result.get("probability"))
    return AnalysisRunSummary(
        id=run.id,
        dataset_id=run.dataset_id,
        dataset_revision=run.dataset_revision,
        analysis_type=run.analysis_type,
        status=run.status,
        created_at=run.created_at,
        is_stale=run_is_stale(run, dataset),
        engine_version=run.engine_version,
        distribution=distribution_data.get("label"),
        mode=result.get("mode", "event") if run.analysis_type == "probability" else None,
        formula=result.get("calculation_formula", result.get("event_formula")),
        result_preview=float(preview) if preview is not None else None,
        input_kind=input_data.get("kind"),
        presentation=summary_presentation(run),
    )


def summary_presentation(run: AnalysisRun) -> AnalysisRunPresentation:
    if run.analysis_type == "probability":
        return probability_presentation(run)
    if run.analysis_type == "descriptive":
        return descriptive_presentation(run)
    return AnalysisRunPresentation(
        module=run.analysis_type.replace("_", " ").title(),
        title=f"Análisis #{run.id}",
        detail=f"Revisión {run.dataset_revision}",
        result=run.status,
        filter_key=run.analysis_type,
        filter_label=run.analysis_type.replace("_", " ").title(),
        tags=[run.engine_version],
    )


def probability_presentation(run: AnalysisRun) -> AnalysisRunPresentation:
    result = run.result_data or {}
    distribution = result.get("distribution") or {}
    mode = result.get("mode", "event")
    mode_labels = {"pdf": "PDF / PMF", "cdf": "CDF", "inverse": "Inversa", "event": "Evento"}
    preview = result.get("result_value", result.get("probability"))
    percentage = mode in {"event", "cdf"} or (
        mode == "pdf" and distribution.get("kind") == "discrete"
    )
    if preview is None:
        preview_text = "Sin vista previa"
    elif percentage:
        preview_text = f"{float(preview) * 100:.2f} %"
    else:
        preview_text = f"{float(preview):.6g}"
    input_data = result.get("input") or {}
    tags = [run.engine_version]
    if input_data.get("kind") == "column":
        tags.insert(0, "Por columna")
    return AnalysisRunPresentation(
        module="Probabilidad",
        title=f"{distribution.get('label', 'Probabilidad')} · {mode_labels.get(mode, mode)}",
        detail=result.get("calculation_formula", result.get("event_formula", f"Cálculo #{run.id}")),
        result=preview_text,
        filter_key=mode,
        filter_label=mode_labels.get(mode, mode.title()),
        tags=tags,
    )


def descriptive_presentation(run: AnalysisRun) -> AnalysisRunPresentation:
    result = run.result_data or {}
    selection = result.get("selection") or {}
    dataset = result.get("dataset") or {}
    groups = result.get("groups") or []
    first_group = groups[0] if groups else {}
    columns = first_group.get("columns") or []
    names = [column.get("name", column.get("key", "Variable")) for column in columns]
    detail = ", ".join(names[:3])
    if len(names) > 3:
        detail += f" y {len(names) - 3} más"
    group_by = selection.get("group_by")
    tags = [f"Revisión {run.dataset_revision}", run.engine_version]
    if group_by:
        tags.insert(0, f"Agrupado por {group_by}")
    row_count = dataset.get("row_count", first_group.get("row_count", 0))
    return AnalysisRunPresentation(
        module="Estadística descriptiva",
        title=dataset.get("name", f"Análisis #{run.id}"),
        detail=detail or "Resumen de variables",
        result=f"{len(names)} variables · {row_count} observaciones",
        filter_key="grouped" if group_by else "general",
        filter_label="Agrupado" if group_by else "General",
        tags=tags,
    )


@router.get("/analyses/{analysis_id}", response_model=AnalysisRunRead)
async def get_analysis(analysis_id: int, db: Database, email: CurrentEmail) -> AnalysisRunRead:
    run = await db.scalar(
        select(AnalysisRun).where(AnalysisRun.id == analysis_id, AnalysisRun.owner_email == email)
    )
    if run is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Análisis no encontrado")
    dataset = await owned_dataset(db, run.dataset_id, email)
    return serialize_run(run, dataset)
