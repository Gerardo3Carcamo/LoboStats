from fastapi import APIRouter

from app.api.routes import analyses, datasets, hypothesis, system

api_router = APIRouter()
api_router.include_router(system.router)
api_router.include_router(datasets.router)
api_router.include_router(analyses.router)
api_router.include_router(hypothesis.router)
