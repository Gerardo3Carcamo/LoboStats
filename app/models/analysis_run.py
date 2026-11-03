from datetime import datetime
from typing import Any

from sqlalchemy import JSON, DateTime, ForeignKey, Index, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class AnalysisRun(Base):
    __tablename__ = "analysis_runs"
    __table_args__ = (
        Index("ix_analysis_runs_dataset_created", "dataset_id", "created_at"),
        Index("ix_analysis_runs_owner_created", "owner_email", "created_at"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    dataset_id: Mapped[int] = mapped_column(
        ForeignKey("datasets.id", ondelete="CASCADE"), index=True, nullable=False
    )
    owner_email: Mapped[str] = mapped_column(String(254), index=True, nullable=False)
    dataset_revision: Mapped[int] = mapped_column(Integer, nullable=False)
    dataset_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    analysis_type: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(24), default="completed", nullable=False)
    engine_version: Mapped[str] = mapped_column(String(32), nullable=False)
    parameters_data: Mapped[dict[str, Any]] = mapped_column("parameters_json", JSON)
    result_data: Mapped[dict[str, Any]] = mapped_column("result_json", JSON)
    warnings_data: Mapped[list[str]] = mapped_column("warnings_json", JSON)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
