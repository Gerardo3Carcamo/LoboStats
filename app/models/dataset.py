from datetime import datetime
from typing import Any

from sqlalchemy import JSON, DateTime, Index, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class Dataset(Base):
    __tablename__ = "datasets"
    __table_args__ = (Index("ix_datasets_owner_updated", "owner_email", "updated_at"),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    owner_email: Mapped[str] = mapped_column(String(254), index=True)
    name: Mapped[str] = mapped_column(String(120))
    columns_data: Mapped[list[dict[str, Any]]] = mapped_column("columns_json", JSON)
    rows_data: Mapped[list[dict[str, Any]]] = mapped_column("data_json", JSON)
    row_count: Mapped[int] = mapped_column(default=0)
    revision: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    content_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )
