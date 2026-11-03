"""Add dataset revisions and persisted analysis runs.

Revision ID: 20261102_01
Revises: 20260825_01
"""

import sqlalchemy as sa

from alembic import op

revision = "20261102_01"
down_revision = "20260825_01"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "datasets",
        sa.Column("revision", sa.Integer(), server_default="1", nullable=False),
    )
    op.add_column(
        "datasets",
        sa.Column("content_sha256", sa.String(length=64), server_default="", nullable=False),
    )
    op.create_table(
        "analysis_runs",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("dataset_id", sa.Integer(), nullable=False),
        sa.Column("owner_email", sa.String(length=254), nullable=False),
        sa.Column("dataset_revision", sa.Integer(), nullable=False),
        sa.Column("dataset_sha256", sa.String(length=64), nullable=False),
        sa.Column("analysis_type", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=24), nullable=False),
        sa.Column("engine_version", sa.String(length=32), nullable=False),
        sa.Column("parameters_json", sa.JSON(), nullable=False),
        sa.Column("result_json", sa.JSON(), nullable=False),
        sa.Column("warnings_json", sa.JSON(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["dataset_id"], ["datasets.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_analysis_runs_dataset_id", "analysis_runs", ["dataset_id"])
    op.create_index("ix_analysis_runs_owner_email", "analysis_runs", ["owner_email"])
    op.create_index(
        "ix_analysis_runs_dataset_created", "analysis_runs", ["dataset_id", "created_at"]
    )
    op.create_index(
        "ix_analysis_runs_owner_created", "analysis_runs", ["owner_email", "created_at"]
    )


def downgrade() -> None:
    op.drop_index("ix_analysis_runs_owner_created", table_name="analysis_runs")
    op.drop_index("ix_analysis_runs_dataset_created", table_name="analysis_runs")
    op.drop_index("ix_analysis_runs_owner_email", table_name="analysis_runs")
    op.drop_index("ix_analysis_runs_dataset_id", table_name="analysis_runs")
    op.drop_table("analysis_runs")
    op.drop_column("datasets", "content_sha256")
    op.drop_column("datasets", "revision")
