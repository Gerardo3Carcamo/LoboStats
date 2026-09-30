"""Create datasets table.

Revision ID: 20260825_01
Revises:
"""

import sqlalchemy as sa

from alembic import op

revision = "20260825_01"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "datasets",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("owner_email", sa.String(length=254), nullable=False),
        sa.Column("name", sa.String(length=120), nullable=False),
        sa.Column("columns_json", sa.JSON(), nullable=False),
        sa.Column("data_json", sa.JSON(), nullable=False),
        sa.Column("row_count", sa.Integer(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_datasets_owner_email", "datasets", ["owner_email"])
    op.create_index("ix_datasets_owner_updated", "datasets", ["owner_email", "updated_at"])


def downgrade() -> None:
    op.drop_index("ix_datasets_owner_updated", table_name="datasets")
    op.drop_index("ix_datasets_owner_email", table_name="datasets")
    op.drop_table("datasets")
