"""answer modes: mode/label on messages, default mode and Ollama model on users

Revision ID: 0002
Revises: 0001
Create Date: 2026-11-10
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Apply this migration."""
    with op.batch_alter_table("users", schema=None) as batch_op:
        batch_op.add_column(sa.Column("preferred_mode", sa.String(length=16), nullable=False, server_default="auto"))
        batch_op.add_column(sa.Column("ollama_model", sa.String(length=120), nullable=True))
    with op.batch_alter_table("messages", schema=None) as batch_op:
        batch_op.add_column(sa.Column("mode", sa.String(length=16), nullable=True))
        batch_op.add_column(sa.Column("label", sa.String(length=160), nullable=True))


def downgrade() -> None:
    """Revert this migration."""
    with op.batch_alter_table("messages", schema=None) as batch_op:
        batch_op.drop_column("label")
        batch_op.drop_column("mode")
    with op.batch_alter_table("users", schema=None) as batch_op:
        batch_op.drop_column("ollama_model")
        batch_op.drop_column("preferred_mode")
