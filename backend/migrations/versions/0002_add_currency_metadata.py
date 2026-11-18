"""add currency metadata to claims

Revision ID: 0002_add_currency_metadata
Revises: 0001_initial_schema
Create Date: 2026-11-17 22:35:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '0002_add_currency_metadata'
down_revision: Union[str, None] = '0001_initial_schema'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Backward-compatible addition of currency tracking columns to claims table
    op.add_column('claims', sa.Column('currency', sa.String(length=10), server_default='USD', nullable=True))
    op.add_column('claims', sa.Column('original_reimbursement', sa.Float(), nullable=True))
    op.add_column('claims', sa.Column('original_deductible', sa.Float(), nullable=True))


def downgrade() -> None:
    op.drop_column('claims', 'original_deductible')
    op.drop_column('claims', 'original_reimbursement')
    op.drop_column('claims', 'currency')
