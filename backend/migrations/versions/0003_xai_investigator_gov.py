"""xai investigator governance schema

Revision ID: 0003_xai_investigator_gov
Revises: 0002_add_currency_metadata
Create Date: 2026-11-19 18:30:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '0003_xai_investigator_gov'
down_revision: Union[str, None] = '0002_add_currency_metadata'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Safely create tables if they do not exist
    conn = op.get_bind()
    existing_tables = sa.inspect(conn).get_table_names()

    if 'investigator_reviews' not in existing_tables:
        op.create_table(
            'investigator_reviews',
            sa.Column('id', sa.Integer(), primary_key=True),
            sa.Column('claim_id', sa.String(), nullable=False),
            sa.Column('provider_id', sa.String(), nullable=False),
            sa.Column('status', sa.String(), nullable=False),
            sa.Column('priority', sa.String(), nullable=False),
            sa.Column('assigned_to_id', sa.Integer(), nullable=True),
            sa.Column('current_verdict', sa.String(), nullable=True),
            sa.Column('notes', sa.Text(), nullable=True),
            sa.Column('evidence_summary', sa.JSON(), nullable=True),
            sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
            sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False)
        )

    if 'investigator_review_events' not in existing_tables:
        op.create_table(
            'investigator_review_events',
            sa.Column('id', sa.Integer(), primary_key=True),
            sa.Column('review_id', sa.Integer(), nullable=False),
            sa.Column('user_id', sa.Integer(), nullable=True),
            sa.Column('event_type', sa.String(), nullable=False),
            sa.Column('previous_status', sa.String(), nullable=True),
            sa.Column('new_status', sa.String(), nullable=True),
            sa.Column('reason', sa.String(), nullable=True),
            sa.Column('comments', sa.Text(), nullable=True),
            sa.Column('evidence_references', sa.JSON(), nullable=True),
            sa.Column('created_at', sa.DateTime(timezone=True), nullable=False)
        )

    if 'model_monitoring_snapshots' not in existing_tables:
        op.create_table(
            'model_monitoring_snapshots',
            sa.Column('id', sa.Integer(), primary_key=True),
            sa.Column('model_version', sa.String(), nullable=False),
            sa.Column('assessment_level', sa.String(), nullable=False),
            sa.Column('snapshot_timestamp', sa.DateTime(timezone=True), nullable=False),
            sa.Column('prediction_count', sa.Integer(), nullable=False),
            sa.Column('risk_distribution', sa.JSON(), nullable=True),
            sa.Column('feature_drift_metrics', sa.JSON(), nullable=True),
            sa.Column('data_quality_metrics', sa.JSON(), nullable=True),
            sa.Column('drift_detected', sa.Boolean(), nullable=False),
            sa.Column('drift_alert_level', sa.String(), nullable=False),
            sa.Column('alert_summary', sa.Text(), nullable=True),
            sa.Column('created_at', sa.DateTime(timezone=True), nullable=False)
        )

    if 'model_registry' not in existing_tables:
        op.create_table(
            'model_registry',
            sa.Column('id', sa.Integer(), primary_key=True),
            sa.Column('model_id', sa.String(), nullable=False),
            sa.Column('model_name', sa.String(), nullable=False),
            sa.Column('assessment_level', sa.String(), nullable=False),
            sa.Column('version', sa.String(), nullable=False),
            sa.Column('status', sa.String(), nullable=False),
            sa.Column('model_type', sa.String(), nullable=False),
            sa.Column('manifest_path', sa.String(), nullable=True),
            sa.Column('artifact_hashes', sa.JSON(), nullable=True),
            sa.Column('validation_metrics', sa.JSON(), nullable=True),
            sa.Column('calibration_metrics', sa.JSON(), nullable=True),
            sa.Column('feature_specification', sa.JSON(), nullable=True),
            sa.Column('training_dataset_reference', sa.String(), nullable=True),
            sa.Column('approved_by_id', sa.Integer(), nullable=True),
            sa.Column('deployment_date', sa.DateTime(timezone=True), nullable=True),
            sa.Column('rollback_version', sa.String(), nullable=True),
            sa.Column('created_at', sa.DateTime(timezone=True), nullable=False)
        )

    if 'verified_claim_labels' not in existing_tables:
        op.create_table(
            'verified_claim_labels',
            sa.Column('id', sa.Integer(), primary_key=True),
            sa.Column('claim_id', sa.String(), nullable=False),
            sa.Column('review_id', sa.Integer(), nullable=True),
            sa.Column('verified_label', sa.String(), nullable=False),
            sa.Column('confidence_level', sa.String(), nullable=False),
            sa.Column('verified_by_id', sa.Integer(), nullable=True),
            sa.Column('second_reviewer_id', sa.Integer(), nullable=True),
            sa.Column('requires_second_review', sa.Boolean(), nullable=False),
            sa.Column('second_review_status', sa.String(), nullable=False),
            sa.Column('justification', sa.Text(), nullable=False),
            sa.Column('evidence_sources', sa.JSON(), nullable=True),
            sa.Column('label_provenance', sa.String(), nullable=False),
            sa.Column('is_training_eligible', sa.Boolean(), nullable=False),
            sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
            sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False)
        )


def downgrade() -> None:
    op.drop_table('verified_claim_labels')
    op.drop_table('model_registry')
    op.drop_table('model_monitoring_snapshots')
    op.drop_table('investigator_review_events')
    op.drop_table('investigator_reviews')
