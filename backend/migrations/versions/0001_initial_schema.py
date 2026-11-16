"""Initial schema creation for HealthGuard AI Phase 4

Revision ID: 0001_initial_schema
Revises: 
Create Date: 2026-11-16 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa

revision = "0001_initial_schema"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 1. users
    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=100), nullable=False),
        sa.Column("email", sa.String(length=255), nullable=False),
        sa.Column("password_hash", sa.String(length=255), nullable=False),
        sa.Column("role", sa.String(length=50), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False, default=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_users_email"), "users", ["email"], unique=True)
    op.create_index(op.f("ix_users_id"), "users", ["id"], unique=False)

    # 2. model_versions
    op.create_table(
        "model_versions",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("version", sa.String(length=50), nullable=False),
        sa.Column("model_type", sa.String(length=100), nullable=False),
        sa.Column("training_status", sa.String(length=50), nullable=False),
        sa.Column("manifest_path", sa.String(length=255), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False, default=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_model_versions_id"), "model_versions", ["id"], unique=False)
    op.create_index(op.f("ix_model_versions_version"), "model_versions", ["version"], unique=True)

    # 3. providers
    op.create_table(
        "providers",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("provider_id", sa.String(length=50), nullable=False),
        sa.Column("provider_name", sa.String(length=255), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_providers_id"), "providers", ["id"], unique=False)
    op.create_index(op.f("ix_providers_provider_id"), "providers", ["provider_id"], unique=True)

    # 4. beneficiaries
    op.create_table(
        "beneficiaries",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("beneficiary_id", sa.String(length=50), nullable=False),
        sa.Column("renal_disease", sa.Boolean(), nullable=True, default=False),
        sa.Column("chronic_conditions_count", sa.Integer(), nullable=True, default=0),
        sa.Column("is_deceased", sa.Boolean(), nullable=True, default=False),
        sa.Column("date_of_death", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_beneficiaries_beneficiary_id"), "beneficiaries", ["beneficiary_id"], unique=True)
    op.create_index(op.f("ix_beneficiaries_id"), "beneficiaries", ["id"], unique=False)

    # 5. claims
    op.create_table(
        "claims",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("claim_id", sa.String(length=50), nullable=False),
        sa.Column("provider_id", sa.String(length=50), nullable=False),
        sa.Column("beneficiary_id", sa.String(length=50), nullable=True),
        sa.Column("claim_type", sa.String(length=50), nullable=False),
        sa.Column("claim_date", sa.DateTime(), nullable=True),
        sa.Column("claim_end_date", sa.DateTime(), nullable=True),
        sa.Column("admission_date", sa.DateTime(), nullable=True),
        sa.Column("discharge_date", sa.DateTime(), nullable=True),
        sa.Column("reimbursement_amount", sa.Float(), nullable=False),
        sa.Column("deductible_amount", sa.Float(), nullable=False),
        sa.Column("inpatient_los", sa.Integer(), nullable=True),
        sa.Column("attending_physician", sa.String(length=50), nullable=True),
        sa.Column("operating_physician", sa.String(length=50), nullable=True),
        sa.Column("other_physician", sa.String(length=50), nullable=True),
        sa.Column("diagnosis_codes", sa.JSON(), nullable=True),
        sa.Column("procedure_codes", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["beneficiary_id"], ["beneficiaries.beneficiary_id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["provider_id"], ["providers.provider_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_claims_beneficiary_id"), "claims", ["beneficiary_id"], unique=False)
    op.create_index(op.f("ix_claims_claim_id"), "claims", ["claim_id"], unique=True)
    op.create_index(op.f("ix_claims_id"), "claims", ["id"], unique=False)
    op.create_index(op.f("ix_claims_provider_id"), "claims", ["provider_id"], unique=False)

    # 6. provider_risk_scores
    op.create_table(
        "provider_risk_scores",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("provider_id", sa.String(length=50), nullable=False),
        sa.Column("model_version_id", sa.Integer(), nullable=True),
        sa.Column("risk_score", sa.Float(), nullable=False),
        sa.Column("raw_score", sa.Float(), nullable=True),
        sa.Column("risk_tier", sa.String(length=20), nullable=False),
        sa.Column("threshold_low", sa.Float(), nullable=False),
        sa.Column("threshold_high", sa.Float(), nullable=False),
        sa.Column("analysis_snapshot", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["model_version_id"], ["model_versions.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["provider_id"], ["providers.provider_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_provider_risk_scores_created_at"), "provider_risk_scores", ["created_at"], unique=False)
    op.create_index(op.f("ix_provider_risk_scores_id"), "provider_risk_scores", ["id"], unique=False)
    op.create_index(op.f("ix_provider_risk_scores_provider_id"), "provider_risk_scores", ["provider_id"], unique=False)
    op.create_index(op.f("ix_provider_risk_scores_risk_tier"), "provider_risk_scores", ["risk_tier"], unique=False)

    # 7. claim_analyses
    op.create_table(
        "claim_analyses",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("claim_id", sa.String(length=50), nullable=False),
        sa.Column("provider_risk_score_id", sa.Integer(), nullable=True),
        sa.Column("anomaly_score", sa.Float(), nullable=True),
        sa.Column("anomaly_flags", sa.JSON(), nullable=True),
        sa.Column("anomaly_rules_triggered", sa.JSON(), nullable=True),
        sa.Column("analysis_status", sa.String(length=30), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["claim_id"], ["claims.claim_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["provider_risk_score_id"], ["provider_risk_scores.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_claim_analyses_claim_id"), "claim_analyses", ["claim_id"], unique=False)
    op.create_index(op.f("ix_claim_analyses_id"), "claim_analyses", ["id"], unique=False)
    op.create_index(op.f("ix_claim_analyses_provider_risk_score_id"), "claim_analyses", ["provider_risk_score_id"], unique=False)

    # 8. explanations
    op.create_table(
        "explanations",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("claim_analysis_id", sa.Integer(), nullable=True),
        sa.Column("provider_risk_score_id", sa.Integer(), nullable=True),
        sa.Column("explanation_type", sa.String(length=50), nullable=False),
        sa.Column("explanation_title", sa.String(length=255), nullable=False),
        sa.Column("explanation_text", sa.Text(), nullable=False),
        sa.Column("feature_contributions", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["claim_analysis_id"], ["claim_analyses.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["provider_risk_score_id"], ["provider_risk_scores.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_explanations_claim_analysis_id"), "explanations", ["claim_analysis_id"], unique=False)
    op.create_index(op.f("ix_explanations_id"), "explanations", ["id"], unique=False)
    op.create_index(op.f("ix_explanations_provider_risk_score_id"), "explanations", ["provider_risk_score_id"], unique=False)

    # 9. csv_uploads
    op.create_table(
        "csv_uploads",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=True),
        sa.Column("filename", sa.String(length=255), nullable=False),
        sa.Column("total_rows", sa.Integer(), nullable=False),
        sa.Column("successful_rows", sa.Integer(), nullable=False),
        sa.Column("rejected_rows", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=30), nullable=False),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_csv_uploads_id"), "csv_uploads", ["id"], unique=False)
    op.create_index(op.f("ix_csv_uploads_status"), "csv_uploads", ["status"], unique=False)

    # 10. csv_upload_rows
    op.create_table(
        "csv_upload_rows",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("csv_upload_id", sa.Integer(), nullable=False),
        sa.Column("row_number", sa.Integer(), nullable=False),
        sa.Column("claim_id", sa.String(length=50), nullable=True),
        sa.Column("processing_status", sa.String(length=20), nullable=False),
        sa.Column("rejection_reason", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["csv_upload_id"], ["csv_uploads.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_csv_upload_rows_csv_upload_id"), "csv_upload_rows", ["csv_upload_id"], unique=False)
    op.create_index(op.f("ix_csv_upload_rows_id"), "csv_upload_rows", ["id"], unique=False)

    # 11. case_notes
    op.create_table(
        "case_notes",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("provider_id", sa.String(length=50), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=True),
        sa.Column("note_text", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["provider_id"], ["providers.provider_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_case_notes_id"), "case_notes", ["id"], unique=False)
    op.create_index(op.f("ix_case_notes_provider_id"), "case_notes", ["provider_id"], unique=False)

    # 12. audit_logs
    op.create_table(
        "audit_logs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=True),
        sa.Column("action", sa.String(length=100), nullable=False),
        sa.Column("resource_type", sa.String(length=50), nullable=True),
        sa.Column("resource_id", sa.String(length=50), nullable=True),
        sa.Column("details", sa.JSON(), nullable=True),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_audit_logs_action"), "audit_logs", ["action"], unique=False)
    op.create_index(op.f("ix_audit_logs_id"), "audit_logs", ["id"], unique=False)

    # 13. revoked_tokens
    op.create_table(
        "revoked_tokens",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("token", sa.String(length=512), nullable=False),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_revoked_tokens_id"), "revoked_tokens", ["id"], unique=False)
    op.create_index(op.f("ix_revoked_tokens_token"), "revoked_tokens", ["token"], unique=True)


def downgrade() -> None:
    op.drop_table("revoked_tokens")
    op.drop_table("audit_logs")
    op.drop_table("case_notes")
    op.drop_table("csv_upload_rows")
    op.drop_table("csv_uploads")
    op.drop_table("explanations")
    op.drop_table("claim_analyses")
    op.drop_table("provider_risk_scores")
    op.drop_table("claims")
    op.drop_table("beneficiaries")
    op.drop_table("providers")
    op.drop_table("model_versions")
    op.drop_table("users")
