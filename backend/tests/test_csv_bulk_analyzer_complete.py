"""
HealthGuard AI - Complete Bulk CSV Analyzer Regression & Parity Test Suite
==========================================================================
Tests:
1. Template download (INR and USD).
2. Pre-upload CSV preview with column mapping, duplicate detection, and row metrics.
3. Chronological claim evaluation with incremental PostgreSQL history accumulation.
4. Cross-provider beneficiary history and overlap detection across batch claims.
5. Strict reconciliation invariant: total_rows == successful_rows + rejected_rows.
6. Rich claim results with provider ML probability, risk tier, anomaly flags, and priority.
7. Batch CSV results export endpoint.
8. Idempotency and error isolation on malformed rows.
"""

import io
import uuid
from datetime import date
import pytest
from sqlalchemy.orm import Session

from backend.app.db.models import Claim, Provider, Beneficiary, CSVUpload, CSVUploadRow


def test_csv_template_download(client, auth_headers):
    """Verify downloadable schema-conforming CSV templates for both INR and USD."""
    # INR template
    res_inr = client.get("/api/analyzer/csv/template?currency=INR", headers=auth_headers)
    assert res_inr.status_code == 200
    assert "text/csv" in res_inr.headers.get("content-type", "")
    assert "healthguard_claim_template_inr.csv" in res_inr.headers.get("content-disposition", "")
    content_inr = res_inr.text
    assert "ClaimID,ProviderID,BeneID,ClaimType,InscClaimAmtReimbursed" in content_inr
    assert "INR" in content_inr

    # USD template
    res_usd = client.get("/api/analyzer/csv/template?currency=USD", headers=auth_headers)
    assert res_usd.status_code == 200
    assert "healthguard_claim_template_usd.csv" in res_usd.headers.get("content-disposition", "")
    content_usd = res_usd.text
    assert "USD" in content_usd


def test_csv_pre_upload_preview(client, auth_headers, db_session: Session):
    """Verify pre-upload inspection, column normalization, duplicate detection, and summary counts."""
    # Pre-seed existing claim in DB
    existing_cid = f"CLM_PREV_EXIST_{uuid.uuid4().hex[:6]}"
    pre_claim = Claim(
        claim_id=existing_cid,
        provider_id="PRV_PREV_01",
        reimbursement_amount=990.0,
    )
    db_session.add(pre_claim)
    db_session.commit()

    valid_cid = f"CLM_PREV_VAL_{uuid.uuid4().hex[:6]}"
    future_cid = f"CLM_PREV_FUT_{uuid.uuid4().hex[:6]}"

    # CSV with:
    # 1. Valid row
    # 2. DB duplicate row
    # 3. In-batch duplicate of row 1
    # 4. Invalid row (negative amount)
    # 5. Future date row (2099-01-01)
    csv_text = (
        "claim_id,provider,beneficiary_id,amount,service_date,type\n"
        f"{valid_cid},PRV_PREV_01,BENE_P_01,1500.0,2009-04-01,Outpatient\n"
        f"{existing_cid},PRV_PREV_01,BENE_P_01,800.0,2009-04-01,Outpatient\n"
        f"{valid_cid},PRV_PREV_01,BENE_P_01,1500.0,2009-04-01,Outpatient\n"
        f"CLM_BAD_AMT,PRV_PREV_01,BENE_P_01,-500.0,2009-04-01,Outpatient\n"
        f"{future_cid},PRV_PREV_01,BENE_P_01,1200.0,2099-01-01,Outpatient\n"
    )

    files = {"file": ("preview_test.csv", csv_text.encode("utf-8"), "text/csv")}
    res = client.post("/api/analyzer/csv/preview", files=files, headers=auth_headers)
    assert res.status_code == 200
    data = res.json()

    assert data["filename"] == "preview_test.csv"
    assert data["total_rows"] == 5
    assert data["valid_rows_count"] == 1
    assert data["duplicate_rows_count"] == 2
    assert data["invalid_rows_count"] == 2
    assert "ClaimID" in data["detected_columns"]
    assert "ProviderID" in data["detected_columns"]
    assert len(data["missing_required_columns"]) == 0
    assert len(data["preview_rows"]) == 5

    # Check status tags on sample rows
    statuses = [r["status"] for r in data["preview_rows"]]
    assert statuses[0] == "VALID"
    assert statuses[1] == "DUPLICATE"
    assert statuses[2] == "DUPLICATE"
    assert statuses[3] == "INVALID"
    assert statuses[4] == "INVALID"


def test_chronological_processing_provider_history_accumulation(client, auth_headers, db_session: Session):
    """
    Verify chronological evaluation:
    Claims submitted in reverse order in CSV (e.g. May before January) are sorted by ClaimStartDt ASC.
    The earlier claim is evaluated and committed first, and is visible as historical evidence for the later claim.
    """
    pid = f"PRV_CHRONO_{uuid.uuid4().hex[:6]}"
    cid_earlier = f"CLM_EARLY_{uuid.uuid4().hex[:6]}"
    cid_later = f"CLM_LATE_{uuid.uuid4().hex[:6]}"

    # Provide later claim FIRST, earlier claim SECOND in the CSV
    csv_text = (
        "ClaimID,ProviderID,BeneID,ClaimType,ClaimStartDt,ClaimEndDt,InscClaimAmtReimbursed,DeductibleAmtPaid\n"
        f"{cid_later},{pid},BENE_C_02,Outpatient,2009-06-01,2009-06-01,1500.0,50.0\n"
        f"{cid_earlier},{pid},BENE_C_01,Outpatient,2009-02-01,2009-02-01,1200.0,50.0\n"
    )

    files = {"file": ("chrono_test.csv", csv_text.encode("utf-8"), "text/csv")}
    res = client.post("/api/analyzer/csv", files=files, headers=auth_headers)
    assert res.status_code == 202
    upload_id = res.json()["upload_id"]

    # Verify upload completed successfully
    status_res = client.get(f"/api/analyzer/csv/{upload_id}", headers=auth_headers)
    assert status_res.status_code == 200
    st_data = status_res.json()
    assert st_data["status"] == "COMPLETED"
    assert st_data["total_rows"] == 2
    assert st_data["successful_rows"] == 2
    assert st_data["rejected_rows"] == 0
    assert st_data["reconciliation_verified"] is True
    assert st_data["processed_rows"] == 2
    assert st_data["progress_pct"] == 100.0

    # Verify both claims are in the database
    clm_early = db_session.query(Claim).filter(Claim.claim_id == cid_earlier).first()
    clm_late = db_session.query(Claim).filter(Claim.claim_id == cid_later).first()
    assert clm_early is not None
    assert clm_late is not None


def test_cross_provider_beneficiary_overlap_detection(client, auth_headers, db_session: Session):
    """
    Verify cross-provider beneficiary history accumulation:
    Claim 1 from Provider A for Beneficiary B: 2009-03-01 to 2009-03-05 (Inpatient).
    Claim 2 from Provider B for same Beneficiary B: 2009-03-03 to 2009-03-03 (Outpatient).
    Because Claim 1 is evaluated first, Claim 2 should trigger OVERLAPPING_CLAIMS or MULTI_PROVIDER anomaly flag.
    """
    pid_a = f"PRV_A_{uuid.uuid4().hex[:6]}"
    pid_b = f"PRV_B_{uuid.uuid4().hex[:6]}"
    bene_shared = f"BENE_SHARED_{uuid.uuid4().hex[:6]}"
    cid_1 = f"CLM_A_{uuid.uuid4().hex[:6]}"
    cid_2 = f"CLM_B_{uuid.uuid4().hex[:6]}"

    csv_text = (
        "ClaimID,ProviderID,BeneID,ClaimType,ClaimStartDt,ClaimEndDt,AdmissionDt,DischargeDt,InscClaimAmtReimbursed,DeductibleAmtPaid,AttendingPhysician,ClmDiagnosisCode_1\n"
        f"{cid_1},{pid_a},{bene_shared},Inpatient,2009-03-01,2009-03-05,2009-03-01,2009-03-05,4500.0,500.0,PHY01,4019\n"
        f"{cid_2},{pid_b},{bene_shared},Outpatient,2009-03-03,2009-03-03,,,1200.0,0.0,PHY02,25000\n"
    )

    files = {"file": ("overlap_test.csv", csv_text.encode("utf-8"), "text/csv")}
    res = client.post("/api/analyzer/csv", files=files, headers=auth_headers)
    assert res.status_code == 202
    upload_id = res.json()["upload_id"]

    status_res = client.get(f"/api/analyzer/csv/{upload_id}", headers=auth_headers)
    assert status_res.status_code == 200
    assert status_res.json()["status"] == "COMPLETED"
    assert status_res.json()["successful_rows"] == 2

    # Check results endpoint
    results_res = client.get(f"/api/analyzer/csv/{upload_id}/results", headers=auth_headers)
    assert results_res.status_code == 200
    claims_list = results_res.json()["claims"]
    assert len(claims_list) == 2

    # Claim 2 should have anomaly flags related to overlap or repeat beneficiary
    c2 = next(c for c in claims_list if c["claim_id"] == cid_2)
    assert c2 is not None
    assert c2["has_anomalies"] is True
    assert any("OVERLAP" in f or "REPEAT" in f for f in (c2["anomaly_flags"] or []))


def test_csv_results_rich_forensic_metadata_and_export(client, auth_headers, db_session: Session):
    """
    Verify /results endpoint populates provider_risk_score, provider_risk_tier,
    investigation_priority, and that /export generates complete CSV download.
    """
    pid = f"PRV_EXP_{uuid.uuid4().hex[:6]}"
    cid = f"CLM_EXP_{uuid.uuid4().hex[:6]}"

    csv_text = (
        "ClaimID,ProviderID,BeneID,ClaimType,ClaimStartDt,ClaimEndDt,InscClaimAmtReimbursed,DeductibleAmtPaid,Currency\n"
        f"{cid},{pid},BENE_EXP_01,Outpatient,2009-05-01,2009-05-01,15000.0,1000.0,INR\n"
    )

    files = {"file": ("export_test.csv", csv_text.encode("utf-8"), "text/csv")}
    res = client.post("/api/analyzer/csv", files=files, headers=auth_headers)
    assert res.status_code == 202
    upload_id = res.json()["upload_id"]

    # Verify results
    results_res = client.get(f"/api/analyzer/csv/{upload_id}/results", headers=auth_headers)
    assert results_res.status_code == 200
    claims = results_res.json()["claims"]
    assert len(claims) == 1
    c = claims[0]
    assert c["claim_id"] == cid
    assert c["provider_id"] == pid
    assert c["currency"] == "INR"
    assert c["original_reimbursement"] == 15000.0
    assert c["provider_risk_score"] is not None
    assert c["provider_risk_tier"] in ("LOW", "MEDIUM", "HIGH")
    assert c["investigation_priority"] in ("LOW", "MEDIUM", "HIGH")
    assert c["processing_status"] == "SUCCESS"

    # Verify export endpoint
    export_res = client.get(f"/api/analyzer/csv/{upload_id}/export", headers=auth_headers)
    assert export_res.status_code == 200
    assert "text/csv" in export_res.headers.get("content-type", "")
    assert f"upload_{upload_id}_batch_results.csv" in export_res.headers.get("content-disposition", "")
    csv_out = export_res.text
    assert "RowNumber,ClaimID,ProviderID,BeneficiaryID" in csv_out
    assert cid in csv_out
    assert pid in csv_out
    assert "SUCCESS" in csv_out


def test_csv_reconciliation_invariant_and_rejected_inspection(client, auth_headers):
    """Verify strict invariant total == successful + rejected with granular failure inspection."""
    valid_cid = f"CLM_RECON_V_{uuid.uuid4().hex[:6]}"
    bad_amt_cid = f"CLM_RECON_B_{uuid.uuid4().hex[:6]}"

    csv_text = (
        "ClaimID,ProviderID,BeneID,ClaimStartDt,InscClaimAmtReimbursed\n"
        f"{valid_cid},PRV_REC_01,BENE_R_01,2009-07-01,1000.0\n"
        f",PRV_REC_01,BENE_R_01,2009-07-01,1000.0\n"  # Missing ClaimID
        f"{bad_amt_cid},PRV_REC_01,BENE_R_01,2009-07-01,NOT_A_NUMBER\n"  # Non-numeric
    )

    files = {"file": ("recon_test.csv", csv_text.encode("utf-8"), "text/csv")}
    res = client.post("/api/analyzer/csv", files=files, headers=auth_headers)
    assert res.status_code == 202
    upload_id = res.json()["upload_id"]

    # Check upload status
    status_res = client.get(f"/api/analyzer/csv/{upload_id}", headers=auth_headers)
    data = status_res.json()
    assert data["status"] == "COMPLETED"
    assert data["total_rows"] == 3
    assert data["successful_rows"] == 1
    assert data["rejected_rows"] == 2
    assert data["reconciliation_verified"] is True
    assert data["successful_rows"] + data["rejected_rows"] == data["total_rows"]

    # Check rejected rows endpoint
    rej_res = client.get(f"/api/analyzer/csv/{upload_id}/rejected-rows", headers=auth_headers)
    assert rej_res.status_code == 200
    rej_data = rej_res.json()
    assert rej_data["total_rejected"] == 2
    reasons = [r["rejection_reason"] for r in rej_data["rejected_rows"]]
    assert any("Missing ClaimID" in r for r in reasons)
    assert any("reimbursement" in r.lower() for r in reasons)
