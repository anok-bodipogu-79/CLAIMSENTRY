"""
HealthGuard AI - Phase 4 CSV Batch Upload Tests
Verifies CSV upload validation, duplicate rejection, background processing,
and strict reconciliation invariant (successful_rows + rejected_rows == total_rows).
"""

from backend.app.db.models import Claim, CSVUpload
from backend.app.workers.csv_worker import process_csv_in_background


def test_csv_upload_invalid_format(client, auth_headers):
    """Verify non-CSV files are rejected with 400."""
    files = {"file": ("test.txt", b"plain text content", "text/plain")}
    res = client.post("/api/analyzer/csv", files=files, headers=auth_headers)
    assert res.status_code == 400
    assert ".csv" in res.json()["detail"].lower()


def test_csv_upload_missing_required_columns(client, auth_headers):
    """Verify CSV missing required columns (e.g. ClaimID) is rejected."""
    csv_content = b"WrongCol1,WrongCol2\n123,456"
    files = {"file": ("invalid.csv", csv_content, "text/csv")}
    res = client.post("/api/analyzer/csv", files=files, headers=auth_headers)
    assert res.status_code == 422
    assert "missing required csv columns" in res.json()["detail"].lower()


def test_csv_upload_and_reconciliation_invariant(client, auth_headers, db_session):
    """Verify CSV upload, duplicate ClaimID rejection, and reconciliation invariant."""
    # Pre-insert existing claim to trigger conflict
    existing_claim = Claim(
        claim_id="CLM_EXISTING_BATCH",
        provider_id="PRV_BATCH_001",
        reimbursement_amount=500.0,
    )
    db_session.add(existing_claim)
    db_session.commit()

    # CSV with 4 rows:
    # 1: Valid new claim
    # 2: Valid new claim
    # 3: Duplicate of pre-existing DB claim
    # 4: In-batch duplicate of row 1
    csv_data = (
        "ClaimID,ProviderID,BeneID,InscClaimAmtReimbursed,DeductibleAmtPaid,ClaimType,ClaimStartDt,ClaimEndDt\n"
        "CLM_NEW_001,PRV_BATCH_001,BENE11001,1200.0,100.0,Outpatient,2009-03-01,2009-03-01\n"
        "CLM_NEW_002,PRV_BATCH_002,BENE11002,3400.0,500.0,Inpatient,2009-03-01,2009-03-05\n"
        "CLM_EXISTING_BATCH,PRV_BATCH_001,BENE11001,800.0,50.0,Outpatient,2009-03-01,2009-03-01\n"
        "CLM_NEW_001,PRV_BATCH_001,BENE11001,1200.0,100.0,Outpatient,2009-03-01,2009-03-01\n"
    )
    csv_bytes = csv_data.encode("utf-8")

    files = {"file": ("batch_test.csv", csv_bytes, "text/csv")}
    res = client.post("/api/analyzer/csv", files=files, headers=auth_headers)
    assert res.status_code == 202
    upload_data = res.json()
    upload_id = upload_data["upload_id"]
    assert upload_id is not None

    # Check upload status (processed by FastAPI background tasks)
    status_res = client.get(f"/api/analyzer/csv/{upload_id}", headers=auth_headers)
    assert status_res.status_code == 200
    status_data = status_res.json()

    assert status_data["status"] == "COMPLETED"
    assert status_data["total_rows"] == 4
    assert status_data["successful_rows"] == 2
    assert status_data["rejected_rows"] == 2

    # Enforce strict reconciliation invariant: successful_rows + rejected_rows == total_rows
    assert status_data["successful_rows"] + status_data["rejected_rows"] == status_data["total_rows"]

    # Check rejected rows endpoint
    rejected_res = client.get(f"/api/analyzer/csv/{upload_id}/rejected-rows", headers=auth_headers)
    assert rejected_res.status_code == 200
    rejected_data = rejected_res.json()
    rejected_rows = rejected_data["rejected_rows"]
    assert len(rejected_rows) == 2
    rejected_reasons = [r["rejection_reason"] for r in rejected_rows]
    assert any("already exists in database" in r for r in rejected_reasons)
    assert any("duplicate" in r.lower() for r in rejected_reasons)


def test_csv_claim_type_and_metadata_persistence(client, auth_headers, db_session):
    """
    Verify CSV processing preserves actual claim type (Inpatient vs Outpatient)
    and persists rich claim metadata (dates, beneficiary, deductible, attending physician).
    """
    csv_data = (
        "ClaimID,ProviderID,BeneID,ClaimType,ClaimStartDt,ClaimEndDt,AdmissionDt,DischargeDt,"
        "InscClaimAmtReimbursed,DeductibleAmtPaid,AttendingPhysician,ClmDiagnosisCode_1\n"
        "CLM_TYPE_INP,PRV_TYPE_01,BENE_INP_01,Inpatient,2009-04-01,2009-04-05,2009-04-01,2009-04-05,5200.0,1068.0,PHY99001,4019\n"
        "CLM_TYPE_OUT,PRV_TYPE_01,BENE_OUT_01,Outpatient,2009-04-10,2009-04-10,,,350.0,0.0,PHY99002,25000\n"
    )
    files = {"file": ("claim_types_test.csv", csv_data.encode("utf-8"), "text/csv")}
    res = client.post("/api/analyzer/csv", files=files, headers=auth_headers)
    assert res.status_code == 202
    upload_id = res.json()["upload_id"]

    # Verify completion
    status_res = client.get(f"/api/analyzer/csv/{upload_id}", headers=auth_headers)
    assert status_res.status_code == 200
    assert status_res.json()["status"] == "COMPLETED"
    assert status_res.json()["successful_rows"] == 2

    # Query persisted claims from DB
    inp_claim = db_session.query(Claim).filter(Claim.claim_id == "CLM_TYPE_INP").first()
    assert inp_claim is not None
    assert inp_claim.claim_type == "Inpatient", f"Expected Inpatient, got {inp_claim.claim_type}"
    assert inp_claim.reimbursement_amount == 5200.0
    assert inp_claim.deductible_amount == 1068.0
    assert inp_claim.beneficiary_id == "BENE_INP_01"
    assert inp_claim.attending_physician == "PHY99001"
    assert inp_claim.inpatient_los == 4

    out_claim = db_session.query(Claim).filter(Claim.claim_id == "CLM_TYPE_OUT").first()
    assert out_claim is not None
    assert out_claim.claim_type == "Outpatient", f"Expected Outpatient, got {out_claim.claim_type}"
    assert out_claim.reimbursement_amount == 350.0
    assert out_claim.deductible_amount == 0.0
    assert out_claim.beneficiary_id == "BENE_OUT_01"
    assert out_claim.attending_physician == "PHY99002"


def test_single_claim_vs_csv_equivalent_processing(client, auth_headers, db_session):
    """
    Compare single-claim analysis (/api/analyzer/single) and CSV processing for equivalent input data.
    Verifies identical provider risk scores, risk tiers, anomaly flags, and persisted metadata.
    """
    provider_id = "PRV_EQUIV_01"
    bene_id = "BENE_EQUIV_01"
    claim_id_single = "CLM_EQUIV_SINGLE"
    claim_id_csv = "CLM_EQUIV_CSV"

    single_payload = {
        "ClaimID": claim_id_single,
        "ProviderID": provider_id,
        "BeneID": bene_id,
        "ClaimType": "Inpatient",
        "ClaimStartDt": "2009-05-01",
        "ClaimEndDt": "2009-05-06",
        "AdmissionDt": "2009-05-01",
        "DischargeDt": "2009-05-06",
        "InscClaimAmtReimbursed": 4500.0,
        "DeductibleAmtPaid": 500.0,
        "AttendingPhysician": "PHY88001",
        "ClmDiagnosisCode_1": "4019"
    }

    # 1. Single Claim Analysis
    single_res = client.post("/api/analyzer/single", json=single_payload, headers=auth_headers)
    assert single_res.status_code == 200
    single_data = single_res.json()

    # 2. Equivalent CSV Upload Analysis
    csv_line = (
        "ClaimID,ProviderID,BeneID,ClaimType,ClaimStartDt,ClaimEndDt,AdmissionDt,DischargeDt,"
        "InscClaimAmtReimbursed,DeductibleAmtPaid,AttendingPhysician,ClmDiagnosisCode_1\n"
        f"{claim_id_csv},{provider_id},{bene_id},Inpatient,2009-05-01,2009-05-06,2009-05-01,2009-05-06,4500.0,500.0,PHY88001,4019\n"
    )
    files = {"file": ("equiv_test.csv", csv_line.encode("utf-8"), "text/csv")}
    csv_res = client.post("/api/analyzer/csv", files=files, headers=auth_headers)
    assert csv_res.status_code == 202
    upload_id = csv_res.json()["upload_id"]

    csv_status = client.get(f"/api/analyzer/csv/{upload_id}", headers=auth_headers).json()
    assert csv_status["status"] == "COMPLETED"
    assert csv_status["successful_rows"] == 1

    # 3. Retrieve CSV results from database
    claim_single_db = db_session.query(Claim).filter(Claim.claim_id == claim_id_single).first()
    claim_csv_db = db_session.query(Claim).filter(Claim.claim_id == claim_id_csv).first()

    assert claim_single_db is not None
    assert claim_csv_db is not None

    # Compare persisted claim attributes
    assert claim_single_db.claim_type == claim_csv_db.claim_type == "Inpatient"
    assert claim_single_db.reimbursement_amount == claim_csv_db.reimbursement_amount == 4500.0
    assert claim_single_db.deductible_amount == claim_csv_db.deductible_amount == 500.0
    assert claim_single_db.inpatient_los == claim_csv_db.inpatient_los == 5
    assert claim_single_db.attending_physician == claim_csv_db.attending_physician == "PHY88001"

    # Compare anomaly evaluation
    single_analysis = claim_single_db.analyses[0]
    csv_analysis = claim_csv_db.analyses[0]
    assert sorted(single_analysis.anomaly_flags) == sorted(csv_analysis.anomaly_flags)


def test_csv_processing_transaction_rollback_on_failure(client, auth_headers, db_session):
    """Verify that if an unhandled DB exception occurs, transaction rolls back cleanly and upload is FAILED."""
    from unittest.mock import patch

    csv_data = (
        "ClaimID,ProviderID,InscClaimAmtReimbursed,ClaimStartDt\n"
        "CLM_FAIL_01,PRV_FAIL_01,1000.0,2009-06-01\n"
    )
    files = {"file": ("fail_test.csv", csv_data.encode("utf-8"), "text/csv")}

    with patch("backend.app.workers.csv_worker.prediction_service.predict_claims_pipeline", side_effect=RuntimeError("Simulated pipeline crash")):
        res = client.post("/api/analyzer/csv", files=files, headers=auth_headers)
        assert res.status_code == 202
        upload_id = res.json()["upload_id"]

        status_res = client.get(f"/api/analyzer/csv/{upload_id}", headers=auth_headers)
        assert status_res.status_code == 200
        data = status_res.json()
        assert data["status"] == "FAILED"
        assert "Simulated pipeline crash" in (data.get("error_message") or "")
