"""
HealthGuard AI - Dataset Importer & Database Seeding Utility
Safely populates the database from Phase 1 processed CSV files.
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
from sqlalchemy.orm import Session

from backend.app.db.database import SessionLocal, init_db
from backend.app.db.models import Provider, Claim, Beneficiary, ProviderRiskScore, ModelVersion
from backend.app.core.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent.parent
PROC_DIR = ROOT / "data" / "processed"


def import_processed_dataset(
    max_providers: Optional[int] = None,
    max_claims: Optional[int] = None,
    db: Optional[Session] = None,
) -> Dict[str, Any]:
    """
    Import processed providers, beneficiaries, and sample claims into the database.
    Idempotent: skips existing records to avoid duplicates.
    """
    init_db()
    close_db = False
    if db is None:
        db = SessionLocal()
        close_db = True

    summary = {
        "providers_imported": 0,
        "beneficiaries_imported": 0,
        "claims_imported": 0,
        "model_versions_registered": 0,
        "errors": [],
    }

    try:
        # 1. Register Model Versions (v1 and v2)
        for ver in ["v1", "v2"]:
            manifest_file = ROOT / "ML" / "artifacts" / ver / "model_manifest.json"
            if manifest_file.exists():
                existing_ver = db.query(ModelVersion).filter(ModelVersion.version == ver).first()
                if not existing_ver:
                    mv = ModelVersion(
                        version=ver,
                        model_type="MultiModalStackingEnsemble",
                        training_status="TRAINED",
                        manifest_path=f"ML/artifacts/{ver}/model_manifest.json",
                        is_active=(ver == settings.MODEL_VERSION),
                    )
                    db.add(mv)
                    summary["model_versions_registered"] += 1
        db.commit()

        # 2. Import Providers from provider_labels.csv
        labels_path = PROC_DIR / "provider_labels.csv"
        if labels_path.exists():
            df_labels = pd.read_csv(labels_path)
            if max_providers:
                df_labels = df_labels.head(max_providers)

            existing_pids = set(
                row[0] for row in db.query(Provider.provider_id).all()
            )
            new_providers = []
            for _, row in df_labels.iterrows():
                pid = str(row["ProviderID"]).strip()
                if pid and pid not in existing_pids:
                    new_providers.append(Provider(
                        provider_id=pid,
                        provider_name=f"Healthcare Provider {pid}"
                    ))
                    existing_pids.add(pid)

            if new_providers:
                db.bulk_save_objects(new_providers)
                db.commit()
                summary["providers_imported"] = len(new_providers)
                logger.info(f"Imported {len(new_providers)} new provider records.")

        # 3. Import Beneficiaries from cleaned_training_beneficiaries.csv
        bene_path = PROC_DIR / "cleaned_training_beneficiaries.csv"
        if bene_path.exists():
            df_bene = pd.read_csv(bene_path, nrows=max_claims or 1000)
            existing_bids = set(
                row[0] for row in db.query(Beneficiary.beneficiary_id).all()
            )
            new_benes = []
            for _, row in df_bene.iterrows():
                bid = str(row["BeneID"]).strip()
                if bid and bid not in existing_bids:
                    new_benes.append(Beneficiary(
                        beneficiary_id=bid,
                        renal_disease=bool(row.get("RenalDiseaseIndicator", "0") in ["1", 1, "Y"]),
                        chronic_conditions_count=int(row.get("ChronicCond_Count", 0) or 0),
                        is_deceased=bool(pd.notna(row.get("DOD"))),
                        date_of_death=pd.to_datetime(row.get("DOD"), errors="coerce") if pd.notna(row.get("DOD")) else None,
                    ))
                    existing_bids.add(bid)

            if new_benes:
                db.bulk_save_objects(new_benes)
                db.commit()
                summary["beneficiaries_imported"] = len(new_benes)
                logger.info(f"Imported {len(new_benes)} beneficiary records.")

        # 4. Import Sample Claims from cleaned_training_claims.csv
        claims_path = PROC_DIR / "cleaned_training_claims.csv"
        if claims_path.exists():
            df_claims = pd.read_csv(claims_path, nrows=max_claims or 500)
            existing_cids = set(
                row[0] for row in db.query(Claim.claim_id).all()
            )
            existing_pids = set(
                row[0] for row in db.query(Provider.provider_id).all()
            )
            existing_bids = set(
                row[0] for row in db.query(Beneficiary.beneficiary_id).all()
            )

            new_claims = []
            for _, row in df_claims.iterrows():
                cid = str(row["ClaimID"]).strip()
                pid = str(row["ProviderID"]).strip()

                if cid and cid not in existing_cids:
                    # Ensure provider exists to satisfy foreign key
                    if pid not in existing_pids:
                        db.add(Provider(provider_id=pid, provider_name=f"Healthcare Provider {pid}"))
                        db.flush()
                        existing_pids.add(pid)

                    bid = str(row.get("BeneID", "")).strip() or None
                    if bid and bid not in existing_bids:
                        db.add(Beneficiary(beneficiary_id=bid))
                        db.flush()
                        existing_bids.add(bid)
                    start_dt = pd.to_datetime(row.get("ClaimStartDt"), errors="coerce")
                    end_dt = pd.to_datetime(row.get("ClaimEndDt"), errors="coerce")

                    def _safe_float(val, default=0.0):
                        if pd.isna(val) or val is None or val == "":
                            return default
                        try:
                            f = float(val)
                            return default if pd.isna(f) else f
                        except (ValueError, TypeError):
                            return default

                    def _safe_int(val, default=0):
                        if pd.isna(val) or val is None or val == "":
                            return default
                        try:
                            return int(float(val))
                        except (ValueError, TypeError):
                            return default

                    reimb = _safe_float(row.get("InscClaimAmtReimbursed", 0.0))
                    deduct = _safe_float(row.get("DeductibleAmtPaid", 0.0))
                    los = _safe_int(row.get("InpatientLOS", 0))

                    new_claims.append(Claim(
                        claim_id=cid,
                        provider_id=pid,
                        beneficiary_id=bid,
                        claim_type=str(row.get("ClaimType", "Outpatient")),
                        claim_date=start_dt if pd.notna(start_dt) else None,
                        claim_end_date=end_dt if pd.notna(end_dt) else None,
                        reimbursement_amount=reimb,
                        deductible_amount=deduct,
                        inpatient_los=los,
                        attending_physician=str(row.get("AttendingPhysician", "")).strip() or None,
                    ))
                    existing_cids.add(cid)

            if new_claims:
                db.bulk_save_objects(new_claims)
                db.commit()
                summary["claims_imported"] = len(new_claims)
                logger.info(f"Imported {len(new_claims)} claims records.")

    except Exception as e:
        logger.exception(f"Error during dataset import: {e}")
        db.rollback()
        summary["errors"].append(str(e))
    finally:
        if close_db:
            db.close()

    return summary


if __name__ == "__main__":
    logger.info("Starting HealthGuard AI dataset import...")
    res = import_processed_dataset(max_providers=50, max_claims=200)
    logger.info(f"Import completed. Summary: {res}")
