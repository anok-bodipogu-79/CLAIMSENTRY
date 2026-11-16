"""
HealthGuard AI - Shared Universal Claim Analysis & Validation Pipeline
======================================================================
Unified inference, explanation, anomaly detection, response validation,
and persistence pipeline shared by both Single-Claim Analysis and CSV Uploads.
Ensures identical methodology, deterministic scoring, strict validation,
and seamless compatibility with present and future database records.
"""

import math
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from backend.app.core.config import settings
from backend.app.db.models import (
    Provider,
    Claim,
    Beneficiary,
    ProviderRiskScore,
    ClaimAnalysis,
    Explanation,
    User,
)
from backend.app.db.schemas import SingleClaimRequest, SingleClaimResponse
from backend.app.services.prediction_service import prediction_service
from backend.app.services.anomaly_service import anomaly_service
from backend.app.services.explanation_service import explanation_service

logger = logging.getLogger(__name__)


class AnalysisValidationError(ValueError):
    """Raised when analysis results fail universal integrity validation."""
    pass


class UniversalResponseValidator:
    """
    Central response validation layer enforcing strict numerical, structural,
    and consistency invariants before any analysis response is returned.
    """

    @staticmethod
    def validate_probability_and_tier(
        score: Any,
        tier: Any,
        thresholds: Optional[Dict[str, float]] = None,
    ) -> Tuple[float, str]:
        """Verify probability is a finite number in [0, 1] and tier matches configured thresholds."""
        if score is None or not isinstance(score, (int, float, np.number)):
            raise AnalysisValidationError(f"Invalid probability value '{score}': must be a finite numerical float.")

        score_f = float(score)
        if not math.isfinite(score_f) or math.isnan(score_f):
            raise AnalysisValidationError(f"Invalid probability value '{score_f}': NaN or Infinite value detected.")

        if not (0.0 <= score_f <= 1.0):
            raise AnalysisValidationError(f"Invalid probability value '{score_f}': strictly outside allowed [0.0, 1.0] range.")

        # Threshold evaluation
        if thresholds is None:
            thresholds = {
                "low_medium_cutoff": settings.RISK_THRESHOLD_LOW,
                "medium_high_cutoff": settings.RISK_THRESHOLD_HIGH,
            }

        t_low = float(thresholds.get("low_medium_cutoff", 0.22))
        t_high = float(thresholds.get("medium_high_cutoff", 0.70))

        if score_f >= t_high:
            expected_tier = "HIGH"
        elif score_f >= t_low:
            expected_tier = "MEDIUM"
        else:
            expected_tier = "LOW"

        tier_str = str(tier).strip().upper() if tier else None
        if not tier_str or tier_str not in ("LOW", "MEDIUM", "HIGH"):
            raise AnalysisValidationError(f"Invalid risk tier '{tier}': must be one of ['LOW', 'MEDIUM', 'HIGH'].")

        if tier_str != expected_tier:
            raise AnalysisValidationError(
                f"Risk tier mismatch: score {score_f:.4f} with cutoffs [{t_low}, {t_high}] "
                f"evaluates to '{expected_tier}', but got '{tier_str}'."
            )

        return score_f, tier_str

    @classmethod
    def validate_single_claim_response(
        cls,
        res: Dict[str, Any],
        thresholds: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Validate complete SingleClaimResponse structure and integrity.
        """
        # 1. Required core keys
        required_keys = [
            "claim_id",
            "provider_id",
            "provider_risk_score",
            "provider_risk_tier",
            "model_version",
            "claim_anomaly_findings",
            "anomaly_flags",
            "has_anomalies",
            "explanation_title",
            "explanation",
            "investigation_recommendation",
            "disclaimer",
        ]
        for k in required_keys:
            if k not in res:
                raise AnalysisValidationError(f"Missing mandatory response field: '{k}'")

        # 2. Probability and tier validation
        score_f, tier_str = cls.validate_probability_and_tier(
            score=res["provider_risk_score"],
            tier=res["provider_risk_tier"],
            thresholds=thresholds,
        )

        # 3. Model version check
        ver = str(res["model_version"]).strip()
        if not ver:
            raise AnalysisValidationError("Model version cannot be empty.")

        # 4. Anomaly findings consistency
        flags = res.get("anomaly_flags") or []
        has_anom = res.get("has_anomalies")
        if has_anom != bool(flags):
            raise AnalysisValidationError(
                f"Anomaly consistency violation: has_anomalies ({has_anom}) != bool(anomaly_flags ({flags}))."
            )

        findings = res.get("claim_anomaly_findings") or []
        for f in findings:
            f_flags = f.get("anomaly_flags", [])
            f_is_anom = f.get("is_anomalous")
            f_count = f.get("anomaly_count", len(f_flags))
            if f_is_anom != (len(f_flags) > 0):
                raise AnalysisValidationError(
                    f"Claim finding anomaly inconsistency: is_anomalous ({f_is_anom}) != (len(flags) > 0)."
                )
            if f_count != len(f_flags):
                raise AnalysisValidationError(
                    f"Claim finding count mismatch: anomaly_count ({f_count}) != len(flags) ({len(f_flags)})."
                )
            # Disclaimer check: if no anomalies, must NOT claim unusual indicators were found
            discl = f.get("disclaimer", "")
            if not f_is_anom and "unusual indicators that may require investigation" in discl.lower():
                raise AnalysisValidationError(
                    "Non-anomalous claim disclaimer misleadingly claims unusual indicators were found."
                )

        # 5. Explanation structure validation
        expl = res.get("explanation")
        if not isinstance(expl, dict):
            raise AnalysisValidationError("Explanation must be a structured dictionary.")

        expl_text = expl.get("text", "")
        if not expl_text or not isinstance(expl_text, str):
            raise AnalysisValidationError("Explanation must include non-empty dynamic narrative text.")

        top_factors = expl.get("top_risk_factors", [])
        for factor in top_factors:
            if not isinstance(factor, dict):
                raise AnalysisValidationError(f"Invalid factor entry in top_risk_factors: {factor}")
            if "feature" not in factor:
                raise AnalysisValidationError("Factor missing 'feature' name.")
            shap_val = factor.get("shap_value")
            if shap_val is None or not math.isfinite(float(shap_val)):
                raise AnalysisValidationError(f"Factor {factor.get('feature')} has non-finite SHAP value: {shap_val}")
            act_val = factor.get("actual_value", factor.get("feature_value"))
            if act_val is None or not math.isfinite(float(act_val)):
                raise AnalysisValidationError(f"Factor {factor.get('feature')} has non-finite actual feature value: {act_val}")

        pos_factors = expl.get("positive_contributors", [])
        for factor in pos_factors:
            if float(factor.get("shap_value", 0.0)) <= 0:
                raise AnalysisValidationError(
                    f"Factor {factor.get('feature')} in positive_contributors has non-positive SHAP value: {factor.get('shap_value')}"
                )

        neg_factors = expl.get("negative_contributors", [])
        for factor in neg_factors:
            if float(factor.get("shap_value", 0.0)) >= 0:
                raise AnalysisValidationError(
                    f"Factor {factor.get('feature')} in negative_contributors has non-negative SHAP value: {factor.get('shap_value')}"
                )

        # 6. Temporal explanation validation (if present)
        temporal_expl = expl.get("temporal_explanation")
        if temporal_expl and isinstance(temporal_expl, dict) and "monthly_temporal_impacts" in temporal_expl:
            impacts = temporal_expl["monthly_temporal_impacts"]
            if len(impacts) != 12:
                raise AnalysisValidationError(f"Temporal impacts length {len(impacts)} != 12 months.")
            for imp in impacts:
                is_valid = imp.get("is_valid_observation")
                obs_type = imp.get("observation_type")
                if not is_valid and obs_type != "PADDING":
                    raise AnalysisValidationError("Unobserved month must be marked as PADDING.")

        # 7. Regulatory disclaimer
        disclaimer = res.get("disclaimer", "")
        if not disclaimer or not isinstance(disclaimer, str):
            raise AnalysisValidationError("Mandatory regulatory disclaimer is missing or invalid.")

        return res


class ClaimAnalysisPipeline:
    """
    Unified Claim Analysis Pipeline shared across Single-Claim Analysis and CSV Uploads.
    """

    @staticmethod
    def fetch_provider_historical_claims(
        provider_id: str,
        db: Optional[Session] = None,
    ) -> pd.DataFrame:
        """
        Dynamically query PostgreSQL for all claims previously recorded for this provider,
        and merge with repository reference claims.
        """
        db_records: List[Dict[str, Any]] = []

        if db is not None:
            try:
                claims = db.query(Claim).filter(Claim.provider_id == provider_id).all()
                for c in claims:
                    diag_1 = c.diagnosis_codes[0] if (c.diagnosis_codes and len(c.diagnosis_codes) > 0) else None
                    proc_1 = c.procedure_codes[0] if (c.procedure_codes and len(c.procedure_codes) > 0) else None
                    db_records.append({
                        "ClaimID": c.claim_id,
                        "ProviderID": c.provider_id,
                        "BeneID": c.beneficiary_id,
                        "ClaimType": c.claim_type or "Outpatient",
                        "InscClaimAmtReimbursed": float(c.reimbursement_amount or 0.0),
                        "DeductibleAmtPaid": float(c.deductible_amount or 0.0),
                        "ClaimStartDt": c.claim_date.strftime("%Y-%m-%d") if c.claim_date else None,
                        "ClaimEndDt": c.claim_end_date.strftime("%Y-%m-%d") if c.claim_end_date else None,
                        "AdmissionDt": c.admission_date.strftime("%Y-%m-%d") if c.admission_date else None,
                        "DischargeDt": c.discharge_date.strftime("%Y-%m-%d") if c.discharge_date else None,
                        "InpatientLOS": int(c.inpatient_los or 0),
                        "AttendingPhysician": c.attending_physician,
                        "OperatingPhysician": c.operating_physician,
                        "OtherPhysician": c.other_physician,
                        "ClmDiagnosisCode_1": diag_1,
                        "ClmProcedureCode_1": proc_1,
                    })
            except Exception as e:
                logger.warning(f"Error querying PostgreSQL claims for provider {provider_id}: {e}")

        # Static historical claims from repo
        try:
            repo_df = prediction_service.predictor._load_provider_history(provider_id)
        except Exception:
            repo_df = pd.DataFrame()

        if db_records:
            db_df = pd.DataFrame(db_records)
            if not repo_df.empty:
                db_cids = set(db_df["ClaimID"].astype(str).tolist())
                repo_dedup = repo_df[~repo_df["ClaimID"].astype(str).isin(db_cids)].copy()
                combined = pd.concat([db_df, repo_dedup], ignore_index=True)
            else:
                combined = db_df
        else:
            combined = repo_df.copy()

        return combined

    @classmethod
    def execute_claim_inference(
        cls,
        claims_input: Any,
        provider_id: Optional[str] = None,
        db: Optional[Session] = None,
    ) -> Dict[str, Any]:
        """
        Execute core inference pipeline across single claims or batches.
        """
        # Determine provider_id if single dict
        pid = provider_id
        if not pid and isinstance(claims_input, dict):
            pid = claims_input.get("ProviderID", claims_input.get("provider_id"))

        # Fetch provider history including live PostgreSQL records
        hist_df = None
        if pid:
            hist_df = cls.fetch_provider_historical_claims(provider_id=str(pid), db=db)

        # Run multi-modal ML ensemble inference
        pipeline_res = prediction_service.predict_claims_pipeline(
            claims_input=claims_input,
            provider_id=pid,
            historical_claims_df=hist_df,
        )
        return pipeline_res

    @classmethod
    def analyze_single_claim(
        cls,
        payload: SingleClaimRequest,
        db: Session,
        current_user: Optional[User] = None,
    ) -> SingleClaimResponse:
        """
        Analyze a single claim, run explainability, validate response invariants,
        and persist records to PostgreSQL.
        """
        claim_dict = payload.model_dump(exclude_none=True)

        # 1. Execute shared inference
        try:
            pipeline_res = cls.execute_claim_inference(
                claims_input=claim_dict,
                provider_id=payload.ProviderID,
                db=db,
            )
        except Exception as e:
            raise AnalysisValidationError(f"Inference execution failed: {e}")

        provider_res = pipeline_res.get("provider_result") or {}
        if not provider_res:
            p_map = pipeline_res.get("provider_results", {})
            provider_res = p_map.get(payload.ProviderID, {})

        if not provider_res:
            raise AnalysisValidationError(f"No provider risk evaluation produced for provider {payload.ProviderID}")

        raw_prob = provider_res.get("calibrated_fraud_probability", provider_res.get("fraud_probability"))
        raw_tier = provider_res.get("risk_tier")

        # Validate probability and tier
        prob, risk_tier = UniversalResponseValidator.validate_probability_and_tier(
            score=raw_prob,
            tier=raw_tier,
            thresholds=provider_res.get("risk_thresholds_applied"),
        )

        claims_res = pipeline_res.get("claims_result", [])
        claim_anomaly = claims_res[0] if claims_res else {}
        flags = claim_anomaly.get("anomaly_flags", [])

        # 2. Formulate investigation recommendation
        if risk_tier == "HIGH":
            recommendation = "HIGH PRIORITY: Full clinical and billing audit recommended due to elevated provider fraud risk."
        elif risk_tier == "MEDIUM" or bool(flags):
            recommendation = "MEDIUM PRIORITY: Targeted claims review recommended based on anomalous billing flags."
        else:
            recommendation = "LOW PRIORITY: Normal claim adjudication; no statistical fraud indicators detected."

        # 3. Format explanation dynamically
        explanation_data = explanation_service.format_explanation(provider_res, claims_res)

        # 4. Construct raw response structure
        response_dict = {
            "claim_id": payload.ClaimID,
            "provider_id": payload.ProviderID,
            "provider_risk_score": round(prob, 4),
            "provider_risk_tier": risk_tier,
            "model_version": settings.MODEL_VERSION,
            "claim_anomaly_findings": claims_res,
            "anomaly_flags": flags,
            "has_anomalies": bool(flags),
            "explanation_title": explanation_service.TITLE,
            "explanation": explanation_data,
            "investigation_recommendation": recommendation,
            "disclaimer": settings.REGULATORY_DISCLAIMER,
        }

        # 5. Run Universal Response Validation
        UniversalResponseValidator.validate_single_claim_response(
            response_dict,
            thresholds=provider_res.get("risk_thresholds_applied"),
        )

        # 6. Persist to PostgreSQL in an atomic transaction
        try:
            # Provider
            provider = db.query(Provider).filter(Provider.provider_id == payload.ProviderID).first()
            if not provider:
                provider = Provider(provider_id=payload.ProviderID)
                db.add(provider)
                db.flush()

            # Beneficiary if present
            if payload.BeneID:
                bene = db.query(Beneficiary).filter(Beneficiary.beneficiary_id == payload.BeneID).first()
                if not bene:
                    bene = Beneficiary(beneficiary_id=payload.BeneID)
                    db.add(bene)
                    db.flush()

            # Date parsing helper
            def _parse_dt(s):
                if not s:
                    return None
                try:
                    return pd.to_datetime(s).to_pydatetime()
                except Exception:
                    return None

            clm_start_dt = _parse_dt(payload.ClaimStartDt)
            clm_end_dt = _parse_dt(payload.ClaimEndDt)
            adm_dt = _parse_dt(payload.AdmissionDt)
            dis_dt = _parse_dt(payload.DischargeDt)
            los = max(0, (dis_dt - adm_dt).days) if (adm_dt and dis_dt) else 0

            # Claim
            claim = db.query(Claim).filter(Claim.claim_id == payload.ClaimID).first()
            if not claim:
                claim = Claim(
                    claim_id=payload.ClaimID,
                    provider_id=payload.ProviderID,
                    beneficiary_id=payload.BeneID,
                    claim_type=payload.ClaimType or ("Inpatient" if adm_dt else "Outpatient"),
                    claim_date=clm_start_dt,
                    claim_end_date=clm_end_dt,
                    admission_date=adm_dt,
                    discharge_date=dis_dt,
                    inpatient_los=los,
                    reimbursement_amount=payload.InscClaimAmtReimbursed,
                    deductible_amount=payload.DeductibleAmtPaid or 0.0,
                    attending_physician=payload.AttendingPhysician,
                    operating_physician=payload.OperatingPhysician,
                    other_physician=payload.OtherPhysician,
                    diagnosis_codes=[payload.ClmDiagnosisCode_1] if payload.ClmDiagnosisCode_1 else [],
                    procedure_codes=[payload.ClmProcedureCode_1] if payload.ClmProcedureCode_1 else [],
                )
                db.add(claim)
                db.flush()
            else:
                claim.reimbursement_amount = payload.InscClaimAmtReimbursed
                claim.deductible_amount = payload.DeductibleAmtPaid or 0.0
                if clm_start_dt:
                    claim.claim_date = clm_start_dt
                if clm_end_dt:
                    claim.claim_end_date = clm_end_dt
                if adm_dt:
                    claim.admission_date = adm_dt
                if dis_dt:
                    claim.discharge_date = dis_dt
                claim.inpatient_los = los
                db.flush()

            # Provider Risk Score
            prs = prediction_service.persist_provider_assessment(
                db=db,
                provider_id=payload.ProviderID,
                eval_result=provider_res,
                model_version_str=settings.MODEL_VERSION,
            )

            # Claim Analysis
            ca = anomaly_service.persist_claim_analysis(
                db=db,
                claim_id=payload.ClaimID,
                provider_risk_score_id=prs.id,
                anomaly_record=claim_anomaly,
            )

            # Explanation
            explanation_service.persist_explanation(
                db=db,
                provider_risk_score_id=prs.id,
                claim_analysis_id=ca.id,
                explanation_data=explanation_data,
            )

            db.commit()
        except Exception as e:
            db.rollback()
            raise RuntimeError(f"Database persistence error: {e}")

        return SingleClaimResponse(**response_dict)


claim_analysis_pipeline = ClaimAnalysisPipeline()
