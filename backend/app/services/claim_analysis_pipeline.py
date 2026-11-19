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
from sqlalchemy.exc import IntegrityError

from backend.app.core.config import settings
from backend.app.db.models import (
    Provider,
    Claim,
    Beneficiary,
    ProviderRiskScore,
    ClaimAnalysis,
    Explanation,
    User,
    AuditLog,
)
from backend.app.db.schemas import SingleClaimRequest, SingleClaimResponse
from backend.app.services.prediction_service import prediction_service
from backend.app.services.anomaly_service import anomaly_service
from backend.app.services.explanation_service import explanation_service
from backend.app.services.currency_service import currency_service

logger = logging.getLogger(__name__)


class AnalysisValidationError(ValueError):
    """Raised when analysis results fail universal integrity validation."""
    pass


class AnalysisConflictError(Exception):
    """Raised when concurrent duplicate submissions conflict and cannot be resolved."""
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
                "low_medium_cutoff": getattr(settings, "DEFAULT_LOW_MEDIUM_THRESHOLD", 0.22),
                "medium_high_cutoff": getattr(settings, "DEFAULT_MEDIUM_HIGH_THRESHOLD", 0.70),
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

        # 8. Data availability validation (if present)
        da = res.get("data_availability")
        if da and isinstance(da, dict):
            for k in ["current_claim", "provider_history", "patient_history", "peer_benchmark"]:
                if k not in da or not isinstance(da[k], dict):
                    raise AnalysisValidationError(f"data_availability missing or malformed required section: '{k}'")

        return res


class ClaimAnalysisPipeline:
    """
    Unified Claim Analysis Pipeline shared across Single-Claim Analysis and CSV Uploads.
    """

    @staticmethod
    def fetch_provider_historical_claims(
        provider_id: str,
        db: Optional[Session] = None,
        exclude_claim_id: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Dynamically query PostgreSQL for all claims previously recorded for this provider.
        Static CMS repository datasets are decoupled from live runtime analysis.
        Excludes exclude_claim_id so the current claim is not counted in historical metrics.
        """
        pid_norm = str(provider_id or "").strip().upper()
        return prediction_service.predictor._load_provider_history(
            provider_id=pid_norm,
            include_db=True,
            db=db,
            exclude_claim_id=exclude_claim_id,
        )

    @staticmethod
    def fetch_patient_historical_claims(
        beneficiary_id: str,
        db: Optional[Session] = None,
        exclude_claim_id: Optional[str] = None,
        current_claim_date: Optional[Any] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Dynamically query PostgreSQL strictly for beneficiary profile and historical claims across ALL providers.
        Decoupled from static CMS reference files.
        Excludes exclude_claim_id and enforces chronological ordering to prevent temporal leakage.
        """
        bene_raw = str(beneficiary_id or "").strip()
        bene_norm = bene_raw.upper()
        if not bene_raw:
            return [], {}

        patient_claims: List[Dict[str, Any]] = []
        patient_profile: Dict[str, Any] = {
            "beneficiary_id": bene_norm,
            "renal_disease": 0,
            "chronic_conditions_count": 0,
            "is_deceased": False,
            "date_of_death": None,
        }

        should_close = False
        session = db
        if session is None:
            try:
                from backend.app.db.database import SessionLocal
                session = SessionLocal()
                should_close = True
            except Exception:
                session = None

        if session is not None:
            try:
                bene_rec = session.query(Beneficiary).filter(
                    (Beneficiary.beneficiary_id == bene_raw) | (Beneficiary.beneficiary_id == bene_norm)
                ).first()
                if bene_rec:
                    patient_profile["renal_disease"] = int(bene_rec.renal_disease or 0)
                    patient_profile["chronic_conditions_count"] = int(bene_rec.chronic_conditions_count or 0)
                    patient_profile["is_deceased"] = bool(bene_rec.is_deceased)
                    patient_profile["date_of_death"] = bene_rec.date_of_death.strftime("%Y-%m-%d") if bene_rec.date_of_death else None

                q = session.query(Claim).filter(
                    (Claim.beneficiary_id == bene_raw) | (Claim.beneficiary_id == bene_norm)
                )
                if exclude_claim_id:
                    q = q.filter(Claim.claim_id != str(exclude_claim_id).strip())
                if current_claim_date:
                    try:
                        c_dt = pd.to_datetime(current_claim_date).to_pydatetime()
                        q = q.filter(Claim.claim_date <= c_dt)
                    except Exception as e:
                        logger.warning(f"Error parsing date for patient anti-leakage filter: {e}")
                clm_recs = q.all()
                for c in clm_recs:
                    patient_claims.append({
                        "claim_id": c.claim_id,
                        "provider_id": c.provider_id,
                        "beneficiary_id": c.beneficiary_id,
                        "claim_type": c.claim_type,
                        "reimbursement_amount": float(c.reimbursement_amount or 0.0),
                        "deductible_amount": float(c.deductible_amount or 0.0),
                        "claim_date": c.claim_date.strftime("%Y-%m-%d") if c.claim_date else None,
                        "claim_end_date": c.claim_end_date.strftime("%Y-%m-%d") if c.claim_end_date else None,
                        "admission_date": c.admission_date.strftime("%Y-%m-%d") if c.admission_date else None,
                        "discharge_date": c.discharge_date.strftime("%Y-%m-%d") if c.discharge_date else None,
                        "attending_physician": c.attending_physician,
                        "diagnosis_codes": c.diagnosis_codes or [],
                        "procedure_codes": c.procedure_codes or [],
                    })
            except Exception as e:
                logger.warning(f"Error querying patient history for {bene_norm}: {e}")
            finally:
                if should_close and session is not None:
                    session.close()

        return patient_claims, patient_profile

    @classmethod
    def execute_claim_inference(
        cls,
        claims_input: Any,
        provider_id: Optional[str] = None,
        db: Optional[Session] = None,
        exclude_claim_id: Optional[str] = None,
        patient_historical_claims: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Execute core inference pipeline across single claims or batches.
        """
        # Determine provider_id if single dict
        pid = provider_id
        cid = exclude_claim_id
        if isinstance(claims_input, dict):
            if not pid:
                pid = claims_input.get("ProviderID", claims_input.get("provider_id"))
            if not cid:
                cid = claims_input.get("ClaimID", claims_input.get("claim_id"))

        # Fetch provider history including live PostgreSQL records (excluding current claim)
        hist_df = None
        if pid:
            hist_df = cls.fetch_provider_historical_claims(
                provider_id=str(pid),
                db=db,
                exclude_claim_id=str(cid) if cid else None,
            )

        # Run multi-modal ML ensemble inference
        pipeline_res = prediction_service.predict_claims_pipeline(
            claims_input=claims_input,
            provider_id=pid,
            historical_claims_df=hist_df,
            patient_historical_claims=patient_historical_claims,
        )
        if isinstance(pipeline_res, dict):
            pipeline_res["historical_claims_df"] = hist_df
        return pipeline_res

    @staticmethod
    def formulate_recommendation(risk_tier: str, flags: List[str]) -> str:
        """Formulate transparent decoupled investigation recommendation."""
        has_flags = bool(flags)
        if risk_tier == "HIGH":
            if has_flags:
                return (
                    "HIGH PRIORITY: Full clinical and billing audit recommended due to elevated provider "
                    "fraud risk accompanied by claim-level billing anomalies."
                )
            return "HIGH PRIORITY: Comprehensive provider audit recommended due to elevated systemic fraud risk score."
        elif risk_tier == "MEDIUM":
            if has_flags:
                return (
                    "MEDIUM PRIORITY: Targeted claims review recommended based on moderate provider risk and specific billing anomalies."
                )
            return "MEDIUM PRIORITY: Routine monitoring with focused claim sample audit due to moderate provider statistical risk."
        else:  # LOW risk tier
            if has_flags:
                return (
                    "ROUTINE MONITORING WITH CLAIM-LEVEL REVIEW: Provider fraud risk is LOW. Specific operational "
                    "billing anomaly flag(s) detected on this claim warrant administrative verification (e.g. duplicate "
                    "code check or outlier validation), but do not constitute evidence of systemic provider fraud."
                )
            return "LOW PRIORITY: Normal claim adjudication; no statistical fraud indicators detected (Provider Risk: LOW)."

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
        # Currency Normalization Boundary
        cur = (payload.Currency or "USD").strip().upper()
        rate = currency_service.get_exchange_rate() if cur == "INR" else 1.0
        orig_reimb = float(payload.InscClaimAmtReimbursed)
        orig_ded = float(payload.DeductibleAmtPaid or 0.0)

        if cur == "INR":
            usd_reimb = currency_service.inr_to_usd(orig_reimb, inr_per_usd=rate)
            usd_ded = currency_service.inr_to_usd(orig_ded, inr_per_usd=rate)
            applied_rate = rate
        else:
            usd_reimb = orig_reimb
            usd_ded = orig_ded
            applied_rate = 1.0

        # 0. Duplicate Claim Idempotency & Conflict Check
        existing_claim = db.query(Claim).filter(Claim.claim_id == payload.ClaimID).first()
        if existing_claim:
            reimb_diff = abs(float(existing_claim.reimbursement_amount or 0.0) - usd_reimb)
            if hasattr(existing_claim, "original_reimbursement") and existing_claim.original_reimbursement is not None and getattr(existing_claim, "currency", None) == cur:
                reimb_diff = min(reimb_diff, abs(float(existing_claim.original_reimbursement) - orig_reimb))
            prov_match = (existing_claim.provider_id == payload.ProviderID)
            bene_match = (existing_claim.beneficiary_id or "") == (payload.BeneID or "")
            type_match = True
            if payload.ClaimType and existing_claim.claim_type:
                type_match = (existing_claim.claim_type.strip().lower() == payload.ClaimType.strip().lower())
            ded_diff = abs(float(existing_claim.deductible_amount or 0.0) - usd_ded)
            if hasattr(existing_claim, "original_deductible") and existing_claim.original_deductible is not None and getattr(existing_claim, "currency", None) == cur:
                ded_diff = min(ded_diff, abs(float(existing_claim.original_deductible) - orig_ded))

            is_exact_match = (prov_match and reimb_diff < 0.01 and bene_match and type_match and ded_diff < 0.01)

            if is_exact_match:
                # Exact duplicate submission: check if existing ClaimAnalysis exists
                existing_ca = (
                    db.query(ClaimAnalysis)
                    .filter(ClaimAnalysis.claim_id == payload.ClaimID)
                    .order_by(ClaimAnalysis.id.desc())
                    .first()
                )
                if existing_ca and existing_ca.provider_risk_score:
                    existing_prs = existing_ca.provider_risk_score
                    existing_expl = (
                        db.query(Explanation)
                        .filter(Explanation.claim_analysis_id == existing_ca.id)
                        .order_by(Explanation.id.desc())
                        .first()
                    )
                    logger.info(f"Exact duplicate claim submission for {payload.ClaimID}: returning existing analysis idempotently.")

                    dup_flags = existing_ca.anomaly_flags or []
                    dup_tier = existing_prs.risk_tier
                    dup_rec = cls.formulate_recommendation(dup_tier, dup_flags)

                    findings = [{
                        "claim_id": existing_claim.claim_id,
                        "is_anomalous": bool(dup_flags),
                        "anomaly_count": len(dup_flags),
                        "anomaly_flags": dup_flags,
                        "anomaly_score": round(float(existing_ca.anomaly_score or 0.0), 4),
                        "rules_triggered": existing_ca.anomaly_rules_triggered or [],
                        "disclaimer": settings.REGULATORY_DISCLAIMER,
                    }]

                    expl_dict = {
                        "title": existing_expl.explanation_title if existing_expl else explanation_service.TITLE,
                        "text": existing_expl.explanation_text if existing_expl else "",
                        "top_risk_factors": (existing_expl.feature_contributions.get("top_risk_factors", []) if existing_expl and existing_expl.feature_contributions else []),
                        "positive_contributors": (existing_expl.feature_contributions.get("positive_contributors", []) if existing_expl and existing_expl.feature_contributions else []),
                        "negative_contributors": (existing_expl.feature_contributions.get("negative_contributors", []) if existing_expl and existing_expl.feature_contributions else []),
                        "temporal_explanation": None,
                        "disclaimer": settings.REGULATORY_DISCLAIMER,
                    }

                    mv_str = (
                        existing_prs.model_version.version
                        if existing_prs.model_version and hasattr(existing_prs.model_version, "version")
                        else settings.MODEL_VERSION
                    )

                    dup_trends = cls.get_provider_monthly_trends(
                        provider_id=payload.ProviderID,
                        db=db,
                        currency=cur,
                        period="12_months",
                    )
                    dup_inv_rep = (
                        existing_expl.feature_contributions.get("investigator_report")
                        if existing_expl and existing_expl.feature_contributions
                        else None
                    )

                    dup_da = (
                        existing_expl.feature_contributions.get("data_availability")
                        if existing_expl and existing_expl.feature_contributions
                        else (dup_inv_rep.get("data_availability") if isinstance(dup_inv_rep, dict) else None)
                    )

                    dup_prov_hist = dup_da.get("provider_history") if isinstance(dup_da, dict) else None
                    dup_bene_hist = (dup_da.get("beneficiary_history") or dup_da.get("patient_history")) if isinstance(dup_da, dict) else None
                    dup_base_models = (
                        existing_prs.analysis_snapshot.get("base_model_probabilities")
                        if existing_prs and existing_prs.analysis_snapshot
                        else None
                    )

                    return SingleClaimResponse(
                        claim_id=payload.ClaimID,
                        provider_id=payload.ProviderID,
                        provider_risk_score=round(float(existing_prs.risk_score), 4),
                        provider_risk_tier=dup_tier,
                        model_version=mv_str,
                        claim_anomaly_findings=findings,
                        anomaly_flags=dup_flags,
                        has_anomalies=bool(dup_flags),
                        explanation_title=explanation_service.TITLE,
                        explanation=expl_dict,
                        investigation_recommendation=dup_rec,
                        disclaimer=settings.REGULATORY_DISCLAIMER,
                        currency=cur,
                        original_reimbursement=round(orig_reimb, 2),
                        original_deductible=round(orig_ded, 2),
                        reimbursement_inr=round(orig_reimb if cur == "INR" else orig_reimb * rate, 2),
                        reimbursement_usd=round(usd_reimb, 2),
                        exchange_rate_applied=applied_rate,
                        exchange_rate_source=currency_service.BENCHMARK_SOURCE,
                        investigator_report=dup_inv_rep,
                        base_model_probabilities=dup_base_models,
                        monthly_trends=dup_trends,
                        data_availability=dup_da,
                        provider_history=dup_prov_hist,
                        beneficiary_history=dup_bene_hist,
                    )
            else:
                # Conflicting details for existing ClaimID
                existing_curr = getattr(existing_claim, "currency", "USD") or "USD"
                existing_amt_str = f"₹{existing_claim.original_reimbursement:,.2f}" if (existing_curr == "INR" and getattr(existing_claim, "original_reimbursement", None) is not None) else f"${existing_claim.reimbursement_amount:,.2f}"
                raise AnalysisValidationError(
                    f"Conflict: Claim ID '{payload.ClaimID}' already exists in database with differing details "
                    f"(Recorded Provider: {existing_claim.provider_id}, Recorded Bene: {existing_claim.beneficiary_id}, "
                    f"Reimbursement: {existing_amt_str}, Claim Type: {existing_claim.claim_type}). "
                    f"Conflicting duplicate claim submissions are rejected to prevent historical claim modification."
                )

        claim_dict = payload.model_dump(exclude_none=True)
        # Ensure claim_dict amounts are converted to USD for ML model inference
        claim_dict["InscClaimAmtReimbursed"] = usd_reimb
        claim_dict["DeductibleAmtPaid"] = usd_ded
        claim_dict["Currency"] = "USD"
        claim_dict["_currency_converted"] = True

        # Query patient history and clinical profile strictly from PostgreSQL (excluding current claim)
        patient_claims, patient_profile = cls.fetch_patient_historical_claims(
            beneficiary_id=payload.BeneID,
            db=db,
            exclude_claim_id=payload.ClaimID,
            current_claim_date=payload.ClaimStartDt,
        )
        patient_prior_count = len(patient_claims)
        if patient_profile.get("date_of_death"):
            claim_dict["DOD"] = patient_profile["date_of_death"]
            claim_dict["_DOD"] = patient_profile["date_of_death"]
        if patient_profile.get("renal_disease"):
            claim_dict["RenalDiseaseIndicator"] = patient_profile["renal_disease"]
        if patient_profile.get("chronic_conditions_count"):
            claim_dict["ChronicCondCount"] = patient_profile["chronic_conditions_count"]

        # 1. Execute shared inference (excluding current claim from provider history)
        try:
            pipeline_res = cls.execute_claim_inference(
                claims_input=claim_dict,
                provider_id=payload.ProviderID,
                db=db,
                exclude_claim_id=payload.ClaimID,
                patient_historical_claims=patient_claims,
            )
        except Exception as e:
            raise AnalysisValidationError(f"Inference execution failed: {e}")

        provider_res = pipeline_res.get("provider_result") or {}
        hist_df = pipeline_res.get("historical_claims_df")
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

        # Build evaluated checks audit trail for all 8 standard checks
        flags_set = set(flags)
        is_inpatient = (payload.ClaimType or "").lower() == "inpatient" or bool(payload.AdmissionDt)
        has_phys = bool(payload.AttendingPhysician)
        diag_code = getattr(payload, "ClmDiagnosisCode_1", None) or getattr(payload, "DiagnosisGroupCode", None)
        has_diag = bool(diag_code)
        has_proc = bool(getattr(payload, "ClmProcedureCode_1", None))

        prov_prior_count = len(hist_df) if (hist_df is not None and not hist_df.empty) else 0

        # Comprehensive Beneficiary Historical Analytics (across ALL providers, excluding current claim)
        distinct_bene_providers = sorted(list({str(c["provider_id"]) for c in patient_claims if c.get("provider_id")}))
        bene_tot_reimb = sum(float(c.get("reimbursement_amount") or 0.0) for c in patient_claims)
        valid_bene_dates = [c["claim_date"] for c in patient_claims if c.get("claim_date")]
        bene_first_dt = min(valid_bene_dates) if valid_bene_dates else None
        bene_last_dt = max(valid_bene_dates) if valid_bene_dates else None

        bene_prov_counts = {}
        for c in patient_claims:
            p = str(c.get("provider_id") or "UNKNOWN")
            bene_prov_counts[p] = bene_prov_counts.get(p, 0) + 1

        # Check service date overlaps across all providers for this beneficiary
        current_start = payload.ClaimStartDt
        current_end = payload.ClaimEndDt or payload.ClaimStartDt
        overlap_found = False
        overlapping_claims_info = []
        if current_start and current_end and patient_claims:
            try:
                c_start_dt = pd.to_datetime(current_start)
                c_end_dt = pd.to_datetime(current_end)
                for c in patient_claims:
                    # Multi-provider overlap requires concurrent active claims from DIFFERENT providers
                    if str(c.get("provider_id") or "").strip() == str(payload.ProviderID or "").strip():
                        continue
                    prior_start = c.get("claim_date")
                    prior_end = c.get("claim_end_date") or prior_start
                    if prior_start and prior_end:
                        p_start_dt = pd.to_datetime(prior_start)
                        p_end_dt = pd.to_datetime(prior_end)
                        if max(c_start_dt, p_start_dt) <= min(c_end_dt, p_end_dt):
                            overlap_found = True
                            overlapping_claims_info.append({
                                "claim_id": c["claim_id"],
                                "provider_id": c["provider_id"],
                                "period": f"{prior_start} to {prior_end}",
                            })
            except Exception as e:
                logger.warning(f"Error checking date overlap: {e}")

        if overlap_found and "OVERLAPPING_CLAIMS" not in flags:
            flags.append("OVERLAPPING_CLAIMS")
            flags_set.add("OVERLAPPING_CLAIMS")
            if claim_anomaly:
                claim_anomaly["anomaly_flags"] = flags
                claim_anomaly["anomaly_count"] = len(flags)
                claim_anomaly["is_anomalous"] = True

        # Comprehensive Provider Historical Analytics (excluding current claim)
        prov_tot_reimb = float(hist_df["InscClaimAmtReimbursed"].sum()) if (hist_df is not None and not hist_df.empty and "InscClaimAmtReimbursed" in hist_df.columns) else 0.0
        prov_unique_benes = int(hist_df["BeneID"].nunique()) if (hist_df is not None and not hist_df.empty and "BeneID" in hist_df.columns) else 0
        prov_avg_reimb = round(prov_tot_reimb / prov_prior_count, 2) if prov_prior_count > 0 else 0.0

        prov_dates = []
        if hist_df is not None and not hist_df.empty and "ClaimStartDt" in hist_df:
            prov_dates = hist_df["ClaimStartDt"].dropna().tolist()
        prov_first_dt = str(min(prov_dates))[:10] if prov_dates else None
        prov_last_dt = str(max(prov_dates))[:10] if prov_dates else None

        provider_history_dict = {
            "available": bool(prov_prior_count > 0),
            "status": "RECORDED" if prov_prior_count > 0 else "NEW_PROVIDER",
            "prior_claim_count": prov_prior_count,
            "first_claim_date": prov_first_dt,
            "last_claim_date": prov_last_dt,
            "total_reimbursement": round(prov_tot_reimb, 2),
            "distinct_beneficiaries_count": prov_unique_benes,
            "avg_reimbursement": prov_avg_reimb,
            "details": f"{prov_prior_count} prior claims recorded in HealthGuard AI." if prov_prior_count > 0 else "New Provider (0 prior claims recorded in HealthGuard AI). Evaluating based on current encounter snapshot.",
        }

        beneficiary_history_dict = {
            "available": bool(patient_prior_count > 0),
            "status": "RECORDED" if patient_prior_count > 0 else "NEW_PATIENT",
            "prior_claim_count": patient_prior_count,
            "first_claim_date": bene_first_dt,
            "last_claim_date": bene_last_dt,
            "total_reimbursement": round(bene_tot_reimb, 2),
            "associated_providers_count": len(distinct_bene_providers),
            "associated_providers": distinct_bene_providers,
            "claims_by_provider": bene_prov_counts,
            "overlap_detected": overlap_found,
            "overlapping_claims": overlapping_claims_info,
            "chronic_conditions_count": int(patient_profile.get("chronic_conditions_count", 0)),
            "renal_disease": bool(patient_profile.get("renal_disease", 0)),
            "is_deceased": bool(patient_profile.get("is_deceased", False)),
            "date_of_death": patient_profile.get("date_of_death"),
            "recent_activity": patient_claims[:5],
            "details": (
                f"{patient_prior_count} prior encounters recorded across {len(distinct_bene_providers)} provider(s) in HealthGuard AI."
                if patient_prior_count > 0
                else "New Beneficiary (0 prior encounters recorded in HealthGuard AI)."
            ),
        }

        # Construct comprehensive Data Availability & Completeness profile
        data_availability = {
            "current_claim": {
                "available": True,
                "claim_id": payload.ClaimID,
                "provider_id": payload.ProviderID,
                "beneficiary_id": payload.BeneID,
                "status": "RECORDED",
                "details": "Claim encounter details submitted and validated in real time.",
            },
            "clinical_coding": {
                "available": bool(diag_code or has_proc),
                "has_diagnosis": bool(diag_code),
                "has_procedure": bool(has_proc),
                "status": "RECORDED" if (diag_code or has_proc) else "PARTIAL",
                "details": "Clinical diagnosis and procedure codes recorded." if (diag_code and has_proc) else "Standard diagnosis or procedure coding recorded.",
            },
            "provider_history": provider_history_dict,
            "patient_history": beneficiary_history_dict,
            "beneficiary_history": beneficiary_history_dict,
            "peer_benchmark": {
                "available": False,
                "status": "INSUFFICIENT_DATA",
                "details": "Peer benchmark unavailable — insufficient application data. Offline CMS reference tables are decoupled from live analysis.",
            },
            "historical_trends": {
                "available": bool(prov_prior_count > 0),
                "status": "RECORDED" if prov_prior_count > 0 else "NO_HISTORY",
                "details": "Multi-month billing trajectory generated from authentic database records." if prov_prior_count > 0 else "No historical claims available in HealthGuard AI for this provider.",
            },
        }

        evaluated_checks = [
            {
                "rule_name": "Chronological Date Validity",
                "status": "FLAGGED" if "INVALID_CLAIM_DATES" in flags_set else "PASSED",
                "description": "Validates ClaimStart <= ClaimEnd and Admission <= Discharge.",
                "details": "Chronological date sequence verified." if "INVALID_CLAIM_DATES" not in flags_set else "Invalid chronological date sequence detected."
            },
            {
                "rule_name": "Physician Identification",
                "status": "FLAGGED" if "MISSING_PHYSICIAN_INFO" in flags_set else "PASSED",
                "description": "Verifies presence of attending/operating physician identifiers.",
                "details": f"Attending physician {payload.AttendingPhysician or 'provided'} recorded." if "MISSING_PHYSICIAN_INFO" not in flags_set else "Missing attending physician identifier."
            },
            {
                "rule_name": "Clinical Diagnostic Coding",
                "status": "FLAGGED" if "MISSING_DIAGNOSIS_INFO" in flags_set else "PASSED",
                "description": "Verifies valid diagnosis code presence.",
                "details": f"Diagnosis code {diag_code or 'verified'} recorded." if "MISSING_DIAGNOSIS_INFO" not in flags_set else "Missing diagnosis codes."
            },
            {
                "rule_name": "Clinical Procedure Reporting",
                "status": "FLAGGED" if "MISSING_PROCEDURE_INFO" in flags_set else ("PASSED" if has_proc or not is_inpatient else "FLAGGED"),
                "description": "Verifies surgical/medical procedure coding for inpatient encounters.",
                "details": "Procedure code recorded and verified." if has_proc else ("Routine outpatient encounter (no surgical procedure required)." if not is_inpatient else "No procedure code recorded for inpatient hospitalization.")
            },
            {
                "rule_name": "Inpatient Length of Stay",
                "status": "FLAGGED" if "UNUSUAL_CLAIM_DURATION" in flags_set else ("PASSED" if is_inpatient else "NOT_APPLICABLE"),
                "description": "Evaluates inpatient hospital length of stay within 0 to 30 days clinical bounds.",
                "details": "Inpatient stay duration within clinical baseline." if is_inpatient else "Outpatient encounter (length of stay check not applicable)."
            },
            {
                "rule_name": "Peer Reimbursement Threshold",
                "status": "FLAGGED" if "HIGH_REIMBURSEMENT" in flags_set else "PASSED",
                "description": "Compares claim reimbursement against training peer 95th percentile benchmark ($57,000 / ₹47,59,500).",
                "details": "Claim reimbursement within normal peer P95 distribution." if "HIGH_REIMBURSEMENT" not in flags_set else "Claim reimbursement exceeds peer 95th percentile threshold."
            },
            {
                "rule_name": "Repeat Beneficiary Frequency",
                "status": "UNAVAILABLE" if (prov_prior_count == 0 or patient_prior_count == 0) else ("FLAGGED" if "REPEAT_BENEFICIARY" in flags_set else "PASSED"),
                "description": "Screens for rapid successive claim submissions for the same beneficiary.",
                "details": (
                    "UNAVAILABLE — No previous beneficiary claims found in PostgreSQL." if patient_prior_count == 0
                    else ("Provider history insufficient in HealthGuard AI to evaluate beneficiary repetition." if prov_prior_count == 0
                          else ("High repeat beneficiary volume detected across provider history." if "REPEAT_BENEFICIARY" in flags_set
                                else "No suspicious repeat beneficiary clustering in provider history."))
                )
            },
            {
                "rule_name": "Multi-Provider Service Date Overlap",
                "status": "UNAVAILABLE" if patient_prior_count == 0 else ("FLAGGED" if overlap_found else "PASSED"),
                "description": "Screens for concurrent active service dates for the same beneficiary across multiple providers.",
                "details": (
                    "No previous beneficiary encounters recorded to evaluate service overlap." if patient_prior_count == 0
                    else (f"Concurrent billing conflict: Active claim overlaps with prior claim from Provider {overlapping_claims_info[0]['provider_id']} ({overlapping_claims_info[0]['period']})." if overlap_found
                          else "No concurrent service date overlap detected across provider records.")
                )
            },
            {
                "rule_name": "Post-Death Service Date",
                "status": ("FLAGGED" if "POST_DEATH_CLAIM" in flags_set else "PASSED") if patient_profile.get("date_of_death") else "UNAVAILABLE",
                "description": "Checks whether claim service date occurs after recorded beneficiary death.",
                "details": (f"Service date occurs after recorded beneficiary death date ({patient_profile.get('date_of_death')})." if "POST_DEATH_CLAIM" in flags_set else f"Service date verified prior to recorded beneficiary death date ({patient_profile.get('date_of_death')}).") if patient_profile.get("date_of_death") else "Beneficiary date of death not recorded in submission or database (check unavailable)."
            },
            {
                "rule_name": "Exact Duplicate Billing Verification",
                "status": "FLAGGED" if "DUP_EXACT_CLAIM" in flags_set else "PASSED",
                "description": "Detects identical claim submissions matching beneficiary, provider, service date, and reimbursement.",
                "details": "Exact duplicate claim detected." if "DUP_EXACT_CLAIM" in flags_set else "No exact duplicate submissions found.",
            },
            {
                "rule_name": "Intra-Claim Duplicate Procedure Verification",
                "status": "FLAGGED" if "CLIN_INTRA_CLAIM_DUPLICATE_PROCEDURE" in flags_set else "PASSED",
                "description": "Verifies that the same procedure code is not repeated across procedure columns on the same claim.",
                "details": "Duplicate procedure codes detected within claim." if "CLIN_INTRA_CLAIM_DUPLICATE_PROCEDURE" in flags_set else "Procedure codes unique within claim.",
            },
            {
                "rule_name": "Same-Day Procedure Repetition Across Claims",
                "status": "FLAGGED" if "DUP_SAME_DAY_PROCEDURE" in flags_set else "PASSED",
                "description": "Detects identical surgical/clinical procedures billed for the beneficiary on the same service date across claims.",
                "details": "Repeated same-day procedure billing detected." if "DUP_SAME_DAY_PROCEDURE" in flags_set else "No same-day repeated procedures across claims.",
            },
            {
                "rule_name": "Provider Baseline Reimbursement Comparison",
                "status": "FLAGGED" if "FIN_PROVIDER_HIST_DEVIATION" in flags_set else ("PASSED" if prov_prior_count >= 5 else "NOT_APPLICABLE"),
                "description": "Flags claims deviating > 3x from provider's own historical median reimbursement.",
                "details": "Reimbursement substantially exceeds provider historical baseline." if "FIN_PROVIDER_HIST_DEVIATION" in flags_set else ("Within expected historical variance." if prov_prior_count >= 5 else "Insufficient history (< 5 prior claims)."),
            },
            {
                "rule_name": "Outpatient Ambulatory Cap Check",
                "status": "FLAGGED" if "FIN_OUTPATIENT_EXTREME_REIMBURSEMENT" in flags_set else "PASSED",
                "description": "Checks if outpatient reimbursement exceeds $15,000.00 standard ambulatory threshold.",
                "details": "Outpatient reimbursement outlier." if "FIN_OUTPATIENT_EXTREME_REIMBURSEMENT" in flags_set else "Outpatient reimbursement within clinical cap.",
            },
            {
                "rule_name": "Provider Volume Surge Anomaly Check",
                "status": "FLAGGED" if "PROV_SUDDEN_VOLUME_SURGE" in flags_set else ("PASSED" if prov_prior_count >= 10 else "NOT_APPLICABLE"),
                "description": "Compares daily claim volume against the provider's historical daily average.",
                "details": "Abnormal daily claim submission volume surge detected." if "PROV_SUDDEN_VOLUME_SURGE" in flags_set else ("Daily volume consistent with historical baseline." if prov_prior_count >= 10 else "Insufficient history (< 10 claims)."),
            },
            {
                "rule_name": "Beneficiary Multi-Provider Hopping Check",
                "status": "FLAGGED" if "BENE_MULTI_PROVIDER_HOPPING" in flags_set else "PASSED",
                "description": "Flags beneficiaries visiting >= 3 distinct healthcare providers within 30 days.",
                "details": "High-frequency multi-provider encounters detected in 30 days." if "BENE_MULTI_PROVIDER_HOPPING" in flags_set else "Beneficiary provider frequency within normal range.",
            },
            {
                "rule_name": "Beneficiary 30-Day Encounter Velocity Check",
                "status": "FLAGGED" if "BENE_HIGH_CLAIM_VELOCITY" in flags_set else "PASSED",
                "description": "Checks if beneficiary generated >= 6 claims within a 30-day period.",
                "details": "Excessive 30-day claim velocity detected." if "BENE_HIGH_CLAIM_VELOCITY" in flags_set else "Encounter velocity within normal bounds.",
            },
            {
                "rule_name": "ICD Coding Version and Effective Date Validation",
                "status": "FLAGGED" if "CLIN_CODE_VERSION_DISCREPANCY" in flags_set else "PASSED",
                "description": "Verifies that diagnosis codes conform to effective CMS date boundaries (ICD-9 vs ICD-10).",
                "details": "Coding version discrepancy detected relative to service date." if "CLIN_CODE_VERSION_DISCREPANCY" in flags_set else "Coding version consistent with service date.",
            },
            {
                "rule_name": "Diagnosis Code Syntax Verification",
                "status": "FLAGGED" if "CLIN_INVALID_ICD9_SYNTAX" in flags_set else "PASSED",
                "description": "Verifies diagnosis code syntax against authoritative ICD structure.",
                "details": "Malformed diagnosis code syntax detected." if "CLIN_INVALID_ICD9_SYNTAX" in flags_set else "Diagnosis codes conform to standard syntax.",
            },
            {
                "rule_name": "National Correct Coding Initiative (NCCI) Procedure Edits",
                "status": "UNAVAILABLE_MISSING_DEPENDENCY",
                "description": "Evaluates mutually exclusive CPT/HCPCS procedure pairs and unbundling edits.",
                "details": "Authoritative dependency unavailable: NCCI edit files require 5-character CPT/HCPCS modifier pairs absent in this CMS dataset.",
            },
            {
                "rule_name": "Medically Unlikely Units (MUE) Validation",
                "status": "UNAVAILABLE_MISSING_DEPENDENCY",
                "description": "Checks maximum allowable units of service per procedure code on a single day of service.",
                "details": "Authoritative dependency unavailable: MUE reference tables require CPT/HCPCS unit quantities absent in this CMS dataset.",
            },
        ]
        if claim_anomaly:
            claim_anomaly["evaluated_checks"] = evaluated_checks

        # 2. Formulate investigation recommendation (Decoupled Policy)
        recommendation = cls.formulate_recommendation(risk_tier, flags)

        # 3. Format explanation dynamically with Investigator Decision Report
        claim_meta = {
            "claim_id": payload.ClaimID,
            "provider_id": payload.ProviderID,
            "beneficiary_id": payload.BeneID,
            "claim_type": payload.ClaimType,
            "original_reimbursement": orig_reimb,
            "currency": cur,
        }
        has_history = prov_prior_count > 0
        first_dt = None
        last_dt = None
        tot_claims = 1
        active_months = 1
        if has_history and hist_df is not None:
            tot_claims = len(hist_df)
            if "ClaimStartDt" in hist_df and not hist_df["ClaimStartDt"].dropna().empty:
                first_dt = str(hist_df["ClaimStartDt"].dropna().min())[:10]
            if "ClaimEndDt" in hist_df and not hist_df["ClaimEndDt"].dropna().empty:
                last_dt = str(hist_df["ClaimEndDt"].dropna().max())[:10]
            elif "ClaimStartDt" in hist_df and not hist_df["ClaimStartDt"].dropna().empty:
                last_dt = str(hist_df["ClaimStartDt"].dropna().max())[:10]

            if "ClaimStartDt" in hist_df:
                try:
                    dt_series = pd.to_datetime(hist_df["ClaimStartDt"], errors="coerce").dropna()
                    active_months = max(1, len(dt_series.dt.to_period("M").unique()))
                except Exception:
                    active_months = 1
        else:
            first_dt = payload.ClaimStartDt
            last_dt = payload.ClaimEndDt or payload.ClaimStartDt

        prov_hist_summary = {
            "total_claims_recorded": tot_claims,
            "historical_claims_prior": tot_claims,
            "total_claims_evaluated_including_current": tot_claims + 1,
            "first_claim_date": first_dt,
            "last_claim_date": last_dt,
            "active_months": active_months,
        }
        explanation_data = explanation_service.format_explanation(
            eval_result=provider_res,
            claim_anomalies=claims_res,
            currency=cur,
            exchange_rate=applied_rate,
            claim_metadata=claim_meta,
            provider_history_summary=prov_hist_summary,
            data_availability=data_availability,
        )

        # Monthly trends for visual charts
        monthly_trends = cls.get_provider_monthly_trends(
            provider_id=payload.ProviderID,
            db=db,
            currency=cur,
            period="12_months",
            current_claim={
                "claim_id": payload.ClaimID,
                "date": payload.ClaimStartDt,
                "reimb_usd": usd_reimb,
            },
            p_history=hist_df,
        )

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
            "anomaly_score": round(float(claim_anomaly.get("anomaly_score", len(flags))), 2),
            "explanation_title": explanation_service.TITLE,
            "explanation": explanation_data,
            "investigator_report": explanation_data.get("investigator_report"),
            "base_model_probabilities": provider_res.get("base_model_probabilities"),
            "monthly_trends": monthly_trends,
            "data_availability": data_availability,
            "provider_history": provider_history_dict,
            "beneficiary_history": beneficiary_history_dict,
            "investigation_recommendation": recommendation,
            "disclaimer": settings.REGULATORY_DISCLAIMER,
            "currency": cur,
            "original_reimbursement": round(orig_reimb, 2),
            "original_deductible": round(orig_ded, 2),
            "reimbursement_inr": round(orig_reimb if cur == "INR" else orig_reimb * rate, 2),
            "reimbursement_usd": round(usd_reimb, 2),
            "exchange_rate_applied": applied_rate,
            "exchange_rate_source": currency_service.BENCHMARK_SOURCE,
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
                    reimbursement_amount=usd_reimb,
                    deductible_amount=usd_ded,
                    currency=cur,
                    original_reimbursement=orig_reimb,
                    original_deductible=orig_ded,
                    attending_physician=payload.AttendingPhysician,
                    operating_physician=payload.OperatingPhysician,
                    other_physician=payload.OtherPhysician,
                    diagnosis_codes=[payload.ClmDiagnosisCode_1] if payload.ClmDiagnosisCode_1 else [],
                    procedure_codes=[payload.ClmProcedureCode_1] if payload.ClmProcedureCode_1 else [],
                )
                db.add(claim)
                db.flush()
            else:
                # Historical claim records are immutable to protect data integrity
                logger.info(f"Existing historical claim {payload.ClaimID} preserved without modification.")

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

            # Audit Log
            valid_uid = None
            if current_user and getattr(current_user, "id", None):
                try:
                    from app.models.user import User as UserModel
                    if db.query(UserModel.id).filter(UserModel.id == current_user.id).first():
                        valid_uid = current_user.id
                except Exception:
                    valid_uid = None

            audit_entry = AuditLog(
                user_id=valid_uid,
                action="CLAIM_FRAUD_ASSESSMENT",
                resource_type="CLAIM",
                resource_id=payload.ClaimID,
                details={
                    "claim_id": payload.ClaimID,
                    "provider_id": payload.ProviderID,
                    "model_version": settings.MODEL_VERSION,
                    "currency": cur,
                    "original_reimbursement": orig_reimb,
                    "original_deductible": orig_ded,
                    "reimbursement_usd": usd_reimb,
                    "deductible_usd": usd_ded,
                    "exchange_rate_applied": applied_rate,
                    "exchange_rate_source": currency_service.BENCHMARK_SOURCE,
                    "base_model_outputs": provider_res.get("base_model_probabilities", {}),
                    "calibrated_probability": round(prob, 4),
                    "risk_tier": risk_tier,
                    "feature_count": provider_res.get("feature_count", 51),
                    "analysis_id": ca.id,
                    "anomaly_flag_count": len(flags),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )
            db.add(audit_entry)
            db.flush()

            db.commit()
        except IntegrityError as ie:
            db.rollback()
            logger.warning(
                f"Concurrent duplicate claim submission detected for ClaimID {payload.ClaimID}: {ie}"
            )
            # Check if another concurrent request successfully completed analysis for this claim
            existing_ca = (
                db.query(ClaimAnalysis)
                .filter(ClaimAnalysis.claim_id == payload.ClaimID)
                .order_by(ClaimAnalysis.id.desc())
                .first()
            )
            if existing_ca:
                logger.info(
                    f"Idempotent resolution for concurrent claim {payload.ClaimID}: returning computed analysis."
                )
                return SingleClaimResponse(**response_dict)
            else:
                raise AnalysisConflictError(
                    f"Concurrent submission conflict for claim {payload.ClaimID}. The claim is already being processed or persisted."
                )
        except Exception as e:
            db.rollback()
            raise RuntimeError(f"Database persistence error: {e}")

        return SingleClaimResponse(**response_dict)

    @classmethod
    def get_provider_monthly_trends(
        cls,
        provider_id: str,
        db: Session,
        currency: str = "INR",
        period: str = "12_months",
        current_claim: Optional[Dict[str, Any]] = None,
        p_history: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """
        Aggregate authentic monthly reimbursement and claim activity for a provider.
        Sources from PostgreSQL claims database and CMS historical repository without fabrication.
        Zero-fills inactive calendar months to return a continuous, ordered 12-month calendar series.
        Converts reimbursement to requested currency.
        """
        cur = str(currency or "INR").strip().upper()
        if p_history is None:
            p_history = cls.fetch_provider_historical_claims(provider_id=provider_id, db=db)

        if p_history is None or p_history.empty:
            return {
                "provider_id": provider_id,
                "period": period,
                "currency": cur,
                "total_reimbursement": 0.0,
                "total_claims": 0,
                "months": [],
                "has_historical_records": False,
                "message": "No historical claims available in HealthGuard AI for this provider. Historical billing trends will dynamically generate as encounter claims are recorded.",
            }

        monthly_map: Dict[str, Dict[str, Any]] = {}
        all_dts: List[pd.Timestamp] = []

        if p_history is not None and not p_history.empty:
            df = p_history.copy()
            date_col = "ClaimStartDt" if "ClaimStartDt" in df else ("ClaimDate" if "ClaimDate" in df else None)
            amt_col = "InscClaimAmtReimbursed" if "InscClaimAmtReimbursed" in df else None

            if date_col and amt_col:
                df["dt"] = pd.to_datetime(df[date_col], errors="coerce")
                df = df.dropna(subset=["dt"])
                if not df.empty:
                    all_dts.extend(df["dt"].tolist())
                    df["month"] = df["dt"].dt.strftime("%Y-%m")
                    df["amt"] = pd.to_numeric(df[amt_col], errors="coerce").fillna(0.0)

                    grouped = df.groupby("month").agg(
                        reimb=("amt", "sum"),
                        cnt=("month", "count"),
                    ).reset_index()

                    for _, row in grouped.iterrows():
                        m_str = str(row["month"])
                        usd_val = float(row["reimb"])
                        if cur == "INR":
                            reimb_val = round(currency_service.usd_to_inr(usd_val), 2)
                        else:
                            reimb_val = round(usd_val, 2)

                        monthly_map[m_str] = {
                            "month": m_str,
                            "reimbursement": reimb_val,
                            "claim_count": int(row["cnt"]),
                        }

        # Include current claim if provided and not already present in p_history
        if current_claim and current_claim.get("date"):
            try:
                c_cid = str(current_claim.get("claim_id", ""))
                already_in_hist = False
                if p_history is not None and not p_history.empty and "ClaimID" in p_history.columns:
                    already_in_hist = c_cid in set(p_history["ClaimID"].astype(str).tolist())

                c_dt = pd.to_datetime(current_claim["date"], errors="coerce")
                if pd.notna(c_dt):
                    all_dts.append(c_dt)
                    if not already_in_hist:
                        c_month = c_dt.strftime("%Y-%m")
                        c_usd = float(current_claim.get("reimb_usd") or 0.0)
                        c_reimb = round(currency_service.usd_to_inr(c_usd), 2) if cur == "INR" else round(c_usd, 2)
                        if c_month in monthly_map:
                            monthly_map[c_month]["reimbursement"] = round(monthly_map[c_month]["reimbursement"] + c_reimb, 2)
                            monthly_map[c_month]["claim_count"] += 1
                        else:
                            monthly_map[c_month] = {
                                "month": c_month,
                                "reimbursement": c_reimb,
                                "claim_count": 1,
                            }
            except Exception as e:
                logger.warning(f"Error merging current claim into monthly trends: {e}")

        # Construct continuous calendar month window with zero-filled inactive months
        result_months: List[Dict[str, Any]] = []
        if all_dts:
            num_months = 6 if period == "6_months" else 12
            max_dt = max(all_dts)
            anchor_month_start = pd.Timestamp(year=max_dt.year, month=max_dt.month, day=1)
            start_month = anchor_month_start - pd.DateOffset(months=num_months - 1)
            consecutive_months = pd.date_range(start=start_month, end=anchor_month_start, freq="MS").strftime("%Y-%m").tolist()

            for m in consecutive_months:
                if m in monthly_map:
                    result_months.append(monthly_map[m])
                else:
                    result_months.append({
                        "month": m,
                        "reimbursement": 0.0,
                        "claim_count": 0,
                    })

        tot_reimb = round(sum(m["reimbursement"] for m in result_months), 2)
        tot_cnt = sum(m["claim_count"] for m in result_months)

        return {
            "provider_id": provider_id,
            "period": period,
            "currency": cur,
            "total_reimbursement": tot_reimb,
            "total_claims": tot_cnt,
            "months": result_months,
            "has_historical_records": True,
            "message": None,
        }


claim_analysis_pipeline = ClaimAnalysisPipeline()
