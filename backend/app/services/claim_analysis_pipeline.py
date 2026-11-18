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
                        monthly_trends=dup_trends,
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
        p_history = cls.fetch_provider_historical_claims(provider_id=payload.ProviderID, db=db)
        has_history = p_history is not None and not p_history.empty
        first_dt = None
        last_dt = None
        tot_claims = 1
        active_months = 1
        if has_history:
            tot_claims = len(p_history)
            if "ClaimStartDt" in p_history and not p_history["ClaimStartDt"].dropna().empty:
                first_dt = str(p_history["ClaimStartDt"].dropna().min())[:10]
            if "ClaimEndDt" in p_history and not p_history["ClaimEndDt"].dropna().empty:
                last_dt = str(p_history["ClaimEndDt"].dropna().max())[:10]
            elif "ClaimStartDt" in p_history and not p_history["ClaimStartDt"].dropna().empty:
                last_dt = str(p_history["ClaimStartDt"].dropna().max())[:10]

            if "ClaimStartDt" in p_history:
                try:
                    dt_series = pd.to_datetime(p_history["ClaimStartDt"], errors="coerce").dropna()
                    active_months = max(1, len(dt_series.dt.to_period("M").unique()))
                except Exception:
                    active_months = 1
        else:
            first_dt = payload.ClaimStartDt
            last_dt = payload.ClaimEndDt or payload.ClaimStartDt

        prov_hist_summary = {
            "total_claims_recorded": tot_claims,
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
        )

        # Monthly trends for visual charts
        monthly_trends = cls.get_provider_monthly_trends(
            provider_id=payload.ProviderID,
            db=db,
            currency=cur,
            period="12_months",
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
            "explanation_title": explanation_service.TITLE,
            "explanation": explanation_data,
            "investigator_report": explanation_data.get("investigator_report"),
            "base_model_probabilities": provider_res.get("base_model_probabilities"),
            "monthly_trends": monthly_trends,
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
            audit_entry = AuditLog(
                user_id=current_user.id if current_user else None,
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
    ) -> Dict[str, Any]:
        """
        Aggregate authentic monthly reimbursement and claim activity for a provider.
        Sources from PostgreSQL claims database and CMS historical repository without fabrication.
        Converts reimbursement to requested currency.
        """
        cur = str(currency or "INR").strip().upper()
        p_history = cls.fetch_provider_historical_claims(provider_id=provider_id, db=db)

        monthly_map: Dict[str, Dict[str, Any]] = {}

        if p_history is not None and not p_history.empty:
            df = p_history.copy()
            date_col = "ClaimStartDt" if "ClaimStartDt" in df else ("ClaimDate" if "ClaimDate" in df else None)
            amt_col = "InscClaimAmtReimbursed" if "InscClaimAmtReimbursed" in df else None

            if date_col and amt_col:
                df["dt"] = pd.to_datetime(df[date_col], errors="coerce")
                df = df.dropna(subset=["dt"])
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

        # Also merge any DB claims not captured in CSV
        db_claims = db.query(Claim).filter(Claim.provider_id == provider_id).all()
        for clm in db_claims:
            clm_dt = clm.claim_date or clm.admission_date
            if clm_dt:
                m_str = clm_dt.strftime("%Y-%m")
                if m_str not in monthly_map:
                    usd_val = float(clm.reimbursement_amount or 0.0)
                    if cur == "INR":
                        reimb_val = round(currency_service.usd_to_inr(usd_val), 2)
                    else:
                        reimb_val = round(usd_val, 2)
                    monthly_map[m_str] = {
                        "month": m_str,
                        "reimbursement": reimb_val,
                        "claim_count": 1,
                    }

        sorted_months = sorted(monthly_map.keys())
        if period == "6_months" and len(sorted_months) > 6:
            sorted_months = sorted_months[-6:]
        elif period == "12_months" and len(sorted_months) > 12:
            sorted_months = sorted_months[-12:]

        result_months = [monthly_map[m] for m in sorted_months]
        tot_reimb = round(sum(m["reimbursement"] for m in result_months), 2)
        tot_cnt = sum(m["claim_count"] for m in result_months)

        return {
            "provider_id": provider_id,
            "period": period,
            "currency": cur,
            "total_reimbursement": tot_reimb,
            "total_claims": tot_cnt,
            "months": result_months,
        }


claim_analysis_pipeline = ClaimAnalysisPipeline()
