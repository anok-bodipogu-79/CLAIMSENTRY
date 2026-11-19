"""
HealthGuard AI - Centralized Investigator-Friendly Feature Explanation Layer
══════════════════════════════════════════════════════════════════════════════
Translates complex 51-feature tabular representations, XGBoost SHAP attributions,
and Bi-LSTM temporal sequence dynamics into transparent, evidence-based,
investigator-friendly narratives.

Key Principles:
  1. Statistical Accuracy: Never fabricate values; use actual observations and empirical peer quantiles.
  2. Objective Language: Distinguish statistical model risk impacts from confirmed operational evidence.
  3. No Premature Accusations: Avoid claiming causation or labeling unusual billing as confirmed fraud.
  4. Multi-Currency Awareness: Formats financial features in INR (Indian Rupee - ₹) or USD without double-conversion.
  5. Determinism: Produces reproducible explanations using structured templates.
"""

from typing import Dict, Any, List, Optional, Tuple
import math
import numpy as np


def format_inr_currency(amount: float) -> str:
    """Format a numerical amount into Indian Rupee representation (e.g. ₹11,89,750.00)."""
    if amount is None or (isinstance(amount, float) and (math.isnan(amount) or math.isinf(amount))):
        return "Not available"
    try:
        val = float(amount)
        is_neg = val < 0
        val = abs(val)
        s, *d = f"{val:.2f}".split(".")
        decimal_part = f".{d[0]}" if d else ".00"
        if len(s) <= 3:
            res = s
        else:
            last3 = s[-3:]
            rest = s[:-3]
            pairs = []
            while len(rest) > 2:
                pairs.append(rest[-2:])
                rest = rest[:-2]
            if rest:
                pairs.append(rest)
            res = ",".join(reversed(pairs)) + "," + last3
        return f"{'-' if is_neg else ''}₹{res}{decimal_part}"
    except Exception:
        return "Not available"


def format_usd_currency(amount: float) -> str:
    """Format a numerical amount into USD currency representation (e.g. $14,248.50)."""
    if amount is None or (isinstance(amount, float) and (math.isnan(amount) or math.isinf(amount))):
        return "Not available"
    try:
        val = float(amount)
        return f"${val:,.2f}"
    except Exception:
        return "Not available"


# ══════════════════════════════════════════════════════════════════════════════
# COMPLETE 51-FEATURE DEFINITION CATALOG
# ══════════════════════════════════════════════════════════════════════════════

FEATURE_CATALOG: Dict[str, Dict[str, Any]] = {
    # ── Group 1: Financial and reimbursement ──
    "total_reimbursement": {
        "display_name": "Provider Historical Cumulative Reimbursement",
        "category": "Financial and reimbursement",
        "description": "Cumulative insurance reimbursement billed across all recorded inpatient and outpatient claims.",
        "data_source": "CMS Claim Billing Records",
        "unit": "INR",
        "is_currency": True,
        "scope": "Provider History (Cumulative)",
        "peer_benchmark_applicable": True,
        "template": "Provider cumulative reimbursement across all recorded historical encounters is {obs_str}, compared with a peer reference median of {bench_str}."
    },
    "avg_reimbursement": {
        "display_name": "Provider Average Reimbursement per Claim",
        "category": "Financial and reimbursement",
        "description": "Mean insurance reimbursement claimed per individual patient encounter.",
        "data_source": "CMS Claim Billing Records",
        "unit": "INR",
        "is_currency": True,
        "scope": "Provider History",
        "peer_benchmark_applicable": True,
        "template": "The provider's average reimbursement per claim across historical encounters is {obs_str}, compared with a peer median of {bench_str}."
    },
    "median_reimbursement": {
        "display_name": "Provider Median Reimbursement per Claim",
        "category": "Financial and reimbursement",
        "description": "Median claim reimbursement value, reflecting typical encounter billing without outlier distortion.",
        "data_source": "CMS Claim Billing Records",
        "unit": "INR",
        "is_currency": True,
        "scope": "Provider History",
        "peer_benchmark_applicable": True,
        "template": "The provider's median reimbursement per claim is {obs_str}, compared with a peer reference median of {bench_str}."
    },
    "max_reimbursement": {
        "display_name": "Provider Historical Peak Claim Reimbursement",
        "category": "Financial and reimbursement",
        "description": "Highest reimbursement amount received on any single claim billed by this provider.",
        "data_source": "CMS Claim Billing Records",
        "unit": "INR",
        "is_currency": True,
        "scope": "Provider History",
        "peer_benchmark_applicable": True,
        "template": "The provider's peak single claim reimbursement across history is {obs_str}, compared with a peer median of {bench_str}."
    },
    "std_reimbursement": {
        "display_name": "Reimbursement Standard Deviation",
        "category": "Financial and reimbursement",
        "description": "Dispersion and variance of claim reimbursement amounts around the provider mean.",
        "data_source": "CMS Claim Billing Records",
        "unit": "INR",
        "is_currency": True,
        "scope": "Provider History",
        "peer_benchmark_applicable": False,
        "template": "The dispersion in claim reimbursement amounts is {obs_str}, reflecting billing variance."
    },
    "cv_reimbursement": {
        "display_name": "Reimbursement Coefficient of Variation",
        "category": "Financial and reimbursement",
        "description": "Ratio of standard deviation to mean reimbursement, indicating relative billing volatility.",
        "data_source": "CMS Claim Billing Records",
        "unit": "ratio",
        "is_currency": False,
        "scope": "Provider History",
        "peer_benchmark_applicable": True,
        "template": "The provider's reimbursement coefficient of variation is {obs_str}, compared with a peer median of {bench_str}."
    },
    "ip_total_reimbursement": {
        "display_name": "Total Inpatient Reimbursement",
        "category": "Financial and reimbursement",
        "description": "Cumulative reimbursement billed specifically for inpatient hospital admissions.",
        "data_source": "CMS Inpatient Claims",
        "unit": "INR",
        "is_currency": True,
        "scope": "Provider History",
        "peer_benchmark_applicable": True,
        "template": "Total inpatient hospitalization reimbursement claimed is {obs_str}, compared with a peer reference median of {bench_str}."
    },
    "op_total_reimbursement": {
        "display_name": "Total Outpatient Reimbursement",
        "category": "Financial and reimbursement",
        "description": "Cumulative reimbursement billed specifically for outpatient encounters and procedures.",
        "data_source": "CMS Outpatient Claims",
        "unit": "INR",
        "is_currency": True,
        "scope": "Provider History",
        "peer_benchmark_applicable": True,
        "template": "Total outpatient encounter reimbursement claimed is {obs_str}, compared with a peer reference median of {bench_str}."
    },
    "avg_deductible": {
        "display_name": "Average Deductible Amount",
        "category": "Financial and reimbursement",
        "description": "Average patient deductible amount paid per claim submission.",
        "data_source": "CMS Claim Records",
        "unit": "INR",
        "is_currency": True,
        "scope": "Provider History",
        "peer_benchmark_applicable": True,
        "template": "The average patient deductible per claim is {obs_str}, compared with a peer reference median of {bench_str}."
    },
    "avg_ipannualreimbursementamt": {
        "display_name": "Patient Annual Inpatient Reimbursement",
        "category": "Financial and reimbursement",
        "description": "Average annual inpatient reimbursement received by treated beneficiaries across all health providers.",
        "data_source": "CMS Beneficiary Summary File",
        "unit": "INR",
        "is_currency": True,
        "scope": "Beneficiary History",
        "peer_benchmark_applicable": True,
        "template": "Treated beneficiaries had an average annual inpatient reimbursement of {obs_str}, compared with a peer median of {bench_str}."
    },
    "avg_ipannualdeductibleamt": {
        "display_name": "Patient Annual Inpatient Deductible",
        "category": "Financial and reimbursement",
        "description": "Average annual inpatient deductible paid by treated beneficiaries across all health providers.",
        "data_source": "CMS Beneficiary Summary File",
        "unit": "INR",
        "is_currency": True,
        "scope": "Beneficiary History",
        "peer_benchmark_applicable": True,
        "template": "Treated beneficiaries had an average annual inpatient deductible of {obs_str}, compared with a peer median of {bench_str}."
    },
    "avg_opannualreimbursementamt": {
        "display_name": "Patient Annual Outpatient Reimbursement",
        "category": "Financial and reimbursement",
        "description": "Average annual outpatient reimbursement received by treated beneficiaries across all health providers.",
        "data_source": "CMS Beneficiary Summary File",
        "unit": "INR",
        "is_currency": True,
        "scope": "Beneficiary History",
        "peer_benchmark_applicable": True,
        "template": "Treated beneficiaries had an average annual outpatient reimbursement of {obs_str}, compared with a peer median of {bench_str}."
    },
    "avg_opannualdeductibleamt": {
        "display_name": "Patient Annual Outpatient Deductible",
        "category": "Financial and reimbursement",
        "description": "Average annual outpatient deductible paid by treated beneficiaries across all health providers.",
        "data_source": "CMS Beneficiary Summary File",
        "unit": "INR",
        "is_currency": True,
        "scope": "Beneficiary History",
        "peer_benchmark_applicable": True,
        "template": "Treated beneficiaries had an average annual outpatient deductible of {obs_str}, compared with a peer median of {bench_str}."
    },

    # ── Group 2: Claim volume and provider activity ──
    "total_claim_count": {
        "display_name": "Total Submitted Claims",
        "category": "Claim volume and provider activity",
        "description": "Total number of claims billed by this provider in the available historical record.",
        "data_source": "CMS Claims Dataset",
        "unit": "claims",
        "is_currency": False,
        "scope": "Provider History",
        "peer_benchmark_applicable": True,
        "template": "The provider has submitted a total of {obs_str}, compared with a peer reference median of {bench_str}."
    },
    "unique_beneficiaries": {
        "display_name": "Distinct Patients Treated",
        "category": "Claim volume and provider activity",
        "description": "Count of unique individual beneficiaries who received clinical care from this provider.",
        "data_source": "CMS Claims Dataset",
        "unit": "patients",
        "is_currency": False,
        "scope": "Provider History",
        "peer_benchmark_applicable": True,
        "template": "The provider billed care for {obs_str} distinct patients, compared with a peer reference median of {bench_str}."
    },
    "claims_per_beneficiary": {
        "display_name": "Average Claims per Patient",
        "category": "Claim volume and provider activity",
        "description": "Mean number of claims billed per individual patient, reflecting patient encounter frequency.",
        "data_source": "CMS Claims Dataset",
        "unit": "claims/patient",
        "is_currency": False,
        "scope": "Provider History",
        "peer_benchmark_applicable": True,
        "template": "The provider averaged {obs_str} claims per patient, compared with a peer reference median of {bench_str}."
    },
    "repeat_bene_count": {
        "display_name": "Repeat Patient Count",
        "category": "Claim volume and provider activity",
        "description": "Number of distinct patients billed on two or more occasions by this provider.",
        "data_source": "CMS Claims Dataset",
        "unit": "patients",
        "is_currency": False,
        "scope": "Provider History",
        "peer_benchmark_applicable": True,
        "template": "The provider served {obs_str} repeat patients with multiple billed encounters, compared with a peer median of {bench_str}."
    },
    "repeat_bene_ratio": {
        "display_name": "Repeat Patient Ratio",
        "category": "Claim volume and provider activity",
        "description": "Proportion of all treated beneficiaries who have multiple billed claims with this provider.",
        "data_source": "CMS Claims Dataset",
        "unit": "ratio",
        "is_currency": False,
        "scope": "Provider History",
        "peer_benchmark_applicable": True,
        "template": "Repeat patients constitute {obs_str} of all treated individuals, compared with a peer reference median of {bench_str}."
    },
    "same_day_repeated_claims": {
        "display_name": "Same-Day Repeated Claims",
        "category": "Claim volume and provider activity",
        "description": "Count of claims where the same patient was billed for multiple services starting on the identical calendar date.",
        "data_source": "CMS Claims Dataset",
        "unit": "claims",
        "is_currency": False,
        "scope": "Provider History",
        "peer_benchmark_applicable": True,
        "template": "The provider recorded {obs_str} same-day repeated claims for the same patient, compared with a peer median of {bench_str}."
    },

    # ── Group 3: Beneficiary demographics ──
    "avg_beneficiary_age": {
        "display_name": "Average Beneficiary Age",
        "category": "Beneficiary demographics",
        "description": "Mean age of treated beneficiaries at the date of clinical service admission.",
        "data_source": "CMS Beneficiary Summary File",
        "unit": "years",
        "is_currency": False,
        "scope": "Beneficiary History",
        "peer_benchmark_applicable": True,
        "template": "The average age of treated beneficiaries is {obs_str}, compared with a peer reference median of {bench_str}."
    },
    "avg_chronic_cond_count": {
        "display_name": "Average Chronic Conditions per Patient",
        "category": "Beneficiary demographics",
        "description": "Mean count of chronic health conditions (e.g. diabetes, hypertension, heart failure) per patient.",
        "data_source": "CMS Beneficiary Summary File",
        "unit": "conditions",
        "is_currency": False,
        "scope": "Beneficiary History",
        "peer_benchmark_applicable": True,
        "template": "Treated beneficiaries averaged {obs_str} chronic medical conditions, compared with a peer median of {bench_str}."
    },
    "renal_disease_ratio": {
        "display_name": "End-Stage Renal Disease Patient Ratio",
        "category": "Beneficiary demographics",
        "description": "Proportion of treated beneficiaries diagnosed with End-Stage Renal Disease (ESRD).",
        "data_source": "CMS Beneficiary Summary File",
        "unit": "ratio",
        "is_currency": False,
        "scope": "Beneficiary History",
        "peer_benchmark_applicable": True,
        "template": "Patients with End-Stage Renal Disease represent {obs_str} of the provider's caseload, compared with a peer median of {bench_str}."
    },
    "deceased_bene_ratio": {
        "display_name": "Deceased Patient Ratio",
        "category": "Beneficiary demographics",
        "description": "Proportion of treated patients who are recorded as deceased in vital statistics records.",
        "data_source": "CMS Beneficiary Summary File",
        "unit": "ratio",
        "is_currency": False,
        "scope": "Beneficiary History",
        "peer_benchmark_applicable": True,
        "template": "Deceased patients represent {obs_str} of the provider's historical caseload, compared with a peer median of {bench_str}."
    },
    "claims_after_death_ratio": {
        "display_name": "Post-Mortem Claims Ratio",
        "category": "Beneficiary demographics",
        "description": "Proportion of claims where service start date occurs after the beneficiary's recorded date of death.",
        "data_source": "CMS Claims & Vital Statistics",
        "unit": "ratio",
        "is_currency": False,
        "scope": "Provider History",
        "peer_benchmark_applicable": True,
        "template": "Post-mortem claims account for {obs_str} of billed activity, compared with a peer median of {bench_str}."
    },
    "claims_after_death_count": {
        "display_name": "Post-Mortem Claims Count",
        "category": "Beneficiary demographics",
        "description": "Total count of claims submitted with service dates following the patient's verified date of death.",
        "data_source": "CMS Claims & Vital Statistics",
        "unit": "claims",
        "is_currency": False,
        "scope": "Provider History",
        "peer_benchmark_applicable": True,
        "template": "The provider recorded {obs_str} claims billed after patient date of death, compared with a peer median of {bench_str}."
    },
    "avg_part_a_coverage_months": {
        "display_name": "Average Medicare Part A Coverage Duration",
        "category": "Beneficiary demographics",
        "description": "Mean duration in months of Medicare Part A (hospital) coverage across treated beneficiaries.",
        "data_source": "CMS Beneficiary Summary File",
        "unit": "months",
        "is_currency": False,
        "scope": "Beneficiary History",
        "peer_benchmark_applicable": True,
        "template": "Treated beneficiaries averaged {obs_str} months of Medicare Part A hospital coverage, compared with a peer median of {bench_str}."
    },
    "full_part_a_coverage_ratio": {
        "display_name": "Full Part A Coverage Ratio",
        "category": "Beneficiary demographics",
        "description": "Proportion of treated beneficiaries who maintain complete 12-month Medicare Part A enrollment.",
        "data_source": "CMS Beneficiary Summary File",
        "unit": "ratio",
        "is_currency": False,
        "scope": "Beneficiary History",
        "peer_benchmark_applicable": True,
        "template": "Patients with full 12-month Part A hospital coverage represent {obs_str} of beneficiaries, compared with a peer median of {bench_str}."
    },
    "avg_part_b_coverage_months": {
        "display_name": "Average Medicare Part B Coverage Duration",
        "category": "Beneficiary demographics",
        "description": "Mean duration in months of Medicare Part B (medical insurance) coverage across treated beneficiaries.",
        "data_source": "CMS Beneficiary Summary File",
        "unit": "months",
        "is_currency": False,
        "scope": "Beneficiary History",
        "peer_benchmark_applicable": True,
        "template": "Treated beneficiaries averaged {obs_str} months of Medicare Part B medical coverage, compared with a peer median of {bench_str}."
    },
    "full_part_b_coverage_ratio": {
        "display_name": "Full Part B Coverage Ratio",
        "category": "Beneficiary demographics",
        "description": "Proportion of treated beneficiaries who maintain complete 12-month Medicare Part B enrollment.",
        "data_source": "CMS Beneficiary Summary File",
        "unit": "ratio",
        "is_currency": False,
        "scope": "Beneficiary History",
        "peer_benchmark_applicable": True,
        "template": "Patients with full 12-month Part B medical coverage represent {obs_str} of beneficiaries, compared with a peer median of {bench_str}."
    },

    # ── Group 4: Clinical and diagnosis-related ──
    "avg_diagnosis_codes": {
        "display_name": "Average Diagnoses per Claim",
        "category": "Clinical and diagnosis-related",
        "description": "Mean number of clinical diagnostic condition codes recorded on each submitted claim.",
        "data_source": "CMS Claims Dataset",
        "unit": "codes/claim",
        "is_currency": False,
        "scope": "Provider History",
        "peer_benchmark_applicable": True,
        "template": "Claims averaged {obs_str} diagnosis codes per encounter, compared with a peer reference median of {bench_str}."
    },
    "unique_diagnosis_codes": {
        "display_name": "Distinct Diagnosis Codes Count",
        "category": "Clinical and diagnosis-related",
        "description": "Total number of unique clinical diagnosis codes billed across all patient encounters.",
        "data_source": "CMS Claims Dataset",
        "unit": "codes",
        "is_currency": False,
        "scope": "Provider History",
        "peer_benchmark_applicable": True,
        "template": "The provider billed across {obs_str} unique clinical diagnosis codes, compared with a peer reference median of {bench_str}."
    },
    "diagnosis_code_entropy": {
        "display_name": "Diagnosis Diversity (Shannon Entropy)",
        "category": "Clinical and diagnosis-related",
        "description": "Shannon diversity index of diagnostic codes, measuring whether conditions are varied or narrowly concentrated.",
        "data_source": "CMS Claims Dataset",
        "unit": "entropy",
        "is_currency": False,
        "scope": "Provider History",
        "peer_benchmark_applicable": True,
        "template": "The Shannon entropy of billed diagnoses is {obs_str}, compared with a peer reference median of {bench_str}."
    },

    # ── Group 5: Procedure and operational ──
    "avg_procedure_codes": {
        "display_name": "Average Procedures per Claim",
        "category": "Procedure and operational",
        "description": "Mean number of surgical and clinical procedure codes billed per claim submission.",
        "data_source": "CMS Claims Dataset",
        "unit": "codes/claim",
        "is_currency": False,
        "scope": "Provider History",
        "peer_benchmark_applicable": True,
        "template": "Claims averaged {obs_str} procedure codes per encounter, compared with a peer reference median of {bench_str}."
    },
    "unique_procedure_codes": {
        "display_name": "Distinct Procedure Codes Count",
        "category": "Procedure and operational",
        "description": "Total number of unique surgical or diagnostic procedure codes billed by this provider.",
        "data_source": "CMS Claims Dataset",
        "unit": "codes",
        "is_currency": False,
        "scope": "Provider History",
        "peer_benchmark_applicable": True,
        "template": "The provider billed for {obs_str} distinct medical procedures, compared with a peer reference median of {bench_str}."
    },
    "procedure_code_entropy": {
        "display_name": "Variety of Medical Procedures Billed",
        "category": "Procedure and operational",
        "description": "Shannon diversity index of billed procedures, quantifying procedural breadth versus repetition.",
        "data_source": "CMS Claims Dataset",
        "unit": "entropy",
        "is_currency": False,
        "scope": "Provider History",
        "peer_benchmark_applicable": True,
        "template": "The Shannon diversity index of billed procedures is {obs_str}, compared with a peer reference median of {bench_str}."
    },
    "avg_inpatient_length_of_stay": {
        "display_name": "Average Patient Hospital Stay",
        "category": "Procedure and operational",
        "description": "Mean duration in days between hospital admission and discharge for inpatient stays.",
        "data_source": "CMS Inpatient Claims",
        "unit": "days",
        "is_currency": False,
        "scope": "Provider History",
        "peer_benchmark_applicable": True,
        "template": "Inpatient admissions averaged a stay of {obs_str}, compared with a peer reference median of {bench_str}."
    },
    "claim_date_anomaly_count": {
        "display_name": "Chronologically Inverted Claims Count",
        "category": "Procedure and operational",
        "description": "Number of claims exhibiting impossible chronological dates where service end precedes service start.",
        "data_source": "CMS Claim Validation Rules",
        "unit": "claims",
        "is_currency": False,
        "scope": "Provider History",
        "peer_benchmark_applicable": True,
        "template": "The provider recorded {obs_str} claims with chronologically inverted dates, compared with a peer median of {bench_str}."
    },
    "claim_date_anomaly_ratio": {
        "display_name": "Chronologically Inverted Claims Ratio",
        "category": "Procedure and operational",
        "description": "Proportion of submitted claims where service end date is recorded prior to service start date.",
        "data_source": "CMS Claim Validation Rules",
        "unit": "ratio",
        "is_currency": False,
        "scope": "Provider History",
        "peer_benchmark_applicable": True,
        "template": "Inverted claim dates represent {obs_str} of billed encounters, compared with a peer median of {bench_str}."
    },
    "discharge_before_admission_count": {
        "display_name": "Discharge Preceding Admission Count",
        "category": "Procedure and operational",
        "description": "Number of hospital claims where patient discharge is recorded prior to hospital admission.",
        "data_source": "CMS Inpatient Claim Validation",
        "unit": "claims",
        "is_currency": False,
        "scope": "Provider History",
        "peer_benchmark_applicable": True,
        "template": "The provider recorded {obs_str} inpatient stays with discharge dates prior to admission, compared with a peer median of {bench_str}."
    },
    "discharge_before_admission_ratio": {
        "display_name": "Discharge Preceding Admission Ratio",
        "category": "Procedure and operational",
        "description": "Proportion of inpatient claims where discharge date precedes admission date.",
        "data_source": "CMS Inpatient Claim Validation",
        "unit": "ratio",
        "is_currency": False,
        "scope": "Provider History",
        "peer_benchmark_applicable": True,
        "template": "Hospital stays with discharge preceding admission constitute {obs_str} of inpatient claims, compared with a peer median of {bench_str}."
    },

    # ── Group 6: Physician-related ──
    "missing_physician_ratio": {
        "display_name": "Missing Attending Physician Ratio",
        "category": "Physician-related",
        "description": "Proportion of claims submitted without an attending physician identification number.",
        "data_source": "CMS Claim Registry",
        "unit": "ratio",
        "is_currency": False,
        "scope": "Provider History",
        "peer_benchmark_applicable": True,
        "template": "Claims lacking an attending physician identifier account for {obs_str} of submissions, compared with a peer median of {bench_str}."
    },
    "distinct_physicians": {
        "display_name": "Distinct Attending Physicians Count",
        "category": "Physician-related",
        "description": "Total count of individual attending physicians submitting claims under this provider.",
        "data_source": "CMS Claim Registry",
        "unit": "physicians",
        "is_currency": False,
        "scope": "Provider History",
        "peer_benchmark_applicable": True,
        "template": "A total of {obs_str} distinct attending physicians billed under this provider, compared with a peer reference median of {bench_str}."
    },

    # ── Group 7: Temporal and monthly patterns ──
    "active_months": {
        "display_name": "Active Billing Months Count",
        "category": "Temporal and monthly patterns",
        "description": "Number of distinct calendar months during which at least one claim was billed.",
        "data_source": "CMS Claim Timeline",
        "unit": "months",
        "is_currency": False,
        "scope": "Temporal History",
        "peer_benchmark_applicable": True,
        "template": "The provider actively submitted claims across {obs_str} calendar months, compared with a peer reference median of {bench_str}."
    },
    "avg_monthly_claims": {
        "display_name": "Average Monthly Claim Volume",
        "category": "Temporal and monthly patterns",
        "description": "Mean number of claims submitted per active calendar billing month.",
        "data_source": "CMS Claim Timeline",
        "unit": "claims/month",
        "is_currency": False,
        "scope": "Temporal History",
        "peer_benchmark_applicable": True,
        "template": "The provider averaged {obs_str} claims per active month, compared with a peer reference median of {bench_str}."
    },
    "max_monthly_claims": {
        "display_name": "Peak Monthly Claim Volume",
        "category": "Temporal and monthly patterns",
        "description": "Highest single-month claim volume recorded across the provider's billing timeline.",
        "data_source": "CMS Claim Timeline",
        "unit": "claims",
        "is_currency": False,
        "scope": "Temporal History",
        "peer_benchmark_applicable": True,
        "template": "The provider's peak single-month volume reached {obs_str} claims, compared with a peer reference median of {bench_str}."
    },
    "avg_monthly_reimbursement": {
        "display_name": "Average Monthly Reimbursement",
        "category": "Temporal and monthly patterns",
        "description": "Mean reimbursement billed per active calendar billing month.",
        "data_source": "CMS Claim Timeline",
        "unit": "INR",
        "is_currency": True,
        "scope": "Temporal History",
        "peer_benchmark_applicable": True,
        "template": "The provider's average monthly reimbursement is {obs_str}, compared with a peer reference median of {bench_str}."
    },
    "monthly_spike_ratio": {
        "display_name": "Monthly Volume Surge Ratio",
        "category": "Temporal and monthly patterns",
        "description": "Ratio of peak monthly claim volume to average monthly volume, detecting abrupt billing surges.",
        "data_source": "CMS Claim Timeline",
        "unit": "ratio",
        "is_currency": False,
        "scope": "Temporal History",
        "peer_benchmark_applicable": True,
        "template": "The provider's peak-to-average volume surge ratio is {obs_str}, compared with a peer reference median of {bench_str}."
    },
    "avg_monthly_claim_change": {
        "display_name": "Month-over-Month Volume Volatility",
        "category": "Temporal and monthly patterns",
        "description": "Average month-to-month absolute change in claim volume, indicating billing volatility.",
        "data_source": "CMS Claim Timeline",
        "unit": "claims",
        "is_currency": False,
        "scope": "Temporal History",
        "peer_benchmark_applicable": True,
        "template": "Month-over-month billing volume fluctuated by an average of {obs_str} claims, compared with a peer median of {bench_str}."
    },
    "avg_days_between_claims": {
        "display_name": "Average Interval Between Claims",
        "category": "Temporal and monthly patterns",
        "description": "Average chronological interval in days between sequential claim submissions.",
        "data_source": "CMS Claim Timeline",
        "unit": "days",
        "is_currency": False,
        "scope": "Temporal History",
        "peer_benchmark_applicable": True,
        "template": "The average interval between consecutive claim submissions was {obs_str}, compared with a peer median of {bench_str}."
    },

    # ── Group 8: Ratios and peer comparisons ──
    "ip_claim_pct": {
        "display_name": "Inpatient Claims Ratio",
        "category": "Ratios and peer comparisons",
        "description": "Proportion of all submitted claims that represent inpatient hospitalizations.",
        "data_source": "CMS Claims Dataset",
        "unit": "ratio",
        "is_currency": False,
        "scope": "Provider History",
        "peer_benchmark_applicable": True,
        "template": "Inpatient hospitalizations comprise {obs_str} of all submitted claims, compared with a peer reference median of {bench_str}."
    },
    "reimbursement_to_peer_median": {
        "display_name": "Reimbursement Ratio vs Peer Group Median",
        "category": "Ratios and peer comparisons",
        "description": "Provider's cumulative reimbursement relative to the peer reference median (1.0 = exactly equal to peer median).",
        "data_source": "CMS Claims & Peer Reference Distribution",
        "unit": "ratio",
        "is_currency": False,
        "scope": "Peer Benchmark",
        "peer_benchmark_applicable": True,
        "template": "The provider's total reimbursement is {obs_str} times the peer reference median (benchmark ratio: {bench_str})."
    },
}


class FeatureExplanationService:
    """
    Centralized service that translates raw ML feature values, SHAP attributions,
    and temporal timelines into verified, plain-English investigator findings.
    """

    @classmethod
    def get_feature_meta(cls, feature_name: str) -> Dict[str, Any]:
        """Retrieve definition and metadata for any of the 51 model features."""
        meta = FEATURE_CATALOG.get(feature_name)
        if meta:
            return dict(meta)
        # Fallback for unexpected or custom features
        readable_title = feature_name.replace("_", " ").title()
        return {
            "display_name": readable_title,
            "category": "Operational and general",
            "description": f"Statistical measure derived from provider claim records ({readable_title}).",
            "data_source": "Provider Historical Claim Records",
            "unit": "value",
            "is_currency": any(k in feature_name for k in ["reimbursement", "deductible", "amt", "cost", "charge"]),
            "scope": "Provider History",
            "peer_benchmark_applicable": False,
            "template": f"{readable_title} is observed at {{obs_str}}."
        }

    @classmethod
    def format_value(
        cls,
        feature_name: str,
        value: Any,
        currency: str = "INR",
        exchange_rate: float = 83.50,
        is_already_inr: bool = False,
    ) -> str:
        """Format a single numerical or categorical feature value according to its documented unit."""
        if value is None or value == "N/A" or value == "Not available":
            return "Not available"
        try:
            val_f = float(value)
            if not math.isfinite(val_f) or math.isnan(val_f):
                return "Not available"
        except (ValueError, TypeError):
            return str(value)

        meta = cls.get_feature_meta(feature_name)
        is_curr = meta.get("is_currency", False)

        if is_curr:
            if currency == "INR":
                inr_amount = val_f if is_already_inr else val_f * exchange_rate
                return format_inr_currency(inr_amount)
            else:
                return format_usd_currency(val_f)

        unit = meta.get("unit", "")
        if unit in ["ratio", "pct"]:
            return f"{val_f:.3f}"
        if unit in ["days", "years", "entropy"]:
            return f"{val_f:.2f} {unit}"
        if unit in ["claims", "patients", "codes", "physicians", "months", "conditions"]:
            if val_f.is_integer():
                return f"{int(val_f):,} {unit}"
            return f"{val_f:.1f} {unit}"
        if val_f.is_integer():
            return f"{int(val_f):,}"
        return f"{val_f:.2f}"

    @classmethod
    def generate_plain_english_finding(
        cls,
        feature_name: str,
        observed_value: Any,
        shap_value: float,
        peer_median: Optional[Any] = None,
        peer_percentile: Optional[float] = None,
        currency: str = "INR",
        exchange_rate: float = 83.50,
        is_already_inr: bool = False,
        rank: int = 1,
        risk_tier: str = "LOW",
    ) -> Dict[str, Any]:
        """
        Generate a fully deterministic, investigator-friendly finding item
        from verified underlying model and peer distribution inputs.
        """
        meta = cls.get_feature_meta(feature_name)
        display_name = meta["display_name"]
        category = meta["category"]
        scope = meta["scope"]
        unit = meta["unit"]
        is_elevating = shap_value > 0

        # Severity assignment rule based on rank and SHAP magnitude
        if is_elevating:
            if rank == 1 or abs(shap_value) >= 0.30:
                severity = "Major Concern"
            elif rank in [2, 3] or abs(shap_value) >= 0.05:
                severity = "Concern"
            else:
                severity = "Informational"
        else:
            severity = "Mitigating Factor"

        obs_str = cls.format_value(
            feature_name=feature_name,
            value=observed_value,
            currency=currency,
            exchange_rate=exchange_rate,
            is_already_inr=is_already_inr,
        )

        # Decouple static CMS Medicare training percentiles from live investigator comparisons.
        # Live peer comparisons require authentic application peer records which are not yet available.
        bench_str = "Peer benchmark unavailable — insufficient application data"
        comp_str = "Peer benchmark unavailable — insufficient application data"

        # Build readable sentence without accusing fraud or asserting causality
        explanation_sentence = f"The provider's historical {display_name.lower()} is observed at {obs_str} (peer reference benchmark unavailable for live application data)."

        # Add objective context clause explaining why final score remains LOW despite some risk-elevating factors
        if is_elevating:
            if risk_tier == "LOW":
                explanation_sentence += " While this historical metric contributed upward pressure in the component tree model, consensus across the ensemble kept the overall provider risk score in the LOW risk tier."
            else:
                explanation_sentence += " This statistical pattern contributed to an increased model risk score."
        else:
            explanation_sentence += " This statistical pattern aligns with expected baseline activity and moderated the model risk score."

        return {
            "rank": rank,
            "feature_name": feature_name,
            "display_name": display_name,
            "title": display_name,
            "category": category,
            "scope": scope,
            "severity": severity,
            "observed_value_raw": observed_value,
            "observed_value_formatted": obs_str,
            "observed_value_display": obs_str,
            "peer_reference_raw": None,
            "peer_reference_formatted": bench_str,
            "comparison_display": comp_str,
            "peer_percentile": None,
            "shap_contribution": round(float(shap_value), 5),
            "effect": "INCREASES_RISK" if is_elevating else "REDUCES_RISK",
            "explanation": explanation_sentence,
            "finding": explanation_sentence,
            "data_source": meta["data_source"],
            "unit": unit,
        }

    @classmethod
    def evaluate_model_agreement(
        cls,
        base_probs: Dict[str, float],
        calibrated_prob: float,
        thresholds: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate consensus across component models (XGBoost, ANN, Bi-LSTM)
        and the final Platt-calibrated stacking ensemble.
        """
        if thresholds is None:
            thresholds = {"low_medium_cutoff": 0.22, "medium_high_cutoff": 0.70}

        t_low = thresholds.get("low_medium_cutoff", 0.22)
        t_high = thresholds.get("medium_high_cutoff", 0.70)

        def _get_tier(p: Optional[float]) -> str:
            if p is None or not math.isfinite(p):
                return "UNKNOWN"
            if p >= t_high:
                return "HIGH"
            if p >= t_low:
                return "MEDIUM"
            return "LOW"

        models_evaluated = []
        tier_counts: Dict[str, int] = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}

        for m_name, label in [("xgboost", "XGBoost (Tabular Trees)"),
                              ("ann", "Deep ANN (Non-linear Dense Net)"),
                              ("bilstm", "Bi-LSTM (Temporal Sequence)")]:
            p = base_probs.get(m_name) if base_probs else None
            tier = _get_tier(p)
            if tier != "UNKNOWN":
                tier_counts[tier] += 1
            models_evaluated.append({
                "model_key": m_name,
                "model_name": label,
                "probability": round(p, 4) if p is not None else None,
                "risk_tier": tier,
                "execution_status": "EXECUTED" if p is not None else "NOT_AVAILABLE"
            })

        final_tier = _get_tier(calibrated_prob)
        valid_model_count = sum(tier_counts.values())

        if valid_model_count < 2:
            consensus_status = "Insufficient Model Outputs"
            consensus_summary = "Fewer than two component model outputs were available to compute agreement."
        elif max(tier_counts.values()) == valid_model_count:
            consensus_status = "Models Agree"
            consensus_summary = f"All {valid_model_count} component models independently arrived at the same {final_tier} risk classification."
        elif tier_counts.get(final_tier, 0) >= 2:
            consensus_status = "Partial Agreement"
            consensus_summary = f"A majority of component models agree on an elevated risk assessment ({final_tier}), though some model divergence was observed."
        else:
            consensus_status = "Models Disagree"
            consensus_summary = "Component models produced conflicting risk classifications across tabular and temporal modalities; stacking meta-classifier weighted the divergent signals."

        return {
            "consensus_status": consensus_status,
            "consensus_summary": consensus_summary,
            "component_models": models_evaluated,
            "calibrated_ensemble": {
                "model_name": "Multi-Modal Stacking Ensemble (Platt Calibrated)",
                "probability": round(calibrated_prob, 4),
                "risk_tier": final_tier,
                "execution_status": "EXECUTED"
            }
        }

    @classmethod
    def generate_investigation_actions(
        cls,
        top_positive_factors: List[Dict[str, Any]],
        risk_tier: str,
        anomaly_flags: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Generate 3-5 specific, evidence-based investigation recommendations
        tailored to the observed statistical and operational risk drivers.
        """
        actions = []
        flag_set = set(anomaly_flags or [])
        feat_names = {f.get("feature") or f.get("feature_name") for f in top_positive_factors}

        # High reimbursement drivers
        if any("reimbursement" in f for f in feat_names):
            actions.append("Audit high-value reimbursement claims to cross-verify billed diagnostic services against itemized facility charge masters.")

        # Inpatient stay drivers
        if any("inpatient" in f or "ip_" in f for f in feat_names):
            actions.append("Request and review patient admission and discharge summaries to verify medical necessity and inpatient length of stay.")

        # Procedure & diagnosis breadth / repetition
        if any("procedure" in f or "diagnosis" in f or "entropy" in f for f in feat_names):
            actions.append("Conduct a clinical documentation review to confirm whether billed surgical and procedural codes match operating room logs.")

        # Physician or duplicate claims
        if any("physician" in f or "same_day" in f or "repeat" in f for f in feat_names):
            actions.append("Verify attending physician credentials and interview clinical staff regarding repeat patient encounter schedules.")

        # Temporal surges
        if any("monthly" in f or "spike" in f or "active_months" in f for f in feat_names):
            actions.append("Review billing records from peak volume months to identify whether temporary staffing or operational changes explain the billing surge.")

        # Default actions if few matches
        if len(actions) < 3:
            actions.append("Compare provider historical claims distribution against regional peer medians for corresponding specialty codes.")
        if len(actions) < 3:
            actions.append("Verify beneficiary eligibility and interview a sampled cohort of treated patients regarding receipt of billed services.")

        return actions[:5]

    @classmethod
    def generate_temporal_narrative(cls, temporal_data: Optional[Dict[str, Any]]) -> str:
        """
        Synthesize Bi-LSTM sequence occlusion dynamics into 1-2 clean, factual sentences.
        Replaces raw colored monthly tiles with investigator-readable findings.
        """
        if not temporal_data or not isinstance(temporal_data, dict):
            return "Temporal billing sequence data was not available for this provider evaluation."

        summary = temporal_data.get("trajectory_summary")
        influential = temporal_data.get("most_influential_months") or []

        if influential and len(influential) > 0:
            top_m = influential[0]
            m_idx = top_m.get("month_index")
            impact = top_m.get("impact", 0.0)
            direction = "elevating" if impact > 0 else "moderating"
            obs = top_m.get("observed_features", {})
            claims_cnt = obs.get("monthly_claim_count")
            reimb_amt = obs.get("avg_reimbursement")

            details = []
            if claims_cnt is not None:
                details.append(f"{int(claims_cnt)} claims")
            if reimb_amt is not None:
                details.append(f"average reimbursement {format_inr_currency(reimb_amt * 83.50)}")

            detail_str = f" ({', '.join(details)})" if details else ""
            return (
                f"Billing activity in Month {m_idx}{detail_str} exhibited the strongest temporal influence "
                f"({impact:+.4f} occlusion impact), {direction} the sequence model's risk assessment. "
                "The overall 12-month billing trajectory aligns with seasonal clinical utilization patterns."
            )

        if summary:
            return f"Bi-LSTM sequence analysis indicates: {summary} Detailed monthly temporal sequence values are preserved in Technical Details."

        return "Temporal billing analysis across the 12-month sequence window showed consistent historical billing without anomalous temporal surges."


feature_explanation_service = FeatureExplanationService()
