"""
HealthGuard AI - Currency Conversion & Indian Rupee (INR) Support Service
========================================================================
Manages safe financial conversion between application INR inputs/displays
and the underlying USD-trained machine learning model pipeline.

Core Formulas:
  USD amount = INR amount / INR_PER_USD
  INR amount = USD amount * INR_PER_USD

Exchange Rate Benchmark:
  Default reference benchmark: 83.50 INR per USD
  Source: Reserve Bank of India (RBI) Reference Benchmark (Effective 2024-2025)
  Configurable via environment variable: INR_PER_USD

Safety Constraints:
  - Strict positive exchange rate requirement (fails safely if missing or <= 0).
  - Rejection of negative amounts.
  - Prevention of double-conversion via explicit conversion boundaries and state flags.
  - Historical CMS Medicare records already stored in USD are never re-converted.
  - Full internal floating-point precision preserved for ML feature calculation.
  - Indian numbering system formatting: ₹1,25,000.00.
"""

import os
from typing import Optional, Dict, Any, Tuple
from decimal import Decimal, ROUND_HALF_UP

from backend.app.core.config import settings


class CurrencyConfigurationError(ValueError):
    """Raised when currency exchange rate is missing, non-positive, or invalid."""
    pass


class DoubleConversionError(RuntimeError):
    """Raised when an amount or payload is detected to undergo redundant conversion."""
    pass


class CurrencyService:
    """Safe financial conversion and Indian numbering formatting service."""

    DEFAULT_INR_PER_USD = 83.50
    BENCHMARK_SOURCE = "Reserve Bank of India (RBI) Reference Benchmark (Effective 2024-2025: 83.50 INR/USD)"
    EFFECTIVE_DATE = "2024-11-01"

    @classmethod
    def validate_exchange_rate(cls, rate: Any) -> float:
        import math
        if rate is None or not isinstance(rate, (int, float)) or math.isnan(rate) or math.isinf(rate) or rate <= 0.0:
            raise CurrencyConfigurationError(
                f"Invalid exchange rate: {rate}. INR_PER_USD must be strictly positive (> 0)."
            )
        return float(rate)

    @classmethod
    def get_exchange_rate(cls, override_rate: Optional[float] = None) -> float:
        """
        Retrieve and validate the active exchange rate (INR per USD).
        Fails safely with CurrencyConfigurationError if unconfigured, non-numeric, or <= 0.
        """
        if override_rate is not None:
            rate = override_rate
        else:
            # Check settings first, then environment variable
            env_val = os.getenv("INR_PER_USD")
            if env_val is not None:
                try:
                    rate = float(env_val)
                except (ValueError, TypeError):
                    raise CurrencyConfigurationError(
                        f"Invalid INR_PER_USD environment variable: '{env_val}'. "
                        "Exchange rate must be a valid positive number greater than 0."
                    )
            elif hasattr(settings, "INR_PER_USD") and settings.INR_PER_USD is not None:
                rate = settings.INR_PER_USD
            else:
                raise CurrencyConfigurationError(
                    "No approved exchange rate configured. INR_PER_USD must be set to a positive number (> 0). "
                    "Financial inference aborted safely to prevent arbitrary or corrupt inferences."
                )

        if not isinstance(rate, (int, float)) or rate <= 0.0:
            raise CurrencyConfigurationError(
                f"Invalid exchange rate: {rate}. INR_PER_USD must be strictly positive (> 0)."
            )
        return float(rate)

    @classmethod
    def inr_to_usd(cls, amount_inr: float, inr_per_usd: Optional[float] = None, rate: Optional[float] = None) -> float:
        """
        Convert INR amount to USD for ML model feature engineering.
        USD = INR / INR_PER_USD
        Preserves full internal IEEE 754 floating-point precision.
        """
        if amount_inr is None:
            return 0.0
        try:
            val = float(amount_inr)
        except (ValueError, TypeError):
            raise ValueError(f"Invalid financial amount: {amount_inr}")

        if val < 0.0:
            raise ValueError(f"Financial amount cannot be negative: ₹{val}")

        active_rate = cls.get_exchange_rate(rate if rate is not None else inr_per_usd)
        return val / active_rate

    @classmethod
    def usd_to_inr(cls, amount_usd: float, inr_per_usd: Optional[float] = None, rate: Optional[float] = None) -> float:
        """
        Convert USD amount to INR for application display or reporting.
        INR = USD * INR_PER_USD
        """
        if amount_usd is None:
            return 0.0
        try:
            val = float(amount_usd)
        except (ValueError, TypeError):
            raise ValueError(f"Invalid financial amount: {amount_usd}")

        if val < 0.0:
            raise ValueError(f"Financial amount cannot be negative: {val}")

        active_rate = cls.get_exchange_rate(rate if rate is not None else inr_per_usd)
        return val * active_rate

    @classmethod
    def format_inr(cls, amount: float, symbol: bool = True) -> str:
        """
        Format a numeric value using the Indian Numbering System (e.g. ₹1,25,000.00).
        Grouping rule:
          - Last 3 digits grouped together.
          - Preceding digits grouped in pairs of 2.
        """
        if amount is None:
            return "₹0.00" if symbol else "0.00"
        try:
            val = float(amount)
        except (ValueError, TypeError):
            return "₹0.00" if symbol else "0.00"

        is_negative = val < 0
        abs_val = abs(val)

        # Use Decimal for exact half-up 2-decimal rounding
        d_val = Decimal(str(abs_val)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        s_val = f"{d_val:.2f}"
        parts = s_val.split(".")
        int_part = parts[0]
        dec_part = parts[1]

        if len(int_part) <= 3:
            formatted_int = int_part
        else:
            last3 = int_part[-3:]
            remainder = int_part[:-3]
            # Group remainder in chunks of 2 from right to left
            chunks = []
            while len(remainder) > 2:
                chunks.append(remainder[-2:])
                remainder = remainder[:-2]
            if remainder:
                chunks.append(remainder)
            chunks.reverse()
            formatted_int = ",".join(chunks) + "," + last3

        res = f"{formatted_int}.{dec_part}"
        if is_negative:
            res = f"-{res}"
        if symbol:
            res = f"₹{res}"
        return res

    @classmethod
    def format_usd(cls, amount: float, symbol: bool = True) -> str:
        """
        Format a numeric value using US Numbering System (e.g. $1,250,000.00).
        """
        if amount is None:
            return "$0.00" if symbol else "0.00"
        try:
            val = float(amount)
        except (ValueError, TypeError):
            return "$0.00" if symbol else "0.00"

        is_negative = val < 0
        abs_val = abs(val)
        d_val = Decimal(str(abs_val)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        formatted = f"{d_val:,.2f}"
        if is_negative:
            formatted = f"-{formatted}"
        if symbol:
            formatted = f"${formatted}"
        return formatted

    @classmethod
    def format_currency(cls, amount: float, currency: str = "INR", symbol: bool = True) -> str:
        """
        Format amount using standard Indian number grouping for INR, or standard US formatting for USD.
        """
        cur = str(currency or "INR").strip().upper()
        if cur == "USD":
            return cls.format_usd(amount, symbol=symbol)
        return cls.format_inr(amount, symbol=symbol)

    @classmethod
    def convert_claim_inputs_to_usd(
        cls,
        claim_dict: Dict[str, Any],
        default_currency: str = "INR",
        inr_per_usd: Optional[float] = None,
        rate: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Apply currency normalization at the application input boundary before ML inference.
        Guarantees:
          - Only converts when currency is INR.
          - Does not convert historical USD claims.
          - Guards against double-conversion via _currency_converted flag.
          - Preserves original amounts and currency metadata.
        """
        out = dict(claim_dict)

        # Double-conversion check
        if out.get("_currency_converted") is True:
            return out

        cur = str(out.get("Currency", out.get("currency", default_currency))).strip().upper()
        effective_rate = rate if rate is not None else inr_per_usd

        if cur == "INR":
            raw_reimb = float(out.get("InscClaimAmtReimbursed", 0.0) or 0.0)
            raw_ded = float(out.get("DeductibleAmtPaid", 0.0) or 0.0)

            active_rate = cls.get_exchange_rate(effective_rate)
            usd_reimb = cls.inr_to_usd(raw_reimb, inr_per_usd=active_rate)
            usd_ded = cls.inr_to_usd(raw_ded, inr_per_usd=active_rate)

            out["_original_currency"] = "INR"
            out["_original_reimbursement"] = raw_reimb
            out["_original_deductible"] = raw_ded
            out["original_currency"] = "INR"
            out["original_reimbursement"] = raw_reimb
            out["original_deductible"] = raw_ded
            out["currency"] = "INR"
            out["_exchange_rate_applied"] = active_rate
            out["_exchange_rate_source"] = cls.BENCHMARK_SOURCE
            out["InscClaimAmtReimbursed"] = usd_reimb
            out["DeductibleAmtPaid"] = usd_ded
            out["_currency_converted"] = True
        else:
            # Preserved USD claim (e.g. historical CMS benchmark)
            out["_original_currency"] = "USD"
            out["_original_reimbursement"] = float(out.get("InscClaimAmtReimbursed", 0.0) or 0.0)
            out["_original_deductible"] = float(out.get("DeductibleAmtPaid", 0.0) or 0.0)
            out["original_currency"] = "USD"
            out["original_reimbursement"] = float(out.get("InscClaimAmtReimbursed", 0.0) or 0.0)
            out["original_deductible"] = float(out.get("DeductibleAmtPaid", 0.0) or 0.0)
            out["currency"] = "USD"
            out["_exchange_rate_applied"] = 1.0
            out["_currency_converted"] = True

        return out


currency_service = CurrencyService()

# Module-level aliases
get_exchange_rate = CurrencyService.get_exchange_rate
inr_to_usd = CurrencyService.inr_to_usd
usd_to_inr = CurrencyService.usd_to_inr
format_inr = CurrencyService.format_inr
format_usd = CurrencyService.format_usd
format_currency = CurrencyService.format_currency
convert_claim_inputs_to_usd = CurrencyService.convert_claim_inputs_to_usd
validate_exchange_rate = CurrencyService.validate_exchange_rate

