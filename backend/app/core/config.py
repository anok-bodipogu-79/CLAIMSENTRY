"""
HealthGuard AI - Application Configuration & Environment Settings
"""

import os
import sys
from typing import List, Union, Optional
from urllib.parse import urlparse
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, model_validator


class Settings(BaseSettings):
    PROJECT_NAME: str = "HealthGuard AI"
    VERSION: str = "1.0.0"
    API_PREFIX: str = "/api"
    
    # Environment & Database selection
    ENVIRONMENT: str = Field(
        default="development",
        description="Application environment: production, development, or test"
    )
    USE_SQLITE: bool = Field(
        default=False,
        description="Explicit flag allowing SQLite only in development or test modes"
    )
    
    # Database - Required via environment outside test mode; no hardcoded credentials
    DATABASE_URL: Optional[str] = Field(
        default=None,
        description="Primary PostgreSQL database connection URL. Must be provided via environment outside test mode."
    )
    SQLITE_FALLBACK_URL: str = "sqlite:///./backend/healthguard.db"
    
    # JWT & Security - Required via environment outside test mode; no hardcoded secret
    JWT_SECRET_KEY: Optional[str] = Field(
        default=None,
        description="Secret key for JWT signature (must be >= 32 characters in production/dev)"
    )
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440  # 24 hours
    
    # CORS
    CORS_ORIGINS: Union[str, List[str]] = "http://localhost:3000,http://localhost:5173,http://localhost:8000"
    
    # Machine Learning Settings
    MODEL_VERSION: str = "v2"
    DEFAULT_LOW_MEDIUM_THRESHOLD: float = 0.22
    DEFAULT_MEDIUM_HIGH_THRESHOLD: float = 0.70
    
    # Currency & Exchange Rate Settings
    APPLICATION_CURRENCY: str = "INR"
    ML_TRAINING_CURRENCY: str = "USD"
    INR_PER_USD: Optional[float] = Field(
        default=83.50,
        description="Configurable exchange rate: INR per USD. Benchmark reference rate: 83.50 INR per USD."
    )
    EXCHANGE_RATE_SOURCE: str = "Reserve Bank of India (RBI) Reference Benchmark (Effective 2024-2025: 83.50 INR/USD)"
    EXCHANGE_RATE_EFFECTIVE_DATE: str = "2024-11-01"

    # Disclaimers & Standard Titles
    REGULATORY_DISCLAIMER: str = "This is a risk assessment to guide investigation. It is not proof of fraud."
    EXPLANATION_TITLE: str = "AI Explanation – Why This Claim Received This Risk Score"

    model_config = SettingsConfigDict(
        env_file=(".env", "../.env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

    @model_validator(mode="after")
    def validate_security_and_environment(self) -> "Settings":
        env = (self.ENVIRONMENT or os.getenv("ENVIRONMENT") or "development").lower()
        self.ENVIRONMENT = env
        
        is_test = (env == "test")

        # 1. Enforce Database URL configuration
        db_url = self.DATABASE_URL
        if is_test:
            if not db_url:
                db_url = os.getenv("DATABASE_URL") or "sqlite:///:memory:"
        else:
            if not db_url or not db_url.strip():
                raise ValueError(
                    "Configuration requirement violated: DATABASE_URL must be provided via environment variables. "
                    "HealthGuard AI enforces explicit database configuration without hardcoded fallback credentials."
                )
            if any(token in db_url for token in ["<DB_PASSWORD>", "<DB_USER>", "<DB_HOST>", "<DB_NAME>"]):
                raise ValueError(
                    "Configuration requirement violated: DATABASE_URL contains unconfigured template placeholders. "
                    "Please configure real database connection credentials in your environment."
                )
            if env == "production" and db_url.startswith("sqlite"):
                raise ValueError(
                    "Security requirement violated: SQLite is strictly prohibited as a database engine in production. "
                    "A valid PostgreSQL connection URL must be configured."
                )
        self.DATABASE_URL = db_url

        # 2. Enforce secure JWT Secret Key configuration
        secret = self.JWT_SECRET_KEY
        if is_test:
            if not secret:
                secret = os.getenv("JWT_SECRET_KEY") or "test-secret-key-32-chars-minimum-length-for-testing"
        else:
            if not secret or not secret.strip():
                raise ValueError(
                    "Security requirement violated: A secure JWT_SECRET_KEY must be provided via environment variables. "
                    "Hardcoded default JWT secrets are strictly prohibited outside test mode."
                )
            if len(secret.strip()) < 32:
                raise ValueError(
                    "Security requirement violated: JWT_SECRET_KEY must be at least 32 characters in length."
                )
            
            # Reject known default, placeholder, or trivially weak secrets
            rejected_secrets = {
                "default",
                "secret",
                "healthguard-super-secret",
                "healthguard-super-secret-jwt-key-change-in-production",
                "healthguard-super-secret-jwt-key-2026-production-ready",
                "<generate_and_insert_strong_random_secret_key_min_32_chars>",
                "12345678901234567890123456789012",
                "change-me-change-me-change-me-32",
                "passwordpasswordpasswordpassword",
            }
            cleaned = secret.strip().lower()
            if cleaned in rejected_secrets or "change-in-production" in cleaned or "changeme" in cleaned:
                raise ValueError(
                    "Security requirement violated: The configured JWT_SECRET_KEY matches a known default or placeholder value. "
                    "A cryptographically secure random secret must be configured."
                )
        self.JWT_SECRET_KEY = secret

        # 3. Validate exchange rate configuration if specified
        rate_val = self.INR_PER_USD
        if rate_val is not None:
            try:
                rate_float = float(rate_val)
                if rate_float <= 0.0:
                    raise ValueError("INR_PER_USD must be a positive number greater than 0.")
                self.INR_PER_USD = rate_float
            except (TypeError, ValueError) as rate_err:
                raise ValueError(f"Invalid currency configuration: INR_PER_USD must be a positive number (> 0): {rate_err}")

        return self

    def get_masked_database_url(self) -> str:
        """Return database URL with password safely masked for logs and exception reporting."""
        if not self.DATABASE_URL:
            return "<none>"
        try:
            parsed = urlparse(self.DATABASE_URL)
            if parsed.password:
                masked_netloc = f"{parsed.username}:****@{parsed.hostname}"
                if parsed.port:
                    masked_netloc += f":{parsed.port}"
                return self.DATABASE_URL.replace(parsed.netloc, masked_netloc)
            return self.DATABASE_URL
        except Exception:
            return "<masked_database_url>"

    @property
    def cors_origins_list(self) -> List[str]:
        if isinstance(self.CORS_ORIGINS, list):
            return self.CORS_ORIGINS
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",") if origin.strip()]


# In module import, initialize settings using test-safe defaults if in test mode
_init_kwargs = {}
if os.getenv("DATABASE_URL"):
    _init_kwargs["DATABASE_URL"] = os.getenv("DATABASE_URL")
elif "pytest" in sys.modules:
    _init_kwargs["DATABASE_URL"] = "sqlite:///:memory:"

if os.getenv("JWT_SECRET_KEY"):
    _init_kwargs["JWT_SECRET_KEY"] = os.getenv("JWT_SECRET_KEY")
elif "pytest" in sys.modules:
    _init_kwargs["JWT_SECRET_KEY"] = "test-secret-key-32-chars-minimum-length-for-testing"

if os.getenv("ENVIRONMENT"):
    _init_kwargs["ENVIRONMENT"] = os.getenv("ENVIRONMENT")
elif "pytest" in sys.modules:
    _init_kwargs["ENVIRONMENT"] = "test"

settings = Settings(**_init_kwargs)
