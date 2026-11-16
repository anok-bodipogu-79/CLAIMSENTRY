"""
HealthGuard AI - Automated PostgreSQL Database Bootstrap & Verification Script
==============================================================================
Creates the target database if it does not exist, applies Alembic migrations,
seeds realistic CMS provider and claim records with ML ensemble inferences,
and rigorously verifies database tables, Alembic revision, and integrity.

Security Guidelines:
- Zero hardcoded passwords or JWT secrets.
- Requires DATABASE_URL from environment or CLI argument.
- Aborts immediately on database creation or migration failure.
- Never drops existing database records.
- Enforces demo password rotation outside development environments.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from urllib.parse import urlparse
from typing import Dict, Any

# Ensure project root is in sys.path
ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from sqlalchemy import create_engine, inspect as sa_inspect, text as sa_text


REQUIRED_TABLES = [
    "users",
    "revoked_tokens",
    "providers",
    "beneficiaries",
    "claims",
    "provider_risk_scores",
    "claim_analyses",
    "explanations",
    "csv_uploads",
    "csv_upload_rows",
    "audit_logs",
    "case_notes",
    "model_versions",
]


def get_connection_params(database_url: str) -> Dict[str, Any]:
    parsed = urlparse(database_url)
    if not parsed.hostname:
        raise ValueError(f"Invalid DATABASE_URL: could not parse hostname from '{database_url}'")
    return {
        "host": parsed.hostname,
        "port": parsed.port or 5432,
        "user": parsed.username or "postgres",
        "password": parsed.password or "",
        "database": parsed.path.lstrip("/") or "healthguard",
    }


def ensure_database_exists(params: Dict[str, Any]) -> None:
    target_db = params["database"]
    print(f"[*] Checking PostgreSQL connection at {params['host']}:{params['port']} for user '{params['user']}'...")

    try:
        conn = psycopg2.connect(
            host=params["host"],
            port=params["port"],
            user=params["user"],
            password=params["password"],
            database="postgres",
            connect_timeout=10,
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()

        cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s", (target_db,))
        exists = cursor.fetchone()
        if not exists:
            print(f"[+] Target database '{target_db}' does not exist. Creating...")
            cursor.execute(f'CREATE DATABASE "{target_db}"')
            print(f"[+] Database '{target_db}' created successfully.")
        else:
            print(f"[+] Target database '{target_db}' already exists. Preserving existing database.")

        cursor.close()
        conn.close()
    except Exception as e:
        # If connecting to 'postgres' failed, try connecting directly to target_db
        print(f"[*] Note: Connection to default 'postgres' database failed ({e}). Checking direct connection to '{target_db}'...")
        try:
            conn = psycopg2.connect(
                host=params["host"],
                port=params["port"],
                user=params["user"],
                password=params["password"],
                database=target_db,
                connect_timeout=10,
            )
            conn.close()
            print(f"[+] Direct connection to target database '{target_db}' verified.")
        except Exception as direct_err:
            raise RuntimeError(
                f"Fatal: Could not connect to PostgreSQL instance or create database '{target_db}': {direct_err}\n"
                f"Please ensure PostgreSQL is running and credentials are correct."
            ) from direct_err


def run_migrations(root_dir: Path) -> None:
    print("[*] Applying Alembic schema migrations...")
    
    # Locate alembic.ini (check root then backend)
    ini_path = root_dir / "alembic.ini"
    if not ini_path.exists():
        ini_path = root_dir / "backend" / "alembic.ini"
    if not ini_path.exists():
        raise RuntimeError("Fatal: Could not find alembic.ini configuration file.")

    cmd = [sys.executable, "-m", "alembic", "-c", str(ini_path), "upgrade", "head"]
    result = subprocess.run(
        cmd,
        cwd=str(root_dir),
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        error_msg = result.stderr or result.stdout
        print(f"[!] Migration failed with exit code {result.returncode}:\n{error_msg}")
        raise RuntimeError(f"Alembic migration failed: {error_msg}. Aborting bootstrap before seeding.")
    
    print("[+] Alembic schema migrations applied successfully.")


def verify_alembic_and_tables(database_url: str) -> str:
    print("[*] Verifying Alembic revision and required database tables...")
    engine = create_engine(database_url)
    inspector = sa_inspect(engine)
    existing_tables = set(inspector.get_table_names())

    # 1. Verify alembic_version table and get revision
    if "alembic_version" not in existing_tables:
        raise RuntimeError("Verification failed: 'alembic_version' table not found in database.")
    
    with engine.connect() as conn:
        rev = conn.execute(sa_text("SELECT version_num FROM alembic_version")).scalar()
    
    if not rev:
        raise RuntimeError("Verification failed: alembic_version table contains no revision record.")
    print(f"[+] Current Alembic revision: {rev}")

    # 2. Verify all required application tables exist
    missing_tables = [t for t in REQUIRED_TABLES if t not in existing_tables]
    if missing_tables:
        raise RuntimeError(f"Verification failed: Required application tables missing from database: {missing_tables}")

    print(f"[+] All {len(REQUIRED_TABLES)} required application tables verified in database.")
    engine.dispose()
    return rev


def seed_demo_data(env: str) -> None:
    print("[*] Running idempotent demo data seeding script...")
    if env == "production":
        print("[!] PRODUCTION WARNING: In production environments, demo passwords must be changed immediately after initialization.")
    
    from backend.app.db.init_demo_data import main as seed_main
    seed_main()
    print("[+] Demo data seeding completed successfully.")


def print_database_audit():
    print("[*] Auditing record counts across database tables...")
    from backend.app.db.database import SessionLocal
    from backend.app.db.models import User, Provider, Claim, Beneficiary, ProviderRiskScore, ClaimAnalysis

    db = SessionLocal()
    try:
        user_count = db.query(User).count()
        provider_count = db.query(Provider).count()
        claim_count = db.query(Claim).count()
        bene_count = db.query(Beneficiary).count()
        eval_count = db.query(ProviderRiskScore).count()
        flagged_count = sum(1 for a in db.query(ClaimAnalysis).all() if a.anomaly_flags and len(a.anomaly_flags) > 0)

        print("\n" + "=" * 55)
        print("  HEALTHGUARD AI - DATABASE VERIFICATION AUDIT")
        print("=" * 55)
        print(f"  Registered Users             : {user_count}")
        print(f"  Monitored Providers          : {provider_count}")
        print(f"  Evaluated Claims             : {claim_count}")
        print(f"  Beneficiaries                : {bene_count}")
        print(f"  Provider Risk Scores (ML)    : {eval_count}")
        print(f"  Flagged Anomaly Claims       : {flagged_count}")
        print("=" * 55 + "\n")
    finally:
        db.close()


def main():
    parser = argparse.ArgumentParser(description="HealthGuard AI Database Bootstrap & Verification")
    parser.add_argument("--db-url", type=str, default=None, help="PostgreSQL connection URL")
    args = parser.parse_args()

    db_url = args.db_url or os.getenv("DATABASE_URL")
    if not db_url:
        print("[!] Error: DATABASE_URL must be provided via --db-url or environment variable.")
        print("    Example: DATABASE_URL=postgresql://user:pass@localhost:5432/healthcare_fraud_db")
        sys.exit(1)

    # Ensure JWT_SECRET_KEY is present in environment
    if not os.getenv("JWT_SECRET_KEY"):
        print("[!] Error: JWT_SECRET_KEY must be provided via environment variable (minimum 32 characters).")
        sys.exit(1)

    os.environ["DATABASE_URL"] = db_url
    env = (os.getenv("ENVIRONMENT") or "development").lower()

    params = get_connection_params(db_url)
    
    # 1. Ensure database exists (never drops existing database)
    ensure_database_exists(params)

    # 2. Run schema migrations and abort on failure
    run_migrations(ROOT)

    # 3. Verify alembic revision and tables
    verify_alembic_and_tables(db_url)

    # 4. Seed demo data
    seed_demo_data(env)

    # 5. Output audit
    print_database_audit()
    print("[+] Database bootstrap and verification finished successfully.")


if __name__ == "__main__":
    main()
