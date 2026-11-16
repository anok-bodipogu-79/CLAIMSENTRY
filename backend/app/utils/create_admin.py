"""
HealthGuard AI - Secure Administrator Account Provisioning Utility
Creates or updates an administrator account directly in the database.
"""

import sys
import argparse
import getpass
from pathlib import Path

# Ensure project root is in sys.path
ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

from backend.app.db.database import SessionLocal, init_db
from backend.app.db.models import User
from backend.app.core.security import hash_password


def provision_admin(name: str, email: str, password: str) -> None:
    init_db()
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.email == email.strip().lower()).first()
        hashed = hash_password(password)
        if user:
            user.name = name.strip()
            user.password_hash = hashed
            user.role = "admin"
            user.is_active = True
            db.commit()
            print(f"[SUCCESS] User '{email}' promoted/updated to administrator role.")
        else:
            new_user = User(
                name=name.strip(),
                email=email.strip().lower(),
                password_hash=hashed,
                role="admin",
                is_active=True,
            )
            db.add(new_user)
            db.commit()
            print(f"[SUCCESS] Administrator '{email}' created successfully.")
    finally:
        db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Provision HealthGuard AI Administrator")
    parser.add_argument("--name", default="System Administrator", help="Admin Full Name")
    parser.add_argument("--email", required=True, help="Admin Email Address")
    parser.add_argument("--password", help="Admin Password (prompts securely if omitted)")

    args = parser.parse_args()
    pwd = args.password
    if not pwd:
        pwd = getpass.getpass("Enter Administrator Password: ")
        pwd_confirm = getpass.getpass("Confirm Administrator Password: ")
        if pwd != pwd_confirm:
            print("[ERROR] Passwords do not match.")
            sys.exit(1)

    if len(pwd) < 8:
        print("[ERROR] Password must be at least 8 characters long.")
        sys.exit(1)

    provision_admin(args.name, args.email, pwd)
