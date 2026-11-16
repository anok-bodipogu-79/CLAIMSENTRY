"""
HealthGuard AI - Phase 4 Authentication & Authorization Tests
Verifies user registration, login, token generation, /me, and role protection.
"""

from backend.app.core.security import create_access_token


def test_user_registration_success(client):
    """Verify new user registration with hashed password."""
    payload = {
        "email": "new_investigator@healthguard.ai",
        "password": "StrongPassword999!",
        "name": "Dr. Sarah Connor",
        "role": "investigator",
    }
    response = client.post("/api/auth/register", json=payload)
    assert response.status_code == 201
    data = response.json()
    assert data["email"] == payload["email"]
    assert data["name"] == payload["name"]
    assert "password" not in data
    assert "password_hash" not in data


def test_user_registration_duplicate_email(client, test_user):
    """Verify duplicate email registration is rejected with 400."""
    payload = {
        "email": test_user.email,
        "password": "AnotherPassword123!",
        "name": "Imposter User",
    }
    response = client.post("/api/auth/register", json=payload)
    assert response.status_code == 400
    assert "already exists" in response.json()["detail"].lower()


def test_login_success(client, test_user):
    """Verify login with correct credentials yields valid Bearer token."""
    response = client.post(
        "/api/auth/login",
        json={"email": test_user.email, "password": "SecurePass123!"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"
    assert data["user"]["email"] == test_user.email


def test_login_invalid_password(client, test_user):
    """Verify login with incorrect password returns 401."""
    response = client.post(
        "/api/auth/login",
        json={"email": test_user.email, "password": "WrongPassword!"},
    )
    assert response.status_code == 401
    assert "invalid email or password" in response.json()["detail"].lower()


def test_login_nonexistent_user(client):
    """Verify login with non-existent user returns 401."""
    response = client.post(
        "/api/auth/login",
        json={"email": "ghost@healthguard.ai", "password": "AnyPassword123!"},
    )
    assert response.status_code == 401


def test_get_current_user_me(client, auth_headers, test_user):
    """Verify /api/auth/me returns current user details with valid token."""
    response = client.get("/api/auth/me", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert data["email"] == test_user.email
    assert data["role"] == test_user.role


def test_get_current_user_invalid_token(client):
    """Verify /api/auth/me rejects invalid token with 401."""
    headers = {"Authorization": "Bearer invalid.fake.token"}
    response = client.get("/api/auth/me", headers=headers)
    assert response.status_code == 401


def test_protected_endpoint_requires_auth(client):
    """Verify protected endpoints reject unauthenticated requests."""
    response = client.get("/api/providers")
    assert response.status_code == 401


def test_registration_forces_investigator_role(client):
    """Verify registration strictly assigns investigator role even if payload requests admin."""
    payload = {
        "email": "attempted_admin@healthguard.ai",
        "password": "StrongPassword999!",
        "name": "Role Hacker",
        "role": "admin"  # Malicious attempt to register as admin
    }
    response = client.post("/api/auth/register", json=payload)
    assert response.status_code == 201
    data = response.json()
    assert data["role"].lower() == "investigator", "Public registration must assign investigator role"


def test_logout_revokes_token(client, auth_headers):
    """Verify logging out revokes the token so subsequent requests return 401."""
    # 1. Successful request before logout
    res_before = client.get("/api/auth/me", headers=auth_headers)
    assert res_before.status_code == 200

    # 2. Call logout
    logout_res = client.post("/api/auth/logout", headers=auth_headers)
    assert logout_res.status_code == 200
    assert logout_res.json()["status"] == "success"

    # 3. Subsequent request with the same token must be rejected with 401
    res_after = client.get("/api/auth/me", headers=auth_headers)
    assert res_after.status_code == 401
    assert "revoked" in res_after.json()["detail"].lower()


def test_promote_user_requires_admin(client, auth_headers, admin_headers, test_user):
    """Verify only authorized administrators can promote users or assign roles."""
    # 1. Non-admin (investigator) attempt must be rejected with 403
    forbidden_res = client.put(
        f"/api/auth/users/{test_user.id}/role",
        json={"role": "admin"},
        headers=auth_headers
    )
    assert forbidden_res.status_code == 403

    # 2. Admin attempt succeeds
    success_res = client.put(
        f"/api/auth/users/{test_user.id}/role",
        json={"role": "admin"},
        headers=admin_headers
    )
    assert success_res.status_code == 200
    assert success_res.json()["role"] == "admin"


def test_settings_modification_restricted_to_admin(client, auth_headers, admin_headers):
    """Verify model version switching and risk threshold updates require administrator role."""
    # Investigator attempt -> 403
    inv_res = client.put(
        "/api/settings",
        json={"model_version": "v1"},
        headers=auth_headers
    )
    assert inv_res.status_code == 403

    # Admin attempt -> 200
    admin_res = client.put(
        "/api/settings",
        json={"model_version": "v1"},
        headers=admin_headers
    )
    assert admin_res.status_code == 200
    assert admin_res.json()["model_version"] == "v1"

    # Revert back to v2
    client.put("/api/settings", json={"model_version": "v2"}, headers=admin_headers)


def test_csv_upload_ownership_enforcement(client, auth_headers, admin_headers, test_user, db_session):
    """Verify users cannot access CSV uploads owned by other users, while admins have oversight."""
    from backend.app.db.models import CSVUpload, User
    from backend.app.core.security import hash_password, create_access_token

    # Create a second user (Other Investigator)
    other_user = User(
        name="Other Investigator",
        email="other@healthguard.ai",
        password_hash=hash_password("Pass1234!"),
        role="investigator",
        is_active=True
    )
    db_session.add(other_user)
    db_session.commit()

    other_token = create_access_token(data={"sub": other_user.email, "role": other_user.role, "uid": other_user.id})
    other_headers = {"Authorization": f"Bearer {other_token}"}

    # Create an upload owned by test_user
    upload = CSVUpload(
        user_id=test_user.id,
        filename="user1_claims.csv",
        status="COMPLETED",
        total_rows=1,
        successful_rows=1,
        rejected_rows=0
    )
    db_session.add(upload)
    db_session.commit()

    # 1. Owner can access
    owner_res = client.get(f"/api/analyzer/csv/{upload.id}", headers=auth_headers)
    assert owner_res.status_code == 200

    # 2. Other investigator is rejected with 403 Forbidden
    other_res = client.get(f"/api/analyzer/csv/{upload.id}", headers=other_headers)
    assert other_res.status_code == 403
    assert "not have permission" in other_res.json()["detail"].lower()

    # 3. Administrator has oversight
    admin_res = client.get(f"/api/analyzer/csv/{upload.id}", headers=admin_headers)
    assert admin_res.status_code == 200
