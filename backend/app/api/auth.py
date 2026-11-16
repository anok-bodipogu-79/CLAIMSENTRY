"""
HealthGuard AI - Authentication API Endpoints
"""

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from backend.app.core.security import hash_password, verify_password, create_access_token
from backend.app.core.dependencies import get_current_user, get_current_admin, oauth2_scheme
from backend.app.db.database import get_db
from backend.app.db.models import User, RevokedToken
from backend.app.db.schemas import (
    UserRegisterRequest,
    UserLoginRequest,
    UserResponse,
    TokenResponse,
    UserRoleUpdateRequest,
)

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def register(payload: UserRegisterRequest, db: Session = Depends(get_db)):
    """Register a new user account with securely hashed credentials. Role is strictly assigned as investigator."""
    existing_user = db.query(User).filter(User.email == payload.email).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A user with this email address already exists.",
        )

    pwd_hash = hash_password(payload.password)
    user = User(
        name=payload.name,
        email=payload.email,
        password_hash=pwd_hash,
        role="investigator",  # Public registration strictly defaults to investigator role
        is_active=True,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@router.post("/login", response_model=TokenResponse)
def login(payload: UserLoginRequest, db: Session = Depends(get_db)):
    """Authenticate with email and password to receive a JWT access token."""
    user = db.query(User).filter(User.email == payload.email).first()
    if not user or not verify_password(payload.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is deactivated.",
        )

    token = create_access_token(data={"sub": user.email, "role": user.role, "uid": user.id})
    return {
        "access_token": token,
        "token_type": "bearer",
        "user": user,
    }


@router.get("/me", response_model=UserResponse)
def get_current_user_profile(current_user: User = Depends(get_current_user)):
    """Get profile of the currently authenticated user."""
    return current_user


@router.post("/logout")
def logout(
    token: str = Depends(oauth2_scheme),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Secure logout by blacklisting the active token."""
    existing = db.query(RevokedToken).filter(RevokedToken.token == token).first()
    if not existing:
        revoked = RevokedToken(token=token)
        db.add(revoked)
        db.commit()
    return {
        "message": f"Successfully logged out user {current_user.email}. Session revoked.",
        "status": "success",
    }


@router.put("/users/{user_id}/role", response_model=UserResponse)
def update_user_role(
    user_id: int,
    payload: UserRoleUpdateRequest,
    current_admin: User = Depends(get_current_admin),
    db: Session = Depends(get_db),
):
    """Restrict administrator creation and promotion to authorized administrators."""
    target_user = db.query(User).filter(User.id == user_id).first()
    if not target_user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User ID {user_id} not found."
        )

    allowed_roles = ["investigator", "admin", "auditor"]
    if payload.role.lower() not in allowed_roles:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid role '{payload.role}'. Must be one of {allowed_roles}."
        )

    target_user.role = payload.role.lower()
    db.commit()
    db.refresh(target_user)
    return target_user
