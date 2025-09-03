from fastapi import APIRouter, Depends, HTTPException, status
from app.auth.models import (
    UserRegister, UserLogin, ForgotPassword, ResetPassword,
    UserResponse, TokenResponse, PasswordResetResponse, GetOTP, VerifyOTP, OTPResponse
)
from app.auth.service import AuthService
from app.database.mongodb_manager import MongoDBManager
from app.config import get_settings
from typing import Dict, Any

router = APIRouter(prefix="/auth", tags=["Authentication"])

async def get_auth_service() -> AuthService:
    """Dependency to get auth service"""
    mongodb_manager = MongoDBManager(get_settings().mongodb_uri)
    await mongodb_manager.initialize()
    return AuthService(mongodb_manager.db.users)

@router.post("/get-otp", response_model=OTPResponse)
async def get_otp(
    otp_data: GetOTP,
    auth_service: AuthService = Depends(get_auth_service)
):
    """Get OTP for email verification during registration"""
    try:
        result = await auth_service.get_otp(otp_data)
        return result
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to generate OTP")

@router.post("/register", response_model=Dict[str, Any])
async def register(
    user_data: UserRegister,
    auth_service: AuthService = Depends(get_auth_service)
):
    """Register a new user with OTP verification"""
    try:
        result = await auth_service.verify_otp_and_register(user_data)
        return result
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Registration failed")

@router.post("/login", response_model=Dict[str, Any])
async def login(
    user_data: UserLogin,
    auth_service: AuthService = Depends(get_auth_service)
):
    """Login user and return access token"""
    try:
        result = await auth_service.login_user(user_data)
        return result
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Login failed")

@router.post("/forgot-password", response_model=PasswordResetResponse)
async def forgot_password(
    forgot_data: ForgotPassword,
    auth_service: AuthService = Depends(get_auth_service)
):
    """Initiate password reset process"""
    try:
        result = await auth_service.forgot_password(forgot_data)
        return result
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Password reset failed")

@router.post("/reset-password", response_model=PasswordResetResponse)
async def reset_password(
    reset_data: ResetPassword,
    auth_service: AuthService = Depends(get_auth_service)
):
    """Reset password using reset token"""
    try:
        result = await auth_service.reset_password(reset_data)
        return result
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Password reset failed")