from pydantic import BaseModel, EmailStr
from typing import Optional
from datetime import datetime

class UserRegister(BaseModel):
    email: EmailStr
    password: str
    full_name: str
    otp: str  # Add OTP field for registration

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class ForgotPassword(BaseModel):
    email: EmailStr

class ResetPassword(BaseModel):
    token: str
    new_password: str

class GetOTP(BaseModel):
    email: EmailStr
    full_name: str

class VerifyOTP(BaseModel):
    email: EmailStr
    otp: str

class UserResponse(BaseModel):
    id: str
    email: str
    full_name: str
    is_active: bool
    is_verified: bool  # Add verification status
    created_at: datetime
    updated_at: datetime

class TokenResponse(BaseModel):
    access_token: str
    token_type: str
    expires_in: int
    user: UserResponse

class PasswordResetResponse(BaseModel):
    message: str

class OTPResponse(BaseModel):
    message: str
    otp: str  # For development/testing - remove in production