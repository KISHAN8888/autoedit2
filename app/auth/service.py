import jwt
import bcrypt
import random
import string
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from motor.motor_asyncio import AsyncIOMotorCollection
import logging
from app.auth.models import UserRegister, UserLogin, ForgotPassword, ResetPassword, GetOTP, VerifyOTP
from app.config import get_settings

logger = logging.getLogger(__name__)

class AuthService:
    def __init__(self, users_collection: AsyncIOMotorCollection):
        self.users_collection = users_collection
        self.settings = get_settings()
        
        # JWT settings
        self.secret_key = self.settings.jwt_secret_key
        self.algorithm = "HS256"
        self.access_token_expire_minutes = 30
        
        # Password reset settings
        self.reset_token_expire_minutes = 15
        
        # OTP settings
        self.otp_expire_minutes = 10
        self.otp_length = 6

    def _generate_otp(self) -> str:
        """Generate a random 6-digit OTP"""
        return ''.join(random.choices(string.digits, k=self.otp_length))

    def _hash_password(self, password: str) -> str:
        """Hash a password using bcrypt"""
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        return hashed.decode('utf-8')

    def _verify_password(self, password: str, hashed_password: str) -> bool:
        """Verify a password against its hash"""
        return bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8'))

    def _create_access_token(self, data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
        """Create a JWT access token"""
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)
        
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt

    def _create_reset_token(self, email: str) -> str:
        """Create a password reset token"""
        expire = datetime.utcnow() + timedelta(minutes=self.reset_token_expire_minutes)
        to_encode = {"email": email, "exp": expire, "type": "password_reset"}
        return jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)

    def _convert_objectid_to_string(self, data: Any) -> Any:
        """Recursively convert ObjectId to string in nested data structures"""
        if hasattr(data, '_id'):
            data['_id'] = str(data['_id'])
        return data

    async def get_otp(self, otp_data: GetOTP) -> Dict[str, Any]:
        """Generate and send OTP for email verification during registration"""
        try:
            # Check if user already exists
            existing_user = await self.users_collection.find_one({"email": otp_data.email.lower()})
            if existing_user:
                raise ValueError("User with this email already exists")

            # Generate OTP
            otp = self._generate_otp()
            otp_expires = datetime.utcnow() + timedelta(minutes=self.otp_expire_minutes)

            # Store OTP in temporary collection or update existing
            await self.users_collection.update_one(
                {"email": otp_data.email.lower()},
                {
                    "$set": {
                        "email": otp_data.email.lower(),
                        "full_name": otp_data.full_name,
                        "otp": otp,
                        "otp_expires": otp_expires,
                        "created_at": datetime.utcnow(),
                        "updated_at": datetime.utcnow()
                    }
                },
                upsert=True
            )

            # TODO: Send OTP via email in production
            # For now, just log it (remove in production)
            logger.info(f"OTP generated for {otp_data.email}: {otp}")

            return {
                "success": True,
                "message": "OTP sent to your email",
                "otp": otp  # Remove this in production
            }

        except Exception as e:
            logger.error(f"Error generating OTP: {e}")
            raise

    async def verify_otp_and_register(self, user_data: UserRegister) -> Dict[str, Any]:
        """Verify OTP and complete user registration"""
        try:
            # Find the temporary user record with OTP
            temp_user = await self.users_collection.find_one({
                "email": user_data.email.lower(),
                "otp": user_data.otp,
                "otp_expires": {"$gt": datetime.utcnow()}
            })
    
            if not temp_user:
                raise ValueError("Invalid OTP or OTP expired")
    
            # Hash password
            hashed_password = self._hash_password(user_data.password)
            
            # Update the existing document instead of inserting a new one
            await self.users_collection.update_one(
                {"_id": temp_user["_id"]},
                {
                    "$set": {
                        "password": hashed_password,
                        "is_active": True,
                        "is_verified": True,
                        "updated_at": datetime.utcnow()
                    },
                    "$unset": {
                        "otp": "",
                        "otp_expires": ""
                    }
                }
            )
            
            # Get the updated user document
            user_doc = await self.users_collection.find_one({"_id": temp_user["_id"]})
            
            # Remove password from response
            user_doc.pop("password", None)
            
            # Convert ObjectId to string
            user_doc["_id"] = str(user_doc["_id"])
            
            logger.info(f"User registered successfully with email verification: {user_data.email}")
            return {"success": True, "user": user_doc}
            
        except Exception as e:
            logger.error(f"Error in OTP verification and registration: {e}")
            raise
    
    async def authenticate_user(self, user_data: UserLogin) -> Optional[Dict[str, Any]]:
        """Authenticate a user and return user data if successful"""
        try:
            # Find user by email
            user = await self.users_collection.find_one({"email": user_data.email.lower()})
            if not user:
                return None
            
            # Check if user is active
            if not user.get("is_active", True):
                return None
            
            # Check if user is verified
            if not user.get("is_verified", False):
                raise ValueError("Email not verified. Please verify your email first.")
            
            # Verify password
            if not self._verify_password(user_data.password, user["password"]):
                return None
            
            # Remove password from response
            user.pop("password", None)
            
            # Convert ObjectId to string
            user["_id"] = str(user["_id"])
            
            logger.info(f"User authenticated successfully: {user_data.email}")
            return user
            
        except Exception as e:
            logger.error(f"Error authenticating user: {e}")
            raise

    async def login_user(self, user_data: UserLogin) -> Dict[str, Any]:
        """Login a user and return access token"""
        user = await self.authenticate_user(user_data)
        if not user:
            raise ValueError("Invalid email or password")
        
        # Create access token
        access_token_expires = timedelta(minutes=self.access_token_expire_minutes)
        access_token = self._create_access_token(
            data={"sub": str(user["_id"]), "email": user["email"]},
            expires_delta=access_token_expires
        )
        
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "expires_in": self.access_token_expire_minutes * 60,
            "user": user
        }

    async def forgot_password(self, forgot_data: ForgotPassword) -> Dict[str, Any]:
        """Initiate password reset process"""
        try:
            # Check if user exists
            user = await self.users_collection.find_one({"email": forgot_data.email.lower()})
            if not user:
                # Don't reveal if user exists or not for security
                return {"message": "If the email exists, a password reset link has been sent"}
            
            # Create reset token
            reset_token = self._create_reset_token(forgot_data.email)
            
            # Store reset token in user document
            await self.users_collection.update_one(
                {"email": forgot_data.email.lower()},
                {
                    "$set": {
                        "reset_token": reset_token,
                        "reset_token_expires": datetime.utcnow() + timedelta(minutes=self.reset_token_expire_minutes),
                        "updated_at": datetime.utcnow()
                    }
                }
            )
            
            # TODO: Send email with reset token
            # For now, just log it (in production, send email)
            logger.info(f"Password reset token created for {forgot_data.email}: {reset_token}")
            
            return {"message": "If the email exists, a password reset link has been sent"}
            
        except Exception as e:
            logger.error(f"Error in forgot password: {e}")
            raise

    async def reset_password(self, reset_data: ResetPassword) -> Dict[str, Any]:
        """Reset password using reset token"""
        try:
            # Decode and verify reset token
            try:
                payload = jwt.decode(reset_data.token, self.secret_key, algorithms=[self.algorithm])
                email = payload.get("email")
                token_type = payload.get("type")
                
                if not email or token_type != "password_reset":
                    raise ValueError("Invalid reset token")
                    
            except jwt.ExpiredSignatureError:
                raise ValueError("Reset token has expired")
            except jwt.InvalidTokenError:
                raise ValueError("Invalid reset token")
            
            # Find user and verify token
            user = await self.users_collection.find_one({
                "email": email,
                "reset_token": reset_data.token,
                "reset_token_expires": {"$gt": datetime.utcnow()}
            })
            
            if not user:
                raise ValueError("Invalid or expired reset token")
            
            # Hash new password
            hashed_password = self._hash_password(reset_data.new_password)
            
            # Update password and clear reset token
            await self.users_collection.update_one(
                {"email": email},
                {
                    "$set": {
                        "password": hashed_password,
                        "updated_at": datetime.utcnow()
                    },
                    "$unset": {
                        "reset_token": "",
                        "reset_token_expires": ""
                    }
                }
            )
            
            logger.info(f"Password reset successfully for {email}")
            return {"message": "Password reset successfully"}
            
        except Exception as e:
            logger.error(f"Error resetting password: {e}")
            raise

    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify JWT token and return payload"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("Token expired")
            return None
        except jwt.InvalidTokenError:
            logger.warning("Invalid token")
            return None