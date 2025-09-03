from fastapi import HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional
from bson import ObjectId
from app.auth.service import AuthService
from app.database.mongodb_manager import MongoDBManager
from app.config import get_settings

security = HTTPBearer()

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    mongodb_manager: MongoDBManager = Depends(lambda: MongoDBManager(get_settings().mongodb_uri))
) -> dict:
    """Get current authenticated user from JWT token"""
    try:
        # Initialize MongoDB if not already done
        if not mongodb_manager.db:
            await mongodb_manager.initialize()
        
        # Create auth service
        auth_service = AuthService(mongodb_manager.db.users)
        
        # Verify token
        payload = auth_service.verify_token(credentials.credentials)
        if not payload:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired token",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Get user from database - Convert string ID to ObjectId
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        try:
            # Convert string ID to ObjectId
            object_id = ObjectId(user_id)
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid user ID format",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        user = await mongodb_manager.db.users.find_one({"_id": object_id})
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Check if user is active
        if not user.get("is_active", True):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User account is deactivated",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Check if user is verified
        if not user.get("is_verified", False):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Email not verified. Please verify your email first.",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Remove password from user object
        user.pop("password", None)
        
        # Convert ObjectId to string
        user["_id"] = str(user["_id"])
        
        return user
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )