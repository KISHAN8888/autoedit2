# app/middleware/security.py
from fastapi import Request, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware
import time
import redis
import hashlib
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware using Redis"""
    
    def __init__(self, app, calls: int = 100, period: int = 60, redis_host: str = 'redis', redis_port: int = 6379):
        super().__init__(app)
        self.calls = calls
        self.period = period
        
        try:
            self.redis_client = redis.Redis(host=redis_host, port=redis_port, db=1, decode_responses=True)
            # Test connection
            self.redis_client.ping()
            logger.info(f"Rate limiting initialized: {calls} calls per {period} seconds")
        except redis.ConnectionError:
            logger.warning("Redis not available for rate limiting")
            self.redis_client = None

    async def dispatch(self, request: Request, call_next):
        if not self.redis_client:
            # If Redis is not available, skip rate limiting
            return await call_next(request)
        
        # Skip rate limiting for health checks
        if request.url.path == "/health":
            return await call_next(request)
        
        # Get client identifier (IP + User-Agent hash for better uniqueness)
        client_ip = request.client.host
        user_agent = request.headers.get("user-agent", "")
        client_id = hashlib.md5(f"{client_ip}:{user_agent}".encode()).hexdigest()
        
        key = f"rate_limit:{client_id}"
        
        try:
            current = self.redis_client.get(key)
            if current is None:
                # First request from this client
                self.redis_client.setex(key, self.period, 1)
            else:
                current = int(current)
                if current >= self.calls:
                    logger.warning(f"Rate limit exceeded for client {client_ip}")
                    raise HTTPException(
                        status_code=429, 
                        detail="Rate limit exceeded. Please try again later.",
                        headers={"Retry-After": str(self.period)}
                    )
                self.redis_client.incr(key)
        except redis.RedisError as e:
            logger.error(f"Redis error in rate limiting: {e}")
            # Continue without rate limiting if Redis fails
        
        response = await call_next(request)
        
        # Add rate limit headers
        if self.redis_client:
            try:
                remaining = max(0, self.calls - int(self.redis_client.get(key) or 0))
                response.headers["X-RateLimit-Limit"] = str(self.calls)
                response.headers["X-RateLimit-Remaining"] = str(remaining)
                response.headers["X-RateLimit-Reset"] = str(int(time.time()) + self.period)
            except:
                pass  # Don't fail the request if we can't add headers
        
        return response

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all responses"""
    
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        
        # Security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        
        # Only add HSTS in production with HTTPS
        if request.url.scheme == "https":
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        
        # Content Security Policy
        csp = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: https:; "
            "connect-src 'self'; "
            "frame-ancestors 'none';"
        )
        response.headers["Content-Security-Policy"] = csp
        
        return response

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log all incoming requests for monitoring"""
    
    def __init__(self, app, log_body: bool = False):
        super().__init__(app)
        self.log_body = log_body

    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        # Log request
        client_ip = request.client.host
        method = request.method
        url = str(request.url)
        user_agent = request.headers.get("user-agent", "")
        
        # Don't log health check requests to reduce noise
        if request.url.path != "/health":
            logger.info(f"Request: {method} {url} from {client_ip} - {user_agent}")
        
        response = await call_next(request)
        
        # Log response
        process_time = time.time() - start_time
        status_code = response.status_code
        
        if request.url.path != "/health":
            logger.info(f"Response: {status_code} - {process_time:.3f}s")
        
        # Add timing header
        response.headers["X-Process-Time"] = str(process_time)
        
        return response

class FileSizeValidationMiddleware(BaseHTTPMiddleware):
    """Validate file upload sizes before processing"""
    
    def __init__(self, app, max_size: int = 500 * 1024 * 1024):  # 500MB default
        super().__init__(app)
        self.max_size = max_size

    async def dispatch(self, request: Request, call_next):
        # Check Content-Length header for file uploads
        if request.method == "POST" and request.url.path == "/process-video":
            content_length = request.headers.get("content-length")
            if content_length:
                try:
                    size = int(content_length)
                    if size > self.max_size:
                        logger.warning(f"File upload too large: {size} bytes from {request.client.host}")
                        raise HTTPException(
                            status_code=413,
                            detail=f"File too large. Maximum size: {self.max_size // (1024*1024)}MB"
                        )
                except ValueError:
                    pass  # Invalid content-length header, let the app handle it
        
        return await call_next(request)

# Authentication middleware (optional, for protected endpoints)
class BearerTokenAuth(HTTPBearer):
    """Simple Bearer token authentication"""
    
    def __init__(self, valid_tokens: Optional[list] = None):
        super().__init__()
        self.valid_tokens = valid_tokens or []
    
    async def __call__(self, request: Request) -> Optional[HTTPAuthorizationCredentials]:
        # Skip authentication for public endpoints
        public_paths = ["/health", "/docs", "/openapi.json"]
        if request.url.path in public_paths:
            return None
            
        credentials = await super().__call__(request)
        
        if self.valid_tokens and credentials.credentials not in self.valid_tokens:
            raise HTTPException(
                status_code=403,
                detail="Invalid authentication token"
            )
        
        return credentials