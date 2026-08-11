import time

import bcrypt
import jwt
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from app.core.config import get_settings

ALGORITHM = "HS256"
TOKEN_TTL_SECONDS = 7 * 24 * 3600

# Requests to these exact paths skip auth. Everything else — every /ingest, /signals,
# /backtest, /paper-trade call — requires a valid bearer token, since this is a
# single-user tool with real (if demo-account) trade execution behind /paper-trade and
# a limited Twelve Data quota behind /ingest.
PUBLIC_PATHS = {"/", "/auth/login"}


def hash_password(plaintext: str) -> str:
    return bcrypt.hashpw(plaintext.encode(), bcrypt.gensalt()).decode()


def verify_credentials(username: str, password: str) -> bool:
    settings = get_settings()
    if not settings.auth_username or not settings.auth_password_hash:
        return False
    if username != settings.auth_username:
        return False
    return bcrypt.checkpw(password.encode(), settings.auth_password_hash.encode())


def create_token(username: str) -> str:
    settings = get_settings()
    payload = {"sub": username, "exp": int(time.time()) + TOKEN_TTL_SECONDS}
    return jwt.encode(payload, settings.auth_secret_key, algorithm=ALGORITHM)


def verify_token(token: str) -> bool:
    settings = get_settings()
    if not settings.auth_secret_key:
        return False
    try:
        jwt.decode(token, settings.auth_secret_key, algorithms=[ALGORITHM])
    except jwt.PyJWTError:
        return False
    return True


class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.method == "OPTIONS" or request.url.path in PUBLIC_PATHS:
            return await call_next(request)

        settings = get_settings()
        service_token = request.headers.get("x-service-token", "")
        if settings.auth_secret_key2 and service_token == settings.auth_secret_key2:
            return await call_next(request)

        auth_header = request.headers.get("authorization", "")
        if not auth_header.startswith("Bearer ") or not verify_token(auth_header.removeprefix("Bearer ")):
            return JSONResponse({"detail": "Not authenticated"}, status_code=401)

        return await call_next(request)
