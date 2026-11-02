"""Password hashing, session tokens, credential encryption, login throttling and SSRF protection."""
import base64
import hashlib
import ipaddress
import json
import socket
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from urllib.parse import urlparse

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError
from cryptography.fernet import Fernet
from fastapi import Cookie, Depends, Header, HTTPException, status
from sqlalchemy.orm import Session

from .config import settings
from .db import User, get_db

COOKIE = "nexus_session"
_hasher = PasswordHasher()


def hash_password(password: str) -> str:
    return _hasher.hash(password)


def verify_password(hashed: str, password: str) -> bool:
    try:
        return _hasher.verify(hashed, password)
    except (VerificationError, InvalidHashError):
        return False


def make_token(user: User) -> str:
    exp = datetime.now(UTC) + timedelta(minutes=settings().token_ttl_minutes)
    return jwt.encode({"sub": user.id, "exp": exp}, settings().secret_key, algorithm="HS256")


@dataclass
class Principal:
    """Who is asking. Every data access takes one of these, so tenancy cannot be forgotten."""
    id: str
    workspace_id: str
    role: str
    name: str = ""
    email: str = ""

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"


def current_user(
    db: Session = Depends(get_db),
    session: str | None = Cookie(default=None, alias=COOKIE),
    authorization: str | None = Header(default=None),
) -> Principal:
    token = session
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization[7:]
    denied = HTTPException(status.HTTP_401_UNAUTHORIZED, "Not authenticated")
    if not token:
        raise denied
    try:
        claims = jwt.decode(token, settings().secret_key, algorithms=["HS256"])
    except jwt.PyJWTError:
        raise denied
    user = db.get(User, claims.get("sub"))
    if not user or not user.active:
        raise denied
    return Principal(user.id, user.workspace_id, user.role, user.name, user.email)


def require_admin(p: Principal = Depends(current_user)) -> Principal:
    if not p.is_admin:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Admin role required")
    return p


def _fernet() -> Fernet:
    return Fernet(base64.urlsafe_b64encode(hashlib.sha256(settings().secret_key.encode()).digest()))


def encrypt(data: dict) -> str:
    return _fernet().encrypt(json.dumps(data).encode()).decode() if data else ""


def decrypt(blob: str) -> dict:
    return json.loads(_fernet().decrypt(blob.encode())) if blob else {}


class Throttle:
    """Sliding-window limiter, per key, in process."""

    def __init__(self, limit: int, seconds: float):
        self.limit, self.seconds = limit, seconds
        self.hits: dict[str, deque] = defaultdict(deque)

    def check(self, key: str) -> None:
        q, t = self.hits[key], time.monotonic()
        while q and t - q[0] > self.seconds:
            q.popleft()
        if len(q) >= self.limit:
            raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "Too many attempts; try again shortly")
        q.append(t)


login_throttle = Throttle(10, 300)


def assert_public_url(url: str) -> str:
    """Refuse URLs that resolve to internal addresses, so crawling cannot reach private services."""
    parts = urlparse(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise ValueError("Only http(s) URLs are allowed")
    try:
        infos = socket.getaddrinfo(parts.hostname, parts.port or 443)
    except socket.gaierror as e:
        raise ValueError(f"Cannot resolve {parts.hostname}") from e
    for info in infos:
        ip = ipaddress.ip_address(info[4][0].split("%")[0])
        if not ip.is_global:
            raise ValueError("URL resolves to a non-public address")
    return url
