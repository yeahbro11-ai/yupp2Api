"""Authentication dependencies for FastAPI routes."""

from __future__ import annotations

from typing import Optional

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .dependencies import get_runtime_state
from .state import RuntimeState

security = HTTPBearer(auto_error=False)


def authenticate_client(
    state: RuntimeState = Depends(get_runtime_state),
    auth: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> None:
    """Verify client API key from Authorization header."""
    if not state.valid_client_keys:
        raise HTTPException(
            status_code=503,
            detail="Service unavailable: Client API keys not configured on server.",
        )

    if not auth or not auth.credentials:
        raise HTTPException(
            status_code=401,
            detail="API key required in Authorization header.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if auth.credentials not in state.valid_client_keys:
        raise HTTPException(status_code=403, detail="Invalid client API key.")
