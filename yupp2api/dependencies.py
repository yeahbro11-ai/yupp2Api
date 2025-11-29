"""Common FastAPI dependencies."""

from __future__ import annotations

from fastapi import Request

from .config import Settings
from .state import RuntimeState


def get_settings(request: Request) -> Settings:  # pragma: no cover - trivial accessor
    return request.app.state.settings  # type: ignore[attr-defined]


def get_runtime_state(request: Request) -> RuntimeState:  # pragma: no cover - trivial accessor
    return request.app.state.runtime_state  # type: ignore[attr-defined]
