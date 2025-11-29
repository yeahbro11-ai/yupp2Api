"""Routers exposing model listing endpoints."""

from __future__ import annotations

import time

from fastapi import APIRouter, Depends

from ..auth import authenticate_client
from ..dependencies import get_runtime_state
from ..models import ModelInfo, ModelList
from ..state import RuntimeState

router = APIRouter()


def _build_model_list(state: RuntimeState) -> ModelList:
    data = [
        ModelInfo(
            id=model.get("label", "unknown"),
            created=int(time.time()),
            owned_by=model.get("publisher", "unknown"),
        )
        for model in state.models
    ]
    return ModelList(data=data)


@router.get("/v1/models", response_model=ModelList)
async def list_v1_models(
    state: RuntimeState = Depends(get_runtime_state),
    _: None = Depends(authenticate_client),
) -> ModelList:
    """List available models - authenticated."""
    return _build_model_list(state)


@router.get("/models", response_model=ModelList)
async def list_models_no_auth(
    state: RuntimeState = Depends(get_runtime_state),
) -> ModelList:
    """List available models without authentication for compatibility."""
    return _build_model_list(state)
