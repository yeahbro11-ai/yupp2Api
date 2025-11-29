"""FastAPI application factory for Yupp2API."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware

from .bootstrap import bootstrap_state
from .config import Settings, get_settings
from .routers import chat_router, models_router
from .state import RuntimeState


def create_app(settings: Optional[Settings] = None) -> FastAPI:
    settings = settings or get_settings()
    runtime_state = RuntimeState()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        print("Starting Yupp.ai OpenAI API Adapter server...")
        bootstrap_state(runtime_state, settings)
        print("Server initialization completed.")
        yield
        print("Server shutdown completed.")

    app = FastAPI(title="Yupp.ai OpenAI API Adapter", lifespan=lifespan)
    app.state.settings = settings
    app.state.runtime_state = runtime_state

    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(models_router.router)
    app.include_router(chat_router.router)

    return app
