"""Smoke tests for main FastAPI routes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest
from fastapi.testclient import TestClient

from yupp2api.app import create_app
from yupp2api.config import Settings
from yupp2api.models import ChatCompletionChoice, ChatCompletionResponse, ChatMessage


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    model_file = tmp_path / "model.json"
    model_data = [
        {
            "label": "test-model",
            "name": "internal-model-name",
            "publisher": "test",
        }
    ]
    model_file.write_text(json.dumps(model_data), encoding="utf-8")
    return Settings(
        client_api_keys=["sk-test"],
        yupp_tokens=["token-test"],
        model_file=model_file,
        host="127.0.0.1",
        port=8001,
        debug_mode=False,
        max_error_count=3,
        error_cooldown=1,
    )


@pytest.fixture()
def client(settings: Settings):
    app = create_app(settings)
    with TestClient(app) as test_client:
        yield test_client


def test_models_endpoint_returns_data(client) -> None:
    response = client.get("/models")
    assert response.status_code == 200
    data = response.json()
    assert data["data"][0]["id"] == "test-model"


def test_chat_completion_non_stream(monkeypatch, client) -> None:
    def fake_session_post(*_: Any, **__: Any):
        class _DummyResponse:
            status_code = 200

            def iter_lines(self):
                return iter([])

            def raise_for_status(self):
                return None

        return _DummyResponse()

    class _DummySession:
        def post(self, *args: Any, **kwargs: Any):
            return fake_session_post(*args, **kwargs)

    monkeypatch.setattr(
        "yupp2api.routers.chat_router.create_requests_session",
        lambda settings: _DummySession(),
    )

    fake_response = ChatCompletionResponse(
        model="test-model",
        choices=[ChatCompletionChoice(message=ChatMessage(role="assistant", content="hello"))],
    )

    monkeypatch.setattr(
        "yupp2api.routers.chat_router.build_yupp_non_stream_response",
        lambda *args, **kwargs: fake_response,
    )

    payload: Dict[str, Any] = {
        "model": "test-model",
        "messages": [
            {"role": "user", "content": "hi"},
        ],
        "stream": False,
    }

    response = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer sk-test"},
        json=payload,
    )
    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "hello"
