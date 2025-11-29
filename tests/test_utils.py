"""Tests for utility functions."""

from __future__ import annotations

from yupp2api.models import ChatMessage
from yupp2api.utils import clean_model_name, format_messages_for_yupp


def test_format_messages_single_user() -> None:
    messages = [ChatMessage(role="user", content="Hello")]
    result = format_messages_for_yupp(messages)
    assert "Human: Hello" in result
    assert result.endswith("Assistant:")


def test_format_messages_with_system() -> None:
    messages = [
        ChatMessage(role="system", content="Be helpful"),
        ChatMessage(role="user", content="Hello"),
    ]
    result = format_messages_for_yupp(messages)
    assert result.startswith("Be helpful")
    assert "Human: Hello" in result


def test_clean_model_name() -> None:
    assert clean_model_name("gpt-4\n\n") == "gpt-4"
    assert clean_model_name("model 🔥 test") == "model test"
