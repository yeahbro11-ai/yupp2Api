"""Tests for configuration module."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from yupp2api.config import Settings


def test_settings_valid_required_fields() -> None:
    settings = Settings(
        client_api_keys=["sk-1"],
        yupp_tokens=["token-1"],
    )
    assert settings.client_api_keys == ["sk-1"]
    assert settings.yupp_tokens == ["token-1"]
    assert settings.debug_mode is False


def test_settings_empty_keys_raises_validation_error() -> None:
    with pytest.raises(ValidationError):
        Settings(
            client_api_keys=[],
            yupp_tokens=["token-1"],
        )


def test_settings_empty_tokens_raises_validation_error() -> None:
    with pytest.raises(ValidationError):
        Settings(
            client_api_keys=["sk-1"],
            yupp_tokens=[],
        )
