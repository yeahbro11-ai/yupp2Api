"""Application configuration and settings helpers."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class Settings(BaseModel):
    """Centralized application configuration."""

    client_api_keys: List[str] = Field(..., description="Comma separated CLIENT_API_KEYS value")
    yupp_tokens: List[str] = Field(..., description="Comma separated YUPP_TOKENS value")
    model_file: Path = Field(default=Path("./model/model.json"), description="Path to cached model file")
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8001)
    debug_mode: bool = Field(default=False)
    max_error_count: int = Field(default=3)
    error_cooldown: int = Field(default=300)
    http_proxy: Optional[str] = Field(default=None)
    https_proxy: Optional[str] = Field(default=None)
    no_proxy: Optional[str] = Field(default=None)

    @field_validator("client_api_keys", "yupp_tokens")
    @classmethod
    def _ensure_values(cls, value: List[str], info):
        if not value:
            raise ValueError(f"{info.field_name} cannot be empty")
        return value

    @property
    def proxies(self) -> dict[str, Optional[str]]:
        return {
            "http": self.http_proxy,
            "https": self.https_proxy,
            "no_proxy": self.no_proxy,
        }


@lru_cache()
def get_settings() -> Settings:
    """Load settings from environment variables."""

    import os
    from dotenv import load_dotenv

    load_dotenv()

    def _split_env_list(raw: Optional[str]) -> List[str]:
        if not raw:
            return []
        return [item.strip() for item in raw.split(",") if item.strip()]

    model_file = os.getenv("MODEL_FILE", "./model/model.json")

    return Settings(
        client_api_keys=_split_env_list(os.getenv("CLIENT_API_KEYS")),
        yupp_tokens=_split_env_list(os.getenv("YUPP_TOKENS")),
        model_file=Path(model_file),
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8001")),
        debug_mode=os.getenv("DEBUG_MODE", "false").lower() == "true",
        max_error_count=int(os.getenv("MAX_ERROR_COUNT", "3")),
        error_cooldown=int(os.getenv("ERROR_COOLDOWN", "300")),
        http_proxy=os.getenv("HTTP_PROXY"),
        https_proxy=os.getenv("HTTPS_PROXY"),
        no_proxy=os.getenv("NO_PROXY"),
    )
