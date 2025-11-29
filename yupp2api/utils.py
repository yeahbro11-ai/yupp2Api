"""Utility functions used across modules."""

from __future__ import annotations

import json
import re
from typing import List

import requests
from requests import Session

from .config import Settings
from .models import ChatMessage


def create_requests_session(settings: Settings) -> Session:
    """Create a configured requests session with retries and proxy support."""
    session = requests.Session()
    proxies = {key: value for key, value in settings.proxies.items() if value}

    if proxies:
        # Requests only cares about HTTP/HTTPS for the session-level proxies; include no_proxy if set.
        session.proxies.update(proxies)
    else:
        session.proxies = {"http": None, "https": None}

    adapter = requests.adapters.HTTPAdapter(max_retries=3)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def format_messages_for_yupp(messages: List[ChatMessage]) -> str:
    """
    Convert multi-turn conversation to Yupp single-turn format.

    Yupp expects a special Human/Assistant format in one string.
    """
    formatted: List[str] = []

    system_messages = [msg for msg in messages if msg.role == "system"]
    if system_messages:
        for sys_msg in system_messages:
            content = (
                sys_msg.content
                if isinstance(sys_msg.content, str)
                else json.dumps(sys_msg.content)
            )
            formatted.append(content)

    user_assistant_msgs = [msg for msg in messages if msg.role != "system"]
    for msg in user_assistant_msgs:
        role = "Human" if msg.role == "user" else "Assistant"
        content = (
            msg.content if isinstance(msg.content, str) else json.dumps(msg.content)
        )
        formatted.append(f"\n\n{role}: {content}")

    if not formatted or not formatted[-1].strip().startswith("Assistant:"):
        formatted.append("\n\nAssistant:")

    result = "".join(formatted)
    if result.startswith("\n\n"):
        result = result[2:]

    return result


def clean_model_name(model_name: str) -> str:
    """Remove newlines, emojis, and special characters from model name."""
    if not model_name:
        return model_name
    cleaned = re.sub(r"[\n\r\t\f\v]", " ", model_name)
    cleaned = re.sub(r"[^\w\s\-_\(\)\.\/\[\]]+", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def log_debug(settings: Settings, message: str) -> None:
    """Log message only if DEBUG_MODE is enabled."""
    if settings.debug_mode:
        print(f"[DEBUG] {message}")
