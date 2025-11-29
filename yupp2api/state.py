"""Runtime state container for the FastAPI application."""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Dict, List, Set

from .models import YuppAccount


@dataclass
class RuntimeState:
    """Holds mutable runtime data that changes while the app is running."""

    valid_client_keys: Set[str] = field(default_factory=set)
    accounts: List[YuppAccount] = field(default_factory=list)
    models: List[Dict[str, Any]] = field(default_factory=list)
    account_rotation_lock: Lock = field(default_factory=Lock)
