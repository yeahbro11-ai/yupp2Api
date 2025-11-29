"""Helpers for loading runtime state from configuration."""

from __future__ import annotations

import json

from .config import Settings
from .state import RuntimeState
from .tokens import initialize_accounts


def load_client_api_keys(state: RuntimeState, settings: Settings) -> None:
    """Populate the runtime state's valid client keys set."""
    state.valid_client_keys = set(settings.client_api_keys)
    print(f"Successfully loaded {len(state.valid_client_keys)} client API keys from configuration.")


def load_yupp_models(state: RuntimeState, settings: Settings) -> None:
    """Load Yupp models from configured file or auto-fetch if missing."""
    model_file = settings.model_file

    if not model_file.exists():
        print(f"Model file {model_file} not found, attempting to auto-fetch...")
        try:
            from model import fetch_and_save_models

            success = fetch_and_save_models(str(model_file))
            if success:
                print(f"Fetched and saved model data to {model_file}")
            else:
                print("Auto-fetch failed, using empty model list")
                state.models = []
                return
        except Exception as exc:  # pylint: disable=broad-except
            print(f"Failed to auto-fetch model data: {exc}")
            state.models = []
            return

    try:
        with model_file.open("r", encoding="utf-8") as file:
            loaded = json.load(file)
            if not isinstance(loaded, list):
                print(f"Warning: {model_file} should contain a list of model objects.")
                state.models = []
                return
            state.models = loaded
            print(f"Successfully loaded {len(state.models)} models from {model_file}.")
    except FileNotFoundError:
        print(f"Error: {model_file} not found. Model list will be empty.")
        state.models = []
    except Exception as exc:  # pylint: disable=broad-except
        print(f"Error loading {model_file}: {exc}")
        state.models = []


def bootstrap_state(state: RuntimeState, settings: Settings) -> None:
    """Load all runtime resources from configuration."""
    load_client_api_keys(state, settings)
    initialize_accounts(state, settings)
    load_yupp_models(state, settings)
