"""
Legacy entry point for backward compatibility.

This file maintains the original CLI for running `python yyapi.py`.
The modular implementation is now in the yupp2api package.
"""

import uvicorn

from yupp2api.app import create_app
from yupp2api.config import get_settings

app = create_app()


def main() -> None:
    """Launch the uvicorn server with configured settings."""
    settings = get_settings()

    if settings.debug_mode:
        print("Debug mode enabled")

    app = create_app(settings)

    print("\n--- Yupp.ai OpenAI API Adapter ---")
    print(f"Debug Mode: {settings.debug_mode}")
    print("Endpoints:")
    print("  GET  /v1/models (Client API Key Auth)")
    print("  GET  /models (No Auth)")
    print("  POST /v1/chat/completions (Client API Key Auth)")
    print(f"\nClient API Keys: {len(settings.client_api_keys)}")
    print(f"Yupp.ai Accounts: {len(settings.yupp_tokens)}")
    print("------------------------------------")
    print(f"Starting server on {settings.host}:{settings.port}")

    uvicorn.run(app, host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()
