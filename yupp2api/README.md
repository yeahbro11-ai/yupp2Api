# Yupp2API Package Structure

This directory contains the refactored, modular implementation of the Yupp2API service.

## Directory Layout

```
yupp2api/
├── __init__.py           # Package marker
├── app.py                # FastAPI application factory
├── auth.py               # Authentication dependencies
├── bootstrap.py          # Startup initialization helpers
├── config.py             # Centralized Pydantic Settings
├── dependencies.py       # Common FastAPI dependencies
├── models.py             # Pydantic request/response models
├── state.py              # Runtime state container
├── tokens.py             # Token rotation and account management
├── utils.py              # Utility functions
├── core/
│   ├── __init__.py
│   └── stream.py         # Stream processing logic
└── routers/
    ├── __init__.py
    ├── chat_router.py    # /v1/chat/completions endpoint
    └── models_router.py  # /v1/models and /models endpoints
```

## Key Modules

### config.py
Centralized configuration using Pydantic Settings. Loads and validates all environment variables:
- CLIENT_API_KEYS
- YUPP_TOKENS
- MODEL_FILE
- HOST, PORT
- DEBUG_MODE
- MAX_ERROR_COUNT, ERROR_COOLDOWN
- HTTP_PROXY, HTTPS_PROXY

### app.py
FastAPI application factory that:
- Creates the app instance with lifespan management
- Attaches middleware (CORS, TrustedHost)
- Includes routers
- Stores settings and runtime state in app.state

### models.py
All Pydantic models for type safety:
- ChatMessage, ChatCompletionRequest, ChatCompletionResponse
- ModelInfo, ModelList
- StreamChoice, StreamResponse
- YuppAccount (TypedDict)

### state.py
RuntimeState dataclass holding mutable app data:
- valid_client_keys (Set)
- accounts (List[YuppAccount])
- models (List[Dict])
- account_rotation_lock (Lock)

### tokens.py
Account management logic:
- initialize_accounts() - Load from config
- get_best_yupp_account() - Smart selection with cooldown
- claim_yupp_reward() - Reward claiming

### core/stream.py
Stream processing:
- yupp_stream_generator() - Converts Yupp stream to OpenAI SSE format
- build_yupp_non_stream_response() - Builds complete response from stream

## Usage

```python
from yupp2api.app import create_app
from yupp2api.config import get_settings

settings = get_settings()
app = create_app(settings)
```

## Testing

Basic tests are in the `tests/` directory at project root:
- test_app.py - Smoke tests for endpoints
- test_config.py - Configuration validation tests
- test_utils.py - Utility function tests

Run tests with:
```bash
pytest
```
