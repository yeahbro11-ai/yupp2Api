import json
import os
import re
import time
import uuid
import threading
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, TypedDict, Union, Generator
import requests
from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


def create_requests_session():
    """Create a configured requests session with proxy disabled"""
    session = requests.Session()
    session.proxies = {"http": None, "https": None}
    adapter = requests.adapters.HTTPAdapter(max_retries=3)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


class YuppAccount(TypedDict):
    token: str
    is_valid: bool
    last_used: float
    error_count: int


VALID_CLIENT_KEYS: set = set()
YUPP_ACCOUNTS: List[YuppAccount] = []
YUPP_MODELS: List[Dict[str, Any]] = []
account_rotation_lock = threading.Lock()
DEBUG_MODE = False


class ChatMessage(BaseModel):
    role: str
    content: Union[str, List[Dict[str, Any]]]
    reasoning_content: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    stream: bool = True
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelInfo]


class ChatCompletionChoice(BaseModel):
    message: ChatMessage
    index: int = 0
    finish_reason: str = "stop"


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionChoice]
    usage: Dict[str, int] = Field(
        default_factory=lambda: {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
    )


class StreamChoice(BaseModel):
    delta: Dict[str, Any] = Field(default_factory=dict)
    index: int = 0
    finish_reason: Optional[str] = None


class StreamResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex}")
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[StreamChoice]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown lifecycle management"""
    # Startup
    print("Starting Yupp.ai OpenAI API Adapter server...")
    load_client_api_keys()
    load_yupp_accounts()
    load_yupp_models()
    print("Server initialization completed.")

    yield
    # Shutdown
    print("Server shutdown completed.")


app = FastAPI(title="Yupp.ai OpenAI API Adapter", lifespan=lifespan)

app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
security = HTTPBearer(auto_error=False)


def log_debug(message: str):
    """Debug logging function"""
    if DEBUG_MODE:
        print(f"[DEBUG] {message}")


def load_client_api_keys():
    """Load client API keys from environment variables"""
    global VALID_CLIENT_KEYS

    env_keys = os.getenv("CLIENT_API_KEYS")
    if not env_keys:
        print(
            "Error: CLIENT_API_KEYS environment variable not found. Client authentication will fail."
        )
        VALID_CLIENT_KEYS = set()
        return

    try:
        # Support multiple keys separated by commas
        keys = [key.strip() for key in env_keys.split(",") if key.strip()]
        VALID_CLIENT_KEYS = set(keys)
        print(
            f"Successfully loaded {len(VALID_CLIENT_KEYS)} client API keys from environment variables."
        )
    except Exception as e:
        print(f"Error parsing CLIENT_API_KEYS environment variable: {e}")
        VALID_CLIENT_KEYS = set()


def load_yupp_accounts():
    """Load Yupp accounts from environment variables"""
    global YUPP_ACCOUNTS
    YUPP_ACCOUNTS = []

    env_tokens = os.getenv("YUPP_TOKENS")
    if not env_tokens:
        print("Error: YUPP_TOKENS environment variable not found. API calls will fail.")
        return

    try:
        # Support multiple tokens separated by commas
        tokens = [token.strip() for token in env_tokens.split(",") if token.strip()]
        for token in tokens:
            YUPP_ACCOUNTS.append(
                {
                    "token": token,
                    "is_valid": True,
                    "last_used": 0,
                    "error_count": 0,
                }
            )
        print(
            f"Successfully loaded {len(YUPP_ACCOUNTS)} Yupp accounts from environment variables."
        )
    except Exception as e:
        print(f"Error parsing YUPP_TOKENS environment variable: {e}")


def load_yupp_models():
    """Load Yupp models from model.json, auto-fetch if file doesn't exist"""
    global YUPP_MODELS
    model_file = os.getenv("MODEL_FILE", "./model/model.json")

    # Ensure the model file exists
    if not os.path.exists(model_file):
        print(f"Model file {model_file} not found. Attempting to fetch model data automatically...")
        try:
            # Import and invoke helper from model.py
            from model import fetch_and_save_models

            success = fetch_and_save_models(model_file)
            if success:
                print(f"Automatically fetched and saved model data to {model_file}.")
            else:
                print("Automatic model fetch failed. Proceeding with an empty model list.")
                YUPP_MODELS = []
                return
        except ImportError as e:
            print(f"Unable to import model.py module: {e}")
            YUPP_MODELS = []
            return
        except Exception as e:
            print(f"Error while fetching model data automatically: {e}")
            YUPP_MODELS = []
            return

    # Load the model file content
    try:
        with open(model_file, "r", encoding="utf-8") as f:
            YUPP_MODELS = json.load(f)
            if not isinstance(YUPP_MODELS, list):
                YUPP_MODELS = []
                print(f"Warning: {model_file} should contain a list of model objects.")
                return
            print(f"Successfully loaded {len(YUPP_MODELS)} models from {model_file}.")
    except FileNotFoundError:
        print(f"Error: {model_file} not found. Model list will be empty.")
        YUPP_MODELS = []
    except Exception as e:
        print(f"Error loading {model_file}: {e}")
        YUPP_MODELS = []


def get_best_yupp_account() -> Optional[YuppAccount]:
    """Get the best available Yupp account using a smart selection algorithm."""
    max_error_count = int(os.getenv("MAX_ERROR_COUNT", "3"))
    error_cooldown = int(os.getenv("ERROR_COOLDOWN", "300"))

    with account_rotation_lock:
        now = time.time()
        valid_accounts = [
            acc
            for acc in YUPP_ACCOUNTS
            if acc["is_valid"]
            and (
                acc["error_count"] < max_error_count
                or now - acc["last_used"] > error_cooldown
            )
        ]

        if not valid_accounts:
            return None

        # Reset error count for accounts that have been in cooldown
        for acc in valid_accounts:
            if (
                acc["error_count"] >= max_error_count
                and now - acc["last_used"] > error_cooldown
            ):
                acc["error_count"] = 0

        # Sort by last used (oldest first) and error count (lowest first)
        valid_accounts.sort(key=lambda x: (x["last_used"], x["error_count"]))
        account = valid_accounts[0]
        account["last_used"] = now
        return account


def format_messages_for_yupp(messages: List[ChatMessage]) -> str:
    """Format multi-turn conversation into Yupp's single-turn format"""
    formatted = []

    # Process system messages
    system_messages = [msg for msg in messages if msg.role == "system"]
    if system_messages:
        for sys_msg in system_messages:
            content = (
                sys_msg.content
                if isinstance(sys_msg.content, str)
                else json.dumps(sys_msg.content)
            )
            formatted.append(content)

    # Process user and assistant messages
    user_assistant_msgs = [msg for msg in messages if msg.role != "system"]
    for msg in user_assistant_msgs:
        role = "Human" if msg.role == "user" else "Assistant"
        content = (
            msg.content if isinstance(msg.content, str) else json.dumps(msg.content)
        )
        formatted.append(f"\n\n{role}: {content}")

    # Ensure it ends with "Assistant:"
    if not formatted or not formatted[-1].strip().startswith("Assistant:"):
        formatted.append("\n\nAssistant:")

    result = "".join(formatted)
    # Strip leading double newline if present
    if result.startswith("\n\n"):
        result = result[2:]

    return result


async def authenticate_client(
    auth: Optional[HTTPAuthorizationCredentials] = Depends(security),
):
    """Authenticate client based on API key in Authorization header"""
    if not VALID_CLIENT_KEYS:
        raise HTTPException(
            status_code=503,
            detail="Service unavailable: Client API keys not configured on server.",
        )

    if not auth or not auth.credentials:
        raise HTTPException(
            status_code=401,
            detail="API key required in Authorization header.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if auth.credentials not in VALID_CLIENT_KEYS:
        raise HTTPException(status_code=403, detail="Invalid client API key.")


def get_models_list_response() -> ModelList:
    """Helper to construct ModelList response from cached models."""
    model_infos = [
        ModelInfo(
            id=model.get("label", "unknown"),
            created=int(time.time()),
            owned_by=model.get("publisher", "unknown"),
        )
        for model in YUPP_MODELS
    ]
    return ModelList(data=model_infos)


@app.get("/v1/models", response_model=ModelList)
async def list_v1_models(_: None = Depends(authenticate_client)):
    """List available models - authenticated"""
    return get_models_list_response()


@app.get("/models", response_model=ModelList)
async def list_models_no_auth():
    """List available models without authentication - for client compatibility"""
    return get_models_list_response()


def claim_yupp_reward(account: YuppAccount, reward_id: str):
    """Claim a pending Yupp reward synchronously"""
    try:
        log_debug(f"Claiming reward {reward_id}...")
        url = "https://yupp.ai/api/trpc/reward.claim?batch=1"
        payload = {"0": {"json": {"rewardId": reward_id}}}
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36 Edg/137.0.0.0",
            "Content-Type": "application/json",
            "sec-fetch-site": "same-origin",
            "Cookie": f"__Secure-yupp.session-token={account['token']}",
        }
        session = create_requests_session()
        response = session.post(url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        balance = data[0]["result"]["data"]["json"]["currentCreditBalance"]
        print(f"Reward claimed successfully. New balance: {balance}")
        return balance
    except Exception as e:
        print(f"Failed to claim reward {reward_id}. Error: {e}")
        return None


def yupp_stream_generator(
    response_lines, model_id: str, account: YuppAccount
) -> Generator[str, None, None]:
    """Process Yupp streaming response and convert it to OpenAI format"""
    stream_id = f"chatcmpl-{uuid.uuid4().hex}"
    created_time = int(time.time())

    # Clean model name: remove newlines and emojis
    def clean_model_name(model_name: str) -> str:
        """Clean model name by removing newlines and emojis"""
        if not model_name:
            return model_name
        # Remove newlines and other invisible characters
        cleaned = re.sub(r"[\n\r\t\f\v]", " ", model_name)
        # Remove emojis and other special characters
        cleaned = re.sub(r"[^\w\s\-_\(\)\.\/\[\]]+", "", cleaned)
        # Remove excess whitespace
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    clean_model_id = clean_model_name(model_id)

    # Send initial role
    yield f"data: {StreamResponse(id=stream_id, created=created_time, model=clean_model_id, choices=[StreamChoice(delta={'role': 'assistant'})]).model_dump_json()}\n\n"

    line_pattern = re.compile(b"^([0-9a-fA-F]+):(.*)")
    chunks = {}
    target_stream_id = None
    reward_info = None
    is_thinking = False
    thinking_content = ""
    normal_content = ""
    select_stream = [None, None]  # Initialize select_stream
    processed_content = set()  # Track processed content to avoid duplicates

    def extract_ref_id(ref):
        """Extract ID from reference string, e.g., extract '123' from '$@123'"""
        return (
            ref[2:] if ref and isinstance(ref, str) and ref.startswith("$@") else None
        )

    def is_valid_content(content: str) -> bool:
        """Check if content is valid, avoiding over-filtering"""
        if not content or content in [None, "", "$undefined"]:
            return False

        # Filter out obvious system messages
        if content.startswith("\\n\\<streaming stopped") or content.startswith(
            "\n\\<streaming stopped"
        ):
            return False

        # Filter out pure UUIDs
        if re.match(
            r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
            content.strip(),
        ):
            return False

        # Filter out empty content
        if len(content.strip()) == 0:
            return False

        # Filter out obvious system markers
        if content.strip() in ["$undefined", "undefined", "null", "NULL"]:
            return False

        return True

    def process_content_chunk(content: str, chunk_id: str):
        """Process a single content chunk"""
        nonlocal is_thinking, thinking_content, normal_content

        if not is_valid_content(content):
            return

        # Avoid processing the same content twice
        content_hash = hash(content)
        if content_hash in processed_content:
            return
        processed_content.add(content_hash)

        log_debug(f"Processing chunk {chunk_id} with content: '{content[:50]}...'")

        # Handle thinking process
        if "<think>" in content or "</think>" in content:
            yield from process_thinking_content(content)
        elif is_thinking:
            thinking_content += content
            yield f"data: {StreamResponse(id=stream_id, created=created_time, model=clean_model_id, choices=[StreamChoice(delta={'reasoning_content': content})]).model_dump_json()}\n\n"
        else:
            normal_content += content
            yield f"data: {StreamResponse(id=stream_id, created=created_time, model=clean_model_id, choices=[StreamChoice(delta={'content': content})]).model_dump_json()}\n\n"

    def process_thinking_content(content: str):
        """Process content containing thinking tags"""
        nonlocal is_thinking, thinking_content, normal_content

        if "<think>" in content:
            parts = content.split("<think>", 1)
            if parts[0]:  # Content before thinking tag
                normal_content += parts[0]
                yield f"data: {StreamResponse(id=stream_id, created=created_time, model=clean_model_id, choices=[StreamChoice(delta={'content': parts[0]})]).model_dump_json()}\n\n"

            is_thinking = True
            thinking_part = parts[1]

            if "</think>" in thinking_part:
                think_parts = thinking_part.split("</think>", 1)
                thinking_content += think_parts[0]
                yield f"data: {StreamResponse(id=stream_id, created=created_time, model=clean_model_id, choices=[StreamChoice(delta={'reasoning_content': think_parts[0]})]).model_dump_json()}\n\n"

                is_thinking = False
                if think_parts[1]:  # Content after thinking tag
                    normal_content += think_parts[1]
                    yield f"data: {StreamResponse(id=stream_id, created=created_time, model=clean_model_id, choices=[StreamChoice(delta={'content': think_parts[1]})]).model_dump_json()}\n\n"
            else:
                thinking_content += thinking_part
                yield f"data: {StreamResponse(id=stream_id, created=created_time, model=clean_model_id, choices=[StreamChoice(delta={'reasoning_content': thinking_part})]).model_dump_json()}\n\n"

        elif "</think>" in content and is_thinking:
            parts = content.split("</think>", 1)
            thinking_content += parts[0]
            yield f"data: {StreamResponse(id=stream_id, created=created_time, model=clean_model_id, choices=[StreamChoice(delta={'reasoning_content': parts[0]})]).model_dump_json()}\n\n"

            is_thinking = False
            if parts[1]:  # Content after thinking tag
                normal_content += parts[1]
                yield f"data: {StreamResponse(id=stream_id, created=created_time, model=clean_model_id, choices=[StreamChoice(delta={'content': parts[1]})]).model_dump_json()}\n\n"

    try:
        log_debug("Starting to process response lines...")
        line_count = 0

        for line in response_lines:
            line_count += 1
            if not line:
                continue

            match = line_pattern.match(line)
            if not match:
                log_debug(
                    f"Line {line_count}: No pattern match for line: {line[:50]}..."
                )
                continue

            chunk_id, chunk_data = match.groups()
            chunk_id = chunk_id.decode()

            try:
                data = json.loads(chunk_data) if chunk_data != b"{}" else {}
                chunks[chunk_id] = data
                log_debug(f"Parsed chunk {chunk_id}: {str(data)[:100]}...")
            except json.JSONDecodeError:
                log_debug(f"Failed to parse JSON for chunk {chunk_id}: {chunk_data}")
                continue

            # Handle reward information
            if chunk_id == "a":
                reward_info = data
                log_debug(f"Found reward info: {reward_info}")

            # Handle initial stream setup
            elif chunk_id == "1":
                if isinstance(data, dict):
                    left_stream = data.get("leftStream", {})
                    right_stream = data.get("rightStream", {})
                    select_stream = [left_stream, right_stream]
                    log_debug(
                        f"Found stream setup: left={left_stream}, right={right_stream}"
                    )

            elif chunk_id == "e":
                if isinstance(data, dict):
                    for i, selection in enumerate(data.get("modelSelections", [])):
                        if selection.get("selectionSource") == "USER_SELECTED":
                            if i < len(select_stream) and isinstance(
                                select_stream[i], dict
                            ):
                                target_stream_id = extract_ref_id(
                                    select_stream[i].get("next")
                                )
                                log_debug(f"Found target stream ID: {target_stream_id}")
                            break

            # Process target stream content
            elif target_stream_id and chunk_id == target_stream_id:
                if isinstance(data, dict):
                    content = data.get("curr", "")
                    if content:
                        log_debug(
                            f"Processing target stream content: '{content[:50]}...'"
                        )
                        yield from process_content_chunk(content, chunk_id)

                        # Update target stream ID
                        target_stream_id = extract_ref_id(data.get("next"))
                        if target_stream_id:
                            log_debug(
                                f"Updated target stream ID to: {target_stream_id}"
                            )

            # Fallback: process any chunk containing "curr"
            elif isinstance(data, dict) and "curr" in data:
                content = data.get("curr", "")
                if content:
                    log_debug(
                        f"Processing fallback chunk {chunk_id} with content: '{content[:50]}...'"
                    )
                    yield from process_content_chunk(content, chunk_id)

        log_debug(f"Finished processing {line_count} lines")

    except Exception as e:
        log_debug(f"Stream processing error: {e}")
        print(f"Stream processing error: {e}")
        yield f"data: {json.dumps({'error': str(e)})}\n\n"

    finally:
        # Send completion signal
        yield f"data: {StreamResponse(id=stream_id, created=created_time, model=clean_model_id, choices=[StreamChoice(delta={}, finish_reason='stop')]).model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"

        # Claim any pending reward
        if reward_info and "unclaimedRewardInfo" in reward_info:
            reward_id = reward_info["unclaimedRewardInfo"].get("rewardId")
            if reward_id:
                try:
                    claim_yupp_reward(account, reward_id)
                except Exception as e:
                    print(f"Failed to claim reward in background: {e}")

        log_debug(
            f"Stream processing completed. Total content: {len(normal_content)} chars, thinking: {len(thinking_content)} chars"
        )


def build_yupp_non_stream_response(
    response_lines, model_id: str, account: YuppAccount
) -> ChatCompletionResponse:
    """Build a non-streaming response from Yupp stream events"""
    full_content = ""
    full_reasoning_content = ""
    reward_id = None

    # Store model name extracted from the streaming response
    response_model_name = model_id

    for event in yupp_stream_generator(response_lines, model_id, account):
        if event.startswith("data:"):
            data_str = event[5:].strip()
            if data_str == "[DONE]":
                break

            try:
                data = json.loads(data_str)
                if "error" in data:
                    raise HTTPException(
                        status_code=500, detail=data["error"]["message"]
                    )

                # Extract model name from the first valid response (already cleaned)
                if "model" in data and not response_model_name:
                    response_model_name = data["model"]

                delta = data.get("choices", [{}])[0].get("delta", {})
                if "content" in delta:
                    full_content += delta["content"]
                if "reasoning_content" in delta:
                    full_reasoning_content += delta["reasoning_content"]
            except json.JSONDecodeError:
                continue

    # Build complete response
    return ChatCompletionResponse(
        model=response_model_name,
        choices=[
            ChatCompletionChoice(
                message=ChatMessage(
                    role="assistant",
                    content=full_content,
                    reasoning_content=(
                        full_reasoning_content if full_reasoning_content else None
                    ),
                )
            )
        ],
    )


@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest, _: None = Depends(authenticate_client)
):
    """Create chat completions by proxying requests to Yupp.ai"""
    # Find the requested model
    model_info = next((m for m in YUPP_MODELS if m.get("label") == request.model), None)
    if not model_info:
        raise HTTPException(
            status_code=404, detail=f"Model '{request.model}' not found."
        )

    model_name = model_info.get("name")
    if not model_name:
        raise HTTPException(
            status_code=404, detail=f"Model '{request.model}' has no 'name' field."
        )

    if not request.messages:
        raise HTTPException(
            status_code=400, detail="No messages provided in the request."
        )

    log_debug(
        f"Processing request for model: {request.model} (Yupp name: {model_name})"
    )

    # Format messages for Yupp
    question = format_messages_for_yupp(request.messages)
    log_debug(f"Formatted question: {question[:100]}...")

    # Try all available accounts
    for attempt in range(len(YUPP_ACCOUNTS)):
        account = get_best_yupp_account()
        if not account:
            raise HTTPException(
                status_code=503, detail="No valid Yupp.ai accounts available."
            )

        try:
            # Build request payload
            url_uuid = str(uuid.uuid4())
            url = f"https://yupp.ai/chat/{url_uuid}?stream=true"

            payload = [
                url_uuid,
                str(uuid.uuid4()),
                question,
                "$undefined",
                "$undefined",
                [],
                "$undefined",
                [{"modelName": model_name, "promptModifierId": "$undefined"}],
                "text",
                False,
                "$undefined",
            ]

            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36 Edg/137.0.0.0",
                "Accept": "text/x-component",
                "Accept-Encoding": "gzip, deflate, br, zstd",
                "Content-Type": "application/json",
                "next-action": "7fbcb7bc0fcb4b0833ac4d1a1981315749f0dc7c09",
                "sec-fetch-site": "same-origin",
                "Cookie": f"__Secure-yupp.session-token={account['token']}",
            }

            log_debug(
                f"Sending request to Yupp.ai with account token ending in ...{account['token'][-4:]}"
            )

            # Send request
            session = create_requests_session()
            response = session.post(
                url,
                data=json.dumps(payload),
                headers=headers,
                stream=True,
            )
            response.raise_for_status()

            # Handle response
            if request.stream:
                log_debug("Returning processed response stream")
                return StreamingResponse(
                    yupp_stream_generator(
                        response.iter_lines(), request.model, account
                    ),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "X-Accel-Buffering": "no",
                    },
                )
            else:
                log_debug("Building non-stream response")

                return build_yupp_non_stream_response(
                    response.iter_lines(), request.model, account
                )

        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code
            error_detail = e.response.text
            print(f"Yupp.ai API error ({status_code}): {error_detail}")

            with account_rotation_lock:
                if status_code in [401, 403]:
                    account["is_valid"] = False
                    print(
                        f"Account ...{account['token'][-4:]} marked as invalid due to auth error."
                    )
                elif status_code in [429, 500, 502, 503, 504]:
                    account["error_count"] += 1
                    print(
                        f"Account ...{account['token'][-4:]} error count: {account['error_count']}"
                    )
                else:
                    # Client error; do not try other accounts
                    raise HTTPException(status_code=status_code, detail=error_detail)

        except Exception as e:
            print(f"Request error: {e}")
            with account_rotation_lock:
                account["error_count"] += 1

    # All attempts failed
    raise HTTPException(
        status_code=503, detail="All attempts to contact Yupp.ai API failed."
    )


def main():
    """Entry point: start the Yupp.ai OpenAI API Adapter service"""
    import uvicorn
    from dotenv import load_dotenv

    # Load environment variables
    load_dotenv()

    # Configure global settings
    global DEBUG_MODE
    DEBUG_MODE = os.environ.get("DEBUG_MODE", "false").lower() == "true"

    if DEBUG_MODE:
        print("Debug mode enabled")

    # Validate required environment variables
    if not os.getenv("CLIENT_API_KEYS"):
        print("Warning: CLIENT_API_KEYS environment variable not set.")
    if not os.getenv("YUPP_TOKENS"):
        print("Warning: YUPP_TOKENS environment variable not set.")

    # Load configuration
    load_client_api_keys()
    load_yupp_accounts()
    load_yupp_models()

    # Display startup info
    print("\n--- Yupp.ai OpenAI API Adapter ---")
    print(f"Debug Mode: {DEBUG_MODE}")
    print("Endpoints:")
    print("  GET  /v1/models (Client API Key Auth)")
    print("  GET  /models (No Auth)")
    print("  POST /v1/chat/completions (Client API Key Auth)")

    print(f"\nClient API Keys: {len(VALID_CLIENT_KEYS)}")
    if YUPP_ACCOUNTS:
        print(f"Yupp.ai Accounts: {len(YUPP_ACCOUNTS)}")
    else:
        print("Yupp.ai Accounts: None loaded. Check YUPP_TOKENS environment variable.")
    if YUPP_MODELS:
        models = sorted([m.get("label", m.get("id", "unknown")) for m in YUPP_MODELS])
        print(f"Yupp.ai Models: {len(YUPP_MODELS)}")
        print(
            f"Available models: {', '.join(models[:5])}{'...' if len(models) > 5 else ''}"
        )
    else:
        model_file = os.getenv("MODEL_FILE", "model.json")
        print(f"Yupp.ai Models: None loaded. Check {model_file}.")
    print("------------------------------------")

    # Start server
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8001"))

    print(f"Starting server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
