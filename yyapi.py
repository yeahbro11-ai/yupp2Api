import json
import os
import re
import time
import uuid
import threading
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, TypedDict, Union, Generator
from enum import Enum
import requests
from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


def create_requests_session():
    """创建配置好的 requests session"""
    session = requests.Session()
    session.proxies = {"http": None, "https": None}
    adapter = requests.adapters.HTTPAdapter(max_retries=3)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


class ErrorType(Enum):
    """Classification of error types for better handling"""
    AUTH_ERROR = "auth_error"           # 401, 403 - invalid token
    RATE_LIMIT = "rate_limit"           # 429 - too many requests
    SERVER_ERROR = "server_error"       # 500, 502, 503, 504 - upstream issues
    NETWORK_ERROR = "network_error"     # Connection, timeout errors
    CLIENT_ERROR = "client_error"       # 4xx client-side errors
    STREAM_ERROR = "stream_error"       # Errors during streaming
    PARSE_ERROR = "parse_error"         # JSON or response parsing errors
    UNKNOWN_ERROR = "unknown_error"     # Catch-all


def mask_token(token: str, visible_chars: int = 4) -> str:
    """Safely mask a token for logging, showing only last N characters"""
    if not token or len(token) <= visible_chars:
        return "***"
    return f"...{token[-visible_chars:]}"


def classify_error(error: Exception) -> ErrorType:
    """Classify an exception into an ErrorType for appropriate handling"""
    if isinstance(error, requests.exceptions.HTTPError):
        status_code = error.response.status_code
        if status_code in [401, 403]:
            return ErrorType.AUTH_ERROR
        elif status_code == 429:
            return ErrorType.RATE_LIMIT
        elif status_code in [500, 502, 503, 504]:
            return ErrorType.SERVER_ERROR
        elif 400 <= status_code < 500:
            return ErrorType.CLIENT_ERROR
    elif isinstance(error, (requests.exceptions.ConnectionError, 
                           requests.exceptions.Timeout,
                           requests.exceptions.ConnectTimeout,
                           requests.exceptions.ReadTimeout)):
        return ErrorType.NETWORK_ERROR
    elif isinstance(error, json.JSONDecodeError):
        return ErrorType.PARSE_ERROR
    
    return ErrorType.UNKNOWN_ERROR


def log_error(error_type: ErrorType, message: str, account_token: Optional[str] = None, 
              endpoint: Optional[str] = None, exception: Optional[Exception] = None):
    """Centralized error logging with context"""
    masked_token = mask_token(account_token) if account_token else "N/A"
    endpoint_str = endpoint if endpoint else "N/A"
    exception_str = f" | Exception: {str(exception)}" if exception else ""
    
    log_line = f"[ERROR] Type: {error_type.value} | Token: {masked_token} | Endpoint: {endpoint_str} | {message}{exception_str}"
    print(log_line)


def should_retry_error(error_type: ErrorType) -> bool:
    """Determine if an error type should trigger a retry with another account"""
    return error_type in [
        ErrorType.AUTH_ERROR,
        ErrorType.RATE_LIMIT,
        ErrorType.SERVER_ERROR,
        ErrorType.NETWORK_ERROR,
        ErrorType.UNKNOWN_ERROR
    ]


def should_mark_invalid(error_type: ErrorType) -> bool:
    """Determine if an error should mark the account as permanently invalid"""
    return error_type == ErrorType.AUTH_ERROR


def should_increment_error_count(error_type: ErrorType) -> bool:
    """Determine if an error should increment the account's error count"""
    return error_type in [
        ErrorType.RATE_LIMIT,
        ErrorType.SERVER_ERROR,
        ErrorType.NETWORK_ERROR,
        ErrorType.UNKNOWN_ERROR
    ]


def log_account_event(account: Optional[Dict[str, Any]], message: str):
    """Log account-specific events with masked tokens"""
    token = account.get("token") if account else None
    masked_token = mask_token(token)
    print(f"[ACCOUNT] Token {masked_token}: {message}")


def set_account_cooldown(account: YuppAccount, cooldown_seconds: Optional[int] = None):
    """Apply cooldown to an account, preventing selection until expiry"""
    if cooldown_seconds is None:
        cooldown_seconds = int(os.getenv("ERROR_COOLDOWN", "300"))
    account["cooldown_until"] = time.time() + max(cooldown_seconds, 1)


def mark_account_invalid(account: YuppAccount, reason: str):
    """Permanently mark an account as invalid"""
    account["is_valid"] = False
    account["cooldown_until"] = float("inf")
    log_account_event(account, f"Marked invalid: {reason}")


def increment_account_error(account: YuppAccount):
    """Increment error count for account"""
    account["error_count"] += 1


def reset_account_after_cooldown(account: YuppAccount):
    """Reset account error state once cooldown finishes"""
    account["error_count"] = 0
    account["cooldown_until"] = 0
    log_account_event(account, "Cooldown elapsed, account re-enabled")


def build_error_detail(message: str, error_type: ErrorType, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    detail = {"message": message, "type": error_type.value}
    if extra:
        detail.update(extra)
    return detail


def raise_tokens_unavailable(reason: str, failures: Optional[List[str]] = None):
    extra = {"failures": failures} if failures else None
    raise HTTPException(
        status_code=503,
        detail=build_error_detail(reason, ErrorType.SERVER_ERROR, extra),
    )


class YuppAccount(TypedDict):
    token: str
    is_valid: bool
    last_used: float
    error_count: int
    cooldown_until: float


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
    """应用启动和关闭时的生命周期管理"""
    # 启动时执行
    print("Starting Yupp.ai OpenAI API Adapter server...")
    load_client_api_keys()
    load_yupp_accounts()
    load_yupp_models()
    print("Server initialization completed.")

    yield
    # 关闭时执行
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
    """Debug日志函数"""
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
        # 支持逗号分隔的多个密钥
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
        # 支持逗号分隔的多个token
        tokens = [token.strip() for token in env_tokens.split(",") if token.strip()]
        for token in tokens:
            YUPP_ACCOUNTS.append(
                {
                    "token": token,
                    "is_valid": True,
                    "last_used": 0,
                    "error_count": 0,
                    "cooldown_until": 0,
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

    # 检查模型文件是否存在
    if not os.path.exists(model_file):
        print(f"模型文件 {model_file} 不存在，尝试自动获取模型数据...")
        try:
            # 导入并调用 model.py 中的函数
            from model import fetch_and_save_models

            success = fetch_and_save_models(model_file)
            if success:
                print(f"成功自动获取并保存模型数据到 {model_file}")
            else:
                print(f"自动获取模型数据失败，将使用空的模型列表")
                YUPP_MODELS = []
                return
        except ImportError as e:
            print(f"无法导入 model.py 模块: {e}")
            YUPP_MODELS = []
            return
        except Exception as e:
            print(f"自动获取模型数据时发生错误: {e}")
            YUPP_MODELS = []
            return

    # 加载模型文件
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
    """Get the best available Yupp account using enhanced rotation logic."""
    max_error_count = int(os.getenv("MAX_ERROR_COUNT", "3"))
    cooldown_seconds = int(os.getenv("ERROR_COOLDOWN", "300"))

    with account_rotation_lock:
        now = time.time()
        candidates: List[YuppAccount] = []

        for account in YUPP_ACCOUNTS:
            if not account["is_valid"]:
                continue

            cooldown_until = account.get("cooldown_until", 0)
            if cooldown_until and cooldown_until > now:
                continue

            if account["error_count"] >= max_error_count:
                if not cooldown_until or cooldown_until <= now:
                    set_account_cooldown(account, cooldown_seconds)
                    log_account_event(
                        account,
                        f"Reached max error count ({account['error_count']}), entering cooldown",
                    )
                continue

            if cooldown_until and cooldown_until <= now and account["error_count"] > 0:
                reset_account_after_cooldown(account)

            candidates.append(account)

        if not candidates:
            return None

        # Sort by last used (oldest first) and error count (lowest first)
        candidates.sort(key=lambda x: (x["last_used"], x["error_count"]))
        account = candidates[0]
        account["last_used"] = now
        return account


def format_messages_for_yupp(messages: List[ChatMessage]) -> str:
    """将多轮对话格式化为Yupp单轮对话格式"""
    formatted = []

    # 处理系统消息
    system_messages = [msg for msg in messages if msg.role == "system"]
    if system_messages:
        for sys_msg in system_messages:
            content = (
                sys_msg.content
                if isinstance(sys_msg.content, str)
                else json.dumps(sys_msg.content)
            )
            formatted.append(content)

    # 处理用户和助手消息
    user_assistant_msgs = [msg for msg in messages if msg.role != "system"]
    for msg in user_assistant_msgs:
        role = "Human" if msg.role == "user" else "Assistant"
        content = (
            msg.content if isinstance(msg.content, str) else json.dumps(msg.content)
        )
        formatted.append(f"\n\n{role}: {content}")

    # 确保以Assistant:结尾
    if not formatted or not formatted[-1].strip().startswith("Assistant:"):
        formatted.append("\n\nAssistant:")

    result = "".join(formatted)
    # 如果以\n\n开头，则删除
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
    """同步领取Yupp奖励 with improved error handling"""
    if not reward_id:
        log_debug("No reward ID provided, skipping claim")
        return None

    try:
        log_debug(f"Claiming reward {reward_id[:16]}... for token {mask_token(account['token'])}")
        url = "https://yupp.ai/api/trpc/reward.claim?batch=1"
        payload = {"0": {"json": {"rewardId": reward_id}}}
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36 Edg/137.0.0.0",
            "Content-Type": "application/json",
            "sec-fetch-site": "same-origin",
            "Cookie": f"__Secure-yupp.session-token={account['token']}",
        }
        session = create_requests_session()
        response = session.post(url, json=payload, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        balance = data[0]["result"]["data"]["json"]["currentCreditBalance"]
        print(f"[REWARD] Claimed successfully for token {mask_token(account['token'])}. New balance: {balance}")
        return balance
    except requests.exceptions.Timeout as e:
        log_error(ErrorType.NETWORK_ERROR, f"Reward claim timeout for reward {reward_id[:16]}...", 
                  account['token'], "reward.claim", e)
        return None
    except requests.exceptions.HTTPError as e:
        error_type = classify_error(e)
        log_error(error_type, f"Reward claim HTTP error for reward {reward_id[:16]}...", 
                  account['token'], "reward.claim", e)
        return None
    except Exception as e:
        error_type = classify_error(e)
        log_error(error_type, f"Reward claim failed for reward {reward_id[:16]}...", 
                  account['token'], "reward.claim", e)
        return None


def yupp_stream_generator(
    response_lines, model_id: str, account: YuppAccount
) -> Generator[str, None, None]:
    """处理Yupp的流式响应并转换为OpenAI格式"""
    stream_id = f"chatcmpl-{uuid.uuid4().hex}"
    created_time = int(time.time())

    # 清理模型名称，移除换行符和表情符号
    def clean_model_name(model_name: str) -> str:
        """清理模型名称，移除换行符和表情符号"""
        if not model_name:
            return model_name
        # 移除换行符和其他不可见字符
        cleaned = re.sub(r"[\n\r\t\f\v]", " ", model_name)
        # 移除表情符号和其他特殊字符
        cleaned = re.sub(r"[^\w\s\-_\(\)\.\/\[\]]+", "", cleaned)
        # 移除多余的空格
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    clean_model_id = clean_model_name(model_id)

    # 发送初始角色
    yield f"data: {StreamResponse(id=stream_id, created=created_time, model=clean_model_id, choices=[StreamChoice(delta={'role': 'assistant'})]).model_dump_json()}\n\n"

    line_pattern = re.compile(b"^([0-9a-fA-F]+):(.*)")
    chunks = {}
    target_stream_id = None
    reward_info = None
    is_thinking = False
    thinking_content = ""
    normal_content = ""
    select_stream = [None, None]  # 初始化 select_stream
    processed_content = set()  # 用于追踪已处理的内容，避免重复

    def extract_ref_id(ref):
        """从引用字符串中提取ID，例如从'$@123'提取'123'"""
        return (
            ref[2:] if ref and isinstance(ref, str) and ref.startswith("$@") else None
        )

    def is_valid_content(content: str) -> bool:
        """检查内容是否有效，避免过度过滤"""
        if not content or content in [None, "", "$undefined"]:
            return False

        # 移除明显的系统消息
        if content.startswith("\\n\\<streaming stopped") or content.startswith(
            "\n\\<streaming stopped"
        ):
            return False

        # 移除纯UUID（更宽松的检查）
        if re.match(
            r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
            content.strip(),
        ):
            return False

        # 移除过短的内容（但允许单字符）
        if len(content.strip()) == 0:
            return False

        # 移除明显的系统标记
        if content.strip() in ["$undefined", "undefined", "null", "NULL"]:
            return False

        return True

    def process_content_chunk(content: str, chunk_id: str):
        """处理单个内容块"""
        nonlocal is_thinking, thinking_content, normal_content

        if not is_valid_content(content):
            return

        # 避免重复处理相同的内容
        content_hash = hash(content)
        if content_hash in processed_content:
            return
        processed_content.add(content_hash)

        log_debug(f"Processing chunk {chunk_id} with content: '{content[:50]}...'")

        # 处理思考过程
        if "<think>" in content or "</think>" in content:
            yield from process_thinking_content(content)
        elif is_thinking:
            thinking_content += content
            yield f"data: {StreamResponse(id=stream_id, created=created_time, model=clean_model_id, choices=[StreamChoice(delta={'reasoning_content': content})]).model_dump_json()}\n\n"
        else:
            normal_content += content
            yield f"data: {StreamResponse(id=stream_id, created=created_time, model=clean_model_id, choices=[StreamChoice(delta={'content': content})]).model_dump_json()}\n\n"

    def process_thinking_content(content: str):
        """处理包含思考标签的内容"""
        nonlocal is_thinking, thinking_content, normal_content

        if "<think>" in content:
            parts = content.split("<think>", 1)
            if parts[0]:  # 思考标签前的内容
                normal_content += parts[0]
                yield f"data: {StreamResponse(id=stream_id, created=created_time, model=clean_model_id, choices=[StreamChoice(delta={'content': parts[0]})]).model_dump_json()}\n\n"

            is_thinking = True
            thinking_part = parts[1]

            if "</think>" in thinking_part:
                think_parts = thinking_part.split("</think>", 1)
                thinking_content += think_parts[0]
                yield f"data: {StreamResponse(id=stream_id, created=created_time, model=clean_model_id, choices=[StreamChoice(delta={'reasoning_content': think_parts[0]})]).model_dump_json()}\n\n"

                is_thinking = False
                if think_parts[1]:  # 思考标签后的内容
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
            if parts[1]:  # 思考标签后的内容
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

            # 处理奖励信息
            if chunk_id == "a":
                reward_info = data
                log_debug(f"Found reward info: {reward_info}")

            # 处理初始设置信息
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

            # 处理目标流内容
            elif target_stream_id and chunk_id == target_stream_id:
                if isinstance(data, dict):
                    content = data.get("curr", "")
                    if content:
                        log_debug(
                            f"Processing target stream content: '{content[:50]}...'"
                        )
                        yield from process_content_chunk(content, chunk_id)

                        # 更新目标流ID
                        target_stream_id = extract_ref_id(data.get("next"))
                        if target_stream_id:
                            log_debug(
                                f"Updated target stream ID to: {target_stream_id}"
                            )

            # 备用逻辑：处理任何包含"curr"的chunk
            elif isinstance(data, dict) and "curr" in data:
                content = data.get("curr", "")
                if content:
                    log_debug(
                        f"Processing fallback chunk {chunk_id} with content: '{content[:50]}...'"
                    )
                    yield from process_content_chunk(content, chunk_id)

        log_debug(f"Finished processing {line_count} lines")

    except Exception as e:
        error_type = (
            ErrorType.NETWORK_ERROR
            if isinstance(e, requests.exceptions.RequestException)
            else ErrorType.STREAM_ERROR
        )
        message = (
            "Upstream stream closed unexpectedly"
            if error_type == ErrorType.NETWORK_ERROR
            else "Stream processing error"
        )
        log_error(error_type, message, account.get("token"), "chat.stream", e)
        error_payload = {
            "error": {
                "message": f"{message}: {str(e)[:200]}",
                "type": error_type.value,
            }
        }
        yield f"data: {json.dumps(error_payload)}\n\n"

    finally:
        # 发送完成信号
        yield f"data: {StreamResponse(id=stream_id, created=created_time, model=clean_model_id, choices=[StreamChoice(delta={}, finish_reason='stop')]).model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"

        # 领取奖励
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
    """构建非流式响应"""
    full_content = ""
    full_reasoning_content = ""
    reward_id = None

    # 用于存储从流式响应中获取的模型名称
    response_model_name = model_id

    for event in yupp_stream_generator(response_lines, model_id, account):
        if event.startswith("data:"):
            data_str = event[5:].strip()
            if data_str == "[DONE]":
                break

            try:
                data = json.loads(data_str)
                if "error" in data:
                    error_payload = data["error"]
                    error_message = error_payload.get(
                        "message", "Upstream stream reported an error."
                    )
                    error_value = error_payload.get(
                        "type", ErrorType.STREAM_ERROR.value
                    )
                    error_type = next(
                        (etype for etype in ErrorType if etype.value == error_value),
                        ErrorType.STREAM_ERROR,
                    )
                    raise HTTPException(
                        status_code=500,
                        detail=build_error_detail(error_message, error_type),
                    )

                # 从第一个有效响应中获取模型名称（已经清理过）
                if "model" in data and not response_model_name:
                    response_model_name = data["model"]

                delta = data.get("choices", [{}])[0].get("delta", {})
                if "content" in delta:
                    full_content += delta["content"]
                if "reasoning_content" in delta:
                    full_reasoning_content += delta["reasoning_content"]
            except json.JSONDecodeError:
                continue

    # 构建完整响应
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
    """使用Yupp.ai创建聊天完成 with enhanced error handling"""
    # 查找模型
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

    # 格式化消息
    question = format_messages_for_yupp(request.messages)
    log_debug(f"Formatted question: {question[:100]}...")

    if not YUPP_ACCOUNTS:
        raise_tokens_unavailable(
            "No Yupp.ai tokens configured on server. Please set YUPP_TOKENS.")

    # 尝试所有账户
    failure_reasons: List[str] = []
    request_timeout = float(os.getenv("YUPP_REQUEST_TIMEOUT", "45"))
    endpoint_label = "chat.completions"

    def record_failure(acc: YuppAccount, reason: str):
        failure_reasons.append(f"{mask_token(acc['token'])}: {reason}")

    for attempt in range(len(YUPP_ACCOUNTS)):
        account = get_best_yupp_account()
        if not account:
            all_invalid = all(not acc["is_valid"] for acc in YUPP_ACCOUNTS)
            in_cooldown = sum(
                1 for acc in YUPP_ACCOUNTS if acc.get("cooldown_until", 0) > time.time()
            )

            if all_invalid:
                raise_tokens_unavailable(
                    "All upstream tokens are marked invalid (authentication failed). Please check token configuration.",
                    failure_reasons,
                )
            elif in_cooldown > 0:
                raise_tokens_unavailable(
                    f"All upstream tokens temporarily unavailable ({in_cooldown} in cooldown due to errors). Please try again later.",
                    failure_reasons,
                )
            else:
                raise_tokens_unavailable(
                    "No valid Yupp.ai accounts available. Please check configuration.",
                    failure_reasons,
                )

        try:
            # 构建请求
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
                f"Sending request to Yupp.ai with account token ending in {mask_token(account['token'])}"
            )

            # 发送请求
            session = create_requests_session()
            response = session.post(
                url,
                data=json.dumps(payload),
                headers=headers,
                stream=True,
                timeout=request_timeout,
            )
            response.raise_for_status()

            # 处理响应
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

        except requests.exceptions.HTTPError as http_error:
            status_code = http_error.response.status_code
            error_type = classify_error(http_error)
            log_error(
                error_type,
                f"Upstream HTTP error {status_code}",
                account.get("token"),
                endpoint_label,
                http_error,
            )
            record_failure(account, f"http_{status_code}")

            with account_rotation_lock:
                if should_mark_invalid(error_type):
                    mark_account_invalid(account, f"HTTP {status_code}")
                elif should_increment_error_count(error_type):
                    increment_account_error(account)
                    set_account_cooldown(account)

            if error_type == ErrorType.CLIENT_ERROR:
                raise HTTPException(
                    status_code=status_code,
                    detail=build_error_detail(
                        "Yupp.ai rejected the request. Please validate your payload.",
                        error_type,
                        {"upstream_status": status_code},
                    ),
                )

        except requests.exceptions.Timeout as timeout_error:
            log_error(
                ErrorType.NETWORK_ERROR,
                "Request to Yupp.ai timed out",
                account.get("token"),
                endpoint_label,
                timeout_error,
            )
            record_failure(account, "timeout")
            with account_rotation_lock:
                increment_account_error(account)
                set_account_cooldown(account)

        except requests.exceptions.ConnectionError as conn_error:
            log_error(
                ErrorType.NETWORK_ERROR,
                "Network error contacting Yupp.ai",
                account.get("token"),
                endpoint_label,
                conn_error,
            )
            record_failure(account, "connection_error")
            with account_rotation_lock:
                increment_account_error(account)
                set_account_cooldown(account)

        except requests.exceptions.RequestException as request_error:
            error_type = classify_error(request_error)
            log_error(
                error_type,
                "Unexpected requests exception calling Yupp.ai",
                account.get("token"),
                endpoint_label,
                request_error,
            )
            record_failure(account, error_type.value)
            with account_rotation_lock:
                increment_account_error(account)
                set_account_cooldown(account)

        except Exception as unexpected_error:
            log_error(
                ErrorType.UNKNOWN_ERROR,
                "Unexpected error while handling chat completion",
                account.get("token"),
                endpoint_label,
                unexpected_error,
            )
            record_failure(account, "unexpected_error")
            with account_rotation_lock:
                increment_account_error(account)
                set_account_cooldown(account, 60)

    raise_tokens_unavailable(
        "All attempts to contact Yupp.ai API failed. Please try again later.",
        failure_reasons,
    )


def main():
    """主函数：启动 Yupp.ai OpenAI API Adapter 服务"""
    import uvicorn
    from dotenv import load_dotenv

    # 加载环境变量
    load_dotenv()

    # 设置全局配置
    global DEBUG_MODE
    DEBUG_MODE = os.environ.get("DEBUG_MODE", "false").lower() == "true"

    if DEBUG_MODE:
        print("Debug mode enabled")

    # 检查必要的环境变量
    if not os.getenv("CLIENT_API_KEYS"):
        print("Warning: CLIENT_API_KEYS environment variable not set.")
    if not os.getenv("YUPP_TOKENS"):
        print("Warning: YUPP_TOKENS environment variable not set.")

    # 加载配置
    load_client_api_keys()
    load_yupp_accounts()
    load_yupp_models()

    # 显示启动信息
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

    # 启动服务器
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8001"))

    print(f"Starting server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
