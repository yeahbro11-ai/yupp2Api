"""Router exposing /v1/chat/completions endpoint."""

from __future__ import annotations

import json
import uuid
from typing import Union

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
import requests

from ..auth import authenticate_client
from ..config import Settings
from ..core.stream import build_yupp_non_stream_response, yupp_stream_generator
from ..dependencies import get_runtime_state, get_settings
from ..models import ChatCompletionRequest, ChatCompletionResponse
from ..state import RuntimeState
from ..tokens import get_best_yupp_account
from ..utils import create_requests_session, format_messages_for_yupp, log_debug

router = APIRouter()


@router.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    state: RuntimeState = Depends(get_runtime_state),
    settings: Settings = Depends(get_settings),
    _: None = Depends(authenticate_client),
) -> Union[ChatCompletionResponse, StreamingResponse]:
    """Create chat completion using Yupp.ai."""

    model_info = next((m for m in state.models if m.get("label") == request.model), None)
    if not model_info:
        raise HTTPException(status_code=404, detail=f"Model '{request.model}' not found.")

    model_name = model_info.get("name")
    if not model_name:
        raise HTTPException(status_code=404, detail=f"Model '{request.model}' has no 'name' field.")

    if not request.messages:
        raise HTTPException(status_code=400, detail="No messages provided in the request.")

    log_debug(settings, f"Processing request for model: {request.model} (Yupp name: {model_name})")

    question = format_messages_for_yupp(request.messages)
    log_debug(settings, f"Formatted question: {question[:100]}...")

    for _ in range(len(state.accounts)):
        account = get_best_yupp_account(state, settings)
        if not account:
            raise HTTPException(status_code=503, detail="No valid Yupp.ai accounts available.")

        try:
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

            log_debug(settings, f"Sending request to Yupp.ai with account token ending in ...{account['token'][-4:]}")

            session = create_requests_session(settings)
            response = session.post(
                url,
                data=json.dumps(payload),
                headers=headers,
                stream=True,
            )
            response.raise_for_status()

            if request.stream:
                log_debug(settings, "Returning processed response stream")
                return StreamingResponse(
                    yupp_stream_generator(response.iter_lines(), request.model, account, settings),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "X-Accel-Buffering": "no",
                    },
                )
            else:
                log_debug(settings, "Building non-stream response")
                return build_yupp_non_stream_response(response.iter_lines(), request.model, account, settings)

        except requests.exceptions.HTTPError as exc:
            status_code = exc.response.status_code
            error_detail = exc.response.text
            print(f"Yupp.ai API error ({status_code}): {error_detail}")

            with state.account_rotation_lock:
                if status_code in [401, 403]:
                    account["is_valid"] = False
                    print(f"Account ...{account['token'][-4:]} marked as invalid due to auth error.")
                elif status_code in [429, 500, 502, 503, 504]:
                    account["error_count"] += 1
                    print(f"Account ...{account['token'][-4:]} error count: {account['error_count']}")
                else:
                    raise HTTPException(status_code=status_code, detail=error_detail)

        except Exception as exc:  # pylint: disable=broad-except
            print(f"Request error: {exc}")
            with state.account_rotation_lock:
                account["error_count"] += 1

    raise HTTPException(status_code=503, detail="All attempts to contact Yupp.ai API failed.")
