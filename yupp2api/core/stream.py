"""Streaming helpers for converting Yupp responses to OpenAI format."""

from __future__ import annotations

import json
import re
import time
import uuid
from typing import Dict, Generator

from fastapi import HTTPException

from ..config import Settings
from ..models import (
    ChatCompletionChoice,
    ChatCompletionResponse,
    ChatMessage,
    StreamChoice,
    StreamResponse,
    YuppAccount,
)
from ..tokens import claim_yupp_reward
from ..utils import clean_model_name, log_debug


def yupp_stream_generator(response_lines, model_id: str, account: YuppAccount, settings: Settings) -> Generator[str, None, None]:
    """Process Yupp stream response and yield OpenAI formatted SSE chunks."""
    stream_id = f"chatcmpl-{uuid.uuid4().hex}"
    created_time = int(time.time())
    clean_model_id = clean_model_name(model_id)

    yield (
        "data: "
        + StreamResponse(
            id=stream_id,
            created=created_time,
            model=clean_model_id,
            choices=[StreamChoice(delta={"role": "assistant"})],
        ).model_dump_json()
        + "\n\n"
    )

    line_pattern = re.compile(b"^([0-9a-fA-F]+):(.*)")
    chunks: Dict[str, Dict] = {}
    target_stream_id = None
    reward_info = None
    is_thinking = False
    thinking_content = ""
    normal_content = ""
    select_stream = [None, None]
    processed_content = set()

    def extract_ref_id(ref):
        return ref[2:] if ref and isinstance(ref, str) and ref.startswith("$@") else None

    def is_valid_content(content: str) -> bool:
        if not content or content in [None, "", "$undefined"]:
            return False
        if content.startswith("\\n\\<streaming stopped") or content.startswith("\n\\<streaming stopped"):
            return False
        if re.match(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", content.strip()):
            return False
        if len(content.strip()) == 0:
            return False
        if content.strip() in ["$undefined", "undefined", "null", "NULL"]:
            return False
        return True

    def process_content_chunk(content: str, chunk_id: str):
        nonlocal is_thinking, thinking_content, normal_content
        if not is_valid_content(content):
            return
        content_hash = hash(content)
        if content_hash in processed_content:
            return
        processed_content.add(content_hash)
        log_debug(settings, f"Processing chunk {chunk_id} with content: '{content[:50]}...'")
        if "<think>" in content or "</think>" in content:
            yield from process_thinking_content(content)
        elif is_thinking:
            thinking_content += content
            yield (
                "data: "
                + StreamResponse(
                    id=stream_id,
                    created=created_time,
                    model=clean_model_id,
                    choices=[StreamChoice(delta={"reasoning_content": content})],
                ).model_dump_json()
                + "\n\n"
            )
        else:
            normal_content += content
            yield (
                "data: "
                + StreamResponse(
                    id=stream_id,
                    created=created_time,
                    model=clean_model_id,
                    choices=[StreamChoice(delta={"content": content})],
                ).model_dump_json()
                + "\n\n"
            )

    def process_thinking_content(content: str):
        nonlocal is_thinking, thinking_content, normal_content
        if "<think>" in content:
            parts = content.split("<think>", 1)
            if parts[0]:
                normal_content += parts[0]
                yield (
                    "data: "
                    + StreamResponse(
                        id=stream_id,
                        created=created_time,
                        model=clean_model_id,
                        choices=[StreamChoice(delta={"content": parts[0]})],
                    ).model_dump_json()
                    + "\n\n"
                )
            is_thinking = True
            thinking_part = parts[1]
            if "</think>" in thinking_part:
                think_parts = thinking_part.split("</think>", 1)
                thinking_content += think_parts[0]
                yield (
                    "data: "
                    + StreamResponse(
                        id=stream_id,
                        created=created_time,
                        model=clean_model_id,
                        choices=[StreamChoice(delta={"reasoning_content": think_parts[0]})],
                    ).model_dump_json()
                    + "\n\n"
                )
                is_thinking = False
                if think_parts[1]:
                    normal_content += think_parts[1]
                    yield (
                        "data: "
                        + StreamResponse(
                            id=stream_id,
                            created=created_time,
                            model=clean_model_id,
                            choices=[StreamChoice(delta={"content": think_parts[1]})],
                        ).model_dump_json()
                        + "\n\n"
                    )
            else:
                thinking_content += thinking_part
                yield (
                    "data: "
                    + StreamResponse(
                        id=stream_id,
                        created=created_time,
                        model=clean_model_id,
                        choices=[StreamChoice(delta={"reasoning_content": thinking_part})],
                    ).model_dump_json()
                    + "\n\n"
                )
        elif "</think>" in content and is_thinking:
            parts = content.split("</think>", 1)
            thinking_content += parts[0]
            yield (
                "data: "
                + StreamResponse(
                    id=stream_id,
                    created=created_time,
                    model=clean_model_id,
                    choices=[StreamChoice(delta={"reasoning_content": parts[0]})],
                ).model_dump_json()
                + "\n\n"
            )
            is_thinking = False
            if parts[1]:
                normal_content += parts[1]
                yield (
                    "data: "
                    + StreamResponse(
                        id=stream_id,
                        created=created_time,
                        model=clean_model_id,
                        choices=[StreamChoice(delta={"content": parts[1]})],
                    ).model_dump_json()
                    + "\n\n"
                )

    try:
        log_debug(settings, "Starting to process response lines...")
        line_count = 0
        for line in response_lines:
            line_count += 1
            if not line:
                continue
            match = line_pattern.match(line)
            if not match:
                log_debug(settings, f"Line {line_count}: No pattern match for line: {line[:50]}...")
                continue
            chunk_id, chunk_data = match.groups()
            chunk_id = chunk_id.decode()
            try:
                data = json.loads(chunk_data) if chunk_data != b"{}" else {}
                chunks[chunk_id] = data
                log_debug(settings, f"Parsed chunk {chunk_id}: {str(data)[:100]}...")
            except json.JSONDecodeError:
                log_debug(settings, f"Failed to parse JSON for chunk {chunk_id}: {chunk_data}")
                continue
            if chunk_id == "a":
                reward_info = data
                log_debug(settings, f"Found reward info: {reward_info}")
            elif chunk_id == "1":
                if isinstance(data, dict):
                    left_stream = data.get("leftStream", {})
                    right_stream = data.get("rightStream", {})
                    select_stream = [left_stream, right_stream]
                    log_debug(settings, f"Found stream setup: left={left_stream}, right={right_stream}")
            elif chunk_id == "e":
                if isinstance(data, dict):
                    for i, selection in enumerate(data.get("modelSelections", [])):
                        if selection.get("selectionSource") == "USER_SELECTED":
                            if i < len(select_stream) and isinstance(select_stream[i], dict):
                                target_stream_id = extract_ref_id(select_stream[i].get("next"))
                                log_debug(settings, f"Found target stream ID: {target_stream_id}")
                            break
            elif target_stream_id and chunk_id == target_stream_id:
                if isinstance(data, dict):
                    content = data.get("curr", "")
                    if content:
                        log_debug(settings, f"Processing target stream content: '{content[:50]}...'")
                        yield from process_content_chunk(content, chunk_id)
                        target_stream_id = extract_ref_id(data.get("next"))
                        if target_stream_id:
                            log_debug(settings, f"Updated target stream ID to: {target_stream_id}")
            elif isinstance(data, dict) and "curr" in data:
                content = data.get("curr", "")
                if content:
                    log_debug(settings, f"Processing fallback chunk {chunk_id} with content: '{content[:50]}...'")
                    yield from process_content_chunk(content, chunk_id)
        log_debug(settings, f"Finished processing {line_count} lines")
    except Exception as exc:  # pylint: disable=broad-except
        log_debug(settings, f"Stream processing error: {exc}")
        print(f"Stream processing error: {exc}")
        yield f"data: {json.dumps({'error': str(exc)})}\n\n"
    finally:
        yield (
            "data: "
            + StreamResponse(
                id=stream_id,
                created=created_time,
                model=clean_model_id,
                choices=[StreamChoice(delta={}, finish_reason="stop")],
            ).model_dump_json()
            + "\n\n"
        )
        yield "data: [DONE]\n\n"
        if reward_info and "unclaimedRewardInfo" in reward_info:
            reward_id = reward_info["unclaimedRewardInfo"].get("rewardId")
            if reward_id:
                try:
                    claim_yupp_reward(account, reward_id, settings)
                except Exception as exc:  # pylint: disable=broad-except
                    print(f"Failed to claim reward in background: {exc}")
        log_debug(
            settings,
            f"Stream processing completed. Total content: {len(normal_content)} chars, thinking: {len(thinking_content)} chars",
        )


def build_yupp_non_stream_response(response_lines, model_id: str, account: YuppAccount, settings: Settings) -> ChatCompletionResponse:
    """Build non-streaming response from streamed chunks."""
    full_content = ""
    full_reasoning_content = ""
    response_model_name = model_id

    for event in yupp_stream_generator(response_lines, model_id, account, settings):
        if event.startswith("data:"):
            data_str = event[5:].strip()
            if data_str == "[DONE]":
                break
            try:
                data = json.loads(data_str)
                if "error" in data:
                    raise HTTPException(status_code=500, detail=data["error"])
                if "model" in data and not response_model_name:
                    response_model_name = data["model"]
                delta = data.get("choices", [{}])[0].get("delta", {})
                if "content" in delta:
                    full_content += delta["content"]
                if "reasoning_content" in delta:
                    full_reasoning_content += delta["reasoning_content"]
            except json.JSONDecodeError:
                continue
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
