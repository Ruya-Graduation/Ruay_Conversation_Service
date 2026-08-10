"""
app/chat_client.py
──────────────────
Async wrapper for the SBG chat completion API.

The SBG client uses the synchronous `requests` library, so every call is
offloaded to a thread pool via asyncio.to_thread to keep the FastAPI event
loop unblocked.
"""

from __future__ import annotations

import asyncio
import logging

import requests
from fastapi import HTTPException, status

logger = logging.getLogger(__name__)


def _call_chat_sync(
    *,
    base_url: str,
    api_key: str,
    model_id: str,
    system_prompt: str,
    messages: list[dict],
    max_tokens: int,
    timeout: int = 120,
) -> dict:
    """
    Synchronous SBG chat call.  Runs inside asyncio.to_thread.

    Args:
        base_url:      e.g. "https://your-api.example.com/api/v1"
        api_key:       Bearer token for the SBG API.
        model_id:      LLM model identifier, e.g. "anthropic.claude-3-haiku-...".
        system_prompt: The fully-built system prompt from prompt_builder.
        messages:      Conversation history + current user turn, OpenAI-style
                       [{"role": "user"|"assistant", "content": "..."}].
        max_tokens:    Maximum tokens for the model reply.
        timeout:       HTTP request timeout in seconds.

    Returns:
        Parsed JSON response dict from the SBG API.

    Raises:
        HTTPException(502) on any API-level or network error.
    """
    url = f"{base_url.rstrip('/')}/student/chat"

    payload = {
        "model_id": model_id,
        "messages": messages,
        "system_prompt": system_prompt,
        "max_tokens": max_tokens,
    }

    logger.info("Calling SBG chat API | model=%s | messages=%d", model_id, len(messages))

    try:
        response = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=timeout,
        )
        response.raise_for_status()
        return response.json()

    except requests.exceptions.Timeout:
        logger.error("SBG chat API timed out after %ds", timeout)
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=f"LLM API timed out after {timeout}s. Try again or increase the timeout.",
        )
    except requests.exceptions.HTTPError as exc:
        status_code = exc.response.status_code if exc.response is not None else 0
        body = ""
        try:
            body = exc.response.text if exc.response is not None else ""
        except Exception:
            pass
        logger.error("SBG chat API HTTP error %d: %s", status_code, body)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"LLM API returned HTTP {status_code}: {body}",
        )
    except requests.exceptions.RequestException as exc:
        logger.error("SBG chat API request failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"LLM API request failed: {exc}",
        )


async def call_chat(
    *,
    base_url: str,
    api_key: str,
    model_id: str,
    system_prompt: str,
    messages: list[dict],
    max_tokens: int,
) -> dict:
    """
    Async wrapper around _call_chat_sync.

    Returns the raw parsed JSON from the SBG API.
    The caller is responsible for extracting the answer text.
    """
    return await asyncio.to_thread(
        _call_chat_sync,
        base_url=base_url,
        api_key=api_key,
        model_id=model_id,
        system_prompt=system_prompt,
        messages=messages,
        max_tokens=max_tokens,
    )


def extract_answer(response_json: dict) -> str:
    """
    Extract the assistant's reply text from the SBG API response.

    Tries several common response shapes. Falls back to a string
    representation of the whole response if nothing matches.
    """
    # Shape: { "content": [{"text": "..."}] }   (Anthropic/Bedrock style)
    if "content" in response_json:
        content = response_json["content"]
        if isinstance(content, list) and content:
            first = content[0]
            if isinstance(first, dict) and "text" in first:
                return str(first["text"])
            if isinstance(first, str):
                return first

    # Shape: { "choices": [{"message": {"content": "..."}}] }  (OpenAI style)
    if "choices" in response_json:
        choices = response_json["choices"]
        if isinstance(choices, list) and choices:
            message = choices[0].get("message", {})
            if "content" in message:
                return str(message["content"])

    # Shape: { "text": "..." }  or  { "response": "..." }
    for key in ("text", "response", "answer", "output", "result"):
        if key in response_json and isinstance(response_json[key], str):
            return response_json[key]

    # Last resort: return JSON string so the caller always gets something
    import json
    return json.dumps(response_json, ensure_ascii=False)
