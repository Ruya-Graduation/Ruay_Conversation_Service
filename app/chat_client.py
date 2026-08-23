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
    Synchronous OpenRouter chat completion call.

    The system prompt is automatically inserted as the first
    message because OpenRouter uses OpenAI-compatible messages
    rather than a separate system_prompt field.

    Args:
        base_url:
            e.g. "https://openrouter.ai/api/v1"

        api_key:
            OpenRouter API key.

        model_id:
            OpenRouter model ID.

        system_prompt:
            Instructions that define the LLM's behavior.

        messages:
            Conversation history + current user message.

        max_tokens:
            Maximum number of tokens in the response.

        timeout:
            HTTP request timeout in seconds.

    Returns:
        Parsed JSON response from OpenRouter.

    Raises:
        HTTPException(502) on API/network errors.
        HTTPException(504) on timeout.
    """

    url = f"{base_url.rstrip('/')}/chat/completions"

    # ---------------------------------------------------------
    # Build final message list
    # ---------------------------------------------------------
    llm_messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        *messages,
    ]

    payload = {
        "model": model_id,
        "messages": llm_messages,
        "max_tokens": max_tokens,
    }

    logger.info(
        "Calling OpenRouter API | model=%s | messages=%d",
        model_id,
        len(llm_messages),
    )

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
        logger.error(
            "OpenRouter API timed out after %ds",
            timeout,
        )

        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=(
                f"LLM API timed out after {timeout}s. "
                "Try again or increase the timeout."
            ),
        )

    except requests.exceptions.HTTPError as exc:
        response_status = (
            exc.response.status_code
            if exc.response is not None
            else 0
        )

        body = ""

        try:
            body = (
                exc.response.text
                if exc.response is not None
                else ""
            )
        except Exception:
            pass

        logger.error(
            "OpenRouter API HTTP error %d: %s",
            response_status,
            body,
        )

        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=(
                f"LLM API returned HTTP {response_status}: {body}"
            ),
        )

    except requests.exceptions.RequestException as exc:
        logger.error(
            "OpenRouter API request failed: %s",
            exc,
        )

        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"LLM API request failed: {exc}",
        )

# def _call_chat_sync(
#     *,
#     base_url: str,
#     api_key: str,
#     model_id: str,
#     system_prompt: str,
#     messages: list[dict],
#     max_tokens: int,
#     timeout: int = 120,
# ) -> dict:
#     """
#     Synchronous SBG chat call.  Runs inside asyncio.to_thread.

#     Args:
#         base_url:      e.g. "https://your-api.example.com/api/v1"
#         api_key:       Bearer token for the SBG API.
#         model_id:      LLM model identifier, e.g. "anthropic.claude-3-haiku-...".
#         system_prompt: The fully-built system prompt from prompt_builder.
#         messages:      Conversation history + current user turn, OpenAI-style
#                        [{"role": "user"|"assistant", "content": "..."}].
#         max_tokens:    Maximum tokens for the model reply.
#         timeout:       HTTP request timeout in seconds.

#     Returns:
#         Parsed JSON response dict from the SBG API.

#     Raises:
#         HTTPException(502) on any API-level or network error.
#     """
#     url = f"{base_url.rstrip('/')}/student/chat"

#     payload = {
#         "model_id": model_id,
#         "messages": messages,
#         "system_prompt": system_prompt,
#         "max_tokens": max_tokens,
#     }

#     logger.info("Calling SBG chat API | model=%s | messages=%d", model_id, len(messages))

#     try:
#         response = requests.post(
#             url,
#             headers={
#                 "Authorization": f"Bearer {api_key}",
#                 "Content-Type": "application/json",
#             },
#             json=payload,
#             timeout=timeout,
#         )
#         response.raise_for_status()
#         return response.json()

#     except requests.exceptions.Timeout:
#         logger.error("SBG chat API timed out after %ds", timeout)
#         raise HTTPException(
#             status_code=status.HTTP_504_GATEWAY_TIMEOUT,
#             detail=f"LLM API timed out after {timeout}s. Try again or increase the timeout.",
#         )
#     except requests.exceptions.HTTPError as exc:
#         status_code = exc.response.status_code if exc.response is not None else 0
#         body = ""
#         try:
#             body = exc.response.text if exc.response is not None else ""
#         except Exception:
#             pass
#         logger.error("SBG chat API HTTP error %d: %s", status_code, body)
#         raise HTTPException(
#             status_code=status.HTTP_502_BAD_GATEWAY,
#             detail=f"LLM API returned HTTP {status_code}: {body}",
#         )
#     except requests.exceptions.RequestException as exc:
#         logger.error("SBG chat API request failed: %s", exc)
#         raise HTTPException(
#             status_code=status.HTTP_502_BAD_GATEWAY,
#             detail=f"LLM API request failed: {exc}",
#         )


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


# def extract_answer(response_json: dict) -> dict:
#     """
#     Extract structured response data from the SBG API response.
    
#     Returns a dictionary with separated fields:
#     - output_text: The actual answer from the model
#     - request_id: Request identifier
#     - model_id: Model used
#     - region: Deployment region
#     - usage: Token usage statistics
#     - estimated_cost_usd: Estimated cost
#     - actual_cost_usd: Actual cost
#     - status: Request status
#     - (any other fields from the response)
    
#     This allows the .NET server to access all metadata separately.
#     """
#     result = {}
    
#     # Extract the actual answer text (output_text)
#     output_text = None
    
#     # Try different response shapes for the answer
#     # Shape: { "output_text": "..." }  (SBG style)
#     if "output_text" in response_json:
#         output_text = str(response_json["output_text"])
#     # Shape: { "content": [{"text": "..."}] }   (Anthropic/Bedrock style)
#     elif "content" in response_json:
#         content = response_json["content"]
#         if isinstance(content, list) and content:
#             first = content[0]
#             if isinstance(first, dict) and "text" in first:
#                 output_text = str(first["text"])
#             elif isinstance(first, str):
#                 output_text = first
#     # Shape: { "choices": [{"message": {"content": "..."}}] }  (OpenAI style)
#     elif "choices" in response_json:
#         choices = response_json["choices"]
#         if isinstance(choices, list) and choices:
#             message = choices[0].get("message", {})
#             if "content" in message:
#                 output_text = str(message["content"])
#     # Shape: { "text": "..." }  or  { "response": "..." }
#     else:
#         for key in ("text", "response", "answer", "output", "result"):
#             if key in response_json and isinstance(response_json[key], str):
#                 output_text = response_json[key]
#                 break
    
#     # If we still don't have output_text, use the whole response as JSON
#     if output_text is None:
#         import json
#         output_text = json.dumps(response_json, ensure_ascii=False)
    
#     result["output_text"] = output_text
    
#     # Extract all other fields from the response
#     for key, value in response_json.items():
#         if key != "output_text":  # Don't duplicate output_text
#             result[key] = value
    
#     return result

#OLD VERSION WITH BIGS
def extract_answer(response_json: dict) -> dict:
    """
    Extract structured response data from the LLM API response.

    Returns a dictionary with separated fields:
    - output_text: The actual answer from the model
    - request_id: Request identifier (if available)
    - model_id: Model used (if available)
    - provider: LLM provider (if available)
    - usage: Token usage statistics
    - cost: Cost information (if available)
    - status: Request status (if available)
    - (any other fields from the response)

    Supports multiple response formats:
    - SBG style
    - Anthropic/Bedrock style
    - OpenAI/OpenRouter style
    - Generic text/response/answer formats

    This allows the .NET server to access the actual answer
    separately while preserving all response metadata.
    """

    result = {}

    # ---------------------------------------------------------
    # Extract the actual answer text
    # ---------------------------------------------------------
    output_text = None

    # Shape:
    # {
    #     "output_text": "..."
    # }
    if "output_text" in response_json:
        value = response_json["output_text"]

        if value is not None:
            output_text = str(value)

    # Shape:
    # {
    #     "content": [
    #         {
    #             "text": "..."
    #         }
    #     ]
    # }
    #
    # Anthropic / Bedrock style
    elif "content" in response_json:
        content = response_json["content"]

        if isinstance(content, list) and content:
            first = content[0]

            if isinstance(first, dict) and "text" in first:
                output_text = str(first["text"])

            elif isinstance(first, str):
                output_text = first

        elif isinstance(content, str):
            output_text = content

    # Shape:
    #
    # {
    #     "choices": [
    #         {
    #             "message": {
    #                 "role": "assistant",
    #                 "content": "..."
    #             }
    #         }
    #     ]
    # }
    #
    # OpenAI / OpenRouter style
    elif "choices" in response_json:
        choices = response_json["choices"]

        if isinstance(choices, list) and choices:
            first_choice = choices[0]

            if isinstance(first_choice, dict):
                message = first_choice.get("message", {})

                if isinstance(message, dict):
                    content = message.get("content")

                    if isinstance(content, str):
                        output_text = content

                    # Some providers may return structured content
                    elif isinstance(content, list):
                        text_parts = []

                        for item in content:
                            if isinstance(item, dict):
                                text = item.get("text")

                                if isinstance(text, str):
                                    text_parts.append(text)

                            elif isinstance(item, str):
                                text_parts.append(item)

                        if text_parts:
                            output_text = "".join(text_parts)

    # Generic response formats
    #
    # {
    #     "text": "...",
    #     "response": "...",
    #     "answer": "...",
    #     "output": "...",
    #     "result": "..."
    # }
    else:
        for key in (
            "text",
            "response",
            "answer",
            "output",
            "result",
        ):
            value = response_json.get(key)

            if isinstance(value, str):
                output_text = value
                break

    # ---------------------------------------------------------
    # Fallback
    # ---------------------------------------------------------
    # If no known response format was found, return the entire
    # response as JSON rather than returning an empty answer.
    # ---------------------------------------------------------
    if output_text is None:
        import json

        output_text = json.dumps(
            response_json,
            ensure_ascii=False,
        )

    result["output_text"] = output_text

    # ---------------------------------------------------------
    # Preserve all response metadata
    # ---------------------------------------------------------
    #
    # OpenRouter example:
    #
    # id
    # object
    # created
    # model
    # provider
    # choices
    # usage
    # etc.
    #
    # We keep all of it so the .NET server can access it.
    # ---------------------------------------------------------
    for key, value in response_json.items():
        if key != "output_text":
            result[key] = value

    return result

# def extract_answer(response_json: dict) -> dict:
#     """
#     Extract the actual LLM answer while preserving the original
#     gateway response metadata.

#     For OpenAI/OpenRouter-style responses:
#         response_json["choices"][0]["message"]["content"]

#     is mapped to:
#         result["output_text"]

#     All other response metadata is preserved.
#     """

#     result = {}

#     output_text = None

#     # ---------------------------------------------------------
#     # 1. Direct output_text format
#     # ---------------------------------------------------------

#     if isinstance(response_json.get("output_text"), str):
#         output_text = response_json["output_text"]

#     # ---------------------------------------------------------
#     # 2. Anthropic / Bedrock style
#     # ---------------------------------------------------------

#     elif "content" in response_json:
#         content = response_json["content"]

#         if isinstance(content, str):
#             output_text = content

#         elif isinstance(content, list):
#             text_parts = []

#             for item in content:
#                 if isinstance(item, dict):
#                     text = item.get("text")

#                     if isinstance(text, str):
#                         text_parts.append(text)

#                 elif isinstance(item, str):
#                     text_parts.append(item)

#             if text_parts:
#                 output_text = "".join(text_parts)

#     # ---------------------------------------------------------
#     # 3. OpenAI / OpenRouter style
#     # ---------------------------------------------------------

#     elif isinstance(response_json.get("choices"), list):
#         choices = response_json["choices"]

#         if choices:
#             first_choice = choices[0]

#             if isinstance(first_choice, dict):

#                 # Preserve finish reason
#                 result["finish_reason"] = first_choice.get("finish_reason")
#                 result["native_finish_reason"] = first_choice.get(
#                     "native_finish_reason"
#                 )

#                 message = first_choice.get("message")

#                 if isinstance(message, dict):
#                     content = message.get("content")

#                     # Normal response
#                     if isinstance(content, str):
#                         output_text = content

#                     # Some providers return structured content
#                     elif isinstance(content, list):
#                         text_parts = []

#                         for item in content:
#                             if isinstance(item, dict):
#                                 text = item.get("text")

#                                 if isinstance(text, str):
#                                     text_parts.append(text)

#                             elif isinstance(item, str):
#                                 text_parts.append(item)

#                         if text_parts:
#                             output_text = "".join(text_parts)

#     # ---------------------------------------------------------
#     # 4. Generic formats
#     # ---------------------------------------------------------

#     elif output_text is None:
#         for key in (
#             "text",
#             "response",
#             "answer",
#             "output",
#             "result",
#         ):
#             value = response_json.get(key)

#             if isinstance(value, str):
#                 output_text = value
#                 break

#     # ---------------------------------------------------------
#     # 5. Never dump the entire gateway response into answer
#     # ---------------------------------------------------------

#     if output_text is None:
#         output_text = ""

#     result["output_text"] = output_text

#     # ---------------------------------------------------------
#     # 6. Preserve original response metadata
#     # ---------------------------------------------------------

#     for key, value in response_json.items():
#         if key != "output_text":
#             result[key] = value

#     return result