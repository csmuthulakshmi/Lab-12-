"""LLM Gateway: unified interface for LLM providers with retry, fallback, and streaming."""

from __future__ import annotations

import email.utils
import json
import logging
import random
import re
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any, Callable, Iterator, Protocol

import httpx

from genai_platform.config import Settings, get_settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Error hierarchy
# ---------------------------------------------------------------------------

class LLMError(Exception):
    """Base exception for all LLM errors."""
    pass


class RateLimitError(LLMError):
    """Provider returned a 429 or quota limit."""
    def __init__(self, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


class TransientError(LLMError):
    """Network drop, 5xx server error, timeout."""
    pass


class PermanentError(LLMError):
    """Invalid API key, bad model name, unrecoverable 4xx."""
    pass


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total: int = 0


@dataclass
class LLMResult:
    text: str
    usage: Usage
    provider: str
    model: str
    latency_s: float


@dataclass
class Chunk:
    text: str
    usage: Usage | None = None


def estimate_tokens(text: str) -> int:
    """Fast heuristic for token counts (~4 chars/token)."""
    if not text:
        return 0
    return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Retry and Backoff utilities
# ---------------------------------------------------------------------------

def parse_retry_after(headers: dict[str, Any] | None, message: str | None = None) -> float | None:
    """
    Parse retry hints from headers or error message.
    Supports Retry-After (seconds or RFC 2822 date), x-ratelimit-reset-*,
    and Gemini hints like 'retryDelay': '17s' or 'Please retry in 17.5s'.
    """
    if headers:
        # Case-insensitive lookup
        lower_headers = {k.lower(): str(v) for k, v in headers.items()}

        if "retry-after" in lower_headers:
            val = lower_headers["retry-after"].strip()
            # Try numeric
            try:
                sec = float(val)
                if sec >= 0:
                    return sec
            except ValueError:
                # Try HTTP-date
                try:
                    dt = email.utils.parsedate_to_datetime(val)
                    now = datetime.now(timezone.utc)
                    delta = (dt - now).total_seconds()
                    return max(0.0, delta)
                except Exception:
                    pass

        # Check x-ratelimit-reset headers
        for reset_key in ("x-ratelimit-reset", "x-ratelimit-reset-requests", "x-ratelimit-reset-tokens"):
            if reset_key in lower_headers:
                try:
                    val = float(lower_headers[reset_key])
                    # If it's an epoch timestamp (> 1e8)
                    if val > 1e8:
                        now = time.time()
                        return max(0.0, val - now)
                    elif val >= 0:
                        return val
                except ValueError:
                    pass

    if message:
        # Match 'retryDelay': '17s' or 'retryDelay': '17.5s'
        delay_match = re.search(r'retryDelay["\']?\s*:\s*["\']?(\d+(?:\.\d+)?)s?["\']?', message, re.IGNORECASE)
        if delay_match:
            try:
                return float(delay_match.group(1))
            except ValueError:
                pass

        # Match 'Please retry in 17.5s' or 'retry after 17s'
        retry_in_match = re.search(r'(?:please\s+)?retry\s+(?:in|after)\s+(\d+(?:\.\d+)?)s?', message, re.IGNORECASE)
        if retry_in_match:
            try:
                return float(retry_in_match.group(1))
            except ValueError:
                pass

    return None


def backoff_delay(attempt: int, base: float = 1.0, cap: float = 30.0) -> float:
    """Full jitter exponential backoff: random in [0, min(cap, base * 2^attempt)]."""
    max_val = min(cap, base * (2 ** attempt))
    return random.uniform(0.0, max_val)


@dataclass
class RetryConfig:
    max_retries: int = 3
    base_delay: float = 1.0
    cap_delay: float = 30.0


def call_with_retry(
    fn: Callable[[], Any],
    cfg: RetryConfig,
    stats: Counter,
    label: str,
) -> Any:
    """Executes a function with retry on transient/rate-limit errors. Permanent errors fail fast."""
    attempt = 0
    while True:
        try:
            return fn()
        except PermanentError:
            stats[f"{label}_permanent_error"] += 1
            raise
        except RateLimitError as e:
            stats[f"{label}_ratelimit"] += 1
            if attempt >= cfg.max_retries:
                raise
            delay = e.retry_after if (e.retry_after is not None and e.retry_after > 0) else backoff_delay(attempt, cfg.base_delay, cfg.cap_delay)
            stats[f"{label}_retry"] += 1
            logger.warning(f"[{label}] Rate limited (attempt {attempt+1}/{cfg.max_retries}), retrying in {delay:.2f}s: {e}")
            time.sleep(delay)
            attempt += 1
        except TransientError as e:
            stats[f"{label}_transient"] += 1
            if attempt >= cfg.max_retries:
                raise
            delay = backoff_delay(attempt, cfg.base_delay, cfg.cap_delay)
            stats[f"{label}_retry"] += 1
            logger.warning(f"[{label}] Transient error (attempt {attempt+1}/{cfg.max_retries}), retrying in {delay:.2f}s: {e}")
            time.sleep(delay)
            attempt += 1


def stream_with_retry(
    fn: Callable[[], Iterator[Chunk]],
    cfg: RetryConfig,
    stats: Counter,
    label: str,
) -> Iterator[Chunk]:
    """
    Retries ONLY before the first token/chunk is yielded.
    Once any chunk is emitted to the caller, downstream errors are not retried.
    Permanent errors fail fast.
    """
    attempt = 0
    gen: Iterator[Chunk] | None = None
    first_chunk: Chunk | None = None

    while True:
        try:
            gen = fn()
            first_chunk = next(gen)
            break
        except StopIteration:
            return
        except PermanentError:
            stats[f"{label}_permanent_error"] += 1
            raise
        except RateLimitError as e:
            stats[f"{label}_ratelimit"] += 1
            if attempt >= cfg.max_retries:
                raise
            delay = e.retry_after if (e.retry_after is not None and e.retry_after > 0) else backoff_delay(attempt, cfg.base_delay, cfg.cap_delay)
            stats[f"{label}_retry"] += 1
            logger.warning(f"[{label}] Stream rate limit before first token (attempt {attempt+1}), retrying in {delay:.2f}s: {e}")
            time.sleep(delay)
            attempt += 1
        except (TransientError, Exception) as e:
            if isinstance(e, LLMError) and not isinstance(e, TransientError):
                raise
            stats[f"{label}_transient"] += 1
            if attempt >= cfg.max_retries:
                if isinstance(e, LLMError):
                    raise
                raise TransientError(str(e)) from e
            delay = backoff_delay(attempt, cfg.base_delay, cfg.cap_delay)
            stats[f"{label}_retry"] += 1
            logger.warning(f"[{label}] Stream transient error before first token (attempt {attempt+1}), retrying in {delay:.2f}s: {e}")
            time.sleep(delay)
            attempt += 1

    # First token yielded successfully - no further retry allowed
    if first_chunk is not None:
        yield first_chunk
    if gen is not None:
        yield from gen


# ---------------------------------------------------------------------------
# Backend interface and implementations
# ---------------------------------------------------------------------------

class BaseBackend(Protocol):
    name: str
    model: str

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
    ) -> LLMResult:
        ...

    def stream(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
    ) -> Iterator[Chunk]:
        ...


class OllamaBackend:
    name = "ollama"

    def __init__(self, host: str, model: str):
        self.host = host.rstrip("/")
        self.model = model

    def _map_http_error(self, e: Exception) -> LLMError:
        if isinstance(e, (httpx.ConnectError, httpx.TimeoutException, httpx.NetworkError)):
            return TransientError(f"Ollama connection error: {e}")
        if isinstance(e, httpx.HTTPStatusError):
            status = e.response.status_code
            headers = dict(e.response.headers)
            body = e.response.text
            if status == 429:
                return RateLimitError(f"Ollama rate limit: {body}", retry_after=parse_retry_after(headers, body))
            if status >= 500:
                return TransientError(f"Ollama 5xx server error ({status}): {body}")
            if status == 404:
                return PermanentError(f"Ollama model '{self.model}' not found: {body}")
            return PermanentError(f"Ollama client error ({status}): {body}")
        return TransientError(f"Ollama unexpected error: {e}")

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
    ) -> LLMResult:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        }
        if max_tokens:
            payload["options"]["num_predict"] = max_tokens
        if stop:
            payload["options"]["stop"] = stop

        start_time = time.perf_counter()
        try:
            with httpx.Client(timeout=60.0) as client:
                res = client.post(f"{self.host}/api/chat", json=payload)
                res.raise_for_status()
                data = res.json()
        except Exception as e:
            raise self._map_http_error(e) from e

        latency = time.perf_counter() - start_time
        msg = data.get("message", {}).get("content", "")
        # Apply stop sequence if returned
        if stop:
            for s in stop:
                if s in msg:
                    msg = msg.split(s)[0]

        prompt_tokens = data.get("prompt_eval_count") or estimate_tokens(
            "".join(m.get("content", "") for m in messages)
        )
        eval_tokens = data.get("eval_count") or estimate_tokens(msg)
        usage = Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=eval_tokens,
            total=prompt_tokens + eval_tokens,
        )
        return LLMResult(
            text=msg,
            usage=usage,
            provider=self.name,
            model=self.model,
            latency_s=latency,
        )

    def stream(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
    ) -> Iterator[Chunk]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "options": {"temperature": temperature},
        }
        if max_tokens:
            payload["options"]["num_predict"] = max_tokens
        if stop:
            payload["options"]["stop"] = stop

        try:
            client = httpx.Client(timeout=60.0)
            res = client.send(
                client.build_request("POST", f"{self.host}/api/chat", json=payload),
                stream=True,
            )
            res.raise_for_status()
        except Exception as e:
            raise self._map_http_error(e) from e

        accumulated = ""
        prompt_tokens = estimate_tokens("".join(m.get("content", "") for m in messages))

        try:
            for line in res.iter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                content = data.get("message", {}).get("content", "")
                if content:
                    accumulated += content
                    if stop:
                        hit_stop = False
                        for s in stop:
                            if s in accumulated:
                                hit_stop = True
                                break
                        if hit_stop:
                            break
                    yield Chunk(text=content)

                if data.get("done"):
                    eval_tokens = data.get("eval_count") or estimate_tokens(accumulated)
                    prompt_eval = data.get("prompt_eval_count") or prompt_tokens
                    yield Chunk(
                        text="",
                        usage=Usage(
                            prompt_tokens=prompt_eval,
                            completion_tokens=eval_tokens,
                            total=prompt_eval + eval_tokens,
                        ),
                    )
        except Exception as e:
            raise self._map_http_error(e) from e
        finally:
            res.close()
            client.close()


class GeminiBackend:
    name = "gemini"

    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model
        self._client: Any = None

    def _get_client(self) -> Any:
        if not self.api_key:
            raise PermanentError("GEMINI_API_KEY is missing or empty")
        if self._client is None:
            try:
                from google import genai
                self._client = genai.Client(api_key=self.api_key)
            except Exception as e:
                raise PermanentError(f"Failed to initialize Gemini client: {e}") from e
        return self._client

    def _map_error(self, e: Exception) -> LLMError:
        msg = str(e)
        retry_hint = parse_retry_after(None, msg)
        lower_msg = msg.lower()
        if "429" in msg or "quota" in lower_msg or "resource_exhausted" in lower_msg:
            return RateLimitError(f"Gemini rate limit: {msg}", retry_after=retry_hint)
        if "503" in msg or "500" in msg or "unavailable" in lower_msg or "deadline" in lower_msg or "timeout" in lower_msg:
            return TransientError(f"Gemini transient error: {msg}")
        if "api_key" in lower_msg or "permission_denied" in lower_msg or "not_found" in lower_msg or "invalid_argument" in lower_msg or "400" in msg or "403" in msg or "404" in msg:
            return PermanentError(f"Gemini permanent error: {msg}")
        return TransientError(f"Gemini error: {msg}")

    def _format_contents(self, messages: list[dict[str, str]]) -> tuple[str | None, list[Any]]:
        from google.genai import types

        system_instruction: str | None = None
        contents: list[Any] = []

        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "system":
                system_instruction = content
            elif role in ("assistant", "model"):
                contents.append(
                    types.Content(role="model", parts=[types.Part.from_text(text=content)])
                )
            else:
                contents.append(
                    types.Content(role="user", parts=[types.Part.from_text(text=content)])
                )
        return system_instruction, contents

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
    ) -> LLMResult:
        client = self._get_client()
        from google.genai import types

        sys_inst, contents = self._format_contents(messages)
        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            stop_sequences=stop,
            system_instruction=sys_inst,
        )

        start_time = time.perf_counter()
        try:
            response = client.models.generate_content(
                model=self.model,
                contents=contents,
                config=config,
            )
        except Exception as e:
            raise self._map_error(e) from e

        latency = time.perf_counter() - start_time
        text = response.text or ""
        p_tokens = estimate_tokens("".join(m.get("content", "") for m in messages))
        c_tokens = estimate_tokens(text)

        if hasattr(response, "usage_metadata") and response.usage_metadata:
            um = response.usage_metadata
            p_tokens = getattr(um, "prompt_token_count", p_tokens) or p_tokens
            c_tokens = getattr(um, "candidates_token_count", c_tokens) or c_tokens

        usage = Usage(prompt_tokens=p_tokens, completion_tokens=c_tokens, total=p_tokens + c_tokens)
        return LLMResult(
            text=text,
            usage=usage,
            provider=self.name,
            model=self.model,
            latency_s=latency,
        )

    def stream(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
    ) -> Iterator[Chunk]:
        client = self._get_client()
        from google.genai import types

        sys_inst, contents = self._format_contents(messages)
        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            stop_sequences=stop,
            system_instruction=sys_inst,
        )

        try:
            response_stream = client.models.generate_content_stream(
                model=self.model,
                contents=contents,
                config=config,
            )
        except Exception as e:
            raise self._map_error(e) from e

        accumulated = ""
        prompt_tokens = estimate_tokens("".join(m.get("content", "") for m in messages))
        last_usage: Usage | None = None

        try:
            for chunk in response_stream:
                chunk_text = chunk.text or ""
                if chunk_text:
                    accumulated += chunk_text
                    yield Chunk(text=chunk_text)

                if hasattr(chunk, "usage_metadata") and chunk.usage_metadata:
                    um = chunk.usage_metadata
                    p = getattr(um, "prompt_token_count", prompt_tokens) or prompt_tokens
                    c = getattr(um, "candidates_token_count", estimate_tokens(accumulated)) or estimate_tokens(accumulated)
                    last_usage = Usage(prompt_tokens=p, completion_tokens=c, total=p + c)
        except Exception as e:
            raise self._map_error(e) from e

        if not last_usage:
            c = estimate_tokens(accumulated)
            last_usage = Usage(prompt_tokens=prompt_tokens, completion_tokens=c, total=prompt_tokens + c)
        yield Chunk(text="", usage=last_usage)


class OpenAICompatBackend:
    name = "openai"

    def __init__(self, api_key: str, base_url: str, model: str):
        self.api_key = api_key or "sk-placeholder"
        self.base_url = base_url
        self.model = model
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            import openai
            self._client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client

    def _map_error(self, e: Exception) -> LLMError:
        import openai

        if isinstance(e, openai.RateLimitError):
            headers = getattr(e, "response", None) and getattr(e.response, "headers", None)
            retry_hint = parse_retry_after(dict(headers) if headers else None, str(e))
            return RateLimitError(f"OpenAI rate limit: {e}", retry_after=retry_hint)
        if isinstance(e, (openai.APIConnectionError, openai.InternalServerError, openai.APITimeoutError)):
            return TransientError(f"OpenAI transient error: {e}")
        if isinstance(e, (openai.AuthenticationError, openai.NotFoundError, openai.BadRequestError)):
            return PermanentError(f"OpenAI permanent error: {e}")
        return TransientError(f"OpenAI unexpected error: {e}")

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
    ) -> LLMResult:
        client = self._get_client()
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens:
            kwargs["max_tokens"] = max_tokens
        if stop:
            kwargs["stop"] = stop

        start_time = time.perf_counter()
        try:
            resp = client.chat.completions.create(**kwargs)
        except Exception as e:
            raise self._map_error(e) from e

        latency = time.perf_counter() - start_time
        text = resp.choices[0].message.content or ""
        p_tokens = resp.usage.prompt_tokens if resp.usage else estimate_tokens("".join(m.get("content", "") for m in messages))
        c_tokens = resp.usage.completion_tokens if resp.usage else estimate_tokens(text)
        usage = Usage(prompt_tokens=p_tokens, completion_tokens=c_tokens, total=p_tokens + c_tokens)

        return LLMResult(
            text=text,
            usage=usage,
            provider=self.name,
            model=self.model,
            latency_s=latency,
        )

    def stream(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
    ) -> Iterator[Chunk]:
        client = self._get_client()
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if max_tokens:
            kwargs["max_tokens"] = max_tokens
        if stop:
            kwargs["stop"] = stop

        try:
            stream_resp = client.chat.completions.create(**kwargs)
        except Exception as e:
            raise self._map_error(e) from e

        accumulated = ""
        prompt_tokens = estimate_tokens("".join(m.get("content", "") for m in messages))
        last_usage: Usage | None = None

        try:
            for chunk in stream_resp:
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                    c_text = chunk.choices[0].delta.content
                    accumulated += c_text
                    yield Chunk(text=c_text)
                if hasattr(chunk, "usage") and chunk.usage:
                    p = chunk.usage.prompt_tokens
                    c = chunk.usage.completion_tokens
                    last_usage = Usage(prompt_tokens=p, completion_tokens=c, total=p + c)
        except Exception as e:
            raise self._map_error(e) from e

        if not last_usage:
            c = estimate_tokens(accumulated)
            last_usage = Usage(prompt_tokens=prompt_tokens, completion_tokens=c, total=prompt_tokens + c)
        yield Chunk(text="", usage=last_usage)


# ---------------------------------------------------------------------------
# LLM Gateway
# ---------------------------------------------------------------------------

class LLMGateway:
    """Gateway managing LLM requests, retries, and automatic fallback."""

    def __init__(
        self,
        primary: BaseBackend,
        fallback: BaseBackend | None = None,
        retry_config: RetryConfig | None = None,
    ):
        self.primary = primary
        self.fallback = fallback
        self.retry_config = retry_config or RetryConfig()
        self.stats: Counter[str] = Counter()

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
    ) -> LLMResult:
        try:
            return call_with_retry(
                lambda: self.primary.chat(
                    messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stop=stop,
                ),
                self.retry_config,
                self.stats,
                label=self.primary.name,
            )
        except LLMError as primary_err:
            if not self.fallback:
                raise
            logger.warning(f"Primary provider {self.primary.name} failed ({primary_err}). Switching to fallback {self.fallback.name}.")
            self.stats["fallback_used"] += 1
            return call_with_retry(
                lambda: self.fallback.chat(
                    messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stop=stop,
                ),
                self.retry_config,
                self.stats,
                label=self.fallback.name,
            )

    def stream(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
    ) -> Iterator[Chunk]:
        try:
            yield from stream_with_retry(
                lambda: self.primary.stream(
                    messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stop=stop,
                ),
                self.retry_config,
                self.stats,
                label=self.primary.name,
            )
        except LLMError as primary_err:
            if not self.fallback:
                raise
            logger.warning(f"Primary provider {self.primary.name} failed before streaming ({primary_err}). Switching to fallback {self.fallback.name}.")
            self.stats["fallback_used"] += 1
            yield from stream_with_retry(
                lambda:                 self.fallback.stream(
                    messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stop=stop,
                ),
                self.retry_config,
                self.stats,
                label=self.fallback.name,
            )


def build_backend(kind: str, settings: Settings) -> BaseBackend:
    kind = kind.lower().strip()
    if kind == "ollama":
        return OllamaBackend(host=settings.OLLAMA_HOST, model=settings.OLLAMA_MODEL)
    elif kind == "gemini":
        return GeminiBackend(api_key=settings.GEMINI_API_KEY, model=settings.GEMINI_MODEL)
    elif kind == "openai":
        api_key = settings.OPENAI_API_KEY or settings.GEMINI_API_KEY
        return OpenAICompatBackend(
            api_key=api_key,
            base_url=settings.OPENAI_BASE_URL,
            model=settings.OPENAI_MODEL,
        )
    raise ValueError(f"Unknown LLM provider backend: '{kind}'")


@lru_cache(maxsize=1)
def get_gateway() -> LLMGateway:
    settings = get_settings()
    primary = build_backend(settings.LLM_PROVIDER, settings)
    fallback: BaseBackend | None = None
    if settings.LLM_FALLBACK_PROVIDER and settings.LLM_FALLBACK_PROVIDER != "none" and settings.LLM_FALLBACK_PROVIDER != settings.LLM_PROVIDER:
        try:
            fallback = build_backend(settings.LLM_FALLBACK_PROVIDER, settings)
        except Exception as e:
            logger.warning(f"Could not build fallback backend {settings.LLM_FALLBACK_PROVIDER}: {e}")
    return LLMGateway(primary=primary, fallback=fallback)
