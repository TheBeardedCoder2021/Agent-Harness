"""Unified async model router with a real-time token / cost ledger.

Routing summary
---------------
* Claude (opus-5-5, sonnet-5-5, fable-5-2): always ``claude -p`` over the Claude Code
  subscription OAuth session. Actual API cost is $0.00; a *virtual* quota equivalent is tracked.
* Gemini (4-carbon, 4-argon, 3-8-flash):
    Tier 1  ``gemini -p`` (Google subscription OAuth, $0.00) with the prompt piped via stdin.
    Tier 2  Vertex AI through the google-genai SDK (metered USD, counted against the budget).
  Tier 1 -> Tier 2 failover happens on quota errors (429 / RESOURCE_EXHAUSTED / Quota exceeded),
  tracked per model with a 300 s cooldown, or when ``route="vertex"`` is requested explicitly.
  gemini-4-carbon additionally falls back to gemini-4-argon on 404 / 403 / 429.

Notes
-----
* The Gemini CLI has no thinking / temperature flags, so ``effort`` only affects the Vertex tier
  there. Temperature is never set on any Gemini call.
* Gemini 4 Carbon uses *implicit* context caching only (4,096-token minimum, discount applied
  automatically). Explicit ``caches.create`` is never called. Put static context first in the
  prompt so the implicit cache can hit; ``static_context`` is rendered ahead of ``prompt``.
* ``claude -p`` manages its own prompt-cache breakpoints and does not accept ``cache_control``;
  ``build_cache_control_blocks`` is provided for direct Anthropic Messages API callers.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import random
import re
import shutil
import time
import uuid
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal, TypeVar, overload

from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

Effort = Literal["low", "medium", "high", "xhigh", "max"]
RouteChoice = Literal["auto", "subscription", "vertex"]
BillingRoute = Literal["subscription_oauth", "vertex_ai"]
TelemetryCallback = Callable[[dict[str, Any]], Awaitable[None]]

# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------

MODEL_CLAUDE_OPUS = "claude-opus-5-5"
MODEL_CLAUDE_SONNET = "claude-sonnet-5-5"
MODEL_CLAUDE_FABLE = "claude-fable-5-2"
MODEL_GEMINI_CARBON = "gemini-4-carbon"
MODEL_GEMINI_ARGON = "gemini-4-argon"
MODEL_GEMINI_FLASH = "gemini-3-8-flash"

FLASH_PROMPT_TOKEN_CAP = 90_000  # stay clear of the 100K 5x pricing cliff
STRUCTURED_MAX_OUTPUT_TOKENS = 16_384  # caps verbose thinking traces on JSON calls
GEMINI_SUB_COOLDOWN_S = 300.0
CHARS_PER_TOKEN = 3.5  # deliberately conservative heuristic (code/diffs tokenize densely)
VALID_EFFORTS: frozenset[str] = frozenset({"low", "medium", "high", "xhigh", "max"})

GEMINI_CLI_INSTRUCTION = "Follow the piped stdin instructions and return valid JSON."
GEMINI_CLI_INSTRUCTION_TEXT = "Follow the piped stdin instructions."

# Deterministic effort mappings. MINIMAL is intentionally unreachable.
CLAUDE_EFFORT_FLAG: dict[str, str] = {e: e for e in ("low", "medium", "high", "xhigh", "max")}
GEMINI_THINKING_LEVEL: dict[str, str] = {
    "low": "LOW",
    "medium": "MEDIUM",
    "high": "HIGH",
    "xhigh": "HIGH",
    "max": "HIGH",
}


@dataclass(frozen=True, slots=True)
class Pricing:
    """USD per million tokens."""

    input: float
    cached: float
    output: float


# Virtual quota-equivalent rates (Claude subscription; actual API cost is $0).
CLAUDE_VIRTUAL_PRICING: dict[str, Pricing] = {
    MODEL_CLAUDE_OPUS: Pricing(4.00, 0.20, 20.00),
    MODEL_CLAUDE_SONNET: Pricing(2.00, 0.10, 10.00),
    MODEL_CLAUDE_FABLE: Pricing(8.00, 0.20, 40.00),
}

# Vertex AI metered rates. Also used as the virtual-equivalent for Gemini subscription calls.
VERTEX_PRICING: dict[str, Pricing] = {
    MODEL_GEMINI_CARBON: Pricing(2.00, 0.10, 10.00),
    MODEL_GEMINI_ARGON: Pricing(2.50, 0.25, 12.00),
    MODEL_GEMINI_FLASH: Pricing(0.75, 0.075, 3.75),
}

GEMINI_FALLBACK_CHAIN: dict[str, tuple[str, ...]] = {
    MODEL_GEMINI_CARBON: (MODEL_GEMINI_CARBON, MODEL_GEMINI_ARGON),
    MODEL_GEMINI_ARGON: (MODEL_GEMINI_ARGON,),
    MODEL_GEMINI_FLASH: (MODEL_GEMINI_FLASH,),
}

# --------------------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------------------


def _is_number(val: Any) -> bool:
    try:
        float(val)
        return True
    except (ValueError, TypeError):
        return False


class ModelRouterError(Exception):
    """Base class for all router errors."""


class UnsupportedModelError(ModelRouterError, ValueError):
    pass


class ConfigurationError(ModelRouterError):
    """Missing binary, missing env var, or missing SDK. Never retried, never failed over."""


class BudgetExceededError(ModelRouterError):
    def __init__(
        self,
        spent_usd: float | str = 0.0,
        max_budget_usd: float = 0.0,
        estimated_usd: float = 0.0,
    ) -> None:
        if isinstance(spent_usd, str) and not _is_number(spent_usd):
            self.spent_usd = 0.0
            self.max_budget_usd = max_budget_usd
            self.estimated_usd = estimated_usd
            super().__init__(spent_usd)
        else:
            spent = float(spent_usd)
            max_b = float(max_budget_usd)
            est = float(estimated_usd)
            self.spent_usd = spent
            self.max_budget_usd = max_b
            self.estimated_usd = est
            super().__init__(
                f"Budget exceeded: spent ${spent:.4f} + estimated ${est:.4f} "
                f"vs max ${max_b:.4f}"
            )


class FlashTokenCliffError(ModelRouterError):
    def __init__(self, estimated_tokens: int | str = 0, limit: int = FLASH_PROMPT_TOKEN_CAP) -> None:
        if isinstance(estimated_tokens, str) and not _is_number(estimated_tokens):
            self.estimated_tokens = 0
            self.limit = limit
            super().__init__(estimated_tokens)
        else:
            tok = int(estimated_tokens)
            self.limit = limit
            self.estimated_tokens = tok
            super().__init__(
                f"{MODEL_GEMINI_FLASH} prompt is ~{tok:,} tokens; hard guard is "
                f"{limit:,} (the 100K pricing cliff is 5x). Trim the input or use another model."
            )


class StructuredOutputError(ModelRouterError):
    """Provider output could not be validated against the requested schema."""


class ProviderError(ModelRouterError):
    """Non-retryable provider failure."""


class TransientError(ModelRouterError):
    """Retryable failure (timeouts, 5xx, overloaded, connection resets)."""


class SubscriptionQuotaError(ModelRouterError):
    """Subscription (OAuth) quota exhausted or rate limited."""


class ModelUnavailableError(ModelRouterError):
    """Model returned 404 / 403 / 429 (used for the carbon -> argon fallback)."""


# --------------------------------------------------------------------------------------
# Environment & Secret Redaction helpers
# --------------------------------------------------------------------------------------

# Stripped so the CLI uses subscription OAuth instead of metered keys / other providers.
_CLAUDE_STRIP = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "CLAUDE_CODE_USE_BEDROCK",
    "CLAUDE_CODE_USE_VERTEX",
)
_GEMINI_STRIP = ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_GENAI_USE_VERTEXAI")

_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"-----BEGIN (?:[A-Z0-9]+ ){0,4}PRIVATE KEY-----[^\r\n]{0,4096}-----END (?:[A-Z0-9]+ ){0,4}PRIVATE KEY-----"
        r"|-----BEGIN (?:[A-Z0-9]+ ){0,4}PRIVATE KEY-----"
        r"(?:[ \t]*\r?\n[ \t]*[A-Za-z0-9+/=_-]{8,}(?:[ \t]*:[ \t]*[^\r\n]+)?){0,256}"
        r"(?:[ \t]*\r?\n[ \t]*-----END (?:[A-Z0-9]+ ){0,4}PRIVATE KEY-----)?"
    ),  # PEM / OpenSSH / PKCS#8 private key blocks and headers
    re.compile(r"AIza[0-9A-Za-z-_]{20,}"),  # Google API key
    re.compile(r"sk-ant-[a-zA-Z0-9_\-]{15,}"),  # Anthropic API key
    re.compile(r"sk-[a-zA-Z0-9_\-]{15,}"),  # OpenAI / generic key
    re.compile(r"ya29\.[0-9A-Za-z-_]+"),  # Google OAuth token
    re.compile(r"Bearer\s+[a-zA-Z0-9_\-\.]{10,}", re.IGNORECASE),  # Bearer tokens
    re.compile(r"[?&](?:key|token|api_key)=[^&\s]+", re.IGNORECASE),  # Query param keys
    re.compile(r"api[_-]?key[\"'\s:=]+[a-zA-Z0-9_\-]{8,}", re.IGNORECASE),
    re.compile(r"auth[_-]?token[\"'\s:=]+[a-zA-Z0-9_\-]{8,}", re.IGNORECASE),
)


def redact_secrets(text: Any) -> Any:
    """Scrub sensitive API keys, OAuth tokens, and bearer credentials."""
    if not isinstance(text, str):
        return text
    result = text
    for env_var in (
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "ANTHROPIC_API_KEY",
        "CLAUDE_CODE_OAUTH_TOKEN",
        "ANTHROPIC_AUTH_TOKEN",
    ):
        val = os.environ.get(env_var)
        if val and len(val) >= 6:
            result = result.replace(val, "[REDACTED]")
    for pattern in _SECRET_PATTERNS:
        result = pattern.sub("[REDACTED]", result)
    return result


def sanitize_telemetry_payload(obj: Any) -> Any:
    """Recursively redact secrets from telemetry and logging payloads."""
    if isinstance(obj, str):
        return redact_secrets(obj)
    if isinstance(obj, dict):
        return {k: sanitize_telemetry_payload(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_telemetry_payload(v) for v in obj]
    return obj


def _env_without(keys: Sequence[str]) -> dict[str, str]:
    # Copy everything else so HOME/USER/PATH (and USERPROFILE/APPDATA/etc. on Windows) and
    # CLAUDE_CODE_OAUTH_TOKEN survive.
    return {k: v for k, v in os.environ.items() if k not in keys}


def clean_claude_env() -> dict[str, str]:
    return _env_without(_CLAUDE_STRIP)


def clean_gemini_oauth_env() -> dict[str, str]:
    return _env_without(_GEMINI_STRIP)


def __getattr__(name: str) -> dict[str, str]:  # live view so env changes are never stale
    if name == "CLEAN_CLAUDE_ENV":
        return clean_claude_env()
    if name == "CLEAN_GEMINI_OAUTH_ENV":
        return clean_gemini_oauth_env()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# --------------------------------------------------------------------------------------
# Pure helpers
# --------------------------------------------------------------------------------------


def estimate_tokens(text: str) -> int:
    return math.ceil(len(text) / CHARS_PER_TOKEN)


def compute_cost(price: Pricing, input_tokens: int, cached_tokens: int, output_tokens: int) -> float:
    """``input_tokens`` must exclude cached tokens."""
    return (
        input_tokens * price.input + cached_tokens * price.cached + output_tokens * price.output
    ) / 1_000_000


_FENCE_RE = re.compile(r"```[a-zA-Z0-9_-]*[ \t]*\r?\n(.*?)\r?\n?```", re.DOTALL)


def strip_code_fences(text: str) -> str:
    """Return the body of a markdown code fence if present, else the stripped text."""
    stripped = text.strip()
    m = _FENCE_RE.search(stripped)
    return m.group(1).strip() if m else stripped


def extract_json_object(text: str) -> dict[str, Any] | None:
    """Find the CLI JSON envelope in ``text``, tolerating banner lines or logs."""
    decoder = json.JSONDecoder()
    candidates: list[dict[str, Any]] = []
    idx = text.find("{")
    while idx != -1:
        try:
            obj, _ = decoder.raw_decode(text, idx)
        except json.JSONDecodeError:
            idx = text.find("{", idx + 1)
            continue
        if isinstance(obj, dict):
            # Prioritize objects containing known CLI envelope keys
            if any(k in obj for k in ("response", "result", "structured_output", "error", "candidates")):
                return obj
            candidates.append(obj)
        idx = text.find("{", idx + 1)
    return candidates[0] if candidates else None


def build_cache_control_blocks(
    static_context: Sequence[str], dynamic_text: str
) -> list[dict[str, Any]]:
    """Anthropic Messages API content blocks with ephemeral cache_control on static context.

    The API allows at most 4 breakpoints and a breakpoint caches the whole prefix before it,
    so only the last four static blocks are marked.
    """
    marked_from = max(0, len(static_context) - 4)
    blocks: list[dict[str, Any]] = []
    for i, text in enumerate(static_context):
        block: dict[str, Any] = {"type": "text", "text": text}
        if i >= marked_from:
            block["cache_control"] = {"type": "ephemeral"}
        blocks.append(block)
    blocks.append({"type": "text", "text": dynamic_text})
    return blocks


_USAGE_LIMIT_RE = re.compile(r"usage limit|limit reached|hit your limit|out of (extra )?usage", re.I)
_QUOTA_RE = re.compile(
    r"\b429\b|RESOURCE_EXHAUSTED|Quota exceeded|rate.?limit|too many requests", re.I
)
_UNAVAILABLE_RE = re.compile(
    r"\b404\b|\b403\b|NOT_FOUND|PERMISSION_DENIED|model.{0,60}not (found|available)", re.I
)
_TRANSIENT_RE = re.compile(
    r"\b(500|502|503|504|529)\b|UNAVAILABLE|overloaded|ECONNRESET|ETIMEDOUT|ENOTFOUND|"
    r"socket hang up|timed out|DEADLINE_EXCEEDED|\bINTERNAL\b",
    re.I,
)


def _clip(text: str, n: int = 1500) -> str:
    return text if len(text) <= n else text[:n] + "..."


def _classify_claude_failure(text: str) -> ModelRouterError:
    clean_text = redact_secrets(text)
    if _USAGE_LIMIT_RE.search(clean_text):
        return SubscriptionQuotaError(f"Claude subscription limit: {_clip(clean_text)}")
    if _QUOTA_RE.search(clean_text) or _TRANSIENT_RE.search(clean_text):
        return TransientError(f"Claude CLI transient failure: {_clip(clean_text)}")
    return ProviderError(f"Claude CLI failed: {_clip(clean_text)}")


def _classify_gemini_cli_failure(model: str, text: str) -> ModelRouterError:
    clean_text = redact_secrets(text)
    if _QUOTA_RE.search(clean_text) or _USAGE_LIMIT_RE.search(clean_text):
        return SubscriptionQuotaError(f"Gemini subscription quota ({model}): {_clip(clean_text)}")
    if model == MODEL_GEMINI_CARBON and _UNAVAILABLE_RE.search(clean_text):
        return ModelUnavailableError(f"{model} unavailable via CLI: {_clip(clean_text)}")
    if _TRANSIENT_RE.search(clean_text):
        return TransientError(f"Gemini CLI transient failure ({model}): {_clip(clean_text)}")
    return ProviderError(f"Gemini CLI failed ({model}): {_clip(clean_text)}")


# --------------------------------------------------------------------------------------
# Data classes
# --------------------------------------------------------------------------------------


class _AwaitableDict(dict[str, Any]):
    """Dict subclass that can be awaited directly for async / sync compatibility."""

    def __await__(self):
        async def _ret() -> dict[str, Any]:
            return self

        return _ret().__await__()


@dataclass(slots=True)
class UsageRecord:
    call_id: str
    ts: float
    requested_model: str
    model: str  # model that actually served the call
    billing_route: BillingRoute
    effort: str
    input_tokens: int  # non-cached input
    cached_tokens: int
    output_tokens: int  # includes thinking tokens where reported
    actual_cost_usd: float
    virtual_cost_usd: float
    latency_s: float
    tokens_estimated: bool = False
    failover_reason: str | None = None


@dataclass(slots=True)
class RouterResult[R]:
    output: R
    record: UsageRecord


@dataclass(slots=True)
class RouterRequest:
    model: str
    prompt: str
    response_schema: type[BaseModel] | None = None
    effort: Effort = "medium"
    static_context: Sequence[str] = ()
    route: RouteChoice = "auto"
    max_output_tokens: int | None = None


@dataclass(slots=True)
class _Ctx:
    call_id: str
    requested_model: str
    effort: str
    route: RouteChoice
    schema: type[BaseModel] | None
    prompt: str  # rendered: static context first, then the dynamic prompt
    max_output_tokens: int | None
    timeout_s: float
    reserved_usd: float = 0.0
    failover_reason: str | None = None


@dataclass(slots=True)
class _Outcome:
    value: Any
    input_tokens: int
    cached_tokens: int
    output_tokens: int
    model: str
    route: BillingRoute
    estimated: bool = False


# --------------------------------------------------------------------------------------
# Router
# --------------------------------------------------------------------------------------


class ModelRouter:
    def __init__(
        self,
        run_id: str = "default",
        max_budget_usd: float = 4.50,
        on_telemetry_update: TelemetryCallback | None = None,
        *,
        gemini_use_subscription_first: bool = True,
        claude_bin: str = "claude",
        gemini_bin: str = "gemini",
        claude_bare: bool = True,
        max_retries: int = 3,
        backoff_base_s: float = 1.0,
        backoff_max_s: float = 30.0,
        timeout_s: float = 300.0,
        vertex_client_factory: Callable[[str], Any] | None = None,
    ) -> None:
        self.run_id = run_id
        self.max_budget_usd = max_budget_usd
        self.on_telemetry_update = on_telemetry_update
        self.gemini_use_subscription_first = gemini_use_subscription_first
        self.claude_bin = claude_bin
        self.gemini_bin = gemini_bin
        self.claude_bare = claude_bare
        self.max_retries = max_retries
        self.backoff_base_s = backoff_base_s
        self.backoff_max_s = backoff_max_s
        self.timeout_s = timeout_s
        self._vertex_client_factory = vertex_client_factory or self._default_vertex_client

        self._lock = asyncio.Lock()
        self._spent_usd = 0.0
        self._reserved_usd = 0.0
        self._virtual_usd = 0.0
        self._records: list[UsageRecord] = []
        self._by_model: dict[str, dict[str, float]] = {}
        self._vertex_clients: dict[str, Any] = {}
        self._gemini_sub_exhausted_until: dict[str, float] = {}

    # ---- public API --------------------------------------------------------------------

    @property
    def spent_usd(self) -> float:
        return self._spent_usd

    @overload
    async def call(
        self,
        model: str,
        prompt: str,
        *,
        response_schema: type[T],
        effort: Effort = ...,
        static_context: Sequence[str] = ...,
        route: RouteChoice = ...,
        max_output_tokens: int | None = ...,
        timeout_s: float | None = ...,
    ) -> T: ...

    @overload
    async def call(
        self,
        model: str,
        prompt: str,
        *,
        response_schema: None = ...,
        effort: Effort = ...,
        static_context: Sequence[str] = ...,
        route: RouteChoice = ...,
        max_output_tokens: int | None = ...,
        timeout_s: float | None = ...,
    ) -> str: ...

    async def call(self, model: str, prompt: str, **kwargs: Any) -> Any:
        """Run one model call; returns the validated Pydantic instance (or str if no schema)."""
        return (await self.call_detailed(model, prompt, **kwargs)).output

    async def generate(
        self,
        prompt_or_model: str | None = None,
        prompt: str | None = None,
        *,
        model: str | None = None,
        response_schema: type[Any] | None = None,
        effort: Effort = "medium",
        static_context: Sequence[str] = (),
        route: RouteChoice = "auto",
        max_output_tokens: int | None = None,
        timeout_s: float | None = None,
        system_instruction: str | None = None,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Flexible downstream interface matching agent-harness execution semantics."""
        all_models = set(CLAUDE_VIRTUAL_PRICING.keys()) | set(VERTEX_PRICING.keys())

        if prompt is not None:
            resolved_model = model or (prompt_or_model if prompt_or_model in all_models else MODEL_GEMINI_CARBON)
            resolved_prompt = prompt
        elif prompt_or_model is not None:
            if prompt_or_model in all_models and model is None:
                resolved_model = prompt_or_model
                resolved_prompt = kwargs.get("prompt", "")
            else:
                resolved_model = model or MODEL_GEMINI_CARBON
                resolved_prompt = prompt_or_model
        else:
            resolved_model = model or MODEL_GEMINI_CARBON
            resolved_prompt = kwargs.get("prompt", "")

        merged_static = list(static_context)
        if system_instruction:
            merged_static.insert(0, system_instruction)
        elif system_prompt:
            merged_static.insert(0, system_prompt)

        return await self.call(
            resolved_model,
            resolved_prompt,
            response_schema=response_schema,
            effort=effort,
            static_context=merged_static,
            route=route,
            max_output_tokens=max_output_tokens,
            timeout_s=timeout_s,
        )

    async def call_detailed(
        self,
        model: str,
        prompt: str,
        *,
        response_schema: type[BaseModel] | None = None,
        effort: Effort = "medium",
        static_context: Sequence[str] = (),
        route: RouteChoice = "auto",
        max_output_tokens: int | None = None,
        timeout_s: float | None = None,
    ) -> RouterResult[Any]:
        if model not in CLAUDE_VIRTUAL_PRICING and model not in VERTEX_PRICING:
            raise UnsupportedModelError(f"Unsupported model: {model!r}")
        if effort not in VALID_EFFORTS:
            raise ValueError(f"Invalid effort {effort!r}; expected one of {sorted(VALID_EFFORTS)}")
        if route not in ("auto", "subscription", "vertex"):
            raise ValueError(f"Invalid route {route!r}")
        if model in CLAUDE_VIRTUAL_PRICING and route == "vertex":
            raise ValueError("Claude models have no Vertex route")

        ctx = _Ctx(
            call_id=uuid.uuid4().hex[:12],
            requested_model=model,
            effort=effort,
            route=route,
            schema=response_schema,
            prompt=self._render_prompt(static_context, prompt),
            max_output_tokens=max_output_tokens,
            timeout_s=timeout_s or self.timeout_s,
        )

        # Pre-flight guards: nothing is spawned or billed if either trips.
        if model == MODEL_GEMINI_FLASH:
            est = estimate_tokens(
                ctx.prompt
                + self._schema_json(response_schema)
                + (GEMINI_CLI_INSTRUCTION if response_schema is not None else "")
            )
            if est > FLASH_PROMPT_TOKEN_CAP:
                raise FlashTokenCliffError(est)

        # Concurrency & Budget Gate: check and reserve worst-case spend under self._lock
        # before emitting telemetry or starting any network / subprocess operations.
        if self._expects_vertex(model, route):
            est_usd = self._estimate_vertex_cost(
                model,
                ctx.prompt + self._schema_json(response_schema),
                response_schema,
                max_output_tokens,
            )
            await self._set_reservation(ctx, est_usd)
        else:
            await self._check_budget()

        await self._emit(
            "call_started",
            call_id=ctx.call_id,
            model=model,
            effort=effort,
            route_preference=route,
        )
        started = time.monotonic()
        try:
            if model in CLAUDE_VIRTUAL_PRICING:
                outcome = await self._retry(ctx, "claude", lambda: self._claude_attempt(ctx))
            else:
                outcome = await self._gemini_dispatch(ctx)
        except BaseException as exc:
            await self._release(ctx)
            if isinstance(exc, Exception):
                clean_err = redact_secrets(str(exc))
                await self._emit(
                    "call_failed",
                    call_id=ctx.call_id,
                    model=model,
                    error_type=type(exc).__name__,
                    error=_clip(clean_err, 500),
                )
            raise

        record = self._make_record(ctx, outcome, time.monotonic() - started)
        await self._commit(record, ctx)
        await self._emit("call_completed", **asdict(record))
        return RouterResult(output=outcome.value, record=record)

    async def preflight(self, requests: Sequence[RouterRequest]) -> float:
        """Estimate worst-case metered spend for a batch and raise if it cannot fit."""
        total = 0.0
        for req in requests:
            if req.model == MODEL_GEMINI_FLASH:
                rendered = self._render_prompt(req.static_context, req.prompt)
                est = estimate_tokens(
                    rendered
                    + self._schema_json(req.response_schema)
                    + (GEMINI_CLI_INSTRUCTION if req.response_schema is not None else "")
                )
                if est > FLASH_PROMPT_TOKEN_CAP:
                    raise FlashTokenCliffError(est)
            if req.model in CLAUDE_VIRTUAL_PRICING:
                continue
            if req.model not in VERTEX_PRICING:
                raise UnsupportedModelError(f"Unsupported model: {req.model!r}")
            if self._expects_vertex(req.model, req.route):
                rendered = self._render_prompt(req.static_context, req.prompt)
                total += self._estimate_vertex_cost(
                    req.model,
                    rendered + self._schema_json(req.response_schema),
                    req.response_schema,
                    req.max_output_tokens,
                )
        async with self._lock:
            if self._spent_usd >= self.max_budget_usd or (
                self._spent_usd + self._reserved_usd + total > self.max_budget_usd
            ):
                raise BudgetExceededError(
                    self._spent_usd + self._reserved_usd, self.max_budget_usd, total
                )
        return total

    async def call_many(
        self, requests: Sequence[RouterRequest], *, return_exceptions: bool = False
    ) -> list[RouterResult[Any] | BaseException]:
        """Budget-checked parallel dispatch."""
        await self.preflight(requests)
        coros = [
            self.call_detailed(
                r.model,
                r.prompt,
                response_schema=r.response_schema,
                effort=r.effort,
                static_context=r.static_context,
                route=r.route,
                max_output_tokens=r.max_output_tokens,
            )
            for r in requests
        ]
        return list(await asyncio.gather(*coros, return_exceptions=return_exceptions))

    def get_ledger_summary(self) -> dict[str, Any]:
        """Synchronous or awaitable ledger summary for downstream monitoring."""
        by_route: dict[str, dict[str, float]] = {}
        for r in self._records:
            agg = by_route.setdefault(r.billing_route, {"calls": 0, "actual_usd": 0.0})
            agg["calls"] += 1
            agg["actual_usd"] += r.actual_cost_usd
        data = {
            "run_id": self.run_id,
            "max_budget_usd": self.max_budget_usd,
            "spent_usd": self._spent_usd,
            "reserved_usd": self._reserved_usd,
            "budget_remaining_usd": max(0.0, self.max_budget_usd - self._spent_usd),
            "virtual_quota_usd": self._virtual_usd,
            "calls": len(self._records),
            "by_model": {k: dict(v) for k, v in self._by_model.items()},
            "by_route": by_route,
            "records": [asdict(r) for r in self._records],
        }
        return _AwaitableDict(data)

    async def ledger_snapshot(self) -> dict[str, Any]:
        """Thread-safe async ledger snapshot."""
        async with self._lock:
            return self.get_ledger_summary()

    # ---- prompt rendering ----------------------------------------------------------------

    @staticmethod
    def _render_prompt(static_context: Sequence[str], prompt: str) -> str:
        # Static, reusable context goes first so prefix / implicit caching can hit.
        return "\n\n".join([*static_context, prompt])

    @staticmethod
    def _schema_json(schema: type[BaseModel] | None) -> str:
        return json.dumps(schema.model_json_schema()) if schema is not None else ""

    # ---- ledger / budget -------------------------------------------------------------------

    async def _check_budget(self) -> None:
        async with self._lock:
            if self._spent_usd >= self.max_budget_usd:
                raise BudgetExceededError(self._spent_usd, self.max_budget_usd)

    async def _set_reservation(self, ctx: _Ctx, estimate_usd: float) -> None:
        """Reserve worst-case metered spend for this call under self._lock."""
        async with self._lock:
            others = self._reserved_usd - ctx.reserved_usd
            if self._spent_usd >= self.max_budget_usd or (
                self._spent_usd + others + estimate_usd > self.max_budget_usd
            ):
                raise BudgetExceededError(
                    self._spent_usd + others, self.max_budget_usd, estimate_usd
                )
            self._reserved_usd = others + estimate_usd
            ctx.reserved_usd = estimate_usd

    async def _release(self, ctx: _Ctx) -> None:
        async with self._lock:
            self._reserved_usd = max(0.0, self._reserved_usd - ctx.reserved_usd)
            ctx.reserved_usd = 0.0

    async def _commit(self, rec: UsageRecord, ctx: _Ctx) -> None:
        async with self._lock:  # reservation release + spend are one atomic step
            self._reserved_usd = max(0.0, self._reserved_usd - ctx.reserved_usd)
            ctx.reserved_usd = 0.0
            self._spent_usd += rec.actual_cost_usd
            self._virtual_usd += rec.virtual_cost_usd
            self._records.append(rec)
            agg = self._by_model.setdefault(
                rec.model,
                {
                    "calls": 0,
                    "input_tokens": 0,
                    "cached_tokens": 0,
                    "output_tokens": 0,
                    "actual_usd": 0.0,
                    "virtual_usd": 0.0,
                },
            )
            agg["calls"] += 1
            agg["input_tokens"] += rec.input_tokens
            agg["cached_tokens"] += rec.cached_tokens
            agg["output_tokens"] += rec.output_tokens
            agg["actual_usd"] += rec.actual_cost_usd
            agg["virtual_usd"] += rec.virtual_cost_usd

    def _make_record(self, ctx: _Ctx, o: _Outcome, latency_s: float) -> UsageRecord:
        if o.model in CLAUDE_VIRTUAL_PRICING:
            actual, virtual = 0.0, compute_cost(
                CLAUDE_VIRTUAL_PRICING[o.model], o.input_tokens, o.cached_tokens, o.output_tokens
            )
        else:
            cost = compute_cost(
                VERTEX_PRICING[o.model], o.input_tokens, o.cached_tokens, o.output_tokens
            )
            if o.route == "vertex_ai":
                actual, virtual = cost, 0.0
            else:  # subscription: free, but track the equivalent quota consumption
                actual, virtual = 0.0, cost
        return UsageRecord(
            call_id=ctx.call_id,
            ts=time.time(),
            requested_model=ctx.requested_model,
            model=o.model,
            billing_route=o.route,
            effort=ctx.effort,
            input_tokens=o.input_tokens,
            cached_tokens=o.cached_tokens,
            output_tokens=o.output_tokens,
            actual_cost_usd=actual,
            virtual_cost_usd=virtual,
            latency_s=round(latency_s, 3),
            tokens_estimated=o.estimated,
            failover_reason=ctx.failover_reason,
        )

    def _estimate_vertex_cost(
        self, model: str, text: str, schema: type[BaseModel] | None, max_out: int | None
    ) -> float:
        out_cap = self._effective_max_output(schema, max_out) or STRUCTURED_MAX_OUTPUT_TOKENS
        return compute_cost(VERTEX_PRICING[model], estimate_tokens(text), 0, out_cap)

    @staticmethod
    def _effective_max_output(schema: type[BaseModel] | None, requested: int | None) -> int | None:
        if schema is not None:
            return min(requested or STRUCTURED_MAX_OUTPUT_TOKENS, STRUCTURED_MAX_OUTPUT_TOKENS)
        return requested

    # ---- telemetry -------------------------------------------------------------------------

    async def _emit(self, event: str, **fields: Any) -> None:
        if self.on_telemetry_update is None:
            return
        raw_payload = {
            "type": event,
            "run_id": self.run_id,
            "ts": time.time(),
            "cumulative_spent_usd": self._spent_usd,
            "cumulative_virtual_usd": self._virtual_usd,
            "max_budget_usd": self.max_budget_usd,
            "budget_remaining_usd": max(0.0, self.max_budget_usd - self._spent_usd),
            **fields,
        }
        payload = sanitize_telemetry_payload(raw_payload)
        try:
            await self.on_telemetry_update(payload)
        except Exception:  # a broken UI stream must never fail a model call
            logger.exception("telemetry callback failed for event %s", event)

    # ---- retry -----------------------------------------------------------------------------

    async def _sleep(self, seconds: float) -> None:
        await asyncio.sleep(seconds)

    async def _retry(self, ctx: _Ctx, label: str, fn: Callable[[], Awaitable[_Outcome]]) -> _Outcome:
        attempt = 0
        while True:
            try:
                return await fn()
            except (TransientError, StructuredOutputError) as exc:
                if attempt >= self.max_retries:
                    raise
                delay = min(self.backoff_max_s, self.backoff_base_s * 2**attempt)
                delay *= 0.5 + random.random() / 2  # jitter
                clean_err = redact_secrets(str(exc))
                await self._emit(
                    "call_retry",
                    call_id=ctx.call_id,
                    tier=label,
                    attempt=attempt + 1,
                    delay_s=round(delay, 2),
                    error=_clip(clean_err, 300),
                )
                await self._sleep(delay)
                attempt += 1

    # ---- subprocess ------------------------------------------------------------------------

    async def _run_subprocess(
        self, argv: list[str], stdin_text: str, env: dict[str, str], timeout_s: float
    ) -> tuple[int, str, str]:
        exe = shutil.which(argv[0]) or argv[0]
        try:
            proc = await asyncio.create_subprocess_exec(
                exe,
                *argv[1:],
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
        except FileNotFoundError as exc:
            raise ConfigurationError(f"CLI not found on PATH: {argv[0]!r}") from None
        try:
            async with asyncio.timeout(timeout_s):
                out, err = await proc.communicate(stdin_text.encode("utf-8"))
        except TimeoutError as exc:
            await self._kill(proc)
            raise TransientError(f"{argv[0]} timed out after {timeout_s:.0f}s") from None
        except BaseException:  # includes CancelledError: never leak child processes
            await self._kill(proc)
            raise
        return (
            proc.returncode if proc.returncode is not None else -1,
            out.decode("utf-8", errors="replace"),
            err.decode("utf-8", errors="replace"),
        )

    @staticmethod
    async def _kill(proc: asyncio.subprocess.Process) -> None:
        try:
            proc.kill()
        except ProcessLookupError:
            pass
        await proc.wait()

    # ---- validation ------------------------------------------------------------------------

    @staticmethod
    def _validate(schema: type[T], payload: Any, origin: str) -> T:
        try:
            if isinstance(payload, (str, bytes)):
                return schema.model_validate_json(payload)
            return schema.model_validate(payload)
        except (ValidationError, ValueError, TypeError) as exc:
            clean_err = redact_secrets(str(exc))
            raise StructuredOutputError(
                f"{origin} output failed {schema.__name__} validation: {_clip(clean_err, 800)}"
            ) from None

    # ---- Claude ----------------------------------------------------------------------------

    async def _claude_attempt(self, ctx: _Ctx) -> _Outcome:
        model = ctx.requested_model
        argv = [self.claude_bin, "-p"]
        if self.claude_bare:
            argv.append("--bare")
        argv += [
            "--model",
            model,
            "--effort",
            CLAUDE_EFFORT_FLAG[ctx.effort],
            "--tools",
            "",
            "--max-turns",
            "1",
            "--no-session-persistence",
            "--output-format",
            "json",
        ]
        if ctx.schema is not None:
            argv += ["--json-schema", self._schema_json(ctx.schema)]

        rc, stdout, stderr = await self._run_subprocess(
            argv, ctx.prompt, clean_claude_env(), ctx.timeout_s
        )

        outer = extract_json_object(stdout)
        if outer is None:
            raise _classify_claude_failure(f"exit={rc}: {stderr.strip() or stdout.strip()}")
        if outer.get("subtype") == "error_max_structured_output_retries":
            raise StructuredOutputError(
                f"{model} could not produce schema-valid output "
                "(error_max_structured_output_retries)"
            )
        if rc != 0 or outer.get("is_error"):
            detail = outer.get("result") or stderr.strip() or stdout.strip()
            raise _classify_claude_failure(f"exit={rc} subtype={outer.get('subtype')}: {detail}")

        if ctx.schema is not None:
            payload = outer.get("structured_output")
            if payload is None:  # tolerate CLI versions that return JSON in "result"
                payload = strip_code_fences(str(outer.get("result", "")))
            value: Any = self._validate(ctx.schema, payload, model)
        else:
            value = str(outer.get("result", ""))

        usage = outer.get("usage") or {}
        # Cache-write tokens are billed at the input rate; the quota table has no write premium.
        return _Outcome(
            value=value,
            input_tokens=int(usage.get("input_tokens") or 0)
            + int(usage.get("cache_creation_input_tokens") or 0),
            cached_tokens=int(usage.get("cache_read_input_tokens") or 0),
            output_tokens=int(usage.get("output_tokens") or 0),
            model=model,
            route="subscription_oauth",
        )

    # ---- Gemini ----------------------------------------------------------------------------

    def _sub_exhausted(self, model: str) -> bool:
        return time.monotonic() < self._gemini_sub_exhausted_until.get(model, 0.0)

    def _expects_vertex(self, model: str, route: RouteChoice) -> bool:
        if route == "vertex":
            return True
        if route == "subscription":
            return False
        return not self.gemini_use_subscription_first or self._sub_exhausted(model)

    async def _gemini_dispatch(self, ctx: _Ctx) -> _Outcome:
        chain = GEMINI_FALLBACK_CHAIN[ctx.requested_model]
        for i, model in enumerate(chain):
            try:
                return await self._gemini_single(ctx, model)
            except ModelUnavailableError as exc:
                if i + 1 >= len(chain):
                    raise
                ctx.failover_reason = f"{model}_unavailable"
                clean_err = redact_secrets(str(exc))
                await self._emit(
                    "route_failover",
                    call_id=ctx.call_id,
                    from_model=model,
                    to_model=chain[i + 1],
                    reason=_clip(clean_err, 300),
                )
        raise AssertionError("unreachable")  # pragma: no cover

    async def _gemini_single(self, ctx: _Ctx, model: str) -> _Outcome:
        use_sub = (
            ctx.route != "vertex"
            and self.gemini_use_subscription_first
            and not self._sub_exhausted(model)
        )
        if ctx.route == "subscription" and not use_sub:
            raise SubscriptionQuotaError(
                f"Subscription tier unavailable for {model} (disabled or in cooldown) and "
                "route='subscription' forbids Vertex failover"
            )
        if use_sub:
            try:
                return await self._retry(
                    ctx, "subscription_oauth", lambda: self._gemini_cli_attempt(ctx, model)
                )
            except SubscriptionQuotaError as exc:
                self._gemini_sub_exhausted_until[model] = time.monotonic() + GEMINI_SUB_COOLDOWN_S
                if ctx.route == "subscription":
                    raise
                ctx.failover_reason = f"{model}_subscription_quota"
                clean_err = redact_secrets(str(exc))
                await self._emit(
                    "route_failover",
                    call_id=ctx.call_id,
                    from_model=model,
                    to_model=model,
                    from_route="subscription_oauth",
                    to_route="vertex_ai",
                    cooldown_s=GEMINI_SUB_COOLDOWN_S,
                    reason=_clip(clean_err, 300),
                )
        return await self._retry(ctx, "vertex_ai", lambda: self._vertex_attempt(ctx, model))

    async def _gemini_cli_attempt(self, ctx: _Ctx, model: str) -> _Outcome:
        payload = ctx.prompt
        if ctx.schema is not None:
            payload += (
                "\n\nRespond with ONLY a JSON object (no prose, no code fences) that conforms "
                f"to this JSON Schema:\n{self._schema_json(ctx.schema)}"
            )
        argv = [
            self.gemini_bin,
            "-p",
            GEMINI_CLI_INSTRUCTION if ctx.schema is not None else GEMINI_CLI_INSTRUCTION_TEXT,
            "-m",
            model,
            "--approval-mode",
            "plan",
            "--output-format",
            "json",
        ]
        # Prompt goes over stdin: avoids ARG_MAX limits and shell/tool-invocation surprises.
        rc, stdout, stderr = await self._run_subprocess(
            argv, payload, clean_gemini_oauth_env(), ctx.timeout_s
        )

        outer = extract_json_object(stdout)
        if outer is None:
            if rc != 0:
                raise _classify_gemini_cli_failure(
                    model, redact_secrets(f"exit={rc}: {stderr.strip() or stdout.strip()}")
                )
            raise StructuredOutputError(
                f"Gemini CLI returned non-JSON output: {_clip(redact_secrets(stdout))}"
            )

        err_obj = outer.get("error")
        if rc != 0 or err_obj:
            detail = (
                f"exit={rc} {stderr.strip()} {json.dumps(err_obj) if err_obj else ''} {stdout.strip()}"
            )
            raise _classify_gemini_cli_failure(model, redact_secrets(detail))

        response = outer.get("response")
        if response is None:
            response = outer.get("result", "")

        if ctx.schema is not None:
            if isinstance(response, dict):
                value: Any = self._validate(ctx.schema, response, model)
            else:
                value = self._validate(ctx.schema, strip_code_fences(str(response or "")), model)
        else:
            value = str(response or "")

        in_tok, cached, out_tok = self._gemini_cli_tokens(outer.get("stats"))
        estimated = in_tok + cached + out_tok == 0
        if estimated:  # CLI gave no stats: fall back to the heuristic so the ledger is not blind
            in_tok, out_tok = estimate_tokens(payload), estimate_tokens(str(response or ""))
        return _Outcome(
            value=value,
            input_tokens=in_tok,
            cached_tokens=cached,
            output_tokens=out_tok,
            model=model,
            route="subscription_oauth",
            estimated=estimated,
        )

    @staticmethod
    def _gemini_cli_tokens(stats: Any) -> tuple[int, int, int]:
        """Return (non-cached input, cached, output incl. thoughts) summed over all models."""
        prompt = cached = out = 0
        models = stats.get("models") if isinstance(stats, dict) else None
        for entry in (models or {}).values():
            tokens = entry.get("tokens", {}) if isinstance(entry, dict) else {}
            prompt += int(tokens.get("prompt") or 0)
            cached += int(tokens.get("cached") or 0)
            out += int(tokens.get("candidates") or 0) + int(tokens.get("thoughts") or 0)
        return max(0, prompt - cached), cached, out

    # ---- Vertex ----------------------------------------------------------------------------

    @staticmethod
    def _default_vertex_client(location: str = "global") -> Any:
        try:
            from google import genai
        except ImportError as exc:
            raise ConfigurationError(
                "google-genai is not installed (pip install 'google-genai>=1.0')"
            ) from None
        project = os.environ.get("GOOGLE_CLOUD_PROJECT")
        if not project:
            raise ConfigurationError("GOOGLE_CLOUD_PROJECT must be set for Vertex AI failover")
        return genai.Client(vertexai=True, project=project, location=location)

    def _vertex_client(self, model: str) -> Any:
        # Vertex AI routing must use the global endpoint across models.
        location = "global"
        if location not in self._vertex_clients:
            self._vertex_clients[location] = self._vertex_client_factory(location)
        return self._vertex_clients[location]

    def _vertex_config(self, ctx: _Ctx) -> Any:
        try:
            from google.genai import types
        except ImportError:
            if self._vertex_client_factory is not self._default_vertex_client:
                class _MockConfig:
                    def __init__(self, **kw: Any) -> None:
                        self.__dict__.update(kw)

                types = type(
                    "types",
                    (),
                    {
                        "ThinkingConfig": _MockConfig,
                        "GenerateContentConfig": _MockConfig,
                    },
                )
            else:
                raise ConfigurationError("google-genai is not installed") from None
        kwargs: dict[str, Any] = {
            # Single-field ThinkingConfig; never MINIMAL; temperature is deliberately left unset.
            "thinking_config": types.ThinkingConfig(
                thinking_level=GEMINI_THINKING_LEVEL[ctx.effort]
            ),
        }
        max_out = self._effective_max_output(ctx.schema, ctx.max_output_tokens) or (
            STRUCTURED_MAX_OUTPUT_TOKENS if ctx.schema is not None else None
        )
        if max_out:
            kwargs["max_output_tokens"] = max_out
        if ctx.schema is not None:
            kwargs["response_mime_type"] = "application/json"
            kwargs["response_schema"] = ctx.schema
        return types.GenerateContentConfig(**kwargs)

    async def _vertex_attempt(self, ctx: _Ctx, model: str) -> _Outcome:
        await self._set_reservation(
            ctx,
            self._estimate_vertex_cost(
                model, ctx.prompt + self._schema_json(ctx.schema), ctx.schema, ctx.max_output_tokens
            ),
        )
        client = self._vertex_client(model)
        config = self._vertex_config(ctx)
        try:
            # Implicit caching only: no caches.create on any model.
            resp = await asyncio.wait_for(
                client.aio.models.generate_content(
                    model=model, contents=ctx.prompt, config=config
                ),
                timeout=ctx.timeout_s,
            )
        except ModelRouterError:
            raise
        except Exception as exc:
            raise self._classify_vertex_error(model, exc) from None

        text = getattr(resp, "text", None)
        if not text:
            finish = None
            candidates = getattr(resp, "candidates", None)
            if candidates:
                finish = getattr(candidates[0], "finish_reason", None)
            raise StructuredOutputError(f"{model} returned no text (finish_reason={finish})")
        value: Any = (
            self._validate(ctx.schema, strip_code_fences(text), model)
            if ctx.schema is not None
            else text
        )

        usage = getattr(resp, "usage_metadata", None)
        prompt_tok = int(getattr(usage, "prompt_token_count", 0) or 0)
        cached = int(getattr(usage, "cached_content_token_count", 0) or 0)
        out_tok = int(getattr(usage, "candidates_token_count", 0) or 0) + int(
            getattr(usage, "thoughts_token_count", 0) or 0
        )  # thinking is billed as output
        estimated = usage is None
        if estimated:
            prompt_tok, out_tok = estimate_tokens(ctx.prompt), estimate_tokens(text)
        return _Outcome(
            value=value,
            input_tokens=max(0, prompt_tok - cached),
            cached_tokens=cached,
            output_tokens=out_tok,
            model=model,
            route="vertex_ai",
            estimated=estimated,
        )

    @staticmethod
    def _classify_vertex_error(model: str, exc: Exception) -> ModelRouterError:
        code = getattr(exc, "code", None)
        text = redact_secrets(f"{type(exc).__name__}: {exc}")
        is_carbon = model == MODEL_GEMINI_CARBON
        if code in (404, 403, 429) or (code is None and _QUOTA_RE.search(text)):
            if is_carbon:
                return ModelUnavailableError(f"{model} Vertex {code or 'quota'}: {_clip(text)}")
            if code == 429 or code is None:
                return TransientError(f"{model} Vertex rate limited: {_clip(text)}")
            return ProviderError(f"{model} Vertex {code}: {_clip(text)}")
        if isinstance(exc, (asyncio.TimeoutError, TimeoutError, ConnectionError, OSError)):
            return TransientError(f"{model} Vertex network error: {_clip(text)}")
        if (isinstance(code, int) and (code >= 500 or code == 408)) or (
            code is None and _TRANSIENT_RE.search(text)
        ):
            return TransientError(f"{model} Vertex transient {code}: {_clip(text)}")
        return ProviderError(f"{model} Vertex error {code}: {_clip(text)}")


__all__ = [
    "CLAUDE_VIRTUAL_PRICING",
    "FLASH_PROMPT_TOKEN_CAP",
    "GEMINI_SUB_COOLDOWN_S",
    "MODEL_CLAUDE_FABLE",
    "MODEL_CLAUDE_OPUS",
    "MODEL_CLAUDE_SONNET",
    "MODEL_GEMINI_ARGON",
    "MODEL_GEMINI_CARBON",
    "MODEL_GEMINI_FLASH",
    "VERTEX_PRICING",
    "BudgetExceededError",
    "ConfigurationError",
    "Effort",
    "FlashTokenCliffError",
    "ModelRouter",
    "ModelRouterError",
    "ModelUnavailableError",
    "Pricing",
    "ProviderError",
    "RouteChoice",
    "RouterRequest",
    "RouterResult",
    "StructuredOutputError",
    "SubscriptionQuotaError",
    "TransientError",
    "UnsupportedModelError",
    "UsageRecord",
    "build_cache_control_blocks",
    "clean_claude_env",
    "clean_gemini_oauth_env",
    "compute_cost",
    "estimate_tokens",
    "extract_json_object",
    "redact_secrets",
    "strip_code_fences",
]
