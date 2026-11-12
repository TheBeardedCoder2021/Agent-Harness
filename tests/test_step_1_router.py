"""Comprehensive Pre-Commit Audit Test Suite for ModelRouter.

Covers:
1. Concurrency & Race-Condition Gate (atomic reservations, budget enforcement, asyncio.gather)
2. Subscription OAuth & Vertex AI Failover Gate (CLEAN_ENV stripping, CLI flags, per-model 300s cooldown TTL, location="global")
3. Prompt Guard & Prompt Caching Gate (soft 90K gemini-4-hydrogen cap, preflight check, cache block structure)
4. Schema & Exception Integrity Gate (CLI envelope unwrapping, Pydantic v2 parsing, bounded retries on malformed JSON, secret scrubbing)
5. Downstream Interface Compatibility Gate (generate(...), get_ledger_summary(), BudgetExceededError, FlashTokenCliffError)
6. Virtual Cache-Read Rate Parity Gate (Claude subscription quota equivalents)
7. Gemini 4 Hydrogen Gate (flat $1.50 / $0.075 / $7.50 rates, gemini-3-8-flash alias, explicit thinking_level
   on every Vertex call, implicit-only caching, soft 90K guard with the allow_long_context override)
"""

from __future__ import annotations

import ast
import asyncio
from enum import Enum
import json
import logging
import os
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel, Field

from backend.models_router import (
    CLAUDE_VIRTUAL_PRICING,
    FLASH_PROMPT_TOKEN_CAP,
    FOREGROUND_MAX_OUTPUT_TOKENS,
    GEMINI_CONTEXT_LIMIT_TOKENS,
    GEMINI_SUB_COOLDOWN_S,
    HYDROGEN_PROMPT_TOKEN_CAP,
    MODEL_ALIASES,
    MODEL_CLAUDE_FABLE,
    MODEL_CLAUDE_OPUS,
    MODEL_CLAUDE_SONNET,
    MODEL_GEMINI_ARGON,
    MODEL_GEMINI_CARBON,
    MODEL_GEMINI_FLASH,
    MODEL_GEMINI_HYDROGEN,
    STRUCTURED_HIGH_MAX_OUTPUT_TOKENS,
    STRUCTURED_MAX_OUTPUT_TOKENS,
    VERTEX_PRICING,
    BudgetExceededError,
    ConfigurationError,
    FlashTokenCliffError,
    HydrogenPromptCapError,
    ModelRouter,
    ModelRouterError,
    Pricing,
    RouterRequest,
    StructuredOutputError,
    SubscriptionQuotaError,
    TransientError,
    UnsupportedModelError,
    build_cache_control_blocks,
    clean_claude_env,
    clean_gemini_oauth_env,
    compute_cost,
    estimate_tokens,
    extract_json_object,
    redact_secrets,
    resolve_model,
    sanitize_telemetry_payload,
    strip_code_fences,
    thinking_level_for,
)


class SampleSchema(BaseModel):
    name: str = Field(description="Entity name")
    score: int = Field(description="Integer score")


# ======================================================================================
# Gate 1: Concurrency & Race-Condition Gate
# ======================================================================================


def test_concurrent_budget_racing_prevents_overshoot() -> None:
    """Verify self._lock wraps pre-flight budget reservation & post-call mutation so

    parallel asyncio.gather calls cannot race past max_budget_usd.
    """
    async def _run():
        max_budget = 0.50
        router = ModelRouter(
            run_id="test-race",
            max_budget_usd=max_budget,
            gemini_use_subscription_first=False,  # Direct to Vertex for metered spend
            backoff_base_s=0.01,
        )

        call_count = 0
        active_concurrent = 0
        max_observed_concurrent = 0

        async def mock_generate_content(*args: Any, **kwargs: Any) -> Any:
            nonlocal call_count, active_concurrent, max_observed_concurrent
            active_concurrent += 1
            if active_concurrent > max_observed_concurrent:
                max_observed_concurrent = active_concurrent
            call_count += 1
            await asyncio.sleep(0.05)
            active_concurrent -= 1

            mock_resp = MagicMock()
            mock_resp.text = '{"name": "test", "score": 100}'
            mock_resp.usage_metadata = MagicMock()
            mock_resp.prompt_token_count = 1000
            mock_resp.usage_metadata.cached_content_token_count = 0
            mock_resp.usage_metadata.candidates_token_count = 1000
            mock_resp.usage_metadata.thoughts_token_count = 0
            return mock_resp

        mock_client = MagicMock()
        mock_client.aio.models.generate_content = mock_generate_content
        router._vertex_client_factory = lambda loc: mock_client

        tasks = [
            router.call_detailed(
                MODEL_GEMINI_CARBON,
                "short prompt",
                response_schema=SampleSchema,
                route="vertex",
            )
            for _ in range(10)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        successes = [r for r in results if not isinstance(r, BaseException)]
        failures = [r for r in results if isinstance(r, BudgetExceededError)]

        assert len(successes) > 0
        assert len(failures) > 0
        assert len(successes) + len(failures) == 10

        summary = router.get_ledger_summary()
        assert summary["spent_usd"] <= max_budget
        assert router.spent_usd <= max_budget
        assert summary["reserved_usd"] == 0.0

    asyncio.run(_run())


def test_preflight_prevents_batch_overshoot() -> None:
    """Verify preflight rejects a batch if total estimated spend exceeds max_budget."""
    async def _run():
        router = ModelRouter(run_id="test-preflight", max_budget_usd=0.20)
        requests = [
            RouterRequest(model=MODEL_GEMINI_CARBON, prompt="Prompt 1", route="vertex"),
            RouterRequest(model=MODEL_GEMINI_CARBON, prompt="Prompt 2", route="vertex"),
            RouterRequest(model=MODEL_GEMINI_CARBON, prompt="Prompt 3", route="vertex"),
        ]

        with pytest.raises(BudgetExceededError):
            await router.call_many(requests)

    asyncio.run(_run())


# ======================================================================================
# Gate 2: Subscription OAuth & Vertex AI Failover Gate
# ======================================================================================


def test_clean_env_key_scrubbing() -> None:
    """Verify clean_claude_env and clean_gemini_oauth_env strip all API keys and flags."""
    test_env = {
        "ANTHROPIC_API_KEY": "sk-ant-test-secret-12345",
        "ANTHROPIC_AUTH_TOKEN": "auth-token-test-secret",
        "CLAUDE_CODE_USE_BEDROCK": "1",
        "CLAUDE_CODE_USE_VERTEX": "1",
        "GEMINI_API_KEY": "AIzaSyTestSecretKey123456789012345",
        "GOOGLE_API_KEY": "AIzaSyTestSecretKeyGoogle123456789",
        "GOOGLE_GENAI_USE_VERTEXAI": "1",
        "PATH": "/usr/bin:/bin",
        "HOME": "/home/user",
        "USERPROFILE": "C:\\Users\\user",
    }

    with patch.dict(os.environ, test_env, clear=True):
        c_env = clean_claude_env()
        assert "ANTHROPIC_API_KEY" not in c_env
        assert "ANTHROPIC_AUTH_TOKEN" not in c_env
        assert "CLAUDE_CODE_USE_BEDROCK" not in c_env
        assert "CLAUDE_CODE_USE_VERTEX" not in c_env
        assert c_env.get("PATH") == "/usr/bin:/bin"

        g_env = clean_gemini_oauth_env()
        assert "GEMINI_API_KEY" not in g_env
        assert "GOOGLE_API_KEY" not in g_env
        assert "GOOGLE_GENAI_USE_VERTEXAI" not in g_env
        assert g_env.get("HOME") == "/home/user"


def test_claude_cli_args_and_structured_output_unwrapping() -> None:
    """Verify claude -p router calls pass --bare, strip keys, and unwrap structured_output."""
    async def _run():
        router = ModelRouter(run_id="test-claude", claude_bare=True)

        captured_argv: list[str] = []
        captured_env: dict[str, str] = {}

        async def mock_run_subp(argv: list[str], stdin: str, env: dict[str, str], timeout: float):
            nonlocal captured_argv, captured_env
            captured_argv = list(argv)
            captured_env = dict(env)
            stdout = json.dumps({
                "result": "fallback",
                "structured_output": {"name": "AuditBot", "score": 99},
                "usage": {
                    "input_tokens": 50,
                    "cache_read_input_tokens": 10,
                    "cache_creation_input_tokens": 0,
                    "output_tokens": 20,
                },
            })
            return 0, stdout, ""

        router._run_subprocess = mock_run_subp  # type: ignore[assignment]

        res = await router.call(
            MODEL_CLAUDE_SONNET,
            "Analyze security",
            response_schema=SampleSchema,
        )

        assert isinstance(res, SampleSchema)
        assert res.name == "AuditBot"
        assert res.score == 99

        assert "--bare" in captured_argv
        assert captured_argv[0] == "claude"
        assert captured_argv[1] == "-p"
        assert "--model" in captured_argv
        assert MODEL_CLAUDE_SONNET in captured_argv
        assert "ANTHROPIC_API_KEY" not in captured_env

    asyncio.run(_run())


def test_gemini_cli_args_and_envelope_unwrapping() -> None:
    """Verify Tier 1 gemini -p calls pass instruction string, --approval-mode plan,

    strip keys, pipe prompt over stdin, and unwrap outer['response'].
    """
    async def _run():
        router = ModelRouter(run_id="test-gemini-cli")

        captured_argv: list[str] = []
        captured_stdin: str = ""
        captured_env: dict[str, str] = {}

        async def mock_run_subp(argv: list[str], stdin: str, env: dict[str, str], timeout: float):
            nonlocal captured_argv, captured_stdin, captured_env
            captured_argv = list(argv)
            captured_stdin = stdin
            captured_env = dict(env)
            stdout = json.dumps({
                "response": '{"name": "CarbonUnit", "score": 10}',
                "stats": {
                    "models": {
                        "gemini-4-carbon": {
                            "tokens": {"prompt": 100, "cached": 20, "candidates": 30, "thoughts": 5}
                        }
                    }
                },
            })
            return 0, stdout, ""

        router._run_subprocess = mock_run_subp  # type: ignore[assignment]

        res = await router.call(
            MODEL_GEMINI_CARBON,
            "Run unit inspection",
            response_schema=SampleSchema,
        )

        assert isinstance(res, SampleSchema)
        assert res.name == "CarbonUnit"
        assert res.score == 10

        assert captured_argv[0] == "gemini"
        assert captured_argv[1] == "-p"
        assert "--approval-mode" in captured_argv
        plan_idx = captured_argv.index("--approval-mode") + 1
        assert captured_argv[plan_idx] == "plan"
        assert "Run unit inspection" in captured_stdin
        assert "GEMINI_API_KEY" not in captured_env
        assert "GOOGLE_API_KEY" not in captured_env
        assert "GOOGLE_GENAI_USE_VERTEXAI" not in captured_env

    asyncio.run(_run())


def test_per_model_failover_cooldown_isolation() -> None:
    """Verify per-model 300s cooldown failover (_gemini_sub_exhausted_until[model])

    so a gemini-4-carbon 429 never trips gemini-4-hydrogen onto paid Vertex AI.
    """
    async def _run():
        router = ModelRouter(run_id="test-cooldown", backoff_base_s=0.01)

        async def mock_run_subp(argv: list[str], stdin: str, env: dict[str, str], timeout: float):
            if MODEL_GEMINI_CARBON in argv:
                return 1, json.dumps({"error": {"code": 429, "message": "RESOURCE_EXHAUSTED: quota exceeded"}}), ""
            return 0, json.dumps({"response": "Hydrogen CLI success", "stats": {}}), ""

        router._run_subprocess = mock_run_subp  # type: ignore[assignment]

        mock_vertex_generate = AsyncMock()
        mock_resp = MagicMock()
        mock_resp.text = "Vertex fallback for carbon"
        mock_resp.usage_metadata = MagicMock(
            prompt_token_count=50, cached_content_token_count=0, candidates_token_count=20, thoughts_token_count=0
        )
        mock_vertex_generate.return_value = mock_resp

        mock_client = MagicMock()
        mock_client.aio.models.generate_content = mock_vertex_generate
        router._vertex_client_factory = lambda loc: mock_client

        carbon_res = await router.call(MODEL_GEMINI_CARBON, "Carbon prompt")
        assert carbon_res == "Vertex fallback for carbon"

        assert router._sub_exhausted(MODEL_GEMINI_CARBON) is True
        assert router._gemini_sub_exhausted_until[MODEL_GEMINI_CARBON] > time.monotonic() + 200

        assert router._sub_exhausted(MODEL_GEMINI_HYDROGEN) is False
        assert router._sub_exhausted(MODEL_GEMINI_FLASH) is False  # the alias shares its state

        mock_vertex_generate.reset_mock()
        hydrogen_res = await router.call(MODEL_GEMINI_HYDROGEN, "Hydrogen prompt")
        assert hydrogen_res == "Hydrogen CLI success"
        alias_res = await router.call(MODEL_GEMINI_FLASH, "Alias prompt")
        assert alias_res == "Hydrogen CLI success"
        mock_vertex_generate.assert_not_called()

    asyncio.run(_run())


def test_vertex_client_forces_global_location() -> None:
    """Verify location='global' is enforced on Vertex AI routing across models."""
    router = ModelRouter(run_id="test-vertex-loc")
    requested_locations: list[str] = []

    def mock_factory(loc: str) -> Any:
        requested_locations.append(loc)
        return MagicMock()

    router._vertex_client_factory = mock_factory

    with patch.dict(os.environ, {"GOOGLE_CLOUD_LOCATION": "us-central1"}):
        router._vertex_client(MODEL_GEMINI_CARBON)
        router._vertex_client(MODEL_GEMINI_ARGON)
        router._vertex_client(MODEL_GEMINI_HYDROGEN)
        router._vertex_client(MODEL_GEMINI_FLASH)

    assert requested_locations == ["global"]


# ======================================================================================
# Gate 3: Prompt Guard & Prompt Caching Gate
# ======================================================================================


@pytest.mark.parametrize("model", [MODEL_GEMINI_HYDROGEN, MODEL_GEMINI_FLASH])
def test_hydrogen_prompt_token_guard_trips_before_anything_is_spawned(model: str) -> None:
    """Verify gemini-4-hydrogen (and its alias) trips the 90,000-token guard before any process starts."""
    async def _run():
        router = ModelRouter(run_id="test-cliff")
        router._run_subprocess = AsyncMock()  # type: ignore[assignment]

        giant_prompt = "x" * int(HYDROGEN_PROMPT_TOKEN_CAP * 3.6)

        with pytest.raises(FlashTokenCliffError) as exc_info:
            await router.call(model, giant_prompt)

        assert exc_info.value.limit == HYDROGEN_PROMPT_TOKEN_CAP == FLASH_PROMPT_TOKEN_CAP == 90_000
        assert exc_info.value.estimated_tokens > HYDROGEN_PROMPT_TOKEN_CAP
        router._run_subprocess.assert_not_called()

    asyncio.run(_run())


@pytest.mark.parametrize("model", [MODEL_GEMINI_HYDROGEN, MODEL_GEMINI_FLASH])
def test_hydrogen_prompt_token_guard_in_preflight(model: str) -> None:
    """Verify preflight rejects any gemini-4-hydrogen request exceeding 90,000 tokens."""
    async def _run():
        router = ModelRouter(run_id="test-cliff-preflight")
        giant_prompt = "x" * int(HYDROGEN_PROMPT_TOKEN_CAP * 3.6)

        requests = [
            RouterRequest(model=MODEL_GEMINI_CARBON, prompt="Small prompt"),
            RouterRequest(model=model, prompt=giant_prompt),
        ]

        with pytest.raises(HydrogenPromptCapError):
            await router.preflight(requests)

    asyncio.run(_run())


def test_build_cache_control_blocks() -> None:
    """Verify cache control blocks put static context first with at most 4 breakpoints."""
    static = [f"Context block {i}" for i in range(6)]
    blocks = build_cache_control_blocks(static, "Dynamic user query")

    assert len(blocks) == 7
    assert "cache_control" not in blocks[0]
    assert "cache_control" not in blocks[1]
    assert blocks[2]["cache_control"] == {"type": "ephemeral"}
    assert blocks[5]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in blocks[6]


# ======================================================================================
# Gate 4: Schema & Exception Integrity Gate
# ======================================================================================


def test_extract_json_object_tolerates_banners_and_logs() -> None:
    """Verify extract_json_object locates the CLI response envelope even with leading banners."""
    raw_output = """
    Connecting to service...
    {"status": "info", "log": "ready"}
    {"response": "{\\"name\\": \\"ParsedEntity\\", \\"score\\": 42}", "stats": {}}
    Trailing garbage line
    """
    obj = extract_json_object(raw_output)
    assert obj is not None
    assert "response" in obj


def test_malformed_json_triggers_bounded_retries() -> None:
    """Verify malformed JSON from provider triggers bounded retries up to max_retries."""
    async def _run():
        router = ModelRouter(run_id="test-retries", max_retries=2, backoff_base_s=0.01)

        attempts = 0

        async def mock_run_subp(argv: list[str], stdin: str, env: dict[str, str], timeout: float):
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                return 0, '{"response": "{\\"name\\": \\"Broken", "stats": {}}', ""
            return 0, '{"response": "{\\"name\\": \\"Fixed\\", \\"score\\": 88}", "stats": {}}', ""

        router._run_subprocess = mock_run_subp  # type: ignore[assignment]

        res = await router.call(
            MODEL_GEMINI_CARBON,
            "Test prompt",
            response_schema=SampleSchema,
        )

        assert isinstance(res, SampleSchema)
        assert res.name == "Fixed"
        assert res.score == 88
        assert attempts == 3

    asyncio.run(_run())


def test_malformed_json_exhausts_retries() -> None:
    """Verify malformed JSON that fails repeatedly raises StructuredOutputError after max_retries."""
    async def _run():
        router = ModelRouter(run_id="test-retries-fail", max_retries=2, backoff_base_s=0.01)

        async def mock_run_subp(argv: list[str], stdin: str, env: dict[str, str], timeout: float):
            return 0, '{"response": "not valid json at all", "stats": {}}', ""

        router._run_subprocess = mock_run_subp  # type: ignore[assignment]

        with pytest.raises(StructuredOutputError):
            await router.call(MODEL_GEMINI_CARBON, "Test prompt", response_schema=SampleSchema)

    asyncio.run(_run())


def test_secret_redaction_scrubs_keys_and_tokens() -> None:
    """Verify API keys and OAuth tokens are redacted from strings and telemetry payloads."""
    secret_text = (
        "Error calling API with AIzaSyD9V-1234567890123456789012345 "
        "and sk-ant-api03-abcdef1234567890abcdef1234567890 "
        "and ya29.a0ARrdaM-sensitive-google-token"
    )

    scrubbed = redact_secrets(secret_text)
    assert "AIzaSy" not in scrubbed
    assert "sk-ant" not in scrubbed
    assert "ya29" not in scrubbed
    assert "[REDACTED]" in scrubbed

    payload = {
        "error": "Failed with token ya29.a0ARrdaM-12345",
        "nested": {"details": "Key: AIzaSyD9V-1234567890123456789012345"},
    }
    safe_payload = sanitize_telemetry_payload(payload)
    assert "ya29" not in safe_payload["error"]
    assert "AIzaSy" not in safe_payload["nested"]["details"]


def test_telemetry_callback_receives_scrubbed_payload() -> None:
    """Verify SSE telemetry payloads never leak secrets on failure."""
    async def _run():
        emitted_events: list[dict[str, Any]] = []

        async def on_telemetry(payload: dict[str, Any]) -> None:
            emitted_events.append(payload)

        router = ModelRouter(
            run_id="test-telemetry-redact",
            on_telemetry_update=on_telemetry,
            backoff_base_s=0.01,
            max_retries=0,
        )

        async def mock_run_subp(argv: list[str], stdin: str, env: dict[str, str], timeout: float):
            return 1, "", "Connection failed with key AIzaSyD9V-1234567890123456789012345"

        router._run_subprocess = mock_run_subp  # type: ignore[assignment]

        with pytest.raises(ModelRouterError):
            await router.call(MODEL_GEMINI_CARBON, "Prompt")

        assert len(emitted_events) >= 2
        failed_event = [e for e in emitted_events if e["type"] == "call_failed"][0]
        assert "AIzaSy" not in failed_event["error"]
        assert "[REDACTED]" in failed_event["error"]

    asyncio.run(_run())


# ======================================================================================
# Gate 5: Downstream Interface Compatibility Gate
# ======================================================================================


def test_generate_downstream_interface() -> None:
    """Verify async def generate(...) handles flexible positional and keyword argument patterns."""
    async def _run():
        router = ModelRouter(run_id="test-gen")

        async def mock_run_subp(argv: list[str], stdin: str, env: dict[str, str], timeout: float):
            return 0, json.dumps({"response": "Generated response", "stats": {}}), ""

        router._run_subprocess = mock_run_subp  # type: ignore[assignment]

        # Pattern 1: prompt positional
        r1 = await router.generate("Hello world")
        assert r1 == "Generated response"

        # Pattern 2: model positional, prompt keyword
        r2 = await router.generate(MODEL_GEMINI_CARBON, prompt="Hello carbon")
        assert r2 == "Generated response"

        # Pattern 3: keyword model & prompt
        r3 = await router.generate(model=MODEL_GEMINI_CARBON, prompt="Hello keyword")
        assert r3 == "Generated response"

    asyncio.run(_run())


def test_get_ledger_summary_sync_and_async() -> None:
    """Verify get_ledger_summary() is callable synchronously and is awaitable."""
    router = ModelRouter(run_id="test-summary", max_budget_usd=5.00)

    s_sync = router.get_ledger_summary()
    assert isinstance(s_sync, dict)
    assert s_sync["run_id"] == "test-summary"
    assert s_sync["max_budget_usd"] == 5.00
    assert s_sync["spent_usd"] == 0.0

    async def check_async():
        s_async = await router.get_ledger_summary()
        assert isinstance(s_async, dict)
        assert s_async["max_budget_usd"] == 5.00

    asyncio.run(check_async())


def test_flexible_error_constructors() -> None:
    """Verify BudgetExceededError and FlashTokenCliffError support both string messages and numeric parameters."""
    e1 = BudgetExceededError(1.5, 4.5, 0.5)
    assert e1.spent_usd == 1.5
    assert e1.max_budget_usd == 4.5
    assert "$1.5000" in str(e1)

    e2 = BudgetExceededError("Custom budget exceeded message")
    assert "Custom budget exceeded message" in str(e2)

    f1 = FlashTokenCliffError(95000, 90000)
    assert f1.estimated_tokens == 95000
    assert "95,000" in str(f1)
    assert MODEL_GEMINI_HYDROGEN in str(f1) and "allow_long_context" in str(f1)
    assert HydrogenPromptCapError is FlashTokenCliffError  # one class, two names

    f2 = FlashTokenCliffError("Custom cliff error")
    assert "Custom cliff error" in str(f2)


# ======================================================================================
# Gate 6: Virtual Cache-Read Rate Parity Gate
# ======================================================================================


def test_claude_virtual_cache_read_rates_and_ledger_parity() -> None:
    """Verify Claude virtual cache-read rates ($0.20 Opus, $0.10 Sonnet, $0.20 Fable) and ledger parity."""
    assert CLAUDE_VIRTUAL_PRICING[MODEL_CLAUDE_OPUS].cached == pytest.approx(0.20)
    assert CLAUDE_VIRTUAL_PRICING[MODEL_CLAUDE_SONNET].cached == pytest.approx(0.10)
    assert CLAUDE_VIRTUAL_PRICING[MODEL_CLAUDE_FABLE].cached == pytest.approx(0.20)

    # 1M cached tokens alone (0 uncached input, 0 output) must match the exact $/MTok rate
    assert compute_cost(CLAUDE_VIRTUAL_PRICING[MODEL_CLAUDE_OPUS], 0, 1_000_000, 0) == pytest.approx(0.20)
    assert compute_cost(CLAUDE_VIRTUAL_PRICING[MODEL_CLAUDE_SONNET], 0, 1_000_000, 0) == pytest.approx(0.10)
    assert compute_cost(CLAUDE_VIRTUAL_PRICING[MODEL_CLAUDE_FABLE], 0, 1_000_000, 0) == pytest.approx(0.20)

    async def _run() -> None:
        router = ModelRouter(run_id="cache-parity")

        async def mock_run_subp(
            argv: list[str], stdin: str, env: dict[str, str], timeout: float
        ) -> tuple[int, str, str]:
            envelope = {
                "result": "ok",
                "usage": {
                    "input_tokens": 0,
                    "cache_read_input_tokens": 1_000_000,
                    "output_tokens": 0,
                },
            }
            return 0, json.dumps(envelope), ""

        router._run_subprocess = mock_run_subp  # type: ignore[assignment]
        with patch("backend.models_router.shutil.which", return_value="/usr/bin/claude"):
            for model, expected_virtual in (
                (MODEL_CLAUDE_OPUS, 0.20),
                (MODEL_CLAUDE_SONNET, 0.10),
                (MODEL_CLAUDE_FABLE, 0.20),
            ):
                await router.call(model, "Check cache parity")
                summary = router.get_ledger_summary()
                assert summary["spent_usd"] == 0.0
                model_stats = summary["by_model"][model]
                assert model_stats["cached_tokens"] == 1_000_000
                assert model_stats["actual_usd"] == 0.0
                assert model_stats["virtual_usd"] == pytest.approx(expected_virtual)

    asyncio.run(_run())


# ======================================================================================
# Gate 7: Gemini 4 Hydrogen Gate
# ======================================================================================

BACKEND_DIR = Path(__file__).parents[1] / "backend"
HYDROGEN_IDS = [MODEL_GEMINI_HYDROGEN, MODEL_GEMINI_FLASH]  # the primary id and its alias


def _usage(prompt: int, cached: int = 0, candidates: int = 0, thoughts: int = 0) -> MagicMock:
    return MagicMock(
        prompt_token_count=prompt,
        cached_content_token_count=cached,
        candidates_token_count=candidates,
        thoughts_token_count=thoughts,
    )


class _VertexDouble:
    """google-genai client double. Records ``generate_content`` calls; any ``caches`` access fails."""

    def __init__(self, text: str = "ok", usage: Any = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.text = text
        self.usage = usage
        self.aio = self  # client.aio.models.generate_content(...)
        self.models = self

    async def generate_content(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return MagicMock(text=self.text, usage_metadata=self.usage)

    @property
    def caches(self) -> Any:  # client.caches / client.aio.caches
        raise AssertionError("explicit context caching (caches.create) must never be used")


def _vertex_call(
    model: str,
    prompt: str = "hello",
    *,
    effort: str = "medium",
    usage: Any = None,
    max_budget_usd: float = 4.50,
    **kwargs: Any,
) -> tuple[Any, _VertexDouble, ModelRouter]:
    """One metered call (``route="vertex"``) through a router wired to a ``_VertexDouble``."""
    client = _VertexDouble(usage=usage)

    async def _run() -> Any:
        router = ModelRouter(
            run_id="hydrogen-gate", max_budget_usd=max_budget_usd, gemini_use_subscription_first=False
        )
        router._vertex_client_factory = lambda loc: client
        result = await router.call_detailed(model, prompt, effort=effort, route="vertex", **kwargs)
        return result, router

    result, router = asyncio.run(_run())
    return result, client, router


def _install_fake_genai(
    monkeypatch: pytest.MonkeyPatch, accepted: tuple[str, ...] = ("LOW", "MEDIUM", "HIGH", "XHIGH")
) -> None:
    """Replace ``google.genai.types`` so the tests do not depend on the installed SDK version.

    ``accepted`` models the SDK's ``ThinkingLevel`` enum: anything else raises ``ValueError``
    (pydantic's ``ValidationError`` is one).
    """

    ThinkingLevel = Enum(
        "ThinkingLevel",
        {name: name for name in accepted},
        type=str,
    )

    class FakeThinkingConfig:
        def __init__(self, thinking_level: str) -> None:
            if thinking_level not in accepted:
                raise ValueError(f"{thinking_level!r} is not a valid ThinkingLevel")
            self.thinking_level = thinking_level

    class FakeGenerateContentConfig:
        def __init__(self, **kwargs: Any) -> None:
            self.__dict__.update(kwargs)

    types_mod = ModuleType("google.genai.types")
    types_mod.ThinkingLevel = ThinkingLevel  # type: ignore[attr-defined]
    types_mod.ThinkingConfig = FakeThinkingConfig  # type: ignore[attr-defined]
    types_mod.GenerateContentConfig = FakeGenerateContentConfig  # type: ignore[attr-defined]
    genai_mod = ModuleType("google.genai")
    genai_mod.types = types_mod  # type: ignore[attr-defined]
    google_mod = ModuleType("google")
    google_mod.genai = genai_mod  # type: ignore[attr-defined]
    google_mod.__path__ = []  # type: ignore[attr-defined]
    for name, module in (("google", google_mod), ("google.genai", genai_mod), ("google.genai.types", types_mod)):
        monkeypatch.setitem(sys.modules, name, module)


@pytest.fixture
def fake_genai(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_genai(monkeypatch)


def test_hydrogen_is_registered_with_flat_vertex_rates_and_the_flash_alias_shares_them() -> None:
    """Verify $1.50 input / $0.075 implicit-cached / $7.50 output per MTok, and the alias mapping."""
    assert MODEL_GEMINI_HYDROGEN == "gemini-4-hydrogen"
    assert MODEL_GEMINI_FLASH == "gemini-3-8-flash"
    rate = VERTEX_PRICING[MODEL_GEMINI_HYDROGEN]
    assert rate == Pricing(1.50, 0.075, 7.50)
    assert VERTEX_PRICING[MODEL_GEMINI_FLASH] == rate

    assert MODEL_ALIASES == {MODEL_GEMINI_FLASH: MODEL_GEMINI_HYDROGEN}
    assert resolve_model(MODEL_GEMINI_FLASH) == MODEL_GEMINI_HYDROGEN
    assert resolve_model(MODEL_GEMINI_HYDROGEN) == MODEL_GEMINI_HYDROGEN
    assert resolve_model(MODEL_GEMINI_CARBON) == MODEL_GEMINI_CARBON

    assert compute_cost(rate, 1_000_000, 0, 0) == pytest.approx(1.50)
    assert compute_cost(rate, 0, 1_000_000, 0) == pytest.approx(0.075)
    assert compute_cost(rate, 0, 0, 1_000_000) == pytest.approx(7.50)
    # the other Gemini 4 models are untouched by the migration
    assert VERTEX_PRICING[MODEL_GEMINI_CARBON] == Pricing(2.00, 0.10, 10.00)
    assert VERTEX_PRICING[MODEL_GEMINI_ARGON] == Pricing(2.50, 0.25, 12.00)


@pytest.mark.parametrize("model", HYDROGEN_IDS)
def test_vertex_call_bills_hydrogen_at_flat_rates_with_the_implicit_cache_discount(model: str) -> None:
    """Verify the ledger prices non-cached input, implicit-cached input and output+thinking separately."""
    usage = _usage(prompt=2_000, cached=1_000, candidates=300, thoughts=200)
    result, client, router = _vertex_call(model, usage=usage)

    record = result.record
    assert (record.input_tokens, record.cached_tokens, record.output_tokens) == (1_000, 1_000, 500)
    assert record.actual_cost_usd == pytest.approx((1_000 * 1.50 + 1_000 * 0.075 + 500 * 7.50) / 1_000_000)
    assert record.billing_route == "vertex_ai" and record.virtual_cost_usd == 0.0
    assert record.model == MODEL_GEMINI_HYDROGEN and record.requested_model == model
    assert client.calls[0]["model"] == MODEL_GEMINI_HYDROGEN  # the alias never reaches Vertex
    assert set(router.get_ledger_summary()["by_model"]) == {MODEL_GEMINI_HYDROGEN}


def test_hydrogen_pricing_is_flat_across_the_old_100k_cliff() -> None:
    """Verify cost scales linearly through 100K prompt tokens (no long-context price step)."""
    rate = VERTEX_PRICING[MODEL_GEMINI_HYDROGEN]
    assert compute_cost(rate, 150_000, 0, 0) == pytest.approx(3 * compute_cost(rate, 50_000, 0, 0))

    long_prompt = "x" * int(3.5 * 120_000)  # ~120K estimated tokens, above the soft guard
    result, _, _ = _vertex_call(
        MODEL_GEMINI_HYDROGEN,
        long_prompt,
        usage=_usage(prompt=120_000, candidates=100),
        allow_long_context=True,
    )
    assert result.record.actual_cost_usd == pytest.approx((120_000 * 1.50 + 100 * 7.50) / 1_000_000)


@pytest.mark.parametrize("model", HYDROGEN_IDS)
@pytest.mark.parametrize(
    ("effort", "expected"),
    [("low", "LOW"), ("medium", "MEDIUM"), ("high", "HIGH"), ("xhigh", "XHIGH"), ("max", "XHIGH")],
)
def test_every_hydrogen_vertex_call_passes_an_explicit_thinking_level(
    fake_genai: None, model: str, effort: str, expected: str
) -> None:
    """Verify thinking_level is always sent (hydrogen runs at HIGH when it is omitted)."""
    assert thinking_level_for(model, effort) == expected
    _, client, _ = _vertex_call(model, effort=effort)
    config = client.calls[0]["config"]
    assert config.thinking_config is not None
    assert config.thinking_config.thinking_level == expected
    assert not hasattr(config, "temperature")  # temperature is never set on a Gemini call


@pytest.mark.parametrize("model", [MODEL_GEMINI_CARBON, MODEL_GEMINI_ARGON])
@pytest.mark.parametrize(
    ("effort", "expected"),
    [("low", "LOW"), ("medium", "MEDIUM"), ("high", "HIGH"), ("xhigh", "HIGH"), ("max", "HIGH")],
)
def test_other_gemini_4_models_keep_their_thinking_levels(
    fake_genai: None, model: str, effort: str, expected: str
) -> None:
    """Verify XHIGH is hydrogen-only: carbon and argon still clamp the top efforts to HIGH."""
    _, client, _ = _vertex_call(model, effort=effort)
    assert client.calls[0]["config"].thinking_config.thinking_level == expected


def test_hydrogen_xhigh_degrades_to_an_explicit_high_when_the_sdk_predates_it(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Verify an SDK whose ThinkingLevel lacks XHIGH gets HIGH (what omission would give), not a crash."""
    _install_fake_genai(monkeypatch, accepted=("LOW", "MEDIUM", "HIGH"))
    with caplog.at_level(logging.WARNING, logger="backend.models_router"):
        for effort in ("xhigh", "max"):
            _, client, _ = _vertex_call(MODEL_GEMINI_HYDROGEN, effort=effort)
            assert client.calls[0]["config"].thinking_config.thinking_level == "HIGH"
    assert "XHIGH" in caplog.text


def test_vertex_calls_never_use_explicit_context_caching(fake_genai: None) -> None:
    """Verify implicit-only caching: no model ever touches ``caches`` (the double fails if one does)."""
    for model in (*HYDROGEN_IDS, MODEL_GEMINI_CARBON, MODEL_GEMINI_ARGON):
        result, client, _ = _vertex_call(model, usage=_usage(prompt=6_000, cached=4_096, candidates=10))
        assert result.record.cached_tokens == 4_096  # the implicit discount is read from usage_metadata
        assert len(client.calls) == 1


def test_backend_source_never_references_explicit_context_caching() -> None:
    """Verify (statically) that no backend module reaches for the SDK's ``caches`` API."""
    offenders: list[str] = []
    for path in sorted(BACKEND_DIR.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        offenders += [
            f"{path.name}:{node.lineno}"
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute) and node.attr == "caches"
        ]
    assert offenders == []


@pytest.mark.parametrize("model", HYDROGEN_IDS)
def test_flash_alias_is_served_as_hydrogen_on_the_cli_and_in_the_ledger(model: str) -> None:
    """Verify the CLI gets the canonical -m value and the quota-equivalent uses the hydrogen rates."""

    async def _run() -> None:
        router = ModelRouter(run_id="alias-cli")
        captured: list[list[str]] = []

        async def mock_run_subp(argv: list[str], stdin: str, env: dict[str, str], timeout: float):
            captured.append(list(argv))
            stats = {"models": {MODEL_GEMINI_HYDROGEN: {"tokens": {"prompt": 1_000, "cached": 200, "candidates": 100, "thoughts": 50}}}}
            return 0, json.dumps({"response": "fine", "stats": stats}), ""

        router._run_subprocess = mock_run_subp  # type: ignore[assignment]
        result = await router.call_detailed(model, "Say fine")

        assert len(captured) == 1 and captured[0][captured[0].index("-m") + 1] == MODEL_GEMINI_HYDROGEN
        record = result.record
        assert record.model == MODEL_GEMINI_HYDROGEN and record.requested_model == model
        assert record.billing_route == "subscription_oauth" and record.actual_cost_usd == 0.0
        assert record.virtual_cost_usd == pytest.approx((800 * 1.50 + 200 * 0.075 + 150 * 7.50) / 1_000_000)
        assert set(router.get_ledger_summary()["by_model"]) == {MODEL_GEMINI_HYDROGEN}

    asyncio.run(_run())


def test_hydrogen_and_its_alias_share_one_subscription_cooldown() -> None:
    """Verify a quota error on either spelling puts both on Vertex, tracked under the canonical id."""

    async def _run() -> None:
        router = ModelRouter(run_id="alias-cooldown", backoff_base_s=0.01)
        cli_calls = 0

        async def mock_run_subp(argv: list[str], stdin: str, env: dict[str, str], timeout: float):
            nonlocal cli_calls
            cli_calls += 1
            return 1, json.dumps({"error": {"code": 429, "message": "RESOURCE_EXHAUSTED: quota exceeded"}}), ""

        client = _VertexDouble(text="vertex answer")
        router._run_subprocess = mock_run_subp  # type: ignore[assignment]
        router._vertex_client_factory = lambda loc: client

        assert await router.call(MODEL_GEMINI_HYDROGEN, "first") == "vertex answer"
        assert cli_calls == 1
        assert list(router._gemini_sub_exhausted_until) == [MODEL_GEMINI_HYDROGEN]
        assert router._sub_exhausted(MODEL_GEMINI_FLASH) is True

        assert await router.call(MODEL_GEMINI_FLASH, "second") == "vertex answer"
        assert cli_calls == 1  # the alias went straight to Vertex
        assert [call["model"] for call in client.calls] == [MODEL_GEMINI_HYDROGEN] * 2

    asyncio.run(_run())


def test_allow_long_context_overrides_the_soft_guard_on_every_entry_point(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Verify call / call_detailed / generate / preflight / call_many honour allow_long_context."""
    giant = "x" * int(HYDROGEN_PROMPT_TOKEN_CAP * 3.6)

    async def _run() -> None:
        router = ModelRouter(run_id="override")
        seen_tokens: list[int] = []

        async def mock_run_subp(argv: list[str], stdin: str, env: dict[str, str], timeout: float):
            seen_tokens.append(estimate_tokens(stdin))
            return 0, json.dumps({"response": "long ok", "stats": {}}), ""

        router._run_subprocess = mock_run_subp  # type: ignore[assignment]

        with pytest.raises(HydrogenPromptCapError):  # the default is still guarded
            await router.call(MODEL_GEMINI_HYDROGEN, giant)
        assert seen_tokens == []

        with caplog.at_level(logging.WARNING, logger="backend.models_router"):
            assert await router.call(MODEL_GEMINI_HYDROGEN, giant, allow_long_context=True) == "long ok"
            assert (await router.call_detailed(MODEL_GEMINI_FLASH, giant, allow_long_context=True)).output == "long ok"
            assert await router.generate(MODEL_GEMINI_HYDROGEN, prompt=giant, allow_long_context=True) == "long ok"
        assert len(seen_tokens) == 3 and min(seen_tokens) > HYDROGEN_PROMPT_TOKEN_CAP
        assert "allow_long_context=True" in caplog.text

        allowed = RouterRequest(model=MODEL_GEMINI_HYDROGEN, prompt=giant, allow_long_context=True)
        assert await router.preflight([allowed]) == 0.0  # subscription route: no metered spend
        assert [r.output for r in await router.call_many([allowed])] == ["long ok"]
        with pytest.raises(HydrogenPromptCapError):
            await router.preflight([RouterRequest(model=MODEL_GEMINI_HYDROGEN, prompt=giant)])

    asyncio.run(_run())


def test_allow_long_context_does_not_bypass_the_budget_gate() -> None:
    """Verify the override only lifts the size guard: the metered reservation still has to fit."""
    long_prompt = "x" * int(3.5 * 120_000)  # ~120K tokens -> ~$0.18 of input alone
    client = _VertexDouble()

    async def _run() -> None:
        router = ModelRouter(run_id="override-budget", max_budget_usd=0.01)
        router._vertex_client_factory = lambda loc: client
        with pytest.raises(BudgetExceededError):
            await router.call(MODEL_GEMINI_HYDROGEN, long_prompt, route="vertex", allow_long_context=True)

    asyncio.run(_run())
    assert client.calls == []


def test_thinking_level_resolves_non_str_enum_with_and_without_xhigh(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Verify ThinkingLevel enum resolution when ThinkingLevel is a non-str Enum."""

    def _install_non_str_enum(members: dict[str, int]) -> type[Enum]:
        enum_cls = Enum("ThinkingLevel", members)

        class PermissiveThinkingConfig:
            def __init__(self, thinking_level: Any) -> None:
                # Deliberately does NOT raise ValueError so we prove enum lookup handles fallback.
                self.thinking_level = thinking_level

        class FakeGenerateContentConfig:
            def __init__(self, **kwargs: Any) -> None:
                self.__dict__.update(kwargs)

        types_mod = ModuleType("google.genai.types")
        types_mod.ThinkingLevel = enum_cls  # type: ignore[attr-defined]
        types_mod.ThinkingConfig = PermissiveThinkingConfig  # type: ignore[attr-defined]
        types_mod.GenerateContentConfig = FakeGenerateContentConfig  # type: ignore[attr-defined]
        genai_mod = ModuleType("google.genai")
        genai_mod.types = types_mod  # type: ignore[attr-defined]
        google_mod = ModuleType("google")
        google_mod.genai = genai_mod  # type: ignore[attr-defined]
        google_mod.__path__ = []  # type: ignore[attr-defined]
        for name, module in (
            ("google", google_mod),
            ("google.genai", genai_mod),
            ("google.genai.types", types_mod),
        ):
            monkeypatch.setitem(sys.modules, name, module)
        return enum_cls

    # Case 1: XHIGH present in non-str Enum -> resolves ThinkingLevel.XHIGH
    full_enum = _install_non_str_enum({"LOW": 1, "MEDIUM": 2, "HIGH": 3, "XHIGH": 4})
    client = _VertexDouble()

    async def _run_xhigh() -> None:
        router = ModelRouter(run_id="enum-with-xhigh")
        router._vertex_client_factory = lambda loc: client
        await router.call(MODEL_GEMINI_HYDROGEN, "hi", effort="xhigh", route="vertex")

    asyncio.run(_run_xhigh())
    assert client.calls[-1]["config"].thinking_config.thinking_level is full_enum.XHIGH  # type: ignore[attr-defined]

    # Case 2: XHIGH absent in non-str Enum -> falls back to ThinkingLevel.HIGH with warning
    legacy_enum = _install_non_str_enum({"LOW": 1, "MEDIUM": 2, "HIGH": 3})
    client2 = _VertexDouble()

    async def _run_legacy() -> None:
        router = ModelRouter(run_id="enum-no-xhigh")
        router._vertex_client_factory = lambda loc: client2
        with caplog.at_level(logging.WARNING, logger="backend.models_router"):
            await router.call(MODEL_GEMINI_HYDROGEN, "hi", effort="xhigh", route="vertex")

    asyncio.run(_run_legacy())
    assert client2.calls[-1]["config"].thinking_config.thinking_level is legacy_enum.HIGH  # type: ignore[attr-defined]
    assert "ThinkingLevel lacks XHIGH" in caplog.text


def test_zero_thinking_budget_and_zero_minimal_runtime_and_ast(fake_genai: None) -> None:
    """Verify thinking_budget and MINIMAL are never used at runtime or in models_router.py AST."""
    client = _VertexDouble()

    async def _run() -> None:
        router = ModelRouter(run_id="no-budget-no-minimal")
        router._vertex_client_factory = lambda loc: client
        for effort in ("low", "medium", "high", "xhigh", "max"):
            await router.call(MODEL_GEMINI_HYDROGEN, "hi", effort=effort, route="vertex")  # type: ignore[arg-type]

    asyncio.run(_run())
    for call in client.calls:
        tc = call["config"].thinking_config
        assert not hasattr(tc, "thinking_budget")
        assert tc.thinking_level != "MINIMAL"

    router_path = Path(__file__).resolve().parents[1] / "backend" / "models_router.py"
    tree = ast.parse(router_path.read_text(encoding="utf-8"), filename=str(router_path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            assert node.attr != "thinking_budget"
        elif isinstance(node, ast.keyword):
            assert node.arg != "thinking_budget"
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            assert node.value != "MINIMAL"


def test_structured_output_budgets_foreground_cap_and_unified_context_limit(
    fake_genai: None,
) -> None:
    """Verify structured output defaults by effort, 65,536 cap, and 1,048,576 context limit."""

    class _Item(BaseModel):
        ok: bool = True

    client = _VertexDouble(text='{"ok": true}')

    async def _run() -> None:
        router = ModelRouter(run_id="output-budgets", max_budget_usd=500.0)
        router._vertex_client_factory = lambda loc: client

        # low / medium -> 16,384
        for eff in ("low", "medium"):
            await router.call(
                MODEL_GEMINI_HYDROGEN, "hi", response_schema=_Item, effort=eff, route="vertex"  # type: ignore[arg-type]
            )
            assert client.calls[-1]["config"].max_output_tokens == STRUCTURED_MAX_OUTPUT_TOKENS == 16_384

        # high / xhigh / max -> 32,768
        for eff in ("high", "xhigh", "max"):
            await router.call(
                MODEL_GEMINI_HYDROGEN, "hi", response_schema=_Item, effort=eff, route="vertex"  # type: ignore[arg-type]
            )
            assert (
                client.calls[-1]["config"].max_output_tokens
                == STRUCTURED_HIGH_MAX_OUTPUT_TOKENS
                == 32_768
            )

        # Explicit caller max_output_tokens=24,576 is honored on structured calls
        await router.call(
            MODEL_GEMINI_HYDROGEN,
            "hi",
            response_schema=_Item,
            effort="medium",
            max_output_tokens=24_576,
            route="vertex",
        )
        assert client.calls[-1]["config"].max_output_tokens == 24_576

        # Explicit caller max_output_tokens=100,000 is clamped to 65,536
        await router.call(
            MODEL_GEMINI_HYDROGEN,
            "hi",
            max_output_tokens=100_000,
            route="vertex",
        )
        assert client.calls[-1]["config"].max_output_tokens == FOREGROUND_MAX_OUTPUT_TOKENS == 65_536

        # Invalid max_output_tokens rejected before spawning or billing
        for bad_val in (0, -1, True):
            with pytest.raises(ValueError, match="max_output_tokens must be a positive integer"):
                await router.call(MODEL_GEMINI_HYDROGEN, "hi", max_output_tokens=bad_val, route="vertex")  # type: ignore[arg-type]

        # Clamp max_output_tokens when prompt_tokens + max_output_tokens > 1,048,576
        near_limit_prompt = "x" * int(1_040_000 * 3.5)
        await router.call(
            MODEL_GEMINI_HYDROGEN,
            near_limit_prompt,
            response_schema=_Item,
            effort="high",
            route="vertex",
            allow_long_context=True,
        )
        prompt_toks = estimate_tokens(near_limit_prompt + router._schema_json(_Item))
        expected_clamped = GEMINI_CONTEXT_LIMIT_TOKENS - prompt_toks
        assert 1 <= expected_clamped < STRUCTURED_HIGH_MAX_OUTPUT_TOKENS
        assert client.calls[-1]["config"].max_output_tokens == expected_clamped

        # >= 1,048,576 prompt tokens raises HydrogenPromptCapError even with allow_long_context=True
        over_limit_prompt = "x" * int(GEMINI_CONTEXT_LIMIT_TOKENS * 3.6)
        with pytest.raises(HydrogenPromptCapError):
            await router.call(
                MODEL_GEMINI_HYDROGEN,
                over_limit_prompt,
                route="vertex",
                allow_long_context=True,
            )
        with pytest.raises(HydrogenPromptCapError):
            await router.preflight(
                [RouterRequest(model=MODEL_GEMINI_CARBON, prompt=over_limit_prompt, allow_long_context=True)]
            )

    asyncio.run(_run())


