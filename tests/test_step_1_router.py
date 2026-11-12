"""Comprehensive Pre-Commit Audit Test Suite for ModelRouter.

Covers:
1. Concurrency & Race-Condition Gate (atomic reservations, budget enforcement, asyncio.gather)
2. Subscription OAuth & Vertex AI Failover Gate (CLEAN_ENV stripping, CLI flags, per-model 300s cooldown TTL, location="global")
3. Pricing Cliff & Prompt Caching Gate (>90K Flash token cap, preflight check, cache block structure)
4. Schema & Exception Integrity Gate (CLI envelope unwrapping, Pydantic v2 parsing, bounded retries on malformed JSON, secret scrubbing)
5. Downstream Interface Compatibility Gate (generate(...), get_ledger_summary(), BudgetExceededError, FlashTokenCliffError)
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel, Field

from backend.models_router import (
    CLAUDE_VIRTUAL_PRICING,
    FLASH_PROMPT_TOKEN_CAP,
    GEMINI_SUB_COOLDOWN_S,
    MODEL_CLAUDE_FABLE,
    MODEL_CLAUDE_OPUS,
    MODEL_CLAUDE_SONNET,
    MODEL_GEMINI_ARGON,
    MODEL_GEMINI_CARBON,
    MODEL_GEMINI_FLASH,
    VERTEX_PRICING,
    BudgetExceededError,
    ConfigurationError,
    FlashTokenCliffError,
    ModelRouter,
    ModelRouterError,
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
    sanitize_telemetry_payload,
    strip_code_fences,
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

    so a gemini-4-carbon 429 never trips gemini-3-8-flash onto paid Vertex AI.
    """
    async def _run():
        router = ModelRouter(run_id="test-cooldown", backoff_base_s=0.01)

        async def mock_run_subp(argv: list[str], stdin: str, env: dict[str, str], timeout: float):
            if MODEL_GEMINI_CARBON in argv:
                return 1, json.dumps({"error": {"code": 429, "message": "RESOURCE_EXHAUSTED: quota exceeded"}}), ""
            return 0, json.dumps({"response": "Flash CLI success", "stats": {}}), ""

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

        assert router._sub_exhausted(MODEL_GEMINI_FLASH) is False

        mock_vertex_generate.reset_mock()
        flash_res = await router.call(MODEL_GEMINI_FLASH, "Flash prompt")
        assert flash_res == "Flash CLI success"
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
        router._vertex_client(MODEL_GEMINI_FLASH)

    assert requested_locations == ["global"]


# ======================================================================================
# Gate 3: Pricing Cliff & Prompt Caching Gate
# ======================================================================================


def test_flash_prompt_token_cliff_hard_guard() -> None:
    """Verify gemini-3-8-flash enforces a hard pre-flight cap at 90,000 tokens before network calls."""
    async def _run():
        router = ModelRouter(run_id="test-cliff")
        router._run_subprocess = AsyncMock()  # type: ignore[assignment]

        giant_prompt = "x" * int(FLASH_PROMPT_TOKEN_CAP * 3.6)

        with pytest.raises(FlashTokenCliffError) as exc_info:
            await router.call(MODEL_GEMINI_FLASH, giant_prompt)

        assert exc_info.value.limit == FLASH_PROMPT_TOKEN_CAP
        assert exc_info.value.estimated_tokens > FLASH_PROMPT_TOKEN_CAP
        router._run_subprocess.assert_not_called()

    asyncio.run(_run())


def test_flash_prompt_token_cliff_in_preflight() -> None:
    """Verify preflight rejects any gemini-3-8-flash request exceeding 90,000 tokens."""
    async def _run():
        router = ModelRouter(run_id="test-cliff-preflight")
        giant_prompt = "x" * int(FLASH_PROMPT_TOKEN_CAP * 3.6)

        requests = [
            RouterRequest(model=MODEL_GEMINI_CARBON, prompt="Small prompt"),
            RouterRequest(model=MODEL_GEMINI_FLASH, prompt=giant_prompt),
        ]

        with pytest.raises(FlashTokenCliffError):
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

