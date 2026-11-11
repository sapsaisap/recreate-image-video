"""Free-tier guard, per-task model chains and daily-quota memory in call_gemini."""
import dataclasses
from datetime import datetime

import pytest
from google.genai.errors import ClientError

import recreate_video as rv

CHAINS = {
    "analysis": ("gemini-2.5-flash", "gemini-3.5-flash-lite"),
    "text": ("gemini-3.5-flash-lite", "gemini-2.5-flash-lite"),
    "metadata": ("gemini-2.5-flash-lite", "gemini-3.5-flash-lite"),
    "story": ("gemini-2.5-flash-lite", "gemini-3.5-flash-lite"),
}
NOV_11 = datetime(2026, 11, 11, 12, 0, tzinfo=rv._PACIFIC)


@pytest.fixture
def env(tmp_path, monkeypatch):
    cfg = dataclasses.replace(rv.CONFIG, work_dir=tmp_path, max_wait_seconds=600, task_models=dict(CHAINS))
    monkeypatch.setattr(rv, "CONFIG", cfg)
    monkeypatch.setattr(rv._STATE, "quota", None)
    monkeypatch.setattr(rv, "_pacific_now", lambda: NOV_11)
    waits = []
    monkeypatch.setattr(rv, "_wait_with_countdown", lambda seconds, reason, notify: waits.append(seconds))
    return waits


def quota_error(quota_id, value="20", retry="37s"):
    body = {"error": {
        "code": 429,
        "message": "You exceeded your current quota, please check your plan and billing details.",
        "status": "RESOURCE_EXHAUSTED",
        "details": [
            {"@type": "type.googleapis.com/google.rpc.QuotaFailure", "violations": [
                {"quotaMetric": "generativelanguage.googleapis.com/generate_content_free_tier_requests",
                 "quotaId": quota_id, "quotaValue": value},
            ]},
            {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": retry},
        ],
    }}
    return ClientError(429, body)


PER_DAY = "GenerateRequestsPerDayPerProjectPerModel-FreeTier"
PER_MINUTE = "GenerateRequestsPerMinutePerProjectPerModel-FreeTier"


def failing_on(errors):
    """fn for call_gemini: raises errors[model] (once per entry) else returns the model."""
    calls = []

    def fn(model):
        calls.append(model)
        queue = errors.get(model)
        if queue:
            raise queue.pop(0)
        return model

    return fn, calls


class TestDailyQuota:
    def test_switches_to_next_model(self, env):
        fn, calls = failing_on({"gemini-2.5-flash": [quota_error(PER_DAY)]})
        assert rv.call_gemini(fn, task="analysis", throttle=False) == "gemini-3.5-flash-lite"
        assert calls == ["gemini-2.5-flash", "gemini-3.5-flash-lite"]

    def test_used_up_model_is_not_retried_by_later_calls(self, env):
        fn, _ = failing_on({"gemini-2.5-flash": [quota_error(PER_DAY)]})
        rv.call_gemini(fn, task="analysis", throttle=False)
        fn2, calls2 = failing_on({})
        rv.call_gemini(fn2, task="analysis", throttle=False)
        assert calls2 == ["gemini-3.5-flash-lite"]

    def test_all_used_up_raises_without_sending_a_request(self, env):
        for model in CHAINS["analysis"]:
            rv._mark_unavailable(model, "daily free quota used up")
        fn, calls = failing_on({})
        with pytest.raises(rv.QuotaExhaustedError, match="midnight Pacific"):
            rv.call_gemini(fn, task="analysis", throttle=False)
        assert calls == []

    def test_memory_survives_restart(self, env):
        rv._mark_unavailable("gemini-2.5-flash", "daily free quota used up")
        rv._STATE.quota = None  # new process: state is re-read from disk
        assert "gemini-2.5-flash" in rv._unavailable_models()

    def test_memory_clears_on_next_pacific_day(self, env, monkeypatch):
        rv._mark_unavailable("gemini-2.5-flash", "daily free quota used up")
        monkeypatch.setattr(rv, "_pacific_now", lambda: datetime(2026, 11, 12, 0, 1, tzinfo=rv._PACIFIC))
        assert rv._unavailable_models() == {}

    def test_calls_are_counted_per_model(self, env):
        fn, _ = failing_on({"gemini-2.5-flash": [quota_error(PER_DAY)]})
        rv.call_gemini(fn, task="analysis", throttle=False)
        assert rv.quota_snapshot()["calls"] == {"gemini-2.5-flash": 1, "gemini-3.5-flash-lite": 1}


class TestUnusableModels:
    def test_limit_zero_means_not_on_free_tier(self, env):
        fn, _ = failing_on({"gemini-2.5-flash": [quota_error(PER_DAY, value="0")]})
        rv.call_gemini(fn, task="analysis", throttle=False)
        assert rv._unavailable_models()["gemini-2.5-flash"] == "not available on the free tier"

    def test_model_not_found_skips_instead_of_failing(self, env):
        missing = ClientError(404, {"error": {"code": 404, "status": "NOT_FOUND",
                                              "message": "models/gemini-2.5-flash is not found"}})
        fn, calls = failing_on({"gemini-2.5-flash": [missing]})
        assert rv.call_gemini(fn, task="analysis", throttle=False) == "gemini-3.5-flash-lite"
        assert rv._unavailable_models()["gemini-2.5-flash"] == "model not found"

    def test_other_client_errors_stay_fatal(self, env):
        bad = ClientError(400, {"error": {"code": 400, "status": "INVALID_ARGUMENT", "message": "bad schema"}})
        fn, calls = failing_on({"gemini-2.5-flash": [bad]})
        with pytest.raises(rv.GeminiCallError):
            rv.call_gemini(fn, task="analysis", throttle=False)
        assert calls == ["gemini-2.5-flash"]
        assert rv._unavailable_models() == {}


class TestPerMinute:
    def test_waits_google_retry_delay_on_same_model(self, env):
        fn, calls = failing_on({"gemini-2.5-flash": [quota_error(PER_MINUTE, retry="37s")]})
        assert rv.call_gemini(fn, task="analysis", throttle=False) == "gemini-2.5-flash"
        assert calls == ["gemini-2.5-flash", "gemini-2.5-flash"]
        assert len(env) == 1 and 38.0 <= env[0] <= 40.0
        assert rv._unavailable_models() == {}


class TestFilesCalls:
    def test_task_none_ignores_daily_memory(self, env):
        for model in rv.CONFIG.all_models():
            rv._mark_unavailable(model, "daily free quota used up")
        fn, calls = failing_on({})
        assert rv.call_gemini(fn, task=None, throttle=False) == ""
        assert rv.quota_snapshot()["calls"] == {}


class TestFreeTierGuard:
    def test_paid_model_is_rejected(self, env):
        cfg = dataclasses.replace(rv.CONFIG, task_models={**CHAINS, "text": ("gemini-3.8-flash",)})
        problems = cfg.problems()
        assert any("gemini-3.8-flash" in p and "free tier" in p for p in problems)

    def test_default_chains_are_all_free(self):
        for _, default in rv.TASK_MODEL_ENV.values():
            assert set(default.split(",")) <= rv.FREE_TIER_MODELS

    def test_shut_down_model_is_dropped_from_chains(self, env, monkeypatch):
        assert "gemini-2.5-flash" in rv.CONFIG.models_for("analysis")
        monkeypatch.setattr(rv, "_pacific_now", lambda: datetime(2026, 12, 1, 0, 0, tzinfo=rv._PACIFIC))
        assert rv.CONFIG.models_for("analysis") == ["gemini-3.5-flash-lite"]
