"""Free-tier guard, per-task model chains and daily-quota memory in call_gemini."""
import dataclasses
import sys
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
    monkeypatch.setattr(rv._STATE, "cooldowns", {})
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


def busy_error():
    from google.genai.errors import ServerError
    return ServerError(503, {"error": {"code": 503, "status": "UNAVAILABLE",
                                       "message": "The model is overloaded. Please try again later."}})


def single_model_chain(monkeypatch):
    monkeypatch.setattr(rv, "CONFIG", dataclasses.replace(
        rv.CONFIG, task_models={**rv.CONFIG.task_models, "analysis": ("gemini-2.5-flash",)}))


class TestPerMinute:
    def test_switches_at_once_when_another_model_is_free(self, env):
        fn, calls = failing_on({"gemini-2.5-flash": [quota_error(PER_MINUTE, retry="37s")]})
        assert rv.call_gemini(fn, task="analysis", throttle=False) == "gemini-3.5-flash-lite"
        assert calls == ["gemini-2.5-flash", "gemini-3.5-flash-lite"] and env == []  # no waiting
        assert rv._cooling_models()["gemini-2.5-flash"] == pytest.approx(37, abs=1)
        assert rv._unavailable_models() == {}  # only a short cooldown, not "used up"

    def test_waits_google_retry_delay_when_no_other_model(self, env, monkeypatch):
        single_model_chain(monkeypatch)
        fn, calls = failing_on({"gemini-2.5-flash": [quota_error(PER_MINUTE, retry="37s")]})
        assert rv.call_gemini(fn, task="analysis", throttle=False) == "gemini-2.5-flash"
        assert calls == ["gemini-2.5-flash", "gemini-2.5-flash"]
        assert len(env) == 1 and 38.0 <= env[0] <= 40.0


class TestBusyModels:
    def test_one_quick_retry_on_the_same_model(self, env):
        fn, calls = failing_on({"gemini-2.5-flash": [busy_error()]})
        assert rv.call_gemini(fn, task="analysis", throttle=False) == "gemini-2.5-flash"
        assert calls == ["gemini-2.5-flash", "gemini-2.5-flash"]
        assert len(env) == 1 and env[0] <= 2.5

    def test_still_busy_switches_and_cools_down(self, env):
        fn, calls = failing_on({"gemini-2.5-flash": [busy_error(), busy_error()]})
        assert rv.call_gemini(fn, task="analysis", throttle=False) == "gemini-3.5-flash-lite"
        assert calls == ["gemini-2.5-flash", "gemini-2.5-flash", "gemini-3.5-flash-lite"]
        assert rv._cooling_models()["gemini-2.5-flash"] == pytest.approx(rv.COOLDOWN_BUSY_SECONDS, abs=2)

    def test_next_call_starts_on_a_model_that_is_answering(self, env):
        rv._cool_down("gemini-2.5-flash", 300)
        fn, calls = failing_on({})
        rv.call_gemini(fn, task="analysis", throttle=False)
        assert calls == ["gemini-3.5-flash-lite"]

    def test_cooldown_ends(self, env, monkeypatch):
        rv._cool_down("gemini-2.5-flash", 300)
        clock = rv.time.monotonic() + 301
        monkeypatch.setattr(rv.time, "monotonic", lambda: clock)
        fn, calls = failing_on({})
        rv.call_gemini(fn, task="analysis", throttle=False)
        assert calls == ["gemini-2.5-flash"]

    def test_every_model_busy_backs_off_instead_of_failing(self, env):
        fn, calls = failing_on({"gemini-2.5-flash": [busy_error()] * 2, "gemini-3.5-flash-lite": [busy_error()] * 2})
        # Last model standing: it backs off (5 s, 10 s) on that model until it answers.
        assert rv.call_gemini(fn, task="analysis", throttle=False) == "gemini-3.5-flash-lite"
        assert calls == ["gemini-2.5-flash", "gemini-2.5-flash", "gemini-3.5-flash-lite",
                         "gemini-3.5-flash-lite", "gemini-3.5-flash-lite"]
        assert env[0] <= 2.5 and min(env[1:]) >= 5

    def test_sidebar_shows_cooling_models(self, env):
        rv._cool_down("gemini-2.5-flash", 120)
        assert "gemini-2.5-flash" in rv.quota_snapshot()["cooling"]


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


class TestModelOrder:
    def test_newest_working_model_leads_and_slow_lite_is_last(self):
        chain = rv.DEFAULT_MODEL_CHAIN.split(",")
        assert chain[0] == "gemini-3.6-flash" and chain[-1] == "gemini-3.1-flash-lite"
        assert "gemini-3.8-flash" not in chain and "gemini-3.7-flash" not in chain

    def test_search_steps_start_on_models_with_free_search(self):
        chain = rv.SEARCH_MODEL_CHAIN.split(",")
        assert set(chain[:2]) == rv.FREE_SEARCH_MODELS
        assert rv.TASK_MODEL_ENV["metadata"][1] == rv.TASK_MODEL_ENV["story"][1] == rv.SEARCH_MODEL_CHAIN

    def test_every_default_model_is_accepted(self):
        for _, default in rv.TASK_MODEL_ENV.values():
            assert set(default.split(",")) <= rv.FREE_TIER_MODELS


class TestStaleModules:
    @pytest.fixture(autouse=True)
    def restore_app_modules(self, monkeypatch):
        """drop_stale_app_modules removes the app modules from sys.modules;
        registering them first makes pytest put every one of them back, so
        later tests keep using the same module objects."""
        for name in rv.APP_MODULES:
            if name in sys.modules:
                monkeypatch.setitem(sys.modules, name, sys.modules[name])
            else:
                monkeypatch.delitem(sys.modules, name, raising=False)

    def test_changed_files_are_reimported(self, monkeypatch):
        import types
        stale = types.ModuleType("summary_short")
        stale.__file__ = rv.__file__
        stale._LOADED_MTIME = 0.0  # loaded before the file last changed
        monkeypatch.setitem(sys.modules, "summary_short", stale)
        assert "summary_short" in rv.drop_stale_app_modules()
        assert "summary_short" not in sys.modules

    def test_nothing_dropped_when_up_to_date(self, monkeypatch):
        for name in rv.APP_MODULES:
            monkeypatch.delitem(sys.modules, name, raising=False)
        monkeypatch.setitem(sys.modules, "recreate_video", rv)
        monkeypatch.setattr(rv, "_LOADED_MTIME", rv.Path(rv.__file__).stat().st_mtime)
        assert rv.drop_stale_app_modules() == []


class TestYoutubeFormat:
    def test_limit_is_on_the_short_side(self):
        fmt = rv.video_format(720)
        assert "[height<=1280][width<=1280]" in fmt and "[height<=720]" not in fmt
