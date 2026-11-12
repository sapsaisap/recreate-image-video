"""
recreate_video.py -- standalone "any video -> recreation prompt" tool.

Paste one public video URL (YouTube, Shorts, Instagram, TikTok, X, Facebook,
or a direct .mp4 link, max 60 s). The video is downloaded, validated, and
analysed by Gemini (native video understanding: picture + sound in one call),
then -- after the user picks an output length (8 s or 10 s) -- Gemini condenses
it into a timed, scene-by-scene prompt that recreates the video. No video is
generated; the output is the prompt only.

Works on any kind of video (kids, animation, viral, vlog, tutorial, ads, ...),
not just ads. Written for a Google AI Studio *free-tier* key: every Gemini
call goes through call_gemini(), which throttles to GEMINI_RPM and, on a
rate limit, waits (honouring Google's retryDelay) and retries instead of
failing. The only limit it cannot wait out is a used-up daily quota.

Logic is adapted from adsgpt/python-backend/advideo/models/{omni_model,
seedance_v2_5} (URL download, retry pattern, timed prompt layout) but
nothing is imported from there -- this file is fully standalone and talks
to Gemini directly with GEMINI_API_KEY (no Gemini Central proxy).

Run:
    streamlit run recreate_video.py                     # UI
    python recreate_video.py --url <URL> --duration 8   # CLI
Config lives in .env next to this file (see .env.example).
"""

from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import logging
import os
import random
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import types
import unicodedata
import uuid
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, TypeVar
from urllib.parse import urljoin, urlsplit
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from dotenv import load_dotenv
from pydantic import BaseModel, Field

APP_DIR = Path(__file__).resolve().parent
load_dotenv(APP_DIR / ".env")

# ── Streamlit Cloud secret injection ────────────────────────────────────────
# On Streamlit Community Cloud there is no .env file; secrets are stored in
# the app's "Secrets" settings and exposed via st.secrets. We inject them into
# os.environ here so the rest of the code (os.getenv) works unchanged.
def _inject_streamlit_secrets() -> None:
    try:
        import streamlit as _st
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        if get_script_run_ctx() is None:
            return
        secrets = _st.secrets
        # Without a secrets.toml, `key in st.secrets` draws an st.error on the
        # page, which then breaks st.set_page_config(). Check quietly first.
        load_quietly = getattr(secrets, "load_if_toml_exists", None)
        if load_quietly is not None and not load_quietly():
            return
        for key in (
            "GEMINI_API_KEY", "COOKIES_FILE", "DOWNLOAD_PROXY", "GEMINI_RPM",
            "GEMINI_ANALYSIS_MODELS", "GEMINI_TEXT_MODELS", "GEMINI_METADATA_MODELS", "GEMINI_STORY_MODELS",
            "SUMMARY_MAX_VIDEO_SECONDS",
            "DIALOGUE_SCRIPT", "METADATA_LANGUAGE", "URL_ALLOWLIST",
        ):
            if key in secrets and not os.environ.get(key):
                os.environ[key] = str(secrets[key])

        # If full cookies.txt content is provided in Streamlit Cloud secrets
        for cookie_key in ("COOKIES_CONTENT", "COOKIES_TEXT", "COOKIES_DATA"):
            if cookie_key in secrets:
                cookie_content = str(secrets[cookie_key]).strip()
                if cookie_content:
                    cloud_cookies_path = APP_DIR / "cookies.txt"
                    cloud_cookies_path.write_text(cookie_content, encoding="utf-8")
                    os.environ["COOKIES_FILE"] = str(cloud_cookies_path)
                    break
    except Exception:
        pass  # Not running in Streamlit, or secrets not configured — skip.

_inject_streamlit_secrets()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("recreate_video")
_LOADED_MTIME = Path(__file__).stat().st_mtime  # see recreate_video.drop_stale_app_modules

T = TypeVar("T")
Notify = Callable[[str], None]


def _noop_notify(_: str) -> None:
    pass


# ============================================================================
# Config
# ============================================================================

def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "").strip() or default)
    except ValueError:
        return default


def _env_list(name: str, default: str) -> list[str]:
    return [item.strip() for item in os.getenv(name, default).split(",") if item.strip()]


try:
    _PACIFIC: Any = ZoneInfo("America/Los_Angeles")
except ZoneInfoNotFoundError:  # Windows without the tzdata package
    _PACIFIC = timezone(timedelta(hours=-8))


def _pacific_now() -> datetime:
    """Gemini's daily quotas reset at midnight Pacific time."""
    return datetime.now(_PACIFIC)


def _seconds_until_quota_reset() -> int:
    now = _pacific_now()
    midnight = datetime.combine(now.date() + timedelta(days=1), datetime.min.time(), tzinfo=now.tzinfo)
    return max(0, int((midnight - now).total_seconds()))


# Models a free-tier key can use (each one answered on a free key, 2026-11-12).
# gemini-3.8-flash and gemini-3.7-flash are left out: on the free key they
# hung for 60-110 s and then answered 503 "high demand". gemini-2.5-pro is
# gone for new users (404). Add a model only once it answers on a free key.
FREE_TIER_MODELS = frozenset({
    "gemini-3.6-flash", "gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-3.1-flash-lite",
    "gemini-2.5-flash", "gemini-2.5-flash-lite",
})

# Shutdown dates from the deprecations page. From that date the model is
# dropped from every chain: Google routes its requests to another model, so
# keeping it would only spend that other model's quota twice.
MODEL_SHUTDOWN = {"gemini-2.5-flash": date(2026, 12, 1)}

# Which models each kind of Gemini call tries, in order. call_gemini starts
# with the first model that is not used up today and not cooling down after
# a "busy" / timeout, so a struggling model costs seconds, not minutes.
#   analysis: watches the video / looks at the image
#   text:     writes the recreation prompt from the analysis JSON
#   metadata: writes the YouTube / Instagram metadata (Google Search)
#   story:    Summarise to Short's fact-checked script (Google Search)
# Newest full model first; gemini-3.1-flash-lite last (works, but took ~25 s
# for a one-word reply). Steps that use Google Search start on the 2.5
# models, the only ones whose free tier includes it. gemini-2.5-flash drops
# out automatically at its shutdown on 2026-12-01.
DEFAULT_MODEL_CHAIN = (
    "gemini-3.6-flash,gemini-3.5-flash,gemini-2.5-flash,gemini-3.5-flash-lite,gemini-2.5-flash-lite,"
    "gemini-3.1-flash-lite"
)
SEARCH_MODEL_CHAIN = (
    "gemini-2.5-flash,gemini-2.5-flash-lite,gemini-3.6-flash,gemini-3.5-flash,gemini-3.5-flash-lite,"
    "gemini-3.1-flash-lite"
)
TASK_MODEL_ENV = {
    "analysis": ("GEMINI_ANALYSIS_MODELS", DEFAULT_MODEL_CHAIN),
    "text": ("GEMINI_TEXT_MODELS", DEFAULT_MODEL_CHAIN),
    "metadata": ("GEMINI_METADATA_MODELS", SEARCH_MODEL_CHAIN),
    "story": ("GEMINI_STORY_MODELS", SEARCH_MODEL_CHAIN),
}

# Per-request time limits (seconds). A request that hangs fails at this
# point and the next model is tried, instead of waiting ~90 s or more.
TIMEOUT_TEXT_SECONDS = 90  # a full recreation plan with thinking can take ~60 s
TIMEOUT_SEARCH_SECONDS = 90  # Google Search grounding adds a few searches
TIMEOUT_IMAGE_SECONDS = 90
TIMEOUT_VIDEO_SECONDS = 180
TIMEOUT_LONG_VIDEO_SECONDS = 300  # Summarise to Short: sources up to 5 minutes
# Metadata needs little reasoning: a small thinking budget on gemini-2.5-flash
# makes it noticeably faster.
METADATA_THINKING_BUDGET = 512


def gemini_http_options(timeout_seconds: int) -> dict:
    """http_options for one generate_content call (google-genai wants ms)."""
    return {"api_version": "v1beta", "timeout": int(timeout_seconds * 1000)}


@dataclass(frozen=True)
class Config:
    api_key: str = os.getenv("GEMINI_API_KEY", "").strip()
    task_models: dict[str, tuple[str, ...]] = field(default_factory=lambda: {
        task: tuple(_env_list(env_name, default)) for task, (env_name, default) in TASK_MODEL_ENV.items()
    })
    rpm: int = max(1, _env_int("GEMINI_RPM", 8))
    max_wait_seconds: int = max(30, _env_int("RATE_LIMIT_MAX_WAIT_SECONDS", 900))
    max_video_seconds: int = _env_int("MAX_VIDEO_SECONDS", 60)
    # Summarise to Short takes longer sources (YouTube / Instagram up to 5 min).
    summary_max_video_seconds: int = _env_int("SUMMARY_MAX_VIDEO_SECONDS", 300)
    min_video_seconds: int = _env_int("MIN_VIDEO_SECONDS", 1)
    max_download_mb: int = _env_int("MAX_DOWNLOAD_MB", 200)
    output_durations: tuple[int, ...] = tuple(
        int(value) for value in _env_list("OUTPUT_DURATIONS", "8,10") if value.isdigit()
    ) or (8, 10)
    work_dir: Path = (APP_DIR / os.getenv("WORK_DIR", "work")).resolve()
    cookies_file: str = os.getenv("COOKIES_FILE", "").strip() or (
        str(APP_DIR / "cookies.txt") if (APP_DIR / "cookies.txt").is_file() else ""
    )
    download_proxy: str = os.getenv("DOWNLOAD_PROXY", "").strip()
    # Optional: only accept links from these sites (and their subdomains).
    # Empty = any public website. Recommended for a public deployment.
    url_allowlist: tuple[str, ...] = tuple(
        domain.lower().lstrip(".") for domain in _env_list("URL_ALLOWLIST", "")
    )
    # The free tier can only produce text, so the video/image prompts ARE the
    # output. true: show them on screen; false: only save + offer download.
    show_prompt: bool = os.getenv("SHOW_PROMPT", "true").strip().lower() not in {"false", "0", "no", "off"}
    # How dialogue is written in the video prompt: "native" (Telugu script,
    # Devanagari, ...) or "romanized" (Latin letters), for generators that
    # pronounce one better than the other.
    dialogue_script: str = os.getenv("DIALOGUE_SCRIPT", "native").strip().lower()
    # Language of the YouTube / Instagram metadata for non-English content:
    # "auto" (English mixed with romanized Telugu/Hindi, as viewers search),
    # "english" or "native".
    metadata_language: str = os.getenv("METADATA_LANGUAGE", "auto").strip().lower()

    def models_for(self, task: str) -> list[str]:
        """The model chain for one kind of call, minus models already shut down."""
        today = _pacific_now().date()
        return [
            model for model in dict.fromkeys(self.task_models[task])
            if model not in MODEL_SHUTDOWN or today < MODEL_SHUTDOWN[model]
        ]

    def all_models(self) -> list[str]:
        return list(dict.fromkeys(model for task in self.task_models for model in self.models_for(task)))

    def problems(self) -> list[str]:
        issues = []
        if not self.api_key:
            issues.append(
                "GEMINI_API_KEY is not set. "
                "Add it to .env (local) or App Settings → Secrets (Streamlit Cloud)."
            )
        for task, (env_name, _) in TASK_MODEL_ENV.items():
            paid = [model for model in self.task_models[task] if model not in FREE_TIER_MODELS]
            if paid:
                issues.append(
                    f"{env_name} lists {', '.join(paid)}, which the Gemini free tier does not offer. "
                    f"Free-tier models: {', '.join(sorted(FREE_TIER_MODELS))}."
                )
            elif not self.models_for(task):
                issues.append(f"Every model in {env_name} has been shut down -- add a current free-tier model.")
        if self.cookies_file and not Path(self.cookies_file).is_file():
            issues.append(f"COOKIES_FILE points to a missing file: {self.cookies_file}")
        return issues


CONFIG = Config()

for _legacy in ("GEMINI_MODEL", "GEMINI_FALLBACK_MODELS"):
    if os.getenv(_legacy, "").strip():
        logger.warning(
            f"[CONFIG] {_legacy} is no longer used -- set GEMINI_ANALYSIS_MODELS, "
            "GEMINI_TEXT_MODELS and GEMINI_METADATA_MODELS instead (see .env.example)."
        )


# ============================================================================
# Data cleanup: 24-hour startup sweep
# ============================================================================

def _sweep_old_files(max_age_seconds: float = 86400) -> None:
    """Delete all files and empty dirs in work_dir older than max_age_seconds.
    Runs once at startup so stale downloads/prompts never accumulate."""
    work = CONFIG.work_dir
    if not work.exists():
        return
    cutoff = time.time() - max_age_seconds
    for item in sorted(work.rglob("*"), reverse=True):  # deepest first
        try:
            if item.is_file() and item.stat().st_mtime < cutoff:
                item.unlink(missing_ok=True)
                logger.info(f"[CLEANUP] deleted old file: {item.name}")
            elif item.is_dir() and not any(item.iterdir()):
                item.rmdir()
        except OSError:
            pass


# ============================================================================
# URL type detection (pure regex, no network call)
# ============================================================================

class MediaType:
    VIDEO = "video"
    IMAGE = "image"
    AMBIGUOUS = "ambiguous"  # Instagram /p/ without img_index


def host_is(host: str | None, domain: str) -> bool:
    """True for `domain` itself and its subdomains -- never for a host that
    merely contains it (evilinstagram.com, instagram.com.evil.net, netflix.com
    for "x.com")."""
    host = (host or "").lower().rstrip(".")
    return host == domain or host.endswith("." + domain)


def detect_url_type(url: str) -> str:
    """Classify a URL as video, image, or ambiguous without any network call."""
    parsed = urlsplit((url or "").strip())
    host = (parsed.hostname or "").lower()
    path = (parsed.path or "").lower()
    query = (parsed.query or "").lower()
    suffix = Path(path).suffix.lower()

    # Direct file extensions
    if suffix in {".mp4", ".mov", ".mkv", ".webm", ".m4v", ".avi"}:
        return MediaType.VIDEO
    if suffix in {".jpg", ".jpeg", ".png", ".webp", ".gif", ".avif"}:
        return MediaType.IMAGE

    # YouTube
    if host_is(host, "youtube.com") or host_is(host, "youtu.be"):
        return MediaType.VIDEO

    # TikTok
    if host_is(host, "tiktok.com"):
        return MediaType.VIDEO

    # Twitter / X
    if host_is(host, "twitter.com") or host_is(host, "x.com"):
        return MediaType.VIDEO

    # Facebook
    if host_is(host, "facebook.com") or host_is(host, "fb.watch"):
        return MediaType.VIDEO

    # Instagram
    if host_is(host, "instagram.com"):
        if "/reel/" in path or "/tv/" in path:
            return MediaType.VIDEO
        if "/p/" in path:
            # img_index param → definitely image carousel
            if "img_index" in query:
                return MediaType.IMAGE
            # bare /p/ → could be image or video Reel
            return MediaType.AMBIGUOUS
        if "/stories/" in path:
            return MediaType.VIDEO

    return MediaType.VIDEO  # default: try as video


# ============================================================================
# Instagram image downloader (requests-based, bypasses yt-dlp 400 errors)
# ============================================================================

# Instagram serves post images from these CDNs. The image URL comes from the
# fetched page, so anything else is refused rather than downloaded.
INSTAGRAM_CDN_DOMAINS = ("cdninstagram.com", "fbcdn.net")
_INSTAGRAM_PAGE_MAX_BYTES = 5 * 1024 * 1024


def _load_cookie_jar(cookie_file: str):
    """cookies.txt as a cookie jar that keeps each cookie's domain, so requests
    only sends a cookie to the site it belongs to. yt-dlp's jar also reads the
    '#HttpOnly_' lines exporters use for login cookies such as sessionid."""
    from yt_dlp.cookies import YoutubeDLCookieJar

    jar = YoutubeDLCookieJar(cookie_file)
    try:
        jar.load()
    except OSError as exc:  # includes http.cookiejar.LoadError
        logger.warning(f"[COOKIES] could not read {cookie_file}: {exc}")
        return None
    return jar


def _has_cookies_for(jar, domain: str) -> bool:
    return jar is not None and any(host_is(cookie.domain.lstrip("."), domain) for cookie in jar)


@dataclass
class DownloadedImage:
    path: Path
    title: str
    source_url: str


def download_instagram_image(
    url: str,
    destination: Path,
    notify: Notify = _noop_notify,
) -> DownloadedImage:
    """Download an Instagram image post using requests + cookies.txt.
    Extracts og:image URL from the HTML page — avoids yt-dlp's HTTP 400 errors."""
    safe_url = validate_public_video_url(url)
    if not host_is(urlsplit(safe_url).hostname, "instagram.com"):
        raise URLVideoError("This is not an Instagram link.")

    # Load cookies for Instagram login
    cookies = _load_cookie_jar(CONFIG.cookies_file) if CONFIG.cookies_file else None
    if not _has_cookies_for(cookies, "instagram.com"):
        raise URLVideoError(
            "Instagram requires a logged-in session. "
            "Export your Instagram cookies to cookies.txt and set COOKIES_FILE in .env."
        )

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Referer": "https://www.instagram.com/",
    }

    notify("Fetching Instagram post page...")
    try:
        resp = safe_get(
            safe_url, max_bytes=_INSTAGRAM_PAGE_MAX_BYTES, timeout=20, headers=headers,
            cookies=cookies, allowed_domains=("instagram.com",),
        )
    except URLVideoError as exc:
        raise URLVideoError(f"Could not fetch Instagram page: {exc}") from exc

    # Extract og:image (the highest-quality image URL in the page)
    img_url: str | None = None
    og_match = re.search(r'<meta[^>]+property=["\']og:image["\'][^>]+content=["\']([^\'"]+)["\']', resp.text)
    if not og_match:
        og_match = re.search(r'<meta[^>]+content=["\']([^\'"]+)["\'][^>]+property=["\']og:image["\']', resp.text)
    if og_match:
        img_url = og_match.group(1)
    # Fallback: JSON-LD
    if not img_url:
        ld_match = re.search(r'"thumbnailUrl"\s*:\s*"([^"]+)"', resp.text)
        if ld_match:
            img_url = ld_match.group(1)
    if not img_url:
        raise URLVideoError(
            "Could not find the image in this Instagram post. "
            "The post may be private, deleted, or requires a fresh login session."
        )

    # Decode HTML entities in the URL (&amp; → &)
    img_url = img_url.replace("&amp;", "&")

    notify("Downloading image from Instagram CDN...")
    try:
        img_resp = safe_get(
            img_url, max_bytes=CONFIG.max_download_mb * 1024 * 1024,
            allowed_domains=INSTAGRAM_CDN_DOMAINS,
        )
    except URLVideoError as exc:
        raise URLVideoError(f"Image download failed: {exc}") from exc

    # Determine extension from content-type
    ctype = img_resp.headers.get("Content-Type", "image/jpeg").lower()
    ext = ".jpg"
    if "png" in ctype:
        ext = ".png"
    elif "webp" in ctype:
        ext = ".webp"

    destination.mkdir(parents=True, exist_ok=True)
    out_path = destination / f"source{ext}"
    out_path.write_bytes(img_resp.content)

    # Extract title from og:title
    title_match = re.search(r'<meta[^>]+property=["\']og:title["\'][^>]+content=["\']([^\'"]+)["\']', resp.text)
    if not title_match:
        title_match = re.search(r'<meta[^>]+content=["\']([^\'"]+)["\'][^>]+property=["\']og:title["\']', resp.text)
    title = title_match.group(1) if title_match else "Instagram post"

    size_kb = len(img_resp.content) / 1024
    logger.info(f"[IG-IMAGE] downloaded {out_path.name} ({size_kb:.0f} KB) from Instagram CDN")
    return DownloadedImage(path=out_path, title=title, source_url=safe_url)


# Streamlit re-executes this script in a fresh namespace on every rerun, so
# anything that must survive across reruns (the throttle clock, the shared
# Gemini client) lives in a stable stash module instead of module globals.
_STATE = sys.modules.get("_recreate_video_state")
if _STATE is None:
    _STATE = types.ModuleType("_recreate_video_state")
    _STATE.lock = threading.Lock()
    _STATE.last_call = 0.0
    _STATE.client = None
    sys.modules["_recreate_video_state"] = _STATE
if not hasattr(_STATE, "quota_lock"):  # stash created by an older version of this file
    _STATE.quota_lock = threading.Lock()
    _STATE.quota = None
if not hasattr(_STATE, "cooldowns"):
    _STATE.cooldowns = {}  # model -> time.monotonic() until which it is set aside


def get_client():
    from google import genai

    # If an existing client in cache does not have api_version='v1beta', drop it
    client = getattr(_STATE, "client", None)
    if client is not None:
        try:
            if getattr(client._api_client._http_options, "api_version", None) != "v1beta":
                _STATE.client = None
        except Exception:
            _STATE.client = None

    if _STATE.client is None:
        if not CONFIG.api_key:
            raise RuntimeError("GEMINI_API_KEY is not configured.")
        _STATE.client = genai.Client(
            api_key=CONFIG.api_key,
            http_options={"api_version": "v1beta"},
        )
    return _STATE.client


# ============================================================================
# Rate-limit-safe Gemini wrapper
# ============================================================================

class GeminiCallError(RuntimeError):
    """A Gemini failure that waiting/retrying cannot fix (bad key, safety
    block, invalid request) or that outlasted RATE_LIMIT_MAX_WAIT_SECONDS."""


class QuotaExhaustedError(RuntimeError):
    """Every model in a step's chain is used up (or unusable) for today."""


# ── Daily free-tier quota memory ────────────────────────────────────────────
# Which models are used up today, plus this app's call count per model. Shared
# by every session through _STATE and kept on disk, so neither a new session,
# a restart nor the CLI re-spends a request on a model that is already out.
# The record is dropped when the Pacific date changes (= Google's reset).

def _quota_state_path() -> Path:
    return CONFIG.work_dir / "cache" / "quota_state.json"


def _quota_state() -> dict:
    """Today's record. Call with _STATE.quota_lock held."""
    today = _pacific_now().date().isoformat()
    state = _STATE.quota
    if state is None:
        try:
            state = json.loads(_quota_state_path().read_text(encoding="utf-8"))
        except (OSError, ValueError):
            state = {}
    if not isinstance(state, dict) or state.get("day") != today:
        state = {"day": today, "calls": {}, "unavailable": {}}
    _STATE.quota = state
    return state


def _save_quota_state(state: dict) -> None:
    try:
        _quota_state_path().parent.mkdir(parents=True, exist_ok=True)
        _quota_state_path().write_text(json.dumps(state, indent=2), encoding="utf-8")
    except OSError as exc:
        logger.warning(f"[QUOTA] could not save quota state: {exc}")


def _record_call(model: str) -> None:
    with _STATE.quota_lock:
        state = _quota_state()
        state["calls"][model] = state["calls"].get(model, 0) + 1
        _save_quota_state(state)


def _mark_unavailable(model: str, reason: str) -> None:
    with _STATE.quota_lock:
        state = _quota_state()
        state["unavailable"][model] = reason
        _save_quota_state(state)


def _unavailable_models() -> dict[str, str]:
    with _STATE.quota_lock:
        return dict(_quota_state()["unavailable"])


def _search_used_up() -> bool:
    """Today's free Google Search grounding quota is spent. Tracked apart from
    the models: the model itself still answers without search."""
    with _STATE.quota_lock:
        return bool(_quota_state().get("search_used_up"))


def _mark_search_used_up() -> None:
    with _STATE.quota_lock:
        state = _quota_state()
        state["search_used_up"] = True
        _save_quota_state(state)


# ── Short-term cooldowns ────────────────────────────────────────────────────
# A model that is "busy" (503), timed out or hit a per-minute limit is set
# aside for a while, for every session, so the next request starts on a
# model that is answering. In memory only: a restart forgets it.
COOLDOWN_BUSY_SECONDS = 300
QUICK_RETRY_SECONDS = 2.0


def _cooling_models() -> dict[str, float]:
    """Models still cooling down -> seconds left."""
    now = time.monotonic()
    with _STATE.quota_lock:
        return {model: until - now for model, until in _STATE.cooldowns.items() if until > now}


def _cool_down(model: str, seconds: float) -> None:
    with _STATE.quota_lock:
        _STATE.cooldowns[model] = max(_STATE.cooldowns.get(model, 0.0), time.monotonic() + seconds)


def _clear_cooldown(model: str) -> None:
    with _STATE.quota_lock:
        _STATE.cooldowns.pop(model, None)


def quota_snapshot() -> dict:
    """For the UI: today's calls, unusable and cooling models, time to the reset."""
    cooling = _cooling_models()
    with _STATE.quota_lock:
        state = _quota_state()
        return {
            "calls": dict(state["calls"]),
            "unavailable": dict(state["unavailable"]),
            "cooling": cooling,
            "search_used_up": bool(state.get("search_used_up")),
            "reset_in_seconds": _seconds_until_quota_reset(),
        }


def _format_reset_wait() -> str:
    hours, minutes = divmod(_seconds_until_quota_reset() // 60, 60)
    return f"{hours}h {minutes}m"


class _RetryableResponse(RuntimeError):
    """A 200 response whose body was empty/unparsable -- usually transient."""


def _throttle() -> None:
    """Keeps at least 60/GEMINI_RPM seconds between Gemini calls so the
    free-tier per-minute limit is rarely hit in the first place."""
    interval = 60.0 / CONFIG.rpm
    with _STATE.lock:
        wait = _STATE.last_call + interval - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _STATE.last_call = time.monotonic()


def _error_code(exc: Exception) -> int | None:
    code = getattr(exc, "code", None)
    if isinstance(code, int):
        return code
    match = re.match(r"\s*(\d{3})\b", str(exc))
    return int(match.group(1)) if match else None


def _error_details(exc: Exception) -> list[dict]:
    """The google.rpc detail objects (QuotaFailure, RetryInfo, ...) of a Gemini
    API error. google-genai keeps the whole response body in exc.details."""
    payload = getattr(exc, "details", None)
    if isinstance(payload, dict):
        body = payload.get("error", payload)
        payload = body.get("details") if isinstance(body, dict) else None
    return [item for item in payload if isinstance(item, dict)] if isinstance(payload, list) else []


def _quota_violations(exc: Exception) -> list[dict]:
    return [
        violation
        for detail in _error_details(exc)
        if str(detail.get("@type", "")).endswith("QuotaFailure")
        for violation in detail.get("violations") or []
        if isinstance(violation, dict)
    ]


def _retry_delay_seconds(exc: Exception) -> float | None:
    """Google's 429 body carries a RetryInfo detail ("retryDelay": "37s") and
    usually also says "Please retry in 37.2s" in the message."""
    for detail in _error_details(exc):
        if str(detail.get("@type", "")).endswith("RetryInfo"):
            match = re.match(r"(\d+(?:\.\d+)?)s", str(detail.get("retryDelay", "")))
            if match:
                return float(match.group(1))
    text = str(getattr(exc, "details", "") or "") + " " + str(exc)
    for pattern in (
        r"retryDelay['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)s",
        r"retry in (\d+(?:\.\d+)?)\s*s",
    ):
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return float(match.group(1))
    return None


def _is_daily_quota(exc: Exception) -> bool:
    """quotaId is e.g. "GenerateRequestsPerDayPerProjectPerModel-FreeTier".
    The message text is only checked when the error carries no QuotaFailure."""
    violations = _quota_violations(exc)
    if violations:
        return any("perday" in str(v.get("quotaId", "")).lower() for v in violations)
    text = str(getattr(exc, "details", "") or "") + " " + str(exc)
    return bool(re.search(r"per ?day", text, flags=re.IGNORECASE))


def _is_not_on_free_tier(exc: Exception) -> bool:
    """A quota of 0 means this key's tier has no access to the model at all."""
    if any(str(v.get("quotaValue", "")).strip() == "0" for v in _quota_violations(exc)):
        return True
    return bool(re.search(r"\blimit:\s*0\b", str(exc)))


def _is_search_quota(exc: Exception) -> bool:
    """A 429 about the Google Search grounding quota, not the model's own."""
    if _classify(exc) != "rate":
        return False
    violations = _quota_violations(exc)
    if violations:
        return any(
            word in f"{v.get('quotaId', '')} {v.get('quotaMetric', '')}".lower()
            for v in violations for word in ("search", "grounding")
        )
    return "grounding" in str(exc).lower()


def _is_model_missing(exc: Exception) -> bool:
    """404 "models/... is not found": renamed, retired or never existed."""
    return _error_code(exc) == 404


def _unusable_reason(exc: Exception) -> str | None:
    """Why a model cannot serve any more requests today, or None if waiting
    and retrying the same model can still work."""
    if _classify(exc) == "rate":
        if _is_not_on_free_tier(exc):
            return "not available on the free tier"
        if _is_daily_quota(exc):
            return "daily free quota used up"
    elif _is_model_missing(exc):
        return "model not found"
    return None


def _classify(exc: Exception) -> str:
    if isinstance(exc, _RetryableResponse):
        return "transient"
    code = _error_code(exc)
    text = str(exc).upper()
    if code == 429 or "RESOURCE_EXHAUSTED" in text:
        return "rate"
    if code in {500, 502, 503, 504} or any(
        marker in text for marker in ("UNAVAILABLE", "OVERLOADED", "TIMEOUT", "TIMED OUT", "DEADLINE")
    ):
        return "transient"
    if isinstance(exc, (ConnectionError, TimeoutError)) or type(exc).__module__.startswith(("httpx", "httpcore")):
        return "transient"
    return "fatal"


def _readable_error(exc: Exception) -> str:
    code = _error_code(exc)
    message = getattr(exc, "message", None) or " ".join(str(exc).split())
    if code in {400, 403} and "API key" in str(exc):
        return "Gemini rejected the API key. Check GEMINI_API_KEY in .env."
    if code == 404:
        return f"Gemini model not found or not available on this key: {message[:300]}"
    return f"Gemini request failed: {message[:400]}"


def _wait_with_countdown(seconds: float, reason: str, notify: Notify) -> None:
    end = time.monotonic() + seconds
    while True:
        remaining = end - time.monotonic()
        if remaining <= 0:
            return
        notify(f"{reason} -- retrying in {int(remaining) + 1}s")
        time.sleep(min(1.0, remaining))


def call_gemini(
    fn: Callable[[str], T],
    notify: Notify = _noop_notify,
    *,
    task: str | None,
    label: str = "Gemini",
    throttle: bool = True,
) -> T:
    """Runs fn(model_name) until it succeeds, walking the free-tier model
    chain of `task` ("analysis", "text", "metadata", "story"; TASK_MODEL_ENV).

    - Models used up today are skipped without sending a request (shared by
      all sessions until the daily reset at midnight Pacific). Models cooling
      down after "busy" / timeouts / per-minute limits go to the back.
    - 429 per-day limit, 429 "limit: 0" (model not on the free tier) or model
      not found: mark the model unusable for today and move on;
      QuotaExhaustedError once the chain is empty.
    - 429 per-minute limit: if another model is available, cool this one
      down for Google's retryDelay and switch at once; otherwise wait it out.
    - 5xx / overloaded / timeout: one quick retry (~2 s) on the same model,
      then cool it down for COOLDOWN_BUSY_SECONDS and switch. Only when no
      other model is left does it back off (5/10/20/40/60 s).
    - Anything else (bad key, bad request, safety block): GeminiCallError now.

    task=None is for Files API calls: they use no model and no daily quota,
    so fn gets "" and only the per-minute / transient handling applies.
    """
    chain = CONFIG.models_for(task) if task else [""]
    passed_over: set[str] = set()  # set aside for the rest of this call
    current = None
    failures_on_model = 0
    deadline = time.monotonic() + CONFIG.max_wait_seconds

    while True:
        unusable = _unavailable_models() if task else {}
        available = [m for m in chain if m not in unusable]
        if not available:
            details = "; ".join(f"{m}: {unusable.get(m, 'unavailable')}" for m in chain)
            raise QuotaExhaustedError(
                f"No free-tier Gemini model is left for this step today ({details}). "
                f"The free quota resets at midnight Pacific time, in about {_format_reset_wait()}."
            )
        # Every model already set aside in this call: keep going with backoff.
        candidates = [m for m in available if m not in passed_over] or available
        if task:
            cooling = _cooling_models()
            candidates = [m for m in candidates if m not in cooling] + [m for m in candidates if m in cooling]
        model = candidates[0]
        if model != current:
            if current is not None:
                notify(f"Switching to {model}")
            current = model
            failures_on_model = 0
        if throttle:
            _throttle()
        if task:
            _record_call(model)
        try:
            result = fn(model)
            if task:
                _clear_cooldown(model)
            return result
        except (GeminiCallError, QuotaExhaustedError):
            raise
        except Exception as exc:
            kind = _classify(exc)
            reason = _unusable_reason(exc) if task else None
            if reason:
                logger.warning(f"[GEMINI] {label} model={model} {reason}: {exc}")
                _mark_unavailable(model, reason)
                notify(f"{model}: {reason}")
                continue
            if kind == "fatal":
                logger.error(f"[GEMINI] {label} model={model} fatal: {exc}")
                raise GeminiCallError(_readable_error(exc)) from exc

            failures_on_model += 1
            can_switch = bool(task) and any(m != model and m not in passed_over for m in available)
            if can_switch and kind == "rate":
                _cool_down(model, _retry_delay_seconds(exc) or 30.0)
                passed_over.add(model)
                logger.warning(f"[GEMINI] {label} model={model} per-minute limit, switching: {exc}")
                notify(f"{model} is rate-limited -- using the next model")
                continue
            if can_switch and kind == "transient" and failures_on_model >= 2:
                _cool_down(model, COOLDOWN_BUSY_SECONDS)
                passed_over.add(model)
                logger.warning(f"[GEMINI] {label} model={model} busy, switching: {exc}")
                notify(f"{model} is busy -- using the next model")
                continue

            delay = _retry_delay_seconds(exc) if kind == "rate" else None
            if delay is None and can_switch:  # first transient failure: one quick retry
                delay = QUICK_RETRY_SECONDS + random.uniform(0.0, 0.5)
            elif delay is None:
                delay = min(60.0, 5.0 * 2 ** (failures_on_model - 1)) + random.uniform(1.0, 3.0)
            else:
                delay += random.uniform(1.0, 3.0)
            remaining = deadline - time.monotonic()
            if delay > remaining:
                raise GeminiCallError(
                    f"Gemini was still rate-limited/unavailable after waiting "
                    f"{CONFIG.max_wait_seconds}s. Try again in a few minutes, or raise "
                    "RATE_LIMIT_MAX_WAIT_SECONDS."
                ) from exc
            wait_reason = (
                f"Free-tier rate limit hit on {model}"
                if kind == "rate"
                else f"Gemini busy/unavailable on {model}"
            )
            logger.warning(f"[GEMINI] {label} model={model} {kind} failure, waiting {delay:.1f}s: {exc}")
            _wait_with_countdown(delay, wait_reason, notify)


def _parse_structured(response: Any, schema: type[BaseModel]) -> BaseModel:
    """response.parsed when the SDK filled it, otherwise validate the raw
    text. A safety block is fatal; an empty/garbled body is retried."""
    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, schema):
        return parsed

    feedback = getattr(response, "prompt_feedback", None)
    if feedback is not None and getattr(feedback, "block_reason", None):
        raise GeminiCallError(f"Gemini blocked this request ({feedback.block_reason}).")
    candidates = getattr(response, "candidates", None) or []
    finish = str(getattr(candidates[0], "finish_reason", "") or "") if candidates else ""
    if any(word in finish.upper() for word in ("SAFETY", "PROHIBITED", "BLOCKLIST", "RECITATION")):
        raise GeminiCallError(f"Gemini refused to process this video (finish_reason={finish}).")

    text = (getattr(response, "text", None) or "").strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    if not text:
        raise _RetryableResponse(f"Gemini returned an empty response (finish_reason={finish or 'unknown'}).")
    try:
        return schema.model_validate_json(text)
    except Exception as exc:
        raise _RetryableResponse(f"Gemini returned invalid JSON: {exc}") from exc


# ============================================================================
# Download + validation (adapted from omni_model's URLVideoClient)
# ============================================================================

class URLVideoError(ValueError):
    pass


class NoVideoInPost(URLVideoError):
    """The link works but the post holds no video (e.g. an Instagram photo post)."""


def _is_no_video(exc: Exception) -> bool:
    lowered = str(exc).lower()
    return "there is no video in this post" in lowered or "no video formats found" in lowered


SUPPORTED_VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm", ".m4v"}


def validate_public_video_url(url: str) -> str:
    value = (url or "").strip()
    if not value or len(value) > 2048:
        raise URLVideoError("Enter a valid public video URL.")
    parsed = urlsplit(value)
    if parsed.scheme.lower() not in {"http", "https"}:
        raise URLVideoError("Only public http(s) video URLs are supported.")
    if parsed.username or parsed.password:
        raise URLVideoError("URLs containing embedded credentials are not allowed.")
    if not parsed.hostname:
        raise URLVideoError("The URL must include a hostname.")
    try:
        port = parsed.port
    except ValueError as exc:
        raise URLVideoError("The URL contains an invalid port.") from exc
    if port not in {None, 80, 443}:
        raise URLVideoError("Only standard HTTP/HTTPS ports are allowed.")
    host = parsed.hostname.lower()
    if CONFIG.url_allowlist and not any(host_is(host, domain) for domain in CONFIG.url_allowlist):
        raise URLVideoError(
            f"Links from {host} are not enabled on this app. "
            f"Allowed sites: {', '.join(CONFIG.url_allowlist)}."
        )
    _check_public_host(host)
    return value


def _check_public_host(host: str) -> None:
    """Every address the host resolves to must be on the public internet, so a
    link can never reach this server's own network: localhost, 10.x/192.168.x,
    the cloud metadata service at 169.254.169.254, and so on. Also catches IPs
    written as decimals (2130706433) or as IPv4-mapped IPv6 (::ffff:127.0.0.1)."""
    try:
        infos = socket.getaddrinfo(host, None)
    except (socket.gaierror, UnicodeError) as exc:
        raise URLVideoError(f"Could not find the website {host}. Check the link.") from exc
    for info in infos:
        address = ipaddress.ip_address(str(info[4][0]).split("%")[0])
        if address.version == 6 and address.ipv4_mapped:
            address = address.ipv4_mapped
        if not address.is_global:
            raise URLVideoError("This link points to a private or internal network address, which is not allowed.")


_MAX_REDIRECTS = 5


@dataclass
class Fetched:
    url: str  # after redirects
    headers: Mapping[str, str]
    content: bytes
    encoding: str | None = None

    @property
    def text(self) -> str:
        return self.content.decode(self.encoding or "utf-8", errors="replace")


def safe_get(
    url: str,
    *,
    max_bytes: int,
    timeout: float = 30,
    headers: Mapping[str, str] | None = None,
    cookies: Any = None,
    allowed_domains: tuple[str, ...] = (),
) -> Fetched:
    """HTTP GET for the app's own downloads. Each redirect hop is re-checked
    with validate_public_video_url (and against allowed_domains when given),
    and reading stops once the body passes max_bytes. `cookies` should be a
    cookie jar: requests then only sends each cookie to its own domain."""
    import requests

    current = url
    with requests.Session() as session:
        for _ in range(_MAX_REDIRECTS + 1):
            validate_public_video_url(current)
            host = urlsplit(current).hostname
            if allowed_domains and not any(host_is(host, domain) for domain in allowed_domains):
                raise URLVideoError(f"Refusing to download from {host}: not an expected site for this link.")
            try:
                response = session.get(
                    current, headers=headers, cookies=cookies, timeout=timeout,
                    allow_redirects=False, stream=True,
                )
            except requests.RequestException as exc:
                raise URLVideoError(f"Could not reach {host}: {exc}") from exc
            with response:
                if response.is_redirect:
                    current = urljoin(current, response.headers.get("Location", ""))
                    continue
                if response.status_code >= 400:
                    raise URLVideoError(f"{host} answered HTTP {response.status_code}.")
                body = bytearray()
                for chunk in response.iter_content(64 * 1024):
                    body += chunk
                    if len(body) > max_bytes:
                        raise URLVideoError(f"The file is larger than the {max_bytes // 1048576} MB limit.")
                return Fetched(url=current, headers=response.headers, content=bytes(body), encoding=response.encoding)
    raise URLVideoError("The link redirected too many times.")


def ffmpeg_exe() -> str:
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:
        raise RuntimeError(
            "ffmpeg was not found. Install it (winget install Gyan.FFmpeg) "
            "or run: pip install imageio-ffmpeg"
        ) from exc


class _YtdlpLogger:
    """Keeps yt-dlp warnings/errors, drops its per-chunk progress spam."""

    def debug(self, msg: str) -> None:
        pass

    def info(self, msg: str) -> None:
        pass

    def warning(self, msg: str) -> None:
        logger.warning(f"[yt-dlp] {msg}")

    def error(self, msg: str) -> None:
        logger.error(f"[yt-dlp] {msg}")


def _download_error_message(url: str, exc: Exception) -> str:
    raw = str(exc)
    lowered = raw.lower()
    host = (urlsplit(url).hostname or "").lower()
    if "sign in to confirm" in lowered or "not a bot" in lowered:
        return (
            "The video host asked for sign-in (bot check). Export your browser cookies "
            "to a cookies.txt file and set COOKIES_FILE in .env, or set DOWNLOAD_PROXY."
        )
    if "instagram" in host or "instagram" in lowered or (
        "no video formats found" in lowered and "instagram" in host
    ):
        return (
            "Instagram requires a logged-in session for this post. Export your Instagram "
            "browser cookies to cookies.txt and set COOKIES_FILE in .env."
        )
    if "private" in lowered or "members-only" in lowered:
        return "The video is private or restricted, so it cannot be downloaded."
    if "age" in lowered and "restrict" in lowered:
        return "The video is age-restricted. Set COOKIES_FILE to a logged-in session to download it."
    if "this video is not available" in lowered or "video unavailable" in lowered:
        return "This video is not available on the platform (it may be deleted, private, or region-restricted)."
    if "unsupported url" in lowered:
        return "This link doesn't point to a downloadable video."
    return f"Video download failed: {raw[:400]}"


def _is_retryable_block(exc: Exception) -> bool:
    lowered = str(exc).lower()
    return any(marker in lowered for marker in ("sign in to confirm", "not a bot", "429", "too many requests"))


def _mask_proxy_credentials(proxy: str) -> str:
    parsed = urlsplit(proxy)
    if not parsed.username and not parsed.password:
        return proxy
    netloc = parsed.hostname or ""
    if parsed.port:
        netloc = f"{netloc}:{parsed.port}"
    return parsed._replace(netloc=f"***:***@{netloc}").geturl()


@dataclass
class DownloadedVideo:
    path: Path
    title: str
    source_url: str
    extractor: str


def video_format(max_height: int) -> str:
    """yt-dlp format: best video + audio up to `max_height`p, measured on the
    picture's SHORT side -- a vertical 720p Short is 720x1280, so limiting
    the height to 720 would have cut it to 360x640."""
    long_side = round(max_height * 16 / 9)
    limit = f"[height<={long_side}][width<={long_side}]"
    return f"bv*[ext=mp4]{limit}+ba[ext=m4a]/b[ext=mp4]{limit}/bv*{limit}+ba/b{limit}/bv*+ba/b"


def js_runtimes() -> dict[str, dict]:
    """JavaScript runtimes yt-dlp may use to solve YouTube's player challenge
    (needs Deno >= 2.3 or Node >= 22). The `deno` pip package installs Deno
    next to Python's scripts, which may not be on PATH (e.g. Streamlit Cloud)."""
    import sysconfig

    runtimes: dict[str, dict] = {}
    deno_name = "deno.exe" if os.name == "nt" else "deno"
    scripts = [Path(sysconfig.get_path("scripts"))]
    try:
        scripts.append(Path(sysconfig.get_path("scripts", f"{os.name}_user")))
    except KeyError:
        pass
    deno = shutil.which("deno") or next((str(p / deno_name) for p in scripts if (p / deno_name).is_file()), None)
    if deno:
        runtimes["deno"] = {"path": deno}
    node = shutil.which("node")
    if node:
        runtimes["node"] = {"path": node}
    return runtimes or {"deno": {}, "node": {}}


def download_video(
    url: str,
    destination: Path,
    notify: Notify = _noop_notify,
    *,
    max_seconds: int | None = None,
    max_height: int = 720,
) -> DownloadedVideo:
    """max_seconds defaults to MAX_VIDEO_SECONDS. max_height caps the picture:
    720p is plenty for Gemini; modes that reuse real frames ask for more."""
    from yt_dlp import YoutubeDL
    from yt_dlp.utils import DownloadError

    safe_url = validate_public_video_url(url)
    destination.mkdir(parents=True, exist_ok=True)
    max_bytes = CONFIG.max_download_mb * 1024 * 1024
    limit = max_seconds or CONFIG.max_video_seconds
    rejection: dict[str, str] = {}

    def match_filter(info: dict, *, incomplete: bool):
        reason = None
        if info.get("is_live") or info.get("live_status") in {"is_live", "is_upcoming"}:
            reason = "Live streams are not supported."
        duration = info.get("duration")
        if not reason and duration and duration > limit:
            reason = f"This video is {duration:.0f} seconds long. The maximum is {limit} seconds."
        size = info.get("filesize") or info.get("filesize_approx")
        if not reason and size and size > max_bytes:
            reason = f"This video is larger than the {CONFIG.max_download_mb} MB limit."
        if reason:
            rejection["reason"] = reason
        return reason

    def progress_hook(payload: dict) -> None:
        if payload.get("status") != "downloading":
            return
        done = float(payload.get("downloaded_bytes") or 0)
        total = float(payload.get("total_bytes") or payload.get("total_bytes_estimate") or 0)
        if total:
            notify(f"Downloading video... {min(done / total, 0.99) * 100:.0f}%")

    options: dict[str, Any] = {
        "format": video_format(max_height),
        "outtmpl": str(destination / "source.%(ext)s"),
        "merge_output_format": "mp4",
        "noplaylist": True,
        "playlist_items": "1",
        "max_filesize": max_bytes,
        "match_filter": match_filter,
        "progress_hooks": [progress_hook],
        "restrictfilenames": True,
        "quiet": True,
        "no_warnings": True,
        "socket_timeout": 30,
        "retries": 2,
        "fragment_retries": 2,
        "logger": _YtdlpLogger(),
        "ffmpeg_location": ffmpeg_exe(),
        # yt-dlp picks its own YouTube player clients: forcing android/ios
        # left only an old 360p combined format.
        "js_runtimes": js_runtimes(),
    }
    if CONFIG.cookies_file:
        options["cookiefile"] = CONFIG.cookies_file

    proxy = CONFIG.download_proxy
    attempts = [("proxy", True), ("proxy-retry", True), ("direct", False)] if proxy else [("direct", False)]
    info = None
    last_exc: Exception | None = None
    for index, (label, use_proxy) in enumerate(attempts):
        if index == 1:
            time.sleep(3)
        attempt_options = dict(options)
        if use_proxy:
            attempt_options["proxy"] = proxy
            logger.info(f"[DOWNLOAD] attempt={label} proxy={_mask_proxy_credentials(proxy)}")
        else:
            logger.info(f"[DOWNLOAD] attempt={label} proxy=none")
        try:
            with YoutubeDL(attempt_options) as downloader:
                info = downloader.extract_info(safe_url, download=True)
            last_exc = None
            break
        except DownloadError as exc:
            last_exc = exc
            logger.warning(f"[DOWNLOAD] attempt={label} failed: {exc}")
            if rejection or index == len(attempts) - 1 or not _is_retryable_block(exc):
                break

    if rejection:
        raise URLVideoError(rejection["reason"])
    if last_exc is not None and _is_no_video(last_exc):
        raise NoVideoInPost("This post has no video -- it is a photo post.") from last_exc
    if last_exc is not None:
        raise URLVideoError(_download_error_message(safe_url, last_exc)) from last_exc
    if not info or info.get("_type") in {"playlist", "multi_video"}:
        raise URLVideoError("The URL must point to one video, not a playlist or collection.")

    candidates = [
        path for path in destination.glob("source.*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_VIDEO_SUFFIXES
    ]
    if not candidates:
        raise URLVideoError("The URL did not produce a supported video file.")
    source = max(candidates, key=lambda path: path.stat().st_size)
    size = source.stat().st_size
    if size == 0:
        raise URLVideoError("The download produced an empty file -- the source may need sign-in or be unavailable.")
    if size > max_bytes:
        raise URLVideoError(f"The downloaded video is larger than the {CONFIG.max_download_mb} MB limit.")
    logger.info(
        f"[DOWNLOAD] ok {size / 1048576:.1f}MB extractor={info.get('extractor_key') or 'generic'} "
        f"title={str(info.get('title') or '')[:60]}"
    )
    return DownloadedVideo(
        path=source,
        title=str(info.get("title") or "Untitled video"),
        source_url=str(info.get("webpage_url") or safe_url),
        extractor=str(info.get("extractor_key") or info.get("extractor") or "generic"),
    )


class VideoValidationError(ValueError):
    pass


@dataclass
class MediaInfo:
    duration_seconds: float
    width: int
    height: int
    has_audio: bool
    aspect_ratio: str
    cuts: list[float] = field(default_factory=list)  # measured scene cuts (s), see detect_scene_cuts


SCENE_CUT_THRESHOLD = 0.3  # ffmpeg scene score; 0.3 catches hard cuts, not camera moves
MIN_CUT_GAP_SECONDS = 0.3  # closer "cuts" are flashes / glitches
MAX_CUTS = 200


def detect_scene_cuts(path: Path, threshold: float = SCENE_CUT_THRESHOLD) -> list[float]:
    """Times (seconds) where the picture changes sharply -- the real cuts,
    measured by ffmpeg, so scene boundaries are not left to Gemini's guess."""
    try:
        output = subprocess.run(
            [ffmpeg_exe(), "-hide_banner", "-nostdin", "-i", str(path),
             "-vf", f"select='gt(scene,{threshold})',showinfo", "-an", "-f", "null", "-"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300,
        ).stderr
    except subprocess.TimeoutExpired:
        logger.warning("[CUTS] scene detection timed out -- continuing without measured cuts")
        return []
    cuts: list[float] = []
    for value in re.findall(r"pts_time:(\d+(?:\.\d+)?)", output):
        moment = round(float(value), 2)
        if moment > 0 and (not cuts or moment - cuts[-1] >= MIN_CUT_GAP_SECONDS):
            cuts.append(moment)
    return cuts[:MAX_CUTS]


def analysis_fps(duration_seconds: float) -> float:
    """Frames per second Gemini looks at: more for short clips, where every
    fast cut and gesture matters; Gemini's default is 1."""
    if duration_seconds <= 20:
        return 4.0
    if duration_seconds <= 40:
        return 3.0
    return 2.0


_ASPECT_RATIOS = {"16:9": 16 / 9, "9:16": 9 / 16, "1:1": 1.0, "4:5": 4 / 5, "4:3": 4 / 3, "3:4": 3 / 4}


def closest_aspect_ratio(width: int, height: int) -> str:
    if not width or not height:
        return "16:9"
    ratio = width / height
    return min(_ASPECT_RATIOS, key=lambda name: abs(_ASPECT_RATIOS[name] - ratio))


def probe_video(path: Path) -> MediaInfo:
    """Reads duration/resolution/audio from `ffmpeg -i` output, so only the
    ffmpeg binary is needed (imageio-ffmpeg ships no ffprobe)."""
    result = subprocess.run(
        [ffmpeg_exe(), "-hide_banner", "-i", str(path)],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    output = result.stderr
    duration_match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", output)
    if not duration_match:
        raise VideoValidationError("The downloaded file is not a readable video.")
    hours, minutes, seconds = duration_match.groups()
    duration = int(hours) * 3600 + int(minutes) * 60 + float(seconds)

    video_line = next((line for line in output.splitlines() if "Video:" in line), "")
    size_match = re.search(r",\s*(\d{2,5})x(\d{2,5})", video_line)
    if not size_match:
        raise VideoValidationError("The downloaded file has no video stream.")
    width, height = int(size_match.group(1)), int(size_match.group(2))
    rotation = re.search(r"rotat(?:e|ion of)\s*:?\s*(-?\d+(?:\.\d+)?)", output)
    if rotation and abs(round(float(rotation.group(1)))) % 180 == 90:
        width, height = height, width

    return MediaInfo(
        duration_seconds=round(duration, 2),
        width=width,
        height=height,
        has_audio="Audio:" in output,
        aspect_ratio=closest_aspect_ratio(width, height),
    )


def validate_media(media: MediaInfo, max_seconds: int | None = None) -> None:
    limit = max_seconds or CONFIG.max_video_seconds
    if media.duration_seconds > limit + 0.5:
        raise VideoValidationError(
            f"This video is {media.duration_seconds:.0f} seconds long. "
            f"The maximum is {limit} seconds."
        )
    if media.duration_seconds < CONFIG.min_video_seconds:
        raise VideoValidationError(
            f"This video is only {media.duration_seconds:.1f} seconds long. "
            f"The minimum is {CONFIG.min_video_seconds} second(s)."
        )


# ============================================================================
# Video analysis (one Gemini call on the whole video)
# ============================================================================

class Character(BaseModel):
    label: str = Field(description="Short visual label, e.g. 'girl in yellow raincoat', 'orange cartoon cat'. Never a real person's name.")
    appearance: str = Field(description="Exact look: age range, build, hair, skin/fur, outfit, colors, accessories, art-style specifics.")
    role: str = Field(description="What this character does in the video.")


class SourceScene(BaseModel):
    start_seconds: float
    end_seconds: float
    description: str = Field(description="What is on screen: subjects, composition, framing, background.")
    action: str = Field(description="What happens, in order.")
    setting: str
    shot_type: str = Field(description="e.g. extreme close-up, close-up, medium, wide, POV, top-down.")
    camera_movement: str = Field(description="e.g. static, handheld, slow push-in, pan left, orbit, whip pan.")
    subject_movement: str
    transition_out: str = Field(description="How this scene ends/cuts to the next: hard cut, jump cut, match cut, fade, whip, none.")
    on_screen_text: str = Field(description="Any captions/titles/text overlays and their placement, or empty.")
    audio: str = Field(description="What is heard in this scene: who speaks (the words are in audio.transcript), music, sound effects.")


class Utterance(BaseModel):
    start_seconds: float
    end_seconds: float
    speaker: str = Field(description="Who speaks, using the same visual label as in `characters`, or 'narrator (off-screen)'.")
    text: str = Field(
        description="Exactly what is said, word for word, in the language actually spoken and in that "
                    "language's own script (Telugu in Telugu script, Hindi in Devanagari). Never translated. "
                    "Code-mixed speech (e.g. Telugu with English words) is kept exactly as spoken."
    )
    romanized: str = Field(description="The same words transliterated into Latin letters (identical to `text` for English).")
    english: str = Field(description="English translation, for the user to read only.")


class AudioProfile(BaseModel):
    has_speech: bool
    spoken_language: str = Field(description="Main spoken language as an English name, e.g. 'Telugu', 'Hindi', 'English', or 'none'.")
    speech_summary: str = Field(description="Who speaks, tone and what the speech is about. Empty if no speech.")
    voice_character: str = Field(description="Voice type and delivery, e.g. 'excited child voice', 'calm male narrator', 'none'.")
    music: str = Field(description="Music genre, tempo, energy, instruments, or 'none'. Describe sung lyrics here; do not transcribe them.")
    sound_effects: str
    transcript: list[Utterance] = Field(
        description="Every spoken line in order, one entry per sentence or speaker turn, covering all speech "
                    "from start to end. Empty if there is no speech. Sung lyrics are not included."
    )


class VideoAnalysis(BaseModel):
    content_type: str = Field(description="One of: kids, animation, viral_meme, comedy_skit, vlog, tutorial, music_video, advertisement, product_demo, nature, sports, gameplay, cinematic, other.")
    visual_medium: str = Field(description="One of: live_action, 2d_animation, 3d_animation, stop_motion, motion_graphics, screen_recording, mixed.")
    summary: str = Field(description="2-3 sentence summary of the whole video.")
    hook: str = Field(description="What grabs attention in the first 1-3 seconds.")
    story_arc: str = Field(description="Beginning -> middle -> payoff/ending.")
    art_style: str = Field(description="Concrete visual style: rendering/look, texture, realism level, references to a style (not to a copyrighted title).")
    color_palette: str
    lighting: str
    mood: str
    pacing: str = Field(description="Edit rhythm, e.g. 'fast cuts every ~1s', 'one slow continuous take'.")
    camera_style: str
    characters: list[Character]
    settings: list[str]
    scenes: list[SourceScene] = Field(description="Chronological scenes/shots covering the full timeline.")
    audio: AudioProfile
    why_it_works: str = Field(description="What makes this video engaging/viral/effective -- the elements a recreation must keep.")


def _cut_rule(media: MediaInfo) -> str:
    if not media.cuts:
        return (
            "- Split the timeline into chronological scenes at each cut or major change; "
            "scene times must cover 0 to the end without gaps.\n"
        )
    boundaries = ", ".join(f"{t:g}" for t in [0.0, *media.cuts, media.duration_seconds])
    return (
        f"- The video's cuts were measured exactly: scene boundaries at {boundaries} seconds. Make one "
        "scene per interval between consecutive boundaries, with start/end times exactly at these points "
        "(split a long interval further only at a clear change). Never merge two intervals.\n"
    )


def _analysis_prompt(media: MediaInfo) -> str:
    return (
        "You are a video director's assistant. Watch and listen to this entire video "
        "and produce a precise reconstruction blueprint, so another video model could "
        "recreate the same video (same style, characters, action, pacing, and sound) "
        "from text alone. The video can be anything -- kids content, cartoons, a viral "
        "clip, a vlog, a tutorial, an ad -- describe it on its own terms.\n\n"
        f"Known facts: duration {media.duration_seconds:.1f}s, "
        f"{media.width}x{media.height} ({media.aspect_ratio}), "
        f"{'has an audio track' if media.has_audio else 'no audio track'}.\n\n"
        "Rules:\n"
        "- Be concrete and visual: colors, materials, framing, lens feel, motion direction, "
        "timing. Avoid vague words like 'nice' or 'engaging' when you can describe what you see.\n"
        + _cut_rule(media)
        + "- Watch every second: record every action, object, gesture and on-screen change, in order. "
        "Nothing that happens may be left out.\n"
        "- Describe only what is actually seen and heard. Do not invent dialogue or sounds.\n"
        "- Never identify real people by name; describe them by appearance only.\n"
        "- SPEECH: transcribe every spoken line into audio.transcript word for word, with start/end "
        "times. Write it in the language actually spoken and in that language's own script -- "
        "do NOT translate it into English, do not summarise, and do not skip short or filler words. "
        "Keep code-mixed speech exactly as spoken.\n"
        "- Song lyrics are not speech: describe the song in audio.music instead of transcribing it.\n"
        "- Ignore platform UI and watermarks (TikTok/Instagram/YouTube logos, usernames, "
        "like buttons) -- they are not part of the content.\n"
        "Return JSON matching the schema."
    )


def _upload_and_wait(path: Path, notify: Notify):
    from google.genai import types as genai_types

    import mimetypes

    client = get_client()
    mime_type = mimetypes.guess_type(path.name)[0] or "video/mp4"
    uploaded = call_gemini(
        lambda _model: client.files.upload(
            file=str(path),
            config=genai_types.UploadFileConfig(mime_type=mime_type, display_name=path.name),
        ),
        notify,
        task=None,
        label="upload",
        throttle=False,
    )
    started = time.monotonic()
    while True:
        state = str(getattr(uploaded.state, "name", uploaded.state) or "")
        if state == "ACTIVE":
            return uploaded
        if state == "FAILED":
            raise GeminiCallError("Gemini could not process this video file (unsupported codec or corrupt file).")
        if time.monotonic() - started > 300:
            raise GeminiCallError("Gemini took more than 5 minutes to process the uploaded video.")
        notify("Gemini is processing the uploaded video...")
        time.sleep(3)
        uploaded = call_gemini(
            lambda _model: client.files.get(name=uploaded.name),
            notify,
            task=None,
            label="file-status",
            throttle=False,
        )


def analyze_video(path: Path, media: MediaInfo, notify: Notify = _noop_notify) -> VideoAnalysis:
    """One Gemini call over the original video file (no re-encode, which only
    lost detail), sampled at analysis_fps() instead of Gemini's default 1 fps,
    with the measured cuts in the prompt."""
    from google.genai import types as genai_types

    client = get_client()
    notify("Uploading video to Gemini...")
    uploaded = _upload_and_wait(path, notify)
    try:
        notify("Analysing video with Gemini...")
        prompt = _analysis_prompt(media)
        video_part = genai_types.Part(
            file_data=genai_types.FileData(file_uri=uploaded.uri, mime_type=uploaded.mime_type),
            video_metadata=genai_types.VideoMetadata(fps=analysis_fps(media.duration_seconds)),
        )

        def run(model: str) -> VideoAnalysis:
            response = client.models.generate_content(
                model=model,
                contents=[video_part, prompt],
                config=genai_types.GenerateContentConfig(
                    http_options=gemini_http_options(TIMEOUT_VIDEO_SECONDS),
                    response_mime_type="application/json",
                    response_schema=VideoAnalysis,
                    temperature=0.3,
                    automatic_function_calling=genai_types.AutomaticFunctionCallingConfig(disable=True),
                ),
            )
            return _parse_structured(response, VideoAnalysis)

        analysis = call_gemini(run, notify, task="analysis", label="analyze")
        logger.info(
            f"[ANALYZE] ok content_type={analysis.content_type} medium={analysis.visual_medium} "
            f"scenes={len(analysis.scenes)}"
        )
        return analysis
    finally:
        try:
            client.files.delete(name=uploaded.name)
        except Exception as exc:
            logger.warning(f"[ANALYZE] could not delete uploaded file {uploaded.name}: {exc}")


# ============================================================================
# Prompt generation (one Gemini call + a fixed Python template)
# ============================================================================

class PromptBeat(BaseModel):
    label: str = Field(description="Short beat label, e.g. 'Hook', 'Build-up', 'Payoff'.")
    duration_seconds: float
    visual: str = Field(description="What is on screen: subjects, composition, setting, framing.")
    action: str = Field(description="What happens during this beat, in order.")
    camera: str = Field(description="Shot type + camera movement.")
    audio: str = Field(description="Music and sound effects for this beat, or 'none'. Never dialogue -- that goes in dialogue_line_ids.")
    # No on-screen text field on purpose: video prompts are always text-free.
    dialogue_line_ids: list[int] = Field(
        default_factory=list,
        description="Numbers of the transcript lines (L1 = 1, L2 = 2, ...) spoken during this beat, in order. "
                    "Empty if nobody speaks in this beat.",
    )
    source_scenes: list[int] = Field(
        default_factory=list,
        description="Numbers of the source scenes (S1 = 1, S2 = 2, ...) this beat shows.",
    )


class ReferenceImage(BaseModel):
    ref_id: str = Field(description="REF_1, REF_2, ... in order.")
    purpose: str = Field(description="One of: first_frame, character, object, setting, style.")
    what_it_shows: str = Field(description="One line: what this reference image contains.")
    used_in_scenes: list[int] = Field(description="Scene numbers (1-based) where this reference applies.")
    image_prompt: str = Field(description="Complete standalone text-to-image prompt that creates this reference image.")


class RecreationPlan(BaseModel):
    title: str
    logline: str = Field(description="One sentence describing the recreated video.")
    visual_style: str
    characters: list[str] = Field(description="One fixed, detailed visual description per character, reused verbatim across beats.")
    setting: str
    color_and_lighting: str
    mood: str
    overall_audio: str
    beats: list[PromptBeat]
    avoid: list[str]
    reference_images: list[ReferenceImage] = Field(description="0-3 reference images to generate first and attach to the video generator.")



class SocialMetadataDraft(BaseModel):
    """What Gemini writes; _finish_metadata turns it into SocialMetadata."""

    primary_keyword: str = Field(description="The one search phrase most viewers would type to find this content.")
    search_insights: str = Field(
        description="2-4 sentences: what people search for on this topic and what top-performing titles look like."
    )
    youtube_title: str = Field(description="YouTube title, max 100 characters.")
    youtube_description: str = Field(description="YouTube description, 1000-2000 characters.")
    youtube_tags: list[str] = Field(description="30-40 YouTube tags, best first, no '#'.")
    instagram_caption: str = Field(description="Instagram caption without hashtags.")
    instagram_hashtags: list[str] = Field(description="Exactly 5 niche Instagram hashtags, best first.")


class SocialMetadata(BaseModel):
    """Ready-to-paste YouTube and Instagram metadata, already fitted to each
    platform's limits."""

    youtube_title: str
    youtube_description: str
    youtube_tags: str  # comma-separated, within YouTube's 500-character tag limit
    instagram_caption: str
    instagram_hashtags: str  # space-separated
    primary_keyword: str = ""
    grounded: bool = False  # True when Gemini actually ran Google searches for it
    search_queries: list[str] = Field(default_factory=list)
    sources: list[dict[str, str]] = Field(default_factory=list)  # {"title", "uri"}


# How each reference purpose is attached to a video generator. Kept in code
# (not left to Gemini) so the instructions are consistent between runs.
ATTACH_INSTRUCTIONS = {
    "first_frame": "Attach as the START / FIRST FRAME (image-to-video). The video opens on this exact image.",
    "character": "Attach as a REFERENCE IMAGE (character). Keep this character identical to it in every scene.",
    "object": "Attach as a REFERENCE IMAGE (object). Keep this object identical to it whenever it appears.",
    "setting": "Attach as a REFERENCE IMAGE (setting/background). Match this location's look and layout.",
    "style": "Attach as a STYLE REFERENCE. Match its rendering style, palette and lighting, not its content.",
}
REFERENCE_LABELS = {
    "first_frame": "First frame",
    "character": "Character reference",
    "object": "Object reference",
    "setting": "Setting reference",
    "style": "Style reference",
}


def beats_for_duration(seconds: int, scene_count: int = 0) -> int:
    """Compute beat count for the output video.

    Base formula: 0.375 beats/second (8s -> 3, 10s -> 4).
    When scene_count is provided, scales up to match source scene density so
    each beat stays visually tight and varied instead of compressing many
    distinct scenes into one generic block. Hard cap = duration in seconds
    (guarantees every beat is >= 1s minimum).
    """
    base = max(2, min(6, round(seconds * 0.375)))
    if scene_count > 0:
        # ~1 beat per 3 source scenes, but always at least the base count.
        density = max(base, min(seconds, (scene_count + 2) // 3))
        return density
    return base


def _speech_seconds(lines: list[Utterance]) -> float:
    return sum(max(0.0, line.end_seconds - line.start_seconds) for line in lines)


def _dialogue_rules(analysis: VideoAnalysis, duration: int) -> str:
    """Plan-prompt section for speech. Gemini only picks which transcript
    lines go in which beat; render_prompt inserts the original words, so the
    dialogue can never be translated, reworded or shortened."""
    lines = analysis.audio.transcript
    if not lines:
        return ""
    language = analysis.audio.spoken_language.strip() or "the original language"
    speech = _speech_seconds(lines)
    if speech <= duration:
        coverage = f"- All {len(lines)} lines fit in {duration}s: you MUST use every line."
    else:
        coverage = (
            f"- The speech ({speech:.1f}s) is longer than {duration}s: keep the lines that carry the "
            "story and the punchline, in order, and leave out the rest."
        )
    rows = "\n".join(
        f"  L{number} [{line.start_seconds:.1f}s-{line.end_seconds:.1f}s, "
        f"{max(0.0, line.end_seconds - line.start_seconds):.1f}s] {line.speaker}: "
        f"\"{line.text.strip()}\" (meaning: {line.english.strip()})"
        for number, line in enumerate(lines, start=1)
    )
    return (
        f"=== DIALOGUE (spoken language: {language}) ===\n"
        f"- The source speech is listed below as L1-L{len(lines)} ({speech:.1f}s of speech in total).\n"
        "- Do NOT write dialogue yourself. For each beat, put the numbers of the lines spoken in it "
        "in dialogue_line_ids; the original words are inserted later exactly as transcribed.\n"
        "- Use line numbers in increasing order across the beats, each line at most once.\n"
        f"{coverage}\n"
        "- Give every beat at least as many seconds as its lines took in the source.\n"
        "- If the source speaks from its first second, Beat 1 may carry L1 and need not be silent.\n"
        "- The beat `audio` field is for music and sound effects only.\n"
        f"{rows}\n\n"
    )


FAITHFUL, VIRAL, FULL_LENGTH = "faithful", "viral", "full_length"


def _structure_rules(analysis: VideoAnalysis, media: MediaInfo, duration: int, beat_count: int, mode: str) -> str:
    if mode == VIRAL:
        hook_text = (analysis.hook or "the most visually dramatic moment").strip()
        return (
            "=== HOOK-FIRST STRUCTURE (mandatory for viral social media) ===\n"
            f"- Beat 1 label MUST be 'HOOK'. Duration: 1.0-1.5s maximum.\n"
            f"- Beat 1 must show the single most visually dramatic or satisfying shot: \"{hook_text}\".\n"
            "- Beat 1 MUST work with AUDIO MUTED -- it is a visual-only scroll-stopper.\n"
            "- If the payoff/result appears at the END of the source, move it to Beat 1 here.\n"
            "- Last beat: hold on the final payoff/result (the CTA frame, 1.0-2.0s).\n"
            "- Middle beats: compress the key process steps in source chronological order.\n\n"
            "=== BEAT RULES ===\n"
            f"- Exactly {beat_count} beats. Durations must sum to exactly {duration}s. "
            f"Beat 1 max 1.5s, all others min 1.0s.\n"
            "- Use the SCENE LIST below to group scenes into beats. Do not merge scenes "
            "separated by a hard cut unless both are under 0.5s each.\n"
            "- source_scenes: the S numbers each beat shows.\n"
            "- Keep the source's visual medium, art style, characters, setting, "
            "mood, color palette, and camera style.\n\n"
        )
    timing = (
        "- The source is about as long as the output: keep its timing -- each beat lasts as long as its "
        "scenes do in the source.\n"
        if media.duration_seconds <= duration * 1.2 else
        f"- The source ({media.duration_seconds:.0f}s) is longer than {duration}s: compress every scene in "
        "proportion -- shorten scenes, never drop one.\n"
    )
    return (
        "=== FAITHFUL STRUCTURE (mandatory) ===\n"
        "- Recreate the source as it is: the same scenes, in the same order, with the same events.\n"
        "- Beat 1 is the source's own opening and the last beat is the source's own ending. Never move "
        "the ending to the front.\n"
        f"{timing}\n"
        "=== BEAT RULES ===\n"
        f"- About {beat_count} beats (more if needed to keep every scene). Durations must sum to exactly "
        f"{duration}s, each at least 1.0s.\n"
        "- Use the SCENE LIST below: consecutive scenes may share a beat, but keep their order.\n"
        "- source_scenes: the S numbers each beat shows. EVERY S number must appear in at least one beat.\n"
        "- Keep the source's visual medium, art style, characters, settings, mood, colour palette and "
        "camera style.\n\n"
    )


def _shared_rules(media: MediaInfo, dialogue: str) -> str:
    """Rules every recreation prompt shares (faithful, viral and full length)."""
    return (
        "=== NO TEXT RULE (mandatory) ===\n"
        "- The video is completely text-free: no captions, subtitles, title cards, labels, meme text or "
        "text overlays of any kind, in any language, even if the source has them.\n"
        "- Never describe on-screen text in visual, action or camera; describe only what is seen.\n\n"

        "=== SETTING CONSISTENCY (mandatory) ===\n"
        "- Keep every scene's setting exactly as in the source. Describe each setting once in detail and "
        "reuse those exact words whenever it appears again.\n"
        "- If the source stays in one place, every beat keeps that identical setting (same floor, walls, "
        "furniture, background), close-ups included.\n"
        "- If the source moves between places, change the setting only where the source does.\n\n"

        "=== PHYSICS & OBJECT STATE PRECISION ===\n"
        "- For any object in motion (falling, flying, rotating), describe its exact state at the\n"
        "  START and END of each beat: position (height, distance), orientation (tilt angle), speed.\n"
        "- Example: 'Beat start: the ball is 1.2 m above the floor, spinning clockwise. Beat end: it\n"
        "  touches the floor 10 cm to the left of where it started.'\n"
        "- Do NOT leave physics implied -- state every axis of orientation explicitly.\n\n"

        "=== CAMERA RULES (override model zoom-bias) ===\n"
        "- Write camera as: [shot type], [movement]. If the original shot is static write: "
        "'locked-off [shot type], zero camera movement, no zoom, no pan'.\n"
        "- Use cinematography-precise terms: 'macro close-up', 'locked overhead tripod', "
        "'slow continuous optical dolly-in 0.3m over 2s', 'handheld micro-shake'.\n"
        "- Never write just 'static' or 'close-up' without a qualifier.\n\n"

        "=== AUDIO RULES ===\n"
        "- Per beat: timestamp each distinct sound from beat start. Example: "
        "'0.0s: door slam; 0.3s: footsteps; 0.8s: whoosh transition'.\n"
        "- Preserve the source audio character (ASMR, SFX, voice, music genre).\n\n"

        f"{dialogue}"

        "=== CHARACTER & REFERENCE RULES ===\n"
        "- Describe each character once in `characters`; reuse those exact words in every beat.\n"
        "- Song lyrics: describe the song, never copy lyrics.\n"
        "- reference_images: 0-3 images. 'character' for recurring characters (max 2), "
        "'first_frame' when opening composition matters, 'object' for a key prop. "
        "Number REF_1, REF_2... and cite as [REF_n] inside beat text.\n"
        "- Each image_prompt is a complete standalone text-to-image prompt: subject alone, "
        f"neutral pose, plain background, same visual medium, {media.aspect_ratio} for first_frame.\n"
        "- IMPORTANT: When you generate the reference image prompts, describe the object with full "
        "physical precision: exact orientation (face-up/face-down/vertical/horizontal), "
        "exact colour, material, and any attached parts (handles, straps, buttons).\n\n"

        "=== AVOID (global + video-specific) ===\n"
        "- `avoid` MUST include these global items:\n"
        "  * 'On-screen text, captions, subtitles, or title cards of any kind -- video must be completely text-free'\n"
        "  * 'Changing a scene's setting where the source does not -- each scene keeps the source's own setting'\n"
        "  * 'Unprompted camera zoom or pan -- camera is fixed unless the Camera line explicitly says so'\n"
        "  * 'Slow-motion unless the scene label explicitly says slow motion'\n"
        "  * 'Fade-in or fade-out at start; begin on the first frame at full energy'\n"
        "  * 'Background music unless the AUDIO section specifies it'\n"
        "  * 'Letterboxing or pillarboxing; fill the full frame edge-to-edge'\n"
        "- Also add video-specific failure modes (style drift, wrong character count, wrong object orientation).\n"
        "- No real person names, no brand logos, no platform UI or watermarks.\n\n"
    )


def _plan_prompt(analysis: VideoAnalysis, media: MediaInfo, duration: int, beat_count: int,
                 mode: str = VIRAL) -> str:
    """Build the Gemini prompt that turns a VideoAnalysis into a RecreationPlan.

    mode: FAITHFUL keeps the source's scenes, order and events (what the app
    uses by default); VIRAL restructures it hook-first for Reels / Shorts.
    VIRAL stays this function's default so direct callers keep their old
    behaviour; generate_prompt passes the mode explicitly.
    - Scene list passed explicitly so Gemini groups real cuts, not guesses.
    - Camera-precision rules that override Veo/Omni zoom-bias defaults.
    - Per-beat audio timestamps for tighter sound sync.
    - Global AVOID list baked in so model defaults are suppressed.
    """
    # Numbered scene list: gives Gemini real cut points to group from.
    scene_lines = "\n".join(
        f"  S{i + 1} [{s.start_seconds:.1f}s-{s.end_seconds:.1f}s] "
        f"{s.shot_type} | {s.camera_movement} | {s.action[:80].strip()}"
        for i, s in enumerate(analysis.scenes)
    )
    why_works = (analysis.why_it_works or "").strip()
    purpose = "optimised for Instagram Reels / YouTube Shorts" if mode == VIRAL else "that recreates it faithfully"

    return (
        f"Below is a JSON analysis of a {media.duration_seconds:.0f}-second source video "
        f"({len(analysis.scenes)} scenes detected, {media.aspect_ratio} aspect ratio).\n"
        f"Write a recreation plan for ONE {duration}-second {media.aspect_ratio} video {purpose}.\n\n"

        f"{_structure_rules(analysis, media, duration, beat_count, mode)}"

        f"{_shared_rules(media, _dialogue_rules(analysis, duration))}"

        f"=== WHY THE ORIGINAL WORKS (preserve these elements) ===\n"
        f"{why_works}\n\n"

        f"=== SOURCE SCENE LIST ({len(analysis.scenes)} scenes) ===\n"
        f"{scene_lines}\n\n"

        f"=== FULL SOURCE ANALYSIS (JSON) ===\n"
        f"{analysis.model_dump_json(indent=2)}"
    )


def _normalize_durations(beats: list[PromptBeat], total: int) -> list[PromptBeat]:
    """Rescales beat durations to sum exactly to `total`, rounded to 0.5 s."""
    raw = [max(0.5, float(beat.duration_seconds or 0)) for beat in beats]
    scale = total / sum(raw)
    rounded = [max(1.0, round(value * scale * 2) / 2) for value in raw]
    rounded[-1] = max(1.0, total - sum(rounded[:-1]))
    if abs(sum(rounded) - total) > 0.01:  # min-1s clamps pushed us over; spread evenly
        rounded = [round(total / len(beats), 2)] * len(beats)
        rounded[-1] = round(total - sum(rounded[:-1]), 2)
    return [beat.model_copy(update={"duration_seconds": value}) for beat, value in zip(beats, rounded)]


def _fmt_seconds(value: float) -> str:
    return f"{value:g}"


# Global Veo/Omni avoid items prepended to every prompt to suppress model defaults.
_GLOBAL_AVOID = [
    "Unprompted camera zoom or pan -- camera is fixed unless the Camera line explicitly says so",
    "Slow-motion unless the scene label explicitly says 'slow motion'",
    "Fade-in or fade-out at start; begin on the first frame at full energy",
    "Background music unless the AUDIO section specifies it",
    "Letterboxing, pillarboxing, or black bars -- fill the full frame edge-to-edge",
    "On-screen text, captions, subtitles, or title cards of any kind in any frame -- the video must be completely text-free",
    "Changing a scene's room, background or floor where the source does not -- each scene keeps the source's own setting",
]


def _is_english(language: str) -> bool:
    return language.strip().lower() in {"", "english", "en", "none"}


def _dialogue_line(line: Utterance, language: str) -> str:
    words = line.romanized if CONFIG.dialogue_script == "romanized" and line.romanized.strip() else line.text
    spoken_in = "" if _is_english(language) else f" in {language}"
    return f"Dialogue: {line.speaker.strip()} says{spoken_in}: \"{words.strip()}\""


def render_prompt(
    plan: RecreationPlan,
    media: MediaInfo,
    duration: int,
    audio: AudioProfile | None = None,
    mode: str = VIRAL,
) -> str:
    """Render the final text prompt that gets pasted into a video generator.

    - VIRAL: SOCIAL MEDIA OPTIMISED header, Scene 1 annotated as the HOOK.
      FAITHFUL: a "faithful recreation" header, the source's own opening.
      (VIRAL stays the default for direct callers; generate_prompt passes it.)
    - _GLOBAL_AVOID prepended before video-specific avoids to kill zoom bias.
    - With the source `audio`, each beat's transcript lines are written out
      word for word in the spoken language, plus a language + voice header.
    """
    orientation = {"9:16": "vertical", "3:4": "vertical", "4:5": "vertical", "1:1": "square"}.get(
        media.aspect_ratio, "horizontal"
    )
    transcript = audio.transcript if audio else []
    language = audio.spoken_language.strip() if audio else ""
    lines = [f"Create one {duration}-second {orientation} {media.aspect_ratio} video. {plan.logline.strip()}", ""]
    if mode == VIRAL:
        lines += [
            "SOCIAL MEDIA OPTIMISED (Instagram Reels / YouTube Shorts)",
            "- Scene 1 is the scroll-stopper hook: most dramatic visual, works with audio muted.",
            "- Payoff-first structure: show the result first, then the process, then the CTA hold.",
            "- No fade-in at the very start; begin on the first frame at full energy.",
            "",
        ]
    else:
        lines += [
            "FAITHFUL RECREATION",
            "- The same scenes, in the same order, with the same events as the original.",
            "- No fade-in at the very start; begin on the first frame at full energy.",
            "",
        ]
    if transcript:
        spoken = "English" if _is_english(language) else language
        lines += [
            "SPOKEN LANGUAGE AND VOICE",
            f"- All dialogue is spoken in {spoken}, word for word exactly as written in the Dialogue lines. "
            "Do not translate, shorten, reword or skip any of it"
            + ("." if _is_english(language) else ", and do not switch to English."),
        ]
        if audio.voice_character.strip() and audio.voice_character.strip().lower() != "none":
            lines.append(f"- Voice: {audio.voice_character.strip()}.")
        lines.append("")
    if plan.reference_images:
        lines.append("REFERENCE IMAGES (attached)")
        for ref in plan.reference_images:
            shows = ref.what_it_shows.strip().rstrip(".")
            lines.append(f"- [{ref.ref_id}] {REFERENCE_LABELS[ref.purpose]}: {shows}. {_reference_rule(ref)}")
        lines.append("")
    lines += [
        "STYLE",
        f"- Visual style: {plan.visual_style.strip()}",
        f"- Setting: {plan.setting.strip()}",
        f"- Color and lighting: {plan.color_and_lighting.strip()}",
        f"- Mood: {plan.mood.strip()}",
    ]
    characters = [item.strip() for item in plan.characters if item.strip()]
    if characters:
        lines += ["", "CHARACTERS (keep identical in every shot)", *[f"- {item}" for item in characters]]

    lines += ["", "TIMELINE"]
    cursor = 0.0
    for number, beat in enumerate(plan.beats, start=1):
        end = cursor + beat.duration_seconds
        label = beat.label.strip().upper()
        if number == 1 and mode == VIRAL:
            # Enforce hook label and sound-off annotation on the first beat.
            if "HOOK" not in label:
                label = f"HOOK -- {label}"
            sound_note = "" if beat.dialogue_line_ids else "works sound-OFF, "
            scene_header = (
                f"Scene {number} ({_fmt_seconds(cursor)}-{_fmt_seconds(end)} sec) - {label}"
                f"  [scroll-stopper: {sound_note}starts at full energy, no fade-in]"
            )
        else:
            scene_header = f"Scene {number} ({_fmt_seconds(cursor)}-{_fmt_seconds(end)} sec) - {label}"
        lines.append(scene_header)
        lines.append(f"Visual: {beat.visual.strip()}")
        lines.append(f"Action: {beat.action.strip()}")
        lines.append(f"Camera: {beat.camera.strip()}")
        if beat.audio.strip():
            lines.append(f"Audio: {beat.audio.strip()}")
        for line_id in beat.dialogue_line_ids:
            if 1 <= line_id <= len(transcript):
                lines.append(_dialogue_line(transcript[line_id - 1], language))
        lines.append("")
        cursor = end

    if plan.overall_audio.strip():
        lines += ["AUDIO", plan.overall_audio.strip(), ""]

    # Global avoids first (suppress model defaults), then video-specific.
    avoid = list(_GLOBAL_AVOID) + [item.strip() for item in plan.avoid if item.strip()]
    if transcript:
        avoid.append("Translating, rewording, shortening or skipping any dialogue line")
        if not _is_english(language):
            avoid.append(f"Any spoken language other than {language}; no English voice-over")
    avoid.append("watermarks, platform UI, or logos")
    lines += ["AVOID", *[f"- {item}" for item in dict.fromkeys(avoid)], ""]
    lines.append(
        f"Timing is mandatory: the video must be exactly {duration} seconds. Play the scenes in "
        "this order within their time ranges. Do not add an intro or outro outside the timeline."
    )
    return "\n".join(lines).strip()


def _reference_rule(ref: ReferenceImage) -> str:
    """The in-prompt instruction telling the video model how to use one
    attached reference."""
    return {
        "first_frame": "Start the video exactly on this image.",
        "character": "Keep this character identical to the reference in every shot.",
        "object": "Keep this object identical to the reference whenever it appears.",
        "setting": "Match this location's look and layout.",
        "style": "Match its rendering style, palette and lighting.",
    }[ref.purpose]


def _normalize_references(refs: list[ReferenceImage], beat_count: int) -> list[ReferenceImage]:
    """Max 3, known purposes only, at most one first frame, ids kept as
    Gemini wrote them (the beats already cite them as [REF_n])."""
    normalized: list[ReferenceImage] = []
    seen_ids: set[str] = set()
    has_first_frame = False
    for ref in refs:
        purpose = ref.purpose.strip().lower().replace(" ", "_").replace("-", "_")
        if purpose not in ATTACH_INSTRUCTIONS or not ref.image_prompt.strip():
            continue
        if purpose == "first_frame":
            if has_first_frame:
                continue
            has_first_frame = True
        ref_id = ref.ref_id.strip().strip("[]").upper() or f"REF_{len(normalized) + 1}"
        if ref_id in seen_ids:
            continue
        seen_ids.add(ref_id)
        scenes = sorted({n for n in ref.used_in_scenes if 1 <= n <= beat_count}) or [1]
        normalized.append(ref.model_copy(update={"ref_id": ref_id, "purpose": purpose, "used_in_scenes": scenes}))
        if len(normalized) == 3:
            break
    return normalized


def render_reference_sheet(plan: RecreationPlan) -> str:
    """Human instructions: which images to generate first, and how to attach
    each one to the video generator."""
    if not plan.reference_images:
        return (
            "No reference images needed -- the video prompt is fully text-only. "
            "Paste it into your video generator as-is."
        )
    lines = [
        "Free-tier Gemini only writes text, so these images are NOT generated here.",
        "1. Generate each image below with any text-to-image model.",
        "2. Paste the video prompt into your video generator.",
        "3. Attach each image as described. The prompt refers to them as [REF_n].",
        "   If your generator accepts only one image, attach the first-frame image "
        "(or the first character reference).",
        "",
    ]
    for ref in plan.reference_images:
        lines += [
            f"[{ref.ref_id}] {REFERENCE_LABELS[ref.purpose]} -- {ref.what_it_shows.strip()}",
            f"How to attach: {ATTACH_INSTRUCTIONS[ref.purpose]}",
            f"Used in scenes: {', '.join(str(n) for n in ref.used_in_scenes)}",
            "Image generation prompt:",
            ref.image_prompt.strip(),
            "",
        ]
    return "\n".join(lines).strip()


def _clean_dialogue_ids(beats: list[PromptBeat], line_count: int) -> list[PromptBeat]:
    """Keeps valid transcript line numbers, each used once, in plan order."""
    seen: set[int] = set()
    cleaned = []
    for beat in beats:
        ids = [n for n in beat.dialogue_line_ids if 1 <= n <= line_count and n not in seen]
        seen.update(ids)
        cleaned.append(beat.model_copy(update={"dialogue_line_ids": ids}))
    return cleaned


def _missing_lines(beats: list[PromptBeat], line_count: int) -> list[int]:
    used = {n for beat in beats for n in beat.dialogue_line_ids}
    return [n for n in range(1, line_count + 1) if n not in used]


def _dialogue_warnings(beats: list[PromptBeat], transcript: list[Utterance], duration: int) -> list[str]:
    """Plain-language warnings shown above the prompt when speech is cut or
    squeezed into less time than it took in the source."""
    warnings = []
    missing = _missing_lines(beats, len(transcript))
    if missing:
        dropped = "; ".join(f"L{n}: {transcript[n - 1].english.strip()}" for n in missing)
        reason = (
            f"the source has {_speech_seconds(transcript):.1f}s of speech, more than {duration}s"
            if _speech_seconds(transcript) > duration
            else "Gemini left them out even after a retry"
        )
        warnings.append(f"{len(missing)} dialogue line(s) are not in the prompt because {reason} -- {dropped}")
    for number, beat in enumerate(beats, start=1):
        needed = _speech_seconds([transcript[n - 1] for n in beat.dialogue_line_ids])
        if needed > beat.duration_seconds * 1.15:
            warnings.append(
                f"Scene {number} has {needed:.1f}s of dialogue in {beat.duration_seconds:g}s -- the "
                "video generator may rush or cut words. A longer output length avoids this."
            )
    return warnings


def accuracy_report(beat_groups: list[list[PromptBeat]], analysis: VideoAnalysis) -> dict:
    """How much of the source reached the prompt(s): spoken words kept and
    source scenes shown, so dropped content is visible instead of guessed."""
    transcript = analysis.audio.transcript
    used = {n for beats in beat_groups for beat in beats for n in beat.dialogue_line_ids if 1 <= n <= len(transcript)}
    scene_count = len(analysis.scenes)
    covered = {n for beats in beat_groups for beat in beats for n in beat.source_scenes if 1 <= n <= scene_count}
    return {
        "words_kept": sum(len(transcript[n - 1].text.split()) for n in used),
        "words_total": sum(len(line.text.split()) for line in transcript),
        "scenes_covered": len(covered),
        "scenes_total": scene_count,
        "missing_scenes": sorted(set(range(1, scene_count + 1)) - covered),
    }


def accuracy_line(report: dict) -> str:
    """'Words kept 97% (120/124) · Scenes covered 12/12'"""
    if not report:
        return ""
    parts = []
    if report["words_total"]:
        share = 100 * report["words_kept"] / report["words_total"]
        parts.append(f"Words kept {share:.0f}% ({report['words_kept']}/{report['words_total']})")
    if report["scenes_total"]:
        parts.append(f"Scenes covered {report['scenes_covered']}/{report['scenes_total']}")
    return " · ".join(parts)


@dataclass
class GeneratedPrompt:
    duration: int
    prompt: str
    plan: dict
    saved_to: str
    references: list[dict] = field(default_factory=list)
    reference_sheet: str = ""
    social_metadata: "SocialMetadata | None" = None
    social_text: str = ""
    warnings: list[str] = field(default_factory=list)
    mode: str = ""
    accuracy: dict = field(default_factory=dict)  # see accuracy_report
    parts: list[str] = field(default_factory=list)  # full length: one prompt per consecutive part

    def full_text(self) -> str:
        parts = [
            "=== VIDEO GENERATION PROMPT ===\n\n",
            self.prompt,
            "\n\n=== REFERENCE IMAGES ===\n\n",
            self.reference_sheet,
        ]
        if self.social_text:
            parts.append("\n\n=== SOCIAL MEDIA METADATA ===\n\n" + self.social_text)
        return "".join(parts)




# ── YouTube / Instagram metadata ────────────────────────────────────────────
# YouTube counts a tag containing a space as 2 characters longer (it wraps it
# in quotes), plus 1 per comma, and ignores EVERY tag once that total passes
# 500 -- so tags are packed to a target safely below the limit.
YOUTUBE_TITLE_MAX = 100
YOUTUBE_DESCRIPTION_MAX_BYTES = 5000  # bytes, not characters: Telugu/Hindi letters take 3 each
YOUTUBE_TAGS_LIMIT = 500
YOUTUBE_TAGS_TARGET = 480
INSTAGRAM_CAPTION_MAX = 2200
INSTAGRAM_HASHTAGS_MAX = 5  # Instagram has reportedly enforced 5 per post since Dec 2025
# Tags so broad they bring no targeted viewers and only use up a slot.
GENERIC_HASHTAGS = frozenset({
    "#viral", "#reels", "#explorepage", "#explore", "#fyp", "#foryou", "#foryoupage",
    "#trending", "#instagood", "#reelsinstagram", "#viralreels", "#viralvideo", "#reelitfeelit",
})
# Free-tier models whose quota includes Grounding with Google Search (500
# requests/day on the pricing page). gemini-3.5-flash-lite has none for free.
FREE_SEARCH_MODELS = frozenset({"gemini-2.5-flash", "gemini-2.5-flash-lite"})


def _metadata_language_rule(language: str) -> str:
    if _is_english(language) or CONFIG.metadata_language == "english":
        return "- Write everything in English."
    if CONFIG.metadata_language == "native":
        return (
            f"- Write the title, description and caption in {language}, in its own script. "
            f"Tags: {language} script, romanized {language} and English."
        )
    return (
        f"- The content is in {language}. Viewers search in English AND in {language} typed in Latin "
        f"letters (e.g. Tenglish / Hinglish), and sometimes in {language} script. Write the title, "
        f"description and caption in English mixed with the most-searched romanized {language} words. "
        f"Tags: English, romanized {language} and {language}-script variants."
    )


def _metadata_prompt(context: str, kind: str, language: str, search: bool) -> str:
    """kind: "video" (a Short / Reel) or "image" (a post). With search=True the
    reply is plain-text JSON, because 2.5 models cannot combine Google Search
    with a response schema."""
    is_video = kind == "video"
    if search:
        research = (
            "=== RESEARCH FIRST (use Google Search) ===\n"
            "1. Search what people currently type on YouTube and Instagram to find this exact topic.\n"
            f"2. Look at the titles, descriptions and tags of top-performing recent "
            f"{'Shorts and Reels' if is_video else 'posts'} about it.\n"
            "3. Note the exact search phrases, long-tail variants and currently trending terms.\n"
            "Base every keyword on what you found; do not invent trends.\n\n"
        )
        reply = (
            "\nReply with ONLY one JSON object and no other text, with these keys: primary_keyword, "
            "search_insights, youtube_title, youtube_description, youtube_tags (array of strings), "
            "instagram_caption, instagram_hashtags (array of strings)."
        )
    else:
        research = "=== KEYWORDS ===\nUse what you know people search on YouTube and Instagram for this topic.\n\n"
        reply = ""
    title_rule = "End it with ' #Shorts'." if is_video else "No hashtags in the title."
    hashtag_line = "#Shorts first, then 2-4 niche hashtags" if is_video else "3-5 niche hashtags"
    return (
        "You are a YouTube and Instagram growth strategist. Write metadata that gets this "
        f"{'short video' if is_video else 'image post'} found in search and recommended.\n\n"
        f"{research}"
        "=== LANGUAGE ===\n"
        f"{_metadata_language_rule(language)}\n\n"
        "=== YOUTUBE ===\n"
        "- primary_keyword: the single phrase most viewers type to find this.\n"
        f"- youtube_title: max {YOUTUBE_TITLE_MAX} characters, best 50-70. Primary keyword within the "
        f"first 40 characters, plus a curiosity or emotion hook and 1 emoji. {title_rule}\n"
        "- youtube_description: 1000-2000 characters, in this order:\n"
        "  1. One hook sentence containing the primary keyword.\n"
        "  2. Two short paragraphs about the content that naturally use the primary keyword and 4-6 "
        "related search phrases. Write sentences, never a bare list of keywords.\n"
        "  3. A call to action (subscribe, or answer a question in the comments).\n"
        f"  4. Last line: {hashtag_line}.\n"
        "- youtube_tags: 30-40 tags, best first: the primary keyword, its close variants, long-tail "
        "phrases people really search, language variants, and 2-3 broad category tags. 1-5 words "
        "each, no '#'. The app keeps as many as fit in YouTube's 500-character limit.\n\n"
        "=== INSTAGRAM ===\n"
        f"- instagram_caption: max {INSTAGRAM_CAPTION_MAX} characters, best 300-800. The first line "
        "(under 125 characters, shown before '...more') holds the hook and the primary keyword, "
        "because Instagram search reads captions. Then 2-4 short lines with natural keywords. End "
        "with a save or share call to action. No hashtags in the caption.\n"
        f"- instagram_hashtags: exactly {INSTAGRAM_HASHTAGS_MAX}, best first, specific to this "
        "content's topic, niche and community. Never generic tags such as #viral #reels "
        "#explorepage #fyp #trending.\n\n"
        "=== CONTENT ===\n"
        f"{context}\n"
        f"{reply}"
    )


def youtube_tags_cost(tags: list[str]) -> int:
    """Characters YouTube counts against its 500 limit: quotes around tags
    that contain a space, plus one per comma."""
    return sum(len(tag) + (2 if " " in tag else 0) for tag in tags) + max(0, len(tags) - 1)


def split_tags(tags: str) -> list[str]:
    return [tag.strip() for tag in tags.split(",") if tag.strip()]


def _pack_youtube_tags(candidates: list[str], target: int = YOUTUBE_TAGS_TARGET) -> str:
    """Best-first tags, deduplicated, filled up to `target` YouTube-counted
    characters. A tag that does not fit is skipped, so shorter ones after it
    can still use the space."""
    chosen: list[str] = []
    seen: set[str] = set()
    for raw in candidates:
        tag = " ".join(re.sub(r"[#,<>]", " ", raw).split())  # YouTube rejects < and >
        if not tag or tag.casefold() in seen:
            continue
        if youtube_tags_cost([*chosen, tag]) <= target:
            chosen.append(tag)
            seen.add(tag.casefold())
    return ", ".join(chosen)


MAX_FILLER_HASHTAG_LENGTH = 25  # "#telugubiryanirecipe"-length; longer ones nobody searches


def _hashtag(text: str) -> str:
    """'#' + letters/marks/digits/underscore, so Telugu or Hindi tags keep their
    vowel signs; spaces and punctuation are dropped ('phone drop' -> '#phonedrop')."""
    body = "".join(ch for ch in text.lstrip("#") if ch == "_" or unicodedata.category(ch)[0] in "LMN")
    return f"#{body}" if body else ""


def _validate_instagram_hashtags(tags: str | list[str], fill_from: list[str] | None = None) -> str:
    """Exactly INSTAGRAM_HASHTAGS_MAX hashtags when possible: the given ones
    (deduplicated, generic ones such as #viral dropped), topped up from
    `fill_from` (ranked keyword ideas, turned into hashtags) when generic tags
    were removed or Gemini wrote fewer than five."""
    tokens = tags.split() if isinstance(tags, str) else [t for tag in tags for t in tag.split()]
    result: list[str] = []

    def add(tag: str) -> None:
        if tag and tag.casefold() not in GENERIC_HASHTAGS and tag.casefold() not in {t.casefold() for t in result}:
            result.append(tag)

    for token in tokens:
        add(_hashtag(token))
    for idea in fill_from or []:
        if len(result) >= INSTAGRAM_HASHTAGS_MAX:
            break
        tag = _hashtag(idea)
        if len(tag) <= MAX_FILLER_HASHTAG_LENGTH:
            add(tag)
    return " ".join(result[:INSTAGRAM_HASHTAGS_MAX])


def _cut_at_boundary(text: str) -> str:
    """Drops the partial line or word a hard length cut leaves at the end."""
    for separator in ("\n", " "):
        index = text.rfind(separator)
        if index > len(text) // 2:
            return text[:index].rstrip()
    return text


def _fit_title(title: str, shorts: bool) -> str:
    title = " ".join(title.split())
    suffix = " #Shorts" if shorts and "#shorts" not in title.casefold() else ""
    room = YOUTUBE_TITLE_MAX - len(suffix)
    if len(title) > room:
        title = title[:room].rsplit(" ", 1)[0].rstrip(" -|,:")
    return title + suffix


def _finish_metadata(
    draft: SocialMetadataDraft, kind: str, queries: list[str], sources: list[dict[str, str]],
) -> SocialMetadata:
    """Fits Gemini's draft to each platform's limits."""
    description = draft.youtube_description.strip()
    encoded = description.encode("utf-8")
    if len(encoded) > YOUTUBE_DESCRIPTION_MAX_BYTES:
        description = _cut_at_boundary(encoded[:YOUTUBE_DESCRIPTION_MAX_BYTES].decode("utf-8", "ignore"))
    caption = draft.instagram_caption.strip()
    if len(caption) > INSTAGRAM_CAPTION_MAX:
        caption = _cut_at_boundary(caption[:INSTAGRAM_CAPTION_MAX])
    return SocialMetadata(
        youtube_title=_fit_title(draft.youtube_title, shorts=kind == "video"),
        youtube_description=description,
        youtube_tags=_pack_youtube_tags([draft.primary_keyword, *draft.youtube_tags]),
        instagram_caption=caption,
        instagram_hashtags=_validate_instagram_hashtags(
            draft.instagram_hashtags, fill_from=[draft.primary_keyword, *draft.youtube_tags]
        ),
        primary_keyword=draft.primary_keyword.strip(),
        grounded=bool(queries or sources),
        search_queries=queries,
        sources=sources,
    )


def _grounding_info(response: Any) -> tuple[list[str], list[dict[str, str]]]:
    """The Google searches Gemini ran and the pages it used."""
    candidates = getattr(response, "candidates", None) or []
    meta = getattr(candidates[0], "grounding_metadata", None) if candidates else None
    if meta is None:
        return [], []
    queries = [query for query in (meta.web_search_queries or []) if query]
    sources: list[dict[str, str]] = []
    for chunk in meta.grounding_chunks or []:
        web = getattr(chunk, "web", None)
        if web is not None and web.uri and all(s["uri"] != web.uri for s in sources):
            sources.append({"title": web.title or web.domain or web.uri, "uri": web.uri})
    return queries, sources


def _parse_json_reply(response: Any, schema: type[BaseModel]) -> BaseModel:
    """Search replies come back as text, sometimes with prose around the JSON."""
    try:
        return _parse_structured(response, schema)
    except _RetryableResponse:
        text = getattr(response, "text", None) or ""
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise
        try:
            return schema.model_validate_json(text[start:end + 1])
        except ValueError as exc:
            raise _RetryableResponse(f"Gemini returned invalid JSON: {exc}") from exc


def generate_with_optional_search(
    build_prompt: Callable[[bool], str],
    schema: type[BaseModel],
    *,
    task: str,
    label: str,
    notify: Notify = _noop_notify,
    contents_before: list | None = None,
    temperature: float = 0.7,
    thinking_budget: int | None = None,
) -> tuple[BaseModel, list[str], list[dict[str, str]]]:
    """One free-tier Gemini call, grounded in Google Search when it can be.

    On a model whose free tier includes Google Search (FREE_SEARCH_MODELS) the
    call searches; 2.5 models cannot combine search with a response schema, so
    build_prompt(True) must ask for JSON as plain text. Otherwise -- or once
    today's free search quota is used up -- build_prompt(False) runs with the
    schema enforced. Returns (result, searches run, sources used); both lists
    are empty when nothing was searched. `contents_before` (e.g. images) is
    sent ahead of the prompt. `thinking_budget` caps gemini-2.5-flash's
    thinking (faster replies); other models ignore it.
    """
    from google.genai import types as genai_types

    client = get_client()

    def ask(model: str, search: bool):
        options: dict[str, Any] = {
            "http_options": gemini_http_options(TIMEOUT_SEARCH_SECONDS if search else TIMEOUT_TEXT_SECONDS),
            "temperature": temperature,
            "automatic_function_calling": genai_types.AutomaticFunctionCallingConfig(disable=True),
        }
        if thinking_budget is not None and model == "gemini-2.5-flash":
            options["thinking_config"] = genai_types.ThinkingConfig(thinking_budget=thinking_budget)
        if search:
            options["tools"] = [genai_types.Tool(google_search=genai_types.GoogleSearch())]
        else:
            options.update(response_mime_type="application/json", response_schema=schema)
        response = client.models.generate_content(
            model=model,
            contents=[*(contents_before or []), build_prompt(search)],
            config=genai_types.GenerateContentConfig(**options),
        )
        if not search:
            return _parse_structured(response, schema), [], []
        queries, sources = _grounding_info(response)
        return _parse_json_reply(response, schema), queries, sources

    def run(model: str):
        if model in FREE_SEARCH_MODELS and not _search_used_up():
            try:
                return ask(model, search=True)
            except Exception as exc:
                if not _is_search_quota(exc):
                    raise
                logger.warning(f"[GEMINI] {label} model={model} search quota used up: {exc}")
                _mark_search_used_up()
                notify("Today's free Google Search quota is used up -- continuing without search")
                _record_call(model)
        return ask(model, search=False)

    return call_gemini(run, notify, task=task, label=label)


def generate_metadata(context: str, kind: str, language: str, notify: Notify = _noop_notify) -> SocialMetadata:
    """One free-tier Gemini call for YouTube + Instagram metadata, grounded in
    Google Search when today's free quota allows; SocialMetadata.grounded
    says whether it was."""
    notify("Researching keywords and writing social media metadata...")
    draft, queries, sources = generate_with_optional_search(
        lambda search: _metadata_prompt(context, kind, language, search),
        SocialMetadataDraft,
        task="metadata",
        label=f"{kind}-metadata",
        notify=notify,
        thinking_budget=METADATA_THINKING_BUDGET,
    )
    return _finish_metadata(draft, kind, queries, sources)


def generate_social_metadata(
    result: "AnalysisResult",
    plan: "RecreationPlan",
    notify: Notify = _noop_notify,
) -> "SocialMetadata":
    """YouTube + Instagram metadata for the recreated video (one Gemini call)."""
    analysis = VideoAnalysis.model_validate(result.analysis)
    audio = analysis.audio
    speech = " ".join(line.english.strip() for line in audio.transcript)[:600]
    context = "\n".join(part for part in [
        f"Content type: {analysis.content_type} ({analysis.visual_medium})",
        f"Summary: {analysis.summary.strip()[:400]}",
        f"Hook: {analysis.hook.strip()}",
        f"Why it works: {analysis.why_it_works.strip()[:300]}",
        f"Recreated video title: {plan.title}",
        f"Logline: {plan.logline}",
        f"Mood: {plan.mood}",
        f"Spoken language: {audio.spoken_language}" if audio.has_speech else "",
        f"What is said (English meaning): {speech}" if speech else "",
    ] if part)
    language = audio.spoken_language if audio.has_speech else "English"
    return generate_metadata(context, "video", language, notify)


def render_social_metadata(meta: "SocialMetadata") -> str:
    """Render the social metadata into a copy-paste ready text block."""
    tag_cost = youtube_tags_cost(split_tags(meta.youtube_tags))
    lines = [
        (f"🔎 Keywords researched with Google Search ({len(meta.search_queries)} searches)"
         if meta.grounded else
         "⚠️ Not grounded in Google Search -- keywords come from Gemini's own knowledge"),
        "",
        "=== YOUTUBE ===",
        "",
        "📌 TITLE",
        meta.youtube_title,
        "",
        "📄 DESCRIPTION",
        meta.youtube_description,
        "",
        f"🏷️ TAGS  ({tag_cost}/{YOUTUBE_TAGS_LIMIT} as YouTube counts them — paste into YouTube Tags field)",
        meta.youtube_tags,
        "",
        "─" * 50,
        "",
        "=== INSTAGRAM ===",
        "",
        "✍️ CAPTION",
        meta.instagram_caption,
        "",
        "#️⃣ HASHTAGS",
        meta.instagram_hashtags,
    ]
    if meta.search_queries:
        lines += ["", "🔎 SEARCHES USED", "; ".join(meta.search_queries)]
    if meta.sources:
        lines += ["", "SOURCES", *[f"- {source['title']}: {source['uri']}" for source in meta.sources]]
    return "\n".join(lines)

def add_social_metadata(generated: GeneratedPrompt, result: "AnalysisResult",
                        notify: Notify = _noop_notify) -> GeneratedPrompt:
    """The prompt plus its YouTube / Instagram metadata (one Gemini call),
    also written next to the saved prompt."""
    import dataclasses

    plan = RecreationPlan.model_validate(generated.plan)
    social = generate_social_metadata(result, plan, notify)
    updated = dataclasses.replace(generated, social_metadata=social, social_text=render_social_metadata(social))
    saved = Path(updated.saved_to)
    saved.parent.mkdir(parents=True, exist_ok=True)
    saved.write_text(updated.full_text(), encoding="utf-8")
    saved.with_name(f"{saved.stem}_social.json").write_text(social.model_dump_json(indent=2), encoding="utf-8")
    return updated


def generate_prompt(result: "AnalysisResult", duration: int, notify: Notify = _noop_notify,
                    include_social: bool = True, mode: str = FAITHFUL) -> GeneratedPrompt:
    """mode: FAITHFUL (default: same scenes and order), VIRAL (hook first) or
    FULL_LENGTH (consecutive `duration`-second parts covering the whole video)."""
    from google.genai import types as genai_types

    if duration not in CONFIG.output_durations:
        raise ValueError(f"Duration must be one of {CONFIG.output_durations}.")
    if mode == FULL_LENGTH:
        return generate_full_length(result, duration, notify, include_social)
    client = get_client()
    analysis = VideoAnalysis.model_validate(result.analysis)
    media = MediaInfo(**result.media)
    beat_count = beats_for_duration(duration, len(analysis.scenes))
    transcript = analysis.audio.transcript
    request = _plan_prompt(analysis, media, duration, beat_count, mode=mode)
    contents = [request]

    def run(model: str) -> RecreationPlan:
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=genai_types.GenerateContentConfig(
                http_options=gemini_http_options(TIMEOUT_TEXT_SECONDS),
                response_mime_type="application/json",
                response_schema=RecreationPlan,
                temperature=0.7,
                automatic_function_calling=genai_types.AutomaticFunctionCallingConfig(disable=True),
            ),
        )
        plan = _parse_structured(response, RecreationPlan)
        if not plan.beats:
            raise _RetryableResponse("Gemini returned a plan with no beats.")
        return plan

    notify(f"Writing the {duration}-second prompt...")
    plan = call_gemini(run, notify, task="text", label="prompt")
    plan = plan.model_copy(update={"beats": _clean_dialogue_ids(plan.beats, len(transcript))})

    def gaps(candidate: RecreationPlan) -> tuple[list[int], list[int]]:
        """Dropped dialogue lines that fit, and (faithful) source scenes not shown."""
        lines = _missing_lines(candidate.beats, len(transcript)) if _speech_seconds(transcript) <= duration else []
        scenes = accuracy_report([candidate.beats], analysis)["missing_scenes"] if mode == FAITHFUL else []
        return lines, scenes

    missing_lines, missing_scenes = gaps(plan)
    if missing_lines or missing_scenes:
        # Content that should be there is Gemini's mistake: one retry.
        logger.info(f"[PROMPT] missing lines {missing_lines}, scenes {missing_scenes} -- retrying once")
        notify("Some of the source was left out -- asking Gemini to include all of it...")
        correction = []
        if missing_lines:
            correction.append(
                f"You left out dialogue lines {', '.join(f'L{n}' for n in missing_lines)}. All lines fit in "
                f"{duration}s, so every line L1-L{len(transcript)} must appear in exactly one beat's "
                "dialogue_line_ids, in increasing order."
            )
        if missing_scenes:
            correction.append(
                f"You left out source scenes {', '.join(f'S{n}' for n in missing_scenes)}. Every scene must "
                "appear in some beat's source_scenes: shorten scenes instead of dropping them."
            )
        contents = [request + "\n\n=== CORRECTION ===\n" + " ".join(correction) + " Return the complete corrected plan."]
        retry = call_gemini(run, notify, task="text", label="prompt-retry")
        retry = retry.model_copy(update={"beats": _clean_dialogue_ids(retry.beats, len(transcript))})
        retry_lines, retry_scenes = gaps(retry)
        if len(retry_lines) + len(retry_scenes) < len(missing_lines) + len(missing_scenes):
            plan = retry
    if len(plan.beats) != beat_count:
        logger.info(f"[PROMPT] asked for {beat_count} beats, got {len(plan.beats)} -- keeping Gemini's split")
    plan = plan.model_copy(update={
        "beats": _normalize_durations(plan.beats, duration),
        "reference_images": _normalize_references(plan.reference_images, len(plan.beats)),
    })
    warnings = _dialogue_warnings(plan.beats, transcript, duration) if transcript else []
    prompt = render_prompt(plan, media, duration, analysis.audio, mode=mode)
    report = accuracy_report([plan.beats], analysis)
    if mode == FAITHFUL and report["missing_scenes"]:
        warnings.append(
            f"Source scene(s) {', '.join(f'S{n}' for n in report['missing_scenes'])} are not in the prompt. "
            "Full length mode covers the whole video."
        )
    generated = GeneratedPrompt(
        duration=duration,
        prompt=prompt,
        plan=plan.model_dump(),
        saved_to="",
        references=[ref.model_dump() for ref in plan.reference_images],
        reference_sheet=render_reference_sheet(plan),
        warnings=warnings,
        mode=mode,
        accuracy=report,
    )
    return _save_generated(generated, plan, result, notify, include_social, f"{duration}s_{mode}")


def _save_generated(generated: GeneratedPrompt, plan: RecreationPlan, result: "AnalysisResult",
                    notify: Notify, include_social: bool, label: str) -> GeneratedPrompt:
    import dataclasses

    prompts_dir = CONFIG.work_dir / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{result.video_sha[:12]}_{label}_{time.strftime('%Y%m%d-%H%M%S')}"
    generated = dataclasses.replace(generated, saved_to=str(prompts_dir / f"{stem}.txt"))
    if include_social:  # the UI does this separately, after showing the prompt
        try:
            generated = add_social_metadata(generated, result, notify)
        except Exception as exc:
            logger.warning(f"[SOCIAL] metadata generation failed: {exc} -- skipping")
    (prompts_dir / f"{stem}.txt").write_text(generated.full_text(), encoding="utf-8")
    (prompts_dir / f"{stem}.json").write_text(plan.model_dump_json(indent=2), encoding="utf-8")
    logger.info(f"[PROMPT] saved {stem}.txt ({len(generated.prompt)} chars, {accuracy_line(generated.accuracy)})")
    return generated


# ── Full length: consecutive parts covering the whole video ─────────────────

class PartPlan(BaseModel):
    beats: list[PromptBeat]


class FullLengthPlan(BaseModel):
    """One shared look (characters, style, setting) and one beat list per part."""
    title: str
    logline: str = Field(description="One sentence describing the whole recreated video.")
    visual_style: str
    characters: list[str] = Field(description="One fixed, detailed visual description per character, reused verbatim.")
    setting: str
    color_and_lighting: str
    mood: str
    overall_audio: str
    avoid: list[str]
    reference_images: list[ReferenceImage] = Field(description="0-3 reference images shared by all parts.")
    parts: list[PartPlan] = Field(description="One entry per part, in order.")


def part_windows(source_seconds: float, cuts: list[float], part_seconds: int) -> list[tuple[float, float]]:
    """Splits the source into consecutive windows of at most part_seconds,
    ending each at a measured cut when one is within 1.5 s before the limit,
    so no part starts mid-shot. A short remainder (< 1 s) joins the last part."""
    windows: list[tuple[float, float]] = []
    start = 0.0
    while source_seconds - start > 0.05:
        limit = start + part_seconds
        if limit >= source_seconds - 1.0:
            end = source_seconds
        else:
            nearby = [cut for cut in cuts if limit - 1.5 <= cut <= limit and cut > start + 1.0]
            end = max(nearby) if nearby else limit
        windows.append((round(start, 2), round(end, 2)))
        start = end
    return windows


def _window_contents(analysis: VideoAnalysis, window: tuple[float, float], last: bool) -> tuple[list[int], list[int]]:
    """(scene numbers, dialogue line numbers) belonging to one window."""
    start, end = window
    scenes = [n for n, s in enumerate(analysis.scenes, start=1) if s.start_seconds < end and s.end_seconds > start]
    lines = [n for n, line in enumerate(analysis.audio.transcript, start=1)
             if start <= line.start_seconds < end or (last and line.start_seconds >= end)]
    return scenes, lines


def _full_length_prompt(analysis: VideoAnalysis, media: MediaInfo, part_seconds: int,
                        windows: list[tuple[float, float]]) -> str:
    rows = []
    for number, window in enumerate(windows, start=1):
        scenes, lines = _window_contents(analysis, window, number == len(windows))
        rows.append(
            f"  Part {number}: source {window[0]:g}-{window[1]:g}s · scenes "
            f"{', '.join(f'S{n}' for n in scenes) or 'none'} · dialogue {', '.join(f'L{n}' for n in lines) or 'none'}"
        )
    scene_lines = "\n".join(
        f"  S{i + 1} [{s.start_seconds:.1f}s-{s.end_seconds:.1f}s] {s.shot_type} | {s.camera_movement} | "
        f"{s.action[:100].strip()}"
        for i, s in enumerate(analysis.scenes)
    )
    transcript = "\n".join(
        f"  L{n} [{line.start_seconds:.1f}s] {line.speaker}: \"{line.text.strip()}\""
        for n, line in enumerate(analysis.audio.transcript, start=1)
    )
    dialogue = (
        "=== DIALOGUE ===\n"
        "- Do NOT write dialogue yourself. Put the L numbers listed for each part into its beats' "
        "dialogue_line_ids, in order -- every one of them; the original words are inserted later.\n"
        "- Give every beat at least as many seconds as its lines took in the source.\n"
        "- The beat `audio` field is for music and sound effects only.\n\n"
        if analysis.audio.transcript else ""
    )
    return (
        f"Below is a JSON analysis of a {media.duration_seconds:.0f}-second source video "
        f"({len(analysis.scenes)} scenes, {media.aspect_ratio}). Recreate ALL of it, faithfully, as "
        f"{len(windows)} consecutive {part_seconds}-second {media.aspect_ratio} clips that play back to back.\n\n"
        "=== PARTS ===\n" + "\n".join(rows) + "\n\n"
        "=== PART RULES (mandatory) ===\n"
        f"- Return exactly {len(windows)} parts, in order. Each part's beat durations sum to exactly "
        f"{part_seconds}s, each beat at least 1.0s.\n"
        "- Each part recreates only its own source window: the same scenes, in the same order, with the same "
        "events. List the S numbers each beat shows in source_scenes; every scene listed for a part must "
        "appear in that part.\n"
        "- Continuity: each part's first beat starts exactly where the previous part's last beat ends -- same "
        "characters, pose, position, framing and lighting. No recap, no intro, no outro.\n"
        "- Characters, visual style, setting and colour are written once at the top level and shared by every "
        "part.\n\n"
        + _shared_rules(media, dialogue)
        + f"=== SOURCE SCENE LIST ({len(analysis.scenes)} scenes) ===\n{scene_lines}\n\n"
        + (f"=== TRANSCRIPT ===\n{transcript}\n\n" if transcript else "")
        + f"=== FULL SOURCE ANALYSIS (JSON) ===\n{analysis.model_dump_json(indent=2)}"
    )


def generate_full_length(result: "AnalysisResult", part_seconds: int, notify: Notify = _noop_notify,
                         include_social: bool = True) -> GeneratedPrompt:
    """The whole source as consecutive part_seconds prompts (one Gemini call):
    each part covers its own window of the source, opens where the previous
    part ended, and shares one character / style description."""
    from google.genai import types as genai_types

    client = get_client()
    analysis = VideoAnalysis.model_validate(result.analysis)
    media = MediaInfo(**result.media)
    windows = part_windows(media.duration_seconds, media.cuts, part_seconds)
    request = _full_length_prompt(analysis, media, part_seconds, windows)

    def run(model: str) -> FullLengthPlan:
        response = client.models.generate_content(
            model=model,
            contents=[request],
            config=genai_types.GenerateContentConfig(
                http_options=gemini_http_options(TIMEOUT_VIDEO_SECONDS),  # a long, multi-part reply
                response_mime_type="application/json",
                response_schema=FullLengthPlan,
                temperature=0.5,
                automatic_function_calling=genai_types.AutomaticFunctionCallingConfig(disable=True),
            ),
        )
        plan = _parse_structured(response, FullLengthPlan)
        if len(plan.parts) != len(windows) or any(not part.beats for part in plan.parts):
            raise _RetryableResponse(f"Gemini returned {len(plan.parts)} parts instead of {len(windows)}.")
        return plan

    notify(f"Writing {len(windows)} consecutive {part_seconds}-second prompts...")
    full = call_gemini(run, notify, task="text", label="full-length")
    transcript = analysis.audio.transcript
    shared = full.model_dump(exclude={"parts"})
    shared["reference_images"] = _normalize_references(full.reference_images, 1 << 10)
    prompts: list[str] = []
    beat_groups: list[list[PromptBeat]] = []
    warnings: list[str] = []
    previous_end = ""
    for number, (window, part) in enumerate(zip(windows, full.parts), start=1):
        _, allowed = _window_contents(analysis, window, number == len(windows))
        beats = [beat.model_copy(update={"dialogue_line_ids": [n for n in beat.dialogue_line_ids if n in allowed]})
                 for beat in _clean_dialogue_ids(part.beats, len(transcript))]
        beats = _normalize_durations(beats, part_seconds)
        beat_groups.append(beats)
        dropped = [n for n in allowed if n not in {i for b in beats for i in b.dialogue_line_ids}]
        if dropped:
            warnings.append(f"Part {number}: dialogue line(s) {', '.join(f'L{n}' for n in dropped)} left out.")
        part_plan = RecreationPlan(**shared, beats=beats)
        header = [f"PART {number} OF {len(windows)} -- recreates the source from {window[0]:g}s to {window[1]:g}s."]
        if previous_end:
            header.append(f"CONTINUITY: this clip continues directly from part {number - 1}; its first frame "
                          f"matches that part's last frame exactly: {previous_end}")
        prompts.append("\n".join(header) + "\n\n" + render_prompt(part_plan, media, part_seconds, analysis.audio,
                                                                  mode=FAITHFUL))
        previous_end = beats[-1].visual.strip()

    first_plan = RecreationPlan(**shared, beats=beat_groups[0])
    report = accuracy_report(beat_groups, analysis)
    if report["missing_scenes"]:
        warnings.append(f"Source scene(s) {', '.join(f'S{n}' for n in report['missing_scenes'])} are in no part.")
    generated = GeneratedPrompt(
        duration=part_seconds,
        prompt="\n\n".join(f"=== PART {n} OF {len(prompts)} ===\n\n{p}" for n, p in enumerate(prompts, start=1)),
        plan=first_plan.model_dump(),
        saved_to="",
        references=[ref.model_dump() for ref in first_plan.reference_images],
        reference_sheet=render_reference_sheet(first_plan),
        warnings=warnings,
        mode=FULL_LENGTH,
        accuracy=report,
        parts=prompts,
    )
    return _save_generated(generated, first_plan, result, notify, include_social, f"{part_seconds}s_full")


# ============================================================================
# Orchestration + cache
# ============================================================================

@dataclass
class AnalysisResult:
    video_sha: str
    video_path: str
    source_url: str
    title: str
    extractor: str
    media: dict
    analysis: dict
    from_cache: bool = field(default=False)
    analysis_version: int = 0


# Bumped when the analysis itself gets better, so older cached analyses are
# redone once instead of being reused. 2: original file at 2-4 fps with
# measured cuts (1 was a 2 fps CRF 28 re-encode with guessed cuts).
ANALYSIS_VERSION = 2


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _url_index_path() -> Path:
    return CONFIG.work_dir / "cache" / "url_index.json"


def _read_url_index() -> dict[str, str]:
    try:
        return json.loads(_url_index_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _write_url_index(url: str, video_sha: str) -> None:
    index = _read_url_index()
    index[url] = video_sha
    _url_index_path().parent.mkdir(parents=True, exist_ok=True)
    _url_index_path().write_text(json.dumps(index, indent=2), encoding="utf-8")


def _cache_path(video_sha: str) -> Path:
    return CONFIG.work_dir / "cache" / f"{video_sha}.json"


def _load_cached(video_sha: str) -> AnalysisResult | None:
    path = _cache_path(video_sha)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("analysis_version", 0) < ANALYSIS_VERSION:
            logger.info(f"[CACHE] {video_sha[:12]} was analysed by an older, less accurate version -- re-analysing")
            return None
        audio = payload.get("analysis", {}).get("audio", {})
        if "transcript" not in audio:
            # Cached before transcripts existed. With speech it must be
            # re-analysed, or the dialogue would be lost; without, upgrade it.
            if audio.get("has_speech"):
                logger.info(f"[CACHE] {video_sha[:12]} has speech but no transcript -- re-analysing")
                return None
            audio["transcript"] = []
        result = AnalysisResult(**payload)
        VideoAnalysis.model_validate(result.analysis)
        # Video file may have been deleted after analysis — that's OK,
        # we only need the analysis JSON to generate prompts.
        result.from_cache = True
        return result
    except (OSError, ValueError, TypeError):
        return None


def analyze_url(url: str, notify: Notify = _noop_notify) -> AnalysisResult:
    """Download -> validate -> analyse, reusing a cached analysis whenever
    the same URL (or the same video bytes) was analysed before, so switching
    between 8 s and 10 s or re-pasting a link costs no Gemini quota."""
    safe_url = validate_public_video_url(url)

    known_sha = _read_url_index().get(safe_url)
    if known_sha and (cached := _load_cached(known_sha)):
        notify("Using saved analysis for this URL")
        logger.info(f"[CACHE] hit by url sha={known_sha[:12]}")
        return cached

    downloads = CONFIG.work_dir / "downloads"
    temp_dir = downloads / f"tmp_{uuid.uuid4().hex[:10]}"
    try:
        notify("Downloading video...")
        downloaded = download_video(safe_url, temp_dir, notify)
        notify("Checking video...")
        media = probe_video(downloaded.path)
        validate_media(media)

        video_sha = _sha256(downloaded.path)
        final_path = downloads / f"{video_sha}.mp4"
        if not final_path.exists():
            shutil.move(str(downloaded.path), final_path)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    _write_url_index(safe_url, video_sha)
    if cached := _load_cached(video_sha):
        notify("Using saved analysis for this video")
        logger.info(f"[CACHE] hit by content sha={video_sha[:12]}")
        final_path.unlink(missing_ok=True)
        return cached

    notify("Measuring the cuts...")
    media.cuts = detect_scene_cuts(final_path)
    analysis = analyze_video(final_path, media, notify)

    # Immediately delete the downloaded video — only the prompt text is kept.
    # The analysis JSON is written to cache (small); the video is no longer needed.
    try:
        final_path.unlink(missing_ok=True)
        logger.info(f"[CLEANUP] deleted video after analysis: {final_path.name}")
    except OSError as exc:
        logger.warning(f"[CLEANUP] could not delete video {final_path.name}: {exc}")

    result = AnalysisResult(
        video_sha=video_sha,
        video_path=str(final_path),  # path stored for cache key, file is gone
        source_url=downloaded.source_url,
        title=downloaded.title,
        extractor=downloaded.extractor,
        media=asdict(media),
        analysis=analysis.model_dump(),
        analysis_version=ANALYSIS_VERSION,
    )
    _cache_path(video_sha).parent.mkdir(parents=True, exist_ok=True)
    _cache_path(video_sha).write_text(json.dumps(asdict(result), indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def analyze_local_video(path: Path, title: str = "Uploaded video", notify: Notify = _noop_notify) -> AnalysisResult:
    """Validate and analyse a locally provided/uploaded video file, with caching.
    The file is always deleted afterwards (success, failure or cache hit)."""
    try:
        notify("Checking video...")
        media = probe_video(path)
        validate_media(media)

        video_sha = _sha256(path)
        if cached := _load_cached(video_sha):
            notify("Using saved analysis for this video")
            logger.info(f"[CACHE] hit by content sha={video_sha[:12]}")
            return cached

        notify("Measuring the cuts...")
        media.cuts = detect_scene_cuts(path)
        analysis = analyze_video(path, media, notify)
    finally:
        try:
            path.unlink(missing_ok=True)
            logger.info(f"[CLEANUP] deleted uploaded video: {path.name}")
        except OSError as exc:
            logger.warning(f"[CLEANUP] could not delete uploaded video {path.name}: {exc}")

    result = AnalysisResult(
        video_sha=video_sha,
        video_path=str(path),
        source_url=f"upload://{title}",
        title=title,
        extractor="upload",
        media=asdict(media),
        analysis=analysis.model_dump(),
        analysis_version=ANALYSIS_VERSION,
    )
    _cache_path(video_sha).parent.mkdir(parents=True, exist_ok=True)
    _cache_path(video_sha).write_text(json.dumps(asdict(result), indent=2, ensure_ascii=False), encoding="utf-8")
    return result


USER_FACING_ERRORS = (URLVideoError, VideoValidationError, GeminiCallError, QuotaExhaustedError, RuntimeError, ValueError)


# ============================================================================
# Streamlit UI
# ============================================================================

def _supported_kwargs(func: Callable, **kwargs: Any) -> dict[str, Any]:
    """Drops keyword arguments the installed Streamlit doesn't accept, so the
    app runs on older versions (e.g. 1.37, where form_submit_button has no
    icon= and st.code has no wrap_lines=) as well as current ones."""
    import inspect

    try:
        params = inspect.signature(func).parameters
    except (TypeError, ValueError):
        return {}
    return {key: value for key, value in kwargs.items() if key in params}


def _show_copyable(st, text: str) -> None:
    st.code(text, language=None, **_supported_kwargs(st.code, wrap_lines=True))


def render_social(st, social: "SocialMetadata | None") -> None:
    """YouTube / Instagram metadata with copy boxes and the platform counts."""
    if not social:
        st.warning("Social metadata generation failed — try regenerating.")
        return
    if social.grounded:
        st.caption(f"🔎 Keywords researched with Google Search ({len(social.search_queries)} searches). Edit before posting.")
    else:
        st.caption(
            "⚠️ Not grounded in Google Search (today's free search quota is used up, or the search "
            "model was unavailable) — keywords come from Gemini's own knowledge. Edit before posting."
        )
    tab_yt, tab_ig = st.tabs(["▶️ YouTube", "📸 Instagram"])
    with tab_yt:
        st.markdown("**📌 Title** — paste into the YouTube title field")
        _show_copyable(st, social.youtube_title)
        st.caption(f"{len(social.youtube_title)}/{YOUTUBE_TITLE_MAX} characters")
        st.markdown("**📄 Description**")
        _show_copyable(st, social.youtube_description)
        st.caption(f"{len(social.youtube_description.encode('utf-8'))}/{YOUTUBE_DESCRIPTION_MAX_BYTES} bytes")
        tags = split_tags(social.youtube_tags)
        st.markdown(
            f"**🏷️ Tags** — {len(tags)} tags · {youtube_tags_cost(tags)}/{YOUTUBE_TAGS_LIMIT} "
            "as YouTube counts them (quotes included)"
        )
        _show_copyable(st, social.youtube_tags)
    with tab_ig:
        st.markdown("**✍️ Caption + hashtags** — one copy, one paste on your phone")
        _show_copyable(st, f"{social.instagram_caption.rstrip()}\n\n{social.instagram_hashtags}")
        st.caption(
            f"{len(social.instagram_caption)}/{INSTAGRAM_CAPTION_MAX} characters · "
            f"{len(social.instagram_hashtags.split())}/{INSTAGRAM_HASHTAGS_MAX} hashtags"
        )
        with st.expander("Hashtags only (e.g. for the first comment)"):
            _show_copyable(st, social.instagram_hashtags)
    if social.search_queries or social.sources:
        with st.expander("Searches and sources used"):
            if social.search_queries:
                st.markdown("**Searches:** " + "; ".join(social.search_queries))
            for source in social.sources:
                st.markdown(f"- [{source['title']}]({source['uri']})")


def render_social_or_generate(st, social: "SocialMetadata | None", key: str,
                              make: Callable[[Notify], "SocialMetadata"]) -> None:
    """Shows the metadata. When it is missing, generates it here -- the
    prompt is already on screen above -- and on failure shows why, with a
    Retry button, instead of silently leaving it out. `make` generates and
    stores it and returns the SocialMetadata."""
    errors = st.session_state.setdefault("social_errors", {})
    if social is None and key not in errors:
        with st.status("Researching keywords and writing the metadata...", expanded=False) as status:
            try:
                social = make(lambda message: status.update(label=message))
            except USER_FACING_ERRORS as exc:
                errors[key] = str(exc)
                status.update(label="Metadata failed", state="error")
            else:
                status.update(label="Metadata ready", state="complete")
    if social is None:
        st.warning(f"The metadata could not be generated: {errors.get(key, 'unknown error')}")
        if st.button("Retry metadata", key=f"retry_social_{key}"):
            errors.pop(key, None)
            st.rerun()
        return
    render_social(st, social)


def _render_generated(st, generated: "GeneratedPrompt", result: "AnalysisResult | None" = None,
                      prompt_key: str | None = None) -> None:
    references = getattr(generated, "references", []) or []
    full_text = generated.full_text() if hasattr(generated, "full_text") else generated.prompt
    duration = generated.duration
    parts = getattr(generated, "parts", []) or []
    prompt_key = prompt_key or str(duration)

    if not CONFIG.show_prompt:
        st.success(
            f"{duration}-second prompt ready ({len(references)} reference image(s)). "
            "On-screen display is off (SHOW_PROMPT=false) -- download it below."
        )
    else:
        title = f"{len(parts)} consecutive prompts ({duration} s each)" if parts else f"Video generation prompt ({duration} s)"
        st.subheader(title)
        st.caption(
            "Free-tier Gemini only writes text -- no video or image is generated here. "
            "Paste this into your video generator (Seedance, Omni, Veo, Kling, ...)."
        )
        score = accuracy_line(getattr(generated, "accuracy", {}) or {})
        if score:
            st.info(f"**Accuracy:** {score}")
        for warning in getattr(generated, "warnings", []) or []:
            st.warning(warning)
        if parts:
            st.caption("Generate the parts in order; each one continues from the last frame of the previous one.")
            for number, part in enumerate(parts, start=1):
                with st.expander(f"Part {number} of {len(parts)}", expanded=number == 1):
                    _show_copyable(st, part)
        else:
            _show_copyable(st, generated.prompt)

        st.subheader("Reference images to attach")
        if not references:
            st.info("No reference images needed -- the video prompt is text-only. Paste it as-is.")
        else:
            st.info(
                "Generate these images first with any text-to-image model, then attach them to the "
                "video generator as described. The video prompt refers to them as [REF_n]. If your "
                "generator accepts only one image, attach the first-frame image (or the first "
                "character reference)."
            )
            for ref in references:
                purpose = ref.get("purpose", "")
                with st.container(border=True):
                    st.markdown(
                        f"**[{ref.get('ref_id')}] {REFERENCE_LABELS.get(purpose, purpose)}** -- "
                        f"{ref.get('what_it_shows', '')}"
                    )
                    st.markdown(f"**How to attach:** {ATTACH_INSTRUCTIONS.get(purpose, '')}")
                    st.caption(
                        "Used in scenes: " + ", ".join(str(n) for n in ref.get("used_in_scenes", []))
                    )
                    st.markdown("**Image generation prompt:**")
                    _show_copyable(st, ref.get("image_prompt", ""))

    # ── Social Media Metadata ──────────────────────────────────────────────
    st.divider()
    st.subheader("📲 Post to Social Media")
    social = getattr(generated, "social_metadata", None)
    if result is None:
        render_social(st, social)
    else:
        def make(notify: Notify) -> "SocialMetadata":
            updated = add_social_metadata(generated, result, notify)
            st.session_state.video_prompts[prompt_key] = updated
            return updated.social_metadata

        render_social_or_generate(st, social, f"video_{result.video_sha[:12]}_{prompt_key}", make)

    st.divider()
    st.download_button(
        "Download prompts + metadata",
        data=full_text,
        file_name=f"recreate_{prompt_key}.txt",
        mime="text/plain",
        **_supported_kwargs(st.download_button, icon=":material/download:"),
    )
    st.caption(f"Saved to {generated.saved_to}")


def _new_upload(state: Any, key: str, uploaded: Any) -> bytes | None:
    """The uploaded bytes if this exact file has not been analysed yet in this
    session, else None. Files are told apart by a hash of their content, not
    their name: a new 'image.jpg' is analysed, and Streamlit reruns of the same
    upload are not. Removing the file resets it, so re-adding retries."""
    if uploaded is None:
        state.pop(key, None)
        return None
    data = uploaded.getvalue()
    digest = hashlib.sha256(data).hexdigest()
    if state.get(key) == digest:
        return None
    state[key] = digest
    return data


def _save_upload(data: bytes, name: str, folder: str, default_suffix: str) -> Path:
    """Writes an upload to WORK_DIR/folder just before analysis; the analysis
    step deletes it."""
    uploads_dir = CONFIG.work_dir / folder
    uploads_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(name).suffix.lower() or default_suffix
    path = uploads_dir / f"upload_{uuid.uuid4().hex[:8]}{suffix}"
    path.write_bytes(data)
    return path


def _show_image_result(st, status, image_result) -> None:
    st.session_state.image_result = image_result
    st.session_state.image_generated = None
    label = "Analysis loaded from cache" if image_result.from_cache else "Analysis complete"
    status.update(label=label, state="complete", expanded=False)


def _render_quota_sidebar(st) -> None:
    snapshot = quota_snapshot()
    st.markdown("**Gemini free tier · today**")
    for model in CONFIG.all_models():
        reason = snapshot["unavailable"].get(model)
        cooling = snapshot["cooling"].get(model)
        status = (
            f":red[{reason}]" if reason
            else f":orange[busy -- retry in {int(cooling // 60)}m {int(cooling % 60)}s]" if cooling
            else ":green[available]"
        )
        st.caption(f"`{model}` · {snapshot['calls'].get(model, 0)} call(s) · {status}")
    search = ":red[used up]" if snapshot["search_used_up"] else ":green[available]"
    st.caption(f"Google Search for metadata · {search}")
    st.caption(
        f"Counts are this app's requests only. Free quotas reset at midnight "
        f"Pacific time (in {_format_reset_wait()})."
    )


APP_MODULES = ("recreate_video", "recreate_image", "image_to_video", "summary_short", "voiceover", "slideshow")


def drop_stale_app_modules() -> list[str]:
    """Streamlit re-runs this script on every interaction, but modules it
    imports stay loaded in the process. After an in-place update (a git push
    to Streamlit Cloud), a newer module importing a name that the still-loaded
    older recreate_video lacks fails with ImportError. When any app module's
    file changed since it was loaded, all of them are dropped so this run
    imports them fresh. Returns the dropped names."""
    stale = False
    for name in APP_MODULES:
        module = sys.modules.get(name)
        path = getattr(module, "__file__", None)
        if module is None or not path:
            continue
        try:
            changed = getattr(module, "_LOADED_MTIME", None) != Path(path).stat().st_mtime
        except OSError:
            changed = True
        stale = stale or changed
    if not stale:
        return []
    dropped = [name for name in APP_MODULES if sys.modules.pop(name, None) is not None]
    logger.info(f"[RELOAD] app files changed since they were loaded -- re-importing {', '.join(dropped)}")
    return dropped


def run_app() -> None:
    drop_stale_app_modules()
    import streamlit as st
    # Lazy import of image pipeline — only needed when image mode is active.
    from recreate_image import (
        analyze_image_url, analyze_local_image,
        USER_FACING_ERRORS as IMAGE_USER_FACING_ERRORS,
    )

    # ── 24-hour startup cleanup ──────────────────────────────────────────────
    if "_swept" not in st.session_state:
        _sweep_old_files()
        st.session_state["_swept"] = True

    st.set_page_config(page_title="Recreate any post", page_icon="🎬", layout="centered")

    # ── Sidebar navigation ───────────────────────────────────────────────────
    # An image link analysed from the Video tab asks for the Image view; the
    # radio's value can only be set before the radio is drawn.
    if st.session_state.pop("_switch_to_image", False):
        st.session_state["app_mode"] = IMAGE_MODE
    with st.sidebar:
        st.markdown("## 🎬 Recreate")
        mode = st.radio(
            "What are you recreating?",
            options=[VIDEO_MODE, IMAGE_MODE, IMAGE_TO_VIDEO_MODE, SUMMARY_MODE],
            key="app_mode",
        )
        st.divider()
        st.caption(
            "Paste any public link. The app auto-detects image vs video. "
            "Use this toggle to override if needed."
        )
        st.divider()
        _render_quota_sidebar(st)

    # Image to video uses only ffmpeg, so it works without a Gemini key.
    if mode == IMAGE_TO_VIDEO_MODE:
        from image_to_video import render_page

        render_page(st)
        return
    if mode == SUMMARY_MODE:  # shows its own Gemini key check on its first step
        from summary_short import render_page as render_summary_page

        render_summary_page(st)
        return

    is_image_mode = mode == IMAGE_MODE

    problems = CONFIG.problems()
    if problems:
        for problem in problems:
            st.error(problem)
        st.stop()

    # ── Title changes by mode ────────────────────────────────────────────────
    if is_image_mode:
        st.title("Recreate any image post")
        st.caption(
            "Upload a screenshot of any Instagram image post (or paste a direct image URL). "
            "Gemini analyses it and writes copy-paste prompts for image generators (any text in the image "
            "is built into the prompt) + audio + social metadata."
        )
    else:
        st.title("Recreate any video")
        st.caption(
            f"Paste a public video link (YouTube, Instagram Reel, TikTok, direct .mp4 ...) up to "
            f"{CONFIG.max_video_seconds} seconds. Gemini analyses it, then writes the prompts to recreate it."
        )

    # ── Session state ────────────────────────────────────────────────────────
    st.session_state.setdefault("video_result", None)
    st.session_state.setdefault("video_prompts", {})
    st.session_state.setdefault("image_result", None)
    st.session_state.setdefault("image_generated", None)

    # ── Image mode: upload OR URL ────────────────────────────────────────────
    if is_image_mode:
        st.info(
            "📸 **Instagram blocks automated downloads.** "
            "Take a screenshot of the post or save the image, then upload it below. "
            "Or paste a direct image URL (.jpg / .png)."
        )
        uploaded_file = st.file_uploader(
            "Upload image (screenshot / saved post)",
            type=["jpg", "jpeg", "png", "webp"],
            key="image_upload",
        )
        with st.form("image_url_form"):
            img_url = st.text_input(
                "Or paste a direct image URL (.jpg / .png)",
                placeholder="https://example.com/image.jpg",
            )
            img_submitted = st.form_submit_button("🔍 Analyze image", type="primary")

        # ── Analyze a new upload, or a submitted URL ──────────────────────────
        image_upload = _new_upload(st.session_state, "_last_image_upload", uploaded_file)
        if image_upload is not None:
            temp_img = _save_upload(image_upload, uploaded_file.name, "image_downloads", ".jpg")
            with st.status("Analysing image...", expanded=True) as status:
                try:
                    image_result = analyze_local_image(
                        temp_img, uploaded_file.name, lambda msg: status.update(label=msg)
                    )
                except (IMAGE_USER_FACING_ERRORS + USER_FACING_ERRORS) as exc:
                    status.update(label="Analysis failed", state="error")
                    st.error(str(exc))
                else:
                    _show_image_result(st, status, image_result)
        elif img_submitted and img_url.strip():
            with st.status("Downloading and analysing image...", expanded=True) as status:
                try:
                    image_result = analyze_image_url(img_url, lambda msg: status.update(label=msg))
                except (IMAGE_USER_FACING_ERRORS + USER_FACING_ERRORS) as exc:
                    status.update(label="Analysis failed", state="error")
                    st.error(str(exc))
                else:
                    _show_image_result(st, status, image_result)

        _render_image_results(st, is_image_mode)
        return  # image mode done — skip video form below

    # ── Video mode: Upload file or URL input ──────────────────────────────────
    uploaded_video = st.file_uploader(
        "Upload video file (if private, login-blocked, or saved on your device)",
        type=["mp4", "mov", "webm", "m4v"],
        key="video_upload",
        help="Upload up to 60s video. File is automatically deleted after Gemini analysis.",
    )

    video_upload = _new_upload(st.session_state, "_last_video_upload", uploaded_video)
    if video_upload is not None:
        temp_video = _save_upload(video_upload, uploaded_video.name, "video_uploads", ".mp4")

        with st.status("Analysing uploaded video...", expanded=True) as status:
            try:
                video_result = analyze_local_video(
                    temp_video,
                    title=uploaded_video.name,
                    notify=lambda msg: status.update(label=msg),
                )
            except USER_FACING_ERRORS as exc:
                status.update(label="Analysis failed", state="error")
                st.error(str(exc))
            else:
                st.session_state.video_result = video_result
                st.session_state.video_prompts = {}
                label = "Analysis loaded from cache" if video_result.from_cache else "Analysis complete"
                status.update(label=label, state="complete", expanded=False)

    with st.form("analyze_form"):
        url = st.text_input(
            "Or paste a public link (video, Reel, or Instagram photo post)",
            placeholder="https://www.youtube.com/shorts/... or https://www.instagram.com/reel/...",
        )
        submitted = st.form_submit_button("🔍 Analyze link", type="primary")

    # ── Analysis ─────────────────────────────────────────────────────────────
    # Image links go to the image pipeline. An Instagram /p/ link can be either,
    # so the video download is tried first and a photo post falls back.
    url_type = detect_url_type(url) if submitted and url.strip() else MediaType.VIDEO
    if submitted and url.strip() and url_type == MediaType.IMAGE:
        _analyze_image_link(st, url)
    elif submitted and url.strip():
        photo_post = False
        with st.status("Starting video analysis...", expanded=True) as status:
            try:
                video_result = analyze_url(
                    url, lambda msg: status.update(label=msg)
                )
            except NoVideoInPost as exc:
                if url_type == MediaType.AMBIGUOUS:
                    status.update(label="This post is a photo -- analysing it as an image", state="complete")
                    photo_post = True
                else:
                    status.update(label="Analysis failed", state="error")
                    st.error(str(exc))
            except USER_FACING_ERRORS as exc:
                status.update(label="Analysis failed", state="error")
                err_msg = str(exc)
                st.error(err_msg)
                st.info(
                    "💡 **Can't download?** If the post is private or blocked by Instagram/TikTok, "
                    "simply screen-record or save the video and upload it using the file uploader above!"
                )
                if "instagram" in err_msg.lower() or "cookies" in err_msg.lower():
                    st.info(
                        "**Or fix Instagram downloads with cookies:**\n\n"
                        "1. Install [Get cookies.txt LOCALLY](https://chromewebstore.google.com/detail/get-cookiestxt-locally/cclelndahbckbenkjhflpdbgdldlbecc) in Chrome.\n"
                        "2. Go to [instagram.com](https://www.instagram.com) while **logged in** → click extension → **Export** → save as `cookies.txt`.\n"
                        f"3. Drop `cookies.txt` into: `{Path('.').resolve()}`\n\n"
                        "`COOKIES_FILE=cookies.txt` is already set in `.env`."
                    )
            else:
                st.session_state.video_result = video_result
                st.session_state.video_prompts = {}
                label = "Analysis loaded from cache" if video_result.from_cache else "Analysis complete"
                status.update(label=label, state="complete", expanded=False)
        if photo_post:
            _analyze_image_link(st, url)

    _render_video_results(st)


IMAGE_MODE = "🖼️ Image Post"
VIDEO_MODE = "🎬 Video / Reel"
IMAGE_TO_VIDEO_MODE = "🎞️ Image to Video"
SUMMARY_MODE = "📝 Summarise to Short"


def _analyze_image_link(st, url: str) -> None:
    """An image link pasted in the Video tab: analyse it as an image post, then
    rerun into the Image view. The sidebar switch happens at the top of the
    next run, because a widget's value cannot change after it is drawn."""
    from recreate_image import USER_FACING_ERRORS as IMAGE_USER_FACING_ERRORS, analyze_image_url

    with st.status("Downloading and analysing image...", expanded=True) as status:
        try:
            image_result = analyze_image_url(url, lambda msg: status.update(label=msg))
        except (IMAGE_USER_FACING_ERRORS + USER_FACING_ERRORS) as exc:
            status.update(label="Analysis failed", state="error")
            st.error(str(exc))
            return
    st.session_state.image_result = image_result
    st.session_state.image_generated = None
    st.session_state["_switch_to_image"] = True
    st.rerun()


def _render_video_results(st) -> None:
    result: AnalysisResult | None = st.session_state.get("video_result")
    if not result:
        return

    analysis = VideoAnalysis.model_validate(result.analysis)
    media = MediaInfo(**result.media)

    with st.container(border=True):
        st.subheader(result.title)
        left, right = st.columns([2, 3])
        with left:
            # Video file is deleted after analysis — show placeholder if gone
            if Path(result.video_path).exists():
                st.video(result.video_path)
            else:
                st.caption("(video file removed after analysis — only the prompt is kept)")
        with right:
            content_type = analysis.content_type.replace("_", " ")
            medium = analysis.visual_medium.replace("_", " ")
            st.markdown(
                f":blue[**{content_type}**] · :violet[**{medium}**] · "
                f"{media.duration_seconds:.0f}s · {media.aspect_ratio}"
            )
            st.write(analysis.summary)
            st.caption(f"**Hook:** {analysis.hook}")
        transcript = analysis.audio.transcript
        if transcript:
            language = analysis.audio.spoken_language or "original language"
            with st.expander(f"Transcript ({language}, {len(transcript)} lines)"):
                st.caption(
                    "These lines go into the video prompt word for word. Only reuse "
                    "dialogue from content you own or have the rights to."
                )
                for number, line in enumerate(transcript, start=1):
                    st.markdown(
                        f"**L{number}** `{line.start_seconds:.1f}-{line.end_seconds:.1f}s` "
                        f"{line.speaker}: {line.text}"
                    )
                    if line.english.strip() and line.english.strip() != line.text.strip():
                        st.caption(line.english)
        with st.expander("Full analysis"):
            st.json(result.analysis, expanded=False)

    durations = list(CONFIG.output_durations)
    if hasattr(st, "segmented_control"):
        duration = st.segmented_control(
            "Output video length",
            options=durations,
            default=durations[0],
            format_func=lambda seconds: f"{seconds} seconds",
            key="duration",
            **_supported_kwargs(st.segmented_control, required=True),
        ) or durations[0]
    else:
        duration = st.radio(
            "Output video length",
            options=durations,
            index=0,
            format_func=lambda seconds: f"{seconds} seconds",
            horizontal=True,
            key="duration",
        )

    part_count = len(part_windows(media.duration_seconds, media.cuts, duration))
    mode_labels = {
        FAITHFUL: "Faithful — one prompt, same scenes and order as the source",
        VIRAL: "Viral — one prompt, hook first",
        FULL_LENGTH: f"Full length — {part_count} × {duration} s prompts covering the whole video",
    }
    modes = [FAITHFUL, VIRAL] + ([FULL_LENGTH] if part_count > 1 else [])
    mode = st.radio("Recreation mode", modes, format_func=mode_labels.get, key="recreation_mode")
    prompt_key = f"{mode}_{duration}"

    if st.button(
        "Generate prompt", type="primary",
        **_supported_kwargs(st.button, icon=":material/auto_awesome:"),
    ):
        with st.status(f"Writing the {duration}-second prompt...", expanded=True) as status:
            try:
                generated = generate_prompt(result, duration, lambda msg: status.update(label=msg),
                                            include_social=False, mode=mode)
            except USER_FACING_ERRORS as exc:
                status.update(label="Prompt generation failed", state="error")
                st.error(str(exc))
            else:
                st.session_state.video_prompts[prompt_key] = generated
                st.session_state.setdefault("social_errors", {}).pop(
                    f"video_{result.video_sha[:12]}_{prompt_key}", None)
                status.update(label="Prompt ready", state="complete", expanded=False)

    generated: GeneratedPrompt | None = st.session_state.get("video_prompts", {}).get(prompt_key)
    if generated:
        _render_generated(st, generated, result, prompt_key)


def _render_image_results(st, is_image_mode: bool) -> None:
    from recreate_image import (
        analyze_image_url, generate_image_recreation,
        ImageAnalysisResult, GeneratedImagePrompt, ImageAnalysis,
        ImageRecreationPlan,
        render_audio_section,
        USER_FACING_ERRORS as IMAGE_USER_FACING_ERRORS,
    )

    result: ImageAnalysisResult | None = st.session_state.get("image_result")
    if not result:
        return

    from recreate_image import ImageAnalysis as _IA
    analysis = _IA.model_validate(result.analysis)

    with st.container(border=True):
        st.subheader(result.title)
        left, right = st.columns([1, 2])
        with left:
            from pathlib import Path as _Path
            if _Path(result.image_path).exists():
                st.image(result.image_path, use_column_width=True)
            else:
                st.caption("(image removed after analysis)")
        with right:
            content = analysis.content_type.replace("_", " ")
            medium = analysis.visual_medium.replace("_", " ")
            st.markdown(f":blue[**{content}**] · :violet[**{medium}**] · {analysis.aspect_ratio}")
            if analysis.has_text:
                st.markdown(f":orange[**Text in the image:** {len(analysis.text_overlays)} — written into the prompts]")
            st.write(analysis.summary)
            st.caption(f"**Hook:** {analysis.hook}")
            audio = analysis.suggested_audio
            st.caption(
                f"**Suggested audio:** {audio.track_name} — {audio.artist_name} "
                f"| Search: `{audio.instagram_search_query}`"
            )
        with st.expander("Full analysis"):
            st.json(result.analysis, expanded=False)

    if st.button(
        "Generate recreation prompts", type="primary",
        **_supported_kwargs(st.button, icon=":material/auto_awesome:"),
        key="gen_image_btn",
    ):
        with st.status("Generating prompts...", expanded=True) as status:
            try:
                gen = generate_image_recreation(result, lambda msg: status.update(label=msg), include_social=False)
            except IMAGE_USER_FACING_ERRORS as exc:
                status.update(label="Generation failed", state="error")
                st.error(str(exc))
            else:
                st.session_state.image_generated = gen
                st.session_state.setdefault("social_errors", {}).pop(f"image_{result.image_sha[:12]}", None)
                status.update(label="Prompts ready", state="complete", expanded=False)

    gen: GeneratedImagePrompt | None = st.session_state.get("image_generated")
    if not gen:
        return

    from recreate_image import ImageRecreationPlan as _IRP
    plan = _IRP.model_validate(gen.plan)

    if CONFIG.show_prompt:
        tab_nb, tab_flux, tab_sd, tab_audio, tab_social = st.tabs([
            "🍌 Nano Banana", "🌀 FLUX / MJ", "🖥️ Stable Diffusion", "🎵 Audio", "📲 Social",
        ])
        with tab_nb:
            st.subheader("Nano Banana / Nano Banana Pro")
            st.caption("One complete prompt, including any text in the image. Paste it as-is.")
            _show_copyable(st, plan.nano_banana_prompt)
        with tab_flux:
            st.subheader("FLUX / Midjourney / Leonardo")
            _show_copyable(st, plan.flux_midjourney_prompt)
        with tab_sd:
            st.subheader("Stable Diffusion / ComfyUI")
            st.markdown("**Positive:**")
            _show_copyable(st, plan.stable_diffusion_prompt)
            st.markdown("**Negative:**")
            _show_copyable(st, plan.negative_prompt)
        with tab_audio:
            st.subheader("Recommended Audio")
            audio = plan.suggested_audio
            with st.container(border=True):
                st.markdown(f"**{audio.track_name}** by {audio.artist_name}")
                st.markdown(f"Genre: {audio.genre} | Feel: {audio.bpm_feel}")
                st.caption(audio.mood_match)
                st.markdown("**Search on Instagram:**")
                _show_copyable(st, audio.instagram_search_query)
            with st.container(border=True):
                st.markdown(f"**Alt: {audio.alternative_track_name}** by {audio.alternative_artist_name}")
                _show_copyable(st, audio.alternative_search_query)
        with tab_social:
            st.subheader("📲 Social Media")
            from recreate_image import add_image_social

            def make(notify: Notify) -> "SocialMetadata":
                updated = add_image_social(gen, result, notify)
                st.session_state.image_generated = updated
                return updated.social_metadata

            render_social_or_generate(st, gen.social_metadata, f"image_{result.image_sha[:12]}", make)

    st.divider()
    st.download_button(
        "Download all prompts",
        data=gen.full_text(),
        file_name="image_recreation.txt",
        mime="text/plain",
        **_supported_kwargs(st.download_button, icon=":material/download:"),
    )
    st.caption(f"Saved to {gen.saved_to}")


# ============================================================================
# CLI
# ============================================================================

def run_cli() -> int:
    parser = argparse.ArgumentParser(description="Turn any video URL into a recreation prompt.")
    parser.add_argument("--url", required=True, help="Public video URL (max 60 s).")
    parser.add_argument("--duration", type=int, choices=CONFIG.output_durations, default=CONFIG.output_durations[0])
    parser.add_argument("--mode", choices=[FAITHFUL, VIRAL, FULL_LENGTH], default=FAITHFUL,
                        help="faithful: same scenes and order (default); viral: hook first; "
                             "full_length: consecutive prompts covering the whole video.")
    args = parser.parse_args()

    for problem in CONFIG.problems():
        print(f"Config error: {problem}", file=sys.stderr)
    if CONFIG.problems():
        return 2

    def notify(message: str) -> None:
        print(f"\r{message:<90}", end="", flush=True)

    try:
        result = analyze_url(args.url, notify)
        print(f"\rAnalysed: {result.title} ({'cache' if result.from_cache else 'fresh'}){' ' * 40}")
        generated = generate_prompt(result, args.duration, notify, mode=args.mode)
    except USER_FACING_ERRORS as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        return 1
    print("\r" + " " * 90 + "\r", end="")
    if CONFIG.show_prompt:
        print(generated.full_text())
    print(f"(saved to {generated.saved_to})")
    return 0


def _running_in_streamlit() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx() is not None
    except Exception:
        return False


if __name__ == "__main__":
    if _running_in_streamlit():
        run_app()
    else:
        sys.exit(run_cli())
