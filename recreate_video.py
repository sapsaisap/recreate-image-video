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
import json
import logging
import os
import random
import re
import shutil
import subprocess
import sys
import threading
import time
import types
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, TypeVar
from urllib.parse import urlsplit

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
        for key in ("GEMINI_API_KEY", "COOKIES_FILE", "DOWNLOAD_PROXY"):
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


@dataclass(frozen=True)
class Config:
    api_key: str = os.getenv("GEMINI_API_KEY", "").strip()
    model: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()
    fallback_models: tuple[str, ...] = tuple(_env_list("GEMINI_FALLBACK_MODELS", "gemini-2.5-flash-lite"))
    rpm: int = max(1, _env_int("GEMINI_RPM", 8))
    max_wait_seconds: int = max(30, _env_int("RATE_LIMIT_MAX_WAIT_SECONDS", 900))
    max_video_seconds: int = _env_int("MAX_VIDEO_SECONDS", 60)
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
    # The free tier can only produce text, so the video/image prompts ARE the
    # output. true: show them on screen; false: only save + offer download.
    show_prompt: bool = os.getenv("SHOW_PROMPT", "true").strip().lower() not in {"false", "0", "no", "off"}

    @property
    def models(self) -> list[str]:
        return [self.model, *[m for m in self.fallback_models if m != self.model]]

    def problems(self) -> list[str]:
        issues = []
        if not self.api_key:
            issues.append(
                "GEMINI_API_KEY is not set. "
                "Add it to .env (local) or App Settings → Secrets (Streamlit Cloud)."
            )
        if self.cookies_file and not Path(self.cookies_file).is_file():
            issues.append(f"COOKIES_FILE points to a missing file: {self.cookies_file}")
        return issues


CONFIG = Config()


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
    if "youtube.com" in host or "youtu.be" in host:
        return MediaType.VIDEO

    # TikTok
    if "tiktok.com" in host:
        return MediaType.VIDEO

    # Twitter / X
    if "twitter.com" in host or "x.com" in host:
        return MediaType.VIDEO

    # Facebook
    if "facebook.com" in host or "fb.watch" in host:
        return MediaType.VIDEO

    # Instagram
    if "instagram.com" in host:
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

def _load_cookies_from_file(cookie_file: str) -> dict[str, str]:
    """Parse a Netscape cookies.txt file into a plain dict."""
    cookies: dict[str, str] = {}
    try:
        with open(cookie_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split("\t")
                if len(parts) >= 7:
                    cookies[parts[5]] = parts[6]
    except OSError:
        pass
    return cookies


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
    import urllib.parse
    try:
        import requests as _requests
    except ImportError as exc:
        raise URLVideoError(
            "The 'requests' library is not installed. Run: pip install requests"
        ) from exc

    safe_url = validate_public_video_url(url)

    # Load cookies for Instagram login
    cookies: dict[str, str] = {}
    if CONFIG.cookies_file:
        cookies = _load_cookies_from_file(CONFIG.cookies_file)
    if not cookies:
        raise URLVideoError(
            "Instagram requires a logged-in session. "
            "Export your cookies to cookies.txt and set COOKIES_FILE in .env."
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
        resp = _requests.get(safe_url, cookies=cookies, headers=headers, timeout=20)
        resp.raise_for_status()
    except Exception as exc:
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
        img_resp = _requests.get(img_url, timeout=30)
        img_resp.raise_for_status()
    except Exception as exc:
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
    """Every configured model has used up its daily free-tier quota."""


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


def _retry_delay_seconds(exc: Exception) -> float | None:
    """Google's 429 body carries a RetryInfo detail ("retryDelay": "37s") and
    usually also says "Please retry in 37.2s" in the message."""
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
    text = str(getattr(exc, "details", "") or "") + " " + str(exc)
    return bool(re.search(r"per ?day", text, flags=re.IGNORECASE))


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
    label: str = "Gemini",
    throttle: bool = True,
) -> T:
    """Runs fn(model_name) until it succeeds.

    - 429 per-minute limits: wait Google's retryDelay (or 5/10/20/40/60 s
      backoff) and retry the same model, until RATE_LIMIT_MAX_WAIT_SECONDS.
    - 429 per-day limits: waiting cannot help, so switch to the next model
      in GEMINI_FALLBACK_MODELS; QuotaExhaustedError once all are spent.
    - 5xx/overloaded/timeout: backoff on the same model, moving to the next
      model after 3 consecutive failures (free-tier 503 "high demand" spells
      often last minutes, and each failed call can itself take ~90 s).
    - Anything else (bad key, bad request, safety block): GeminiCallError now.
    """
    models = CONFIG.models
    index = 0
    failures_on_model = 0
    deadline = time.monotonic() + CONFIG.max_wait_seconds

    while True:
        model = models[index]
        if throttle:
            _throttle()
        try:
            return fn(model)
        except (GeminiCallError, QuotaExhaustedError):
            raise
        except Exception as exc:
            kind = _classify(exc)
            if kind == "fatal":
                logger.error(f"[GEMINI] {label} model={model} fatal: {exc}")
                raise GeminiCallError(_readable_error(exc)) from exc

            if kind == "rate" and _is_daily_quota(exc):
                logger.warning(f"[GEMINI] {label} model={model} daily quota exhausted")
                if index + 1 < len(models):
                    index += 1
                    failures_on_model = 0
                    notify(f"Daily free quota used up on {model}, switching to {models[index]}")
                    continue
                raise QuotaExhaustedError(
                    "The daily free-tier Gemini quota is used up for every configured model "
                    f"({', '.join(models)}). It resets at midnight Pacific time -- try again "
                    "then, or add another model to GEMINI_FALLBACK_MODELS."
                ) from exc

            failures_on_model += 1
            if kind == "transient" and failures_on_model > 2 and index + 1 < len(models):
                logger.warning(f"[GEMINI] {label} model={model} keeps failing, switching to {models[index + 1]}")
                index += 1
                failures_on_model = 0
                continue

            delay = _retry_delay_seconds(exc) if kind == "rate" else None
            if delay is None:
                delay = min(60.0, 5.0 * 2 ** (failures_on_model - 1))
            delay += random.uniform(1.0, 3.0)
            remaining = deadline - time.monotonic()
            if delay > remaining:
                raise GeminiCallError(
                    f"Gemini was still rate-limited/unavailable after waiting "
                    f"{CONFIG.max_wait_seconds}s. Try again in a few minutes, or raise "
                    "RATE_LIMIT_MAX_WAIT_SECONDS."
                ) from exc
            reason = (
                f"Free-tier rate limit hit on {model}"
                if kind == "rate"
                else f"Gemini busy/unavailable on {model}"
            )
            logger.warning(f"[GEMINI] {label} model={model} {kind} failure, waiting {delay:.1f}s: {exc}")
            _wait_with_countdown(delay, reason, notify)


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
    return value


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


def download_video(url: str, destination: Path, notify: Notify = _noop_notify) -> DownloadedVideo:
    from yt_dlp import YoutubeDL
    from yt_dlp.utils import DownloadError

    safe_url = validate_public_video_url(url)
    destination.mkdir(parents=True, exist_ok=True)
    max_bytes = CONFIG.max_download_mb * 1024 * 1024
    rejection: dict[str, str] = {}

    def match_filter(info: dict, *, incomplete: bool):
        reason = None
        if info.get("is_live") or info.get("live_status") in {"is_live", "is_upcoming"}:
            reason = "Live streams are not supported."
        duration = info.get("duration")
        if not reason and duration and duration > CONFIG.max_video_seconds:
            reason = (
                f"This video is {duration:.0f} seconds long. The maximum is "
                f"{CONFIG.max_video_seconds} seconds."
            )
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
        # Capped at 720p: Gemini samples video at low resolution anyway, and a
        # smaller file uploads faster.
        "format": (
            "bv*[ext=mp4][height<=720]+ba[ext=m4a]/b[ext=mp4][height<=720]/"
            "bv*[height<=720]+ba/b[height<=720]/bv*+ba/b"
        ),
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
        "js_runtimes": {"node": {}},
        "extractor_args": {
            "youtube": {
                "player_client": ["android", "ios", "web"],
            }
        },
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


def validate_media(media: MediaInfo) -> None:
    if media.duration_seconds > CONFIG.max_video_seconds + 0.5:
        raise VideoValidationError(
            f"This video is {media.duration_seconds:.0f} seconds long. "
            f"The maximum is {CONFIG.max_video_seconds} seconds."
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
    audio: str = Field(description="What is heard in this scene: speech (paraphrased), music, sound effects.")


class AudioProfile(BaseModel):
    has_speech: bool
    spoken_language: str = Field(description="Language(s) spoken, or 'none'.")
    speech_summary: str = Field(description="What is said, paraphrased (who speaks, tone, key lines). Empty if no speech.")
    voice_character: str = Field(description="Voice type and delivery, e.g. 'excited child voice', 'calm male narrator', 'none'.")
    music: str = Field(description="Music genre, tempo, energy, instruments, or 'none'.")
    sound_effects: str


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
        "- Split the timeline into chronological scenes at each cut or major change; "
        "scene times must cover 0 to the end without gaps.\n"
        "- Describe only what is actually seen and heard. Do not invent dialogue or sounds.\n"
        "- Never identify real people by name; describe them by appearance only.\n"
        "- Paraphrase speech and lyrics instead of transcribing them word for word.\n"
        "- Ignore platform UI and watermarks (TikTok/Instagram/YouTube logos, usernames, "
        "like buttons) -- they are not part of the content.\n"
        "Return JSON matching the schema."
    )


def _upload_and_wait(path: Path, notify: Notify):
    from google.genai import types as genai_types

    client = get_client()
    uploaded = call_gemini(
        lambda _model: client.files.upload(
            file=str(path),
            config=genai_types.UploadFileConfig(mime_type="video/mp4", display_name=path.name),
        ),
        notify,
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
            label="file-status",
            throttle=False,
        )


def _resample_for_gemini(path: Path, notify: Notify) -> Path:
    """Re-encodes the video at 2fps before uploading to Gemini.

    Gemini's free-tier File API samples uploaded video at 1fps (static mode).
    By pre-sampling to 2fps we double effective frame coverage: sub-second cuts
    in fast-cut videos (which land between Gemini's 1fps samples) become visible
    in the resampled stream. The temp file is placed next to the original and
    cleaned up after the Gemini call completes.
    """
    out_path = path.with_name(f"{path.stem}_2fps.mp4")
    if out_path.exists():
        return out_path
    notify("Pre-processing video for Gemini analysis (2fps resample)...")
    try:
        subprocess.run(
            [
                ffmpeg_exe(), "-y", "-i", str(path),
                "-vf", "fps=2",
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
                "-c:a", "aac", "-b:a", "64k",
                str(out_path),
            ],
            capture_output=True,
            check=True,
        )
        size_mb = out_path.stat().st_size / 1_048_576
        logger.info(f"[RESAMPLE] 2fps pre-sample: {out_path.name} ({size_mb:.1f} MB)")
        return out_path
    except subprocess.CalledProcessError as exc:
        logger.warning(
            f"[RESAMPLE] ffmpeg 2fps resample failed, uploading original: "
            f"{(exc.stderr or b'').decode()[:200]}"
        )
        return path


def analyze_video(path: Path, media: MediaInfo, notify: Notify = _noop_notify) -> VideoAnalysis:
    from google.genai import types as genai_types

    client = get_client()
    # Pre-sample at 2fps to double Gemini's effective frame coverage on fast-cut videos.
    analysis_path = _resample_for_gemini(path, notify)
    notify("Uploading video to Gemini...")
    uploaded = _upload_and_wait(analysis_path, notify)
    try:
        notify("Analysing video with Gemini...")
        prompt = _analysis_prompt(media)

        def run(model: str) -> VideoAnalysis:
            response = client.models.generate_content(
                model=model,
                contents=[uploaded, prompt],
                config=genai_types.GenerateContentConfig(
                    http_options={"api_version": "v1beta"},
                    response_mime_type="application/json",
                    response_schema=VideoAnalysis,
                    temperature=0.3,
                    automatic_function_calling=genai_types.AutomaticFunctionCallingConfig(disable=True),
                ),
            )
            return _parse_structured(response, VideoAnalysis)

        analysis = call_gemini(run, notify, label="analyze")
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
        # Clean up the temp 2fps file (never the original).
        if analysis_path != path and analysis_path.exists():
            try:
                analysis_path.unlink()
            except OSError:
                pass


# ============================================================================
# Prompt generation (one Gemini call + a fixed Python template)
# ============================================================================

class PromptBeat(BaseModel):
    label: str = Field(description="Short beat label, e.g. 'Hook', 'Build-up', 'Payoff'.")
    duration_seconds: float
    visual: str = Field(description="What is on screen: subjects, composition, setting, framing.")
    action: str = Field(description="What happens during this beat, in order.")
    camera: str = Field(description="Shot type + camera movement.")
    audio: str = Field(description="Speech (short, paraphrased), music, sound effects for this beat, or 'none'.")
    on_screen_text: str = Field(description="Short text overlay ONLY if the source relies on it (max ~5 words), else empty.")


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



class SocialMetadata(BaseModel):
    """YouTube and Instagram ready-to-paste viral metadata for the recreated video."""

    youtube_title: str = Field(
        description=(
            "YouTube Shorts title under 65 characters so it is NEVER truncated on mobile. "
            "Front-load the curiosity hook in the first 3-5 words, include 1 emotional emoji, and end with #Shorts. "
            "Example: 'Nobody Expected This In 2070 😳 #Shorts'"
        )
    )
    youtube_description: str = Field(
        description=(
            "YouTube description. Rich SEO structure under 400 characters: "
            "Line 1: Hook sentence featuring the primary search keyword. "
            "Lines 2-3: 2-3 conversational sentences with natural semantic keywords explaining the clip. "
            "Line 4: Engagement CTA ('Subscribe for daily [topic]! What would you do? Comment below 👇'). "
            "Line 5: Exactly 3 hashtags (e.g. '#Shorts #NicheTag1 #NicheTag2')."
        )
    )
    youtube_tags: str = Field(
        description=(
            "Comma-separated YouTube tags. TOTAL string length MUST be under 450 characters (YouTube allows 500 max). "
            "6-10 targeted tags: exact query keyword first, common search terms, niche audience, and 2 broad category tags. "
            "NO # symbols in tags. Each tag 2-4 words max."
        )
    )
    instagram_caption: str = Field(
        description=(
            "Instagram viral caption formatted with line breaks into 3 distinct sections: "
            "1. Micro-Hook (Line 1, under 120 chars before '...more' fold): Bold, relatable or shocked reaction with 1-2 emojis. "
            "2. Context/Story (Lines 2-3): 2-3 short conversational sentences embedding high-traffic search keywords (for Instagram SEO). "
            "3. High-converting CTA (Line 4): Explicit trigger for Shares or Saves (the #1 ranking signals on Reels!). "
            "Example: 'Send this to someone who does this 😭' or 'Save this for later 📌'."
        )
    )
    instagram_hashtags: str = Field(
        description=(
            "Space-separated hashtags, ALL starting with #. "
            "2-3 niche tags relevant specifically to this video's content/topic, "
            "plus #viral #reels #explorepage. Total 5-6 hashtags. "
            "Example: '#phonedrop #futuretech #techfail #viral #reels #explorepage'"
        )
    )


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


def _plan_prompt(analysis: VideoAnalysis, media: MediaInfo, duration: int, beat_count: int) -> str:
    """Build the Gemini prompt that turns a VideoAnalysis into a RecreationPlan.

    Improvements over original:
    - Hook-first mandatory structure (Beat 1 = scroll-stopper, max 1.5 s).
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
    hook_text = (analysis.hook or "the most visually dramatic moment").strip()
    why_works = (analysis.why_it_works or "").strip()

    return (
        f"Below is a JSON analysis of a {media.duration_seconds:.0f}-second source video "
        f"({len(analysis.scenes)} scenes detected, {media.aspect_ratio} aspect ratio).\n"
        f"Write a recreation plan for ONE {duration}-second {media.aspect_ratio} video "
        f"optimised for Instagram Reels / YouTube Shorts.\n\n"

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
        "- Keep the source's visual medium, art style, characters, setting, "
        "mood, color palette, and camera style.\n\n"

        "=== NO TEXT RULE (mandatory) ===\n"
        "- on_screen_text MUST be empty string '' for EVERY beat without exception.\n"
        "- Do NOT reproduce any text overlays, captions, subtitles, or title cards from the source.\n"
        "- Even if the source has meme text, caption text, or language text, set on_screen_text to ''.\n"
        "- Only exception: dialogue text that is plot-essential (speech bubbles in animation). Still omit meme/caption text.\n\n"

        "=== SETTING CONSISTENCY (mandatory) ===\n"
        "- Every beat MUST be set in the EXACT same room/environment as the source.\n"
        "- Describe the setting once in detail and reuse it verbatim in every beat's visual field.\n"
        "- Do NOT change the floor type, wall colour, furniture, or background between beats.\n"
        "- If a beat shows a close-up of an object, the background must still match the same room.\n\n"

        "=== PHYSICS & OBJECT STATE PRECISION ===\n"
        "- For any object in motion (falling, flying, rotating), describe its exact state at the\n"
        "  START and END of each beat: position (height, distance), orientation (tilt angle), speed.\n"
        "- Example for a falling phone: 'Beat start: phone is 1.2m above floor, screen facing down,\n"
        "  tilted 15 degrees. Beat end: phone is 3cm above floor, flat horizontal, thrusters firing.'\n"
        "- Do NOT leave physics implied -- state every axis of orientation explicitly.\n\n"

        "=== CAMERA RULES (override model zoom-bias) ===\n"
        "- Write camera as: [shot type], [movement]. If the original shot is static write: "
        "'locked-off [shot type], zero camera movement, no zoom, no pan'.\n"
        "- Use cinematography-precise terms: 'macro close-up', 'locked overhead tripod', "
        "'slow continuous optical dolly-in 0.3m over 2s', 'handheld micro-shake'.\n"
        "- Never write just 'static' or 'close-up' without a qualifier.\n\n"

        "=== AUDIO RULES ===\n"
        "- Per beat: timestamp each distinct sound from beat start. Example: "
        "'0.0s: knife impact; 0.3s: chicken slide; 0.8s: whoosh transition'.\n"
        "- Preserve the source audio character (ASMR, SFX, voice, music genre).\n\n"

        "=== CHARACTER & REFERENCE RULES ===\n"
        "- Describe each character once in `characters`; reuse those exact words in every beat.\n"
        "- Speech: paraphrase only -- never copy lyrics or lines verbatim. ~2.5 words/second.\n"
        "- reference_images: 0-3 images. 'character' for recurring characters (max 2), "
        "'first_frame' when opening composition matters, 'object' for a key prop. "
        "Number REF_1, REF_2... and cite as [REF_n] inside beat text.\n"
        "- Each image_prompt is a complete standalone text-to-image prompt: subject alone, "
        f"neutral pose, plain background, same visual medium, {media.aspect_ratio} for first_frame.\n"
        "- IMPORTANT: When you generate the reference image prompts, describe the object with full "
        "physical precision: exact orientation (face-up/face-down/vertical/horizontal), "
        "exact colour, material, and any attached elements (e.g. thruster nozzles at corners).\n\n"

        "=== AVOID (global + video-specific) ===\n"
        "- `avoid` MUST include these global items:\n"
        "  * 'On-screen text, captions, subtitles, or title cards of any kind -- video must be completely text-free'\n"
        "  * 'Changing room, background, floor, or setting between beats -- identical setting in every beat'\n"
        "  * 'Unprompted camera zoom or pan -- camera is fixed unless the Camera line explicitly says so'\n"
        "  * 'Slow-motion unless the scene label explicitly says slow motion'\n"
        "  * 'Fade-in or fade-out at start; begin on the first frame at full energy'\n"
        "  * 'Background music unless the AUDIO section specifies it'\n"
        "  * 'Letterboxing or pillarboxing; fill the full frame edge-to-edge'\n"
        "- Also add video-specific failure modes (style drift, wrong character count, wrong object orientation).\n"
        "- No real person names, no brand logos, no platform UI or watermarks.\n\n"

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
    "Changing the room, background, floor type, or setting between scenes -- all beats must share the identical setting",
]


def render_prompt(plan: RecreationPlan, media: MediaInfo, duration: int) -> str:
    """Render the final text prompt that gets pasted into a video generator.

    Improvements over original:
    - SOCIAL MEDIA OPTIMISED header block so the model knows context.
    - Scene 1 annotated as HOOK with sound-off and full-energy instructions.
    - _GLOBAL_AVOID prepended before video-specific avoids to kill zoom bias.
    """
    orientation = {"9:16": "vertical", "3:4": "vertical", "4:5": "vertical", "1:1": "square"}.get(
        media.aspect_ratio, "horizontal"
    )
    lines = [
        f"Create one {duration}-second {orientation} {media.aspect_ratio} video. {plan.logline.strip()}",
        "",
        "SOCIAL MEDIA OPTIMISED (Instagram Reels / YouTube Shorts)",
        "- Scene 1 is the scroll-stopper hook: most dramatic visual, works with audio muted.",
        "- Payoff-first structure: show the result first, then the process, then the CTA hold.",
        "- No fade-in at the very start; begin on the first frame at full energy.",
        "",
    ]
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
        if number == 1:
            # Enforce hook label and sound-off annotation on the first beat.
            if "HOOK" not in label:
                label = f"HOOK -- {label}"
            scene_header = (
                f"Scene {number} ({_fmt_seconds(cursor)}-{_fmt_seconds(end)} sec) - {label}"
                "  [scroll-stopper: works sound-OFF, starts at full energy, no fade-in]"
            )
        else:
            scene_header = f"Scene {number} ({_fmt_seconds(cursor)}-{_fmt_seconds(end)} sec) - {label}"
        lines.append(scene_header)
        lines.append(f"Visual: {beat.visual.strip()}")
        lines.append(f"Action: {beat.action.strip()}")
        lines.append(f"Camera: {beat.camera.strip()}")
        if beat.audio.strip():
            lines.append(f"Audio: {beat.audio.strip()}")
        if beat.on_screen_text.strip():
            lines.append(f"On-screen text: \"{beat.on_screen_text.strip()}\"")
        lines.append("")
        cursor = end

    if plan.overall_audio.strip():
        lines += ["AUDIO", plan.overall_audio.strip(), ""]

    # Global avoids first (suppress model defaults), then video-specific.
    avoid = list(_GLOBAL_AVOID) + [item.strip() for item in plan.avoid if item.strip()]
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




def _social_metadata_prompt(analysis: "VideoAnalysis", plan: "RecreationPlan") -> str:
    """Build the Gemini instruction for generating viral YouTube + Instagram metadata."""
    return (
        "You are an elite viral growth hacker and short-form video SEO specialist.\n"
        "Generate copy-paste ready, algorithmically optimized metadata for YouTube Shorts and Instagram Reels.\n\n"

        "=== PLATFORM LIMITS & VIRAL RULES ===\n\n"

        "1. YOUTUBE SHORTS (Optimized for Click-Through Rate & Search Indexing):\n"
        "   - title: STRICTLY under 65 characters so it is never truncated on mobile feeds.\n"
        "     * Front-load the curiosity hook in the first 3-5 words.\n"
        "     * High curiosity gap or emotional reaction.\n"
        "     * Include 1 emoji and always append #Shorts at the end.\n"
        "     * Example: 'Nobody Expected This In 2070 😳 #Shorts'\n\n"
        "   - description: Rich SEO format (YouTube is the #2 search engine):\n"
        "     * Line 1 (Hook): Clear, punchy sentence featuring the primary search keyword.\n"
        "     * Lines 2-3 (Context): 2-3 engaging context sentences with natural semantic keywords explaining the clip.\n"
        "     * Line 4 (CTA): Engagement trigger (e.g. 'Subscribe for daily [topic]! What would you do? Comment below 👇').\n"
        "     * Line 5 (Hashtags): Exactly 3 top hashtags (#Shorts plus 2 niche keywords).\n"
        "     * Total under 400 characters.\n\n"
        "   - tags: Comma-separated list under 450 characters (YouTube allows 500 max).\n"
        "     * 6-10 targeted tags: exact query keyword first, common search terms, niche audience, and 2 broad category tags.\n"
        "     * NEVER include '#' in tags. Separate strictly with commas.\n\n"

        "2. INSTAGRAM REELS (Optimized for Explore Discovery, Shares & Saves):\n"
        "   - caption: Do NOT write only 3 words! Instagram algorithms heavily index caption text for Search & Explore.\n"
        "     Format with clean line breaks into 3 distinct sections:\n"
        "     * Section 1 - The Micro-Hook (Line 1, under 120 chars before '...more'): Bold, relatable or shocked reaction with 1-2 emojis that makes viewers tap '...more'.\n"
        "     * Section 2 - Story & SEO (Lines 2-3): 2-3 conversational sentences giving context and using natural search keywords.\n"
        "     * Section 3 - The Viral CTA: High-converting trigger for Shares or Saves (the #1 ranking signals on Reels!).\n"
        "       Examples: 'Send this to someone who does this 😭' or 'Save this for your next inspo 📌' or 'Drop a 🔥 if you agree!'.\n\n"
        "   - hashtags: Exactly 5-6 hashtags starting with #.\n"
        "     * Mix 2-3 specific content/niche tags, plus #viral #reels #explorepage.\n"
        "     * Total 5-6 space-separated hashtags.\n\n"

        f"=== VIDEO CONTEXT ===\n"
        f"Content type: {analysis.content_type}\n"
        f"Visual medium: {analysis.visual_medium}\n"
        f"Hook: {(analysis.hook or '').strip()}\n"
        f"Summary: {(analysis.summary or '').strip()[:400]}\n"
        f"Why it works: {(analysis.why_it_works or '').strip()[:300]}\n\n"

        f"=== RECREATION PLAN ===\n"
        f"Title: {plan.title}\n"
        f"Logline: {plan.logline}\n"
        f"Mood: {plan.mood}\n"
    )


def _trim_tags(tags: str, limit: int = 450) -> str:
    """Trim comma-separated tags string to stay within character limit and remove # symbols."""
    cleaned = tags.replace("#", "")
    if len(cleaned) <= limit:
        return cleaned
    parts = [t.strip() for t in cleaned.split(",") if t.strip()]
    result = ""
    for part in parts:
        candidate = result + (", " if result else "") + part
        if len(candidate) > limit:
            break
        result = candidate
    return result


def _validate_instagram_hashtags(tags: str) -> str:
    """Ensure all tokens start with # and #viral #reels #explorepage are present."""
    tokens = tags.split()
    fixed = []
    for t in tokens:
        fixed.append(t if t.startswith("#") else "#" + t)
    mandatory = {"#viral", "#reels", "#explorepage"}
    existing = {t.lower() for t in fixed}
    for tag in mandatory:
        if tag not in existing:
            fixed.append(tag)
    return " ".join(fixed[:6])  # cap at 6


def generate_social_metadata(
    result: "AnalysisResult",
    plan: "RecreationPlan",
    notify: Notify = _noop_notify,
) -> "SocialMetadata":
    """Generate YouTube + Instagram metadata for the recreated video.

    Makes one extra free-tier Gemini call (~200 tokens) after prompt generation.
    Returns a validated SocialMetadata object.
    """
    from google.genai import types as genai_types

    analysis = VideoAnalysis.model_validate(result.analysis)
    client = get_client()
    request = _social_metadata_prompt(analysis, plan)

    def run(model: str) -> SocialMetadata:
        response = client.models.generate_content(
            model=model,
            contents=[request],
            config=genai_types.GenerateContentConfig(
                http_options={"api_version": "v1beta"},
                response_mime_type="application/json",
                response_schema=SocialMetadata,
                temperature=0.8,
                automatic_function_calling=genai_types.AutomaticFunctionCallingConfig(disable=True),
            ),
        )
        meta = _parse_structured(response, SocialMetadata)
        return meta

    notify("Generating social media metadata...")
    meta = call_gemini(run, notify, label="social-metadata")

    # Validate and auto-trim
    meta = meta.model_copy(update={
        "youtube_title": meta.youtube_title[:100],
        "youtube_description": meta.youtube_description[:500],
        "youtube_tags": _trim_tags(meta.youtube_tags, 450),
        "instagram_hashtags": _validate_instagram_hashtags(meta.instagram_hashtags),
    })
    return meta


def render_social_metadata(meta: "SocialMetadata") -> str:
    """Render the social metadata into a copy-paste ready text block."""
    tag_len = len(meta.youtube_tags)
    lines = [
        "=== YOUTUBE ===",
        "",
        "📌 TITLE",
        meta.youtube_title,
        "",
        "📄 DESCRIPTION",
        meta.youtube_description,
        "",
        f"🏷️ TAGS  ({tag_len}/450 chars — paste into YouTube Tags field)",
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
    return "\n".join(lines)

def generate_prompt(result: "AnalysisResult", duration: int, notify: Notify = _noop_notify) -> GeneratedPrompt:
    from google.genai import types as genai_types

    if duration not in CONFIG.output_durations:
        raise ValueError(f"Duration must be one of {CONFIG.output_durations}.")
    client = get_client()
    analysis = VideoAnalysis.model_validate(result.analysis)
    media = MediaInfo(**result.media)
    beat_count = beats_for_duration(duration, len(analysis.scenes))
    request = _plan_prompt(analysis, media, duration, beat_count)

    def run(model: str) -> RecreationPlan:
        response = client.models.generate_content(
            model=model,
            contents=[request],
            config=genai_types.GenerateContentConfig(
                http_options={"api_version": "v1beta"},
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
    plan = call_gemini(run, notify, label="prompt")
    if len(plan.beats) != beat_count:
        logger.info(f"[PROMPT] asked for {beat_count} beats, got {len(plan.beats)} -- keeping Gemini's split")
    plan = plan.model_copy(update={
        "beats": _normalize_durations(plan.beats, duration),
        "reference_images": _normalize_references(plan.reference_images, len(plan.beats)),
    })
    prompt = render_prompt(plan, media, duration)

    prompts_dir = CONFIG.work_dir / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{result.video_sha[:12]}_{duration}s_{time.strftime('%Y%m%d-%H%M%S')}"
    generated = GeneratedPrompt(
        duration=duration,
        prompt=prompt,
        plan=plan.model_dump(),
        saved_to=str(prompts_dir / f"{stem}.txt"),
        references=[ref.model_dump() for ref in plan.reference_images],
        reference_sheet=render_reference_sheet(plan),
    )
    # social metadata populated below
    # Generate social metadata (one extra free-tier Gemini call)
    try:
        social_meta = generate_social_metadata(result, plan, notify)
        social_text = render_social_metadata(social_meta)
        generated = GeneratedPrompt(
            duration=duration,
            prompt=prompt,
            plan=plan.model_dump(),
            saved_to=str(prompts_dir / f"{stem}.txt"),
            references=[ref.model_dump() for ref in plan.reference_images],
            reference_sheet=render_reference_sheet(plan),
            social_metadata=social_meta,
            social_text=social_text,
        )
        (prompts_dir / f"{stem}_social.json").write_text(
            social_meta.model_dump_json(indent=2), encoding="utf-8"
        )
    except Exception as exc:
        logger.warning(f"[SOCIAL] metadata generation failed: {exc} -- skipping")
        # Keep generated as-is without social metadata

    (prompts_dir / f"{stem}.txt").write_text(generated.full_text(), encoding="utf-8")
    (prompts_dir / f"{stem}.json").write_text(plan.model_dump_json(indent=2), encoding="utf-8")
    logger.info(
        f"[PROMPT] saved {stem}.txt ({len(prompt)} chars, {len(plan.reference_images)} reference image(s))"
    )
    return generated


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
        return cached

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
    )
    _cache_path(video_sha).parent.mkdir(parents=True, exist_ok=True)
    _cache_path(video_sha).write_text(json.dumps(asdict(result), indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def analyze_local_video(path: Path, title: str = "Uploaded video", notify: Notify = _noop_notify) -> AnalysisResult:
    """Validate and analyse a locally provided/uploaded video file, with caching."""
    notify("Checking video...")
    media = probe_video(path)
    validate_media(media)

    video_sha = _sha256(path)
    if cached := _load_cached(video_sha):
        notify("Using saved analysis for this video")
        logger.info(f"[CACHE] hit by content sha={video_sha[:12]}")
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
        return cached

    analysis = analyze_video(path, media, notify)

    # Immediately delete the uploaded video — only prompt is kept.
    try:
        path.unlink(missing_ok=True)
        logger.info(f"[CLEANUP] deleted uploaded video after analysis: {path.name}")
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


def _render_generated(st, generated: "GeneratedPrompt") -> None:
    references = getattr(generated, "references", []) or []
    full_text = generated.full_text() if hasattr(generated, "full_text") else generated.prompt
    duration = generated.duration

    if not CONFIG.show_prompt:
        st.success(
            f"{duration}-second prompt ready ({len(references)} reference image(s)). "
            "On-screen display is off (SHOW_PROMPT=false) -- download it below."
        )
    else:
        st.subheader(f"Video generation prompt ({duration} s)")
        st.caption(
            "Free-tier Gemini only writes text -- no video or image is generated here. "
            "Paste this into your video generator (Seedance, Omni, Veo, Kling, ...)."
        )
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
    social = getattr(generated, "social_metadata", None)
    if social:
        st.divider()
        st.subheader("📲 Post to Social Media")
        st.caption(
            "Copy-paste ready metadata for YouTube and Instagram. "
            "Generated by Gemini — edit before posting!"
        )
        tab_yt, tab_ig = st.tabs(["▶️ YouTube", "📸 Instagram"])

        with tab_yt:
            st.markdown("**📌 Title** — copy-paste into YouTube title field")
            _show_copyable(st, social.youtube_title)
            st.caption(f"{len(social.youtube_title)}/100 characters")

            st.markdown("**📄 Description** — copy-paste into YouTube description field")
            _show_copyable(st, social.youtube_description)
            st.caption(f"{len(social.youtube_description)}/500 characters")

            tag_len = len(social.youtube_tags)
            st.markdown(f"**🏷️ Tags** — paste into YouTube Tags field · {tag_len}/450 chars")
            _show_copyable(st, social.youtube_tags)

        with tab_ig:
            st.markdown("**✍️ Caption** — copy-paste as your Instagram caption")
            _show_copyable(st, social.instagram_caption)

            st.markdown("**#️⃣ Hashtags** — copy-paste after your caption")
            _show_copyable(st, social.instagram_hashtags)
            st.caption(f"{len(social.instagram_hashtags.split())} hashtags")

    st.divider()
    st.download_button(
        "Download prompts + metadata",
        data=full_text,
        file_name=f"recreate_{duration}s.txt",
        mime="text/plain",
        **_supported_kwargs(st.download_button, icon=":material/download:"),
    )
    st.caption(f"Saved to {generated.saved_to}")


def run_app() -> None:
    import streamlit as st
    # Lazy import of image pipeline — only needed when image mode is active.
    from recreate_image import (
        analyze_image_url, generate_image_recreation,
        ImageAnalysisResult, GeneratedImagePrompt, ImageAnalysis,
        USER_FACING_ERRORS as IMAGE_USER_FACING_ERRORS,
    )

    # ── 24-hour startup cleanup ──────────────────────────────────────────────
    if "_swept" not in st.session_state:
        _sweep_old_files()
        st.session_state["_swept"] = True

    st.set_page_config(page_title="Recreate any post", page_icon="🎬", layout="centered")

    # ── Sidebar navigation ───────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 🎬 Recreate")
        mode = st.radio(
            "What are you recreating?",
            options=["🎬 Video / Reel", "🖼️ Image Post"],
            index=0,
            key="app_mode",
        )
        st.divider()
        st.caption(
            "Paste any public link. The app auto-detects image vs video. "
            "Use this toggle to override if needed."
        )

    is_image_mode = mode == "🖼️ Image Post"

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
            "Gemini analyses it and writes copy-paste prompts for image generators + text overlays + audio + social metadata."
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

        effective_image = True
        url_type = MediaType.IMAGE

        # ── Determine image source ────────────────────────────────────────────
        source_path: Path | None = None
        source_url_str: str = ""

        if uploaded_file is not None:
            # Save uploaded file to temp location
            uploads_dir = CONFIG.work_dir / "image_downloads"
            uploads_dir.mkdir(parents=True, exist_ok=True)
            suffix = Path(uploaded_file.name).suffix.lower() or ".jpg"
            temp_img = uploads_dir / f"upload_{uuid.uuid4().hex[:8]}{suffix}"
            temp_img.write_bytes(uploaded_file.read())
            source_path = temp_img
            source_url_str = f"upload://{uploaded_file.name}"

        # ── Analyze image on submit or upload ─────────────────────────────────
        should_analyze_image = (img_submitted and img_url.strip()) or (
            uploaded_file is not None
            and st.session_state.get("_last_upload") != uploaded_file.name
        )
        if uploaded_file is not None:
            st.session_state["_last_upload"] = uploaded_file.name

        if should_analyze_image:
            if source_path and source_path.exists():
                # ── Uploaded file path ────────────────────────────────────────
                with st.status("Analysing image...", expanded=True) as status:
                    try:
                        from recreate_image import analyze_image, ImageAnalysisResult
                        notify_fn = lambda msg: status.update(label=msg)
                        analysis = analyze_image(source_path, notify_fn)
                        import hashlib
                        h = hashlib.sha256(source_path.read_bytes()).hexdigest()
                        image_result = ImageAnalysisResult(
                            image_sha=h,
                            image_path=str(source_path),
                            source_url=source_url_str,
                            title=uploaded_file.name,
                            extractor="upload",
                            analysis=analysis.model_dump(),
                        )
                        # Immediately delete temp upload after Gemini processes it
                        try:
                            source_path.unlink(missing_ok=True)
                        except OSError:
                            pass
                    except (IMAGE_USER_FACING_ERRORS + USER_FACING_ERRORS) as exc:
                        status.update(label="Analysis failed", state="error")
                        st.error(str(exc))
                        image_result = None
                    else:
                        st.session_state.image_result = image_result
                        st.session_state.image_generated = None
                        status.update(label="Analysis complete", state="complete", expanded=False)
            elif img_submitted and img_url.strip():
                # ── URL-based image ───────────────────────────────────────────
                with st.status("Downloading and analysing image...", expanded=True) as status:
                    try:
                        image_result = analyze_image_url(
                            img_url, lambda msg: status.update(label=msg)
                        )
                    except (IMAGE_USER_FACING_ERRORS + USER_FACING_ERRORS) as exc:
                        status.update(label="Analysis failed", state="error")
                        st.error(str(exc))
                    else:
                        st.session_state.image_result = image_result
                        st.session_state.image_generated = None
                        status.update(label="Analysis complete", state="complete", expanded=False)

        _render_image_results(st, is_image_mode)
        return  # image mode done — skip video form below

    # ── Video mode: Upload file or URL input ──────────────────────────────────
    uploaded_video = st.file_uploader(
        "Upload video file (if private, login-blocked, or saved on your device)",
        type=["mp4", "mov", "webm", "m4v"],
        key="video_upload",
        help="Upload up to 60s video. File is automatically deleted after Gemini analysis.",
    )

    if (
        uploaded_video is not None
        and st.session_state.get("_last_video_upload") != uploaded_video.name
    ):
        st.session_state["_last_video_upload"] = uploaded_video.name
        uploads_dir = CONFIG.work_dir / "video_uploads"
        uploads_dir.mkdir(parents=True, exist_ok=True)
        suffix = Path(uploaded_video.name).suffix.lower() or ".mp4"
        temp_video = uploads_dir / f"upload_{uuid.uuid4().hex[:8]}{suffix}"
        temp_video.write_bytes(uploaded_video.read())

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
            "Or paste public video URL",
            placeholder="https://www.youtube.com/shorts/... or https://www.instagram.com/reel/...",
        )

        url_type = detect_url_type(url) if url.strip() else MediaType.VIDEO
        effective_image = url_type == MediaType.IMAGE  # direct image URL in video form

        if url_type == MediaType.AMBIGUOUS:
            st.caption("⚠️ This Instagram link could be a photo or a Reel. Select below:")
            post_type = st.radio(
                "Post type",
                options=["🖼️ Image / Photo post", "🎬 Video / Reel"],
                horizontal=True,
                key="ambiguous_type",
                label_visibility="collapsed",
            )
            effective_image = post_type == "🖼️ Image / Photo post"

        btn_label = "🔍 Analyze image" if effective_image else "🔍 Analyze video URL"
        submitted = st.form_submit_button(btn_label, type="primary")

    # ── Analysis ─────────────────────────────────────────────────────────────
    if submitted and url.strip():
        if url_type == MediaType.IMAGE and effective_image:
            st.info("Auto-detected as an **image URL** — use the 🖼️ Image Post tab for image posts.")

        # ── VIDEO pipeline ────────────────────────────────────────────────────
        with st.status("Starting video analysis...", expanded=True) as status:
            try:
                video_result = analyze_url(
                    url, lambda msg: status.update(label=msg)
                )
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

    _render_video_results(st)


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

    if st.button(
        "Generate prompt", type="primary",
        **_supported_kwargs(st.button, icon=":material/auto_awesome:"),
    ):
        with st.status(f"Writing the {duration}-second prompt...", expanded=True) as status:
            try:
                generated = generate_prompt(result, duration, lambda msg: status.update(label=msg))
            except USER_FACING_ERRORS as exc:
                status.update(label="Prompt generation failed", state="error")
                st.error(str(exc))
            else:
                st.session_state.video_prompts[duration] = generated
                status.update(label="Prompt ready", state="complete", expanded=False)

    generated: GeneratedPrompt | None = st.session_state.get("video_prompts", {}).get(duration)
    if generated:
        _render_generated(st, generated)


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
                st.markdown(f":orange[**Text overlays detected:** {len(analysis.text_overlays)}]")
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
                gen = generate_image_recreation(result, lambda msg: status.update(label=msg))
            except IMAGE_USER_FACING_ERRORS as exc:
                status.update(label="Generation failed", state="error")
                st.error(str(exc))
            else:
                st.session_state.image_generated = gen
                status.update(label="Prompts ready", state="complete", expanded=False)

    gen: GeneratedImagePrompt | None = st.session_state.get("image_generated")
    if not gen:
        return

    from recreate_image import ImageRecreationPlan as _IRP
    plan = _IRP.model_validate(gen.plan)

    if CONFIG.show_prompt:
        tab_nb, tab_flux, tab_sd, tab_text, tab_audio, tab_social = st.tabs([
            "🍌 Nano Banana", "🌀 FLUX / MJ", "🖥️ Stable Diffusion",
            "✍️ Text Overlays", "🎵 Audio", "📲 Social",
        ])
        with tab_nb:
            st.subheader("Nano Banana / Nano Banana Pro")
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
        with tab_text:
            st.subheader("Text Overlay Recreation")
            if analysis.has_text and plan.text_overlays:
                st.info(f"{len(plan.text_overlays)} text overlay(s) — add these in Canva/CapCut after generating the base image.")
                for i, ov in enumerate(plan.text_overlays, 1):
                    with st.container(border=True):
                        st.markdown(f"**[{i}] Text:**")
                        _show_copyable(st, ov.text)
                        st.markdown(f"**Position:** {ov.position} | **Style:** {ov.style}")
                        st.caption(f"Canva step: {ov.canva_instruction}")
                st.divider()
                st.markdown("**Full Canva / CapCut Workflow:**")
                _show_copyable(st, plan.canva_workflow)
            else:
                st.info("No text overlays — generate the base image as-is.")
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
            social = gen.social_metadata
            if social:
                yt_tab, ig_tab = st.tabs(["▶️ YouTube", "📸 Instagram"])
                with yt_tab:
                    st.markdown("**📌 Title**")
                    _show_copyable(st, social.youtube_title)
                    st.caption(f"{len(social.youtube_title)}/100 chars")
                    st.markdown("**📄 Description**")
                    _show_copyable(st, social.youtube_description)
                    st.markdown(f"**🏷️ Tags** ({len(social.youtube_tags)}/450 chars)")
                    _show_copyable(st, social.youtube_tags)
                with ig_tab:
                    st.markdown("**✍️ Caption**")
                    _show_copyable(st, social.instagram_caption)
                    st.markdown("**#️⃣ Hashtags**")
                    _show_copyable(st, social.instagram_hashtags)
            else:
                st.warning("Social metadata generation failed — try regenerating.")

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
        generated = generate_prompt(result, args.duration, notify)
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
