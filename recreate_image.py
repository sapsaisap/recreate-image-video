"""
recreate_image.py
=================
Analyses an Instagram image post (or any static image URL) with Gemini Vision
and produces copy-paste prompts for:
  - Image generation  (Nano Banana / Nano Banana Pro / FLUX / Midjourney / SD)
  - Text overlay recreation in Canva / CapCut
  - Audio recommendation  (search term for Instagram audio library)
  - YouTube + Instagram social metadata

No images or audio are generated here — only text prompts.

Run as Streamlit:
    streamlit run recreate_image.py
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, Field

# ── Shared utilities from recreate_video ────────────────────────────────────
from recreate_video import (
    CONFIG,
    TIMEOUT_IMAGE_SECONDS,
    TIMEOUT_TEXT_SECONDS,
    gemini_http_options,
    SUPPORTED_VIDEO_SUFFIXES,
    SocialMetadata,
    _noop_notify,
    _parse_structured,
    _sha256,
    _GLOBAL_AVOID,
    call_gemini,
    closest_aspect_ratio,
    ffmpeg_exe,
    generate_metadata,
    get_client,
    host_is,
    render_social,
    render_social_metadata,
    validate_public_video_url,
    URLVideoError,
    _YtdlpLogger,
    _is_retryable_block,
    _download_error_message,
    _mask_proxy_credentials,
    download_instagram_image,
    DownloadedImage,
)

logger = logging.getLogger(__name__)
Notify = Callable[[str], None]

SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".gif"}


# ============================================================================
# Pydantic schemas
# ============================================================================

class TextOverlay(BaseModel):
    text: str = Field(description="Exact verbatim text visible in the image.")
    position: str = Field(
        description="Where on the image: 'top-center', 'center', 'bottom-center', 'top-left', etc."
    )
    style: str = Field(
        description="Font style description: e.g. 'bold white sans-serif with black drop shadow', "
                    "'handwritten yellow', 'neon pink outlined'."
    )
    purpose: str = Field(
        description="Role of this text: 'meme caption', 'quote', 'call-to-action', 'title', "
                    "'watermark', 'hashtag overlay', 'information'."
    )
    canva_instruction: str = Field(
        description="One-line Canva/CapCut instruction to recreate this text overlay. "
                    "E.g. 'Add a white bold heading text box at the top, font size 80, "
                    "with black drop shadow, no background fill.'"
    )


class SuggestedAudio(BaseModel):
    track_name: str = Field(description="Song title. Use a real, known track.")
    artist_name: str = Field(description="Artist / band name.")
    instagram_search_query: str = Field(
        description="Exact string to type into Instagram audio search to find this track. "
                    "E.g. 'Calm Down Rema Selena Gomez'."
    )
    mood_match: str = Field(
        description="One sentence explaining why this audio fits the image mood/energy."
    )
    genre: str = Field(description="Genre: 'afrobeats', 'lo-fi', 'pop', 'hip-hop', 'ambient', etc.")
    bpm_feel: str = Field(description="'slow and dreamy', 'fast and energetic', 'mid-tempo chill', etc.")
    alternative_track_name: str = Field(description="Backup track title if first is unavailable.")
    alternative_artist_name: str = Field(description="Backup artist name.")
    alternative_search_query: str = Field(description="Instagram search query for the backup track.")


class ImageAnalysis(BaseModel):
    summary: str = Field(description="2-3 sentence description of the image post.")
    hook: str = Field(description="What makes someone stop scrolling on this image.")
    why_it_works: str = Field(description="Viral mechanics — why this image performs well.")
    content_type: str = Field(
        description="One of: quote_card, meme, aesthetic, product, infographic, "
                    "lifestyle, motivational, educational, funny, art."
    )
    visual_medium: str = Field(
        description="One of: photography, illustration, graphic_design, "
                    "collage, screenshot, ai_generated."
    )
    subject: str = Field(description="Detailed description of the main subject/focal point.")
    composition: str = Field(
        description="Layout: 'centered subject with blurred background', "
                    "'rule of thirds', 'full bleed', 'split screen', 'grid'."
    )
    color_palette: str = Field(
        description="Dominant colors: 'warm earth tones — burnt orange, beige, cream', "
                    "'high contrast black and white', 'pastel pink and lavender'."
    )
    lighting: str = Field(
        description="'natural golden hour side-light', 'studio ring light flat', "
                    "'dramatic chiaroscuro', 'bright even daylight'."
    )
    background: str = Field(description="Detailed background description.")
    has_text: bool = Field(description="True if any text is visible in the image.")
    text_overlays: list[TextOverlay] = Field(
        description="Every piece of text visible in the image. Empty list if none."
    )
    mood: str = Field(description="Overall emotional tone of the image.")
    aspect_ratio: str = Field(description="'9:16', '1:1', '4:5', or '16:9'.")
    suggested_audio: SuggestedAudio = Field(
        description="Best audio recommendation for posting this as an Instagram Reel."
    )


class ImageRecreationPlan(BaseModel):
    title: str = Field(description="Short descriptive title for this recreation.")
    logline: str = Field(description="One sentence describing what the recreated image shows.")

    # Image generation prompts
    nano_banana_prompt: str = Field(
        description=(
            "Image generation prompt optimised for Nano Banana / Nano Banana Pro. "
            "Highly detailed, includes: subject, setting, lighting, composition, color palette, "
            "camera angle, lens, mood. Starts with the most important elements. "
            "Example style: 'portrait of a young woman, golden hour backlight, "
            "shallow depth of field, warm earth tones, rule of thirds, "
            "shot on 85mm f/1.4, photorealistic, cinematic'."
        )
    )
    flux_midjourney_prompt: str = Field(
        description=(
            "Prompt for FLUX / Midjourney / Leonardo. Same content as nano_banana_prompt "
            "but with Midjourney-style suffixes: '--ar 9:16 --q 2 --style raw'. "
            "Include the aspect ratio flag appropriate for the post."
        )
    )
    stable_diffusion_prompt: str = Field(
        description=(
            "Prompt for Stable Diffusion / ComfyUI / A1111. "
            "Comma-separated token style. Include quality boosters: "
            "'masterpiece, best quality, ultra-detailed, photorealistic, 8k'."
        )
    )
    negative_prompt: str = Field(
        description=(
            "Negative prompt for SD / ComfyUI. Token-style. "
            "Include: 'blurry, low quality, watermark, text, logo, ugly, deformed, "
            "extra limbs, bad anatomy, nsfw'."
        )
    )

    # Text overlay recreation
    text_overlays: list[TextOverlay] = Field(
        description="Recreated text overlays matching the original. "
                    "Must match original if has_text=True. Empty list if no text."
    )
    canva_workflow: str = Field(
        description=(
            "Step-by-step Canva or CapCut workflow to add all text overlays. "
            "Written as numbered steps. Include: canvas size, background, "
            "font choices, colors, positions, effects."
        )
    )

    # Audio
    suggested_audio: SuggestedAudio

    # Technical
    aspect_ratio: str = Field(description="'9:16' for Reels, '1:1' for feed, '4:5' for portrait feed.")
    avoid: list[str] = Field(
        description="Image-gen specific things to avoid: wrong colors, style drift, "
                    "incorrect text, wrong composition."
    )


# ============================================================================
# Download helpers
# ============================================================================

@dataclass
class DownloadedMedia:
    path: Path
    title: str
    source_url: str
    extractor: str
    media_type: Literal["image", "video"]


def download_media(url: str, destination: Path, notify: Notify = _noop_notify) -> DownloadedMedia:
    """Download any public URL (video or image) using yt-dlp.
    Auto-detects whether the result is an image or video by file extension."""
    from yt_dlp import YoutubeDL
    from yt_dlp.utils import DownloadError

    safe_url = validate_public_video_url(url)
    destination.mkdir(parents=True, exist_ok=True)
    max_bytes = CONFIG.max_download_mb * 1024 * 1024
    rejection: dict[str, str] = {}

    def progress_hook(payload: dict) -> None:
        if payload.get("status") != "downloading":
            return
        done = float(payload.get("downloaded_bytes") or 0)
        total = float(payload.get("total_bytes") or payload.get("total_bytes_estimate") or 0)
        if total:
            notify(f"Downloading... {min(done / total, 0.99) * 100:.0f}%")

    options: dict[str, Any] = {
        "format": "best",
        "outtmpl": str(destination / "source.%(ext)s"),
        "noplaylist": True,
        "playlist_items": "1",
        "max_filesize": max_bytes,
        "progress_hooks": [progress_hook],
        "restrictfilenames": True,
        "quiet": True,
        "no_warnings": True,
        "socket_timeout": 30,
        "retries": 2,
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
    attempts = [(True, "proxy"), (True, "proxy-retry"), (False, "direct")] if proxy else [(False, "direct")]

    info = None
    last_exc: Exception | None = None
    for idx, (use_proxy, label) in enumerate(attempts):
        if idx == 1:
            time.sleep(3)
        opts = dict(options)
        if use_proxy:
            opts["proxy"] = proxy
        try:
            with YoutubeDL(opts) as dl:
                info = dl.extract_info(safe_url, download=True)
            last_exc = None
            break
        except DownloadError as exc:
            last_exc = exc
            if rejection or idx == len(attempts) - 1 or not _is_retryable_block(exc):
                break

    if rejection:
        raise URLVideoError(rejection.get("reason", "Download rejected."))
    if last_exc is not None:
        raise URLVideoError(_download_error_message(safe_url, last_exc)) from last_exc
    if not info:
        raise URLVideoError("Could not extract media info from the URL.")

    # Find downloaded file
    all_candidates = [p for p in destination.glob("source.*") if p.is_file()]
    if not all_candidates:
        raise URLVideoError("The URL did not produce a downloadable file.")
    source = max(all_candidates, key=lambda p: p.stat().st_size)
    if source.stat().st_size == 0:
        raise URLVideoError("Download produced an empty file.")

    suffix = source.suffix.lower()
    if suffix in SUPPORTED_IMAGE_SUFFIXES:
        media_type = "image"
    else:
        media_type = "video"

    logger.info(
        f"[DOWNLOAD] ok type={media_type} ext={suffix} "
        f"title={str(info.get('title') or '')[:60]}"
    )
    return DownloadedMedia(
        path=source,
        title=str(info.get("title") or "Untitled post"),
        source_url=str(info.get("webpage_url") or safe_url),
        extractor=str(info.get("extractor_key") or "generic"),
        media_type=media_type,
    )


# ============================================================================
# Analysis
# ============================================================================

def _analysis_prompt_image() -> str:
    return (
        "You are an expert visual content analyst and viral social media strategist.\n"
        "Analyse the attached image (an Instagram post) in detail.\n\n"

        "=== TEXT EXTRACTION (critical) ===\n"
        "- Read EVERY piece of text visible in the image, verbatim.\n"
        "- For each text: record exact wording, position, font style/colour.\n"
        "- Set has_text=true if ANY text is visible, even small.\n\n"

        "=== VISUAL ANALYSIS ===\n"
        "- Describe subject, composition, colour palette, lighting, background precisely.\n"
        "- Identify the content_type (quote_card, meme, aesthetic, etc.).\n\n"

        "=== AUDIO RECOMMENDATION ===\n"
        "- Suggest the BEST real, known song to use as audio when posting as an Instagram Reel.\n"
        "- The track must match the mood, energy, and content type exactly.\n"
        "- Provide an alternative in case the primary is unavailable.\n"
        "- Give the exact Instagram audio search query for each.\n\n"

        "=== VIRAL MECHANICS ===\n"
        "- Explain what hook makes someone stop scrolling.\n"
        "- Explain why this content performs well on Instagram.\n"
    )


def _valid_aspect(value: str) -> str:
    """'W:H' snapped to the nearest ratio image generators accept; '1:1' if unreadable."""
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*[:x/]\s*(\d+(?:\.\d+)?)\s*", value or "")
    if not match or float(match.group(2)) == 0:
        return "1:1"
    return closest_aspect_ratio(float(match.group(1)), float(match.group(2)))


def _with_aspect_flag(prompt: str, aspect: str) -> str:
    """Exactly one correct Midjourney '--ar', whatever Gemini wrote."""
    cleaned = re.sub(r"\s*--(?:ar|aspect)\s+\S+", "", prompt).rstrip()
    return f"{cleaned} --ar {aspect}"


def _plan_prompt_image(analysis: ImageAnalysis) -> str:
    aspect = _valid_aspect(analysis.aspect_ratio)
    overlays_desc = ""
    if analysis.has_text and analysis.text_overlays:
        overlay_lines = []
        for i, ov in enumerate(analysis.text_overlays, 1):
            overlay_lines.append(
                f"  [{i}] Text: '{ov.text}' | Position: {ov.position} | Style: {ov.style}"
            )
        overlays_desc = "ORIGINAL TEXT OVERLAYS (must be preserved):\n" + "\n".join(overlay_lines) + "\n\n"

    return (
        f"You are a viral content recreation expert specialising in Instagram image posts.\n"
        f"Generate a complete recreation plan for the following image.\n\n"

        f"=== SOURCE ANALYSIS ===\n"
        f"Content type: {analysis.content_type}\n"
        f"Visual medium: {analysis.visual_medium}\n"
        f"Subject: {analysis.subject}\n"
        f"Composition: {analysis.composition}\n"
        f"Color palette: {analysis.color_palette}\n"
        f"Lighting: {analysis.lighting}\n"
        f"Background: {analysis.background}\n"
        f"Mood: {analysis.mood}\n"
        f"Aspect ratio: {aspect}\n"
        f"Hook: {analysis.hook}\n"
        f"Why it works: {analysis.why_it_works}\n\n"

        f"{overlays_desc}"

        "=== IMAGE GENERATION RULES ===\n"
        "- nano_banana_prompt: natural language, most important elements first, very detailed.\n"
        f"- flux_midjourney_prompt: same content + append '--ar {aspect} --q 2 --style raw'.\n"
        "- stable_diffusion_prompt: comma-separated tokens, include 'masterpiece, best quality'.\n"
        "- negative_prompt: always include 'blurry, low quality, watermark, text, ugly, deformed'.\n\n"

        "=== TEXT OVERLAY RULES ===\n"
        "- If has_text=True: text_overlays MUST recreate all original text exactly.\n"
        "- Each overlay needs: text, position, style, purpose, canva_instruction.\n"
        "- canva_workflow: numbered steps to recreate all overlays in Canva/CapCut.\n"
        "- If has_text=False: text_overlays=[]. canva_workflow='No text overlays needed.'.\n\n"

        "=== AVOID RULES ===\n"
        "- Include specific avoid items: wrong color palette, style drift, unwanted text\n"
        "  (if original has no text, note 'no text overlays — pure visual image').\n"
    )


def analyze_image(path: Path, notify: Notify = _noop_notify) -> ImageAnalysis:
    """Send the image to Gemini Vision and return a structured ImageAnalysis."""
    from google.genai import types as genai_types

    notify("Analysing image with Gemini Vision...")
    client = get_client()
    image_bytes = path.read_bytes()
    suffix = path.suffix.lower().lstrip(".")
    mime_map = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png",
                "webp": "image/webp", "gif": "image/gif"}
    mime = mime_map.get(suffix, "image/jpeg")

    def run(model: str) -> ImageAnalysis:
        response = client.models.generate_content(
            model=model,
            contents=[
                genai_types.Part.from_bytes(data=image_bytes, mime_type=mime),
                _analysis_prompt_image(),
            ],
            config=genai_types.GenerateContentConfig(
                http_options=gemini_http_options(TIMEOUT_IMAGE_SECONDS),
                response_mime_type="application/json",
                response_schema=ImageAnalysis,
                temperature=0.4,
                automatic_function_calling=genai_types.AutomaticFunctionCallingConfig(disable=True),
            ),
        )
        return _parse_structured(response, ImageAnalysis)

    return call_gemini(run, notify, task="analysis", label="image-analysis")


def generate_image_plan(
    analysis: ImageAnalysis,
    notify: Notify = _noop_notify,
) -> ImageRecreationPlan:
    """Generate the image recreation plan (prompts + text overlays + audio)."""
    from google.genai import types as genai_types

    notify("Writing image recreation prompts...")
    client = get_client()
    request = _plan_prompt_image(analysis)

    def run(model: str) -> ImageRecreationPlan:
        response = client.models.generate_content(
            model=model,
            contents=[request],
            config=genai_types.GenerateContentConfig(
                http_options=gemini_http_options(TIMEOUT_TEXT_SECONDS),
                response_mime_type="application/json",
                response_schema=ImageRecreationPlan,
                temperature=0.7,
                automatic_function_calling=genai_types.AutomaticFunctionCallingConfig(disable=True),
            ),
        )
        return _parse_structured(response, ImageRecreationPlan)

    plan = call_gemini(run, notify, task="text", label="image-plan")
    aspect = _valid_aspect(analysis.aspect_ratio)
    return plan.model_copy(update={
        "aspect_ratio": aspect,
        "flux_midjourney_prompt": _with_aspect_flag(plan.flux_midjourney_prompt, aspect),
    })


def generate_social_metadata_for_image(
    analysis: ImageAnalysis,
    plan: ImageRecreationPlan,
    notify: Notify = _noop_notify,
) -> SocialMetadata:
    """YouTube + Instagram metadata for the image post (one Gemini call,
    grounded in Google Search when today's free search quota allows)."""
    overlay_text = " | ".join(overlay.text for overlay in analysis.text_overlays if overlay.text.strip())
    context = "\n".join(part for part in [
        f"Content type: {analysis.content_type} ({analysis.visual_medium})",
        f"Summary: {analysis.summary.strip()}",
        f"Subject: {analysis.subject.strip()}",
        f"Hook: {analysis.hook.strip()}",
        f"Mood: {analysis.mood.strip()}",
        f"Recreated post title: {plan.title}",
        f"Logline: {plan.logline}",
        f"Text in the image: {overlay_text}" if overlay_text else "",
    ] if part)
    return generate_metadata(context, "image", "English", notify)


# ============================================================================
# Rendering
# ============================================================================

def render_audio_section(audio: SuggestedAudio) -> str:
    lines = [
        "=== RECOMMENDED AUDIO (for Instagram Reel) ===",
        "",
        f"Primary track:",
        f"  Track : {audio.track_name}",
        f"  Artist: {audio.artist_name}",
        f"  Search on Instagram: {audio.instagram_search_query}",
        f"  Why it fits: {audio.mood_match}",
        f"  Genre: {audio.genre} | Feel: {audio.bpm_feel}",
        "",
        f"Alternative (if primary unavailable):",
        f"  Track : {audio.alternative_track_name}",
        f"  Artist: {audio.alternative_artist_name}",
        f"  Search on Instagram: {audio.alternative_search_query}",
    ]
    return "\n".join(lines)


def render_image_prompts(plan: ImageRecreationPlan, analysis: ImageAnalysis) -> str:
    """Render the full copy-paste output for image recreation."""
    lines = [
        "=== IMAGE RECREATION PROMPTS ===",
        f"Title: {plan.title}",
        f"Logline: {plan.logline}",
        f"Aspect ratio: {plan.aspect_ratio}",
        "",
        "─" * 55,
        "",
        "=== NANO BANANA / NANO BANANA PRO PROMPT ===",
        "(Paste directly into Nano Banana or Nano Banana Pro)",
        "",
        plan.nano_banana_prompt,
        "",
        "─" * 55,
        "",
        "=== FLUX / MIDJOURNEY / LEONARDO PROMPT ===",
        "",
        plan.flux_midjourney_prompt,
        "",
        "─" * 55,
        "",
        "=== STABLE DIFFUSION / COMFYUI PROMPT ===",
        "",
        plan.stable_diffusion_prompt,
        "",
        "=== NEGATIVE PROMPT ===",
        "",
        plan.negative_prompt,
        "",
        "─" * 55,
        "",
    ]

    if analysis.has_text and plan.text_overlays:
        lines += [
            "=== TEXT OVERLAYS (add in Canva / CapCut) ===",
            "(The original image has text — recreate these overlays after generating the base image)",
            "",
        ]
        for i, ov in enumerate(plan.text_overlays, 1):
            lines += [
                f"[{i}] Text: '{ov.text}'",
                f"    Position: {ov.position}",
                f"    Style: {ov.style}",
                f"    Purpose: {ov.purpose}",
                f"    Canva step: {ov.canva_instruction}",
                "",
            ]
        lines += [
            "--- Canva / CapCut Full Workflow ---",
            plan.canva_workflow,
            "",
            "─" * 55,
            "",
        ]
    else:
        lines += [
            "=== TEXT OVERLAYS ===",
            "No text overlays in the original image — generate the base image as-is.",
            "",
            "─" * 55,
            "",
        ]

    lines.append(render_audio_section(plan.suggested_audio))
    return "\n".join(lines)


# ============================================================================
# Orchestration + cache
# ============================================================================

@dataclass
class ImageAnalysisResult:
    image_sha: str
    image_path: str
    source_url: str
    title: str
    extractor: str
    analysis: dict
    from_cache: bool = False


@dataclass
class GeneratedImagePrompt:
    prompt_text: str        # full render_image_prompts() output
    plan: dict
    social_text: str
    social_metadata: "SocialMetadata | None"
    saved_to: str

    def full_text(self) -> str:
        parts = [self.prompt_text]
        if self.social_text:
            parts.append("\n\n=== SOCIAL MEDIA METADATA ===\n\n" + self.social_text)
        return "\n\n".join(parts)


def _image_cache_path(sha: str) -> Path:
    return CONFIG.work_dir / "image_cache" / f"{sha}.json"


def _load_image_cached(sha: str) -> ImageAnalysisResult | None:
    p = _image_cache_path(sha)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        data["from_cache"] = True
        result = ImageAnalysisResult(**data)
        ImageAnalysis.model_validate(result.analysis)
        return result
    except (OSError, ValueError, TypeError) as exc:
        logger.warning(f"[CACHE] ignoring unreadable image cache {p.name}: {exc}")
        return None


def _pixel_aspect_ratio(path: Path) -> str | None:
    try:
        from PIL import Image

        with Image.open(path) as image:
            width, height = image.size
    except Exception as exc:  # unreadable / unsupported format: keep Gemini's value
        logger.warning(f"[IMAGE] could not read size of {path.name}: {exc}")
        return None
    return closest_aspect_ratio(width, height)


def _analyze_image_file(
    path: Path, *, source_url: str, title: str, extractor: str, notify: Notify = _noop_notify,
) -> ImageAnalysisResult:
    """Hash -> cache -> Gemini -> cache, for a downloaded or uploaded image.
    The image file is always deleted afterwards (success, failure or cache
    hit); only the analysis JSON is kept."""
    try:
        image_sha = _sha256(path)
        if cached := _load_image_cached(image_sha):
            notify("Using saved analysis for this image")
            return cached
        analysis = analyze_image(path, notify)
        # A screenshot's pixels include the phone UI, so for uploads Gemini's
        # reading of the post's own shape is kept; downloads use real pixels.
        if extractor != "upload" and (measured := _pixel_aspect_ratio(path)):
            analysis = analysis.model_copy(update={"aspect_ratio": measured})
    finally:
        try:
            path.unlink(missing_ok=True)
            logger.info(f"[CLEANUP] deleted image after analysis: {path.name}")
        except OSError as exc:
            logger.warning(f"[CLEANUP] could not delete image {path.name}: {exc}")

    result = ImageAnalysisResult(
        image_sha=image_sha,
        image_path=str(path),  # stored for reference; the file is deleted
        source_url=source_url,
        title=title,
        extractor=extractor,
        analysis=analysis.model_dump(),
    )
    _image_cache_path(image_sha).parent.mkdir(parents=True, exist_ok=True)
    _image_cache_path(image_sha).write_text(
        json.dumps(asdict(result), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return result


def analyze_local_image(path: Path, title: str = "Uploaded image", notify: Notify = _noop_notify) -> ImageAnalysisResult:
    """Analyse an uploaded image, with caching. The file is deleted afterwards."""
    return _analyze_image_file(path, source_url=f"upload://{title}", title=title, extractor="upload", notify=notify)


def analyze_image_url(url: str, notify: Notify = _noop_notify) -> ImageAnalysisResult:
    """Download and analyse an image post. Returns cached result if available.
    For Instagram URLs: uses the requests-based downloader (bypasses yt-dlp 400 errors).
    For all other URLs: uses yt-dlp as before.
    Image file is deleted immediately after Gemini processes it.
    """
    safe_url = validate_public_video_url(url)
    downloads = CONFIG.work_dir / "image_downloads"
    temp_dir = downloads / f"tmp_{uuid.uuid4().hex[:10]}"

    try:
        is_instagram = host_is(urlsplit(safe_url).hostname, "instagram.com")

        if is_instagram:
            # Instagram: use requests + cookies (bypasses yt-dlp HTTP 400)
            notify("Downloading Instagram image...")
            dl_img = download_instagram_image(safe_url, temp_dir, notify)
            path = dl_img.path
            title = dl_img.title
            source_url = dl_img.source_url
            extractor = "Instagram"
        else:
            # Other URLs: use yt-dlp as before
            notify("Downloading image...")
            downloaded = download_media(safe_url, temp_dir, notify)
            if downloaded.media_type != "image":
                raise URLVideoError(
                    "This URL points to a video, not an image. "
                    "Use the Video tab to recreate videos."
                )
            path = downloaded.path
            title = downloaded.title
            source_url = downloaded.source_url
            extractor = downloaded.extractor

        return _analyze_image_file(path, source_url=source_url, title=title, extractor=extractor, notify=notify)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)



def add_image_social(generated: GeneratedImagePrompt, result: ImageAnalysisResult,
                     notify: Notify = _noop_notify) -> GeneratedImagePrompt:
    """The prompts plus their YouTube / Instagram metadata (one Gemini call),
    also written into the saved prompt file."""
    import dataclasses

    analysis = ImageAnalysis.model_validate(result.analysis)
    plan = ImageRecreationPlan.model_validate(generated.plan)
    social = generate_social_metadata_for_image(analysis, plan, notify)
    updated = dataclasses.replace(generated, social_metadata=social, social_text=render_social_metadata(social))
    saved = Path(updated.saved_to)
    saved.parent.mkdir(parents=True, exist_ok=True)
    saved.write_text(updated.full_text(), encoding="utf-8")
    return updated


def generate_image_recreation(
    result: ImageAnalysisResult,
    notify: Notify = _noop_notify,
    include_social: bool = True,
) -> GeneratedImagePrompt:
    """Generate all prompts: image gen + audio, plus social metadata unless
    include_social is False (the UI adds it after showing the prompts)."""
    analysis = ImageAnalysis.model_validate(result.analysis)
    plan = generate_image_plan(analysis, notify)
    prompt_text = render_image_prompts(plan, analysis)

    prompts_dir = CONFIG.work_dir / "image_prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{result.image_sha[:12]}_{time.strftime('%Y%m%d-%H%M%S')}"
    saved_to = str(prompts_dir / f"{stem}.txt")

    generated = GeneratedImagePrompt(
        prompt_text=prompt_text,
        plan=plan.model_dump(),
        social_text="",
        social_metadata=None,
        saved_to=saved_to,
    )
    if include_social:
        try:
            generated = add_image_social(generated, result, notify)
        except Exception as exc:
            logger.warning(f"[SOCIAL] failed: {exc}")
    (prompts_dir / f"{stem}.txt").write_text(generated.full_text(), encoding="utf-8")
    (prompts_dir / f"{stem}.json").write_text(plan.model_dump_json(indent=2), encoding="utf-8")
    logger.info(f"[IMAGE-PROMPT] saved {stem}.txt")
    return generated


# ============================================================================
# Shared Streamlit helpers
# ============================================================================

def _supported_kwargs(func, **kwargs):
    import inspect
    try:
        params = inspect.signature(func).parameters
    except (TypeError, ValueError):
        return {}
    return {k: v for k, v in kwargs.items() if k in params}


def _show_copyable(st, text: str) -> None:
    st.code(text, language=None, **_supported_kwargs(st.code, wrap_lines=True))


USER_FACING_ERRORS = (URLVideoError, ValueError, RuntimeError)


# ============================================================================
# Streamlit UI
# ============================================================================

def run_app() -> None:
    import streamlit as st

    st.set_page_config(
        page_title="Recreate image post",
        page_icon="🖼️",
        layout="centered",
    )
    st.title("Recreate any Instagram image post")
    st.caption(
        "Paste a public Instagram image post URL. "
        "Gemini analyses it and writes copy-paste prompts for "
        "Nano Banana, Nano Banana Pro, FLUX, Midjourney, SD — plus text overlays, "
        "audio recommendation, and social media metadata."
    )

    problems = CONFIG.problems()
    if problems:
        for p in problems:
            st.error(p)
        st.stop()

    st.session_state.setdefault("image_result", None)
    st.session_state.setdefault("image_generated", None)

    with st.form("image_form"):
        url = st.text_input(
            "Instagram image post URL",
            placeholder="https://www.instagram.com/p/..."
        )
        submitted = st.form_submit_button(
            "Analyze image",
            type="primary",
            **_supported_kwargs(st.form_submit_button, icon=":material/image_search:"),
        )

    if submitted:
        with st.status("Starting...", expanded=True) as status:
            try:
                result = analyze_image_url(url, lambda msg: status.update(label=msg))
            except USER_FACING_ERRORS as exc:
                status.update(label="Analysis failed", state="error")
                st.error(str(exc))
            else:
                st.session_state.image_result = result
                st.session_state.image_generated = None
                label = "Analysis loaded from cache" if result.from_cache else "Analysis complete"
                status.update(label=label, state="complete", expanded=False)

    result: ImageAnalysisResult | None = st.session_state.image_result
    if not result:
        return

    analysis = ImageAnalysis.model_validate(result.analysis)

    # ── Post info ────────────────────────────────────────────────────────────
    with st.container(border=True):
        st.subheader(result.title)
        left, right = st.columns([1, 2])
        with left:
            try:
                st.image(result.image_path, use_column_width=True)
            except Exception:
                st.caption("(preview unavailable)")
        with right:
            content = analysis.content_type.replace("_", " ")
            medium = analysis.visual_medium.replace("_", " ")
            st.markdown(
                f":blue[**{content}**] · :violet[**{medium}**] · "
                f"{analysis.aspect_ratio}"
            )
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

    # ── Generate button ───────────────────────────────────────────────────────
    if st.button(
        "Generate recreation prompts",
        type="primary",
        **_supported_kwargs(st.button, icon=":material/auto_awesome:"),
    ):
        with st.status("Generating prompts...", expanded=True) as status:
            try:
                generated = generate_image_recreation(
                    result, lambda msg: status.update(label=msg)
                )
            except USER_FACING_ERRORS as exc:
                status.update(label="Generation failed", state="error")
                st.error(str(exc))
            else:
                st.session_state.image_generated = generated
                status.update(label="Prompts ready", state="complete", expanded=False)

    generated: GeneratedImagePrompt | None = st.session_state.image_generated
    if not generated:
        return

    plan = ImageRecreationPlan.model_validate(generated.plan)

    if CONFIG.show_prompt:
        # ── Results tabs ──────────────────────────────────────────────────────
        tab_nb, tab_flux, tab_sd, tab_text, tab_audio, tab_social = st.tabs([
            "🍌 Nano Banana",
            "🌀 FLUX / MJ",
            "🖥️ Stable Diffusion",
            "✍️ Text Overlays",
            "🎵 Audio",
            "📲 Social",
        ])

        with tab_nb:
            st.subheader("Nano Banana / Nano Banana Pro Prompt")
            st.caption("Paste this directly into Nano Banana or Nano Banana Pro.")
            _show_copyable(st, plan.nano_banana_prompt)

        with tab_flux:
            st.subheader("FLUX / Midjourney / Leonardo Prompt")
            _show_copyable(st, plan.flux_midjourney_prompt)

        with tab_sd:
            st.subheader("Stable Diffusion / ComfyUI / A1111")
            st.markdown("**Positive prompt:**")
            _show_copyable(st, plan.stable_diffusion_prompt)
            st.markdown("**Negative prompt:**")
            _show_copyable(st, plan.negative_prompt)

        with tab_text:
            st.subheader("Text Overlay Recreation")
            if analysis.has_text and plan.text_overlays:
                st.info(
                    f"The original has {len(plan.text_overlays)} text overlay(s). "
                    "Add these AFTER generating your base image in Canva or CapCut."
                )
                for i, ov in enumerate(plan.text_overlays, 1):
                    with st.container(border=True):
                        st.markdown(f"**[{i}] Text:**")
                        _show_copyable(st, ov.text)
                        st.markdown(
                            f"**Position:** {ov.position} | "
                            f"**Style:** {ov.style} | "
                            f"**Purpose:** {ov.purpose}"
                        )
                        st.caption(f"Canva step: {ov.canva_instruction}")
                st.divider()
                st.markdown("**Full Canva / CapCut Workflow:**")
                _show_copyable(st, plan.canva_workflow)
            else:
                st.info("No text overlays in the original — generate the base image as-is.")

        with tab_audio:
            st.subheader("Recommended Audio for Instagram Reel")
            audio = plan.suggested_audio
            with st.container(border=True):
                st.markdown(f"**Primary: {audio.track_name}** by {audio.artist_name}")
                st.markdown(f"Genre: {audio.genre} | Feel: {audio.bpm_feel}")
                st.caption(audio.mood_match)
                st.markdown("**Search on Instagram audio:**")
                _show_copyable(st, audio.instagram_search_query)
            with st.container(border=True):
                st.markdown(
                    f"**Alternative: {audio.alternative_track_name}** "
                    f"by {audio.alternative_artist_name}"
                )
                st.markdown("**Search on Instagram audio:**")
                _show_copyable(st, audio.alternative_search_query)

        with tab_social:
            st.subheader("📲 Post to Social Media")
            render_social(st, generated.social_metadata)

    st.divider()
    st.download_button(
        "Download all prompts",
        data=generated.full_text(),
        file_name="image_recreation.txt",
        mime="text/plain",
        **_supported_kwargs(st.download_button, icon=":material/download:"),
    )
    st.caption(f"Saved to {generated.saved_to}")


# ============================================================================
# Entry
# ============================================================================

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
        import sys
        print("Run as: streamlit run recreate_image.py")
        sys.exit(0)
