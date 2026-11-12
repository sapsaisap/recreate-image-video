"""
summary_short.py
================
"Summarise to Short": a YouTube / Instagram video of up to 5 minutes becomes
a 10, 15 or 20-second vertical Short / Reel that tells its content in brief:
still images (real frames from the source, or AI images the user makes in
Google Flow from our prompts) with gentle motion, and a voice-over in the
source's language.

Steps (one screen each):
  1. input    link or upload + target length -> Analyse
  2. prompts  per scene: real frame / AI-from-reference / full AI, prompts
  3. images   real frames are filled in; the user uploads the AI images
  4. voice    Edge TTS voice-over (language + male/female), fitted to length
  5. done     render, preview, download

Gemini: 2 free-tier calls per video (analysis + fact-checked script), plus
at most one more to shorten or translate the script.
"""
from __future__ import annotations

import io
import logging
import re
import shutil
import uuid
import zipfile
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

import slideshow
import voiceover
from recreate_video import (
    CONFIG,
    TIMEOUT_LONG_VIDEO_SECONDS,
    TIMEOUT_TEXT_SECONDS,
    USER_FACING_ERRORS,
    MediaInfo,
    Notify,
    _noop_notify,
    _parse_structured,
    _show_copyable,
    _upload_and_wait,
    call_gemini,
    download_video,
    gemini_http_options,
    get_client,
    generate_with_optional_search,
    grounding_label,
    probe_video,
    validate_media,
)

logger = logging.getLogger(__name__)
_LOADED_MTIME = Path(__file__).stat().st_mtime  # see recreate_video.drop_stale_app_modules

DURATIONS = (10, 15, 20)
DEFAULT_DURATION = 15
REAL_FRAME_SHARE = 1 / 3  # at most this share of scenes reuse the creator's footage
# Measured speaking speed of the Edge TTS voices (words per second at +0%).
# Telugu measured at ~1.4 with Mohan; the rest are conservative estimates.
WORDS_PER_SECOND = {"te": 1.4, "ta": 1.4, "ml": 1.3, "kn": 1.5, "mr": 1.8, "gu": 1.9, "bn": 2.0,
                    "hi": 2.3, "en": 2.5}
DEFAULT_WORDS_PER_SECOND = 2.0
TRANSCRIPT_CHARS_IN_PROMPT = 8000
# The script call searches Google and reads ~15 frame images: 74 s was
# measured on a 160 s video, too close to the usual 90 s limit.
STORY_TIMEOUT_SECONDS = 180
VISUAL_TYPES = ("real_frame", "ai_from_reference", "ai")
TYPE_LABELS = {"real_frame": "📷 Real frame", "ai_from_reference": "🎨 AI from reference", "ai": "✨ Full AI"}


# ============================================================================
# Schemas
# ============================================================================

class TranscriptLine(BaseModel):
    start_seconds: float
    text: str = Field(description="Exactly what is said, in the spoken language and its own script.")


class VisualMoment(BaseModel):
    timestamp_seconds: float = Field(description="The single clearest instant of this moment.")
    description: str = Field(description="What is on screen, concretely, in English.")
    why_it_matters: str = Field(description="What this picture explains about the content.")
    kind: Literal["place", "product", "person", "chart_or_text", "screen_or_demo", "action", "object", "other"]
    has_burned_in_text: bool = Field(description="Captions, subtitles or titles drawn on the picture.")
    has_watermark_or_ui: bool = Field(description="A channel logo, watermark or app interface on the picture.")
    has_identifiable_person: bool = Field(description="A real person's face is clearly visible.")
    focus_box: list[int] = Field(description="[ymin, xmin, ymax, xmax] of the main subject, each 0-1000.")


class SourceAnalysis(BaseModel):
    title: str
    spoken_language: str = Field(description="Main spoken language as an English name, e.g. 'Telugu', or 'none'.")
    language_code: str = Field(description="Two-letter code of the spoken language, e.g. 'te', 'hi', 'en'.")
    topic: str
    summary: str = Field(description="3-5 sentences in English: what the video says and shows.")
    key_points: list[str] = Field(description="The main points, in the order the video makes them, in English.")
    transcript: list[TranscriptLine] = Field(description="Everything said, word for word, in order.")
    visual_style: str = Field(description="Look of the footage: live action / animation, colours, lighting.")
    visual_moments: list[VisualMoment] = Field(
        description="8-15 of the clearest, most informative moments, spread across the whole video."
    )


class StoryScene(BaseModel):
    narration: str
    visual_type: Literal["real_frame", "ai_from_reference", "ai"]
    type_reason: str
    moment: int = Field(description="Number of the source moment this scene shows or uses (M1 = 1); 0 for none.")
    image_prompt: str


class FactCheck(BaseModel):
    claim: str
    verdict: Literal["confirmed", "disputed", "unverified"]
    note: str


class StoryPlan(BaseModel):
    hook_title: str
    scenes: list[StoryScene]
    fact_checks: list[FactCheck]


class ScriptRewrite(BaseModel):
    narrations: list[str]


# ============================================================================
# Planning helpers (pure)
# ============================================================================

SCENE_RANGES = {10: (3, 4), 15: (4, 6), 20: (5, 7)}  # shots of roughly 2.5-4 s


def scene_range(duration: int) -> tuple[int, int]:
    """(fewest, most) scenes for a Short of this length."""
    if duration in SCENE_RANGES:
        return SCENE_RANGES[duration]
    return max(2, int(duration / 4 + 0.5)), max(3, int(duration / 2.8 + 0.5))


def words_per_second(language_code: str) -> float:
    return WORDS_PER_SECOND.get((language_code or "").lower(), DEFAULT_WORDS_PER_SECOND)


def word_budget(duration: int, language_code: str) -> int:
    """Total narration words; 8% headroom so the voice rarely needs speeding up."""
    return max(6, int(duration * words_per_second(language_code) * 0.92))


def max_real_frames(scene_count: int) -> int:
    return max(1, int(scene_count * REAL_FRAME_SHARE))


def finalise_plan(plan: StoryPlan, analysis: SourceAnalysis, duration: int) -> StoryPlan:
    """Enforces the rules Gemini may bend: scene count, valid moments, and
    when real footage may be reused (no faces, burned-in text or watermarks,
    at most REAL_FRAME_SHARE of the scenes)."""
    _, most = scene_range(duration)
    scenes = [scene.model_copy() for scene in plan.scenes if scene.narration.strip()]
    if len(scenes) > most:  # fold extra narration into the last kept scene
        extra = " ".join(scene.narration.strip() for scene in scenes[most:])
        scenes = scenes[:most]
        scenes[-1] = scenes[-1].model_copy(update={"narration": f"{scenes[-1].narration.strip()} {extra}"})
    moments = analysis.visual_moments
    real_left = max_real_frames(len(scenes))
    used: set[int] = set()
    fixed = []
    for scene in scenes:
        update: dict[str, Any] = {}
        moment = scene.moment if 1 <= scene.moment <= len(moments) else 0
        if moment in used:
            moment = 0
        update["moment"] = moment
        kind = scene.visual_type
        if kind != "ai" and not moment:
            kind, update["type_reason"] = "ai", "No source moment to use, so a full AI image."
        elif kind == "real_frame":
            source = moments[moment - 1]
            reason = (
                "a real person's face is visible" if source.has_identifiable_person
                else "the frame has burned-in text" if source.has_burned_in_text
                else "the frame has a watermark or app interface" if source.has_watermark_or_ui
                else "enough scenes already reuse real footage" if real_left == 0
                else None
            )
            if reason:
                kind, update["type_reason"] = "ai_from_reference", f"AI from reference: {reason}."
            else:
                real_left -= 1
        if moment:
            used.add(moment)
        update["visual_type"] = kind
        fixed.append(scene.model_copy(update=update))
    return plan.model_copy(update={"scenes": fixed})


def _moment_lines(analysis: SourceAnalysis) -> str:
    lines = []
    for number, moment in enumerate(analysis.visual_moments, start=1):
        flags = [name for name, flag in (("text on screen", moment.has_burned_in_text),
                                         ("watermark/UI", moment.has_watermark_or_ui),
                                         ("real person's face", moment.has_identifiable_person)) if flag]
        lines.append(
            f"  M{number} [{moment.timestamp_seconds:.1f}s] {moment.kind}: {moment.description.strip()} "
            f"-- {moment.why_it_matters.strip()}" + (f" ({', '.join(flags)})" if flags else "")
        )
    return "\n".join(lines)


def story_prompt(analysis: SourceAnalysis, source_seconds: float, duration: int, search: bool,
                 allow_text: bool = False) -> str:
    """allow_text: image prompts may carry the source's important on-screen
    text (quoted); by default scene images are text-free, since they become
    video frames."""
    fewest, most = scene_range(duration)
    language = analysis.spoken_language if analysis.spoken_language.lower() != "none" else "English"
    code = analysis.language_code or "en"
    transcript = "\n".join(f"  [{line.start_seconds:.0f}s] {line.text.strip()}" for line in analysis.transcript)
    if len(transcript) > TRANSCRIPT_CHARS_IN_PROMPT:
        transcript = transcript[:TRANSCRIPT_CHARS_IN_PROMPT] + "\n  ..."
    fact_rule = (
        "Use Google Search to check the 2-4 main factual claims you put in the script."
        if search else
        "From your own knowledge, check the 2-4 main factual claims you put in the script."
    )
    reply = (
        "\nReply with ONLY one JSON object and no other text: {\"hook_title\": str, \"scenes\": [{\"narration\": "
        "str, \"visual_type\": \"real_frame\"|\"ai_from_reference\"|\"ai\", \"type_reason\": str, \"moment\": int, "
        "\"image_prompt\": str}], \"fact_checks\": [{\"claim\": str, \"verdict\": \"confirmed\"|\"disputed\"|"
        "\"unverified\", \"note\": str}]}"
        if search else ""
    )
    return (
        f"You are turning a {source_seconds:.0f}-second video into a {duration}-second vertical Short / Reel that "
        "summarises it for viewers who will not watch the original.\n\n"
        "=== SOURCE ===\n"
        f"Title: {analysis.title}\nTopic: {analysis.topic}\nSummary: {analysis.summary}\n"
        "Key points:\n" + "\n".join(f"  - {point}" for point in analysis.key_points) + "\n"
        f"Look of the footage: {analysis.visual_style}\n"
        f"Transcript ({language}):\n{transcript}\n"
        "Visual moments (their frames are attached above, labelled M1, M2, ...):\n"
        f"{_moment_lines(analysis)}\n\n"
        "=== SCRIPT ===\n"
        f"- {fewest}-{most} scenes. Scene 1 is the hook: the most surprising or useful point, in its first words.\n"
        f"- narration in {language}, written in {language}'s own script (not transliterated), natural spoken style "
        f"in the source's register. About {word_budget(duration, code)} words in total -- never more; the voice "
        f"speaks about {words_per_second(code)} words per second.\n"
        "- Tell the source's real content. Add nothing the source does not say. The last scene ends on the "
        "takeaway, not on 'like and subscribe'.\n\n"
        "=== VISUALS (one per scene) ===\n"
        "- visual_type:\n"
        "  * real_frame: the source moment itself is the clearest picture (a real place, product, chart, screen "
        "or demo whose details must be exact). Never for frames with text on screen, a watermark/UI or a real "
        "person's face.\n"
        "  * ai_from_reference: show what a source moment shows, cleaner or more striking; that moment's frame "
        "is attached as the reference in the image generator.\n"
        "  * ai: ONLY for an idea the video never shows (normally at most one scene).\n"
        "- The Short must show what the video shows: every scene about something on screen in the video MUST "
        "set moment to that moment's number -- real_frame where allowed, otherwise ai_from_reference (faces, "
        "on-screen text and watermarks are fine as a reference; the new image replaces them).\n"
        f"- At most {max_real_frames(most)} real_frame scenes. Use each moment at most once; moment = 0 only for ai.\n"
        "- image_prompt, in English, for EVERY scene (also real_frame, as a fallback): ONE single full-frame "
        "vertical 9:16 shot -- subject, setting, composition, lighting, colours, lens, mood; photorealistic "
        "unless the source is animated. "
        + (
            "Where the source shows important on-screen text for this moment, put it in the prompt inside "
            "double quotes, word for word in its own language and script, with position, font and colour; "
            "no other text. No logos, watermarks, split screens, collages or panels. "
            if allow_text else
            "NO text, letters, numbers, captions, logos, watermarks, split screens, collages or panels. "
        )
        + "No real names or likenesses: describe people generically, and describe recurring "
        "characters identically in every scene.\n"
        "- For ai_from_reference, start the prompt with: \"Using the attached reference image, keep its layout "
        "and subject;\"\n"
        "- type_reason: one short sentence on why this visual type.\n\n"
        "=== FACT CHECK ===\n"
        f"- {fact_rule} fact_checks: claim (English), verdict, and a note naming where it is confirmed or what "
        "disputes it. Leave disputed claims out of the narration.\n"
        "- hook_title: a short title for this Short, in English.\n"
        f"{reply}"
    )


def prompts_text(plan: StoryPlan, analysis: SourceAnalysis, source_url: str) -> str:
    lines = [f"{plan.hook_title}", f"Source: {analysis.title} -- {source_url}", ""]
    for number, scene in enumerate(plan.scenes, start=1):
        moment = analysis.visual_moments[scene.moment - 1] if scene.moment else None
        lines += [
            f"=== Scene {number} · {TYPE_LABELS[scene.visual_type]} ===",
            f"Narration: {scene.narration}",
            f"Reference frame: M{scene.moment} at {moment.timestamp_seconds:.1f}s" if moment else "Reference frame: none",
            "Image prompt:" if scene.visual_type != "real_frame" else "Fallback image prompt:",
            scene.image_prompt,
            "",
        ]
    return "\n".join(lines)


def assign_bulk_uploads(names: list[str], slots: list[int]) -> dict[int, str]:
    """Bulk-uploaded files fill the AI scenes in natural filename order
    (scene2 before scene10)."""
    def natural(name: str):
        return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", name)]

    return dict(zip(slots, sorted(names, key=natural)))


# ============================================================================
# Gemini + ffmpeg steps
# ============================================================================

def acquire_source(url: str, upload: tuple[str, bytes] | None, job: Path, notify: Notify) -> tuple[Path, str, str]:
    """The source video in `job` (download or upload): (path, title, source url)."""
    if upload:
        name, data = upload
        path = job / f"source{Path(name).suffix.lower() or '.mp4'}"
        path.write_bytes(data)
        return path, Path(name).stem, f"upload://{name}"
    notify("Downloading the video...")
    downloaded = download_video(url, job / "download", notify,
                                max_seconds=CONFIG.summary_max_video_seconds, max_height=1080)
    return downloaded.path, downloaded.title, downloaded.source_url


def analyse_source(path: Path, media: MediaInfo, notify: Notify = _noop_notify) -> SourceAnalysis:
    """One Gemini call over the whole video (picture + sound). Low media
    resolution keeps a 5-minute video within the free per-minute token limit."""
    from google.genai import types as genai_types

    client = get_client()
    notify("Uploading the video to Gemini...")
    uploaded = _upload_and_wait(path, notify)
    prompt = (
        "Watch and listen to this whole video. Return: its title and topic; the spoken language; a short "
        "English summary and its key points in order; the full transcript word for word in the spoken "
        "language and its own script (never translated); the look of the footage; and 8-15 of the clearest, "
        "most informative visual moments spread across the whole video, each at the single sharpest instant, "
        "with a focus box around the main subject and flags for on-screen text, watermarks/app UI and "
        f"clearly visible real faces. Known facts: {media.duration_seconds:.0f}s, {media.width}x{media.height}; "
        f"every timestamp is between 0 and {media.duration_seconds:.0f} seconds. "
        "Never name real people; describe them by appearance."
    )
    try:
        video_part = genai_types.Part(
            file_data=genai_types.FileData(file_uri=uploaded.uri, mime_type=uploaded.mime_type),
            video_metadata=genai_types.VideoMetadata(fps=1.0),
        )

        def run(model: str) -> SourceAnalysis:
            response = client.models.generate_content(
                model=model,
                contents=[video_part, prompt],
                config=genai_types.GenerateContentConfig(
                    http_options=gemini_http_options(TIMEOUT_LONG_VIDEO_SECONDS),
                    response_mime_type="application/json",
                    response_schema=SourceAnalysis,
                    media_resolution=genai_types.MediaResolution.MEDIA_RESOLUTION_LOW,
                    temperature=0.3,
                    automatic_function_calling=genai_types.AutomaticFunctionCallingConfig(disable=True),
                ),
            )
            return _parse_structured(response, SourceAnalysis)

        notify("Analysing the video with Gemini...")
        analysis = call_gemini(run, notify, task="analysis", label="summary-analysis")
        return drop_moments_outside(analysis, media.duration_seconds)
    finally:
        try:
            client.files.delete(name=uploaded.name)
        except Exception as exc:
            logger.warning(f"[SUMMARY] could not delete uploaded file {uploaded.name}: {exc}")


def extract_moment_frames(video: Path, analysis: SourceAnalysis, folder: Path, duration: float,
                          notify: Notify = _noop_notify) -> dict[int, Path]:
    """The sharpest frame for each visual moment, as PNG (1-based numbers)."""
    frames: dict[int, Path] = {}
    for number, moment in enumerate(analysis.visual_moments, start=1):
        notify(f"Picking the sharpest frame for moment {number} of {len(analysis.visual_moments)}...")
        try:
            frames[number] = slideshow.extract_best_frame(
                video, moment.timestamp_seconds, folder / f"moment{number}.png", duration=duration
            )
        except slideshow.RenderError as exc:
            logger.warning(f"[SUMMARY] moment {number}: {exc}")
    return frames


def _frame_thumbnails(frames: dict[int, Path]) -> list:
    """Small JPEGs of the moment frames, labelled M1, M2, ... for the script call."""
    from google.genai import types as genai_types
    from PIL import Image

    parts: list = []
    for number, path in sorted(frames.items()):
        with Image.open(path) as image:
            thumb = image.convert("RGB")
            thumb.thumbnail((512, 512))
            buffer = io.BytesIO()
            thumb.save(buffer, format="JPEG", quality=85)
        parts += [f"M{number}:", genai_types.Part.from_bytes(data=buffer.getvalue(), mime_type="image/jpeg")]
    return parts


def write_story(analysis: SourceAnalysis, source_seconds: float, duration: int, frames: dict[int, Path],
                notify: Notify = _noop_notify, allow_text: bool = False,
                ) -> tuple[StoryPlan, list[str], list[dict[str, str]], str]:
    """The fact-checked script + per-scene visuals (one Gemini call, grounded
    in Google Search when the free quota allows)."""
    notify("Writing the script and image prompts (checking facts with Google Search)...")
    plan, queries, sources, model = generate_with_optional_search(
        lambda search: story_prompt(analysis, source_seconds, duration, search, allow_text),
        StoryPlan,
        task="story",
        label="summary-story",
        notify=notify,
        contents_before=_frame_thumbnails(frames),
        timeout_seconds=STORY_TIMEOUT_SECONDS,
    )
    logger.info("[SUMMARY] Gemini's visuals: " + ", ".join(f"{s.visual_type}/M{s.moment}" for s in plan.scenes))
    return drop_missing_frames(finalise_plan(plan, analysis, duration), frames), queries, sources, model


def drop_moments_outside(analysis: SourceAnalysis, duration: float) -> SourceAnalysis:
    """Moments timed after the video ends cannot be real (Gemini once gave
    216 s for a 160 s video); keeping them would point scenes at no frame."""
    kept = [m for m in analysis.visual_moments if 0 <= m.timestamp_seconds <= duration]
    if len(kept) != len(analysis.visual_moments):
        logger.info(f"[SUMMARY] dropped {len(analysis.visual_moments) - len(kept)} moment(s) outside 0-{duration:.0f}s")
    return analysis.model_copy(update={"visual_moments": kept})


def uses_source_frames(plan: StoryPlan) -> bool:
    return any(scene.moment for scene in plan.scenes)


def drop_missing_frames(plan: StoryPlan, frames: dict[int, Path]) -> StoryPlan:
    """A scene whose moment frame could not be extracted becomes full AI."""
    scenes = [
        scene if not scene.moment or scene.moment in frames else scene.model_copy(update={
            "moment": 0, "visual_type": "ai", "type_reason": "The source frame could not be read, so a full AI image.",
        })
        for scene in plan.scenes
    ]
    return plan.model_copy(update={"scenes": scenes})


def rewrite_script(narrations: list[str], instruction: str, notify: Notify = _noop_notify) -> list[str]:
    """Shortens or translates the narration, keeping one line per scene. The
    original is kept if Gemini returns a different number of lines."""
    from google.genai import types as genai_types

    client = get_client()
    numbered = "\n".join(f"{number}. {text}" for number, text in enumerate(narrations, start=1))
    request = (
        f"{instruction}\nKeep exactly {len(narrations)} lines, one per scene, in the same order and meaning. "
        f"Return narrations as a list.\n\n{numbered}"
    )

    def run(model: str) -> ScriptRewrite:
        response = client.models.generate_content(
            model=model, contents=[request],
            config=genai_types.GenerateContentConfig(
                http_options=gemini_http_options(TIMEOUT_TEXT_SECONDS), response_mime_type="application/json",
                response_schema=ScriptRewrite, temperature=0.4,
                automatic_function_calling=genai_types.AutomaticFunctionCallingConfig(disable=True),
            ),
        )
        return _parse_structured(response, ScriptRewrite)

    result = call_gemini(run, notify, task="text", label="summary-rewrite")
    lines = [line.strip() for line in result.narrations]
    return lines if len(lines) == len(narrations) and all(lines) else narrations


def shorten_script(narrations: list[str], ratio: float, language: str, notify: Notify = _noop_notify) -> list[str]:
    percent = max(40, int(ratio * 100) - 5)
    return rewrite_script(
        narrations,
        f"Shorten this {language} voice-over to about {percent}% of its length. Stay in {language}, in its own "
        "script, and keep the most important facts.",
        notify,
    )


def translate_script(narrations: list[str], language: str, notify: Notify = _noop_notify) -> list[str]:
    return rewrite_script(
        narrations,
        f"Translate this voice-over into natural spoken {language}, written in {language}'s own script. Keep "
        "it about as long as the original.",
        notify,
    )


def build_canvases(plan: StoryPlan, analysis: SourceAnalysis, frames: dict[int, Path],
                   images: dict[int, Path]) -> list:
    """One 9:16 canvas per scene: real frames cropped or blurred-fitted around
    their subject, uploaded images filled."""
    canvases = []
    for index, scene in enumerate(plan.scenes):
        if scene.visual_type == "real_frame" and index not in images:
            moment = analysis.visual_moments[scene.moment - 1]
            canvases.append(slideshow.real_frame_canvas(slideshow.load_image(frames[scene.moment]), moment.focus_box))
        else:
            canvases.append(slideshow.fill_canvas(slideshow.load_image(images[index])))
    return canvases


# ============================================================================
# Streamlit page
# ============================================================================

def _state(st) -> dict:
    return st.session_state.setdefault("ss", {"stage": "input"})


def _start_over(st) -> None:
    job = _state(st).get("job")
    if job:
        shutil.rmtree(job, ignore_errors=True)
    st.session_state["ss"] = {"stage": "input"}


def _plan(st) -> StoryPlan:
    return StoryPlan.model_validate(_state(st)["plan"])


def _analysis(st) -> SourceAnalysis:
    return SourceAnalysis.model_validate(_state(st)["analysis"])


def _frames(st) -> dict[int, Path]:
    return {int(k): Path(v) for k, v in _state(st).get("frames", {}).items()}


def _images(st) -> dict[int, Path]:
    return {int(k): Path(v) for k, v in _state(st).get("images", {}).items()}


def _key(st, name: str) -> str:
    """Widget keys include the job, so a new Short never inherits the last
    one's choices or uploads."""
    return f"ss_{name}_{Path(_state(st)['job']).name}"


def render_page(st) -> None:
    st.title("Summarise to Short")
    state = _state(st)
    steps = ["1 · Video", "2 · Prompts", "3 · Images", "4 · Voice-over", "5 · Video"]
    position = ["input", "prompts", "images", "voice", "done"].index(state["stage"])
    st.caption("  →  ".join(f"**{s}**" if i == position else s for i, s in enumerate(steps)))
    {"input": _page_input, "prompts": _page_prompts, "images": _page_images,
     "voice": _page_voice, "done": _page_done}[state["stage"]](st)


def _page_input(st) -> None:
    st.caption(
        f"Paste a YouTube or Instagram link (up to {CONFIG.summary_max_video_seconds // 60} min) or upload a video. "
        "Gemini watches it, then you get a short script, image prompts and a voice-over to make a Short that "
        "summarises it."
    )
    problems = CONFIG.problems()
    for problem in problems:
        st.error(problem)
    url = st.text_input("Video link", placeholder="https://www.youtube.com/watch?v=...", key="ss_url")
    uploaded = st.file_uploader("Or upload a video", type=["mp4", "mov", "webm", "m4v"], key="ss_upload")
    duration = st.radio("Length of the Short", DURATIONS, index=DURATIONS.index(DEFAULT_DURATION),
                        format_func=lambda s: f"{s} seconds", horizontal=True, key="ss_duration")
    allow_text = st.checkbox(
        "Include the source's on-screen text in the image prompts", value=False, key="ss_allow_text",
        help="Off: scene images are text-free (they become video frames). On: important text from the "
             "video (e.g. a title or figure) is written into the prompt, word for word.",
    )
    if not st.button("Analyse", type="primary", disabled=bool(problems) or not (url.strip() or uploaded),
                     key="ss_analyse"):
        return
    job = CONFIG.work_dir / "summary" / uuid.uuid4().hex[:12]
    job.mkdir(parents=True, exist_ok=True)
    with st.status("Starting...", expanded=True) as status:
        notify = lambda message: status.update(label=message)  # noqa: E731
        try:
            upload = (uploaded.name, uploaded.getvalue()) if uploaded is not None else None
            video, title, source_url = acquire_source(url.strip(), upload, job, notify)
            notify("Checking the video...")
            media = probe_video(video)
            validate_media(media, CONFIG.summary_max_video_seconds)
            analysis = analyse_source(video, media, notify)
            frames = extract_moment_frames(video, analysis, job / "frames", media.duration_seconds, notify)
            plan, queries, sources, model = write_story(analysis, media.duration_seconds, duration, frames, notify,
                                                 allow_text=allow_text)
        except (*USER_FACING_ERRORS, slideshow.RenderError) as exc:
            status.update(label="Analysis failed", state="error")
            st.error(str(exc))
            shutil.rmtree(job, ignore_errors=True)
            return
        finally:
            for leftover in [*job.glob("source.*"), job / "download"]:  # the source video is not kept
                shutil.rmtree(leftover, ignore_errors=True) if leftover.is_dir() else leftover.unlink(missing_ok=True)
        status.update(label="Analysis complete", state="complete", expanded=False)
    st.session_state["ss"] = {
        "stage": "prompts", "job": str(job), "duration": duration, "title": title, "source_url": source_url,
        "analysis": analysis.model_dump(), "plan": plan.model_dump(), "queries": queries, "sources": sources,
        "model": model,
        "frames": {str(k): str(v) for k, v in frames.items()}, "images": {},
    }
    st.rerun()


def _page_prompts(st) -> None:
    state, plan, analysis, frames = _state(st), _plan(st), _analysis(st), _frames(st)
    st.subheader(plan.hook_title)
    st.caption(f"From: {state['title']} · {state['source_url']} · {state['duration']} s Short · {len(plan.scenes)} scenes")
    if frames and not uses_source_frames(plan):
        st.warning("No scene uses a frame from the video, so the images may drift from what it shows. "
                   "Analyse again, or switch scenes to 🎨 AI from reference below.")
    st.caption(grounding_label(state["queries"], state["sources"], state.get("model", "")))
    with st.expander(f"Fact check ({'Google Search' if state['queries'] or state['sources'] else 'not grounded'})"):
        for check in plan.fact_checks:
            icon = {"confirmed": "✅", "disputed": "⚠️", "unverified": "❔"}[check.verdict]
            st.markdown(f"{icon} **{check.claim}** — {check.note}")
        for source in state["sources"]:
            st.markdown(f"- [{source['title']}]({source['uri']})")
    st.info(
        "Make the AI images in Google Flow (or any image generator) in 9:16. For **🎨 AI from reference**, "
        "attach the scene's frame as the reference. **📷 Real frame** scenes need nothing."
    )
    for index, scene in enumerate(plan.scenes):
        with st.container(border=True):
            options = [t for t in VISUAL_TYPES if t == "ai" or scene.moment]
            choice = st.selectbox(f"Scene {index + 1}", options, index=options.index(scene.visual_type),
                                  format_func=TYPE_LABELS.get, key=_key(st, f"type_{index}"))
            st.caption(scene.type_reason)
            left, right = st.columns([1, 2])
            with left:
                if scene.moment in frames:
                    moment = analysis.visual_moments[scene.moment - 1]
                    st.image(str(frames[scene.moment]), caption=f"Source at {moment.timestamp_seconds:.1f}s")
                    st.download_button("Download frame", frames[scene.moment].read_bytes(),
                                       file_name=f"scene{index + 1}_reference.png", mime="image/png",
                                       key=_key(st, f"frame_dl_{index}"))
            with right:
                st.markdown(f"**Voice-over:** {scene.narration}")
                if choice != "real_frame":
                    st.markdown("**Image prompt:**")
                    _show_copyable(st, scene.image_prompt)
    text = prompts_text(plan, analysis, state["source_url"])
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as archive:
        archive.writestr("prompts.txt", text)
        for index, scene in enumerate(plan.scenes):
            if scene.moment in frames:
                archive.write(frames[scene.moment], f"scene{index + 1}_reference.png")
    st.download_button("Download prompts + reference frames (zip)", zip_buffer.getvalue(),
                       file_name="short_prompts.zip", mime="application/zip", key="ss_zip")
    back, forward = st.columns(2)
    if back.button("Start over", key="ss_restart_prompts"):
        _start_over(st)
        st.rerun()
    if forward.button("Next: upload images", type="primary", key="ss_to_images"):
        scenes = [scene.model_copy(update={"visual_type": st.session_state.get(_key(st, f"type_{i}"), scene.visual_type)})
                  for i, scene in enumerate(plan.scenes)]
        state["plan"] = plan.model_copy(update={"scenes": scenes}).model_dump()
        state["stage"] = "images"
        st.rerun()


def _save_scene_image(st, index: int, name: str, data: bytes) -> None:
    """Saves an uploaded scene image -- only when it changed, since the
    uploader hands back the same file on every Streamlit rerun."""
    import hashlib

    state = _state(st)
    digest = hashlib.sha256(data).hexdigest()
    hashes = state.setdefault("image_hashes", {})
    if hashes.get(str(index)) == digest and str(index) in state["images"]:
        return
    hashes[str(index)] = digest
    folder = Path(state["job"]) / "images"
    folder.mkdir(parents=True, exist_ok=True)
    for old in folder.glob(f"scene{index + 1}.*"):
        old.unlink(missing_ok=True)
    path = folder / f"scene{index + 1}{Path(name).suffix.lower() or '.png'}"
    path.write_bytes(data)
    state["images"][str(index)] = str(path)


def _page_images(st) -> None:
    state, plan, analysis, frames = _state(st), _plan(st), _analysis(st), _frames(st)
    ai_slots = [i for i, scene in enumerate(plan.scenes) if scene.visual_type != "real_frame"]
    if ai_slots:
        bulk = st.file_uploader(
            f"Upload all {len(ai_slots)} AI images at once (assigned in filename order), or one per scene below",
            type=["png", "jpg", "jpeg", "webp"], accept_multiple_files=True, key=_key(st, "bulk"),
        )
        if bulk:
            by_name = {file.name: file for file in bulk}
            for index, name in assign_bulk_uploads(list(by_name), ai_slots).items():
                _save_scene_image(st, index, name, by_name[name].getvalue())
    images = _images(st)
    for index, scene in enumerate(plan.scenes):
        with st.container(border=True):
            st.markdown(f"**Scene {index + 1}** · {TYPE_LABELS[scene.visual_type]} — {scene.narration}")
            left, right = st.columns([1, 2])
            if scene.visual_type == "real_frame" and index not in images:
                moment = analysis.visual_moments[scene.moment - 1]
                with left:
                    st.image(str(frames[scene.moment]), caption=f"Real frame at {moment.timestamp_seconds:.1f}s")
                with right:
                    st.caption("Filled in from the source video. Upload an image below to replace it.")
            upload = right.file_uploader(f"Image for scene {index + 1}", type=["png", "jpg", "jpeg", "webp"],
                                         key=_key(st, f"img_{index}"), label_visibility="collapsed")
            if upload is not None:
                _save_scene_image(st, index, upload.name, upload.getvalue())
                images = _images(st)
            if index in images:
                left.image(str(images[index]), caption="Your image")
            elif scene.visual_type != "real_frame":
                right.warning("Waiting for this scene's image.")
    missing = [i + 1 for i, s in enumerate(plan.scenes) if s.visual_type != "real_frame" and i not in images]
    back, forward = st.columns(2)
    if back.button("Back to prompts", key="ss_back_prompts"):
        state["stage"] = "prompts"
        st.rerun()
    if forward.button("Next: voice-over", type="primary", disabled=bool(missing), key="ss_to_voice",
                      help=f"Missing images for scene(s) {missing}" if missing else None):
        state["stage"] = "voice"
        st.rerun()


def _page_voice(st) -> None:
    state, plan, analysis = _state(st), _plan(st), _analysis(st)
    try:
        language_options = voiceover.languages()
    except voiceover.VoiceoverError as exc:
        st.error(str(exc))
        language_options = []
    codes = [code for code, _ in language_options]
    source_code = (analysis.language_code or "en").lower().split("-")[0]
    names = dict(language_options)
    left, right = st.columns(2)
    code = left.selectbox("Voice-over language", codes or [source_code],
                          index=codes.index(source_code) if source_code in codes else 0,
                          format_func=lambda c: names.get(c, c), key="ss_lang")
    gender = right.radio("Voice", ["Female", "Male"], horizontal=True, key="ss_gender")
    voices = voiceover.voices_for(code, gender) if codes else []
    voice = st.selectbox("Speaker", voices, format_func=lambda v: v.label, key="ss_voice") if voices else None
    if codes and not voices:
        st.warning(f"No {gender.lower()} voice for {names.get(code, code)} -- pick the other voice.")
    if code != source_code:
        st.caption(f"The script is in {analysis.spoken_language}; it will be translated to {names.get(code, code)} "
                   "(one Gemini call).")
    narrations = [scene.narration for scene in plan.scenes]
    duration = state["duration"]

    if st.button("Generate voice-over", type="primary", disabled=voice is None, key="ss_make_voice"):
        language = names.get(code, code)
        with st.status("Recording the voice-over...", expanded=True) as status:
            notify = lambda message: status.update(label=message)  # noqa: E731
            try:
                script = narrations if code == source_code else translate_script(narrations, language, notify)
                fitted = voiceover.fit_to_duration(
                    script, voice.short_name, duration, notify=notify,
                    shorten=lambda lines, ratio: shorten_script(lines, ratio, language, notify),
                )
            except (*USER_FACING_ERRORS, voiceover.VoiceoverError) as exc:
                status.update(label="Voice-over failed", state="error")
                st.error(str(exc))
            else:
                state["voice"] = _voice_state(fitted, voice.short_name, ".mp3")
                status.update(label="Voice-over ready", state="complete", expanded=False)

    with st.expander("Use my own voice-over instead"):
        own = st.file_uploader("Voice-over audio", type=["mp3", "wav", "m4a"], key="ss_own_voice")
        if own is not None and st.button("Use this audio", key="ss_use_own"):
            try:
                fitted = voiceover.from_uploaded_audio(own.getvalue(), Path(own.name).suffix.lower(), narrations, duration)
            except voiceover.VoiceoverError as exc:
                st.error(str(exc))
            else:
                state["voice"] = _voice_state(fitted, "your upload", Path(own.name).suffix.lower())

    take = state.get("voice")
    if take:
        st.audio(take["audio"], format={".wav": "audio/wav", ".m4a": "audio/mp4"}.get(take["suffix"], "audio/mpeg"))
        for note in take["notes"]:
            st.caption(note)
        st.table([{"Scene": i + 1, "Starts": f"{start:.1f} s", "Lasts": f"{length:.1f} s", "Voice-over": text}
                  for i, (start, length, text) in enumerate(zip(take["starts"], take["durations"], take["scenes"]))])
    back, forward = st.columns(2)
    if back.button("Back to images", key="ss_back_images"):
        state["stage"] = "images"
        st.rerun()
    if forward.button("Next: combine", type="primary", disabled=not take, key="ss_to_done"):
        state["stage"] = "done"
        st.rerun()


def _voice_state(fitted: voiceover.FittedVoiceover, voice_name: str, suffix: str) -> dict:
    return {
        "audio": fitted.take.audio, "suffix": suffix, "voice": voice_name, "scenes": fitted.scenes,
        "starts": fitted.scene_starts, "durations": fitted.scene_durations,
        "video_duration": fitted.video_duration, "notes": fitted.notes,
    }


def _page_done(st) -> None:
    state, plan, analysis = _state(st), _plan(st), _analysis(st)
    take = state["voice"]
    if not state.get("video") and st.button("Combine images and voice-over", type="primary", key="ss_render"):
        job = Path(state["job"])
        voice_path = job / f"voice{take['suffix']}"
        voice_path.write_bytes(take["audio"])
        bar = st.progress(0.0, text="Rendering...")
        try:
            canvases = build_canvases(plan, analysis, _frames(st), _images(st))
            output = slideshow.render_slideshow(
                canvases, take["durations"], job / "short.mp4", voice_path,
                progress=lambda done: bar.progress(min(done, 1.0), text=f"Rendering... {done * 100:.0f}%"),
            )
        except (slideshow.RenderError, OSError, ValueError) as exc:
            st.error(f"Rendering failed: {exc}")
            return
        state["video"] = output.read_bytes()
        output.unlink(missing_ok=True)
        st.rerun()
    if state.get("video"):
        st.subheader(plan.hook_title)
        st.video(state["video"], format="video/mp4")
        st.caption(f"1080x1920 · {take['video_duration']:.1f} s · voice: {take['voice']}")
        stem = re.sub(r"[^\w-]+", "_", plan.hook_title).strip("_")[:50] or "short"
        st.download_button("Download MP4", state["video"], file_name=f"{stem}_9x16.mp4", mime="video/mp4",
                           key="ss_download")
        credit = f"Summary of \"{state['title']}\" -- original: {state['source_url']}"
        st.download_button("Download script + credit", f"{credit}\n\n" + "\n".join(take["scenes"]),
                           file_name=f"{stem}_script.txt", mime="text/plain", key="ss_script_dl")
        st.caption(f"Credit the creator when you post: {credit}")
    back, again = st.columns(2)
    if back.button("Back to voice-over", key="ss_back_voice"):
        state.pop("video", None)
        state["stage"] = "voice"
        st.rerun()
    if again.button("Start a new Short", key="ss_restart_done"):
        _start_over(st)
        st.rerun()
