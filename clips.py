"""Clips from a long video: Gemini watches the whole video once and
recommends the moments worth posting on their own as YouTube Shorts /
Instagram Reels; ffmpeg cuts the picture from the real footage. The
original sound (voice and music) is never used: every clip gets a new
narration written by Gemini in its own words, read by Edge TTS, with
optional background music the user uploads.

Flow (one Streamlit page, three stages):
  1. input   link or upload (max CLIPS_MAX_VIDEO_SECONDS, 30 min) -> speech
             pauses and shot cuts measured by ffmpeg -> one Gemini call
             returns the recommended clips (count, times, hook, score,
             narration); clips that just chop the video into back-to-back
             pieces are sent back once
  2. pick    the clips on a timeline, start/end snapped to pauses or cuts;
             tick, nudge, edit the narration, pick the voice and music
  3. done    each clip rendered 1080x1920 (blurred fill for landscape
             footage), new voice-over at -14 LUFS, with its own YouTube +
             Instagram metadata
"""
from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import uuid
import zipfile
from dataclasses import asdict
from pathlib import Path
from urllib.parse import urlparse

from pydantic import BaseModel, Field

import voiceover
from recreate_video import (
    CONFIG,
    USER_FACING_ERRORS,
    GeminiCallError,
    MediaInfo,
    Notify,
    URLVideoError,
    _noop_notify,
    _parse_structured,
    _upload_and_wait,
    call_gemini,
    detect_scene_cuts,
    download_video,
    ffmpeg_exe,
    gemini_http_options,
    generate_metadata,
    get_client,
    probe_video,
    render_social_or_generate,
)
from summary_short import WORDS_PER_SECOND, words_per_second

logger = logging.getLogger(__name__)
_LOADED_MTIME = Path(__file__).stat().st_mtime  # see recreate_video.drop_stale_app_modules

CLIPS_MAX_VIDEO_SECONDS = int(os.getenv("CLIPS_MAX_VIDEO_SECONDS", "1800") or 1800)
CLIPS_MAX_DOWNLOAD_MB = int(os.getenv("CLIPS_MAX_DOWNLOAD_MB", "1500") or 1500)
TOO_LONG = (f"This video cannot be analysed or made into clips due to its duration. The maximum is "
            f"{CLIPS_MAX_VIDEO_SECONDS / 60:.0f} minutes.")
CLIP_MIN_SECONDS, CLIP_MAX_SECONDS = 15, 60
RECOMMEND_SCORE = 70
MAX_CLIPS = 8
SNAP_SECONDS = 2.0  # how far a start / end may move to reach a pause or cut
CLIPS_TIMEOUT_SECONDS = 600  # one call over up to 30 minutes of video
YOUTUBE_HOSTS = ("youtube.com", "youtu.be")
CHOP_GAP_SECONDS = 5.0  # clips closer than this count as "next to each other"
CHOP_COVERAGE = 0.6  # back-to-back clips covering more than this share = chopped
SHORT_SOURCE_SECONDS = 180  # below this, 0-1 clip is normal
MUSIC_VOLUME = 0.12  # background music under the voice
NARRATION_SHARE = 0.85  # narration fills ~85% of the clip, leaving breathing room
CHOPPED_FIX = (
    "Your previous answer split the video into consecutive back-to-back pieces. That is wrong: do not cover "
    "the video piece by piece. Return only the separate moments that each have a real hook and payoff of "
    "their own, with gaps between them; fewer clips (even 0 or 1) is fine."
)


# ============================================================================
# What Gemini returns
# ============================================================================

class ClipIdea(BaseModel):
    start_seconds: float
    end_seconds: float
    title: str = Field(description="Short working title of the clip, in English.")
    hook: str = Field(description="The exact moment that grabs attention in the clip's first 2 seconds: quote "
                                  "the words said there (in English meaning) or describe exactly what is seen.")
    payoff: str = Field(description="The exact moment the clip ends on: the answer, punchline, result or "
                                    "conclusion, quoted or described.")
    why_it_works: str = Field(description="Why it works as a standalone Short, in one sentence.")
    what_is_said: str = Field(description="English meaning of what is said in the clip, 1-3 sentences; '' if nothing.")
    score: int = Field(description="0-100: how likely it is to hold viewers as a standalone Short. Strict.")
    stands_alone: bool = Field(description="True if it makes sense without the rest of the video.")
    narration: str = Field(
        description="A NEW voice-over for the clip in your own words, in the video's spoken language and its own "
                    "script (English if nobody speaks), sized to the clip's length. It replaces the original "
                    "sound completely.")


class ClipPlan(BaseModel):
    video_summary: str = Field(description="2-3 sentence English summary of the whole video.")
    spoken_language: str = Field(description="Main spoken language as an English name, or 'none'.")
    language_code: str = Field(description="ISO 639-1 code of the spoken language, e.g. 'te', 'en'; 'en' if none.")
    recommended_count: int = Field(description="How many clips scored 70 or more and are worth posting.")
    count_reason: str = Field(description="One or two sentences: why that many clips (not more, not fewer).")
    clips: list[ClipIdea] = Field(description="Up to 12 candidates, best first, without overlaps.")


# ============================================================================
# Measuring: speech pauses and shot cuts (ffmpeg, no Gemini)
# ============================================================================

def speech_pauses(video: Path) -> list[tuple[float, float]]:
    """(start, end) of every pause of 0.35 s or more -- safe places to cut."""
    try:
        output = subprocess.run(
            [ffmpeg_exe(), "-hide_banner", "-nostdin", "-i", str(video), "-vn",
             "-af", "silencedetect=noise=-32dB:d=0.35", "-f", "null", "-"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600,
        ).stderr
    except subprocess.TimeoutExpired:
        logger.warning("[CLIPS] pause detection timed out")
        return []
    starts = [float(v) for v in re.findall(r"silence_start:\s*(-?\d+(?:\.\d+)?)", output)]
    ends = [float(v) for v in re.findall(r"silence_end:\s*(\d+(?:\.\d+)?)", output)]
    return [(max(0.0, a), b) for a, b in zip(starts, ends)]


def snap(value: float, points: list[float], window: float = SNAP_SECONDS) -> float:
    """The nearest point within `window` seconds, or `value` itself."""
    near = [p for p in points if abs(p - value) <= window]
    return min(near, key=lambda p: abs(p - value)) if near else value


def looks_chopped(spans: list[tuple[float, float]], duration: float) -> bool:
    """True when the clips just cover the video piece by piece: they cover
    most of it and most neighbours touch -- not separate moments."""
    spans = sorted(spans)
    if len(spans) < 2 or duration <= 0:
        return False
    covered = sum(max(0.0, end - start) for start, end in spans)
    touching = sum(1 for (_, end), (start, _) in zip(spans, spans[1:]) if start - end < CHOP_GAP_SECONDS)
    return covered > CHOP_COVERAGE * duration and touching * 2 >= len(spans) - 1


def finalise_clips(plan: ClipPlan, duration: float, pauses: list[tuple[float, float]],
                   cuts: list[float]) -> list[dict]:
    """Gemini's clips with start / end moved onto a pause or cut, 15-60 s,
    no overlaps, best first for overlap decisions, then in video order. If
    they still chop the video into back-to-back pieces, only the best ones
    with a gap between them are kept."""
    kept = _pick_clips(plan, duration, pauses, cuts, gap=-2.0)
    if looks_chopped([(c["start"], c["end"]) for c in kept], duration):
        logger.info("[CLIPS] clips are back-to-back -- keeping only separate moments")
        kept = _pick_clips(plan, duration, pauses, cuts, gap=CHOP_GAP_SECONDS)
    kept = kept[:12]  # already best first
    good = [c for c in kept if c["score"] >= RECOMMEND_SCORE and c["stands_alone"]][:MAX_CLIPS]
    for clip in kept:
        clip["recommended"] = any(clip is g for g in good)
    return sorted(kept, key=lambda c: c["start"])


def _pick_clips(plan: ClipPlan, duration: float, pauses: list[tuple[float, float]], cuts: list[float],
                gap: float) -> list[dict]:
    """Best first; a clip is skipped when it comes closer than `gap` seconds
    to one already kept (a negative gap allows that much overlap)."""
    start_points = sorted({round(max(0.0, end - 0.1), 2) for _, end in pauses} | set(cuts) | {0.0})
    end_points = sorted({round(start + 0.15, 2) for start, _ in pauses} | set(cuts) | {round(duration, 2)})
    kept: list[dict] = []
    for idea in sorted(plan.clips, key=lambda c: c.score, reverse=True):
        start = max(0.0, min(idea.start_seconds, duration))
        end = max(0.0, min(idea.end_seconds, duration))
        start, end = snap(start, start_points), snap(end, end_points)
        if end - start > CLIP_MAX_SECONDS:  # end on the last pause before the limit
            earlier = [p for p in end_points if start + CLIP_MIN_SECONDS <= p <= start + CLIP_MAX_SECONDS]
            end = earlier[-1] if earlier else start + CLIP_MAX_SECONDS
        if end - start < CLIP_MIN_SECONDS - 1:
            logger.info(f"[CLIPS] dropped '{idea.title}': {end - start:.1f}s after snapping")
            continue
        if any(start < k["end"] + gap and end > k["start"] - gap for k in kept):
            continue
        kept.append({**idea.model_dump(), "start": round(start, 2), "end": round(end, 2)})
    return kept


# ============================================================================
# Gemini: one call over the whole video
# ============================================================================

def frames_per_second(duration: float) -> float:
    """Fewer frames for longer videos so 30 minutes stays within the free
    per-minute token limit (finding clips leans on the speech anyway)."""
    if duration <= 600:
        return 1.0
    if duration <= 1200:
        return 0.5
    return 0.25


def _clip_prompt(media: MediaInfo, pauses: list[tuple[float, float]], cuts: list[float],
                 correction: str = "") -> str:
    pause_times = ", ".join(f"{(a + b) / 2:.1f}" for a, b in pauses[:400]) or "none measured"
    cut_times = ", ".join(f"{c:.1f}" for c in cuts[:200]) or "none measured"
    rates = ", ".join(f"{code} {rate:g}" for code, rate in WORDS_PER_SECOND.items())
    short = (f"- This video is short ({media.duration_seconds:.0f} s): 0 or 1 clip is normal.\n"
             if media.duration_seconds < SHORT_SOURCE_SECONDS else "")
    return (
        f"You are a short-form video editor. Watch and listen to this whole {media.duration_seconds:.0f}-second "
        "video. Find the moments worth posting on their own as YouTube Shorts / Instagram Reels, each cut "
        f"straight from this footage as ONE continuous piece of {CLIP_MIN_SECONDS}-{CLIP_MAX_SECONDS} seconds.\n\n"
        "Rules:\n"
        "- A clip starts on a hook (a striking line, question, claim or visual in its first 2 seconds) and ends "
        "on a payoff (the answer, punchline, result or conclusion). Never start or end mid-sentence.\n"
        "- It must make sense to someone who never saw the rest of the video: no 'as I said earlier', no "
        "unexplained references.\n"
        "- Score each clip 0-100 for how well it would hold viewers as a standalone Short. Be strict: 70+ only "
        "for clips you would really post.\n"
        "- NEVER split the video into consecutive pieces (0-30 s, 30-60 s, ...). Most of a video is not worth "
        "clipping; gaps between clips are normal. Each clip is a separate moment with its OWN hook and payoff.\n"
        "- The length follows the moment: as short as the moment allows, not as long as the limit.\n"
        "- recommended_count = the number of clips scoring 70+. It follows the content, not the length: roughly "
        "one per 2-4 minutes of strong content, and 0 if nothing stands on its own. Return up to 12 "
        "candidates, best first, including a few 50-69 alternatives. Clips must not overlap.\n"
        + short +
        f"- Times are seconds from the start, between 0 and {media.duration_seconds:.0f}. Start and end at "
        f"speech pauses or shot changes. Speech pauses (seconds): {pause_times}. Shot changes: {cut_times}.\n"
        "- narration: the original sound (voice and music) is removed from the clip, so write a NEW voice-over "
        "in your own words that tells what happens in the clip: open on the hook, end on the payoff. Never "
        "repeat, translate or closely paraphrase what is said in the video, and quote no song lyrics. Write it "
        "in the spoken language and its own script (English if nobody speaks). Size it to the clip: about "
        f"{NARRATION_SHARE:.0%} of (clip seconds x words per second); words per second by language: {rates}, "
        "others 2.0. Plain spoken sentences, no hashtags or emojis, no 'like and subscribe'.\n"
        "- Never name real people; describe them by appearance or role. Ignore platform UI and watermarks.\n"
        + (f"\nIMPORTANT: {correction}\n" if correction else "")
        + "Return JSON matching the schema."
    )


def youtube_watch_url(url: str) -> str:
    """https://www.youtube.com/watch?v=ID for any YouTube link (watch,
    youtu.be, shorts, live), the form Gemini opens; '' if not YouTube."""
    host = (urlparse(url).hostname or "").lower()
    if not any(host == h or host.endswith("." + h) for h in YOUTUBE_HOSTS):
        return ""
    match = re.search(r"(?:v=|youtu\.be/|/shorts/|/live/|/embed/)([A-Za-z0-9_-]{11})", url)
    return f"https://www.youtube.com/watch?v={match.group(1)}" if match else ""


def find_clips(video: Path, source_url: str, media: MediaInfo, pauses: list[tuple[float, float]],
               cuts: list[float], notify: Notify = _noop_notify) -> tuple[ClipPlan, str]:
    """(plan, model). YouTube links are watched by Gemini straight from
    YouTube (free, no upload); anything else is uploaded and deleted after."""
    from google.genai import types as genai_types

    client = get_client()
    metadata = genai_types.VideoMetadata(fps=frames_per_second(media.duration_seconds))
    used: dict[str, str] = {}

    def ask(part):
        plan = ask_once(part, _clip_prompt(media, pauses, cuts))
        if looks_chopped([(c.start_seconds, c.end_seconds) for c in plan.clips], media.duration_seconds):
            logger.info("[CLIPS] Gemini chopped the video into back-to-back pieces -- asking once more")
            notify("Gemini split the video into back-to-back pieces -- asking again for real moments...")
            plan = ask_once(part, _clip_prompt(media, pauses, cuts, CHOPPED_FIX))
        return plan

    def ask_once(part, prompt: str) -> ClipPlan:
        def run(model: str) -> ClipPlan:
            response = client.models.generate_content(
                model=model,
                contents=[part, prompt],
                config=genai_types.GenerateContentConfig(
                    http_options=gemini_http_options(CLIPS_TIMEOUT_SECONDS),
                    response_mime_type="application/json",
                    response_schema=ClipPlan,
                    media_resolution=genai_types.MediaResolution.MEDIA_RESOLUTION_LOW,
                    temperature=0.4,
                    automatic_function_calling=genai_types.AutomaticFunctionCallingConfig(disable=True),
                ),
            )
            used["model"] = model
            return _parse_structured(response, ClipPlan)

        notify("Gemini is watching the whole video and picking the clips...")
        return call_gemini(run, notify, task="analysis", label="clips")

    if youtube := youtube_watch_url(source_url):
        try:
            part = genai_types.Part(file_data=genai_types.FileData(file_uri=youtube), video_metadata=metadata)
            return ask(part), used["model"]
        except GeminiCallError as exc:  # e.g. Gemini cannot open this YouTube video: upload instead
            logger.warning(f"[CLIPS] YouTube link not usable by Gemini ({exc}) -- uploading the file")
    notify("Uploading the video to Gemini...")
    uploaded = _upload_and_wait(video, notify)
    try:
        part = genai_types.Part(
            file_data=genai_types.FileData(file_uri=uploaded.uri, mime_type=uploaded.mime_type),
            video_metadata=metadata)
        return ask(part), used["model"]
    finally:
        try:
            client.files.delete(name=uploaded.name)
        except Exception as exc:
            logger.warning(f"[CLIPS] could not delete uploaded file {uploaded.name}: {exc}")


# ============================================================================
# Rendering (ffmpeg)
# ============================================================================

class ClipRenderError(RuntimeError):
    pass


def render_clip(source: Path, media: MediaInfo, start: float, end: float, output: Path,
                voice: Path, music: Path | None = None) -> Path:
    """One 1080x1920 clip: vertical footage is filled, landscape footage sits
    on a blurred copy of itself. The source's own sound is never used: the
    audio is the new voice-over, with the music (looped, quiet) under it if
    given, normalised to -14 LUFS."""
    length = max(1.0, end - start)
    inputs = ["-ss", f"{start:.3f}", "-t", f"{length:.3f}", "-i", str(source), "-i", str(voice)]
    if music is not None:
        inputs += ["-stream_loop", "-1", "-i", str(music)]
    if media.width / max(1, media.height) <= 0.6:
        video = "[0:v]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,setsar=1,fps=30[v]"
    else:
        video = ("[0:v]split=2[bg0][fg0];"
                 "[bg0]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,boxblur=24:2[bg];"
                 "[fg0]scale=1080:-2[fg];[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1,fps=30[v]")
    loudness = "loudnorm=I=-14:TP=-1.5:LRA=11,aresample=48000"
    voice_chain = "[1:a]aresample=48000,aformat=channel_layouts=stereo,apad"
    if music is not None:
        audio = (f"{voice_chain}[vo];[2:a]aresample=48000,aformat=channel_layouts=stereo,"
                 f"volume={MUSIC_VOLUME}[bg];[vo][bg]amix=inputs=2:duration=first:normalize=0,{loudness}[a]")
    else:
        audio = f"{voice_chain},{loudness}[a]"
    command = [ffmpeg_exe(), "-y", "-hide_banner", "-loglevel", "error", *inputs,
               "-filter_complex", f"{video};{audio}", "-map", "[v]", "-map", "[a]",
               "-c:v", "libx264", "-preset", "veryfast", "-crf", "21", "-pix_fmt", "yuv420p",
               "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", "-t", f"{length:.3f}", str(output)]
    try:
        result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace",
                                timeout=900)
    except subprocess.TimeoutExpired as exc:
        raise ClipRenderError("Rendering the clip took more than 15 minutes.") from exc
    if result.returncode != 0 or not output.is_file():
        raise ClipRenderError(f"ffmpeg could not render the clip: {result.stderr.strip()[-400:]}")
    return output


def metadata_context(clip: dict, plan: ClipPlan) -> str:
    return "\n".join(part for part in [
        f"Content: a {clip['end'] - clip['start']:.0f}-second Short cut from a longer video.",
        f"Clip title idea: {clip['title']}",
        f"Hook: {clip['hook']}",
        f"Payoff: {clip['payoff']}",
        f"Why it works: {clip['why_it_works']}",
        f"What is said (English meaning): {clip['what_is_said']}" if clip.get("what_is_said") else "",
        f"New voice-over of the clip: {clip['narration']}" if clip.get("narration") else "",
        f"The full video: {plan.video_summary.strip()[:400]}",
        f"Spoken language: {plan.spoken_language}" if plan.spoken_language.lower() != "none" else "",
    ] if part)


def _clock(seconds: float) -> str:
    minutes, rest = divmod(max(0.0, seconds), 60)
    return f"{int(minutes)}:{rest:04.1f}"


# ============================================================================
# Page
# ============================================================================

def _state(st) -> dict:
    return st.session_state.setdefault("clips", {"stage": "input"})


def _start_over(st) -> None:
    job = _state(st).get("job")
    if job:
        shutil.rmtree(job, ignore_errors=True)
    st.session_state["clips"] = {"stage": "input"}


def _key(st, name: str) -> str:
    return f"clips_{name}_{Path(_state(st)['job']).name}"


def render_page(st) -> None:
    st.title("Clips from a long video")
    state = _state(st)
    steps = ["1 · Video", "2 · Pick clips", "3 · Clips + metadata"]
    position = ["input", "pick", "done"].index(state["stage"])
    st.caption("  →  ".join(f"**{s}**" if i == position else s for i, s in enumerate(steps)))
    {"input": _page_input, "pick": _page_pick, "done": _page_done}[state["stage"]](st)


def _page_input(st) -> None:
    limit_min = CLIPS_MAX_VIDEO_SECONDS / 60
    st.caption(
        f"Paste a YouTube / Instagram link or upload a video (up to {limit_min:.0f} minutes). Gemini watches it "
        "once and recommends the moments worth posting as Shorts / Reels; the picture is cut from the real "
        "footage with ffmpeg. The original sound is never used: each clip gets a new voice-over written by "
        "Gemini, with optional music you upload."
    )
    st.info("A new voice-over removes the original voice and music, but the picture is still the creator's "
            "footage, and YouTube / Instagram also match video. Post clips only of videos you own or have the "
            "rights to. For someone else's video, Summarise to Short (AI images) is the safer choice.")
    for problem in (problems := CONFIG.problems()):
        st.error(problem)
    url = st.text_input("Video link", placeholder="https://www.youtube.com/watch?v=...", key="clips_url")
    uploaded = st.file_uploader("Or upload a video", type=["mp4", "mov", "webm", "m4v", "mkv"], key="clips_upload")
    if not st.button("Find clips", type="primary", disabled=bool(problems) or not (url.strip() or uploaded),
                     key="clips_find"):
        return
    job = CONFIG.work_dir / "clips" / uuid.uuid4().hex[:12]
    job.mkdir(parents=True, exist_ok=True)
    with st.status("Starting...", expanded=True) as status:
        notify = lambda message: status.update(label=message)  # noqa: E731
        try:
            if uploaded is not None:
                video = job / f"source{Path(uploaded.name).suffix.lower() or '.mp4'}"
                video.write_bytes(uploaded.getvalue())
                title, source_url = Path(uploaded.name).stem, f"upload://{uploaded.name}"
            else:
                notify("Downloading the video...")
                downloaded = download_video(
                    url.strip(), job / "download", notify, max_seconds=CLIPS_MAX_VIDEO_SECONDS, max_height=1080,
                    max_mb=CLIPS_MAX_DOWNLOAD_MB, too_long_message=TOO_LONG,
                )
                video, title, source_url = downloaded.path, downloaded.title, downloaded.source_url
            notify("Checking the video...")
            media = probe_video(video)
            if media.duration_seconds > CLIPS_MAX_VIDEO_SECONDS + 0.5:
                raise URLVideoError(TOO_LONG)
            if media.duration_seconds < CLIP_MIN_SECONDS * 2:
                raise URLVideoError(f"This video is only {media.duration_seconds:.0f} seconds long -- too short to "
                                    "cut into clips. Use Video recreate instead.")
            notify("Finding the speech pauses...")
            pauses = speech_pauses(video) if media.has_audio else []
            notify("Measuring the shot changes...")
            cuts = detect_scene_cuts(video)
            plan, model = find_clips(video, source_url, media, pauses, cuts, notify)
            clips = finalise_clips(plan, media.duration_seconds, pauses, cuts)
        except (*USER_FACING_ERRORS, ClipRenderError) as exc:
            status.update(label="Could not find clips", state="error")
            st.error(str(exc))
            shutil.rmtree(job, ignore_errors=True)
            return
        status.update(label="Clips found", state="complete", expanded=False)
    logger.info(f"[CLIPS] {len(clips)} clips ({sum(c['recommended'] for c in clips)} recommended) by {model}")
    st.session_state["clips"] = {
        "stage": "pick", "job": str(job), "video": str(video), "title": title, "source_url": source_url,
        "media": asdict(media), "plan": plan.model_dump(), "clips": clips, "model": model,
        "rendered": {}, "social": {},
    }
    st.rerun()


def _page_pick(st) -> None:
    state = _state(st)
    plan = ClipPlan.model_validate(state["plan"])
    media = MediaInfo(**state["media"])
    clips = state["clips"]
    recommended = sum(c["recommended"] for c in clips)
    st.subheader(state["title"])
    st.caption(f"{_clock(media.duration_seconds)} · {state['source_url']} · analysed by {state['model']}")
    st.markdown(f"**Gemini recommends {recommended} clip(s)**"
                + (f" ({len(clips) - recommended} more possible)" if len(clips) > recommended else "")
                + f" — {plan.count_reason}")
    if not clips:
        st.warning("Gemini found no moment that works on its own as a Short. Try another video.")
    elif media.duration_seconds < SHORT_SOURCE_SECONDS:
        st.caption("Short video: 0-1 clip is normal. To retell the whole video as a Short, use Summarise to Short.")
    if clips:
        st.markdown(timeline_html(clips, media.duration_seconds), unsafe_allow_html=True)
    language = narration_language(plan)
    st.markdown("**New voice-over** (the original sound is not used)")
    voice = _voice_picker(st, language)
    music = st.file_uploader("Background music (optional, quiet under the voice) -- only music you have rights "
                             "to, e.g. from the YouTube Audio Library", type=["mp3", "wav", "m4a", "aac", "ogg"],
                             key=_key(st, "music"))

    chosen = []
    for index, clip in enumerate(clips):
        with st.container(border=True):
            top = st.columns([3, 1])
            take = top[0].checkbox(f"**{clip['title']}** · score {clip['score']}" + (" · ⭐ recommended"
                                   if clip["recommended"] else ""), value=clip["recommended"],
                                   key=_key(st, f"take_{index}"))
            top[1].caption(f"{_clock(clip['start'])} → {_clock(clip['end'])} ({clip['end'] - clip['start']:.0f} s)")
            st.caption(f"**Hook:** {clip['hook']}  \n**Payoff:** {clip['payoff']}  \n**Why:** {clip['why_it_works']}")
            times = st.columns(2)
            start = times[0].number_input("Start (s)", 0.0, media.duration_seconds, float(clip["start"]), 0.5,
                                          key=_key(st, f"start_{index}"))
            end = times[1].number_input("End (s)", 0.0, media.duration_seconds, float(clip["end"]), 0.5,
                                        key=_key(st, f"end_{index}"))
            narration = st.text_area("Voice-over", clip["narration"], key=_key(st, f"narration_{index}"), height=90)
            budget = round(max(0.0, end - start) * words_per_second(language) * NARRATION_SHARE)
            st.caption(f"{len(narration.split())} words · about {budget} fit in {max(0.0, end - start):.0f} s "
                       "(the voice is sped up or slowed down a little to fit)")
            if take:
                chosen.append({**clip, "index": index, "start": start, "end": end, "narration": narration.strip()})

    bad = [c for c in chosen if not 3 <= c["end"] - c["start"] <= 180]
    for clip in bad:
        st.warning(f"'{clip['title']}': end must be 3-180 s after start.")
    silent = [c for c in chosen if not c["narration"]]
    for clip in silent:
        st.warning(f"'{clip['title']}': write a voice-over -- the original sound is not used.")
    left, right = st.columns(2)
    if right.button("Start over", key=_key(st, "restart")):
        _start_over(st)
        st.rerun()
    if not left.button(f"Render {len(chosen)} clip(s)", type="primary",
                       disabled=not chosen or bool(bad) or bool(silent) or voice is None, key=_key(st, "render")):
        return
    _render_chosen(st, chosen, media, voice, music)


def narration_language(plan: ClipPlan) -> str:
    """The language the narration is written in: the spoken one, English
    when nobody speaks."""
    if plan.spoken_language.strip().lower() in ("", "none"):
        return "en"
    return (plan.language_code or "en").lower().split("-")[0]


def timeline_html(clips: list[dict], duration: float) -> str:
    """A bar for the whole video with each clip marked where it sits, so
    back-to-back chopping is easy to spot."""
    marks = "".join(
        f'<div title="{_clock(c["start"])}-{_clock(c["end"])}" style="position:absolute;top:0;bottom:0;'
        f'left:{100 * c["start"] / duration:.2f}%;width:{max(0.4, 100 * (c["end"] - c["start"]) / duration):.2f}%;'
        f'background:{"#e8590c" if c["recommended"] else "#adb5bd"};border-radius:3px"></div>'
        for c in clips if duration > 0)
    return (f'<div style="position:relative;height:14px;background:rgba(128,128,128,.18);border-radius:3px;'
            f'margin:4px 0 2px">{marks}</div><div style="display:flex;justify-content:space-between;'
            f'font-size:12px;opacity:.7"><span>0:00</span><span>clips on the video (orange = recommended)</span>'
            f'<span>{_clock(duration)}</span></div>')


def _voice_picker(st, language_code: str):
    """The voice for the narration's language (male / female, speaker)."""
    try:
        names = dict(voiceover.languages())
    except voiceover.VoiceoverError as exc:
        st.error(str(exc))
        return None
    left, right = st.columns(2)
    left.caption(f"Language: {names.get(language_code, language_code)} (the video's spoken language)")
    gender = right.radio("Voice", ["Female", "Male"], horizontal=True, key="clips_gender")
    voices = voiceover.voices_for(language_code, gender)
    if not voices:
        st.warning(f"No {gender.lower()} voice for {names.get(language_code, language_code)} -- pick the other "
                   "voice.")
        return None
    return st.selectbox("Speaker", voices, format_func=lambda v: v.label, key="clips_voice")


def fit_clip_voice(clip: dict, voice: str, duration: float, notify: Notify = _noop_notify) -> tuple[bytes, dict]:
    """The narration read by Edge TTS and fitted to the clip. If it still
    runs over at the fastest natural speed, the clip ends a little later
    (never past the video's end). Returns (MP3, the clip with notes)."""
    length = clip["end"] - clip["start"]
    fitted = voiceover.fit_to_duration([clip["narration"]], voice, length, notify=notify)
    clip = {**clip, "notes": list(fitted.notes)}
    if fitted.take.duration > length + voiceover.OVERRUN_TOLERANCE_SECONDS:
        clip["end"] = round(min(duration, clip["start"] + fitted.take.duration + 0.3), 2)
        clip["notes"].append(f"The clip was lengthened to {clip['end'] - clip['start']:.0f} s to fit the voice-over.")
    return fitted.take.audio, clip


def _render_chosen(st, chosen: list[dict], media: MediaInfo, voice, music_upload=None) -> None:
    state = _state(st)
    job = Path(state["job"])
    music = None
    if music_upload is not None:
        music = job / f"music{Path(music_upload.name).suffix.lower() or '.mp3'}"
        music.write_bytes(music_upload.getvalue())
    rendered: dict[str, dict] = {}
    with st.status("Rendering the clips...", expanded=True) as status:
        for number, clip in enumerate(chosen, start=1):
            label = f"Clip {number} of {len(chosen)}: {clip['title']}"
            status.update(label=label)
            try:
                audio, clip = fit_clip_voice(clip, voice.short_name, media.duration_seconds,
                                             lambda message: status.update(label=f"{label} -- {message}"))
                voice_path = job / f"voice_{clip['index']}.mp3"
                voice_path.write_bytes(audio)
                output = job / f"clip_{number:02d}_{int(clip['start'])}s.mp4"
                status.update(label=f"{label} -- rendering")
                render_clip(Path(state["video"]), media, clip["start"], clip["end"], output, voice_path, music)
                rendered[str(clip["index"])] = {**clip, "path": str(output), "music": music is not None}
            except (ClipRenderError, voiceover.VoiceoverError) as exc:
                st.error(f"Clip {number} ({clip['title']}): {exc}")
        status.update(label=f"{len(rendered)} clip(s) ready", state="complete" if rendered else "error",
                      expanded=False)
    if rendered:
        archive_path = job / "clips.zip"
        with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_STORED) as archive:
            for clip in rendered.values():
                archive.write(clip["path"], Path(clip["path"]).name)
        state.update(stage="done", rendered=rendered, social={}, zip=str(archive_path))
        st.rerun()


def _page_done(st) -> None:
    state = _state(st)
    plan = ClipPlan.model_validate(state["plan"])
    rendered = state["rendered"]
    st.success(f"{len(rendered)} clip(s) ready. Each one has its own YouTube and Instagram metadata below.")
    left, mid, right = st.columns(3)
    archive = Path(state.get("zip", ""))
    if archive.name and archive.is_file():
        left.download_button("Download all clips (ZIP)", archive.read_bytes(), file_name="clips.zip",
                             mime="application/zip", key=_key(st, "zip"))
    if mid.button("Back to the clip list", key=_key(st, "back")):
        state["stage"] = "pick"
        st.rerun()
    if right.button("Start over", key=_key(st, "restart_done")):
        _start_over(st)
        st.rerun()

    language = plan.spoken_language if plan.spoken_language.lower() != "none" else "English"
    for number, (index, clip) in enumerate(rendered.items(), start=1):
        st.divider()
        st.subheader(f"Clip {number}: {clip['title']}")
        st.caption(f"{_clock(clip['start'])} → {_clock(clip['end'])} · {clip['end'] - clip['start']:.0f} s"
                   + (" · with music" if clip.get("music") else ""))
        st.caption(f"**Voice-over:** {clip['narration']}")
        for note in clip.get("notes", []):
            st.caption(note)
        path = Path(clip["path"])
        columns = st.columns([1, 2])
        if path.is_file():
            columns[0].video(str(path))
            columns[0].download_button("Download clip", path.read_bytes(), file_name=path.name, mime="video/mp4",
                                       key=_key(st, f"dl_{index}"))
        with columns[1]:
            stored = state["social"].get(index)
            social = None
            if stored:
                from recreate_video import SocialMetadata

                social = SocialMetadata.model_validate(stored)

            def make(notify: Notify, clip=clip, index=index):
                meta = generate_metadata(metadata_context(clip, plan), "video", language, notify)
                state["social"][index] = meta.model_dump()
                return meta

            render_social_or_generate(st, social, f"clip_{Path(state['job']).name}_{index}", make)
