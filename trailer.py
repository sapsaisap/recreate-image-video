"""Trailer / Teaser: clips from a newly released movie trailer or teaser,
posted with the ORIGINAL sound (dialogue and music) -- unlike Clips, which
replaces it with a new voice-over. Gemini watches the trailer once,
recommends the moments worth posting on their own (title reveal, hero
intro, punch dialogue, action peak, song hook) and reads the title card and
credits; Google Search then identifies the movie (title, cast, crew,
release date, platform), and every clip's YouTube / Instagram metadata is
written from those checked facts.

Flow (one Streamlit page, three stages):
  1. input   link or upload (max TRAILER_MAX_VIDEO_SECONDS) -> speech pauses
             and shot cuts (ffmpeg) -> one Gemini call for the clips and the
             on-screen clues -> one Google-Search-grounded call for the movie
  2. pick    the movie facts (editable) and the clips on a timeline; tick,
             nudge; optionally the whole trailer as one vertical video
  3. done    each clip 1080x1920 with its original sound (short fades,
             -14 LUFS) and its own metadata
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import uuid
import zipfile
from dataclasses import asdict
from pathlib import Path

from pydantic import BaseModel, Field

from clips import (
    CHOP_GAP_SECONDS,
    CLIP_MAX_SECONDS,
    CLIP_MIN_SECONDS,
    MAX_ALTERNATIVES,
    RECOMMEND_SCORE,
    TINY_SOURCE_SECONDS,
    UPLOAD_LIMIT_MB,
    ClipRenderError,
    _clock,
    _pick_clips,
    coverage_budget,
    find_clips,
    looks_chopped,
    speech_pauses,
    timeline_html,
    video_filter,
)
from recreate_video import (
    CONFIG,
    METADATA_THINKING_BUDGET,
    USER_FACING_ERRORS,
    MediaInfo,
    Notify,
    SocialMetadata,
    URLVideoError,
    _noop_notify,
    content_texts,
    detect_scene_cuts,
    download_video,
    ffmpeg_exe,
    generate_metadata,
    generate_with_optional_search,
    grounding_label,
    probe_video,
    render_social_or_generate,
)

logger = logging.getLogger(__name__)
_LOADED_MTIME = Path(__file__).stat().st_mtime  # see recreate_video.drop_stale_app_modules

TRAILER_MAX_VIDEO_SECONDS = int(os.getenv("TRAILER_MAX_VIDEO_SECONDS", "600") or 600)
TRAILER_MAX_DOWNLOAD_MB = int(os.getenv("TRAILER_MAX_DOWNLOAD_MB", "600") or 600)
TOO_LONG = (f"This video is too long for Trailer / Teaser. The maximum is {TRAILER_MAX_VIDEO_SECONDS / 60:.0f} "
            "minutes -- for a longer video use Clips from long video.")
FADE_IN_SECONDS = 0.15  # just enough to avoid a click where the clip starts
FADE_OUT_SECONDS = 0.6  # the music / dialogue tails off instead of stopping dead
RIGHTS_NOTE = (
    "The clips keep the trailer's own dialogue and music, so YouTube / Instagram will recognise them: expect a "
    "copyright claim (the film owns the trailer and may take the ad money or, rarely, block it). Studios usually "
    "welcome trailer clips that credit the movie, and the description links the official trailer."
)
MOVIE_FIELDS = [  # (key, label) shown and edited on the pick step, in this order
    ("title", "Movie title"), ("native_title", "Title in its own script"), ("release_date", "Release date"),
    ("platform", "Releasing in / on"), ("languages", "Languages"), ("genre", "Genre"),
    ("director", "Director"), ("music_director", "Music"), ("cast", "Cast"), ("production", "Production"),
    ("official_hashtags", "Official hashtags / handles"),
]


# ============================================================================
# What Gemini returns
# ============================================================================

class TrailerClipIdea(BaseModel):
    start_seconds: float
    end_seconds: float
    title: str = Field(description="Short working title of the clip, in English.")
    moment: str = Field(description="The kind of moment: title reveal, hero intro, punch dialogue, action, emotion, "
                                    "song hook, comedy, villain, climax tease or other.")
    hook: str = Field(description="What grabs attention in the clip's first 2 seconds: quote the line or describe "
                                  "exactly what is seen and heard.")
    payoff: str = Field(description="What the clip ends on: the punchline, reveal, beat drop or title card.")
    why_it_works: str = Field(description="Why it works as a standalone Short, in one sentence.")
    what_is_said: str = Field(description="The dialogue in the clip in its original language and script, with the "
                                          "English meaning in brackets; '' if nothing is said.")
    score: int = Field(description="0-100: how likely it is to hold viewers as a standalone Short. Strict.")
    stands_alone: bool = Field(description="True if it makes sense without the rest of the trailer.")
    on_screen_text: str = Field(description="Text shown on screen during the clip (title, credits, release date), "
                                            "word for word; no watermarks or @handles. '' if none.")


class TrailerPlan(BaseModel):
    video_summary: str = Field(description="2-3 sentence English summary of what the trailer shows.")
    video_kind: str = Field(description="trailer, teaser, glimpse, song promo, promo or other.")
    spoken_language: str = Field(description="Main spoken language as an English name, or 'none'.")
    language_code: str = Field(description="ISO 639-1 code of the spoken language, e.g. 'te', 'hi'; 'en' if none.")
    title_on_screen: str = Field(
        description="The movie's title exactly as shown on its title card, in its own script, plus the Latin "
                    "spelling if that is shown too; '' if there is no title card.")
    credits_on_screen: str = Field(
        description="Every name and line on the credit / end cards, word for word: production house, director, "
                    "music, cast names, release date, platform or 'in cinemas'. '' if none.")
    source_music: str = Field(default="", description="The music: song title and singers if a song with lyrics, "
                                                      "else its genre and mood; 'none' if there is no music.")
    recommended_count: int = Field(description="How many clips scored 70 or more and are worth posting.")
    count_reason: str = Field(description="One or two sentences: why that many clips.")
    clips: list[TrailerClipIdea] = Field(description="Up to 10 candidates, best first, without overlaps.")


class MovieFacts(BaseModel):
    """The movie behind the trailer, found with Google Search. '' = not confirmed."""

    identified: bool = Field(default=False, description="True only if Google Search confirmed which movie this is.")
    title: str = Field(default="", description="The official title in English / Latin letters.")
    native_title: str = Field(default="", description="The official title in its own script, if different.")
    release_date: str = ""
    release_status: str = Field(default="", description="'upcoming' or 'released'.")
    platform: str = Field(default="", description="'In cinemas' or the OTT platform it releases on.")
    languages: str = ""
    genre: str = ""
    director: str = ""
    music_director: str = ""
    cast: str = Field(default="", description="The 3-6 main actors, comma-separated.")
    production: str = ""
    official_hashtags: str = Field(default="", description="The movie's official hashtags and the production "
                                                           "house's handle, space-separated.")
    synopsis: str = Field(default="", description="One line from official sources.")
    note: str = Field(default="", description="Anything still uncertain, in one sentence.")


# ============================================================================
# Gemini: the clips (one call over the whole trailer)
# ============================================================================

def max_trailer_clips(duration: float) -> int:
    """Trailers are dense with standalone moments, so a little more than
    Clips allows: 1 under 30 s, 2 under 90 s, 3 under 4 minutes, else 4."""
    if duration < TINY_SOURCE_SECONDS:
        return 1
    if duration < 90:
        return 2
    return 3 if duration < 240 else 4


def _trailer_prompt(media: MediaInfo, pauses: list[tuple[float, float]], cuts: list[float],
                    correction: str = "") -> str:
    pause_times = ", ".join(f"{(a + b) / 2:.1f}" for a, b in pauses[:300]) or "none measured"
    cut_times = ", ".join(f"{c:.1f}" for c in cuts[:200]) or "none measured"
    cap = max_trailer_clips(media.duration_seconds)
    return (
        f"You are a short-form editor for a movie fan page. Watch and listen to this whole "
        f"{media.duration_seconds:.0f}-second movie trailer or teaser. Find the moments worth posting on their own "
        f"as YouTube Shorts / Instagram Reels, each cut straight from it as ONE continuous piece of "
        f"{CLIP_MIN_SECONDS}-{CLIP_MAX_SECONDS} seconds.\n\n"
        "The clips keep the trailer's ORIGINAL sound (dialogue and music), so:\n"
        "- Never start or end in the middle of a line of dialogue or a sung phrase. Start on a cut, a beat or "
        "right before a line; end after the line lands or on a beat drop / title card.\n"
        "- The best moments: the title reveal, the hero's or heroine's introduction, a punch dialogue, the "
        "action peak, an emotional beat, the song hook, the villain's line, the closing title / release card.\n"
        "- A clip opens on a hook (striking line, image or sound in its first 2 seconds) and ends on a payoff.\n"
        "- It must work for someone who never saw the trailer.\n"
        "- Score each clip 0-100 for how well it would hold viewers as a standalone Short. Strict: 70+ only for "
        "clips you would really post.\n"
        "- NEVER split the trailer into consecutive pieces (0-30 s, 30-60 s, ...). Each clip is a separate "
        "moment with gaps between clips.\n"
        f"- recommended_count = the clips scoring 70+, at most {cap} for this video, together under half of its "
        "length. Return those plus up to 3 alternatives scoring 50-69, best first. Clips must not overlap.\n"
        f"- Times are seconds from the start, between 0 and {media.duration_seconds:.0f}. Speech pauses "
        f"(seconds): {pause_times}. Shot changes: {cut_times}.\n"
        "- title_on_screen and credits_on_screen: read the title card and every credit / end card word for "
        "word -- they are used to find the movie with Google Search, so copy names and dates exactly.\n"
        "- what_is_said: the dialogue in its original language and script, with the English meaning.\n"
        "- Do not name the actors from their faces; only write names that appear on screen.\n"
        + (f"\nIMPORTANT: {correction}\n" if correction else "")
        + "Return JSON matching the schema."
    )


def finalise_trailer_clips(plan: TrailerPlan, duration: float, pauses: list[tuple[float, float]],
                           cuts: list[float]) -> list[dict]:
    """Gemini's clips with start / end moved onto a pause or cut (8-45 s, no
    overlaps), as Clips does. Recommended: score 70+, stands alone, at most
    max_trailer_clips(duration), together at most half the trailer; up to
    MAX_ALTERNATIVES others are kept as extras. In trailer order."""
    cap = max_trailer_clips(duration)

    def best_spans(items: list[dict]) -> list[tuple[float, float]]:
        best = sorted(items, key=lambda c: c["score"], reverse=True)[:cap]
        return [(c["start"], c["end"]) for c in best]

    kept = _pick_clips(plan, duration, pauses, cuts, gap=-2.0)
    if looks_chopped(best_spans(kept), duration):
        logger.info("[TRAILER] clips are back-to-back -- keeping only separate moments")
        kept = _pick_clips(plan, duration, pauses, cuts, gap=CHOP_GAP_SECONDS)
    budget = coverage_budget(duration)
    good: list[dict] = []
    for clip in kept:  # best first
        length = clip["end"] - clip["start"]
        if (clip["score"] >= RECOMMEND_SCORE and clip["stands_alone"] and len(good) < cap
                and sum(g["end"] - g["start"] for g in good) + length <= budget):
            good.append(clip)
    others = [c for c in kept if not any(c is g for g in good)][:MAX_ALTERNATIVES]
    for clip in good + others:
        clip["recommended"] = any(clip is g for g in good)
    return sorted(good + others, key=lambda c: c["start"])


# ============================================================================
# Google Search: which movie is this?
# ============================================================================

def movie_clues(plan: TrailerPlan, video_title: str, source_url: str) -> str:
    return "\n".join(part for part in [
        f"Video title: {video_title}" if video_title else "",
        f"Link: {source_url}" if source_url.startswith("http") else "",
        f"Kind: {plan.video_kind}",
        f"Title card: {plan.title_on_screen}" if plan.title_on_screen.strip() else "",
        f"Credits on screen: {plan.credits_on_screen}" if plan.credits_on_screen.strip() else "",
        f"Spoken language: {plan.spoken_language}" if plan.spoken_language.lower() != "none" else "",
        f"Music: {plan.source_music}" if plan.source_music.strip().lower() not in ("", "none") else "",
        f"What it shows: {plan.video_summary.strip()[:500]}",
    ] if part)


def _movie_prompt(clues: str, search: bool) -> str:
    keys = ", ".join(MovieFacts.model_fields)
    return (
        "Identify the movie this trailer / teaser belongs to and find its official details.\n\n"
        f"=== CLUES FROM THE VIDEO ===\n{clues}\n\n"
        + ("=== RESEARCH (use Google Search) ===\n"
           "1. Search the title card text and the video title together with 'trailer', 'teaser' or 'movie'.\n"
           "2. Confirm on official or reliable pages: the production house, the official YouTube channel, "
           "Wikipedia, major film news sites.\n"
           "3. Find the release date, where it releases (in cinemas or which OTT platform), the languages, genre, "
           "director, music director, the 3-6 main cast and the production house.\n"
           "4. Find the movie's official hashtags and the production house's social handle.\n\n"
           if search else "")
        + "Rules: names exactly as officially credited. A field the search did not confirm stays '' -- never "
          "guess. identified=true only when you are sure which movie this is; if two movies fit, say so in note.\n"
        + (f"\nReply with ONLY one JSON object and no other text, with these keys: {keys}." if search else "")
    )


def identify_movie(plan: TrailerPlan, video_title: str, source_url: str,
                   notify: Notify = _noop_notify) -> tuple[MovieFacts, list[str], list[dict[str, str]], str]:
    """(facts, searches run, sources, model). Only a Google-Search-grounded
    answer is accepted (GeminiCallError / QuotaExhaustedError otherwise)."""
    notify("Finding the movie with Google Search...")
    clues = movie_clues(plan, video_title, source_url)
    facts, queries, sources, model = generate_with_optional_search(
        lambda search: _movie_prompt(clues, search),
        MovieFacts,
        task="metadata",
        label="movie-facts",
        notify=notify,
        temperature=0.2,
        thinking_budget=METADATA_THINKING_BUDGET,
        require_search=True,
    )
    return facts, queries, sources, model


# ============================================================================
# Rendering (ffmpeg): the original sound, never replaced
# ============================================================================

def render_trailer_clip(source: Path, media: MediaInfo, start: float, end: float, output: Path) -> Path:
    """One 1080x1920 clip with the trailer's own sound: a short fade in and
    out at the edges, normalised to -14 LUFS. Same picture as Clips
    (vertical filled, landscape on a blurred copy of itself)."""
    length = max(1.0, end - start)
    inputs = ["-ss", f"{start:.3f}", "-t", f"{length:.3f}", "-i", str(source)]
    fade_out = min(FADE_OUT_SECONDS, length / 4)
    if media.has_audio:
        audio = (f"[0:a]aresample=48000,aformat=channel_layouts=stereo,afade=t=in:st=0:d={FADE_IN_SECONDS:g},"
                 f"afade=t=out:st={length - fade_out:.3f}:d={fade_out:.3f},"
                 "loudnorm=I=-14:TP=-1.5:LRA=11,aresample=48000[a]")
    else:  # a silent source still gets an audio track, as the platforms expect
        inputs += ["-f", "lavfi", "-t", f"{length:.3f}", "-i", "anullsrc=r=48000:cl=stereo"]
        audio = "[1:a]anull[a]"
    command = [ffmpeg_exe(), "-y", "-hide_banner", "-loglevel", "error", *inputs,
               "-filter_complex", f"{video_filter(media)};{audio}", "-map", "[v]", "-map", "[a]",
               "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
               "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", "-t", f"{length:.3f}", str(output)]
    try:
        result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace",
                                timeout=900)
    except subprocess.TimeoutExpired as exc:
        raise ClipRenderError("Rendering the clip took more than 15 minutes.") from exc
    if result.returncode != 0 or not output.is_file():
        raise ClipRenderError(f"ffmpeg could not render the clip: {result.stderr.strip()[-400:]}")
    return output


# ============================================================================
# Metadata: written from the checked movie facts
# ============================================================================

def movie_fact_lines(movie: dict) -> list[str]:
    return [f"{label}: {movie[key].strip()}" for key, label in MOVIE_FIELDS if str(movie.get(key, "")).strip()]


def trailer_metadata_context(clip: dict, plan: TrailerPlan, movie: dict, source_url: str) -> str:
    title = (movie.get("title") or plan.title_on_screen).strip()
    kind = plan.video_kind.strip() or "trailer"
    facts = movie_fact_lines(movie)
    on_screen = content_texts([clip.get("on_screen_text", "")])
    length = clip["end"] - clip["start"]
    return "\n".join(part for part in [
        (f"Content: a {length:.0f}-second Short cut from the official {kind} of the movie {title}, posted with its "
         "original dialogue and music." if title else
         f"Content: a {length:.0f}-second Short cut from a movie {kind}, posted with its original dialogue and music."),
        "Movie facts checked with Google Search -- use these names and dates exactly and add no other names: "
        + "; ".join(facts) if facts else "",
        f"Clip title idea: {clip['title']}",
        f"Moment: {clip.get('moment', '')}" if clip.get("moment") else "",
        f"Hook: {clip['hook']}",
        f"Payoff: {clip['payoff']}",
        f"Dialogue: {clip['what_is_said']}" if clip.get("what_is_said") else "",
        "Text on screen: " + " | ".join(on_screen) if on_screen else "",
        f"The whole {kind}: {plan.video_summary.strip()[:400]}",
        f"Spoken language: {plan.spoken_language}" if plan.spoken_language.lower() != "none" else "",
        f"Trailer music: {plan.source_music.strip()}" if plan.source_music.strip().lower() not in ("", "none") else "",
        "Put the movie title and the word trailer / teaser in the YouTube title, and use the movie's official "
        "hashtags." if title else "",
        f"End the description with the credit line: Official {kind}: {source_url}" if source_url.startswith("http")
        else "",
    ] if part)


# ============================================================================
# Page
# ============================================================================

def _state(st) -> dict:
    return st.session_state.setdefault("trailer", {"stage": "input"})


def _start_over(st) -> None:
    job = _state(st).get("job")
    if job:
        shutil.rmtree(job, ignore_errors=True)
    st.session_state["trailer"] = {"stage": "input"}


def _key(st, name: str) -> str:
    return f"trailer_{name}_{Path(_state(st)['job']).name}"


def render_page(st) -> None:
    st.title("Trailer / Teaser clips")
    state = _state(st)
    steps = ["1 · Trailer", "2 · Movie + clips", "3 · Clips + metadata"]
    position = ["input", "pick", "done"].index(state["stage"])
    st.caption("  →  ".join(f"**{s}**" if i == position else s for i, s in enumerate(steps)))
    {"input": _page_input, "pick": _page_pick, "done": _page_done}[state["stage"]](st)


def _page_input(st) -> None:
    st.caption(
        "Paste the YouTube link of a newly released movie trailer or teaser, or upload it (up to "
        f"{TRAILER_MAX_VIDEO_SECONDS / 60:.0f} minutes). Gemini watches it and recommends the moments worth "
        "posting as Shorts / Reels, Google Search finds the movie (title, cast, crew, release date), and each clip "
        "keeps the trailer's **original dialogue and music** -- no new voice-over."
    )
    st.info(RIGHTS_NOTE)
    for problem in (problems := CONFIG.problems()):
        st.error(problem)
    url = st.text_input("Trailer / teaser link", placeholder="https://www.youtube.com/watch?v=...",
                        key="trailer_url")
    uploaded = st.file_uploader("Or upload the trailer", type=["mp4", "mov", "webm", "m4v", "mkv"],
                                key="trailer_upload")
    st.caption(f"Uploads up to {UPLOAD_LIMIT_MB} MB. A YouTube link is best: Gemini watches it straight from "
               "YouTube, and the video title helps find the movie.")
    if not st.button("Find clips", type="primary", disabled=bool(problems) or not (url.strip() or uploaded),
                     key="trailer_find"):
        return
    job = CONFIG.work_dir / "trailer" / uuid.uuid4().hex[:12]
    job.mkdir(parents=True, exist_ok=True)
    with st.status("Starting...", expanded=True) as status:
        notify = lambda message: status.update(label=message)  # noqa: E731
        try:
            if uploaded is not None:
                video = job / f"source{Path(uploaded.name).suffix.lower() or '.mp4'}"
                uploaded.seek(0)
                with video.open("wb") as out:  # streamed: no second copy of a big upload in memory
                    shutil.copyfileobj(uploaded, out, 8 * 1024 * 1024)
                title, source_url = Path(uploaded.name).stem, f"upload://{uploaded.name}"
            else:
                notify("Downloading the trailer...")
                downloaded = download_video(
                    url.strip(), job / "download", notify, max_seconds=TRAILER_MAX_VIDEO_SECONDS, max_height=1080,
                    max_mb=TRAILER_MAX_DOWNLOAD_MB, too_long_message=TOO_LONG,
                )
                video, title, source_url = downloaded.path, downloaded.title, downloaded.source_url
            notify("Checking the video...")
            media = probe_video(video)
            if media.duration_seconds > TRAILER_MAX_VIDEO_SECONDS + 0.5:
                raise URLVideoError(TOO_LONG)
            if media.duration_seconds < CLIP_MIN_SECONDS:
                raise URLVideoError(f"This video is only {media.duration_seconds:.1f} seconds long. Clips need at "
                                    f"least {CLIP_MIN_SECONDS} seconds of video.")
            notify("Finding the speech pauses...")
            pauses = speech_pauses(video) if media.has_audio else []
            notify("Measuring the shot changes...")
            cuts = detect_scene_cuts(video)
            plan, model = find_clips(video, source_url, media, pauses, cuts, notify, schema=TrailerPlan,
                                     build_prompt=_trailer_prompt, label="trailer")
            clips = finalise_trailer_clips(plan, media.duration_seconds, pauses, cuts)
        except (*USER_FACING_ERRORS, ClipRenderError) as exc:
            status.update(label="Could not find clips", state="error")
            st.error(str(exc))
            shutil.rmtree(job, ignore_errors=True)
            return
        # The movie lookup is best-effort: without it the clips still work and
        # the facts can be typed in on the next step.
        movie, search = MovieFacts(title=plan.title_on_screen.strip()), {"queries": [], "sources": [], "model": ""}
        movie_error = ""
        try:
            movie, queries, sources, search_model = identify_movie(plan, title, source_url, notify)
            search = {"queries": queries, "sources": sources, "model": search_model}
        except USER_FACING_ERRORS as exc:
            movie_error = str(exc)
            logger.warning(f"[TRAILER] movie lookup failed: {exc}")
        status.update(label="Clips found", state="complete", expanded=False)
    logger.info(f"[TRAILER] {len(clips)} clips ({sum(c['recommended'] for c in clips)} recommended) by {model}; "
                f"movie identified={movie.identified}")
    st.session_state["trailer"] = {
        "stage": "pick", "job": str(job), "video": str(video), "title": title, "source_url": source_url,
        "media": asdict(media), "plan": plan.model_dump(), "clips": clips, "model": model,
        "movie": movie.model_dump(), "movie_search": search, "movie_error": movie_error,
        "rendered": {}, "social": {},
    }
    st.rerun()


def _movie_card(st, state: dict) -> None:
    """The movie as Google Search found it; every field can be corrected,
    and the metadata is written from what is here."""
    movie, search = state["movie"], state["movie_search"]
    with st.container(border=True):
        if movie.get("identified"):
            st.markdown(f"**🎬 {movie.get('title') or 'Movie found'}**"
                        + (f" · {movie['native_title']}" if movie.get("native_title") else ""))
            st.caption(grounding_label(search["queries"], search["sources"], search["model"])
                       + ". Check the details -- the metadata is written from them.")
        else:
            st.warning("Google Search could not confirm which movie this is"
                       + (f": {state['movie_error']}" if state.get("movie_error") else ".")
                       + " Fill in what you know -- the metadata is written from these details.")
        if movie.get("synopsis"):
            st.caption(movie["synopsis"])
        if movie.get("note"):
            st.caption(f"Note: {movie['note']}")
        columns = st.columns(2)
        for number, (key, label) in enumerate(MOVIE_FIELDS):
            movie[key] = columns[number % 2].text_input(label, movie.get(key, ""), key=_key(st, f"movie_{key}"))
        if search["sources"]:
            with st.expander("Sources used"):
                for source in search["sources"]:
                    st.markdown(f"- [{source['title']}]({source['uri']})")


def _page_pick(st) -> None:
    state = _state(st)
    plan = TrailerPlan.model_validate(state["plan"])
    media = MediaInfo(**state["media"])
    clips = state["clips"]
    recommended = sum(c["recommended"] for c in clips)
    st.subheader(state["title"])
    st.caption(f"{plan.video_kind or 'trailer'} · {_clock(media.duration_seconds)} · {state['source_url']} · "
               f"analysed by {state['model']}")
    _movie_card(st, state)
    st.markdown(f"**Gemini recommends {recommended} clip(s)**"
                + (f" ({len(clips) - recommended} more possible)" if len(clips) > recommended else "")
                + f" — {plan.count_reason}")
    if clips:
        st.markdown(timeline_html(clips, media.duration_seconds), unsafe_allow_html=True)
    else:
        st.warning("Gemini found no moment that works on its own as a Short. You can still post the whole "
                   f"{plan.video_kind or 'trailer'} below.")
    if clips and not recommended:
        st.warning("No clip is strong enough to recommend. The possible ones are below if you want to try them.")
    st.caption("🔊 Every clip keeps the original dialogue and music, with a short fade at each end.")

    chosen = []
    extras = [(i, c) for i, c in enumerate(clips) if not c["recommended"]]
    for index, clip in enumerate(clips):
        if clip["recommended"]:
            _clip_card(st, index, clip, media, chosen)
    if extras:
        with st.expander(f"Show {len(extras)} more possible clip(s) (not recommended)"):
            for index, clip in extras:
                _clip_card(st, index, clip, media, chosen)
    chosen.sort(key=lambda c: c["start"])
    full = st.checkbox(f"Also make the whole {plan.video_kind or 'trailer'} one vertical video "
                       f"({_clock(media.duration_seconds)})", value=not clips, key=_key(st, "full"))
    if full:
        chosen.append({"index": "full", "title": f"Full {plan.video_kind or 'trailer'}", "moment": "whole video",
                       "start": 0.0, "end": round(media.duration_seconds, 2), "hook": plan.video_summary,
                       "payoff": "", "what_is_said": "", "on_screen_text": plan.title_on_screen, "score": 0})

    bad = [c for c in chosen if c["index"] != "full" and not 3 <= c["end"] - c["start"] <= 180]
    for clip in bad:
        st.warning(f"'{clip['title']}': end must be 3-180 s after start.")
    left, right = st.columns(2)
    if right.button("Start over", key=_key(st, "restart")):
        _start_over(st)
        st.rerun()
    if not left.button(f"Render {len(chosen)} clip(s)", type="primary", disabled=not chosen or bool(bad),
                       key=_key(st, "render")):
        return
    _render_chosen(st, chosen, media)


def _clip_card(st, index: int, clip: dict, media: MediaInfo, chosen: list[dict]) -> None:
    """One clip: tick, times, moment / hook / payoff / dialogue."""
    with st.container(border=True):
        top = st.columns([3, 1])
        take = top[0].checkbox(f"**{clip['title']}** · {clip.get('moment', '')} · score {clip['score']}"
                               + (" · ⭐ recommended" if clip["recommended"] else ""), value=clip["recommended"],
                               key=_key(st, f"take_{index}"))
        top[1].caption(f"{_clock(clip['start'])} → {_clock(clip['end'])} ({clip['end'] - clip['start']:.0f} s)")
        st.caption(f"**Hook:** {clip['hook']}  \n**Payoff:** {clip['payoff']}  \n**Why:** {clip['why_it_works']}"
                   + (f"  \n**Dialogue:** {clip['what_is_said']}" if clip.get("what_is_said") else "")
                   + (f"  \n**Text on screen:** {clip['on_screen_text']}" if clip.get("on_screen_text") else ""))
        times = st.columns(2)
        start = times[0].number_input("Start (s)", 0.0, media.duration_seconds, float(clip["start"]), 0.5,
                                      key=_key(st, f"start_{index}"))
        end = times[1].number_input("End (s)", 0.0, media.duration_seconds, float(clip["end"]), 0.5,
                                    key=_key(st, f"end_{index}"))
        if take:
            chosen.append({**clip, "index": index, "start": start, "end": end})


def _render_chosen(st, chosen: list[dict], media: MediaInfo) -> None:
    state = _state(st)
    job = Path(state["job"])
    rendered: dict[str, dict] = {}
    with st.status("Rendering the clips...", expanded=True) as status:
        for number, clip in enumerate(chosen, start=1):
            status.update(label=f"Clip {number} of {len(chosen)}: {clip['title']}")
            name = "full" if clip["index"] == "full" else f"{number:02d}_{int(clip['start'])}s"
            output = job / f"trailer_clip_{name}.mp4"
            try:
                render_trailer_clip(Path(state["video"]), media, clip["start"], clip["end"], output)
            except ClipRenderError as exc:
                st.error(f"Clip {number} ({clip['title']}): {exc}")
                continue
            rendered[str(clip["index"])] = {**clip, "path": str(output)}
        status.update(label=f"{len(rendered)} clip(s) ready", state="complete" if rendered else "error",
                      expanded=False)
    if rendered:
        archive_path = job / "trailer_clips.zip"
        with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_STORED) as archive:
            for clip in rendered.values():
                archive.write(clip["path"], Path(clip["path"]).name)
        state.update(stage="done", rendered=rendered, social={}, zip=str(archive_path))
        st.rerun()


def _page_done(st) -> None:
    state = _state(st)
    plan = TrailerPlan.model_validate(state["plan"])
    movie = state["movie"]
    rendered = state["rendered"]
    st.success(f"{len(rendered)} clip(s) ready, each with the original sound and its own YouTube and Instagram "
               "metadata below.")
    left, mid, right = st.columns(3)
    archive = Path(state.get("zip", ""))
    if archive.name and archive.is_file():
        left.download_button("Download all clips (ZIP)", archive.read_bytes(), file_name="trailer_clips.zip",
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
        st.caption(f"{_clock(clip['start'])} → {_clock(clip['end'])} · {clip['end'] - clip['start']:.0f} s · "
                   "original sound")
        path = Path(clip["path"])
        columns = st.columns([1, 2])
        if path.is_file():
            columns[0].video(str(path))
            columns[0].download_button("Download clip", path.read_bytes(), file_name=path.name, mime="video/mp4",
                                       key=_key(st, f"dl_{index}"))
        with columns[1]:
            stored = state["social"].get(index)
            social = SocialMetadata.model_validate(stored) if stored else None

            def make(notify: Notify, clip=clip, index=index):
                meta = generate_metadata(trailer_metadata_context(clip, plan, movie, state["source_url"]), "video",
                                         language, notify, audio_language=language)
                # The clip keeps the trailer's own sound: no other audio to suggest.
                meta = meta.model_copy(update={"trending_audio": []})
                state["social"][index] = meta.model_dump()
                return meta

            render_social_or_generate(st, social, f"trailer_{Path(state['job']).name}_{index}", make)
