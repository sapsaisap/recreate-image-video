"""Trailer / Teaser: choosing clips, the movie lookup (Google Search
required), rendering with real ffmpeg (the ORIGINAL sound kept, unlike
Clips), metadata written from the movie facts, and the whole page."""
import inspect
import textwrap
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import recreate_video as rv
import trailer
from tests.test_clips import ffmpeg, make_video, max_volume
from trailer import MovieFacts, TrailerClipIdea, TrailerPlan, finalise_trailer_clips, max_trailer_clips

REPO = Path(__file__).resolve().parent.parent


def idea(start, end, score=85, stands_alone=True, title=None, moment="punch dialogue"):
    return TrailerClipIdea(start_seconds=start, end_seconds=end, title=title or f"clip {start:g}", moment=moment,
                           hook="h", payoff="p", why_it_works="w", what_is_said="నేను వస్తున్నా (I am coming)",
                           score=score, stands_alone=stands_alone, on_screen_text="")


def plan(*ideas, title_on_screen="దేవర | DEVARA"):
    return TrailerPlan(video_summary="An action trailer.", video_kind="trailer", spoken_language="Telugu",
                       language_code="te", title_on_screen=title_on_screen,
                       credits_on_screen="In cinemas 10 Jan", source_music="Background score",
                       recommended_count=len(ideas), count_reason="because", clips=list(ideas))


# ── Choosing clips ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("seconds, cap", [(20, 1), (60, 2), (150, 3), (300, 4)])
def test_trailers_allow_a_few_more_clips_than_clips_mode(seconds, cap):
    assert max_trailer_clips(seconds) == cap


def test_recommends_strong_standalone_moments_and_keeps_extras():
    clips = finalise_trailer_clips(
        plan(idea(5, 18, 95, title="Title reveal"), idea(40, 55, 80, title="Hero intro"),
             idea(90, 104, 72, title="Punch line"), idea(120, 133, 60, title="Song"),
             idea(140, 150, 90, stands_alone=False, title="Needs context")),
        duration=160, pauses=[], cuts=[])
    recommended = [c["title"] for c in clips if c["recommended"]]
    assert recommended == ["Title reveal", "Hero intro", "Punch line"]  # 3 for a 160 s trailer
    assert {c["title"] for c in clips if not c["recommended"]} == {"Song", "Needs context"}
    assert [c["start"] for c in clips] == sorted(c["start"] for c in clips)  # trailer order


def test_prompt_keeps_the_original_sound_and_reads_the_credits():
    media = rv.MediaInfo(duration_seconds=150, width=1920, height=1080, has_audio=True, aspect_ratio="16:9")
    prompt = trailer._trailer_prompt(media, [(10.0, 10.5)], [20.0])
    assert "ORIGINAL sound" in prompt and "Never start or end in the middle of a line of dialogue" in prompt
    assert "title_on_screen and credits_on_screen" in prompt
    assert "narration" not in prompt.lower()  # no new voice-over in this mode
    assert "at most 3 for this video" in prompt


# ── Movie lookup ────────────────────────────────────────────────────────────

def test_movie_lookup_requires_google_search_and_uses_the_on_screen_clues(monkeypatch):
    seen = {}

    def fake_search(build_prompt, schema, **kwargs):
        seen.update(kwargs, prompt=build_prompt(True), schema=schema)
        return MovieFacts(identified=True, title="Devara"), ["devara trailer"], [{"title": "W", "uri": "u"}], "m"

    monkeypatch.setattr(trailer, "generate_with_optional_search", fake_search)
    facts, queries, sources, model = trailer.identify_movie(
        plan(), "DEVARA Official Trailer | NTR", "https://www.youtube.com/watch?v=abcdefghijk")
    assert facts.title == "Devara" and queries == ["devara trailer"] and model == "m"
    assert seen["require_search"] is True and seen["task"] == "metadata" and seen["schema"] is MovieFacts
    prompt = seen["prompt"]
    for clue in ("DEVARA Official Trailer | NTR", "దేవర | DEVARA", "In cinemas 10 Jan", "Spoken language: Telugu"):
        assert clue in prompt
    assert "never guess" in prompt and "release_date" in prompt


def test_metadata_context_is_built_from_the_checked_facts():
    movie = MovieFacts(identified=True, title="Devara", release_date="10 January 2027", cast="A, B",
                       official_hashtags="#Devara").model_dump()
    clip = {**idea(5, 20).model_dump(), "start": 5.0, "end": 20.0}
    context = trailer.trailer_metadata_context(clip, plan(), movie, "https://www.youtube.com/watch?v=abcdefghijk")
    assert context.startswith("Content: a 15-second Short cut from the official trailer of the movie Devara")
    assert "Movie title: Devara; Release date: 10 January 2027; Cast: A, B; Official hashtags / handles: #Devara" \
        in context
    assert "Dialogue: నేను వస్తున్నా (I am coming)" in context
    assert context.endswith("Official trailer: https://www.youtube.com/watch?v=abcdefghijk")


# ── Rendering: the original sound is kept ───────────────────────────────────

def test_clip_keeps_the_original_sound(tmp_path):
    source = make_video(tmp_path / "trailer.mp4", 20)  # landscape, with a steady 300 Hz tone
    media = rv.probe_video(source)
    output = trailer.render_trailer_clip(source, media, 5.0, 15.0, tmp_path / "clip.mp4")
    clip = rv.probe_video(output)
    assert (clip.width, clip.height) == (1080, 1920) and clip.has_audio
    assert clip.duration_seconds == pytest.approx(10, abs=0.2)
    middle = tmp_path / "middle.wav"
    ffmpeg("-ss", "4", "-t", "2", "-i", str(output), "-vn", str(middle))
    assert max_volume(middle) > -20  # the source's own tone, not silence or a new voice


def test_silent_source_still_gets_an_audio_track(tmp_path):
    source = make_video(tmp_path / "teaser.mp4", 12, size="720x1280", audio=False)
    output = trailer.render_trailer_clip(source, rv.probe_video(source), 0.0, 10.0, tmp_path / "clip.mp4")
    clip = rv.probe_video(output)
    assert clip.has_audio and (clip.width, clip.height) == (1080, 1920)


def test_mode_is_in_the_sidebar():
    source = inspect.getsource(rv.run_app)
    assert "TRAILER_MODE" in source and "render_trailer_page" in source
    assert rv.TRAILER_MODE == "🎥 Trailer / Teaser" and "trailer" in rv.APP_MODULES


# ── Whole page, end to end ──────────────────────────────────────────────────

WRAPPER = textwrap.dedent('''
    import dataclasses, sys
    from pathlib import Path
    sys.path.insert(0, {repo!r})
    sys.path.insert(0, {tests!r})
    import streamlit as st
    import recreate_video as rv
    import trailer
    from test_clips import make_video
    from test_trailer import idea, plan

    work = Path({work!r})
    config = dataclasses.replace(rv.CONFIG, api_key="dummy", work_dir=work)
    rv.CONFIG = trailer.CONFIG = config
    calls = st.session_state.setdefault("calls", [])

    def fake_download(url, destination, notify, **limits):
        calls.append(("download", limits["max_seconds"]))
        destination.mkdir(parents=True, exist_ok=True)
        video = make_video(destination / "video.mp4", 100)
        return rv.DownloadedVideo(path=video, title="DEVARA Official Trailer", source_url=url, extractor="youtube")

    def fake_find(video, source_url, media, pauses, cuts, notify, **options):
        calls.append(("find", options["label"], options["schema"].__name__))
        return plan(idea(2, 16, 92, title="Title reveal"), idea(40, 55, 80, title="Hero intro"),
                    idea(70, 84, 55, title="Song")), "gemini-3.6-flash"

    def fake_identify(plan, title, source_url, notify):
        calls.append(("identify", title))
        if {fail_lookup}:
            raise rv.QuotaExhaustedError("today's free Google Search quota is used up")
        return (trailer.MovieFacts(identified=True, title="Devara", release_date="10 January 2027",
                                   cast="Actor One"), ["devara trailer"], [{{"title": "Wiki", "uri": "https://w"}}],
                "gemini-2.5-flash")

    def fake_metadata(context, kind, language, notify, audio_language=""):
        calls.append(("metadata", context, language))
        return rv.SocialMetadata(youtube_title="T #Shorts", youtube_description="D", youtube_tags="a, b",
                                 instagram_caption="C", instagram_hashtags="#a #b #c #d #e",
                                 trending_audio=[rv.TrendingAudio(track="Some song")],
                                 model="gemini-2.5-flash", search_queries=["q"])

    trailer.download_video = fake_download
    trailer.find_clips = fake_find
    trailer.identify_movie = fake_identify
    trailer.generate_metadata = fake_metadata
    trailer.render_page(st)
''')


@pytest.fixture
def page(tmp_path, monkeypatch):
    for name in ["download_video", "find_clips", "identify_movie", "generate_metadata", "CONFIG"]:
        monkeypatch.setattr(trailer, name, getattr(trailer, name))
    monkeypatch.setattr(rv, "CONFIG", rv.CONFIG)

    def build(fail_lookup=False):
        wrapper = tmp_path / "app.py"
        wrapper.write_text(WRAPPER.format(repo=str(REPO), tests=str(REPO / "tests"), work=str(tmp_path / "work"),
                                          fail_lookup=fail_lookup), encoding="utf-8")
        return wrapper
    return build


def click(app, label):
    next(b for b in app.button if b.label == label).click()
    return app.run()


def advance(app, label, wrapper):
    app = click(app, label)
    assert not app.exception, app.exception
    fresh = AppTest.from_file(str(wrapper), default_timeout=300)
    for key in ("trailer", "calls", "social_errors"):
        if key in app.session_state:
            fresh.session_state[key] = app.session_state[key]
    return fresh.run()


def test_whole_flow_from_trailer_link_to_clips_with_original_sound(page):
    wrapper = page()
    app = AppTest.from_file(str(wrapper), default_timeout=300).run()
    assert not app.exception and app.button[0].disabled

    # 1. link -> clips found and the movie identified with Google Search
    app.text_input[0].input("https://www.youtube.com/watch?v=abcdefghijk")
    app = advance(app, "Find clips", wrapper)
    state = app.session_state.trailer
    assert state["stage"] == "pick", [e.value for e in app.error]
    calls = app.session_state.calls
    assert ("download", trailer.TRAILER_MAX_VIDEO_SECONDS) in calls
    assert ("find", "trailer", "TrailerPlan") in calls and ("identify", "DEVARA Official Trailer") in calls
    assert state["movie"]["identified"] and state["movie"]["title"] == "Devara"
    assert any("Researched with Google Search" in c.value for c in app.caption)
    assert any("Gemini recommends 2 clip(s)" in m.value for m in app.markdown)

    # 2. correct the cast, add the whole trailer, render
    cast = next(t for t in app.text_input if t.label == "Cast")
    cast.input("Actor One, Actor Two")
    full = next(c for c in app.checkbox if c.label.startswith("Also make the whole trailer"))
    full.check()
    app = app.run()
    app = advance(app, "Render 3 clip(s)", wrapper)
    state = app.session_state.trailer
    assert state["stage"] == "done" and not app.exception
    rendered = state["rendered"]
    assert [c["title"] for c in rendered.values()] == ["Title reveal", "Hero intro", "Full trailer"]
    for clip in rendered.values():
        media = rv.probe_video(Path(clip["path"]))
        assert (media.width, media.height) == (1080, 1920) and media.has_audio
        assert media.duration_seconds == pytest.approx(clip["end"] - clip["start"], abs=0.25)
        middle = Path(clip["path"]).with_suffix(".mid.wav")
        ffmpeg("-ss", "3", "-t", "2", "-i", clip["path"], "-vn", str(middle))
        assert max_volume(middle) > -20  # the trailer's own sound
    assert rendered["full"]["end"] == pytest.approx(100, abs=0.2)

    # 3. each clip's metadata comes from the checked facts, with the corrected cast
    metadata = [call for call in app.session_state.calls if call[0] == "metadata"]
    assert len(metadata) == 3 and all(language == "Telugu" for _, _, language in metadata)
    assert all("Cast: Actor One, Actor Two" in context and "movie Devara" in context for _, context, _ in metadata)
    assert all(not s["trending_audio"] for s in state["social"].values())  # original sound kept


def test_movie_lookup_failure_still_gives_clips_and_editable_facts(page):
    wrapper = page(fail_lookup=True)
    app = AppTest.from_file(str(wrapper), default_timeout=300).run()
    app.text_input[0].input("https://www.youtube.com/watch?v=abcdefghijk")
    app = advance(app, "Find clips", wrapper)
    state = app.session_state.trailer
    assert state["stage"] == "pick" and not state["movie"]["identified"]
    assert any("could not confirm which movie" in w.value for w in app.warning)
    title = next(t for t in app.text_input if t.label == "Movie title")
    assert title.value == "దేవర | DEVARA"  # prefilled from the title card
    assert [c["title"] for c in state["clips"]] == ["Title reveal", "Hero intro", "Song"]


def test_too_long_trailer_is_rejected_before_any_gemini_call(page, monkeypatch):
    wrapper = page()
    long_media = rv.MediaInfo(duration_seconds=trailer.TRAILER_MAX_VIDEO_SECONDS + 60, width=1920, height=1080,
                              has_audio=True, aspect_ratio="16:9")
    text = wrapper.read_text(encoding="utf-8").replace(
        "trailer.render_page(st)", f"trailer.probe_video = lambda path: rv.MediaInfo(**{long_media.__dict__!r})\n"
                                   "trailer.render_page(st)")
    wrapper.write_text(text, encoding="utf-8")
    monkeypatch.setattr(trailer, "probe_video", trailer.probe_video)
    app = AppTest.from_file(str(wrapper), default_timeout=300).run()
    app.text_input[0].input("https://www.youtube.com/watch?v=abcdefghijk")
    app = click(app, "Find clips")
    assert [e.value for e in app.error] == [trailer.TOO_LONG]
    assert not any(call[0] in ("find", "identify") for call in app.session_state.calls)
