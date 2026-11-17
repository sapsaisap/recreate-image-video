"""Clips from a long video: snapping and choosing clips, no back-to-back
chopping, the Gemini call (faked), rendering with real ffmpeg (new voice-over
only, never the original sound), and the whole page from link to clips with
their own metadata."""
import subprocess
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest
from streamlit.testing.v1 import AppTest

import clips
import recreate_video as rv
from clips import ClipIdea, ClipPlan, finalise_clips, snap
from recreate_video import MediaInfo

REPO = Path(__file__).resolve().parent.parent


def idea(start, end, score=80, stands_alone=True, title=None):
    return ClipIdea(start_seconds=start, end_seconds=end, title=title or f"clip {start:g}", hook="h", payoff="p",
                    why_it_works="w", what_is_said="said", score=score, stands_alone=stands_alone,
                    narration="ఈ రహస్య మసాలా రుచిని మార్చేస్తుంది")


def plan(*ideas, count=None):
    return ClipPlan(video_summary="A cooking video.", spoken_language="Telugu", language_code="te",
                    recommended_count=len(ideas) if count is None else count, count_reason="because",
                    clips=list(ideas))


def ffmpeg(*args):
    subprocess.run([rv.ffmpeg_exe(), "-y", "-loglevel", "error", *args], capture_output=True, check=True)


def make_tone(path, seconds, frequency=900):
    ffmpeg("-f", "lavfi", "-i", f"sine=frequency={frequency}:duration={seconds}", str(path))
    return path


def max_volume(path):
    """Loudest sample in dB (-91 = digital silence)."""
    output = subprocess.run([rv.ffmpeg_exe(), "-hide_banner", "-i", str(path), "-af", "volumedetect", "-f", "null",
                             "-"], capture_output=True, text=True, encoding="utf-8", errors="replace").stderr
    import re
    return float(re.search(r"max_volume:\s*(-?[\d.]+) dB", output).group(1))


def make_video(path, seconds, size="1280x720", audio=True):
    sound = ["-f", "lavfi", "-i", f"sine=frequency=300:duration={seconds}"] if audio else []
    ffmpeg("-f", "lavfi", "-i", f"testsrc2=size={size}:rate=30:duration={seconds}", *sound,
           "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", *(["-c:a", "aac"] if audio else []),
           "-shortest", str(path))
    return path


# ── Choosing clips ──────────────────────────────────────────────────────────

class TestSnap:
    def test_moves_to_the_nearest_point_in_reach(self):
        assert snap(10.0, [7.0, 11.2, 12.0]) == 11.2

    def test_stays_when_nothing_is_near(self):
        assert snap(10.0, [5.0, 15.0]) == 10.0


class TestFinaliseClips:
    def test_start_and_end_land_on_pauses_and_cuts(self):
        clip, = finalise_clips(plan(idea(20.0, 50.0)), 300, pauses=[(18.9, 19.6)], cuts=[51.2])
        assert clip["start"] == 19.5 and clip["end"] == 51.2  # pause end - 0.1, the cut

    def test_end_inside_a_pause_keeps_the_last_word(self):
        clip, = finalise_clips(plan(idea(20.0, 49.0)), 300, pauses=[(48.0, 49.5)], cuts=[])
        assert clip["end"] == 48.15

    def test_too_long_clip_ends_on_the_last_pause_before_60_s(self):
        clip, = finalise_clips(plan(idea(10.0, 100.0)), 300, pauses=[(40.0, 40.5), (65.0, 65.5)], cuts=[])
        assert clip["start"] == 10.0 and clip["end"] == 65.15

    def test_too_long_without_pauses_is_cut_at_60_s(self):
        clip, = finalise_clips(plan(idea(10.0, 100.0)), 300, pauses=[], cuts=[])
        assert clip["end"] - clip["start"] == 60

    def test_too_short_clip_is_dropped(self):
        assert finalise_clips(plan(idea(10.0, 18.0)), 300, [], []) == []

    def test_times_past_the_end_are_clamped(self):
        clip, = finalise_clips(plan(idea(270.0, 400.0)), 300, [], [])
        assert clip["end"] == 300

    def test_overlap_keeps_the_better_clip(self):
        kept = finalise_clips(plan(idea(10, 40, score=60, title="weak"), idea(20, 50, score=90, title="strong")),
                              300, [], [])
        assert [c["title"] for c in kept] == ["strong"]

    def test_touching_clips_are_both_kept(self):
        assert len(finalise_clips(plan(idea(10, 40), idea(40, 70)), 300, [], [])) == 2

    def test_recommended_needs_score_70_and_standing_alone(self):
        kept = finalise_clips(plan(idea(100, 130, score=90), idea(10, 40, score=65),
                                   idea(200, 230, score=85, stands_alone=False)), 300, [], [])
        assert [c["start"] for c in kept] == [10, 100, 200]  # video order
        assert [c["recommended"] for c in kept] == [False, True, False]

    def test_at_most_eight_recommended_and_twelve_kept(self):
        many = [idea(i * 40.0, i * 40.0 + 30, score=95 - i) for i in range(14)]
        kept = finalise_clips(plan(*many), 14 * 40, [], [])
        assert len(kept) == 12 and sum(c["recommended"] for c in kept) == clips.MAX_CLIPS


class TestChopping:
    def test_back_to_back_pieces_are_chopping(self):
        assert clips.looks_chopped([(0, 30), (30, 60), (60, 90), (90, 120)], 120)

    def test_two_halves_of_a_short_video_are_chopping(self):
        assert clips.looks_chopped([(2, 60), (62, 118)], 120)

    def test_separate_moments_are_not(self):
        assert not clips.looks_chopped([(20, 50), (300, 340), (600, 640)], 900)
        assert not clips.looks_chopped([(10, 40)], 120)

    def test_chopped_answer_keeps_only_separate_best_moments(self):
        pieces = plan(idea(0, 30, 70), idea(30, 60, 92), idea(60, 90, 75), idea(90, 120, 80))
        kept = finalise_clips(pieces, 120, [], [])
        assert [(c["start"], c["score"]) for c in kept] == [(30, 92), (90, 80)]
        assert not clips.looks_chopped([(c["start"], c["end"]) for c in kept], 120)


class TestGeminiRequest:
    @pytest.mark.parametrize("seconds, fps", [(300, 1.0), (600, 1.0), (900, 0.5), (1800, 0.25)])
    def test_fewer_frames_for_longer_videos(self, seconds, fps):
        assert clips.frames_per_second(seconds) == fps

    @pytest.mark.parametrize("url", [
        "https://www.youtube.com/watch?v=EmcJiDu9mJI&t=10", "https://youtu.be/EmcJiDu9mJI?si=x",
        "https://youtube.com/shorts/EmcJiDu9mJI", "https://m.youtube.com/live/EmcJiDu9mJI",
    ])
    def test_youtube_links_become_watch_urls(self, url):
        assert clips.youtube_watch_url(url) == "https://www.youtube.com/watch?v=EmcJiDu9mJI"

    @pytest.mark.parametrize("url", [
        "https://www.instagram.com/reel/abc/", "https://youtube.com.evil.example/watch?v=EmcJiDu9mJI",
        "upload://talk.mp4", "https://www.youtube.com/@channel",
    ])
    def test_other_links_are_not_youtube(self, url):
        assert clips.youtube_watch_url(url) == ""

    def test_prompt_gives_pauses_cuts_and_length(self):
        media = MediaInfo(duration_seconds=600, width=1280, height=720, has_audio=True, aspect_ratio="16:9")
        prompt = clips._clip_prompt(media, [(10.0, 11.0)], [42.04])
        assert "600-second" in prompt and "15-60 seconds" in prompt
        assert "Speech pauses (seconds): 10.5" in prompt and "Shot changes: 42.0" in prompt
        assert "recommended_count" in prompt and "Never name real people" in prompt
        assert "NEVER split the video into consecutive pieces" in prompt and "0 or 1 clip" not in prompt
        assert "NEW voice-over" in prompt and "Never repeat, translate or closely paraphrase" in prompt
        assert "te 1.4" in prompt and "IMPORTANT" not in prompt

    def test_short_video_and_correction(self):
        media = MediaInfo(duration_seconds=120, width=1280, height=720, has_audio=True, aspect_ratio="16:9")
        prompt = clips._clip_prompt(media, [], [], clips.CHOPPED_FIX)
        assert "0 or 1 clip is normal" in prompt and "IMPORTANT: Your previous answer split the video" in prompt

    def test_metadata_context_is_about_one_clip(self):
        clip = {**idea(10, 40).model_dump(), "start": 10.0, "end": 40.0, "title": "Secret spice"}
        context = clips.metadata_context(clip, plan())
        assert "30-second Short" in context and "Secret spice" in context and "Spoken language: Telugu" in context
        assert "New voice-over of the clip: ఈ రహస్య" in context
        silent = plan().model_copy(update={"spoken_language": "none"})
        assert "Spoken language" not in clips.metadata_context({**clip, "what_is_said": ""}, silent)


class FakeClient:
    def __init__(self, replies):
        self.replies, self.requests, self.deleted = list(replies), [], []
        self.models = SimpleNamespace(generate_content=self.generate)
        self.files = SimpleNamespace(delete=lambda name: self.deleted.append(name))

    def generate(self, model, contents, config):
        self.requests.append(contents[0])
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return SimpleNamespace(parsed=reply)


@pytest.fixture
def gemini(monkeypatch):
    def install(replies):
        client = FakeClient(replies)
        monkeypatch.setattr(clips, "get_client", lambda: client)
        monkeypatch.setattr(clips, "call_gemini", lambda run, notify, **kw: run("gemini-3.6-flash"))
        monkeypatch.setattr(clips, "_upload_and_wait", lambda path, notify: SimpleNamespace(
            uri="https://files/abc", mime_type="video/mp4", name="files/abc"))
        return client
    return install


MEDIA = MediaInfo(duration_seconds=600, width=1280, height=720, has_audio=True, aspect_ratio="16:9")


class TestFindClips:
    def test_youtube_is_watched_from_youtube_without_upload(self, gemini):
        client = gemini([plan(idea(10, 40))])
        result, model = clips.find_clips(Path("x.mp4"), "https://youtu.be/EmcJiDu9mJI", MEDIA, [], [])
        part = client.requests[0]
        assert part.file_data.file_uri == "https://www.youtube.com/watch?v=EmcJiDu9mJI"
        assert part.video_metadata.fps == 1.0 and model == "gemini-3.6-flash" and not client.deleted
        assert result.clips[0].start_seconds == 10

    def test_youtube_refused_falls_back_to_upload(self, gemini):
        client = gemini([rv.GeminiCallError("cannot open"), plan()])
        clips.find_clips(Path("x.mp4"), "https://youtu.be/EmcJiDu9mJI", MEDIA, [], [])
        assert client.requests[1].file_data.file_uri == "https://files/abc" and client.deleted == ["files/abc"]

    def test_chopped_answer_is_sent_back_once(self, gemini):
        chopped = plan(idea(0, 150), idea(150, 300), idea(300, 450), idea(450, 600))
        client = gemini([chopped, plan(idea(100, 140))])
        result, _ = clips.find_clips(Path("x.mp4"), "https://youtu.be/EmcJiDu9mJI", MEDIA, [], [])
        assert len(client.requests) == 2 and [c.start_seconds for c in result.clips] == [100]

    def test_good_answer_is_not_asked_again(self, gemini):
        client = gemini([plan(idea(100, 140), idea(400, 430))])
        clips.find_clips(Path("x.mp4"), "https://youtu.be/EmcJiDu9mJI", MEDIA, [], [])
        assert len(client.requests) == 1

    def test_upload_is_deleted_even_when_gemini_fails(self, gemini):
        client = gemini([rv.GeminiCallError("busy")])
        with pytest.raises(rv.GeminiCallError):
            clips.find_clips(Path("x.mp4"), "upload://talk.mp4", MEDIA, [], [])
        assert client.deleted == ["files/abc"]


# ── ffmpeg: pauses and rendering ────────────────────────────────────────────

def test_speech_pauses_finds_the_silence(tmp_path):
    audio = tmp_path / "speech.wav"
    ffmpeg("-f", "lavfi", "-i", "sine=frequency=300:duration=2", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono",
           "-f", "lavfi", "-i", "sine=frequency=300:duration=2",
           "-filter_complex", "[1:a]atrim=duration=1[s];[0:a][s][2:a]concat=n=3:v=0:a=1", str(audio))
    (start, end), = clips.speech_pauses(audio)
    assert start == pytest.approx(2.0, abs=0.1) and end == pytest.approx(3.0, abs=0.1)


class TestRenderClip:
    @pytest.mark.parametrize("size, audio", [("1280x720", True), ("720x1280", False)])
    def test_vertical_1080x1920_at_the_clip_length(self, tmp_path, size, audio):
        source = make_video(tmp_path / "src.mp4", 8, size, audio)
        voice = make_tone(tmp_path / "voice.mp3", 2.5)  # shorter than the clip: padded with silence
        out = clips.render_clip(source, rv.probe_video(source), 2.0, 6.0, tmp_path / "clip.mp4", voice)
        result = rv.probe_video(out)
        assert (result.width, result.height) == (1080, 1920) and result.has_audio
        assert result.duration_seconds == pytest.approx(4.0, abs=0.15)

    def test_original_sound_is_never_used(self, tmp_path):
        source = make_video(tmp_path / "src.mp4", 8)  # loud 300 Hz tone in the source
        assert max_volume(source) > -20
        silent_voice = tmp_path / "silence.wav"
        ffmpeg("-f", "lavfi", "-i", "anullsrc=r=48000:cl=mono", "-t", "3", str(silent_voice))
        out = clips.render_clip(source, rv.probe_video(source), 1.0, 5.0, tmp_path / "clip.mp4", silent_voice)
        assert max_volume(out) < -60

    def test_music_plays_under_the_voice_and_loops(self, tmp_path):
        source = make_video(tmp_path / "src.mp4", 10, audio=False)
        silent_voice = tmp_path / "silence.wav"
        ffmpeg("-f", "lavfi", "-i", "anullsrc=r=48000:cl=mono", "-t", "1", str(silent_voice))
        music = make_tone(tmp_path / "music.mp3", 2, frequency=440)  # shorter than the clip: loops
        out = clips.render_clip(source, rv.probe_video(source), 0.0, 8.0, tmp_path / "clip.mp4", silent_voice,
                                music)
        assert rv.probe_video(out).duration_seconds == pytest.approx(8.0, abs=0.15)
        tail = tmp_path / "tail.wav"
        ffmpeg("-ss", "6", "-i", str(out), "-vn", str(tail))
        assert max_volume(tail) > -40  # music still playing after its own 2 s

    def test_broken_source_raises_a_clear_error(self, tmp_path):
        broken = tmp_path / "broken.mp4"
        broken.write_bytes(b"not a video")
        with pytest.raises(clips.ClipRenderError, match="could not render"):
            clips.render_clip(broken, MEDIA, 0, 5, tmp_path / "clip.mp4", make_tone(tmp_path / "v.mp3", 1))


class TestVoice:
    def fitted(self, monkeypatch, seconds):
        import voiceover as vo

        take = vo.Take(audio=b"mp3", duration=seconds, words=[])
        monkeypatch.setattr(vo, "fit_to_duration", lambda scenes, voice, target, notify=None: vo.FittedVoiceover(
            take=take, scenes=scenes, scene_starts=[0.0], video_duration=max(target, seconds)))

    def test_voice_that_fits_keeps_the_clip(self, monkeypatch):
        self.fitted(monkeypatch, 18.0)
        audio, clip = clips.fit_clip_voice({"start": 10.0, "end": 30.0, "narration": "x"}, "v", 600)
        assert audio == b"mp3" and clip["end"] == 30.0 and clip["notes"] == []

    def test_voice_that_runs_over_lengthens_the_clip_not_past_the_end(self, monkeypatch):
        self.fitted(monkeypatch, 24.0)
        _, clip = clips.fit_clip_voice({"start": 10.0, "end": 30.0, "narration": "x"}, "v", 600)
        assert clip["end"] == 34.3 and "lengthened" in clip["notes"][0]
        _, clip = clips.fit_clip_voice({"start": 580.0, "end": 598.0, "narration": "x"}, "v", 600)
        assert clip["end"] == 600

    def test_narration_language(self):
        assert clips.narration_language(plan()) == "te"
        assert clips.narration_language(plan().model_copy(update={"spoken_language": "none"})) == "en"

    def test_timeline_marks_each_clip(self):
        html = clips.timeline_html([{"start": 30.0, "end": 60.0, "recommended": True},
                                    {"start": 90.0, "end": 105.0, "recommended": False}], 120)
        assert "left:25.00%;width:25.00%" in html and "left:75.00%;width:12.50%" in html


# ── Whole page, end to end ──────────────────────────────────────────────────

WRAPPER = textwrap.dedent('''
    import dataclasses, subprocess, sys
    from pathlib import Path
    sys.path.insert(0, {repo!r})
    sys.path.insert(0, {tests!r})
    import streamlit as st
    import recreate_video as rv
    import clips
    import voiceover as vo
    from test_clips import idea, plan, make_video

    work = Path({work!r})
    config = dataclasses.replace(rv.CONFIG, api_key="dummy", work_dir=work)
    rv.CONFIG = clips.CONFIG = config
    SECONDS = {seconds}
    calls = st.session_state.setdefault("calls", [])

    def fake_download(url, destination, notify, **limits):
        calls.append(("download", limits["max_seconds"], limits["too_long_message"]))
        destination.mkdir(parents=True, exist_ok=True)
        video = make_video(destination / "video.mp4", SECONDS)
        return rv.DownloadedVideo(path=video, title="Village cooking", source_url=url, extractor="youtube")

    def fake_find(video, source_url, media, pauses, cuts, notify):
        calls.append(("find", source_url))
        return plan(idea(2, 20, score=90, title="Secret spice"), idea(35, 53, score=75, title="Taste test"),
                    idea(70, 88, score=55, title="Washing up")), "gemini-3.6-flash"

    def fake_metadata(context, kind, language, notify):
        calls.append(("metadata", context.splitlines()[1], language))
        title = context.splitlines()[1].removeprefix("Clip title idea: ")
        return rv.SocialMetadata(youtube_title=f"{{title}} #Shorts", youtube_description="D", youtube_tags="a, b",
                                 instagram_caption=f"{{title}} caption", instagram_hashtags="#a #b #c #d #e",
                                 model="gemini-2.5-flash", search_queries=[f"{{title}} recipe"])

    def fake_fit(scenes, voice, target, notify=None):
        audio = work / "voice.mp3"
        subprocess.run([rv.ffmpeg_exe(), "-y", "-f", "lavfi", "-i", "sine=frequency=900:duration=3", str(audio)],
                       capture_output=True, check=True)
        calls.append(("voice", scenes[0], voice, round(target, 2)))
        take = vo.Take(audio=audio.read_bytes(), duration=3.0, words=[])
        return vo.FittedVoiceover(take=take, scenes=scenes, scene_starts=[0.0], video_duration=target)

    clips.download_video = fake_download
    clips.find_clips = fake_find
    clips.generate_metadata = fake_metadata
    vo.languages = lambda: [("en", "English"), ("te", "Telugu")]
    vo.voices_for = lambda code, gender=None: [vo.Voice("te-IN-MohanNeural", "Male", "te-IN", "Telugu", "India")]
    vo.fit_to_duration = fake_fit
    clips.render_page(st)
''')


@pytest.fixture
def page(tmp_path, monkeypatch):
    import voiceover as vo

    for module, names in [(clips, ["download_video", "find_clips", "generate_metadata", "CONFIG"]),
                          (vo, ["languages", "voices_for", "fit_to_duration"]), (rv, ["CONFIG"])]:
        for name in names:
            monkeypatch.setattr(module, name, getattr(module, name))

    def build(seconds=100):
        wrapper = tmp_path / "app.py"
        wrapper.write_text(WRAPPER.format(repo=str(REPO), tests=str(REPO / "tests"), work=str(tmp_path / "work"),
                                          seconds=seconds), encoding="utf-8")
        return wrapper
    return build


def click(app, label):
    next(b for b in app.button if b.label == label).click()
    return app.run()


def advance(app, label, wrapper):
    """See tests/test_summary_short.py: continue in a fresh AppTest after a
    step change, carrying over the page's state."""
    app = click(app, label)
    assert not app.exception, app.exception
    fresh = AppTest.from_file(str(wrapper), default_timeout=300)
    for key in ("clips", "calls", "social_errors"):
        if key in app.session_state:
            fresh.session_state[key] = app.session_state[key]
    return fresh.run()


def test_whole_flow_from_link_to_clips_with_their_own_metadata(page):
    wrapper = page()
    app = AppTest.from_file(str(wrapper), default_timeout=300).run()
    assert not app.exception and app.button[0].disabled  # nothing to work on yet

    # 1. link -> clips found
    app.text_input[0].input("https://www.youtube.com/watch?v=EmcJiDu9mJI")
    app = advance(app, "Find clips", wrapper)
    state = app.session_state.clips
    assert state["stage"] == "pick", [e.value for e in app.error]
    assert ("download", clips.CLIPS_MAX_VIDEO_SECONDS, clips.TOO_LONG) in app.session_state.calls
    assert [c["title"] for c in state["clips"]] == ["Secret spice", "Taste test", "Washing up"]
    assert any("Gemini recommends 2 clip(s)" in m.value for m in app.markdown)
    takes = [c for c in app.checkbox if "score" in c.label]
    assert [c.value for c in takes] == [True, True, False]  # recommended ones are ticked

    assert any("orange = recommended" in m.value for m in app.markdown)  # the timeline
    assert any("Language: Telugu" in c.value for c in app.caption)

    # 2. edit the first clip's voice-over, render the two recommended
    narrations = [t for t in app.text_area if t.label == "Voice-over"]
    assert narrations[0].value == "ఈ రహస్య మసాలా రుచిని మార్చేస్తుంది"
    narrations[0].input("కొత్త వాయిస్ ఓవర్")
    app = app.run()
    app = advance(app, "Render 2 clip(s)", wrapper)
    state = app.session_state.clips
    assert state["stage"] == "done" and not app.exception
    rendered = list(state["rendered"].values())
    assert [c["title"] for c in rendered] == ["Secret spice", "Taste test"]
    voices = [call for call in app.session_state.calls if call[0] == "voice"]
    assert [v[1] for v in voices] == ["కొత్త వాయిస్ ఓవర్", "ఈ రహస్య మసాలా రుచిని మార్చేస్తుంది"]
    assert all(v[2] == "te-IN-MohanNeural" for v in voices)
    for clip in rendered:
        media = rv.probe_video(Path(clip["path"]))
        assert (media.width, media.height) == (1080, 1920)
        assert media.duration_seconds == pytest.approx(clip["end"] - clip["start"], abs=0.2)
        tail = Path(clip["path"]).with_suffix(".tail.wav")
        ffmpeg("-ss", "5", "-i", clip["path"], "-vn", str(tail))
        assert max_volume(tail) < -60  # after the 3 s voice: silence, not the source's 300 Hz tone
    import zipfile
    with zipfile.ZipFile(state["zip"]) as archive:
        assert sorted(archive.namelist()) == sorted(Path(c["path"]).name for c in rendered)

    # 3. every clip got its own metadata, in the spoken language
    metadata = [call for call in app.session_state.calls if call[0] == "metadata"]
    assert metadata == [("metadata", "Clip title idea: Secret spice", "Telugu"),
                        ("metadata", "Clip title idea: Taste test", "Telugu")]
    assert {s["youtube_title"] for s in state["social"].values()} == {"Secret spice #Shorts", "Taste test #Shorts"}
    assert [s.value for s in app.subheader][:2] == ["Clip 1: Secret spice", "Clip 2: Taste test"]

    # a rerun shows the stored metadata without asking Gemini again
    app = app.run()
    assert sum(1 for call in app.session_state.calls if call[0] == "metadata") == 2

    # Start over removes the job folder
    job = Path(state["job"])
    app = advance(app, "Start over", wrapper)
    assert app.session_state.clips == {"stage": "input"} and not job.exists()


def test_video_over_30_minutes_is_rejected(page, monkeypatch):
    wrapper = page()
    long_media = MediaInfo(duration_seconds=clips.CLIPS_MAX_VIDEO_SECONDS + 60, width=1280, height=720,
                           has_audio=True, aspect_ratio="16:9")
    text = wrapper.read_text(encoding="utf-8").replace(
        "clips.render_page(st)", f"clips.probe_video = lambda path: rv.MediaInfo(**{long_media.__dict__!r})\n"
                                 "clips.render_page(st)")
    wrapper.write_text(text, encoding="utf-8")
    monkeypatch.setattr(clips, "probe_video", clips.probe_video)
    app = AppTest.from_file(str(wrapper), default_timeout=300).run()
    app.text_input[0].input("https://www.youtube.com/watch?v=EmcJiDu9mJI")
    app = click(app, "Find clips")
    assert [e.value for e in app.error] == [clips.TOO_LONG]
    assert app.session_state.clips == {"stage": "input"}
    assert not any(call[0] == "find" for call in app.session_state.calls)  # no Gemini call
    assert not any((wrapper.parent / "work" / "clips").glob("*/*"))  # the job folder was removed


def test_too_long_message_matches_the_request():
    assert clips.TOO_LONG.startswith("This video cannot be analysed or made into clips due to its duration")
    assert "30 minutes" in clips.TOO_LONG
