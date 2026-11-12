"""Summarise to Short: planning rules, prompt contents, and the whole 5-step
page driven end to end (Gemini and Edge TTS faked; ffmpeg real)."""
import textwrap
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import recreate_video as rv
import summary_short as ss
from summary_short import (
    SourceAnalysis, StoryPlan, StoryScene, VisualMoment, assign_bulk_uploads, drop_missing_frames,
    finalise_plan, scene_range, story_prompt, word_budget,
)

REPO = Path(__file__).resolve().parent.parent


def moment(t, person=False, text=False, ui=False):
    return VisualMoment(timestamp_seconds=t, description=f"thing at {t}s", why_it_matters="shows it",
                        kind="object", has_burned_in_text=text, has_watermark_or_ui=ui,
                        has_identifiable_person=person, focus_box=[100, 400, 900, 600])


def analysis(moments=None, language="Telugu", code="te"):
    return SourceAnalysis(
        title="Healthy habits", spoken_language=language, language_code=code, topic="health",
        summary="Three habits.", key_points=["walk", "water", "sleep"],
        transcript=[{"start_seconds": 0, "text": "ఈ వీడియోలో మూడు విషయాలు"}],
        visual_style="bright vlog", visual_moments=moments or [moment(1.0), moment(2.0), moment(3.0)],
    )


def scene(kind="real_frame", moment_number=1, text="ఒకటి"):
    return StoryScene(narration=text, visual_type=kind, type_reason="Gemini's reason", moment=moment_number,
                      image_prompt="A person walking at sunrise, vertical 9:16")


def plan(*scenes):
    return StoryPlan(hook_title="3 habits", scenes=list(scenes), fact_checks=[])


class TestPlanningRules:
    @pytest.mark.parametrize("duration, expected", [(10, (3, 4)), (15, (4, 6)), (20, (5, 7))])
    def test_scene_range(self, duration, expected):
        assert scene_range(duration) == expected

    def test_word_budget_uses_language_speed(self):
        assert word_budget(15, "te") < word_budget(15, "en")
        assert word_budget(15, "te") == int(15 * 1.4 * 0.92)

    def test_face_text_and_watermark_frames_become_ai_from_reference(self):
        source = analysis([moment(1, person=True), moment(2, text=True), moment(3, ui=True)])
        fixed = finalise_plan(plan(scene(moment_number=1), scene(moment_number=2), scene(moment_number=3)), source, 10)
        assert [s.visual_type for s in fixed.scenes] == ["ai_from_reference"] * 3
        assert "face" in fixed.scenes[0].type_reason and "burned-in text" in fixed.scenes[1].type_reason

    def test_real_frames_are_capped(self):
        source = analysis([moment(1), moment(2), moment(3), moment(4)])
        fixed = finalise_plan(plan(*[scene(moment_number=i) for i in range(1, 5)]), source, 10)
        assert [s.visual_type for s in fixed.scenes].count("real_frame") == ss.max_real_frames(4) == 1
        assert "enough scenes already reuse real footage" in fixed.scenes[1].type_reason

    def test_invalid_or_repeated_moment_becomes_full_ai(self):
        fixed = finalise_plan(plan(scene("ai_from_reference", 9), scene("real_frame", 1), scene("real_frame", 1)),
                              analysis(), 10)
        assert [(s.visual_type, s.moment) for s in fixed.scenes] == [("ai", 0), ("real_frame", 1), ("ai", 0)]

    def test_extra_scenes_fold_into_the_last(self):
        scenes = [scene("ai", 0, f"line{i}") for i in range(6)]
        fixed = finalise_plan(plan(*scenes), analysis(), 10)  # at most 4 for 10 s
        assert len(fixed.scenes) == 4 and fixed.scenes[-1].narration == "line3 line4 line5"

    def test_missing_frame_becomes_full_ai(self):
        fixed = drop_missing_frames(plan(scene("real_frame", 1), scene("ai_from_reference", 2)), {2: Path("x")})
        assert [(s.visual_type, s.moment) for s in fixed.scenes] == [("ai", 0), ("ai_from_reference", 2)]

    def test_bulk_uploads_fill_ai_scenes_in_natural_order(self):
        assigned = assign_bulk_uploads(["scene10.png", "scene2.png", "scene1.png"], [0, 2, 5])
        assert assigned == {0: "scene1.png", 2: "scene2.png", 5: "scene10.png"}


class TestStoryPrompt:
    def test_rules_are_in_the_prompt(self):
        prompt = story_prompt(analysis(), 240, 15, search=True)
        assert "4-6 scenes" in prompt and "in Telugu" in prompt and f"About {word_budget(15, 'te')} words" in prompt
        assert "NO text, letters, numbers" in prompt and "split screens, collages" in prompt
        assert "M1 [1.0s]" in prompt and "Use Google Search" in prompt
        assert "Reply with ONLY one JSON object" in prompt

    def test_without_search_no_json_instruction(self):
        prompt = story_prompt(analysis(), 240, 15, search=False)
        assert "From your own knowledge" in prompt and "Reply with ONLY" not in prompt

    def test_flags_are_shown_for_moments(self):
        prompt = story_prompt(analysis([moment(1, person=True, text=True)]), 60, 10, search=False)
        assert "(text on screen, real person's face)" in prompt


# ── Whole page, end to end ──────────────────────────────────────────────────

WRAPPER = textwrap.dedent('''
    import dataclasses, subprocess, sys
    from pathlib import Path
    sys.path.insert(0, {repo!r})
    sys.path.insert(0, {tests!r})
    import streamlit as st
    import recreate_video as rv
    import summary_short as ss
    import voiceover as vo
    from test_summary_short import analysis, moment

    work = Path({work!r})
    config = dataclasses.replace(rv.CONFIG, api_key="dummy", work_dir=work)
    rv.CONFIG = ss.CONFIG = config

    def fake_acquire(url, upload, job, notify):
        video = job / "source.mp4"
        subprocess.run([rv.ffmpeg_exe(), "-y", "-f", "lavfi", "-i", "testsrc2=size=1280x720:rate=30:duration=4",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(video)], capture_output=True, check=True)
        return video, "Healthy habits", url

    def fake_story(source, seconds, duration, frames, notify, allow_text=False):
        plan = ss.StoryPlan(hook_title="3 habits in 10 seconds", fact_checks=[
            ss.FactCheck(claim="Walking helps", verdict="confirmed", note="WHO")], scenes=[
            ss.StoryScene(narration=f"line {{i}}", visual_type="real_frame", type_reason="clear picture",
                          moment=i, image_prompt="prompt") for i in (1, 2, 3)])
        return plan, ["walking benefits"], [{{"title": "who.int", "uri": "https://example.com"}}]

    def fake_fit(scenes, voice, target, notify=None, shorten=None):
        audio = work / "tone.mp3"
        subprocess.run([rv.ffmpeg_exe(), "-y", "-f", "lavfi", "-i", "sine=frequency=200:duration=3",
                        str(audio)], capture_output=True, check=True)
        take = vo.Take(audio=audio.read_bytes(), duration=3.0, words=[])
        return vo.FittedVoiceover(take=take, scenes=scenes, scene_starts=[0.0, 1.0, 2.0], video_duration=3.0)

    ss.acquire_source = fake_acquire
    ss.analyse_source = lambda path, media, notify: analysis([moment(0.5), moment(1.5), moment(2.5)])
    ss.write_story = fake_story
    vo.languages = lambda: [("en", "English"), ("te", "Telugu")]
    vo.voices_for = lambda code, gender=None: [vo.Voice("te-IN-MohanNeural", "Male", "te-IN", "Telugu", "India")]
    vo.fit_to_duration = fake_fit
    ss.render_page(st)
''')


def click(app, label):
    next(b for b in app.button if b.label == label).click()
    return app.run()


def advance(app, label, wrapper):
    """Clicks a button that moves to another step. AppTest keeps the previous
    step's widgets in its tree after st.rerun() (a real browser drops them)
    and then fails reading their state, so continue in a fresh AppTest that
    carries over the page's saved state."""
    app = click(app, label)
    assert not app.exception, app.exception
    fresh = AppTest.from_file(str(wrapper), default_timeout=180)
    fresh.session_state["ss"] = app.session_state.ss
    return fresh.run()


@pytest.fixture
def restore_patched_modules(monkeypatch):
    """The wrapper swaps functions on modules shared with every other test;
    registering them here makes pytest put the originals back afterwards."""
    import voiceover as vo

    for module, names in [(ss, ["acquire_source", "analyse_source", "write_story", "CONFIG"]),
                          (vo, ["languages", "voices_for", "fit_to_duration"]), (rv, ["CONFIG"])]:
        for name in names:
            monkeypatch.setattr(module, name, getattr(module, name))


def test_whole_flow_from_link_to_video(tmp_path, restore_patched_modules):
    wrapper = tmp_path / "app.py"
    wrapper.write_text(WRAPPER.format(repo=str(REPO), tests=str(REPO / "tests"), work=str(tmp_path / "work")),
                       encoding="utf-8")
    app = AppTest.from_file(str(wrapper), default_timeout=180).run()
    assert not app.exception

    # 1. input -> analyse
    app.text_input[0].input("https://www.youtube.com/watch?v=abc")
    app.radio[0].set_value(10)
    app = advance(app, "Analyse", wrapper)
    state = app.session_state.ss
    assert state["stage"] == "prompts" and len(state["frames"]) == 3 and state["duration"] == 10
    job = Path(state["job"])
    assert not list(job.glob("source.*"))  # the source video is not kept

    # 2. prompts -> images (all real frames: nothing to upload)
    assert [s.value for s in app.subheader] == ["3 habits in 10 seconds"]
    assert len(app.selectbox) == 3
    app = advance(app, "Next: upload images", wrapper)
    assert app.session_state.ss["stage"] == "images"
    assert not next(b for b in app.button if b.label == "Next: voice-over").disabled

    # 3. images -> voice
    app = advance(app, "Next: voice-over", wrapper)
    assert app.session_state.ss["stage"] == "voice"
    app = click(app, "Generate voice-over")
    assert not app.exception and app.session_state.ss["voice"]["video_duration"] == 3.0

    # 4. voice -> combine -> video
    app = advance(app, "Next: combine", wrapper)
    app = advance(app, "Combine images and voice-over", wrapper)
    assert not app.exception and not app.error
    video = app.session_state.ss["video"]
    assert video[4:8] == b"ftyp"
    info_path = tmp_path / "out.mp4"
    info_path.write_bytes(video)
    media = rv.probe_video(info_path)
    assert (media.width, media.height) == (1080, 1920) and media.duration_seconds == pytest.approx(3.0, abs=0.05)

    # Start over removes the job folder
    app = advance(app, "Start a new Short", wrapper)
    assert app.session_state.ss == {"stage": "input"} and not job.exists()


class TestOnScreenTextSwitch:
    def test_text_free_by_default(self):
        assert "NO text, letters, numbers" in story_prompt(analysis(), 120, 15, search=False)

    def test_switch_allows_quoted_source_text(self):
        prompt = story_prompt(analysis(), 120, 15, search=False, allow_text=True)
        assert "NO text, letters" not in prompt and "inside double quotes, word for word" in prompt


class TestSourceFramesUsed:
    def test_story_prompt_requires_moments_for_what_the_video_shows(self):
        prompt = story_prompt(analysis(), 160, 15, search=True)
        assert "MUST set moment to that moment's number" in prompt and "ONLY for an idea the video never shows" in prompt

    def test_moments_after_the_end_are_dropped(self):
        kept = ss.drop_moments_outside(analysis([moment(10.0), moment(216.0)]), 160.0)
        assert [m.timestamp_seconds for m in kept.visual_moments] == [10.0]

    def test_detects_a_plan_without_source_frames(self):
        assert not ss.uses_source_frames(plan(scene("ai", 0), scene("ai", 0)))
        assert ss.uses_source_frames(plan(scene("ai", 0), scene("ai_from_reference", 2)))
