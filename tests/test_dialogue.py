"""Dialogue fidelity: transcript lines reach the video prompt word for word,
in the spoken language, and dropped or squeezed lines are reported."""
import dataclasses
import json
from types import SimpleNamespace

import pytest

import recreate_video as rv
from recreate_video import (
    AnalysisResult, AudioProfile, MediaInfo, PromptBeat, RecreationPlan, SourceScene,
    Utterance, VideoAnalysis, _clean_dialogue_ids, _dialogue_warnings, _missing_lines,
    _plan_prompt, render_prompt,
)

TELUGU_1 = "అమ్మా, ఈ రోజు ఏం వంట చేస్తున్నావు?"
TELUGU_2 = "నీకు ఇష్టమైన బిర్యానీ!"


def make_audio(language="Telugu", lines=None):
    if lines is None:
        lines = [
            Utterance(start_seconds=0.5, end_seconds=2.5, speaker="boy in blue t-shirt",
                      text=TELUGU_1, romanized="Ammaa, ee roju em vanta chestunnavu?",
                      english="Mom, what are you cooking today?"),
            Utterance(start_seconds=3.0, end_seconds=4.5, speaker="woman in red saree",
                      text=TELUGU_2, romanized="Neeku ishtamaina biryani!",
                      english="Your favourite biryani!"),
        ]
    return AudioProfile(
        has_speech=bool(lines), spoken_language=language, speech_summary="Mother and son talk.",
        voice_character="excited young boy, warm mother", music="none", sound_effects="sizzling",
        transcript=lines,
    )


def make_analysis(audio):
    scene = SourceScene(
        start_seconds=0, end_seconds=6, description="Kitchen", action="Cooking", setting="kitchen",
        shot_type="medium", camera_movement="static", subject_movement="stirring",
        transition_out="none", on_screen_text="", audio="speech and sizzling",
    )
    return VideoAnalysis(
        content_type="vlog", visual_medium="live_action", summary="A boy asks his mother what she is cooking.",
        hook="Steam rising", story_arc="question -> reveal", art_style="phone video", color_palette="warm",
        lighting="daylight", mood="happy", pacing="slow", camera_style="handheld", characters=[],
        settings=["kitchen"], scenes=[scene], audio=audio, why_it_works="relatable",
    )


MEDIA = MediaInfo(duration_seconds=6.0, width=1080, height=1920, has_audio=True, aspect_ratio="9:16")


def make_plan(dialogue=((), (1,), (2,)), durations=(1.5, 3.5, 3.0)):
    beats = [
        PromptBeat(label=label, duration_seconds=dur, visual="Kitchen.", action="Cooking.",
                   camera="Locked-off medium, zero camera movement.", audio="sizzling",
                   on_screen_text="", dialogue_line_ids=list(ids))
        for label, dur, ids in zip(("HOOK", "ASK", "REVEAL"), durations, dialogue)
    ]
    return RecreationPlan(
        title="t", logline="A boy asks his mother what is cooking.", visual_style="phone video",
        characters=["boy in blue t-shirt", "woman in red saree"], setting="kitchen",
        color_and_lighting="warm", mood="happy", overall_audio="", beats=beats, avoid=[],
        reference_images=[],
    )


class TestAnalysisPrompt:
    def test_asks_for_verbatim_original_language(self):
        prompt = rv._analysis_prompt(MEDIA)
        assert "word for word" in prompt
        assert "do NOT translate" in prompt
        assert "Paraphrase speech" not in prompt


class TestPlanPrompt:
    def test_lists_every_line_and_requires_all_when_it_fits(self):
        prompt = _plan_prompt(make_analysis(make_audio()), MEDIA, 8, 3)
        assert "DIALOGUE (spoken language: Telugu)" in prompt
        assert TELUGU_1 in prompt and TELUGU_2 in prompt
        assert "you MUST use every line" in prompt
        assert "paraphrase only" not in prompt
        assert "words/second" not in prompt

    def test_too_much_speech_asks_to_keep_key_lines(self):
        long_line = Utterance(start_seconds=0, end_seconds=12, speaker="narrator", text="x",
                              romanized="x", english="x")
        prompt = _plan_prompt(make_analysis(make_audio(lines=[long_line])), MEDIA, 8, 3)
        assert "longer than 8s" in prompt

    def test_no_dialogue_section_without_speech(self):
        prompt = _plan_prompt(make_analysis(make_audio(language="none", lines=[])), MEDIA, 8, 3)
        assert "=== DIALOGUE" not in prompt


class TestRenderPrompt:
    def test_lines_inserted_verbatim_in_telugu(self):
        prompt = render_prompt(make_plan(), MEDIA, 8, make_audio())
        assert f'Dialogue: boy in blue t-shirt says in Telugu: "{TELUGU_1}"' in prompt
        assert f'Dialogue: woman in red saree says in Telugu: "{TELUGU_2}"' in prompt
        assert "All dialogue is spoken in Telugu" in prompt
        assert "do not switch to English" in prompt
        assert "Voice: excited young boy, warm mother." in prompt
        assert "Any spoken language other than Telugu" in prompt

    def test_romanized_mode(self, monkeypatch):
        monkeypatch.setattr(rv, "CONFIG", dataclasses.replace(rv.CONFIG, dialogue_script="romanized"))
        prompt = render_prompt(make_plan(), MEDIA, 8, make_audio())
        assert '"Ammaa, ee roju em vanta chestunnavu?"' in prompt
        assert TELUGU_1 not in prompt

    def test_english_has_no_language_switch_rule(self):
        line = Utterance(start_seconds=0, end_seconds=1, speaker="man", text="Hello there",
                         romanized="Hello there", english="Hello there")
        prompt = render_prompt(make_plan(dialogue=((), (1,), ())), MEDIA, 8, make_audio("English", [line]))
        assert 'Dialogue: man says: "Hello there"' in prompt
        assert "do not switch to English" not in prompt

    def test_hook_with_dialogue_is_not_marked_sound_off(self):
        prompt = render_prompt(make_plan(dialogue=((1,), (2,), ())), MEDIA, 8, make_audio())
        scene1 = next(line for line in prompt.splitlines() if line.startswith("Scene 1"))
        assert "sound-OFF" not in scene1

    def test_without_audio_is_unchanged(self):
        prompt = render_prompt(make_plan(), MEDIA, 8)
        assert "Dialogue:" not in prompt and "SPOKEN LANGUAGE" not in prompt


class TestDialogueChecks:
    def test_clean_ids_drops_invalid_and_repeats(self):
        beats = _clean_dialogue_ids(make_plan(dialogue=((1, 7), (1, 2), (2,))).beats, 2)
        assert [b.dialogue_line_ids for b in beats] == [[1], [2], []]

    def test_missing_lines(self):
        assert _missing_lines(make_plan(dialogue=((), (1,), ())).beats, 2) == [2]

    def test_warns_about_dropped_line(self):
        warnings = _dialogue_warnings(make_plan(dialogue=((), (1,), ())).beats, make_audio().transcript, 8)
        assert any("Your favourite biryani!" in w for w in warnings)

    def test_warns_when_scene_is_too_short_for_its_lines(self):
        plan = make_plan(dialogue=((), (1, 2), ()), durations=(1.5, 2.0, 4.5))
        warnings = _dialogue_warnings(plan.beats, make_audio().transcript, 8)
        assert any(w.startswith("Scene 2 has 3.5s of dialogue in 2s") for w in warnings)

    def test_no_warnings_when_everything_fits(self):
        assert _dialogue_warnings(make_plan().beats, make_audio().transcript, 8) == []


class TestOldCache:
    def write(self, tmp_path, monkeypatch, audio):
        monkeypatch.setattr(rv, "CONFIG", dataclasses.replace(rv.CONFIG, work_dir=tmp_path))
        analysis = make_analysis(make_audio()).model_dump()
        analysis["audio"] = audio
        payload = {"video_sha": "abc", "video_path": "x.mp4", "source_url": "u", "title": "t",
                   "extractor": "e", "media": dataclasses.asdict(MEDIA), "analysis": analysis}
        rv._cache_path("abc").parent.mkdir(parents=True)
        rv._cache_path("abc").write_text(json.dumps(payload), encoding="utf-8")

    def test_old_cache_with_speech_is_reanalysed(self, tmp_path, monkeypatch):
        audio = make_audio().model_dump()
        del audio["transcript"]
        self.write(tmp_path, monkeypatch, audio)
        assert rv._load_cached("abc") is None

    def test_old_cache_without_speech_still_loads(self, tmp_path, monkeypatch):
        audio = make_audio(language="none", lines=[]).model_dump()
        del audio["transcript"]
        self.write(tmp_path, monkeypatch, audio)
        assert rv._load_cached("abc").from_cache is True


class TestGeneratePromptRetry:
    """generate_prompt end to end against a fake Gemini client."""

    def run(self, tmp_path, monkeypatch, plans):
        monkeypatch.setattr(rv, "CONFIG", dataclasses.replace(rv.CONFIG, work_dir=tmp_path))
        monkeypatch.setattr(rv._STATE, "quota", None)
        monkeypatch.setattr(rv, "_throttle", lambda: None)
        requests = []

        def generate_content(model, contents, config):
            if config.response_schema is not RecreationPlan:
                raise RuntimeError("metadata not under test")
            requests.append(contents[0])
            return SimpleNamespace(parsed=plans.pop(0))

        monkeypatch.setattr(rv, "get_client", lambda: SimpleNamespace(
            models=SimpleNamespace(generate_content=generate_content)))
        result = AnalysisResult(video_sha="f" * 64, video_path="x", source_url="u", title="t",
                                extractor="e", media=dataclasses.asdict(MEDIA),
                                analysis=make_analysis(make_audio()).model_dump())
        return rv.generate_prompt(result, 8), requests

    def test_retries_once_when_a_fitting_line_is_dropped(self, tmp_path, monkeypatch):
        generated, requests = self.run(tmp_path, monkeypatch, [
            make_plan(dialogue=((), (1,), ())), make_plan(),
        ])
        assert len(requests) == 2 and "left out dialogue lines L2" in requests[1]
        assert TELUGU_2 in generated.prompt
        assert generated.warnings == []

    def test_keeps_warning_when_retry_does_not_help(self, tmp_path, monkeypatch):
        generated, _ = self.run(tmp_path, monkeypatch, [
            make_plan(dialogue=((), (1,), ())), make_plan(dialogue=((), (1,), ())),
        ])
        assert any("left them out even after a retry" in w for w in generated.warnings)

    def test_no_retry_when_all_lines_used(self, tmp_path, monkeypatch):
        generated, requests = self.run(tmp_path, monkeypatch, [make_plan()])
        assert len(requests) == 1
        assert TELUGU_1 in generated.prompt and TELUGU_2 in generated.prompt


class TestVideoIsTextFree:
    def test_rendered_prompt_has_no_on_screen_text_line(self):
        prompt = render_prompt(make_plan(), MEDIA, 8, make_audio())
        assert 'On-screen text: "' not in prompt  # no per-scene text line
        assert "the video must be completely text-free" in prompt  # the AVOID rule stays

    def test_plan_prompt_forbids_text(self):
        prompt = _plan_prompt(make_analysis(make_audio()), MEDIA, 8, 3)
        assert "NO TEXT RULE" in prompt and "completely text-free" in prompt
        assert "set on_screen_text" not in prompt and "on_screen_text MUST" not in prompt
