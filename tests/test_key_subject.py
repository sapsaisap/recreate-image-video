"""Key subject accuracy: measured size and shape in the analysis and the
prompt, no repeated AVOID items, the original first frame, and the weak-model
warning (Gemini faked; ffmpeg real)."""
import dataclasses
import json
import subprocess

import recreate_video as rv
from recreate_video import FAITHFUL, VIRAL, HeroSubject, ReferenceImage, render_prompt
from tests.test_accuracy import MEDIA, analysis, beat, fake_client, plan_with, result_for, work  # noqa: F401

BED = HeroSubject(
    name="giant pink sofa-bed", why_it_matters="its absurd size",
    size_and_scale="about 10 m wide; the woman is about 1/12 of its width; fills the lower 65% of the frame",
    shape_and_parts="rounded rectangle with low padded walls on all four sides; steps at the front-left corner",
    not_present="no headboard, no visible legs",
)


def ref(purpose, prompt="a prompt"):
    return ReferenceImage(ref_id="REF_1", purpose=purpose, what_it_shows="x", used_in_scenes=[1], image_prompt=prompt)


class TestAnalysisSchema:
    def test_gemini_must_fill_the_new_fields(self):
        schema = rv._StrictVideoAnalysis.model_json_schema()
        assert "hero_subjects" in schema["required"]
        scene = schema["$defs"]["_StrictSourceScene"]
        assert {"camera_position", "composition", "end_state"} <= set(scene["required"])

    def test_older_analyses_still_load(self):
        old = analysis().model_dump()
        old.pop("hero_subjects")
        for scene in old["scenes"]:
            for key in ("camera_position", "composition", "end_state"):
                scene.pop(key)
        loaded = rv.VideoAnalysis.model_validate(old)
        assert loaded.hero_subjects == [] and loaded.scenes[0].camera_position == ""

    def test_analysis_prompt_asks_for_measurements(self):
        prompt = rv._analysis_prompt(MEDIA)
        assert "MEASURE it" in prompt and "not_present" in prompt and "camera_position" in prompt

    def test_strict_answer_becomes_a_normal_analysis(self, monkeypatch):
        strict = rv._StrictVideoAnalysis.model_validate({**analysis().model_dump(), "hero_subjects": [BED.model_dump()]})
        assert rv.VideoAnalysis.model_validate(strict.model_dump()).hero_subjects[0].name == "giant pink sofa-bed"


class TestPromptRules:
    def test_plan_prompt_demands_numbers_and_no_invented_parts(self):
        prompt = rv._plan_prompt(analysis(), MEDIA, 8, 3, mode=FAITHFUL)
        assert "never only 'giant'" in prompt and "Never add a part the analysis does not mention" in prompt
        assert "camera_position" in prompt and "final end_state" in prompt
        assert "MUST include these global items" not in prompt and "Do not repeat them" in prompt


class TestRenderedPrompt:
    def test_key_subject_is_written_out_with_its_numbers(self):
        text = render_prompt(plan_with(beat([1])), MEDIA, 8, mode=FAITHFUL, heroes=[BED])
        assert "KEY SUBJECT (keep its exact size and shape" in text
        assert "the woman is about 1/12 of its width" in text and "Not present (do not add): no headboard" in text
        assert "Changing the giant pink sofa-bed's size, proportions or shape" in text

    def test_no_key_subject_section_without_one(self):
        assert "KEY SUBJECT" not in render_prompt(plan_with(beat([1])), MEDIA, 8, mode=FAITHFUL)

    def test_avoid_items_are_not_repeated(self):
        plan = plan_with(beat([1])).model_copy(update={"avoid": [
            "On-screen text, captions, subtitles, or title cards of any kind -- video must be completely text-free",
            "Changing a scene's setting where the source does not -- each scene keeps the source's own setting",
            "Slow-motion unless the scene label explicitly says slow motion",
            "Letterboxing or pillarboxing; fill the full frame edge-to-edge",
            "A bed with a headboard",
            "a bed with a headboard.",
            "Using a steel pan instead of the iron kadai",
        ]})
        avoid = render_prompt(plan, MEDIA, 8, mode=FAITHFUL).split("AVOID\n", 1)[1].split("\n\n")[0].splitlines()
        assert len(avoid) == len({a.lower() for a in avoid})
        assert sum("slow-motion" in a.lower() for a in avoid) == 1
        assert sum("letterbox" in a.lower() for a in avoid) == 1
        assert sum("text-free" in a for a in avoid) == 1
        assert sum("headboard" in a for a in avoid) == 1
        assert "- Using a steel pan instead of the iron kadai" in avoid  # a cooking pan is not a camera pan

    def test_start_frame_and_object_images_carry_the_measurements(self):
        refs = rv._with_hero_details([ref("first_frame"), ref("character")], [BED])
        assert "about 10 m wide" in refs[0].image_prompt and "no headboard" in refs[0].image_prompt
        assert refs[1].image_prompt == "a prompt"

    def test_generate_prompt_uses_the_key_subject(self, work, monkeypatch):  # noqa: F811
        source = analysis().model_copy(update={"hero_subjects": [BED]})
        plan = plan_with(beat([1, 2, 3])).model_copy(update={"reference_images": [ref("first_frame")]})
        fake_client(monkeypatch, [plan])
        generated = rv.generate_prompt(result_for(source), 8, include_social=False, mode=FAITHFUL)
        assert "KEY SUBJECT" in generated.prompt and "1/12 of its width" in generated.references[0]["image_prompt"]
        assert generated.model


class TestFirstFrame:
    def test_saves_the_opening_frame_once(self, work, tmp_path):  # noqa: F811
        video = tmp_path / "v.mp4"
        subprocess.run([rv.ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                        "color=c=red:s=180x320:d=1:r=24", str(video)], check=True)
        path = rv.save_first_frame(video, "a" * 64)
        assert path.endswith("_first.jpg") and rv.Path(path).stat().st_size > 0
        assert rv.save_first_frame(video, "a" * 64) == path

    def test_missing_video_gives_no_frame(self, work, tmp_path):  # noqa: F811
        assert rv.save_first_frame(tmp_path / "gone.mp4", "b" * 64) == ""

    def test_shown_for_faithful_not_viral(self, tmp_path):
        from streamlit.testing.v1 import AppTest

        frame = tmp_path / "first.jpg"
        subprocess.run([rv.ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                        "color=c=blue:s=90x160", "-frames:v", "1", str(frame)], check=True)

        def page(repo, frame, mode):
            import sys
            sys.path.insert(0, repo)
            import streamlit as st
            import recreate_video as rv
            result = rv.AnalysisResult(video_sha="c" * 64, video_path="x", source_url="u", title="t",
                                       extractor="e", media={}, analysis={}, first_frame=frame)
            generated = rv.GeneratedPrompt(duration=8, prompt="p", plan={}, saved_to="t.txt", mode=mode)
            rv._show_first_frame(st, result, generated, "k")

        for mode, shown in [(FAITHFUL, True), (VIRAL, False)]:
            app = AppTest.from_function(page, args=(str(rv.APP_DIR), str(frame), mode), default_timeout=60).run()
            assert any("original's first frame" in m.value for m in app.markdown) is shown


class TestWeakModels:
    def test_note_names_the_lite_models(self):
        note = rv.weak_model_note("gemini-3.5-flash-lite", "gemini-3.6-flash")
        assert note.startswith("Made with gemini-3.5-flash-lite --") and "quota resets" in note
        assert rv.weak_model_note("gemini-3.6-flash", "gemini-2.5-flash") == ""

    def test_lite_analysis_is_not_cached(self, work):  # noqa: F811
        for model, cached in [("gemini-3.5-flash-lite", False), ("gemini-3.6-flash", True)]:
            result = dataclasses.replace(result_for(analysis()), video_sha=model[-1] * 64, analysis_model=model)
            rv._write_cache(result)
            assert rv._cache_path(result.video_sha).is_file() is cached
            if cached:
                saved = json.loads(rv._cache_path(result.video_sha).read_text(encoding="utf-8"))
                assert saved["analysis_model"] == model
