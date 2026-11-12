"""Accuracy work: measured cuts, frame rate, faithful / viral prompts, the
accuracy score, and full-length parts (Gemini faked; ffmpeg real)."""
import dataclasses
import subprocess
from types import SimpleNamespace

import pytest

import recreate_video as rv
from recreate_video import (
    FAITHFUL, FULL_LENGTH, VIRAL, AnalysisResult, FullLengthPlan, MediaInfo, PartPlan, PromptBeat,
    RecreationPlan, SourceScene, VideoAnalysis, accuracy_line, accuracy_report, part_windows,
)
from tests.test_dialogue import make_audio


def scene(start, end, action="something happens"):
    return SourceScene(start_seconds=start, end_seconds=end, description="room", action=action, setting="kitchen",
                       shot_type="medium", camera_movement="static", subject_movement="none",
                       transition_out="hard cut", on_screen_text="", audio="")


def analysis(scene_count=3, audio=None, duration=9.0):
    step = duration / scene_count
    return VideoAnalysis(
        content_type="vlog", visual_medium="live_action", summary="s", hook="the big reveal", story_arc="a",
        art_style="phone", color_palette="warm", lighting="day", mood="fun", pacing="cuts", camera_style="hand",
        characters=[], settings=["kitchen"],
        scenes=[scene(i * step, (i + 1) * step, f"action {i + 1}") for i in range(scene_count)],
        audio=audio or make_audio(language="none", lines=[]), why_it_works="w",
    )


def beat(scenes=(), lines=(), seconds=2.0, visual="kitchen"):
    return PromptBeat(label="B", duration_seconds=seconds, visual=visual, action="a", camera="locked-off medium",
                      audio="none", dialogue_line_ids=list(lines), source_scenes=list(scenes))


MEDIA = MediaInfo(duration_seconds=9.0, width=1080, height=1920, has_audio=True, aspect_ratio="9:16",
                  cuts=[3.0, 6.0])


class TestCutsAndFrameRate:
    def test_detects_real_cuts(self, tmp_path):
        video = tmp_path / "cuts.mp4"
        subprocess.run([
            rv.ffmpeg_exe(), "-y",
            "-f", "lavfi", "-i", "color=c=red:s=320x240:d=2:r=30",
            "-f", "lavfi", "-i", "color=c=blue:s=320x240:d=2:r=30",
            "-f", "lavfi", "-i", "color=c=green:s=320x240:d=2:r=30",
            "-filter_complex", "[0:v][1:v][2:v]concat=n=3:v=1:a=0", "-c:v", "libx264", "-pix_fmt", "yuv420p",
            str(video)], capture_output=True, check=True)
        cuts = rv.detect_scene_cuts(video)
        assert len(cuts) == 2 and cuts[0] == pytest.approx(2.0, abs=0.1) and cuts[1] == pytest.approx(4.0, abs=0.1)

    @pytest.mark.parametrize("seconds, fps", [(8, 4.0), (20, 4.0), (35, 3.0), (60, 2.0)])
    def test_frame_rate_by_length(self, seconds, fps):
        assert rv.analysis_fps(seconds) == fps

    def test_measured_cuts_are_in_the_analysis_prompt(self):
        prompt = rv._analysis_prompt(MEDIA)
        assert "scene boundaries at 0, 3, 6, 9 seconds" in prompt and "Never merge two intervals" in prompt

    def test_without_cuts_falls_back_to_guessing(self):
        assert "at each cut or major change" in rv._analysis_prompt(dataclasses.replace(MEDIA, cuts=[]))

    def test_old_media_without_cuts_still_loads(self):
        assert MediaInfo(**{"duration_seconds": 5, "width": 1, "height": 1, "has_audio": False,
                            "aspect_ratio": "9:16"}).cuts == []


class TestModes:
    def test_faithful_keeps_order_and_every_scene(self):
        prompt = rv._plan_prompt(analysis(), MEDIA, 8, 3, mode=FAITHFUL)
        assert "FAITHFUL STRUCTURE" in prompt and "Never move the ending to the front" in prompt
        assert "EVERY S number must appear" in prompt and "HOOK-FIRST" not in prompt

    def test_viral_is_hook_first(self):
        prompt = rv._plan_prompt(analysis(), MEDIA, 8, 3, mode=VIRAL)
        assert "HOOK-FIRST STRUCTURE" in prompt and "the big reveal" in prompt

    def test_one_clip_examples_and_single_room_rule_are_gone(self):
        prompt = rv._plan_prompt(analysis(), MEDIA, 8, 3, mode=FAITHFUL)
        assert "thruster" not in prompt and "EXACT same room" not in prompt
        assert "change the setting only where the source does" in prompt

    def test_faithful_render_has_no_hook_label(self):
        plan = RecreationPlan(title="t", logline="l", visual_style="v", characters=[], setting="s",
                              color_and_lighting="c", mood="m", overall_audio="", beats=[beat([1])], avoid=[],
                              reference_images=[])
        faithful = rv.render_prompt(plan, MEDIA, 8, mode=FAITHFUL)
        assert "FAITHFUL RECREATION" in faithful and "HOOK" not in faithful
        assert "HOOK" in rv.render_prompt(plan, MEDIA, 8, mode=VIRAL)


class TestAccuracyScore:
    def test_counts_words_and_scenes(self):
        source = analysis(audio=make_audio())  # 2 Telugu lines: 6 + 3 words
        report = accuracy_report([[beat([1], [1]), beat([3])]], source)
        assert report == {"words_kept": 6, "words_total": 9, "scenes_covered": 2, "scenes_total": 3,
                          "missing_scenes": [2]}
        assert accuracy_line(report) == "Words kept 67% (6/9) · Scenes covered 2/3"

    def test_no_speech_shows_scenes_only(self):
        assert accuracy_line(accuracy_report([[beat([1, 2, 3])]], analysis())) == "Scenes covered 3/3"


class TestPartWindows:
    def test_parts_end_on_nearby_cuts(self):
        assert part_windows(37.0, [7.2, 8.4, 15.1, 23.9, 30.6], 8) == [
            (0.0, 7.2), (7.2, 15.1), (15.1, 23.1), (23.1, 30.6), (30.6, 37.0)]

    def test_short_remainder_joins_the_last_part(self):
        assert part_windows(16.5, [], 8) == [(0.0, 8.0), (8.0, 16.5)]

    def test_source_shorter_than_one_part(self):
        assert part_windows(6.0, [], 8) == [(0.0, 6.0)]


def fake_client(monkeypatch, replies):
    requests = []

    def generate_content(model, contents, config):
        requests.append(contents[0])
        return SimpleNamespace(parsed=replies.pop(0))

    monkeypatch.setattr(rv, "get_client", lambda: SimpleNamespace(models=SimpleNamespace(generate_content=generate_content)))
    monkeypatch.setattr(rv, "_throttle", lambda: None)
    return requests


@pytest.fixture
def work(tmp_path, monkeypatch):
    monkeypatch.setattr(rv, "CONFIG", dataclasses.replace(rv.CONFIG, work_dir=tmp_path))
    monkeypatch.setattr(rv._STATE, "quota", None)
    monkeypatch.setattr(rv._STATE, "cooldowns", {})


def result_for(source: VideoAnalysis, media: MediaInfo = MEDIA) -> AnalysisResult:
    return AnalysisResult(video_sha="f" * 64, video_path="x", source_url="u", title="t", extractor="e",
                          media=dataclasses.asdict(media), analysis=source.model_dump(),
                          analysis_version=rv.ANALYSIS_VERSION)


def plan_with(*beats):
    return RecreationPlan(title="t", logline="l", visual_style="v", characters=[], setting="s",
                          color_and_lighting="c", mood="m", overall_audio="", beats=list(beats), avoid=[],
                          reference_images=[])


class TestFaithfulRetry:
    def test_missing_scene_triggers_one_retry(self, work, monkeypatch):
        requests = fake_client(monkeypatch, [plan_with(beat([1]), beat([3])), plan_with(beat([1]), beat([2, 3]))])
        generated = rv.generate_prompt(result_for(analysis()), 8, include_social=False, mode=FAITHFUL)
        assert len(requests) == 2 and "You left out source scenes S2" in requests[1]
        assert generated.accuracy["scenes_covered"] == 3 and generated.mode == FAITHFUL

    def test_viral_may_drop_scenes_without_retry(self, work, monkeypatch):
        requests = fake_client(monkeypatch, [plan_with(beat([3]))])
        generated = rv.generate_prompt(result_for(analysis()), 8, include_social=False, mode=VIRAL)
        assert len(requests) == 1 and generated.accuracy["scenes_covered"] == 1


class TestFullLength:
    def full(self, parts):
        return FullLengthPlan(title="t", logline="l", visual_style="phone video", characters=["boy in blue"],
                              setting="kitchen", color_and_lighting="warm", mood="fun", overall_audio="",
                              avoid=[], reference_images=[], parts=parts)

    def test_parts_cover_the_whole_video_with_continuity(self, work, monkeypatch):
        source = analysis(scene_count=4, audio=make_audio(), duration=16.0)
        media = MediaInfo(duration_seconds=16.0, width=1080, height=1920, has_audio=True, aspect_ratio="9:16",
                          cuts=[4.0, 8.0, 12.0])
        requests = fake_client(monkeypatch, [self.full([
            PartPlan(beats=[beat([1], [1], 4, "boy asks"), beat([2], [2], 4, "mother smiles at the stove")]),
            PartPlan(beats=[beat([3], [], 4, "pan to the pot"), beat([4], [], 4, "steam rises")]),
        ])])
        generated = rv.generate_prompt(result_for(source, media), 8, include_social=False, mode=FULL_LENGTH)
        assert "2 consecutive 8-second" in requests[0] and "Part 2: source 8-16s" in requests[0]
        assert len(generated.parts) == 2 and generated.mode == FULL_LENGTH
        assert generated.parts[0].startswith("PART 1 OF 2 -- recreates the source from 0s to 8s.")
        assert "matches that part's last frame exactly: mother smiles at the stove" in generated.parts[1]
        assert generated.accuracy["scenes_covered"] == 4 and generated.accuracy["words_kept"] == 9
        assert all("Create one 8-second" in part for part in generated.parts)

    def test_dialogue_from_another_window_is_not_used(self, work, monkeypatch):
        source = analysis(scene_count=4, audio=make_audio(), duration=16.0)  # both lines start in part 1
        media = MediaInfo(duration_seconds=16.0, width=1080, height=1920, has_audio=True, aspect_ratio="9:16")
        fake_client(monkeypatch, [self.full([
            PartPlan(beats=[beat([1, 2], [1], 8)]),
            PartPlan(beats=[beat([3, 4], [2], 8)]),  # L2 belongs to part 1
        ])])
        generated = rv.generate_prompt(result_for(source, media), 8, include_social=False, mode=FULL_LENGTH)
        assert any("Part 1: dialogue line(s) L2 left out" in w for w in generated.warnings)
        assert generated.accuracy["words_kept"] == 6

    def test_wrong_part_count_is_retried(self, work, monkeypatch):
        source = analysis(scene_count=2, duration=16.0)
        media = MediaInfo(duration_seconds=16.0, width=1080, height=1920, has_audio=False, aspect_ratio="9:16")
        monkeypatch.setattr(rv, "_wait_with_countdown", lambda *a, **k: None)
        requests = fake_client(monkeypatch, [
            self.full([PartPlan(beats=[beat([1, 2], [], 8)])]),
            self.full([PartPlan(beats=[beat([1], [], 8)]), PartPlan(beats=[beat([2], [], 8)])]),
        ])
        generated = rv.generate_prompt(result_for(source, media), 8, include_social=False, mode=FULL_LENGTH)
        assert len(requests) == 2 and len(generated.parts) == 2
