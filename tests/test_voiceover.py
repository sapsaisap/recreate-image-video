"""Voice-over fitting and scene timing, with a fake text-to-speech (no network)."""
import subprocess

import pytest

import recreate_video as rv
import voiceover as vo
from voiceover import Take, fit_to_duration, scene_start_times


def fake_synth(words_per_second=2.0, calls=None):
    """Speaks at words_per_second at +0%; rate r speeds it up by (1 + r/100)."""
    def synth(text, voice, rate):
        if calls is not None:
            calls.append(rate)
        words = vo._spoken_words(text)
        step = 1 / (words_per_second * (1 + rate / 100))
        timings = [(i * step, (i + 1) * step, w) for i, w in enumerate(words)]
        return Take(audio=b"mp3", duration=len(words) * step, words=timings, rate_percent=rate)
    return synth


def words(n, word="పదం"):
    return " ".join([word] * n)


class TestFit:
    def test_fits_without_change(self):
        fitted = fit_to_duration([words(10), words(10)], "v", 10, synth=fake_synth())
        assert fitted.take.rate_percent == 0 and fitted.video_duration == 10 and fitted.notes == []

    def test_speeds_up_to_fit(self):
        calls = []
        fitted = fit_to_duration([words(12), words(12)], "v", 10, synth=fake_synth(calls=calls))  # 12 s at +0%
        assert calls == [0, 21] and fitted.take.duration <= 10.25
        assert fitted.video_duration == 10

    def test_speedup_is_capped_and_reported(self):
        fitted = fit_to_duration([words(20), words(20)], "v", 10, synth=fake_synth())  # 20 s at +0%
        assert fitted.take.rate_percent == vo.MAX_SPEEDUP_PERCENT
        assert fitted.video_duration == pytest.approx(16.0, abs=0.01)
        assert any("instead of 10 s" in note for note in fitted.notes)

    def test_shortens_once_when_speed_is_not_enough(self):
        requests = []

        def shorten(scenes, ratio):
            requests.append(round(ratio, 2))
            return [words(9), words(9)]

        fitted = fit_to_duration([words(20), words(20)], "v", 10, synth=fake_synth(), shorten=shorten)
        assert requests == [0.5] and fitted.shortened
        assert fitted.scenes == [words(9), words(9)] and fitted.video_duration == 10

    def test_slows_down_a_little_when_far_too_short(self):
        fitted = fit_to_duration([words(8), words(8)], "v", 10, synth=fake_synth())  # 8 s at +0%
        assert fitted.take.rate_percent == -vo.MAX_SLOWDOWN_PERCENT
        assert fitted.video_duration == 10  # the rest holds the last image


class TestSceneTiming:
    def test_starts_follow_word_timings(self):
        timings = [(i * 0.5, i * 0.5 + 0.4, "w") for i in range(10)]
        assert scene_start_times(["a b c", "d e e f", "g h i j"], timings, 5.0) == [0.0, 1.5, 3.5]

    def test_punctuation_is_not_a_word(self):
        assert vo._spoken_words("అమ్మా , ఈ రోజు ! ?") == ["అమ్మా", "ఈ", "రోజు"]
        assert vo._spoken_words("Hello , world - 2026") == ["Hello", "world", "2026"]

    def test_falls_back_to_text_length_when_timings_do_not_match(self):
        starts = scene_start_times(["aaaa", "aaaa"], [(0.0, 0.1, "x")] * 7, 8.0)
        assert starts == [0.0, 4.0]

    def test_starts_are_at_least_half_a_second_apart(self):
        timings = [(0.0, 0.1, "a"), (0.1, 0.2, "b"), (0.2, 0.3, "c")]
        assert scene_start_times(["a", "b", "c"], timings, 3.0) == [0.0, 0.5, 1.0]

    def test_scene_durations_end_at_video_length(self):
        fitted = fit_to_duration([words(4), words(6)], "v", 10, synth=fake_synth())
        assert sum(fitted.scene_durations) == pytest.approx(fitted.video_duration)


class TestVoices:
    RAW = [
        {"ShortName": "te-IN-ShrutiNeural", "Gender": "Female", "Locale": "te-IN",
         "FriendlyName": "Microsoft Shruti Online (Natural) - Telugu (India)"},
        {"ShortName": "te-IN-MohanNeural", "Gender": "Male", "Locale": "te-IN",
         "FriendlyName": "Microsoft Mohan Online (Natural) - Telugu (India)"},
        {"ShortName": "en-US-GuyNeural", "Gender": "Male", "Locale": "en-US",
         "FriendlyName": "Microsoft Guy Online (Natural) - English (United States)"},
        {"ShortName": "en-IN-PrabhatNeural", "Gender": "Male", "Locale": "en-IN",
         "FriendlyName": "Microsoft Prabhat Online (Natural) - English (India)"},
    ]

    @pytest.fixture(autouse=True)
    def fake_list(self, monkeypatch):
        monkeypatch.setattr(vo, "_all_voices", lambda: tuple(vo._voice_from_raw(r) for r in self.RAW))

    def test_filter_by_language_and_gender(self):
        assert [v.short_name for v in vo.voices_for("te", "female")] == ["te-IN-ShrutiNeural"]
        assert [v.short_name for v in vo.voices_for("te")] == ["te-IN-MohanNeural", "te-IN-ShrutiNeural"]

    def test_indian_accent_first(self):
        assert [v.short_name for v in vo.voices_for("en", "male")] == ["en-IN-PrabhatNeural", "en-US-GuyNeural"]

    def test_labels_and_languages(self):
        assert vo.voices_for("te", "male")[0].label == "Mohan · male · Telugu (India)"
        assert vo.languages() == [("en", "English"), ("te", "Telugu")]


class TestFailures:
    def test_tts_error_becomes_friendly_message(self, monkeypatch):
        import edge_tts

        class Broken:
            def __init__(self, *args, **kwargs):
                pass

            async def stream(self):
                raise RuntimeError("403, message='Invalid response status'")
                yield  # pragma: no cover

        monkeypatch.setattr(edge_tts, "Communicate", Broken)
        with pytest.raises(vo.VoiceoverError, match="upload your own voice-over"):
            vo.synthesize("hello", "en-US-GuyNeural")

    def test_uploaded_voiceover_sets_length(self, tmp_path):
        audio = tmp_path / "v.mp3"
        subprocess.run([rv.ffmpeg_exe(), "-y", "-f", "lavfi", "-i", "sine=frequency=300:duration=12",
                        "-c:a", "libmp3lame", str(audio)], capture_output=True, check=True)
        fitted = vo.from_uploaded_audio(audio.read_bytes(), ".mp3", ["aaaa", "aaaa"], 10)
        assert fitted.video_duration == pytest.approx(12.0, abs=0.1)
        assert fitted.scene_starts == [0.0, pytest.approx(6.0, abs=0.1)]
        assert "12.0 s long" in fitted.notes[0]
