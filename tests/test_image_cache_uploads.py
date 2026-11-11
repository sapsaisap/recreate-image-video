"""Image cache, upload handling and the Midjourney aspect flag."""
import dataclasses
import hashlib
from types import SimpleNamespace

import pytest
from PIL import Image

import recreate_image as ri
import recreate_video as rv
from recreate_image import ImageAnalysis, SuggestedAudio, _valid_aspect, _with_aspect_flag


def make_analysis(aspect="1:1"):
    audio = SuggestedAudio(
        track_name="t", artist_name="a", instagram_search_query="q", mood_match="m", genre="g",
        bpm_feel="b", alternative_track_name="t2", alternative_artist_name="a2", alternative_search_query="q2",
    )
    return ImageAnalysis(
        summary="s", hook="h", why_it_works="w", content_type="aesthetic", visual_medium="photography",
        subject="cup", composition="centered", color_palette="warm", lighting="soft", background="table",
        has_text=False, text_overlays=[], mood="calm", aspect_ratio=aspect, suggested_audio=audio,
    )


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Isolated work dir, and analyze_image replaced by a counter."""
    cfg = dataclasses.replace(rv.CONFIG, work_dir=tmp_path)
    monkeypatch.setattr(rv, "CONFIG", cfg)
    monkeypatch.setattr(ri, "CONFIG", cfg)
    calls = []

    def fake_analyze(path, notify=None):
        calls.append(path.read_bytes())
        return make_analysis(aspect="1:1")

    monkeypatch.setattr(ri, "analyze_image", fake_analyze)
    return tmp_path, calls


def write_image(path, size=(800, 1000), color="red"):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)
    return path


def leftover_images(root):
    return [p for p in root.rglob("*") if p.is_file() and p.suffix in {".png", ".jpg"}]


class TestImageCache:
    def test_second_upload_of_same_image_uses_cache(self, env):
        root, calls = env
        first = ri.analyze_local_image(write_image(root / "in" / "a.png"), "a.png")
        second = ri.analyze_local_image(write_image(root / "in" / "b.png"), "b.png")
        assert len(calls) == 1
        assert not first.from_cache and second.from_cache
        assert second.image_sha == first.image_sha

    def test_different_image_is_analysed(self, env):
        root, calls = env
        ri.analyze_local_image(write_image(root / "in" / "a.png", color="red"), "a.png")
        ri.analyze_local_image(write_image(root / "in" / "b.png", color="blue"), "b.png")
        assert len(calls) == 2

    def test_image_deleted_after_success_and_cache_hit(self, env):
        root, _ = env
        ri.analyze_local_image(write_image(root / "in" / "a.png"), "a.png")
        ri.analyze_local_image(write_image(root / "in" / "b.png"), "b.png")
        assert leftover_images(root) == []

    def test_image_deleted_when_gemini_fails(self, env, monkeypatch):
        root, _ = env

        def boom(path, notify=None):
            raise rv.GeminiCallError("down")

        monkeypatch.setattr(ri, "analyze_image", boom)
        with pytest.raises(rv.GeminiCallError):
            ri.analyze_local_image(write_image(root / "in" / "a.png"), "a.png")
        assert leftover_images(root) == []

    def test_unreadable_cache_is_ignored(self, env):
        root, _ = env
        ri._image_cache_path("abc").parent.mkdir(parents=True, exist_ok=True)
        ri._image_cache_path("abc").write_text("{not json", encoding="utf-8")
        assert ri._load_image_cached("abc") is None


class TestAspectRatio:
    def test_downloaded_image_uses_real_pixels(self, env):
        root, _ = env
        result = ri._analyze_image_file(
            write_image(root / "in" / "a.png", size=(1080, 1350)),
            source_url="https://example.com/a.png", title="a", extractor="generic",
        )
        assert result.analysis["aspect_ratio"] == "4:5"

    def test_upload_keeps_geminis_reading(self, env):
        root, _ = env
        result = ri.analyze_local_image(write_image(root / "in" / "a.png", size=(1080, 2340)), "a.png")
        assert result.analysis["aspect_ratio"] == "1:1"  # the post inside the screenshot, not the phone

    @pytest.mark.parametrize("value, expected", [
        ("9:16", "9:16"), ("4:5", "4:5"), ("2:3", "3:4"), ("1080x1350", "4:5"), ("square", "1:1"), ("", "1:1"),
    ])
    def test_valid_aspect(self, value, expected):
        assert _valid_aspect(value) == expected

    def test_plan_prompt_has_real_ratio_not_placeholder(self):
        prompt = ri._plan_prompt_image(make_analysis(aspect="4:5"))
        assert "{aspect}" not in prompt and "--ar 4:5 --q 2 --style raw" in prompt

    @pytest.mark.parametrize("prompt, expected", [
        ("cat on a sofa --ar 16:9 --q 2", "cat on a sofa --q 2 --ar 4:5"),
        ("cat on a sofa", "cat on a sofa --ar 4:5"),
        ("cat --ar {aspect} --aspect 1:1 --style raw", "cat --style raw --ar 4:5"),
    ])
    def test_aspect_flag_is_corrected(self, prompt, expected):
        assert _with_aspect_flag(prompt, "4:5") == expected


class TestNewUpload:
    def upload(self, data: bytes, name="image.jpg"):
        return SimpleNamespace(name=name, getvalue=lambda: data)

    def test_same_bytes_only_once(self):
        state = {}
        assert rv._new_upload(state, "k", self.upload(b"one")) == b"one"
        assert rv._new_upload(state, "k", self.upload(b"one")) is None

    def test_new_file_with_same_name_is_new(self):
        state = {}
        rv._new_upload(state, "k", self.upload(b"one", "image.jpg"))
        assert rv._new_upload(state, "k", self.upload(b"two", "image.jpg")) == b"two"
        assert state["k"] == hashlib.sha256(b"two").hexdigest()

    def test_removing_the_file_allows_a_retry(self):
        state = {}
        rv._new_upload(state, "k", self.upload(b"one"))
        assert rv._new_upload(state, "k", None) is None
        assert rv._new_upload(state, "k", self.upload(b"one")) == b"one"


class TestLocalVideoCleanup:
    def test_deleted_when_video_is_too_long(self, tmp_path, monkeypatch):
        video = tmp_path / "upload.mp4"
        video.write_bytes(b"fake")
        monkeypatch.setattr(rv, "probe_video", lambda path: rv.MediaInfo(120.0, 1080, 1920, True, "9:16"))
        with pytest.raises(rv.VideoValidationError):
            rv.analyze_local_video(video)
        assert not video.exists()
