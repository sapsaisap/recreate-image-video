"""Upscale to 1080: sizes, the AI model (tiled), the fallback, video, and the page."""
import dataclasses
import io
import subprocess

import numpy as np
import pytest
from PIL import Image
from streamlit.testing.v1 import AppTest

import recreate_video as rv
import upscale

needs_model = pytest.mark.skipif(upscale._session() is None, reason="onnxruntime or the model is not available")


def picture(width, height, mode="RGB"):
    """A detailed test picture (gradients + stripes), so the model has edges to work on."""
    y, x = np.mgrid[0:height, 0:width]
    rgb = np.stack([(x * 255 // max(1, width - 1)), (y * 255 // max(1, height - 1)),
                    ((x // 4 + y // 4) % 2) * 255], axis=-1).astype(np.uint8)
    image = Image.fromarray(rgb).convert(mode) if mode != "RGBA" else Image.fromarray(rgb).convert("RGBA")
    if mode == "RGBA":
        image.putalpha(Image.fromarray(((x < width // 2) * 255).astype(np.uint8)))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def ffmpeg(*args):
    subprocess.run([rv.ffmpeg_exe(), "-y", "-loglevel", "error", *args], capture_output=True, check=True)


class TestSize:
    @pytest.mark.parametrize("size, expected", [
        ((270, 480), (1080, 1920)), ((480, 270), (1920, 1080)), ((500, 500), (1080, 1080)),
        ((400, 500), (1080, 1350)), ((2160, 3840), (1080, 1920)), ((480, 854), (1080, 1920)),
        ((640, 360), (1920, 1080)), ((300, 700), (1080, 2520)), ((333, 520), (1080, 1686)),
    ])
    def test_short_side_1080_same_shape_even(self, size, expected):
        assert upscale.target_size(*size) == expected


class TestImage:
    @needs_model
    def test_small_image_is_ai_upscaled_to_1080(self):
        result = upscale.upscale_image(picture(270, 480))
        assert (result.method, result.width, result.height, result.suffix) == ("ai", 1080, 1920, ".jpg")
        with Image.open(io.BytesIO(result.data)) as image:
            assert image.size == (1080, 1920) and image.format == "JPEG"

    @needs_model
    def test_tiles_leave_no_seams(self):
        rgb = np.asarray(Image.open(io.BytesIO(picture(90, 70))))
        whole = upscale.ai_upscale_x4(rgb, tile=512).astype(int)
        tiled = upscale.ai_upscale_x4(rgb, tile=32).astype(int)
        assert whole.shape == (280, 360, 3) and np.abs(whole - tiled).max() <= 3

    def test_without_the_model_it_still_gives_1080(self, monkeypatch):
        monkeypatch.setattr(upscale, "_session", lambda: None)
        result = upscale.upscale_image(picture(300, 300))
        assert (result.method, result.width, result.height) == ("resize", 1080, 1080)

    def test_bigger_image_is_resized_down(self):
        result = upscale.upscale_image(picture(1440, 2560))
        assert (result.method, result.width, result.height) == ("downscale", 1080, 1920)

    def test_transparency_is_kept(self, monkeypatch):
        monkeypatch.setattr(upscale, "_session", lambda: None)  # speed: the alpha path is the same
        result = upscale.upscale_image(picture(200, 200, "RGBA"))
        with Image.open(io.BytesIO(result.data)) as image:
            assert result.suffix == ".png" and image.mode == "RGBA" and image.size == (1080, 1080)
            assert image.getpixel((50, 540))[3] == 255 and image.getpixel((1030, 540))[3] == 0

    def test_not_an_image(self):
        with pytest.raises(upscale.UpscaleError, match="not a readable image"):
            upscale.upscale_image(b"not an image")


class TestVideo:
    def test_low_quality_video_comes_back_at_1080_with_its_sound(self, tmp_path):
        source = tmp_path / "small.mp4"
        ffmpeg("-f", "lavfi", "-i", "testsrc2=size=480x854:rate=30:duration=2", "-f", "lavfi", "-i",
               "sine=frequency=300:duration=2", "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
               "-c:a", "aac", "-shortest", str(source))
        out = upscale.upscale_video(source, tmp_path / "out.mp4")
        media = rv.probe_video(out)
        assert (media.width, media.height) == (1080, 1920) and media.has_audio
        assert media.duration_seconds == pytest.approx(2.0, abs=0.15)

    def test_too_long_is_refused(self, tmp_path):
        media = rv.MediaInfo(duration_seconds=200, width=480, height=854, has_audio=True, aspect_ratio="9:16")
        with pytest.raises(upscale.UpscaleError, match="maximum is 3 minutes"):
            upscale.upscale_video(tmp_path / "x.mp4", tmp_path / "out.mp4", media)


class TestPage:
    def test_mode_is_in_the_sidebar_and_needs_no_gemini_key(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        app = AppTest.from_file(str(rv.APP_DIR / "recreate_video.py"), default_timeout=60).run()
        app.sidebar.radio[0].set_value(rv.UPSCALE_MODE).run()
        assert not app.exception and [t.value for t in app.title] == ["Upscale to 1080"]
        assert not any("GEMINI_API_KEY" in e.value for e in app.error)

    def test_upscale_shows_before_after_and_download(self, tmp_path):
        def page(repo, data):
            import sys
            sys.path.insert(0, repo)
            import streamlit as st
            import upscale as up

            class Upload:
                name = "old photo.png"

                def getvalue(self):
                    return data

            class PageStreamlit:
                """streamlit with a fake image uploader (AppTest cannot drive a real one)."""

                def __getattr__(self, name):
                    return getattr(st, name)

                def file_uploader(self, label, *args, **kwargs):
                    return Upload() if kwargs.get("key") == "up_image" else None

            up._session = lambda: None  # the page flow, not the model (tested above)
            up.render_page(PageStreamlit())

        app = AppTest.from_function(page, args=(str(rv.APP_DIR), picture(300, 400)), default_timeout=120).run()
        next(b for b in app.button if b.label == "Upscale image").click()
        app.run()
        assert not app.exception and not app.error
        assert any("300x400 → 1080x1440" in s.value for s in app.success)
        name, result = app.session_state.up_image_result
        assert name == "old photo.png" and (result.width, result.height) == (1080, 1440)
        assert upscale._name(name, result.suffix) == "old photo_1080.jpg"
