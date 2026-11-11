"""Image to video: real ffmpeg renders, checked with ffmpeg itself. No Gemini."""
import dataclasses
import io
import subprocess

import pytest
from PIL import Image
from streamlit.testing.v1 import AppTest

import image_to_video as i2v
import recreate_video as rv
from image_to_video import ImageToVideoError, output_filename, prepare_image, render_video

RED, BLUE = (220, 20, 20), (20, 20, 220)


@pytest.fixture(autouse=True)
def work_dir(tmp_path, monkeypatch):
    cfg = dataclasses.replace(rv.CONFIG, work_dir=tmp_path)
    monkeypatch.setattr(i2v, "CONFIG", cfg)
    return tmp_path


def image_bytes(size=(1600, 900), color=RED, fmt="PNG", mode="RGB", **save_kwargs):
    buffer = io.BytesIO()
    Image.new(mode, size, color).save(buffer, format=fmt, **save_kwargs)
    return buffer.getvalue()


def probe(video: bytes, tmp_path):
    path = tmp_path / "probe.mp4"
    path.write_bytes(video)
    info = subprocess.run([rv.ffmpeg_exe(), "-hide_banner", "-i", str(path)], capture_output=True,
                          text=True, encoding="utf-8", errors="replace").stderr
    return rv.probe_video(path), info


def frame_at_middle(video: bytes, tmp_path) -> Image.Image:
    path, png = tmp_path / "frame.mp4", tmp_path / "frame.png"
    path.write_bytes(video)
    subprocess.run([rv.ffmpeg_exe(), "-y", "-ss", "1", "-i", str(path), "-frames:v", "1", str(png)],
                   capture_output=True, check=True)
    return Image.open(png).convert("RGB")


def close(pixel, color, tolerance=12):
    return all(abs(a - b) <= tolerance for a, b in zip(pixel, color))


class TestRender:
    @pytest.mark.parametrize("size", [(1600, 900), (900, 1600), (1000, 1000), (37, 21)])
    def test_always_1080x1920_for_the_chosen_duration(self, size, tmp_path):
        media, info = probe(render_video(image_bytes(size), 10, "photo.png"), tmp_path)
        assert (media.width, media.height) == (1080, 1920) and media.aspect_ratio == "9:16"
        assert abs(media.duration_seconds - 10) <= 0.05

    def test_codec_colour_and_audio(self, tmp_path):
        _, info = probe(render_video(image_bytes(), 3, "photo.png"), tmp_path)
        assert "Video: h264 (High)" in info
        assert "yuv420p(tv, bt709, progressive)" in info
        assert "30 fps" in info and "Audio: aac" in info

    def test_other_durations(self, tmp_path):
        media, _ = probe(render_video(image_bytes(), 1, "photo.png"), tmp_path)
        assert abs(media.duration_seconds - 1) <= 0.05

    def test_landscape_image_gets_black_bars_not_cropped(self, tmp_path):
        frame = frame_at_middle(render_video(image_bytes((1600, 900), RED), 2, "wide.png"), tmp_path)
        assert close(frame.getpixel((540, 960)), RED)       # image in the middle
        assert close(frame.getpixel((540, 40)), (0, 0, 0))  # bar at the top
        assert close(frame.getpixel((540, 1880)), (0, 0, 0))
        assert close(frame.getpixel((5, 960)), RED)         # full width used: nothing cut off

    def test_colours_are_kept(self, tmp_path):
        frame = frame_at_middle(render_video(image_bytes((1080, 1920), BLUE), 2, "blue.png"), tmp_path)
        assert close(frame.getpixel((540, 960)), BLUE)

    def test_temp_folder_is_removed(self, work_dir):
        render_video(image_bytes(), 1, "photo.png")
        with pytest.raises(ImageToVideoError):
            render_video(b"not an image", 1, "broken.png")
        assert list((work_dir / "image_to_video").iterdir()) == []


class TestPrepareImage:
    def prepare(self, data, tmp_path, name="in.img"):
        source, out = tmp_path / name, tmp_path / "out.png"
        source.write_bytes(data)
        size = prepare_image(source, out)
        return size, Image.open(out)

    def test_exif_rotation_is_applied(self, tmp_path):
        exif = Image.Exif()
        exif[0x0112] = 6  # "rotate 90° clockwise to display"
        size, _ = self.prepare(image_bytes((400, 200), fmt="JPEG", exif=exif.tobytes()), tmp_path)
        assert size == (200, 400)

    def test_transparency_becomes_black(self, tmp_path):
        _, out = self.prepare(image_bytes((50, 50), (255, 255, 255, 0), mode="RGBA"), tmp_path)
        assert out.mode == "RGB" and out.getpixel((25, 25)) == (0, 0, 0)

    def test_animated_gif_uses_first_frame(self, tmp_path):
        buffer = io.BytesIO()
        frames = [Image.new("RGB", (40, 40), RED), Image.new("RGB", (40, 40), BLUE)]
        frames[0].save(buffer, format="GIF", save_all=True, append_images=frames[1:])
        _, out = self.prepare(buffer.getvalue(), tmp_path)
        assert close(out.getpixel((20, 20)), RED, tolerance=30)

    def test_embedded_srgb_profile_is_handled(self, tmp_path):
        from PIL import ImageCms
        profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
        _, out = self.prepare(image_bytes((40, 40), RED, fmt="PNG", icc_profile=profile), tmp_path)
        assert close(out.getpixel((20, 20)), RED)

    def test_broken_file(self, tmp_path):
        with pytest.raises(ImageToVideoError, match="not a readable image"):
            self.prepare(b"definitely not an image", tmp_path)

    def test_too_many_megapixels(self, tmp_path, monkeypatch):
        monkeypatch.setattr(i2v, "MAX_MEGAPIXELS", 1)
        with pytest.raises(ImageToVideoError, match="maximum is 1 megapixels"):
            self.prepare(image_bytes((2000, 1000)), tmp_path)


class TestInputs:
    @pytest.mark.parametrize("seconds", [0, 61])
    def test_duration_out_of_range(self, seconds):
        with pytest.raises(ImageToVideoError, match="between 1 and 60"):
            render_video(image_bytes(), seconds)

    @pytest.mark.parametrize("name, expected", [
        ("Holiday photo.JPG", "Holiday_photo_9x16_10s.mp4"),
        ("../../etc/passwd", "passwd_9x16_10s.mp4"),
        ("!!!.png", "image_9x16_10s.mp4"),
    ])
    def test_output_filename(self, name, expected):
        assert output_filename(name, 10) == expected


class TestPage:
    def test_mode_works_without_a_gemini_key(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        app = AppTest.from_file(str(rv.APP_DIR / "recreate_video.py"), default_timeout=60).run()
        app.sidebar.radio[0].set_value(rv.IMAGE_TO_VIDEO_MODE).run()
        assert not app.exception
        assert [t.value for t in app.title] == ["Image to video"]
        assert not any("GEMINI_API_KEY" in e.value for e in app.error)
        assert app.number_input[0].value == 10
        generate = next(b for b in app.button if b.label == "Generate video")
        assert generate.disabled  # nothing uploaded yet

    def test_generate_shows_preview_and_download(self, work_dir):
        # i2v.CONFIG already points at work_dir (autouse fixture, same process).
        def page(repo):
            import io, sys
            sys.path.insert(0, repo)
            import streamlit as st
            from PIL import Image
            import image_to_video as i2v

            buffer = io.BytesIO()
            Image.new("RGB", (1200, 800), (200, 30, 30)).save(buffer, format="PNG")

            class Upload:
                name = "My photo.png"

                def getvalue(self):
                    return buffer.getvalue()

            class PageStreamlit:
                """streamlit with a fake file_uploader for this page only. AppTest
                cannot drive a real uploader, and patching the shared streamlit
                module would leak into every later test."""

                def __getattr__(self, name):
                    return getattr(st, name)

                def file_uploader(self, *args, **kwargs):
                    return Upload()

            i2v.render_page(PageStreamlit())

        app = AppTest.from_function(page, args=(str(rv.APP_DIR),), default_timeout=120).run()
        app.number_input[0].set_value(3)
        next(b for b in app.button if b.label == "Generate video").click()
        app.run()
        assert not app.exception and not app.error
        assert [s.value for s in app.subheader] == ["Your video"]
        assert any("1080x1920 · 3 s · 30 fps" in c.value for c in app.caption)
        assert app.session_state.i2v_result["filename"] == "My_photo_9x16_3s.mp4"
        assert app.session_state.i2v_result["video"][4:8] == b"ftyp"  # an MP4 file
