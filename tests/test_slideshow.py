"""Slideshow renderer: real ffmpeg renders checked for length, format,
loudness, crossfades and motion; canvas fitting; sharp-frame picking."""
import re
import subprocess

import pytest
from PIL import Image, ImageDraw

import recreate_video as rv
import slideshow as ss

RED, BLUE = (220, 20, 20), (20, 20, 220)


def ffmpeg(*args):
    return subprocess.run([rv.ffmpeg_exe(), "-hide_banner", *args], capture_output=True, text=True,
                          encoding="utf-8", errors="replace")


def striped(color, size=(1600, 900)):
    image = Image.new("RGB", size, color)
    draw = ImageDraw.Draw(image)
    for x in range(0, size[0], 60):
        draw.line([(x, 0), (x, size[1])], fill="white", width=3)
    return image


def frame_at(video, seconds, tmp_path):
    png = tmp_path / f"f{seconds}.png"
    ffmpeg("-y", "-ss", str(seconds), "-i", str(video), "-frames:v", "1", str(png))
    return Image.open(png).convert("RGB")


def tone(path, seconds=3, volume_db=-30):
    ffmpeg("-y", "-f", "lavfi", "-i", f"sine=frequency=220:duration={seconds}", "-af", f"volume={volume_db}dB",
           str(path))
    return path


@pytest.fixture(scope="module")
def rendered(tmp_path_factory):
    folder = tmp_path_factory.mktemp("render")
    canvases = [ss.fill_canvas(striped(RED)), ss.fill_canvas(striped(BLUE))]
    out = ss.render_slideshow(canvases, [1.5, 1.5], folder / "out.mp4", tone(folder / "voice.mp3", 2, -30))
    return out, folder


class TestRender:
    def test_exact_length_and_format(self, rendered):
        video, _ = rendered
        info = ffmpeg("-i", str(video)).stderr
        assert "Duration: 00:00:03.00" in info
        assert "1080x1920" in info and "30 fps" in info and "yuv420p(tv, bt709, progressive)" in info
        assert "Audio: aac" in info

    def test_voice_is_normalised_to_minus_14_lufs(self, rendered):
        video, _ = rendered
        report = ffmpeg("-i", str(video), "-af", "ebur128=framelog=quiet", "-f", "null", "-").stderr
        loudness = float(re.findall(r"I:\s+(-?[\d.]+) LUFS", report)[-1])
        assert loudness == pytest.approx(ss.TARGET_LUFS, abs=1.0)

    def test_crossfade_blends_at_the_cut(self, rendered):
        video, folder = rendered
        red, blue, middle = (frame_at(video, t, folder).getpixel((540, 300)) for t in (0.5, 2.5, 1.5))
        assert red[0] > 150 and blue[2] > 150
        assert 60 < middle[0] < 180 and 60 < middle[2] < 180  # half red, half blue

    def test_camera_moves(self, rendered):
        video, folder = rendered
        assert frame_at(video, 0.1, folder).tobytes() != frame_at(video, 1.0, folder).tobytes()

    def test_silent_when_no_voice(self, tmp_path):
        video = ss.render_slideshow([ss.fill_canvas(striped(RED))], [1.0], tmp_path / "silent.mp4")
        assert "Duration: 00:00:01.00" in ffmpeg("-i", str(video)).stderr

    def test_rejects_mismatched_inputs(self, tmp_path):
        with pytest.raises(ss.RenderError):
            ss.render_slideshow([ss.fill_canvas(striped(RED))], [1.0, 2.0], tmp_path / "x.mp4")
        with pytest.raises(ss.RenderError):
            ss.render_slideshow([ss.fill_canvas(striped(RED))], [0], tmp_path / "x.mp4")


class TestCanvases:
    def test_fill_covers_the_canvas(self):
        assert ss.fill_canvas(striped(RED)).size == ss.CANVAS_SIZE

    def test_fill_centres_on_focus_box(self):
        image = Image.new("RGB", (1600, 900), BLUE)
        image.paste(Image.new("RGB", (100, 900), RED), (1100, 0))  # subject right of centre
        canvas = ss.fill_canvas(image, focus_box=[0, 690, 1000, 750])
        assert canvas.getpixel((canvas.width // 2, canvas.height // 2))[0] > 150

    def test_crop_stops_at_the_image_edge(self):
        image = Image.new("RGB", (1600, 900), BLUE)
        image.paste(Image.new("RGB", (100, 900), RED), (1500, 0))  # subject at the very edge
        canvas = ss.fill_canvas(image, focus_box=[0, 940, 1000, 1000])
        assert canvas.getpixel((canvas.width - 5, canvas.height // 2))[0] > 150  # no empty border

    def test_blurred_fit_keeps_the_whole_image(self):
        canvas = ss.blurred_fit_canvas(Image.new("RGB", (1600, 900), RED))
        assert canvas.size == ss.CANVAS_SIZE
        center = canvas.getpixel((canvas.width // 2, canvas.height // 2))
        top = canvas.getpixel((canvas.width // 2, 20))
        assert center[0] > 200 and top[0] < center[0]  # sharp image in the middle, dimmed blur around it

    @pytest.mark.parametrize("size, box, expected", [
        ((1080, 1920), None, True),                   # already 9:16
        ((1920, 1080), [100, 450, 900, 560], True),    # narrow subject, enough pixels
        ((1920, 1080), [100, 100, 900, 900], False),   # subject wider than a 9:16 crop
        ((1920, 1080), None, False),                   # no idea where the subject is
        ((1280, 720), [100, 450, 900, 560], False),    # 405 px wide crop would be soft
    ])
    def test_crop_keeps_focus(self, size, box, expected):
        assert ss.crop_keeps_focus(size, box) is expected


class TestBestFrame:
    def test_picks_a_sharp_frame_over_a_blurred_one(self, tmp_path):
        video = tmp_path / "src.mp4"
        ffmpeg("-y", "-f", "lavfi", "-i", "testsrc2=size=640x360:rate=30:duration=2",
               "-vf", "boxblur=enable='lt(t,1)':luma_radius=15:chroma_radius=10",
               "-c:v", "libx264", "-pix_fmt", "yuv420p", str(video))
        best = ss.extract_best_frame(video, 0.9, tmp_path / "best.png", duration=2.0)
        blurred = frame_at(video, 0.5, tmp_path)
        with Image.open(best) as chosen:
            assert ss._sharpness(chosen) > 2 * ss._sharpness(blurred)

    def test_unreadable_video(self, tmp_path):
        bad = tmp_path / "bad.mp4"
        bad.write_bytes(b"nope")
        with pytest.raises(ss.RenderError):
            ss.extract_best_frame(bad, 1.0, tmp_path / "x.png")
