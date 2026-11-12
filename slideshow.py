"""
slideshow.py
============
Still images + one voice-over -> a 9:16 MP4 for Shorts / Reels.

Every scene gets gentle camera motion (eased zoom in / zoom out / pan) and
neighbouring scenes crossfade. Frames are drawn with Pillow at sub-pixel
precision and piped straight into ffmpeg, which avoids the visible jitter of
ffmpeg's own zoompan filter (it rounds every frame to whole pixels). The
voice is normalised to -14 LUFS, the loudness Shorts and Reels play at.

Also here: picking the sharpest real frame from the source video, and
fitting real frames / AI images onto the 9:16 canvas.
"""
from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

from recreate_video import ffmpeg_exe

logger = logging.getLogger(__name__)

OUTPUT_WIDTH, OUTPUT_HEIGHT = 1080, 1920
FPS = 30
CRF = 17
ZOOM = 0.10  # total zoom over one scene; 6% (the reference project) is barely visible
CROSSFADE_SECONDS = 0.3
SUPERSAMPLE = 1.25  # canvases are drawn larger, so zoomed-in frames stay sharp
MOTIONS = ("zoom_in", "pan_right", "zoom_out", "pan_left")
TARGET_LUFS, TRUE_PEAK_DB, LOUDNESS_RANGE = -14.0, -1.5, 11.0
MIN_CROP_WIDTH = 540  # a 9:16 crop narrower than this (in source pixels) looks soft
RENDER_TIMEOUT_SECONDS = 900
_WORKERS = max(1, min(8, (os.cpu_count() or 2) - 1))  # leave one core for ffmpeg's encoder
_BATCH = 24  # frames drawn ahead of the encoder (~150 MB at 1080x1920)

CANVAS_SIZE = (round(OUTPUT_WIDTH * SUPERSAMPLE), round(OUTPUT_HEIGHT * SUPERSAMPLE))


class RenderError(RuntimeError):
    """ffmpeg failed, or an input could not be used."""


# ============================================================================
# Canvases: fitting any image onto the 9:16 frame
# ============================================================================

def load_image(path: Path):
    """RGB image with EXIF rotation and colour profile applied (same rules as
    Image to Video, so phone photos and transparent PNGs come out right)."""
    from PIL import Image
    from image_to_video import prepare_image

    with tempfile.TemporaryDirectory() as folder:
        normalised = Path(folder) / "image.png"
        prepare_image(path, normalised)
        with Image.open(normalised) as image:
            return image.convert("RGB").copy()


def _focus_center(size: tuple[int, int], focus_box: list[int] | None) -> tuple[float, float]:
    width, height = size
    if focus_box and len(focus_box) == 4:
        y0, x0, y1, x1 = focus_box  # Gemini's box order, 0-1000
        return (x0 + x1) / 2000 * width, (y0 + y1) / 2000 * height
    return width / 2, height / 2


def fill_canvas(image, focus_box: list[int] | None = None):
    """Covers the whole canvas (cropping the overflow), centred on the focus box."""
    from PIL import Image

    target_w, target_h = CANVAS_SIZE
    scale = max(target_w / image.width, target_h / image.height)
    resized = image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS)
    cx, cy = _focus_center(resized.size, focus_box)
    left = min(max(cx - target_w / 2, 0), resized.width - target_w)
    top = min(max(cy - target_h / 2, 0), resized.height - target_h)
    return resized.crop((round(left), round(top), round(left) + target_w, round(top) + target_h))


def blurred_fit_canvas(image):
    """The whole image, centred, over a blurred and dimmed copy of itself --
    the usual way a landscape clip is shown in a vertical Short."""
    from PIL import Image, ImageEnhance, ImageFilter

    target_w, target_h = CANVAS_SIZE
    background = fill_canvas(image).filter(ImageFilter.GaussianBlur(radius=target_w * 0.03))
    background = ImageEnhance.Brightness(background).enhance(0.55)
    scale = min(target_w / image.width, target_h / image.height)
    foreground = image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS)
    background.paste(foreground, ((target_w - foreground.width) // 2, (target_h - foreground.height) // 2))
    return background


def crop_keeps_focus(size: tuple[int, int], focus_box: list[int] | None) -> bool:
    """A full-height 9:16 crop of this frame contains the whole focus box and
    is wide enough (in source pixels) to stay sharp."""
    width, height = size
    crop_width = height * OUTPUT_WIDTH / OUTPUT_HEIGHT
    if crop_width >= width:  # already as narrow as 9:16
        return True
    if not focus_box or len(focus_box) != 4 or crop_width < MIN_CROP_WIDTH:
        return False
    _, x0, _, x1 = focus_box
    return (x1 - x0) / 1000 * width <= crop_width * 0.95


def real_frame_canvas(image, focus_box: list[int] | None = None):
    """A frame from the source video: cropped to fill when the subject fits a
    9:16 crop, otherwise shown whole over a blurred background."""
    if crop_keeps_focus(image.size, focus_box):
        return fill_canvas(image, focus_box)
    return blurred_fit_canvas(image)


# ============================================================================
# Picking the sharpest real frame
# ============================================================================

def _sharpness(image) -> float:
    from PIL import ImageFilter, ImageStat

    gray = image.convert("L")
    gray.thumbnail((640, 640))
    return ImageStat.Stat(gray.filter(ImageFilter.FIND_EDGES)).var[0]


def extract_best_frame(video: Path, at_seconds: float, output: Path, *, window: float = 0.4,
                       samples: int = 5, duration: float | None = None) -> Path:
    """Saves the sharpest frame within +-window of at_seconds as PNG. Motion
    blur and crossfade frames score low, so a clean frame is chosen."""
    from PIL import Image

    end = (duration - 0.05) if duration else None
    times = [at_seconds - window + 2 * window * i / max(1, samples - 1) for i in range(samples)]
    times = sorted({round(min(max(t, 0.0), end) if end else max(t, 0.0), 3) for t in times})
    best, best_score = None, -1.0
    with tempfile.TemporaryDirectory() as folder:
        for index, moment in enumerate(times):
            frame_path = Path(folder) / f"frame{index}.png"
            subprocess.run(
                [ffmpeg_exe(), "-hide_banner", "-nostdin", "-y", "-ss", f"{moment:.3f}", "-i", str(video),
                 "-frames:v", "1", str(frame_path)],
                capture_output=True, timeout=60,
            )
            if not frame_path.is_file():
                continue
            with Image.open(frame_path) as frame:
                score = _sharpness(frame)
                if score > best_score:
                    best, best_score = frame.convert("RGB").copy(), score
    if best is None:
        raise RenderError(f"Could not read a frame at {at_seconds:.1f}s from the source video.")
    output.parent.mkdir(parents=True, exist_ok=True)
    best.save(output, format="PNG")
    return output


# ============================================================================
# Audio: loudness normalisation
# ============================================================================

def normalise_voice(audio: Path, output: Path) -> Path:
    """Two-pass EBU R128 loudness normalisation to TARGET_LUFS, written as
    48 kHz WAV so the final mux encodes the audio only once."""
    base = f"loudnorm=I={TARGET_LUFS}:TP={TRUE_PEAK_DB}:LRA={LOUDNESS_RANGE}"
    measured = subprocess.run(
        [ffmpeg_exe(), "-hide_banner", "-nostdin", "-i", str(audio), "-af", f"{base}:print_format=json",
         "-f", "null", "-"],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300,
    ).stderr
    match = re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", measured)
    audio_filter = "anull"
    if match:
        values = json.loads(match.group(0))
        if values.get("input_i") not in (None, "-inf") and float(values["input_i"]) > -70:
            audio_filter = (
                f"{base}:measured_I={values['input_i']}:measured_TP={values['input_tp']}"
                f":measured_LRA={values['input_lra']}:measured_thresh={values['input_thresh']}"
                f":offset={values['target_offset']}:linear=true"
            )
    result = subprocess.run(
        [ffmpeg_exe(), "-hide_banner", "-nostdin", "-y", "-i", str(audio), "-af", audio_filter,
         "-ar", "48000", "-ac", "2", "-c:a", "pcm_s16le", str(output)],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300,
    )
    if result.returncode != 0 or not output.is_file():
        raise RenderError(f"Could not process the voice-over: {result.stderr.strip()[-300:]}")
    return output


# ============================================================================
# Motion + crossfades -> MP4
# ============================================================================

def _ease(progress: float) -> float:
    progress = min(max(progress, 0.0), 1.0)
    return progress * progress * (3 - 2 * progress)


def draw_frame(canvas, motion: str, progress: float):
    """One output frame of `canvas` at `progress` (0-1) through its motion."""
    from PIL import Image

    canvas_w, canvas_h = canvas.size
    p = _ease(progress)
    if motion == "zoom_in":
        scale, shift = 1 + ZOOM * p, 0.0
    elif motion == "zoom_out":
        scale, shift = 1 + ZOOM * (1 - p), 0.0
    else:  # pans run at a fixed zoom so there is room to move
        scale = 1 + ZOOM * 0.8
        shift = (2 * p - 1) * (1 if motion == "pan_right" else -1)
    visible_w, visible_h = canvas_w / scale, canvas_h / scale
    center_x = canvas_w / 2 + shift * (canvas_w - visible_w) / 2 * 0.95
    center_y = canvas_h / 2
    # AFFINE maps each output pixel (x, y) to canvas (a*x + c, e*y + f).
    a, e = visible_w / OUTPUT_WIDTH, visible_h / OUTPUT_HEIGHT
    c, f = center_x - visible_w / 2, center_y - visible_h / 2
    return canvas.transform((OUTPUT_WIDTH, OUTPUT_HEIGHT), Image.Transform.AFFINE, (a, 0, c, 0, e, f),
                            resample=Image.Resampling.BICUBIC)


def _ffmpeg_command(audio: Path | None, total: float, output: Path) -> list[str]:
    audio_input = ["-i", str(audio)] if audio else ["-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=48000"]
    return [
        ffmpeg_exe(), "-hide_banner", "-nostdin", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{OUTPUT_WIDTH}x{OUTPUT_HEIGHT}", "-r", str(FPS), "-i", "-",
        *audio_input,
        "-map", "0:v", "-map", "1:a",
        "-vf", (
            "scale=out_color_matrix=bt709:out_range=tv:flags=accurate_rnd+full_chroma_int,format=yuv420p,"
            "setparams=colorspace=bt709:color_primaries=bt709:color_trc=bt709:range=tv"
        ),
        "-c:v", "libx264", "-preset", "medium", "-crf", str(CRF), "-profile:v", "high",
        "-pix_fmt", "yuv420p", "-g", str(FPS * 2),
        "-af", "apad", "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
        "-t", f"{total:.3f}",
        "-movflags", "+faststart",
        str(output),
    ]


def render_slideshow(
    canvases: list,
    durations: list[float],
    output: Path,
    voice: Path | None = None,
    progress: Callable[[float], None] = lambda _: None,
) -> Path:
    """canvases[i] (from fill_canvas / real_frame_canvas / blurred_fit_canvas)
    is shown for durations[i] seconds. Crossfades are centred on each cut, so
    the video lasts exactly sum(durations). `voice` (any audio file) is
    loudness-normalised and laid under it; silence if None."""
    if not canvases or len(canvases) != len(durations):
        raise RenderError("Each scene needs one image and one duration.")
    if any(d <= 0 for d in durations):
        raise RenderError("Every scene must last more than 0 seconds.")
    from PIL import Image

    total = sum(durations)
    starts = [sum(durations[:i]) for i in range(len(durations))]
    half = CROSSFADE_SECONDS / 2
    frame_count = round(total * FPS)
    output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as folder:
        audio = normalise_voice(voice, Path(folder) / "voice.wav") if voice else None
        process = subprocess.Popen(_ffmpeg_command(audio, total, output), stdin=subprocess.PIPE,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

        def scene_frame(index: int, t: float):
            # Motion spans the scene plus both crossfade halves, so it never stops mid-fade.
            span_start = starts[index] - (half if index else 0)
            span = durations[index] + (half if index else 0) + (half if index < len(durations) - 1 else 0)
            return draw_frame(canvases[index], MOTIONS[index % len(MOTIONS)], (t - span_start) / span)

        def frame_bytes(number: int) -> bytes:
            t = number / FPS
            index = max(i for i, start in enumerate(starts) if start <= t + 1e-9)
            frame = scene_frame(index, t)
            # Inside a crossfade window: blend with the neighbouring scene.
            if index > 0 and t < starts[index] + half:
                alpha = 0.5 + (t - starts[index]) / CROSSFADE_SECONDS
                frame = Image.blend(scene_frame(index - 1, t), frame, alpha)
            elif index < len(starts) - 1 and t > starts[index + 1] - half:
                alpha = 0.5 - (starts[index + 1] - t) / CROSSFADE_SECONDS
                frame = Image.blend(frame, scene_frame(index + 1, t), alpha)
            return frame.tobytes()

        try:
            # Pillow releases the GIL while drawing, so frames are drawn on
            # several threads; small batches keep memory bounded and order kept.
            with ThreadPoolExecutor(max_workers=_WORKERS) as pool:
                for batch_start in range(0, frame_count, _BATCH):
                    batch = range(batch_start, min(frame_count, batch_start + _BATCH))
                    for data in pool.map(frame_bytes, batch):
                        process.stdin.write(data)
                    progress(batch.stop / frame_count)
            process.stdin.close()
        except (BrokenPipeError, OSError):
            pass  # ffmpeg exited early; its error is read below
        try:
            stderr = process.communicate(timeout=RENDER_TIMEOUT_SECONDS)[1].decode("utf-8", "replace")
        except subprocess.TimeoutExpired as exc:
            process.kill()
            raise RenderError("Rendering took too long and was stopped.") from exc
    if process.returncode != 0 or not output.is_file() or output.stat().st_size == 0:
        logger.error(f"[SLIDESHOW] ffmpeg failed: {stderr[-2000:]}")
        output.unlink(missing_ok=True)
        raise RenderError(f"ffmpeg could not render the video: {stderr.strip()[-300:]}")
    progress(1.0)
    logger.info(f"[SLIDESHOW] {len(canvases)} scenes, {total:.2f}s -> {output.name}")
    return output
