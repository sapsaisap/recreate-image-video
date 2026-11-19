"""
upscale.py
==========
Upscale to 1080: a low-quality image or video comes back with its short
side at 1080 pixels (1080x1080, 1080x1350, 1080x1920, 1920x1080 ...), the
same shape, nothing cropped. No Gemini call.

- Images: Real-ESRGAN (realesr-general-x4v3, BSD-3-Clause, models/) run on
  the CPU with onnxruntime, in tiles so memory stays small; it recovers
  detail and removes JPEG blocks. Then a lanczos resize to the exact size.
  If the model cannot load, a lanczos resize with light sharpening is used.
- Videos: ffmpeg -- lanczos resize, light noise removal and light
  sharpening, the original sound kept (it is the user's own video). AI on
  every frame would take far too long on a CPU.
"""
from __future__ import annotations

import functools
import io
import logging
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path

from recreate_video import CONFIG, MediaInfo, ffmpeg_exe, probe_video

logger = logging.getLogger(__name__)
_LOADED_MTIME = Path(__file__).stat().st_mtime  # see recreate_video.drop_stale_app_modules

TARGET = 1080  # the short side of the result
MODEL_PATH = Path(__file__).parent / "models" / "realesr-general-x4v3.onnx"
MODEL_SCALE = 4
TILE, TILE_PAD = 256, 24  # model input tiles; 24 px of overlap leaves no visible seam (max 2/255)
STANDARD_LONG_SIDES = (1080, 1350, 1440, 1920)  # 1:1, 4:5, 3:4, 9:16 at a 1080 short side
MAX_MEGAPIXELS = 50  # images larger than this are refused
MAX_AI_MEGAPIXELS = 2.5  # bigger inputs (already near 1080) are resized without the model
MAX_VIDEO_SECONDS = 180
VIDEO_FILTER = "hqdn3d=1.5:1.5:6:6,unsharp=5:5:0.6:5:5:0.0"  # after the resize
IMAGE_TYPES = ["jpg", "jpeg", "png", "webp"]
VIDEO_TYPES = ["mp4", "mov", "webm", "m4v", "mkv"]


class UpscaleError(ValueError):
    """A problem the user can fix (unreadable file, too big, too long) or an ffmpeg failure."""


@dataclass
class UpscaledImage:
    data: bytes
    width: int
    height: int
    source_width: int
    source_height: int
    method: str  # "ai", "resize" (fallback) or "downscale" (it was bigger than 1080 already)
    suffix: str  # ".jpg" or ".png" (transparency kept)


def target_size(width: int, height: int, target: int = TARGET) -> tuple[int, int]:
    """Short side `target`, same shape, both sides even (video encoders need
    it). Within 1% of a standard shape (9:16, 4:5, 3:4, 1:1) the exact
    standard size is used: 480x854 gives 1080x1920, not 1080x1922."""
    long_side = max(width, height) * target / min(width, height)
    standard = [s * target // 1080 for s in STANDARD_LONG_SIDES]
    near = [s for s in standard if abs(long_side - s) <= 0.01 * s]
    long_side = near[0] if near else max(2, int(round(long_side / 2)) * 2)
    return (target, long_side) if width <= height else (long_side, target)


@functools.lru_cache(maxsize=1)
def _session():
    """The onnxruntime session, or None if the model or library is missing."""
    try:
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.log_severity_level = 3
        return ort.InferenceSession(str(MODEL_PATH), options, providers=["CPUExecutionProvider"])
    except Exception as exc:
        logger.warning(f"[UPSCALE] AI model not available, using a plain resize: {exc}")
        return None


def ai_upscale_x4(rgb, tile: int = TILE, pad: int = TILE_PAD):
    """HxWx3 uint8 RGB -> 4H x 4W x 3 uint8, through the model tile by tile."""
    import numpy as np

    session = _session()
    height, width = rgb.shape[:2]
    planar = rgb.astype(np.float32).transpose(2, 0, 1) / 255.0
    out = np.zeros((height * MODEL_SCALE, width * MODEL_SCALE, 3), dtype=np.uint8)
    for y0 in range(0, height, tile):
        for x0 in range(0, width, tile):
            y1, x1 = min(y0 + tile, height), min(x0 + tile, width)
            py0, px0, py1, px1 = max(0, y0 - pad), max(0, x0 - pad), min(height, y1 + pad), min(width, x1 + pad)
            result = session.run(None, {"image": planar[None, :, py0:py1, px0:px1]})[0][0]
            oy, ox = (y0 - py0) * MODEL_SCALE, (x0 - px0) * MODEL_SCALE
            part = result[:, oy:oy + (y1 - y0) * MODEL_SCALE, ox:ox + (x1 - x0) * MODEL_SCALE]
            out[y0 * MODEL_SCALE:y1 * MODEL_SCALE, x0 * MODEL_SCALE:x1 * MODEL_SCALE] = (
                np.clip(part, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8).transpose(1, 2, 0)
    return out


def upscale_image(data: bytes) -> UpscaledImage:
    """The image with its short side at 1080: AI-upscaled when smaller,
    resized down when bigger. EXIF rotation applied; transparency kept."""
    import numpy as np
    from PIL import Image, ImageFilter, ImageOps, UnidentifiedImageError

    try:
        with Image.open(io.BytesIO(data)) as opened:
            if opened.width * opened.height > MAX_MEGAPIXELS * 1_000_000:
                raise UpscaleError(f"This image is {opened.width}x{opened.height}. The maximum is "
                                   f"{MAX_MEGAPIXELS} megapixels.")
            opened.seek(0)
            image = ImageOps.exif_transpose(opened)
            image.load()
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError) as exc:
        raise UpscaleError("This file is not a readable image. Upload a JPG, PNG or WebP.") from exc

    alpha = None
    if image.mode in ("RGBA", "LA", "PA") or (image.mode == "P" and "transparency" in image.info):
        alpha = image.convert("RGBA").getchannel("A")
        if alpha.getextrema() == (255, 255):  # fully opaque: no need for a PNG
            alpha = None
    rgb = image.convert("RGB")
    size = target_size(*rgb.size)
    if size[0] <= rgb.width:
        method, result = "downscale", rgb.resize(size, Image.LANCZOS)
    elif rgb.width * rgb.height <= MAX_AI_MEGAPIXELS * 1_000_000 and _session() is not None:
        method = "ai"
        result = Image.fromarray(ai_upscale_x4(np.asarray(rgb))).resize(size, Image.LANCZOS)
    else:
        method = "resize"
        result = rgb.resize(size, Image.LANCZOS).filter(ImageFilter.UnsharpMask(radius=2, percent=60, threshold=2))
    buffer = io.BytesIO()
    if alpha is not None:
        result.putalpha(alpha.resize(size, Image.LANCZOS))
        result.save(buffer, format="PNG", optimize=True)
        suffix = ".png"
    else:
        result.save(buffer, format="JPEG", quality=95, subsampling=0)
        suffix = ".jpg"
    return UpscaledImage(buffer.getvalue(), size[0], size[1], rgb.width, rgb.height, method, suffix)


def upscale_video(source: Path, output: Path, media: MediaInfo | None = None) -> Path:
    """The video with its short side at 1080, same shape and length, sound kept."""
    media = media or probe_video(source)
    if media.duration_seconds > MAX_VIDEO_SECONDS:
        raise UpscaleError(f"This video is {media.duration_seconds / 60:.1f} minutes long. The maximum is "
                           f"{MAX_VIDEO_SECONDS // 60} minutes.")
    width, height = target_size(media.width, media.height)
    picture = (f"scale={width}:{height}:flags=lanczos+accurate_rnd+full_chroma_int,setsar=1"
               + (f",{VIDEO_FILTER}" if width > media.width else ""))
    command = [ffmpeg_exe(), "-y", "-hide_banner", "-loglevel", "error", "-i", str(source), "-map", "0:v:0",
               *(["-map", "0:a:0", "-c:a", "aac", "-b:a", "192k"] if media.has_audio else []),
               "-vf", picture, "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
               "-movflags", "+faststart", str(output)]
    try:
        result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace",
                                timeout=1800)
    except subprocess.TimeoutExpired as exc:
        raise UpscaleError("Upscaling took more than 30 minutes.") from exc
    if result.returncode != 0 or not output.is_file():
        raise UpscaleError(f"ffmpeg could not upscale the video: {result.stderr.strip()[-400:]}")
    return output


# ============================================================================
# Page
# ============================================================================

def _name(original: str, suffix: str) -> str:
    return f"{Path(original).stem or 'upscaled'}_1080{suffix}"


def render_page(st) -> None:
    st.title("Upscale to 1080")
    st.caption("Upload a low-quality image or video and get it back with its short side at 1080 pixels "
               "(1080x1920 for 9:16, 1080x1080 for square ...), the same shape, nothing cropped. "
               "No Gemini quota is used.")
    tab_image, tab_video = st.tabs(["🖼️ Image", "🎬 Video"])
    with tab_image:
        _image_tab(st)
    with tab_video:
        _video_tab(st)


def _image_tab(st) -> None:
    st.caption("AI upscaling (Real-ESRGAN, on the server's CPU): recovers detail and removes blur and JPEG "
               "blocks. Takes a few seconds.")
    uploaded = st.file_uploader("Upload image", type=IMAGE_TYPES, key="up_image")
    if uploaded is None:
        return
    if st.button("Upscale image", type="primary", key="up_image_go"):
        with st.spinner("Upscaling..."):
            try:
                st.session_state["up_image_result"] = (uploaded.name, upscale_image(uploaded.getvalue()))
            except UpscaleError as exc:
                st.error(str(exc))
                return
    stored = st.session_state.get("up_image_result")
    if not stored or stored[0] != uploaded.name:
        return
    result: UpscaledImage = stored[1]
    how = {"ai": "AI upscaled", "resize": "resized and sharpened (AI model not available)",
           "downscale": "already bigger than 1080, resized down"}[result.method]
    st.success(f"{result.source_width}x{result.source_height} → {result.width}x{result.height} · {how}")
    before, after = st.columns(2)
    before.image(uploaded.getvalue(), caption="Before", use_column_width=True)
    after.image(result.data, caption="After", use_column_width=True)
    st.download_button("Download image", result.data, file_name=_name(uploaded.name, result.suffix),
                       mime="image/png" if result.suffix == ".png" else "image/jpeg", key="up_image_dl")


def _video_tab(st) -> None:
    st.caption(f"Up to {MAX_VIDEO_SECONDS // 60} minutes. Sharp resize with light noise removal and sharpening; "
               "the sound is kept. It makes the video clean at 1080 but cannot add detail that is not there.")
    uploaded = st.file_uploader("Upload video", type=VIDEO_TYPES, key="up_video")
    if uploaded is None:
        return
    if st.button("Upscale video", type="primary", key="up_video_go"):
        job = CONFIG.work_dir / "upscale" / uuid.uuid4().hex[:12]
        job.mkdir(parents=True, exist_ok=True)
        source = job / f"source{Path(uploaded.name).suffix.lower() or '.mp4'}"
        source.write_bytes(uploaded.getvalue())
        output = job / "upscaled.mp4"
        with st.spinner("Upscaling the video..."):
            try:
                media = probe_video(source)
                upscale_video(source, output, media)
                st.session_state["up_video_result"] = (uploaded.name, output.read_bytes(),
                                                       (media.width, media.height), target_size(media.width,
                                                                                                media.height))
            except (UpscaleError, RuntimeError, ValueError) as exc:
                st.error(str(exc))
                return
            finally:
                source.unlink(missing_ok=True)
                output.unlink(missing_ok=True)
                job.rmdir()
    stored = st.session_state.get("up_video_result")
    if not stored or stored[0] != uploaded.name:
        return
    name, data, (w0, h0), (w1, h1) = stored
    st.success(f"{w0}x{h0} → {w1}x{h1} · {len(data) / 1048576:.1f} MB")
    st.video(data, format="video/mp4")
    st.download_button("Download video", data, file_name=_name(name, ".mp4"), mime="video/mp4", key="up_video_dl")
