"""
image_to_video.py
=================
Turns one still image into an MP4 for Reels / Shorts: the image stays on
screen, unchanged, for the chosen number of seconds. No zoom, pan or fade,
and no Gemini call -- only Pillow and ffmpeg.

The output is always 9:16 at 1080x1920. An image of another shape is fitted
inside the frame with black bars, so nothing is cropped.
"""
from __future__ import annotations

import hashlib
import io
import logging
import re
import shutil
import subprocess
import uuid
from pathlib import Path

from recreate_video import CONFIG, ffmpeg_exe

logger = logging.getLogger(__name__)

OUTPUT_WIDTH, OUTPUT_HEIGHT = 1080, 1920  # 9:16, the Reels / Shorts frame
FPS = 30
CRF = 16  # visually lossless; a still image costs almost nothing extra at this quality
MIN_SECONDS, MAX_SECONDS, DEFAULT_SECONDS = 1, 60, 10
MAX_MEGAPIXELS = 50
BACKGROUND = (0, 0, 0)  # bars, and what transparent pixels become
FFMPEG_TIMEOUT_SECONDS = 300
UPLOAD_TYPES = ["jpg", "jpeg", "png", "webp"]


class ImageToVideoError(ValueError):
    """A problem the user can fix (unreadable image, bad duration) or an ffmpeg failure."""


# ============================================================================
# Image preparation (Pillow)
# ============================================================================

def _to_srgb(image):
    """Phone photos often carry a wide-gamut colour profile (e.g. Display P3).
    ffmpeg ignores it, so colours would come out dull: convert to sRGB first."""
    icc = image.info.get("icc_profile")
    if not icc or image.mode not in ("RGB", "CMYK"):
        return image
    try:
        from PIL import ImageCms

        source = ImageCms.ImageCmsProfile(io.BytesIO(icc))
        return ImageCms.profileToProfile(image, source, ImageCms.createProfile("sRGB"), outputMode="RGB")
    except Exception as exc:  # broken profile: keep the pixels as they are
        logger.warning(f"[IMAGE2VIDEO] could not apply colour profile: {exc}")
        return image


def prepare_image(source: Path, destination: Path) -> tuple[int, int]:
    """Writes `source` as a plain sRGB PNG that ffmpeg reads exactly as the
    viewer saw it: EXIF rotation applied (ffmpeg ignores it), first frame of an
    animation, transparency flattened onto BACKGROUND. Returns (width, height)."""
    from PIL import Image, ImageOps, UnidentifiedImageError

    try:
        with Image.open(source) as image:
            width, height = image.size  # read from the header, before decoding
            if width * height > MAX_MEGAPIXELS * 1_000_000:
                raise ImageToVideoError(
                    f"This image is {width}x{height} ({width * height / 1e6:.0f} MP). "
                    f"The maximum is {MAX_MEGAPIXELS} megapixels."
                )
            image.seek(0)
            frame = _to_srgb(ImageOps.exif_transpose(image))
            if frame.mode in ("RGBA", "LA", "PA") or (frame.mode == "P" and "transparency" in frame.info):
                rgba = frame.convert("RGBA")
                flat = Image.new("RGB", rgba.size, BACKGROUND)
                flat.paste(rgba, mask=rgba.getchannel("A"))
                frame = flat
            else:
                frame = frame.convert("RGB")
            frame.save(destination, format="PNG")
            return frame.size
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError) as exc:
        raise ImageToVideoError("This file is not a readable image. Upload a JPG, PNG or WebP.") from exc


# ============================================================================
# Rendering (ffmpeg)
# ============================================================================

def build_ffmpeg_command(image: Path, output: Path, seconds: int, ffmpeg: str = "ffmpeg") -> list[str]:
    """Still image -> H.264 MP4 at OUTPUT_WIDTH x OUTPUT_HEIGHT.

    - Lanczos scaling to fit inside the frame, then black bars (no crop).
    - The RGB -> YUV conversion and the file's colour tags are both BT.709,
      so colours match the image (ffmpeg's default matrix shifts them).
    - yuv420p + faststart: plays on phones and in browsers, uploads anywhere.
    - A silent AAC track, because some editors and upload pages mishandle
      video without one.
    """
    video_filter = (
        f"scale={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}:force_original_aspect_ratio=decrease:force_divisible_by=2"
        ":flags=lanczos+accurate_rnd+full_chroma_int:out_color_matrix=bt709:out_range=tv,"
        "format=yuv420p,"
        f"pad={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}:(ow-iw)/2:(oh-ih)/2:color=black,"
        "setsar=1,"
        # Tag the frames themselves: newer ffmpeg takes colour tags from the
        # frames, so the -color_* options alone leave primaries "unknown".
        "setparams=colorspace=bt709:color_primaries=bt709:color_trc=bt709:range=tv"
    )
    return [
        ffmpeg, "-hide_banner", "-nostdin", "-y",
        "-loop", "1", "-framerate", str(FPS), "-i", str(image),
        "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=48000",
        "-map", "0:v", "-map", "1:a",
        "-vf", video_filter,
        "-c:v", "libx264", "-preset", "slow", "-tune", "stillimage", "-crf", str(CRF),
        "-profile:v", "high", "-pix_fmt", "yuv420p", "-g", str(FPS * 2),
        "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv",
        "-c:a", "aac", "-b:a", "128k",
        "-t", str(seconds),
        "-movflags", "+faststart",
        str(output),
    ]


def render_video(image_bytes: bytes, seconds: int, source_name: str = "image") -> bytes:
    """The MP4 for one image, as bytes. Works in its own temp folder under
    WORK_DIR, which is deleted afterwards whatever happens."""
    if not MIN_SECONDS <= seconds <= MAX_SECONDS:
        raise ImageToVideoError(f"Choose a duration between {MIN_SECONDS} and {MAX_SECONDS} seconds.")
    job = CONFIG.work_dir / "image_to_video" / uuid.uuid4().hex[:12]
    job.mkdir(parents=True, exist_ok=True)
    try:
        source = job / f"source{Path(source_name).suffix.lower() or '.img'}"
        source.write_bytes(image_bytes)
        prepared = job / "prepared.png"
        width, height = prepare_image(source, prepared)
        output = job / "output.mp4"
        command = build_ffmpeg_command(prepared, output, seconds, ffmpeg_exe())
        try:
            result = subprocess.run(
                command, capture_output=True, text=True, encoding="utf-8", errors="replace",
                timeout=FFMPEG_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired as exc:
            raise ImageToVideoError(f"ffmpeg took more than {FFMPEG_TIMEOUT_SECONDS}s and was stopped.") from exc
        if result.returncode != 0 or not output.is_file() or output.stat().st_size == 0:
            detail = " ".join(result.stderr.strip().splitlines()[-3:])
            logger.error(f"[IMAGE2VIDEO] ffmpeg failed ({result.returncode}): {result.stderr[-2000:]}")
            raise ImageToVideoError(f"ffmpeg could not render the video: {detail[:300]}")
        video = output.read_bytes()
        logger.info(
            f"[IMAGE2VIDEO] {width}x{height} image -> {OUTPUT_WIDTH}x{OUTPUT_HEIGHT} {seconds}s "
            f"({len(video) / 1048576:.1f} MB)"
        )
        return video
    finally:
        shutil.rmtree(job, ignore_errors=True)


def output_filename(source_name: str, seconds: int) -> str:
    stem = re.sub(r"[^\w-]+", "_", Path(source_name).stem).strip("_")[:60] or "image"
    return f"{stem}_9x16_{seconds}s.mp4"


# ============================================================================
# Streamlit page
# ============================================================================

def render_page(st) -> None:
    st.title("Image to video")
    st.caption(
        f"Upload one image and choose how long it stays on screen. You get a {OUTPUT_WIDTH}x{OUTPUT_HEIGHT} "
        f"(9:16) MP4 at {FPS} fps with no zoom, pan or fade. Images of another shape get black bars, "
        "so nothing is cropped. No Gemini quota is used."
    )
    uploaded = st.file_uploader("Upload image", type=UPLOAD_TYPES, key="i2v_upload")
    seconds = int(st.number_input(
        "Duration (seconds)", min_value=MIN_SECONDS, max_value=MAX_SECONDS,
        value=DEFAULT_SECONDS, step=1, key="i2v_seconds",
    ))
    image_bytes = uploaded.getvalue() if uploaded is not None else None
    image_sha = hashlib.sha256(image_bytes).hexdigest() if image_bytes else None
    if image_bytes:
        st.image(image_bytes, caption=uploaded.name, width=240)

    if st.button("Generate video", type="primary", disabled=image_bytes is None, key="i2v_generate"):
        previous = st.session_state.get("i2v_result")
        if not previous or previous["key"] != (image_sha, seconds):  # same image + duration: reuse
            with st.spinner(f"Rendering a {seconds}-second video..."):
                try:
                    video = render_video(image_bytes, seconds, source_name=uploaded.name)
                except (ImageToVideoError, RuntimeError) as exc:
                    st.error(str(exc))
                    return
            st.session_state["i2v_result"] = {
                "key": (image_sha, seconds),
                "video": video,
                "filename": output_filename(uploaded.name, seconds),
            }

    result = st.session_state.get("i2v_result")
    if not result or result["key"][0] != image_sha:  # only for the image that is uploaded now
        return
    st.divider()
    st.subheader("Your video")
    st.video(result["video"], format="video/mp4")
    st.caption(
        f"{OUTPUT_WIDTH}x{OUTPUT_HEIGHT} · {result['key'][1]} s · {FPS} fps · H.264 MP4 · "
        f"{len(result['video']) / 1048576:.1f} MB"
    )
    st.download_button(
        "Download MP4", data=result["video"], file_name=result["filename"], mime="video/mp4",
        key="i2v_download",
    )
