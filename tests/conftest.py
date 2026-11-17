import json, sys
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from recreate_video import (
    MediaInfo, VideoAnalysis, AnalysisResult,
    RecreationPlan, PromptBeat, ReferenceImage,
)

CACHE_DIR = PROJECT_ROOT / "work" / "cache"

SHA_CHICKEN = "e0710bb8682339cb372613ecf57f85ae81822f0c718b62f1c829626d94cfac7c"
SHA_CATS    = "9d3bb393f4feee4b2224990d681675ba35e04a83998f4d8d443b5340e182adc1"
SHA_YT      = "c523678b1cb3b5302b74c16c7aface30d147901478454d3f6e0153837d2215e6"


def _load(sha):
    path = CACHE_DIR / (sha + ".json")
    if not path.is_file():
        pytest.skip(f"needs a cached analysis ({path.name}) that only exists on the machine that made it")
    return json.loads(path.read_text(encoding="utf-8"))


def _media(sha, seconds):
    """The cached video's media, or a plain vertical clip of the same kind:
    render_prompt only uses the length and shape."""
    if (CACHE_DIR / (sha + ".json")).is_file():
        return MediaInfo(**_load(sha)["media"])
    return MediaInfo(duration_seconds=seconds, width=1080, height=1920, has_audio=True, aspect_ratio="9:16")


@pytest.fixture(scope="session")
def chicken_cache():  return _load(SHA_CHICKEN)

@pytest.fixture(scope="session")
def cats_cache():  return _load(SHA_CATS)

@pytest.fixture(scope="session")
def yt_cache():  return _load(SHA_YT)

@pytest.fixture(scope="session")
def chicken_analysis(chicken_cache):
    return VideoAnalysis.model_validate(chicken_cache["analysis"])

@pytest.fixture(scope="session")
def cats_analysis(cats_cache):
    return VideoAnalysis.model_validate(cats_cache["analysis"])

@pytest.fixture(scope="session")
def yt_analysis(yt_cache):
    return VideoAnalysis.model_validate(yt_cache["analysis"])

@pytest.fixture(scope="session")
def chicken_media():  return _media(SHA_CHICKEN, 8.0)

@pytest.fixture(scope="session")
def cats_media():  return _media(SHA_CATS, 8.0)

@pytest.fixture(scope="session")
def yt_media(yt_cache):  return MediaInfo(**yt_cache["media"])

@pytest.fixture(scope="session")
def chicken_result(chicken_cache):  return AnalysisResult(**chicken_cache)

@pytest.fixture(scope="session")
def cats_result(cats_cache):  return AnalysisResult(**cats_cache)


def make_mock_plan(
    beat_labels=("HOOK", "PROCESS", "PAYOFF"),
    beat_durations=(1.5, 4.0, 2.5),
    with_refs=True,
):
    beats = [
        PromptBeat(
            label=lbl,
            duration_seconds=dur,
            visual="Close-up of the subject.",
            action="Subject moves into frame naturally.",
            camera="Locked-off overhead tripod, zero camera movement, no zoom, no pan.",
            audio="0.0s: impact sound.",
            on_screen_text="",
        )
        for lbl, dur in zip(beat_labels, beat_durations)
    ]
    refs = (
        [
            ReferenceImage(
                ref_id="REF_1",
                purpose="character",
                what_it_shows="Adult hands in black gloves",
                used_in_scenes=[1, 2, 3],
                image_prompt="Close-up of gloved hands on a plain background.",
            )
        ]
        if with_refs
        else []
    )
    return RecreationPlan(
        title="Test Video",
        logline="A satisfying cooking tutorial recreated for Instagram Reels.",
        visual_style="Clean overhead food photography, high-contrast studio lighting.",
        characters=["Adult hands in tight black nitrile gloves with forearm tattoo."],
        setting="Dark brown wooden butcher block countertop.",
        color_and_lighting="Bright even overhead lighting, vibrant orange-red and green.",
        mood="Efficient, satisfying, appetizing.",
        overall_audio="Prominent ASMR-style sound effects throughout.",
        beats=beats,
        avoid=["Inconsistent food texture", "Wrong number of fingers"],
        reference_images=refs,
    )
