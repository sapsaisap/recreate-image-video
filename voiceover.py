"""
voiceover.py
============
Voice-over with Microsoft Edge's online text-to-speech (the `edge-tts`
package): free, no key, no Gemini quota, natural voices for Telugu, Hindi,
English and many more.

The whole script is spoken as ONE take, so the voice flows naturally across
scenes, and edge-tts reports when every word starts -- each scene's image
then appears exactly when its words begin. The take is fitted to the target
length by adjusting the speaking rate (and, if that is not enough, by asking
for a shorter script once).

edge-tts uses the service behind Edge's "Read aloud". It is unofficial:
Microsoft can rate-limit it (HTTP 403) or change it, which is why the UI also
accepts an uploaded voice-over (from_uploaded_audio).
"""
from __future__ import annotations

import asyncio
import logging
import math
import re
import subprocess
import tempfile
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Callable

from recreate_video import ffmpeg_exe

logger = logging.getLogger(__name__)
_LOADED_MTIME = Path(__file__).stat().st_mtime  # see recreate_video.drop_stale_app_modules

TICKS_PER_SECOND = 10_000_000  # edge-tts word offsets are in 100 ns units
MAX_SPEEDUP_PERCENT = 25  # faster than this starts to sound rushed
MAX_SLOWDOWN_PERCENT = 10
OVERRUN_TOLERANCE_SECONDS = 0.25
UNDERRUN_TOLERANCE_SECONDS = 1.5  # a little silence at the end is fine
PREFERRED_REGION = "IN"  # e.g. en-IN before en-US when both exist


class VoiceoverError(RuntimeError):
    """Text-to-speech failed (network, rate limit) or an audio file was unreadable."""


@dataclass(frozen=True)
class Voice:
    short_name: str  # what edge-tts needs, e.g. "te-IN-ShrutiNeural"
    gender: str  # "Female" / "Male"
    locale: str  # "te-IN"
    language_name: str  # "Telugu"
    region_name: str  # "India"

    @property
    def language_code(self) -> str:
        return self.locale.split("-")[0].lower()

    @property
    def label(self) -> str:
        person = self.short_name.split("-")[-1].removesuffix("Neural").removesuffix("Multilingual")
        return f"{person} · {self.gender.lower()} · {self.language_name} ({self.region_name})"


@dataclass
class Take:
    audio: bytes  # MP3
    duration: float
    words: list[tuple[float, float, str]]  # (start, end, word) in seconds
    rate_percent: int = 0


@dataclass
class FittedVoiceover:
    take: Take
    scenes: list[str]  # the narration actually spoken (may be shortened)
    scene_starts: list[float]  # when each scene's words begin
    video_duration: float  # target length, or the take's length if it still runs over
    shortened: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def scene_durations(self) -> list[float]:
        ends = [*self.scene_starts[1:], self.video_duration]
        return [round(end - start, 3) for start, end in zip(self.scene_starts, ends)]


# ============================================================================
# Voices
# ============================================================================

def _voice_from_raw(raw: dict) -> Voice:
    # FriendlyName: "Microsoft Shruti Online (Natural) - Telugu (India)"
    described = raw.get("FriendlyName", "").rsplit(" - ", 1)[-1]
    match = re.match(r"(.+?)\s*\((.+)\)\s*$", described)
    language, region = (match.group(1), match.group(2)) if match else (raw["Locale"], "")
    return Voice(short_name=raw["ShortName"], gender=raw.get("Gender", ""), locale=raw["Locale"],
                 language_name=language, region_name=region)


@lru_cache(maxsize=1)
def _all_voices() -> tuple[Voice, ...]:
    import edge_tts

    try:
        raw = asyncio.run(edge_tts.list_voices())
    except Exception as exc:
        raise VoiceoverError(f"Could not load the Microsoft Edge voice list: {exc}") from exc
    return tuple(_voice_from_raw(item) for item in raw)


def list_voices() -> list[Voice]:
    return list(_all_voices())


def languages() -> list[tuple[str, str]]:
    """(language code, name) for every language with at least one voice, by name."""
    names: dict[str, str] = {}
    for voice in list_voices():
        names.setdefault(voice.language_code, voice.language_name)
    return sorted(names.items(), key=lambda item: item[1])


def voices_for(language_code: str, gender: str | None = None) -> list[Voice]:
    """Voices for a language (e.g. "te"), optionally one gender, Indian accents
    first, then alphabetical."""
    code = language_code.lower()
    matches = [
        voice for voice in list_voices()
        if voice.language_code == code and (gender is None or voice.gender.lower() == gender.lower())
    ]
    return sorted(matches, key=lambda v: (not v.locale.endswith(f"-{PREFERRED_REGION}"), v.locale, v.short_name))


# ============================================================================
# Synthesis
# ============================================================================

def media_duration(path: Path) -> float:
    output = subprocess.run(
        [ffmpeg_exe(), "-hide_banner", "-i", str(path)],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    ).stderr
    match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", output)
    if not match:
        raise VoiceoverError("The audio file could not be read.")
    hours, minutes, seconds = match.groups()
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def _audio_duration(audio: bytes, suffix: str = ".mp3") -> float:
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / f"voice{suffix}"
        path.write_bytes(audio)
        return media_duration(path)


def synthesize(text: str, voice: str, rate_percent: int = 0) -> Take:
    """One take of `text` with per-word timings."""
    import edge_tts

    async def run() -> tuple[bytes, list[tuple[float, float, str]]]:
        communicate = edge_tts.Communicate(text, voice, rate=f"{rate_percent:+d}%", boundary="WordBoundary")
        audio = bytearray()
        words: list[tuple[float, float, str]] = []
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio += chunk["data"]
            elif chunk["type"] == "WordBoundary":
                start = chunk["offset"] / TICKS_PER_SECOND
                words.append((start, start + chunk["duration"] / TICKS_PER_SECOND, chunk["text"]))
        return bytes(audio), words

    try:
        audio, words = asyncio.run(run())
    except Exception as exc:
        raise VoiceoverError(
            f"Microsoft Edge text-to-speech failed ({type(exc).__name__}: {str(exc)[:160]}). "
            "Try again in a minute, or upload your own voice-over."
        ) from exc
    if not audio:
        raise VoiceoverError("Microsoft Edge text-to-speech returned no audio. Try again, or upload your own voice-over.")
    return Take(audio=audio, duration=_audio_duration(audio), words=words, rate_percent=rate_percent)


# ============================================================================
# Fitting and scene timing
# ============================================================================

def _spoken_words(text: str) -> list[str]:
    """Words as edge-tts counts them: whitespace-separated tokens that contain
    a letter, mark (Indic vowel signs) or digit -- not stray punctuation."""
    return [token for token in text.split() if any(unicodedata.category(ch)[0] in "LMN" for ch in token)]


def scene_start_times(scenes: list[str], words: list[tuple[float, float, str]], total: float) -> list[float]:
    """When each scene's first word is spoken. Falls back to splitting the
    take by text length when the word timings do not line up with the script."""
    counts = [len(_spoken_words(scene)) for scene in scenes]
    expected = sum(counts)
    if words and expected and abs(len(words) - expected) <= max(1, round(expected * 0.1)):
        starts, index = [], 0
        for count in counts:
            starts.append(words[min(index, len(words) - 1)][0])
            index += count
    else:
        lengths = [max(1, len(scene.strip())) for scene in scenes]
        starts, cursor = [], 0.0
        for length in lengths:
            starts.append(cursor)
            cursor += total * length / sum(lengths)
    starts[0] = 0.0
    # Keep starts strictly increasing, at least 0.5 s apart.
    for i in range(1, len(starts)):
        starts[i] = max(starts[i], starts[i - 1] + 0.5)
    return [round(start, 3) for start in starts]


def _rate_for(normal_duration: float, target: float) -> int:
    """Rate (+%) at which a take that lasts `normal_duration` at +0% fits `target`."""
    return math.ceil((normal_duration / target - 1) * 100) + 1


def fit_to_duration(
    scenes: list[str],
    voice: str,
    target_seconds: float,
    *,
    synth: Callable[[str, str, int], Take] = synthesize,
    shorten: Callable[[list[str], float], list[str]] | None = None,
    notify: Callable[[str], None] = lambda _: None,
) -> FittedVoiceover:
    """Speaks the scenes as one take and fits it to `target_seconds`:
    - too long: speed up (up to +MAX_SPEEDUP_PERCENT); if that is not enough
      and `shorten` is given, ask once for a shorter script and fit again;
    - clearly too short: slow down a little (up to -MAX_SLOWDOWN_PERCENT);
      any remaining gap just holds the last image.
    """
    notes: list[str] = []
    shortened = False
    notify("Recording the voice-over...")
    take = synth(" ".join(scenes), voice, 0)
    for _ in range(3):
        normal = take.duration * (1 + take.rate_percent / 100)  # length at +0%
        if take.duration > target_seconds + OVERRUN_TOLERANCE_SECONDS:
            rate = _rate_for(normal, target_seconds)
            if rate > MAX_SPEEDUP_PERCENT and shorten is not None and not shortened:
                notify("The script is too long for this length -- shortening it...")
                scenes = shorten(scenes, target_seconds / normal)
                shortened = True
                notes.append("The script was shortened to fit the length.")
                take = synth(" ".join(scenes), voice, 0)
                continue
            rate = min(rate, MAX_SPEEDUP_PERCENT)
            if rate == take.rate_percent:
                break
            notify(f"Fitting the voice-over to {target_seconds:g} s (speed {rate:+d}%)...")
            take = synth(" ".join(scenes), voice, rate)
        elif take.duration < target_seconds - UNDERRUN_TOLERANCE_SECONDS and take.rate_percent > -MAX_SLOWDOWN_PERCENT:
            rate = max(-MAX_SLOWDOWN_PERCENT, _rate_for(normal, target_seconds) - 1)
            if rate >= take.rate_percent:
                break
            notify(f"Fitting the voice-over to {target_seconds:g} s (speed {rate:+d}%)...")
            take = synth(" ".join(scenes), voice, rate)
        else:
            break

    if take.duration > target_seconds + OVERRUN_TOLERANCE_SECONDS:
        notes.append(
            f"The voice-over is {take.duration:.1f} s even at {take.rate_percent:+d}% speed, so the video "
            f"is {take.duration:.1f} s instead of {target_seconds:g} s."
        )
    if take.rate_percent:
        notes.append(f"Speaking speed adjusted by {take.rate_percent:+d}% to fit {target_seconds:g} s.")
    video_duration = round(max(target_seconds, take.duration), 3)
    return FittedVoiceover(
        take=take, scenes=scenes, scene_starts=scene_start_times(scenes, take.words, take.duration),
        video_duration=video_duration, shortened=shortened, notes=notes,
    )


def from_uploaded_audio(audio: bytes, suffix: str, scenes: list[str], target_seconds: float) -> FittedVoiceover:
    """The user's own voice-over. No word timings, so scenes are split by
    script length; the video lasts the target or the audio, whichever is longer."""
    duration = _audio_duration(audio, suffix)
    take = Take(audio=audio, duration=duration, words=[])
    notes = []
    if duration > target_seconds + OVERRUN_TOLERANCE_SECONDS:
        notes.append(f"The uploaded voice-over is {duration:.1f} s, so the video is {duration:.1f} s long.")
    return FittedVoiceover(
        take=take, scenes=scenes, scene_starts=scene_start_times(scenes, [], duration),
        video_duration=round(max(target_seconds, duration), 3), notes=notes,
    )
