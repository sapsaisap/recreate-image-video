# 🎬 Sri Recreate Content

An AI-powered content analysis and recreation studio built with Streamlit and Google Gemini. Transform any public video (Reel, Short, TikTok) or image into production-ready prompts, reference sheets, Canva workflows, trending audio recommendations, and viral social metadata.

---

## ✨ Features

### 🎬 Video & Reel Recreation
- **URL or Direct File Upload**: Paste links from YouTube Shorts, Instagram Reels, TikTok, or directly upload an `.mp4`/`.mov`/`.webm` file (ideal for private posts or geo-restricted content).
- **Native Video Understanding**: Uses Google Gemini to analyze visuals, speech, rhythm, lighting, camera angles, and soundtrack in a single multimodal call.
- **Timed Scene-by-Scene Prompts**: Condenses videos into timed 8s or 10s recreation prompts with hook-first pacing.
- **AI Reference Sheet**: Automatically suggests up to 3 image prompts (first frame, character, setting, object) for image-to-video generators (Veo, Runway, Luma, Kling, Sora).

### 🖼️ Image Post Recreation
- **Screenshot Upload or Direct Image URL**: Upload any screenshot from Instagram, Pinterest, or Twitter.
- **3 Generator Formats**:
  - 🍌 **Nano Banana / Nano Banana Pro**
  - 🌀 **FLUX / Midjourney / Leonardo**
  - 🖥️ **Stable Diffusion / ComfyUI** (with positive and negative prompts)
- **Text Overlay Extraction**: Detects on-screen typography, fonts, positions, and outputs a ready-to-use Canva / CapCut layout recreation workflow.

### 🎯 Recreation modes and accuracy
- **Faithful** (default): one prompt with the source's own scenes, order and events, compressed to 8/10 s. Every source scene must appear; if one is missing Gemini is asked once more.
- **Viral**: one prompt restructured hook-first for Reels / Shorts.
- **Full length**: the whole video as consecutive 8/10 s prompts, split at real cuts. Each part opens where the previous one ended and shares one character/style description.
- Gemini watches the **original file** at 2–4 fps (Gemini's default is 1) and gets the **cuts measured by ffmpeg** as scene boundaries.
- Every prompt shows an **accuracy score**, e.g. "Words kept 97% · Scenes covered 12/12", and lists anything left out.

### 📝 Summarise to Short
Turn a YouTube / Instagram video of up to 5 minutes into a **10, 15 or 20-second** vertical Short that tells its content in brief — images + voice-over, no video model.
1. **Video**: paste a link or upload, pick the length, click **Analyse**. Gemini watches the whole video (transcript, key points, the clearest visual moments) and writes a fact-checked script (Google Search on the free tier when available) in the source's language.
2. **Prompts**: one card per scene, each a **📷 real frame** (the sharpest frame from the source, for places, products, charts, demos), **🎨 AI from reference** (attach the shown frame in Google Flow) or **✨ full AI**. Change the type per scene; download all prompts + reference frames.
3. **Images**: real frames are filled in; upload your AI images per scene or all at once (filename order).
4. **Voice-over**: Microsoft Edge TTS (free) — language detected from the video, male/female voice, one natural take fitted to the length by adjusting speed (and shortening the script once if needed). You can upload your own voice-over instead.
5. **Video**: gentle zoom/pan per image, crossfades, voice at -14 LUFS, 1080×1920 — preview and download, with a credit line for the original creator.

Real footage is limited to about a third of the scenes and never used for frames with faces, on-screen text or watermarks (those become "AI from reference"). Uses 2 Gemini calls per video.

### 🎞️ Image to Video
- **Third option in the sidebar**: upload one image (JPG, PNG, WebP), pick a duration (1–60 s, default 10) and click **Generate video**; preview and download the MP4 on the same page.
- **Always 9:16, 1080×1920, 30 fps**, with no zoom, pan or fade. Images of another shape are fitted with black bars, so nothing is cropped.
- **Quality first**: lossless image preparation (EXIF rotation, colour profile converted to sRGB, transparency flattened), Lanczos scaling, H.264 High at CRF 16 tuned for still images, BT.709 colour, silent AAC track, fast-start MP4.
- Uses only ffmpeg — no Gemini key or quota needed.

### 🗣️ Dialogue That Matches the Source
- **Word-for-word transcript** in the spoken language and its own script (Telugu, Hindi, ...), with an English translation for you to read.
- **Exact lines in the video prompt**: Gemini only decides which line goes in which scene; the app inserts the original words, so dialogue is never translated, reworded or shortened. The prompt states the spoken language and voice.
- **Warnings** when a line had to be left out or a scene is too short for its dialogue. Set `DIALOGUE_SCRIPT=romanized` if your generator pronounces Latin letters better.

### 🎵 Audio & Social Metadata
- **Audio Matching**: Suggests matching music tracks, BPM feel, genres, and exact Instagram search queries.
- **Search-backed Social Pack**: keywords are researched with Google Search (free on `gemini-2.5-flash-lite`, 500/day) and the searches and sources are shown. YouTube tags are packed to ~480 of YouTube's own 500-character count (tags with spaces cost 2 extra for quotes — over 500 YouTube ignores them all), descriptions up to 5,000 bytes, Instagram captions up to 2,200 characters with 5 niche hashtags (no generic `#viral`/`#reels`).

### 🔒 Privacy & Automated Cleanup
- **Immediate Media Cleanup**: Videos and images are deleted as soon as Gemini finishes analyzing them. Only the generated prompt text is retained.
- **24-Hour Cache Sweep**: Old prompt files and cached data older than 24 hours are automatically purged on startup.
- **Privacy by Default**: `.env`, cookies, and temp working folders are fully excluded from version control.

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10 or 3.11
- [FFmpeg](https://ffmpeg.org/download.html) installed and accessible in your system PATH
- A free Google Gemini API key from [Google AI Studio](https://aistudio.google.com/)

### 2. Clone the Repository
```bash
git clone https://github.com/Sridhar2061uppari/sri-recreate-content.git
cd sri-recreate-content
```

### 3. Create and Activate Virtual Environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Open `.env` and add your Google Gemini API key:
```ini
GEMINI_API_KEY=your_actual_gemini_api_key_here
```

#### Free tier only
The app is built for a **free** Gemini key and accepts only free-tier models
(`gemini-3.6-flash`, `gemini-3.5-flash`, `gemini-3.5-flash-lite`, `gemini-3.1-flash-lite`,
`gemini-2.5-flash`, `gemini-2.5-flash-lite`).
- **Never link a billing account** to the key's Google Cloud project. A project without billing cannot be charged; when a limit is hit, Gemini returns an error and the app waits or switches model.
- Each step has its own model chain (`GEMINI_ANALYSIS_MODELS`, `GEMINI_TEXT_MODELS`, `GEMINI_METADATA_MODELS`, `GEMINI_STORY_MODELS`). A model whose daily quota is used up is skipped until the reset at midnight Pacific time; the sidebar shows today's usage.
- **Google Search** is free only on `gemini-2.5-flash` and `gemini-2.5-flash-lite` (500 searches a day, shared). YouTube / Instagram metadata is only written from real search results, so it uses just those two; if neither can search (busy, or the day's searches are used up), the app says so with a Retry button instead of guessing. The Summarise to Short script starts on them and falls back to the newest models without search, marked "not grounded". Analysis and prompts don't search and start on `gemini-3.6-flash`.
- Limits are per Google Cloud project, not per key — extra keys in the same project add no quota.
- On the free tier, Google may use what you send to improve its products. Don't upload confidential material.

*(Optional)* If you want automated Instagram Reel downloads for accounts requiring a login session, export your browser cookies with the Chrome extension **Get cookies.txt LOCALLY** and save the file as `cookies.txt` in the project root.

---

## 💻 Running the App

### Interactive Web Studio (Streamlit)
```bash
streamlit run recreate_video.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

### Command-Line Interface (CLI)
```bash
python recreate_video.py --url "https://www.youtube.com/shorts/YOUR_VIDEO_ID" --duration 8
```

---

## 📁 Project Structure

```
├── recreate_video.py              # Main unified Streamlit app & video recreation pipeline
├── recreate_image.py              # Image analysis, multi-model prompt generation & Canva workflows
├── image_to_video.py              # Still image -> 9:16 MP4 with ffmpeg (no Gemini)
├── summary_short.py               # Summarise to Short: analysis, script, 5-step page
├── voiceover.py                   # Edge TTS voice-over fitted to a target length
├── slideshow.py                   # Images + voice -> 9:16 MP4 with motion and crossfades
├── recreate_transition_video.py   # Transition video analysis pipeline
├── tests/                         # Unit tests for prompt builders and metadata schemas
├── requirements.txt               # Project dependencies
├── .env.example                   # Environment configuration template
└── .gitignore                     # Excludes secrets, cookies, and temporary media
```

---

## 🛡️ License
Distributed under the MIT License.