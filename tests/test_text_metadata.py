"""Metadata built around the text on the image / on screen (memes, quotes,
captions), in the text's language, with what is trending now (Gemini faked)."""
import dataclasses
import json

import recreate_image as ri
import recreate_video as rv
from tests.test_accuracy import analysis, plan_with, beat, result_for, work  # noqa: F401
from tests.test_image_recreation import (  # noqa: F401
    mock_analysis_no_text, mock_analysis_with_text, mock_audio, mock_overlay, mock_plan_with_text,
)


class TestContentTexts:
    def test_keeps_the_message_and_drops_the_rest(self):
        texts = ["When mom says 5 minutes", "@foodie_channel", "Follow me for more", "none", "",
                 "when MOM says 5 minutes", "TikTok watermark bottom-right", "Me at 2 AM:"]
        assert rv.content_texts(texts) == ["When mom says 5 minutes", "Me at 2 AM:"]

    def test_length_is_capped(self):
        assert rv.content_texts(["a" * 500, "b" * 500], limit=900) == ["a" * 500]


class TestImageMetadata:
    def test_text_and_its_meaning_lead_the_context(self, mock_analysis_with_text, mock_plan_with_text):
        meme = mock_analysis_with_text.model_copy(update={
            "text_meaning": "Jokes that future phones will save themselves when dropped.",
            "meme_format": "Phones in 2070",
            "text_overlays": [*mock_analysis_with_text.text_overlays,
                              ri.TextOverlay(text="@memes_daily", position="bottom-right", style="small",
                                             purpose="watermark", canva_instruction="")],
        })
        lines = ri.image_metadata_context(meme, mock_plan_with_text).splitlines()
        assert lines[1] == "Text on the image (the main message -- build the metadata around it): Phones in 2070:"
        assert lines[2].startswith("What the text means / the joke: Jokes that future phones")
        assert lines[3] == "Meme format / trend: Phones in 2070"
        assert not any("@memes_daily" in line for line in lines)

    def test_no_text_no_text_lines(self, mock_analysis_no_text, mock_plan_with_text):
        context = ri.image_metadata_context(mock_analysis_no_text, mock_plan_with_text)
        assert "Text on the image" not in context and "the joke" not in context

    def test_metadata_is_in_the_texts_language(self, mock_analysis_with_text, mock_analysis_no_text):
        telugu = mock_analysis_with_text.model_copy(update={"text_language": "Telugu"})
        assert ri.image_metadata_language(telugu) == "Telugu"
        assert ri.image_metadata_language(mock_analysis_with_text) == "English"  # language not given
        assert ri.image_metadata_language(mock_analysis_no_text.model_copy(update={"text_language": "Hindi"})) \
            == "English"  # no text on the image

    def test_generate_passes_context_and_language(self, mock_analysis_with_text, mock_plan_with_text, monkeypatch):
        seen = {}
        monkeypatch.setattr(ri, "generate_metadata", lambda context, kind, language, notify: seen.update(
            context=context, kind=kind, language=language))
        ri.generate_social_metadata_for_image(mock_analysis_with_text.model_copy(update={"text_language": "Hindi"}),
                                              mock_plan_with_text)
        assert seen["kind"] == "image" and seen["language"] == "Hindi" and "Phones in 2070:" in seen["context"]

    def test_gemini_must_fill_the_text_fields(self):
        required = set(ri._StrictImageAnalysis.model_json_schema()["required"])
        assert {"text_meaning", "text_language", "meme_format"} <= required
        assert "joke or point" in ri._analysis_prompt_image()

    def test_old_cached_analysis_is_redone_once(self, mock_analysis_with_text, tmp_path, monkeypatch):
        monkeypatch.setattr(ri, "CONFIG", dataclasses.replace(ri.CONFIG, work_dir=tmp_path))
        old = mock_analysis_with_text.model_dump()
        for key in ("text_meaning", "text_language", "meme_format"):
            old.pop(key)
        for sha, data in [("old", old), ("new", mock_analysis_with_text.model_dump())]:
            path = ri._image_cache_path(sha)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"image_sha": sha, "image_path": "x", "source_url": "u", "title": "t",
                                        "extractor": "e", "analysis": data}), encoding="utf-8")
        assert ri._load_image_cached("old") is None
        assert ri._load_image_cached("new").from_cache


class TestVideoMetadata:
    def test_on_screen_text_and_key_subjects_are_in_the_context(self, monkeypatch):
        source = analysis()
        scenes = [source.scenes[0].model_copy(update={"on_screen_text": "Me after one bite"}),
                  source.scenes[1].model_copy(update={"on_screen_text": "@chef_ravi watermark"}),
                  source.scenes[2]]
        source = source.model_copy(update={"scenes": scenes, "hero_subjects": [
            rv.HeroSubject(name="iron kadai", why_it_matters="x", size_and_scale="x", shape_and_parts="x",
                           not_present="x")]})
        seen = {}
        monkeypatch.setattr(rv, "generate_metadata", lambda context, kind, language, notify: seen.update(
            context=context))
        rv.generate_social_metadata(result_for(source), plan_with(beat([1])))
        lines = seen["context"].splitlines()
        assert lines[1] == ("Text on screen (often the main message -- build the metadata around it): "
                            "Me after one bite")
        assert "Key subjects: iron kadai" in lines and "Settings: kitchen" in lines
        assert "@chef_ravi" not in seen["context"]


class TestPromptAndTrending:
    def test_prompt_puts_text_first_and_asks_for_trends(self):
        prompt = rv._metadata_prompt("Content: meme", "image", "English", search=True)
        assert "=== TEXT ON SCREEN ===" in prompt and "Never ignore it" in prompt
        assert "trending RIGHT NOW" in prompt and "search its key phrase too" in prompt
        assert "1-2 that are trending right now" in prompt and "trending_now" in prompt
        assert "#viral #reels" in prompt  # generic tags stay banned

    def test_trending_is_kept_and_shown(self):
        draft = rv.SocialMetadataDraft(
            primary_keyword="phone drop meme", search_insights="x", youtube_title="Phones in 2070 😳",
            youtube_description="d", youtube_tags=["phone drop meme"], instagram_caption="c",
            instagram_hashtags=["#phonememe", "#techhumor", "#futuretech", "#gadgets", "#memesdaily"],
            trending_now=["'Phones in 2070' format is trending on Reels", "#phonememe came from it"])
        meta = rv._finish_metadata(draft, "image", ["phone meme"], [], "gemini-2.5-flash")
        assert meta.trending_now == "'Phones in 2070' format is trending on Reels #phonememe came from it"
        assert "📈 TRENDING NOW" in rv.render_social_metadata(meta)

    def test_reply_without_trending_still_works(self):
        draft = rv.SocialMetadataDraft(primary_keyword="k", search_insights="x", youtube_title="t",
                                       youtube_description="d", youtube_tags=[], instagram_caption="c",
                                       instagram_hashtags=[])
        assert rv._finish_metadata(draft, "video", [], []).trending_now == ""
        assert rv.SocialMetadata.model_validate({"youtube_title": "t", "youtube_description": "d",
                                                 "youtube_tags": "", "instagram_caption": "c",
                                                 "instagram_hashtags": ""}).trending_now == ""
