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
        monkeypatch.setattr(ri, "generate_metadata", lambda context, kind, language, notify, audio_language="": seen.update(
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
        monkeypatch.setattr(rv, "generate_metadata", lambda context, kind, language, notify, audio_language="": seen.update(
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
        assert "1 trending right now" in prompt and "trending_now" in prompt
        assert "#viral #reels" in prompt  # generic tags stay banned

    def test_broad_high_view_tags_are_part_of_the_mix(self):
        video = rv._metadata_prompt("Content: recipe", "video", "Telugu", search=True)
        assert "2 broad community tags for its topic and language / region that have big audiences" in video
        assert "Check which BROAD hashtags" in video and "3-5 broad high-volume tags" in video
        assert "#Shorts, then 1 broad and 1-2 niche hashtags (YouTube shows the first 3 above the title)" in video
        image = rv._metadata_prompt("Content: meme", "image", "English", search=True)
        assert "3 hashtags, 1 broad and 2 niche" in image
        assert rv._validate_instagram_hashtags(["#telugumemes", "#viral", "#memes", "#fyp", "#3amscroll"],
                                               fill_from=[]).split() == ["#telugumemes", "#memes", "#3amscroll"]

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


class TestTrendingAudio:
    def test_sounds_follow_the_contents_language(self):
        telugu = rv._metadata_prompt("Content: recipe", "video", "Telugu", search=True)
        assert "pick trending Telugu songs and sounds first (at least 2 of the 3)" in telugu
        english = rv._metadata_prompt("Content: meme", "image", "English", search=True)
        assert "trending worldwide" in english and "trending Telugu" not in english

    def test_prompt_asks_for_sounds_that_fit_and_are_searched(self):
        prompt = rv._metadata_prompt("Content: meme", "image", "English", search=True)
        assert "sounds / songs are trending or widely used RIGHT NOW" in prompt
        assert "fit THIS meme or video" in prompt and "the sound that goes with its format comes first" in prompt
        assert "trending_audio (array of objects" in prompt and "never 'Original Sound'" in prompt

    def test_gemini_shapes_are_accepted(self):
        def draft(audio):
            return rv.SocialMetadataDraft(primary_keyword="k", search_insights="x", youtube_title="t",
                                          youtube_description="d", youtube_tags=[], instagram_caption="c",
                                          instagram_hashtags=[], trending_audio=audio)
        full = draft([{"track_name": "Kurchi Madathapetti", "artist_name": "Thaman S", "language": "Telugu",
                       "reason": "used on most Telugu food reels"}, "Monkey Spinning Monkeys", {"artist": "x"},
                      {"track": "a"}, {"track": "b"}])
        assert [a.track for a in full.trending_audio] == ["Kurchi Madathapetti", "Monkey Spinning Monkeys", "a"]
        first = full.trending_audio[0]
        assert first.search_query == "Kurchi Madathapetti Thaman S" and first.why.startswith("used on")
        assert draft({"song": "Oh No", "by": "Capone"}).trending_audio[0].label == "Oh No — Capone"
        assert draft(None).trending_audio == [] and draft("").trending_audio == []
        assert draft([{"track": "Original Sound - me at 3am"}, "original audio"]).trending_audio == []

    def test_kept_shown_and_old_metadata_still_loads(self):
        draft = rv.SocialMetadataDraft(primary_keyword="k", search_insights="x", youtube_title="t",
                                       youtube_description="d", youtube_tags=[], instagram_caption="c",
                                       instagram_hashtags=[], trending_audio=[
                                           {"track": "Oh No", "artist": "Capone", "language": "English",
                                            "search_query": "oh no oh no", "why": "the classic fail meme sound"}])
        meta = rv._finish_metadata(draft, "image", ["q"], [])
        text = rv.render_social_metadata(meta)
        assert "🎵 TRENDING AUDIO" in text and "- Oh No — Capone [English] — search: oh no oh no" in text
        assert rv.SocialMetadata.model_validate_json(meta.model_dump_json()).trending_audio[0].track == "Oh No"
        old = rv.SocialMetadata.model_validate({"youtube_title": "t", "youtube_description": "d",
                                                "youtube_tags": "", "instagram_caption": "c",
                                                "instagram_hashtags": ""})
        assert old.trending_audio == [] and "TRENDING AUDIO" not in rv.render_social_metadata(old)

    def test_image_pick_follows_the_text_and_its_language(self):
        prompt = ri._analysis_prompt_image()
        assert "fit what the image and its text are about" in prompt and "audience's language" in prompt


class TestAudioLanguage:
    """The trending sounds follow the audience: sidebar choice > what the post shows > metadata language."""

    @staticmethod
    def prompt_for(monkeypatch, language, audio_language="", chosen=""):
        seen = {}
        monkeypatch.setattr(rv, "chosen_audio_language", lambda: chosen)

        def fake(build_prompt, schema, **kwargs):
            seen["prompt"] = build_prompt(True)
            return (rv.SocialMetadataDraft(primary_keyword="k", search_insights="x", youtube_title="t",
                                           youtube_description="d", youtube_tags=[], instagram_caption="c",
                                           instagram_hashtags=[]), ["q"], [], "m")
        monkeypatch.setattr(rv, "generate_with_optional_search", fake)
        rv.generate_metadata("Content: biryani photo", "image", language, audio_language=audio_language)
        return seen["prompt"]

    def test_the_audience_language_beats_the_text_language(self, monkeypatch):
        assert "pick trending Telugu songs" in self.prompt_for(monkeypatch, "English", audio_language="Telugu")
        assert "Write everything in English." in self.prompt_for(monkeypatch, "English", audio_language="Telugu")

    def test_the_sidebar_choice_wins(self, monkeypatch):
        prompt = self.prompt_for(monkeypatch, "Telugu", audio_language="Telugu", chosen="Hindi")
        assert "pick trending Hindi songs" in prompt and "pick trending Telugu songs" not in prompt

    def test_without_a_language_the_region_is_read_from_the_content(self, monkeypatch):
        prompt = self.prompt_for(monkeypatch, "English")
        assert "clearly belongs to a region or language" in prompt and "list it first" in prompt

    def test_no_choice_outside_the_app(self):
        assert rv.chosen_audio_language() == "" and rv.AUDIO_LANGUAGES[0] == "Auto"

    def test_image_audience_is_asked_for_and_passed(self, mock_analysis_no_text, mock_plan_with_text, monkeypatch):
        assert "audience_language" in ri._StrictImageAnalysis.model_json_schema()["required"]
        seen = {}
        monkeypatch.setattr(ri, "generate_metadata", lambda context, kind, language, notify, audio_language="":
                            seen.update(language=language, audio=audio_language))
        food = mock_analysis_no_text.model_copy(update={"audience_language": "Telugu"})
        ri.generate_social_metadata_for_image(food, mock_plan_with_text)
        assert seen == {"language": "English", "audio": "Telugu"}

    def test_the_source_song_reaches_the_video_metadata(self, monkeypatch):
        source = analysis()
        source = source.model_copy(update={"audio": source.audio.model_copy(update={
            "music": "Kurchi Madathapetti (Guntur Kaaram), Thaman S -- mass beat"})})
        seen = {}
        monkeypatch.setattr(rv, "generate_metadata", lambda context, kind, language, notify, audio_language="":
                            seen.update(context=context))
        rv.generate_social_metadata(result_for(source), plan_with(beat([1])))
        assert "Source music: Kurchi Madathapetti (Guntur Kaaram)" in seen["context"]
        assert "recognisable song, start with its title" in rv.AudioProfile.model_fields["music"].description
