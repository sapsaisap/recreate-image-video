"""
test_social_metadata.py - YouTube / Instagram metadata: platform limits (tags
packed to YouTube's own count, description in bytes, 5 niche hashtags), the
search-grounded generation path and its no-search fallback. Zero API calls.
"""
import dataclasses
from types import SimpleNamespace

import pytest
from google.genai.errors import ClientError

import recreate_video as rv
from recreate_video import (
    GeneratedPrompt, SocialMetadata, SocialMetadataDraft, YOUTUBE_TAGS_LIMIT,
    _finish_metadata, _fit_title, _pack_youtube_tags, _validate_instagram_hashtags,
    render_social_metadata, split_tags, youtube_tags_cost,
)


@pytest.fixture
def mock_meta():
    return SocialMetadata(
        youtube_title="Phone Drop Saved By Thrusters 😳 #Shorts",
        youtube_description="What if your phone could save itself? Futuristic thruster tech.\n#Shorts #phonedrop",
        youtube_tags="phone drop, futuristic technology, phone thrusters, tech memes",
        instagram_caption="Phone drop… saved by thrusters 😳\nSave this for later 📌",
        instagram_hashtags="#phonedrop #futuretech #techfail #gadgets #scifi",
    )


def make_draft(**overrides):
    values = dict(
        primary_keyword="telugu biryani recipe",
        search_insights="People search 'biryani recipe telugu' and 'hyderabadi biryani'.",
        youtube_title="Telugu Biryani Recipe in 60 Seconds 🔥",
        youtube_description="Make Hyderabadi biryani at home.\n\nSubscribe for more!\n#Shorts #biryani",
        youtube_tags=["biryani recipe telugu", "hyderabadi biryani", "biryani in telugu"],
        instagram_caption="Biryani in 60 seconds 🔥\nSave this recipe 📌",
        instagram_hashtags=["#biryani", "#telugufood", "#viral", "hyderabadibiryani", "#reels", "#foodie", "#homecooking"],
    )
    values.update(overrides)
    return SocialMetadataDraft(**values)


# ── YouTube tag packing ─────────────────────────────────────────────────────

class TestYouTubeTags:
    def test_cost_counts_quotes_and_commas(self):
        # "phone drop" = 10 + 2 quotes, "tech" = 4, one comma
        assert youtube_tags_cost(["phone drop", "tech"]) == 17

    def test_packs_up_to_target_and_never_over_limit(self):
        candidates = [f"telugu cooking idea number {i}" for i in range(60)]
        packed = _pack_youtube_tags(candidates)
        cost = youtube_tags_cost(split_tags(packed))
        assert 440 <= cost <= 480 < YOUTUBE_TAGS_LIMIT

    def test_skips_a_long_tag_but_keeps_filling(self):
        packed = _pack_youtube_tags(["a" * 20, "b" * 30, "c"], target=25)
        assert split_tags(packed) == ["a" * 20, "c"]

    def test_cleans_and_dedupes(self):
        packed = _pack_youtube_tags(["#Biryani", "biryani", "<script>", "a,b"])
        assert split_tags(packed) == ["Biryani", "script", "a b"]

    def test_finish_puts_primary_keyword_first(self):
        meta = _finish_metadata(make_draft(), "video", [], [])
        assert split_tags(meta.youtube_tags)[0] == "telugu biryani recipe"


# ── Title and description limits ────────────────────────────────────────────

class TestTitleAndDescription:
    def test_shorts_suffix_added_for_video(self):
        assert _fit_title("Biryani in 60 seconds", shorts=True) == "Biryani in 60 seconds #Shorts"

    def test_no_suffix_for_image(self):
        assert _fit_title("Biryani poster", shorts=False) == "Biryani poster"

    def test_title_trimmed_to_100_at_a_word(self):
        title = _fit_title("word " * 40, shorts=True)
        assert len(title) <= 100 and title.endswith(" #Shorts") and "  " not in title

    def test_description_capped_in_bytes(self):
        telugu = "బిర్యానీ " * 1000  # 3 bytes per letter
        meta = _finish_metadata(make_draft(youtube_description=telugu), "video", [], [])
        assert len(meta.youtube_description.encode("utf-8")) <= 5000
        assert meta.youtube_description.endswith("బిర్యానీ")

    def test_caption_capped_at_2200(self):
        meta = _finish_metadata(make_draft(instagram_caption="food " * 1000), "video", [], [])
        assert len(meta.instagram_caption) <= 2200


# ── Instagram hashtags ──────────────────────────────────────────────────────

class TestInstagramHashtags:
    def test_drops_generic_tags_and_caps_at_5(self):
        result = _finish_metadata(make_draft(), "video", [], []).instagram_hashtags
        assert result == "#biryani #telugufood #hyderabadibiryani #foodie #homecooking"

    def test_adds_missing_hash_prefix(self):
        assert _validate_instagram_hashtags("phonedrop futuretech") == "#phonedrop #futuretech"

    def test_no_duplicates_case_insensitive(self):
        assert _validate_instagram_hashtags("#Biryani #biryani") == "#Biryani"

    def test_strips_punctuation(self):
        assert _validate_instagram_hashtags("#phone-drop! #tech.") == "#phonedrop #tech"

    def test_keeps_telugu_vowel_signs(self):
        assert _validate_instagram_hashtags("#తెలుగు") == "#తెలుగు"


# ── Rendering ───────────────────────────────────────────────────────────────

class TestRenderSocialMetadata:
    def test_sections_and_fields_present(self, mock_meta):
        text = render_social_metadata(mock_meta)
        for part in ("=== YOUTUBE ===", "=== INSTAGRAM ===", mock_meta.youtube_title,
                     mock_meta.youtube_description, mock_meta.youtube_tags,
                     mock_meta.instagram_caption, mock_meta.instagram_hashtags):
            assert part in text

    def test_tag_count_uses_youtube_limit(self, mock_meta):
        assert f"/{YOUTUBE_TAGS_LIMIT} as YouTube counts them" in render_social_metadata(mock_meta)

    def test_says_when_not_grounded(self, mock_meta):
        assert "Not grounded in Google Search" in render_social_metadata(mock_meta)

    def test_lists_searches_and_sources_when_grounded(self, mock_meta):
        meta = mock_meta.model_copy(update={
            "grounded": True, "search_queries": ["phone drop shorts"],
            "sources": [{"title": "youtube.com", "uri": "https://example.com/x"}],
        })
        text = render_social_metadata(meta)
        assert "Keywords researched with Google Search (1 searches)" in text
        assert "phone drop shorts" in text and "https://example.com/x" in text


class TestGeneratedPromptFullText:
    def test_full_text_includes_social_section(self, mock_meta):
        gp = GeneratedPrompt(duration=8, prompt="p", plan={}, saved_to="t.txt",
                             social_metadata=mock_meta, social_text=render_social_metadata(mock_meta))
        full = gp.full_text()
        assert "=== SOCIAL MEDIA METADATA ===" in full and "=== YOUTUBE ===" in full

    def test_full_text_without_social_no_section(self):
        gp = GeneratedPrompt(duration=8, prompt="p", plan={}, saved_to="t.txt")
        assert "=== SOCIAL MEDIA METADATA ===" not in gp.full_text()


# ── generate_metadata against a fake Gemini ─────────────────────────────────

def grounded_response(draft):
    text = "Here is the metadata you asked for:\n" + draft.model_dump_json() + "\nGood luck!"
    grounding = SimpleNamespace(
        web_search_queries=["telugu biryani recipe shorts", "hyderabadi biryani"],
        grounding_chunks=[
            SimpleNamespace(web=SimpleNamespace(uri="https://example.com/a", title="youtube.com", domain=None)),
            SimpleNamespace(web=SimpleNamespace(uri="https://example.com/a", title="youtube.com", domain=None)),
        ],
    )
    return SimpleNamespace(parsed=None, text=text, prompt_feedback=None,
                           candidates=[SimpleNamespace(finish_reason="STOP", grounding_metadata=grounding)])


def search_quota_error():
    return ClientError(429, {"error": {
        "code": 429, "status": "RESOURCE_EXHAUSTED", "message": "Quota exceeded.",
        "details": [{"@type": "type.googleapis.com/google.rpc.QuotaFailure", "violations": [
            {"quotaId": "SearchGroundingRequestsPerDayPerProject-FreeTier", "quotaValue": "500"}]}],
    }})


class TestGenerateMetadata:
    @pytest.fixture
    def fake(self, tmp_path, monkeypatch):
        monkeypatch.setattr(rv, "CONFIG", dataclasses.replace(rv.CONFIG, work_dir=tmp_path, task_models={
            **rv.CONFIG.task_models, "metadata": ("gemini-2.5-flash-lite", "gemini-3.5-flash-lite"),
        }))
        monkeypatch.setattr(rv._STATE, "quota", None)
        monkeypatch.setattr(rv, "_throttle", lambda: None)
        calls = []
        replies = []

        def generate_content(model, contents, config):
            calls.append({"model": model, "search": bool(config.tools), "schema": config.response_schema,
                          "prompt": contents[0]})
            reply = replies.pop(0)
            if isinstance(reply, Exception):
                raise reply
            return reply

        monkeypatch.setattr(rv, "get_client", lambda: SimpleNamespace(
            models=SimpleNamespace(generate_content=generate_content)))
        return calls, replies

    def test_grounded_on_free_search_model(self, fake):
        calls, replies = fake
        replies.append(grounded_response(make_draft()))
        meta = rv.generate_metadata("Content: biryani", "video", "Telugu")
        assert calls[0]["search"] and calls[0]["schema"] is None
        assert "use Google Search" in calls[0]["prompt"] and "Tenglish" in calls[0]["prompt"]
        assert meta.grounded and meta.search_queries == ["telugu biryani recipe shorts", "hyderabadi biryani"]
        assert meta.sources == [{"title": "youtube.com", "uri": "https://example.com/a"}]
        assert meta.youtube_title.endswith("#Shorts")

    def test_search_quota_falls_back_without_search_on_same_model(self, fake):
        calls, replies = fake
        replies += [search_quota_error(), SimpleNamespace(parsed=make_draft())]
        meta = rv.generate_metadata("Content: biryani", "video", "English")
        assert [(c["model"], c["search"]) for c in calls] == [
            ("gemini-2.5-flash-lite", True), ("gemini-2.5-flash-lite", False)]
        assert calls[1]["schema"] is SocialMetadataDraft
        assert not meta.grounded
        assert rv._search_used_up() and rv._unavailable_models() == {}

    def test_skips_search_once_used_up_today(self, fake):
        calls, replies = fake
        rv._mark_search_used_up()
        replies.append(SimpleNamespace(parsed=make_draft()))
        rv.generate_metadata("Content: biryani", "image", "English")
        assert calls[0]["search"] is False

    def test_no_search_on_model_without_free_search(self, fake):
        calls, replies = fake
        rv._mark_unavailable("gemini-2.5-flash-lite", "daily free quota used up")
        replies.append(SimpleNamespace(parsed=make_draft()))
        meta = rv.generate_metadata("Content: poster", "image", "English")
        assert calls[0]["model"] == "gemini-3.5-flash-lite" and calls[0]["search"] is False
        assert not meta.youtube_title.endswith("#Shorts")

    def test_answer_without_searching_is_asked_again(self, fake):
        calls, replies = fake
        ungrounded = SimpleNamespace(parsed=None, text=make_draft().model_dump_json(), prompt_feedback=None,
                                     candidates=[SimpleNamespace(finish_reason="STOP", grounding_metadata=None)])
        replies.extend([ungrounded, grounded_response(make_draft())])
        meta = rv.generate_metadata("Content: biryani", "video", "English")
        assert len(calls) == 2 and all(c["search"] for c in calls)
        assert "run Google Search now" in calls[1]["prompt"] and meta.grounded

    def test_keeps_the_answer_if_it_still_does_not_search(self, fake):
        calls, replies = fake
        ungrounded = lambda: SimpleNamespace(parsed=None, text=make_draft().model_dump_json(), prompt_feedback=None,
                                             candidates=[SimpleNamespace(finish_reason="STOP", grounding_metadata=None)])
        replies.extend([ungrounded(), ungrounded()])
        meta = rv.generate_metadata("Content: biryani", "video", "English")
        assert len(calls) == 2 and not meta.grounded and meta.youtube_title


class TestHashtagFill:
    def test_fills_up_to_five_from_keyword_ideas(self):
        draft = make_draft(instagram_hashtags=["#viral", "#biryani", "#reels"])
        tags = _finish_metadata(draft, "video", [], []).instagram_hashtags.split()
        assert tags == ["#biryani", "#telugubiryanirecipe", "#biryanirecipetelugu", "#hyderabadibiryani",
                        "#biryaniintelugu"]

    def test_long_ideas_are_not_used_as_hashtags(self):
        tags = _validate_instagram_hashtags("#food", fill_from=["a very long search phrase nobody types", "biryani"])
        assert tags == "#food #biryani"


class TestMetadataRetry:
    def test_failure_shows_reason_and_retry_works(self):
        from streamlit.testing.v1 import AppTest

        def page(repo):
            import sys
            sys.path.insert(0, repo)
            import streamlit as st
            import recreate_video as rv

            def make(notify):
                st.session_state["attempts"] = st.session_state.get("attempts", 0) + 1
                if st.session_state["attempts"] == 1:
                    raise rv.GeminiCallError("gemini-2.5-flash is busy")
                st.session_state["social"] = rv.SocialMetadata(
                    youtube_title="T #Shorts", youtube_description="D", youtube_tags="a, b",
                    instagram_caption="C", instagram_hashtags="#a #b #c #d #e")
                return st.session_state["social"]

            rv.render_social_or_generate(st, st.session_state.get("social"), "k", make)

        app = AppTest.from_function(page, args=(str(rv.APP_DIR),), default_timeout=60).run()
        assert any("gemini-2.5-flash is busy" in w.value for w in app.warning)
        app = app.run()  # a plain rerun does not retry on its own
        assert app.session_state.attempts == 1
        next(b for b in app.button if b.label == "Retry metadata").click()
        app = app.run()
        assert app.session_state.attempts == 2 and not app.warning
        assert [t.label for t in app.tabs] == ["▶️ YouTube", "📸 Instagram"]
