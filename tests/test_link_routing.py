"""Links pasted in the Video tab reach the right pipeline: image links and
Instagram photo posts go to image analysis and the app switches to the Image
view; Reels stay on the video pipeline."""
import socket
import textwrap
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest
from yt_dlp.utils import DownloadError

import recreate_video as rv

REPO = Path(__file__).resolve().parent.parent

# Runs the real app with the network-facing steps replaced. The fakes record
# which pipeline each link reached in st.session_state["calls"].
WRAPPER = textwrap.dedent('''
    import dataclasses, sys
    sys.path.insert(0, {repo!r})
    sys.path.insert(0, {tests!r})
    import streamlit as st
    import recreate_image as ri
    import recreate_video as rv
    from test_image_cache_uploads import make_analysis

    rv.CONFIG = dataclasses.replace(rv.CONFIG, api_key="dummy", work_dir=__import__("pathlib").Path({work!r}))
    st.session_state.setdefault("calls", [])

    def fake_video(url, notify=None):
        st.session_state.calls.append(("video", url))
        if "/p/" in url:
            raise rv.NoVideoInPost("This post has no video -- it is a photo post.")
        raise rv.URLVideoError("stop here")

    def fake_image(url, notify=None):
        st.session_state.calls.append(("image", url))
        return ri.ImageAnalysisResult(image_sha="a" * 64, image_path="gone.jpg", source_url=url,
                                      title="Photo post", extractor="Instagram",
                                      analysis=make_analysis().model_dump())

    rv.analyze_url = fake_video
    ri.analyze_image_url = fake_image
    rv.run_app()
''')


@pytest.fixture
def app(tmp_path):
    wrapper = tmp_path / "app.py"
    wrapper.write_text(WRAPPER.format(repo=str(REPO), tests=str(REPO / "tests"), work=str(tmp_path / "work")),
                       encoding="utf-8")
    return AppTest.from_file(str(wrapper), default_timeout=60).run()


def submit(app, url):
    app.text_input[0].input(url)
    next(b for b in app.button if b.label == "🔍 Analyze link").click()
    return app.run()


class TestVideoTabRouting:
    def test_direct_image_link_goes_to_image_view(self, app):
        app = submit(app, "https://example.com/photo.jpg")
        assert app.session_state.calls == [("image", "https://example.com/photo.jpg")]
        assert app.session_state.app_mode == rv.IMAGE_MODE
        assert not app.exception
        assert any(h.value == "Photo post" for h in app.subheader)

    def test_instagram_photo_post_falls_back_to_image(self, app):
        app = submit(app, "https://www.instagram.com/p/abc123/")
        assert app.session_state.calls == [
            ("video", "https://www.instagram.com/p/abc123/"),
            ("image", "https://www.instagram.com/p/abc123/"),
        ]
        assert app.session_state.app_mode == rv.IMAGE_MODE

    def test_reel_stays_on_video_pipeline(self, app):
        app = submit(app, "https://www.instagram.com/reel/abc123/")
        assert app.session_state.calls == [("video", "https://www.instagram.com/reel/abc123/")]
        assert app.session_state.app_mode == rv.VIDEO_MODE
        assert any("stop here" in e.value for e in app.error)


class TestNoVideoDetection:
    def test_instagram_photo_post_raises_no_video(self, tmp_path, monkeypatch):
        monkeypatch.setattr(rv.socket, "getaddrinfo",
                            lambda *a, **k: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("157.240.1.174", 0))])

        class FakeYDL:
            def __init__(self, options):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def extract_info(self, url, download):
                raise DownloadError("ERROR: [Instagram] abc123: There is no video in this post")

        monkeypatch.setattr("yt_dlp.YoutubeDL", FakeYDL)
        with pytest.raises(rv.NoVideoInPost):
            rv.download_video("https://www.instagram.com/p/abc123/", tmp_path)

    def test_other_errors_are_not_no_video(self):
        assert not rv._is_no_video(Exception("Sign in to confirm you're not a bot"))
        assert rv._is_no_video(Exception("ERROR: No video formats found!"))
