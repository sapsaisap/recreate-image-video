"""URL safety: links may not reach internal networks, redirects and sizes are
checked on every hop, and Instagram cookies only ever go to Instagram."""
import dataclasses
import socket

import pytest
import requests

import recreate_video as rv
from recreate_video import URLVideoError, detect_url_type, host_is, safe_get, validate_public_video_url

PUBLIC_IP = "142.250.183.14"


def resolve_to(monkeypatch, mapping):
    """Fake DNS: hostname -> IP (anything unknown resolves to a public IP)."""
    def getaddrinfo(host, *args, **kwargs):
        ip = mapping.get(host, PUBLIC_IP)
        family = socket.AF_INET6 if ":" in ip else socket.AF_INET
        return [(family, socket.SOCK_STREAM, 6, "", (ip, 0))]
    monkeypatch.setattr(rv.socket, "getaddrinfo", getaddrinfo)


class TestInternalAddressesBlocked:
    @pytest.mark.parametrize("url", [
        "http://127.0.0.1/",
        "http://localhost/",
        "http://[::1]/",
        "http://[::ffff:127.0.0.1]/",
        "http://2130706433/",          # 127.0.0.1 written as a decimal
        "http://169.254.169.254/latest/meta-data/",
        "http://10.0.0.1/",
        "http://192.168.1.10/",
        "http://100.64.0.1/",          # carrier-grade NAT
        "http://0.0.0.0/",
    ])
    def test_rejected(self, url):
        with pytest.raises(URLVideoError):
            validate_public_video_url(url)

    def test_public_name_resolving_to_private_ip_is_rejected(self, monkeypatch):
        resolve_to(monkeypatch, {"innocent.example": "10.1.2.3"})
        with pytest.raises(URLVideoError, match="private or internal"):
            validate_public_video_url("https://innocent.example/video.mp4")

    def test_public_host_is_accepted(self, monkeypatch):
        resolve_to(monkeypatch, {})
        url = "https://www.youtube.com/shorts/abc"
        assert validate_public_video_url(url) == url

    def test_unknown_host_gives_clear_error(self, monkeypatch):
        def fail(*args, **kwargs):
            raise socket.gaierror("no such host")
        monkeypatch.setattr(rv.socket, "getaddrinfo", fail)
        with pytest.raises(URLVideoError, match="Could not find the website"):
            validate_public_video_url("https://no-such-site.example/")


class TestAllowlist:
    def test_only_listed_sites_accepted(self, monkeypatch):
        resolve_to(monkeypatch, {})
        monkeypatch.setattr(rv, "CONFIG", dataclasses.replace(rv.CONFIG, url_allowlist=("youtube.com",)))
        assert validate_public_video_url("https://m.youtube.com/watch?v=1")
        with pytest.raises(URLVideoError, match="not enabled"):
            validate_public_video_url("https://www.instagram.com/p/abc/")


class TestHostMatching:
    @pytest.mark.parametrize("host, domain, expected", [
        ("instagram.com", "instagram.com", True),
        ("www.instagram.com", "instagram.com", True),
        ("evilinstagram.com", "instagram.com", False),
        ("instagram.com.evil.net", "instagram.com", False),
        ("netflix.com", "x.com", False),
        (None, "instagram.com", False),
    ])
    def test_host_is(self, host, domain, expected):
        assert host_is(host, domain) is expected

    def test_query_string_does_not_make_a_link_instagram(self):
        assert detect_url_type("https://evil.example/?u=instagram.com/p/x") == rv.MediaType.VIDEO
        assert detect_url_type("https://www.instagram.com/p/abc/") == rv.MediaType.AMBIGUOUS


class FakeResponse:
    def __init__(self, status=200, body=b"ok", location=None):
        self.status_code = status
        self.headers = {"Location": location} if location else {"Content-Type": "image/jpeg"}
        self.is_redirect = location is not None
        self.encoding = "utf-8"
        self._body = body

    def iter_content(self, size):
        for start in range(0, len(self._body), size):
            yield self._body[start:start + size]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def fake_web(monkeypatch):
    """requests.Session.get replaced by a URL -> FakeResponse table; records calls."""
    resolve_to(monkeypatch, {"internal.example": "10.0.0.5"})
    pages, calls = {}, []

    def get(self, url, **kwargs):
        calls.append({"url": url, **kwargs})
        return pages[url]

    monkeypatch.setattr(requests.Session, "get", get)
    return pages, calls


class TestSafeGet:
    def test_follows_public_redirect(self, fake_web):
        pages, _ = fake_web
        pages["https://a.example/x"] = FakeResponse(302, location="/y")
        pages["https://a.example/y"] = FakeResponse(body=b"hello")
        fetched = safe_get("https://a.example/x", max_bytes=100)
        assert fetched.content == b"hello" and fetched.url == "https://a.example/y"

    def test_redirect_to_internal_address_is_refused(self, fake_web):
        pages, calls = fake_web
        pages["https://a.example/x"] = FakeResponse(302, location="http://internal.example/admin")
        with pytest.raises(URLVideoError, match="private or internal"):
            safe_get("https://a.example/x", max_bytes=100)
        assert [c["url"] for c in calls] == ["https://a.example/x"]

    def test_redirect_off_allowed_domains_is_refused(self, fake_web):
        pages, _ = fake_web
        pages["https://scontent.cdninstagram.com/i.jpg"] = FakeResponse(302, location="https://evil.example/i.jpg")
        with pytest.raises(URLVideoError, match="not an expected site"):
            safe_get("https://scontent.cdninstagram.com/i.jpg", max_bytes=100,
                     allowed_domains=rv.INSTAGRAM_CDN_DOMAINS)

    def test_stops_reading_past_size_limit(self, fake_web):
        pages, _ = fake_web
        pages["https://a.example/big"] = FakeResponse(body=b"x" * 5000)
        with pytest.raises(URLVideoError, match="larger than"):
            safe_get("https://a.example/big", max_bytes=1000)

    def test_too_many_redirects(self, fake_web):
        pages, _ = fake_web
        pages["https://a.example/loop"] = FakeResponse(302, location="/loop")
        with pytest.raises(URLVideoError, match="too many times"):
            safe_get("https://a.example/loop", max_bytes=100)

    def test_http_error_is_reported(self, fake_web):
        pages, _ = fake_web
        pages["https://a.example/gone"] = FakeResponse(404)
        with pytest.raises(URLVideoError, match="HTTP 404"):
            safe_get("https://a.example/gone", max_bytes=100)


COOKIES_TXT = (
    "# Netscape HTTP Cookie File\n"
    "#HttpOnly_.instagram.com\tTRUE\t/\tTRUE\t0\tsessionid\tSECRET_SESSION\n"
    ".instagram.com\tTRUE\t/\tTRUE\t0\tcsrftoken\tcsrf123\n"
    ".google.com\tTRUE\t/\tTRUE\t0\tSID\tGOOGLE_SECRET\n"
)


@pytest.fixture
def cookie_file(tmp_path):
    path = tmp_path / "cookies.txt"
    path.write_text(COOKIES_TXT, encoding="utf-8")
    return path


def cookie_header(jar, url):
    prepared = requests.Session().prepare_request(requests.Request("GET", url, cookies=jar))
    return prepared.headers.get("Cookie", "")


class TestCookies:
    def test_httponly_session_cookie_is_loaded(self, cookie_file):
        jar = rv._load_cookie_jar(str(cookie_file))
        assert {c.name for c in jar} == {"sessionid", "csrftoken", "SID"}
        assert rv._has_cookies_for(jar, "instagram.com")

    def test_instagram_cookies_only_go_to_instagram(self, cookie_file):
        jar = rv._load_cookie_jar(str(cookie_file))
        to_instagram = cookie_header(jar, "https://www.instagram.com/p/abc/")
        assert "SECRET_SESSION" in to_instagram and "GOOGLE_SECRET" not in to_instagram
        assert cookie_header(jar, "https://evil.example/?instagram.com") == ""

    def test_missing_file_means_no_cookies(self, tmp_path):
        assert rv._load_cookie_jar(str(tmp_path / "nope.txt")) is None
        assert not rv._has_cookies_for(None, "instagram.com")


class TestInstagramDownloader:
    @pytest.fixture
    def setup(self, fake_web, cookie_file, tmp_path, monkeypatch):
        monkeypatch.setattr(rv, "CONFIG", dataclasses.replace(rv.CONFIG, cookies_file=str(cookie_file)))
        return fake_web

    def page(self, image_url):
        return FakeResponse(body=(
            f'<meta property="og:image" content="{image_url}">'
            '<meta property="og:title" content="A post">'
        ).encode())

    def test_downloads_from_cdn_without_cookies(self, setup, tmp_path):
        pages, calls = setup
        pages["https://www.instagram.com/p/abc/"] = self.page("https://scontent.cdninstagram.com/v/i.jpg?a=1&amp;b=2")
        pages["https://scontent.cdninstagram.com/v/i.jpg?a=1&b=2"] = FakeResponse(body=b"JPEGDATA")
        result = rv.download_instagram_image("https://www.instagram.com/p/abc/", tmp_path / "out")
        assert result.path.read_bytes() == b"JPEGDATA" and result.title == "A post"
        assert calls[0]["cookies"] is not None and calls[1]["cookies"] is None

    def test_image_url_outside_instagram_cdn_is_refused(self, setup, tmp_path):
        pages, calls = setup
        pages["https://www.instagram.com/p/abc/"] = self.page("https://evil.example/steal.jpg")
        with pytest.raises(URLVideoError, match="not an expected site"):
            rv.download_instagram_image("https://www.instagram.com/p/abc/", tmp_path / "out")
        assert len(calls) == 1

    def test_non_instagram_link_is_refused_before_any_request(self, setup, tmp_path):
        _, calls = setup
        with pytest.raises(URLVideoError, match="not an Instagram link"):
            rv.download_instagram_image("https://evil.example/?instagram.com", tmp_path / "out")
        assert calls == []


class TestDirectImageDownload:
    def test_image_link_is_fetched_directly(self, fake_web, tmp_path):
        import recreate_image as ri
        pages, calls = fake_web
        response = FakeResponse(body=b"PNGDATA")
        response.headers = {"Content-Type": "image/png"}
        pages["https://images.example/photo"] = response
        downloaded = ri.download_direct_image("https://images.example/photo", tmp_path)
        assert downloaded.path.read_bytes() == b"PNGDATA" and downloaded.path.suffix == ".png"
        assert "Mozilla" in calls[0]["headers"]["User-Agent"]  # some hosts refuse non-browser clients

    def test_web_page_is_left_to_yt_dlp(self, fake_web, tmp_path):
        import recreate_image as ri
        pages, _ = fake_web
        page = FakeResponse(body=b"<html></html>")
        page.headers = {"Content-Type": "text/html; charset=utf-8"}
        pages["https://site.example/post"] = page
        assert ri.download_direct_image("https://site.example/post", tmp_path) is None
