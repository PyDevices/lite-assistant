"""
The web tools, with the network faked.
"""
import os
import unittest
from unittest import mock

from lite_assistant.tools import web

ARTICLE = ("<html><head><title>Board news</title></head><body><nav>Home | About</nav><article><h1>Board news</h1>"
           + "".join(f"<p>Day {i}: the new board has a camera and a microphone, and it talks to a server on the "
                     f"local network {i} times.</p>" for i in range(80))
           + "</article></body></html>")


class Page:
    def __init__(self, url, text):
        self.url, self.text = url, text

    def raise_for_status(self):
        pass


class TestSearch(unittest.TestCase):
    def test_duckduckgo_results_are_trimmed(self):
        found = [{"title": "A", "href": "https://a.example", "body": "word " * 200}]
        with mock.patch.dict(os.environ), mock.patch("ddgs.DDGS") as ddgs:
            os.environ.pop("SEARXNG_URL", None)
            ddgs.return_value.text.return_value = found
            results = web.search_web("boards")["results"]
        self.assertEqual(results[0]["url"], "https://a.example")
        self.assertLessEqual(len(results[0]["snippet"]), web.SNIPPET_CHARS + 1)

    def test_news(self):
        found = [{"title": "N", "url": "https://n.example", "body": "b", "date": "2026-10-26", "source": "S"}]
        with mock.patch.dict(os.environ), mock.patch("ddgs.DDGS") as ddgs:
            os.environ.pop("SEARXNG_URL", None)
            ddgs.return_value.news.return_value = found
            self.assertEqual(web.search_web("boards", news=True)["results"][0]["date"], "2026-10-26")

    def test_news_that_finds_nothing_searches_the_web(self):
        from ddgs.exceptions import DDGSException
        with mock.patch.dict(os.environ), mock.patch("ddgs.DDGS") as ddgs:
            os.environ.pop("SEARXNG_URL", None)
            ddgs.return_value.news.side_effect = DDGSException("No results found.")
            ddgs.return_value.text.return_value = [{"title": "W", "href": "https://w.example", "body": "b"}]
            self.assertEqual(web.search_web("race winner", news=True)["results"][0]["title"], "W")

    def test_searxng_when_set(self):
        reply = mock.Mock(json=lambda: {"results": [{"title": "S", "url": "https://s.example", "content": "c"}]})
        with mock.patch.dict(os.environ, {"SEARXNG_URL": "http://searx.lan:8080/"}), \
                mock.patch("httpx.get", return_value=reply) as get:
            self.assertEqual(web.search_web("boards")["results"][0]["title"], "S")
        self.assertEqual(get.call_args.args[0], "http://searx.lan:8080/search")

    def test_nothing_found(self):
        with mock.patch.dict(os.environ), mock.patch("ddgs.DDGS") as ddgs:
            os.environ.pop("SEARXNG_URL", None)
            ddgs.return_value.text.return_value = []
            self.assertIn("note", web.search_web("zzz"))


class TestReadPage(unittest.TestCase):
    def test_reads_the_main_text_and_cuts_it(self):
        with mock.patch.object(web, "public", return_value=True), \
                mock.patch("httpx.get", return_value=Page("https://news.example/a", ARTICLE)):
            page = web.read_web_page("https://news.example/a")
        self.assertIn("camera and a microphone", page["text"])
        self.assertNotIn("About", page["text"])
        self.assertLessEqual(len(page["text"]), web.PAGE_CHARS + 1)
        self.assertIn("note", page)

    def test_private_addresses_are_refused(self):
        for url in ("http://localhost:5000/v1/models", "http://192.168.1.10/", "file:///etc/passwd", "http://127.0.0.1/"):
            self.assertFalse(web.public(url), url)
            self.assertIn("error", web.read_web_page(url))


if __name__ == "__main__":
    unittest.main()
