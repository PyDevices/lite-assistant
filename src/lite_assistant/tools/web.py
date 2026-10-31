"""
The web tools: search the web, and read a page. Searches go to DuckDuckGo,
through the ddgs package, or to a SearXNG server if SEARXNG_URL names one.
"""
import ipaddress
import os
import socket
import urllib.parse
from typing import Any, Dict, List

RESULTS = 5
SNIPPET_CHARS = 300
# A page's text is cut to this, to leave room in a 4096-token context
PAGE_CHARS = 4000
TIMEOUT = 15
USER_AGENT = "Mozilla/5.0 (compatible; lite-assistant)"


def searxng(query: str, news: bool) -> List[Dict[str, Any]]:
    import httpx
    url = os.environ["SEARXNG_URL"].rstrip("/") + "/search"
    params = {"q": query, "format": "json", **({"categories": "news"} if news else {})}
    found = httpx.get(url, params=params, timeout=TIMEOUT, headers={"User-Agent": USER_AGENT})
    found.raise_for_status()
    return [{"title": r.get("title", ""), "url": r.get("url", ""), "snippet": r.get("content", ""),
             **({"date": r["publishedDate"]} if r.get("publishedDate") else {})}
            for r in found.json().get("results", [])[:RESULTS]]


def duckduckgo(query: str, news: bool) -> List[Dict[str, Any]]:
    from ddgs import DDGS
    from ddgs.exceptions import DDGSException

    def found(search):
        # ddgs raises when there are no results
        try:
            return search(query, max_results=RESULTS) or []
        except DDGSException as exc:
            if "no results" in str(exc).lower():
                return []
            raise
    # News search finds less, so the web stands in when it finds nothing
    stories = found(DDGS().news) if news else []
    if stories:
        return [{"title": r.get("title", ""), "url": r.get("url", ""), "snippet": r.get("body", ""),
                 "date": r.get("date", ""), "source": r.get("source", "")} for r in stories]
    return [{"title": r.get("title", ""), "url": r.get("href", ""), "snippet": r.get("body", "")}
            for r in found(DDGS().text)]


def search_web(query: str, news: bool = False) -> Dict[str, Any]:
    """
    Searches the web. Use it for anything current, such as news, prices,
    weather, sports or events, and for facts you aren't sure of. Answer from
    the results, and say which site said so. read_web_page reads a result in full.

    Args:
        query: What to search for, as you'd type it into a search engine.
        news: True for recent news stories rather than web pages.
    """
    results = (searxng if os.environ.get("SEARXNG_URL") else duckduckgo)(query, news)
    for result in results:
        if len(result["snippet"]) > SNIPPET_CHARS:
            result["snippet"] = result["snippet"][:SNIPPET_CHARS].rsplit(" ", 1)[0] + "…"
    return {"results": results} if results else {"results": [], "note": f"Nothing was found for {query!r}."}


def public(url: str) -> bool:
    """ Whether a URL is http or https on a public address, not this machine or its network. """
    parts = urllib.parse.urlparse(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        return False
    try:
        addresses = {info[4][0] for info in socket.getaddrinfo(parts.hostname, None)}
    except socket.gaierror:
        return False
    return all(ipaddress.ip_address(a.split("%")[0]).is_global for a in addresses)


def read_web_page(url: str) -> Dict[str, Any]:
    """
    Reads a web page's main text, such as an article from search_web's
    results, when its snippet isn't enough to answer.

    Args:
        url: The page's address, starting with http:// or https://.
    """
    import httpx
    import trafilatura
    if not public(url):
        return {"error": f"{url} isn't a public web page; only http and https addresses on the internet can be read."}
    page = httpx.get(url, timeout=TIMEOUT, follow_redirects=True, headers={"User-Agent": USER_AGENT})
    page.raise_for_status()
    # A redirect could lead somewhere private
    if not public(str(page.url)):
        return {"error": f"{url} redirects to an address that isn't public."}
    text = trafilatura.extract(page.text, url=str(page.url), include_comments=False) or ""
    if not text.strip():
        return {"url": str(page.url), "error": "The page has no text to read; it may need JavaScript."}
    title = (trafilatura.extract_metadata(page.text) or None)
    cut = len(text) > PAGE_CHARS
    return {"url": str(page.url), **({"title": title.title} if title and title.title else {}),
            "text": text[:PAGE_CHARS].rsplit(" ", 1)[0] + "…" if cut else text,
            **({"note": f"Only the first {PAGE_CHARS} characters of {len(text)}."} if cut else {})}
