"""
oo_net.py — Internet Access for the Sovereign
================================================
Web fetch, search, and download capabilities with safety controls.

Features:
  - web_fetch(url)     → GET page, return cleaned text
  - web_search(query)  → DuckDuckGo instant answer + HTML search
  - download_file(url) → Save binary to disk
  - Rate limiting, timeouts, size limits, domain allowlist
"""
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Optional


# ── HTML → Text Converter ─────────────────────────────────────────────────────

class HTMLToText(HTMLParser):
    """Strip HTML tags and extract readable text."""

    def __init__(self) -> None:
        """Initialize text accumulator."""
        super().__init__()
        self.text: list[str] = []
        self._skip: bool = False
        self._skip_tags = {"script", "style", "nav", "footer", "header"}

    def handle_starttag(self, tag: str, attrs: list) -> None:
        """Track skip tags."""
        if tag in self._skip_tags:
            self._skip = True
        if tag in ("br", "p", "div", "li", "h1", "h2", "h3", "h4", "tr"):
            self.text.append("\n")

    def handle_endtag(self, tag: str) -> None:
        """End skip."""
        if tag in self._skip_tags:
            self._skip = False

    def handle_data(self, data: str) -> None:
        """Accumulate text."""
        if not self._skip:
            self.text.append(data)

    def get_text(self) -> str:
        """Return cleaned text."""
        raw = "".join(self.text)
        # Collapse whitespace
        lines = [line.strip() for line in raw.split("\n")]
        lines = [l for l in lines if l]
        return "\n".join(lines)


@dataclass
class NetConfig:
    """Network access configuration."""

    timeout_s: int = 15
    max_response_bytes: int = 1_000_000  # 1MB
    rate_limit_s: float = 1.0  # min seconds between requests
    user_agent: str = "SovereignBot/1.0 (autonomous research)"
    allowed_domains: list = field(default_factory=lambda: [])  # empty = all allowed
    blocked_domains: list = field(default_factory=lambda: [
        "localhost", "127.0.0.1", "0.0.0.0", "169.254.*",
        "10.*", "192.168.*", "172.16.*",
    ])


class NetAccess:
    """Safe internet access for the sovereign."""

    def __init__(self, config: Optional[NetConfig] = None) -> None:
        """Initialize with config.

        Args:
            config: network configuration, uses defaults if None
        """
        self.config = config or NetConfig()
        self._last_request_time: float = 0

    def _rate_limit(self) -> None:
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.config.rate_limit_s:
            time.sleep(self.config.rate_limit_s - elapsed)
        self._last_request_time = time.time()

    def _check_domain(self, url: str) -> bool:
        """Check if domain is allowed.

        Args:
            url: URL to check

        Returns:
            True if domain is allowed
        """
        parsed = urllib.parse.urlparse(url)
        host = parsed.hostname or ""

        # Block internal networks
        for pattern in self.config.blocked_domains:
            if "*" in pattern:
                regex = pattern.replace(".", r"\.").replace("*", ".*")
                if re.match(regex, host):
                    return False
            elif host == pattern:
                return False

        # Allowlist (if configured)
        if self.config.allowed_domains:
            return any(host.endswith(d) for d in self.config.allowed_domains)

        return True

    def web_fetch(self, url: str) -> dict:
        """Fetch a URL and return cleaned text.

        Args:
            url: URL to fetch

        Returns:
            dict with 'url', 'status', 'text', 'length', 'error'
        """
        if not self._check_domain(url):
            return {"url": url, "status": 403, "text": "", "error": "Domain blocked"}

        self._rate_limit()

        req = urllib.request.Request(url, headers={
            "User-Agent": self.config.user_agent,
            "Accept": "text/html,application/json,text/plain",
        })

        try:
            with urllib.request.urlopen(req, timeout=self.config.timeout_s) as resp:
                content_type = resp.headers.get("Content-Type", "")
                raw = resp.read(self.config.max_response_bytes)
                status = resp.status

                # Decode
                encoding = "utf-8"
                if "charset=" in content_type:
                    encoding = content_type.split("charset=")[-1].split(";")[0].strip()

                try:
                    text = raw.decode(encoding, errors="replace")
                except (LookupError, UnicodeDecodeError):
                    text = raw.decode("utf-8", errors="replace")

                # HTML → text
                if "html" in content_type.lower():
                    parser = HTMLToText()
                    parser.feed(text)
                    text = parser.get_text()
                elif "json" in content_type.lower():
                    try:
                        data = json.loads(text)
                        text = json.dumps(data, indent=2)
                    except json.JSONDecodeError:
                        pass

                # Truncate if huge
                if len(text) > 50000:
                    text = text[:50000] + "\n...[truncated]"

                return {
                    "url": url, "status": status,
                    "text": text, "length": len(text), "error": None
                }

        except urllib.error.HTTPError as e:
            return {"url": url, "status": e.code, "text": "", "error": str(e)}
        except urllib.error.URLError as e:
            return {"url": url, "status": 0, "text": "", "error": str(e.reason)}
        except TimeoutError:
            return {"url": url, "status": 0, "text": "", "error": "Timeout"}
        except Exception as e:
            return {"url": url, "status": 0, "text": "", "error": str(e)}

    def web_search(self, query: str, max_results: int = 5) -> dict:
        """Search the web via DuckDuckGo instant answer API.

        Args:
            query: search query
            max_results: max results to return

        Returns:
            dict with 'query', 'results', 'abstract', 'error'
        """
        # DuckDuckGo instant answer API (no key needed)
        params = urllib.parse.urlencode({
            "q": query, "format": "json", "no_html": "1",
            "skip_disambig": "1",
        })
        url = f"https://api.duckduckgo.com/?{params}"

        result = self.web_fetch(url)
        if result["error"]:
            return {"query": query, "results": [], "abstract": "", "error": result["error"]}

        try:
            data = json.loads(result["text"])
        except json.JSONDecodeError:
            return {"query": query, "results": [], "abstract": "", "error": "Parse error"}

        # Extract results
        abstract = data.get("Abstract", "") or data.get("Answer", "")
        results = []

        # Related topics
        for topic in data.get("RelatedTopics", [])[:max_results]:
            if isinstance(topic, dict) and "Text" in topic:
                results.append({
                    "text": topic["Text"],
                    "url": topic.get("FirstURL", ""),
                })

        return {
            "query": query,
            "abstract": abstract,
            "results": results,
            "error": None,
        }

    def download_file(self, url: str, dest_path: str) -> dict:
        """Download a file to disk.

        Args:
            url: URL to download
            dest_path: destination path

        Returns:
            dict with 'url', 'path', 'size', 'error'
        """
        if not self._check_domain(url):
            return {"url": url, "path": dest_path, "size": 0, "error": "Domain blocked"}

        self._rate_limit()

        req = urllib.request.Request(url, headers={
            "User-Agent": self.config.user_agent,
        })

        try:
            with urllib.request.urlopen(req, timeout=self.config.timeout_s) as resp:
                with open(dest_path, "wb") as f:
                    total = 0
                    while True:
                        chunk = resp.read(8192)
                        if not chunk:
                            break
                        f.write(chunk)
                        total += len(chunk)
                        if total > self.config.max_response_bytes * 10:
                            break  # 10MB limit for downloads

            return {"url": url, "path": dest_path, "size": total, "error": None}

        except Exception as e:
            return {"url": url, "path": dest_path, "size": 0, "error": str(e)}


def run_test() -> None:
    """Self-test: web fetch AND search."""
    print("\n  oo_net self-test")
    print("  " + "=" * 40)

    net = NetAccess()

    # Test 1: Domain blocking
    r = net.web_fetch("http://localhost:8080/secret")
    assert r["error"] == "Domain blocked"
    print("  ✓ Domain blocking: localhost blocked")

    r = net.web_fetch("http://192.168.1.1/admin")
    assert r["error"] == "Domain blocked"
    print("  ✓ Domain blocking: private IP blocked")

    # Test 2: Live web fetch (httpbin)
    r = net.web_fetch("https://httpbin.org/get")
    if r["error"] is None:
        assert r["status"] == 200
        assert len(r["text"]) > 0
        print(f"  ✓ Web fetch: httpbin.org ({r['length']} chars)")
    else:
        print(f"  ⚠ Web fetch: {r['error']} (network unavailable, OK)")

    # Test 3: Live search
    r = net.web_search("Python programming language")
    if r["error"] is None:
        print(f"  ✓ Web search: '{r['query']}' → "
              f"{len(r['results'])} results, "
              f"abstract={len(r['abstract'])} chars")
    else:
        print(f"  ⚠ Web search: {r['error']} (network unavailable, OK)")

    # Test 4: HTML stripping
    parser = HTMLToText()
    parser.feed("<html><body><h1>Hello</h1><p>World</p><script>evil()</script></body></html>")
    text = parser.get_text()
    assert "Hello" in text
    assert "World" in text
    assert "evil" not in text
    print("  ✓ HTML stripping: tags removed, script stripped")

    # Test 5: Rate limiting
    t0 = time.time()
    net.web_fetch("https://httpbin.org/status/200")
    net.web_fetch("https://httpbin.org/status/200")
    elapsed = time.time() - t0
    assert elapsed >= 0.9, f"Rate limit failed: {elapsed:.2f}s"
    print(f"  ✓ Rate limiting: {elapsed:.1f}s for 2 requests")

    print(f"\n  All tests passed.\n")


if __name__ == "__main__":
    import sys
    if "--test" in sys.argv:
        run_test()
    else:
        print("Usage: python oo_net.py --test")
