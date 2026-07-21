"""
AI Digest Data Collector — v2
==============================
Collects data from arXiv, HackerNews, and lab RSS feeds for three snapshots.

Snapshots (short windows, complete RSS):
  A: Jan 20–31 2026  — MCP governance, OpenAI vs Anthropic, Microsoft
  B: Feb 15–28 2026  — Claude Sonnet 4.6, SWE-bench 80.8%, sabotage report
  C: Mar  1–15 2026  — GPT-5.4, AlphaEvolve, MCP 97M installs

Usage:
  uv run collect

Output:
  data/snapshot_A_jan2026.json
  data/snapshot_B_feb2026.json
  data/snapshot_C_mar2026.json
"""

import json
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent

import feedparser
import requests

# ── Snapshot configuration ────────────────────────────────────────────────────

SNAPSHOTS = [
    {
        "id":            "A_jan2026",
        "label":         "Jan 20–31 2026 — MCP governance, Microsoft pivot, revenue race",
        "date_from":     "2026-01-20",
        "date_to":       "2026-01-31",
        "arxiv_from":    "20260120",
        "arxiv_to":      "20260131",
        "hn_from":       1737331200,
        "hn_to":         1738367999,
        "jordan_keywords": ["mcp", "linux foundation", "microsoft", "anthropic", "revenue", "governance", "enterprise"],
        "alex_keywords":   ["protocol", "sdk", "open source", "benchmark", "architecture", "inference"],
    },
    {
        "id":            "B_feb2026",
        "label":         "Feb 15–28 2026 — Claude Sonnet 4.6, SWE-bench 80.8%, sabotage report",
        "date_from":     "2026-02-15",
        "date_to":       "2026-02-28",
        "arxiv_from":    "20260215",
        "arxiv_to":      "20260228",
        "hn_from":       1739577600,
        "hn_to":         1740787199,
        "jordan_keywords": ["enterprise", "coding", "vendor", "adoption", "safety", "report", "revenue"],
        "alex_keywords":   ["swe-bench", "context window", "token", "computer use", "benchmark", "api", "sonnet"],
    },
    {
        "id":            "C_mar2026",
        "label":         "Mar 1–15 2026 — GPT-5.4, AlphaEvolve, MCP 97M installs",
        "date_from":     "2026-03-01",
        "date_to":       "2026-03-15",
        "arxiv_from":    "20260301",
        "arxiv_to":      "20260315",
        "hn_from":       1740787200,
        "hn_to":         1741996799,
        "jordan_keywords": ["gpt-5.4", "human baseline", "agent", "workforce", "standard", "install"],
        "alex_keywords":   ["osworld", "alphaevolve", "compute", "evolutionary", "coding agent", "mcp sdk"],
    },
]

# ── RSS feeds ─────────────────────────────────────────────────────────────────

RSS_FEEDS = {
    "anthropic":   "https://www.anthropic.com/news/rss.xml",
    "openai":      "https://openai.com/blog/rss.xml",
    "deepmind":    "https://deepmind.google/blog/rss.xml",
    "huggingface": "https://huggingface.co/blog/feed.xml",
    "langchain":   "https://blog.langchain.dev/rss/",
    "mistral":     "https://mistral.ai/news/rss.xml",
}

ARXIV_CATEGORIES = ["cs.AI", "cs.LG", "cs.CL"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def clamp(text: str, max_chars: int = 600) -> str:
    return text[:max_chars] + "..." if len(text) > max_chars else text


def fetch_article(url: str, timeout: int = 20) -> str:
    """Fetch full article text via Jina reader (handles JS-rendered pages)."""
    try:
        resp = requests.get(
            "https://r.jina.ai/" + url,
            timeout=timeout,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        resp.raise_for_status()
    except requests.RequestException:
        return ""

    text = resp.text

    # Append any GitHub / HuggingFace links found in the markdown
    links = re.findall(
        r'https?://(?:github\.com|huggingface\.co)/[\w\-\.]+/[\w\-\.]+',
        text,
    )
    unique_links = list(dict.fromkeys(links))
    if unique_links:
        text += "\n\nLinks: " + "  ".join(unique_links[:10])

    return text


def strip_html(text: str) -> str:
    """Strip HTML tags from text."""
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def parse_dt(date_str: str) -> datetime | None:
    """Parse a datetime string into a datetime object."""
    if not date_str:
        return None
    formats = [
        "%a, %d %b %Y %H:%M:%S %z",
        "%a, %d %b %Y %H:%M:%S GMT",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S.%f%z",
    ]
    for fmt in formats:
        try:
            dt = datetime.strptime(date_str.strip(), fmt)
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            continue
    return None


def in_window(date_str: str, date_from: str, date_to: str) -> bool:
    """Check if a date string falls within the specified date range."""
    dt = parse_dt(date_str)
    if not dt:
        return False
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    d_from = datetime.strptime(date_from, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    d_to   = datetime.strptime(date_to,   "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return d_from <= dt <= d_to


def score_relevance(
    items: list[dict],
    alex_keywords: list[str],
    jordan_keywords: list[str],
) -> None:
    """Adds alex_score and jordan_score to each item: keyword hit count per persona.
    Used by generate_digests.py to rank items before generation.
    """
    for item in items:
        haystack = (
            item.get("title", "") + " " +
            item.get("abstract", item.get("summary", ""))
        ).lower()
        item["alex_score"]   = sum(1 for kw in alex_keywords   if kw in haystack)
        item["jordan_score"] = sum(1 for kw in jordan_keywords if kw in haystack)


# ── Source 1: arXiv ───────────────────────────────────────────────────────────

def fetch_arxiv(snapshot: dict, max_per_cat: int = 15) -> list[dict]:
    """Fetches papers from arXiv API for specified categories and date range."""
    papers = []

    for cat in ARXIV_CATEGORIES:
        print(f"    arXiv [{cat}] ...")
        params = {
            "search_query": (
                f"cat:{cat} AND "
                f"submittedDate:[{snapshot['arxiv_from']}0000 "
                f"TO {snapshot['arxiv_to']}2359]"
            ),
            "sortBy":      "submittedDate",
            "sortOrder":   "descending",
            "max_results": max_per_cat,
        }
        try:
            resp = requests.get(
                "http://export.arxiv.org/api/query",
                params=params,
                timeout=30,
            )
            resp.raise_for_status()
        except requests.RequestException as e:
            print(f"      ✗ {e}")
            time.sleep(5)
            continue

        ns   = {"a": "http://www.w3.org/2005/Atom"}
        root = ET.fromstring(resp.text)

        for entry in root.findall("a:entry", ns):

            def txt(tag: str) -> str:
                el = entry.find(tag, ns)
                return el.text.strip() if el is not None and el.text else ""

            raw_id   = txt("a:id")
            arxiv_id = raw_id.split("/abs/")[-1].strip()
            abstract = txt("a:summary").replace("\n", " ")

            url = raw_id
            for link in entry.findall("a:link", ns):
                if link.get("type") == "text/html":
                    url = link.get("href", raw_id)
                    break

            # GitHub link sometimes appears in abstract (e.g. via Papers with Code)
            gh = re.search(r"https?://github\.com/[\w\-]+/[\w\-]+", abstract)

            authors = [
                a.find("a:name", ns).text
                for a in entry.findall("a:author", ns)
                if a.find("a:name", ns) is not None
            ]

            papers.append({
                "source":     "arxiv",
                "category":   cat,
                "arxiv_id":   arxiv_id,
                "url":        url,
                "github_url": gh.group(0) if gh else "",
                "title":      txt("a:title").replace("\n", " "),
                "abstract":   clamp(abstract),
                "authors":    authors[:5],
                "published":  txt("a:published"),
            })

        time.sleep(3)   # arXiv rate limit

    seen, unique = set(), []
    for p in papers:
        if p["arxiv_id"] not in seen:
            seen.add(p["arxiv_id"])
            unique.append(p)

    print(f"    → {len(unique)} papers")
    return unique


# ── Source 2: HackerNews ──────────────────────────────────────────────────────

def fetch_hackernews(snapshot: dict, max_results: int = 30) -> list[dict]:
    """Fetches papers from hackernews API for specified categories and date range."""
    posts = []

    for tag in ["ai", "machine-learning"]:
        print(f"    HackerNews [{tag}] ...")
        params = {
            "tags":          tag,
            "numericFilters": (
                f"created_at_i>{snapshot['hn_from']},"
                f"created_at_i<{snapshot['hn_to']},"
                f"points>30"
            ),
            "hitsPerPage":   max_results // 2,
            "attributesToRetrieve": "title,url,points,num_comments,created_at,objectID,author",
        }
        try:
            resp = requests.get(
                "https://hn.algolia.com/api/v1/search",
                params=params,
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
        except (requests.RequestException, json.JSONDecodeError) as e:
            print(f"      ✗ {e}")
            continue

        for hit in data.get("hits", []):
            hn_id = hit.get("objectID", "")
            posts.append({
                "source":       "hackernews",
                "hn_id":        hn_id,
                "url":          hit.get("url") or f"https://news.ycombinator.com/item?id={hn_id}",
                "title":        hit.get("title", ""),
                "points":       hit.get("points", 0),
                "num_comments": hit.get("num_comments", 0),
                "author":       hit.get("author", ""),
                "published":    hit.get("created_at", ""),
            })

        time.sleep(1)

    seen, unique = set(), []
    for p in sorted(posts, key=lambda x: x["points"], reverse=True):
        key = p["title"].lower()[:50]
        if key not in seen:
            seen.add(key)
            unique.append(p)

    print(f"    → {len(unique)} posts")
    return unique[:max_results]


# ── Source 3: RSS ─────────────────────────────────────────────────────────────

def fetch_rss(snapshot: dict, max_per_feed: int = 8) -> list[dict]:
    """For snapshots within the last 3 months RSS is complete. feedparser handles all major formats (RSS 2.0, Atom)."""
    posts = []

    for source, feed_url in RSS_FEEDS.items():
        print(f"    RSS [{source}] ...")
        try:
            feed = feedparser.parse(feed_url)
        except Exception as e:
            print(f"      ✗ {e}")
            continue

        count = 0
        for entry in feed.entries:
            pub = entry.get("published", entry.get("updated", ""))
            if not in_window(pub, snapshot["date_from"], snapshot["date_to"]):
                continue

            url = entry.get("link", "")
            content = fetch_article(url) if url else ""

            if not content:
                print(f"      ✗ empty content — skipping {entry.get('title', '')[:50]}")
                continue

            posts.append({
                "source":    f"blog_{source}",
                "url":       url,
                "title":     entry.get("title", "").strip(),
                "summary":   clamp(content, 3000),
                "published": pub,
                "tags":      [t.get("term", "") for t in entry.get("tags", [])],
            })

            count += 1
            if count >= max_per_feed:
                break

            time.sleep(1)

        time.sleep(0.5)

    print(f"    → {len(posts)} RSS posts")
    return posts


# ── Collect ───────────────────────────────────────────────────────────────────

def collect(snapshot: dict) -> dict:
    """Collects data from all sources for a given snapshot configuration."""
    print(f"\n{'═' * 62}")
    print(f"  Snapshot {snapshot['id']}: {snapshot['label']}")
    print(f"  Window: {snapshot['date_from']} → {snapshot['date_to']}")
    print(f"{'═' * 62}")

    arxiv = fetch_arxiv(snapshot)
    hn    = fetch_hackernews(snapshot)
    rss   = fetch_rss(snapshot)

    # Score relevance per persona — generate_digests.py uses these to rank items
    for items in [arxiv, hn, rss]:
        score_relevance(items, snapshot["alex_keywords"], snapshot["jordan_keywords"])

    return {
        "snapshot_id":  snapshot["id"],
        "label":        snapshot["label"],
        "date_from":    snapshot["date_from"],
        "date_to":      snapshot["date_to"],
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "persona_keywords": {
            "alex":   snapshot["alex_keywords"],
            "jordan": snapshot["jordan_keywords"],
        },
        "counts": {
            "arxiv": len(arxiv),
            "hn":    len(hn),
            "rss":   len(rss),
        },
        "arxiv": arxiv,
        "hn":    hn,
        "rss":   rss,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    out_dir = ROOT / "artifacts" / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("AI Digest Collector")
    print(f"Snapshots: {len(SNAPSHOTS)}")
    print(f"Sources: arXiv ({len(ARXIV_CATEGORIES)} categories) + HackerNews + RSS ({len(RSS_FEEDS)} feeds)")

    for snapshot in SNAPSHOTS:
        out_path = out_dir / f"snapshot_{snapshot['id']}.json"

        if out_path.exists():
            print(f"\n  {snapshot['id']} already exists — skipping")
            print(f"  Delete {out_path} to re-collect.")
            continue

        data = collect(snapshot)

        out_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False)
        )

        total = sum(data["counts"].values())
        print(f"\n✓  {out_path.name}")
        print(f"   arXiv {data['counts']['arxiv']} | "
              f"HN {data['counts']['hn']} | "
              f"RSS {data['counts']['rss']} | "
              f"Total {total}")

        time.sleep(2)

    print("\n\n✅ All snapshots collected → ./data/")
    print("Next step: uv run gen-trainset")


if __name__ == "__main__":
    main()
