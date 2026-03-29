#!/usr/bin/env python3
"""
web_research.py — multi-source web research for chitta-research.

Searches arXiv, bioRxiv, GitHub, and general web for a query.
Outputs structured JSON with findings to stdout.

Usage:
    python3 web_research.py --query "DiscussionRoom synthesis quality LLM"
    python3 web_research.py --query "..." --sources arxiv,biorxiv,github,web --limit 5

The output JSON is consumed by Adhvaryu as an observation artifact.
"""

import argparse
import json
import sys
import time
import urllib.request
import urllib.parse
import urllib.error
import xml.etree.ElementTree as ET

TIMEOUT = 15  # seconds per request


def search_arxiv(query: str, limit: int) -> list[dict]:
    q = urllib.parse.quote(query)
    url = (f"https://export.arxiv.org/api/query?search_query=all:{q}"
           f"&start=0&max_results={limit}&sortBy=relevance")
    try:
        with urllib.request.urlopen(url, timeout=TIMEOUT) as r:
            xml = r.read().decode()
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        root = ET.fromstring(xml)
        results = []
        for entry in root.findall("atom:entry", ns):
            title   = (entry.findtext("atom:title", "", ns) or "").strip().replace("\n", " ")
            summary = (entry.findtext("atom:summary", "", ns) or "").strip().replace("\n", " ")[:300]
            link    = next((a.get("href","") for a in entry.findall("atom:link", ns)
                            if a.get("rel") == "alternate"), "")
            published = (entry.findtext("atom:published", "", ns) or "")[:10]
            authors = [a.findtext("atom:name", "", ns)
                       for a in entry.findall("atom:author", ns)][:3]
            results.append({
                "source": "arxiv",
                "title": title,
                "summary": summary,
                "url": link,
                "published": published,
                "authors": authors,
            })
        return results
    except Exception as e:
        return [{"source": "arxiv", "error": str(e)}]


def search_biorxiv(query: str, limit: int) -> list[dict]:
    # bioRxiv search API
    q = urllib.parse.quote(query)
    url = f"https://api.biorxiv.org/details/biorxiv/2020-01-01/{time.strftime('%Y-%m-%d')}/0/{limit}"
    # bioRxiv doesn't have free text search in the details API, use the search endpoint
    search_url = f"https://www.biorxiv.org/search/{q}?limit={limit}&format=json"
    try:
        req = urllib.request.Request(
            search_url,
            headers={"User-Agent": "chitta-research/1.0 (scientific research tool)"}
        )
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            data = json.loads(r.read().decode())
        results = []
        for item in data.get("collection", [])[:limit]:
            results.append({
                "source": "biorxiv",
                "title": item.get("title", ""),
                "summary": (item.get("abstract", "") or "")[:300],
                "url": f"https://www.biorxiv.org/content/{item.get('doi','')}" if item.get("doi") else "",
                "published": item.get("date", ""),
                "authors": [a.get("name", "") for a in item.get("authors", {}).get("author", [])[:3]],
            })
        return results
    except Exception as e:
        # Fallback: try the content API with keyword search
        try:
            fallback_url = f"https://api.biorxiv.org/details/biorxiv/2023-01-01/{time.strftime('%Y-%m-%d')}/0/{limit}"
            req2 = urllib.request.Request(
                fallback_url,
                headers={"User-Agent": "chitta-research/1.0"}
            )
            with urllib.request.urlopen(req2, timeout=TIMEOUT) as r:
                data = json.loads(r.read().decode())
            # Filter by relevance to query terms
            terms = set(query.lower().split())
            results = []
            for item in data.get("collection", []):
                text = (item.get("title", "") + " " + item.get("abstract", "")).lower()
                if sum(1 for t in terms if t in text) >= 2:
                    results.append({
                        "source": "biorxiv",
                        "title": item.get("title", ""),
                        "summary": (item.get("abstract", "") or "")[:300],
                        "url": f"https://www.biorxiv.org/content/{item.get('doi','')}",
                        "published": item.get("date", ""),
                        "authors": [],
                    })
                    if len(results) >= limit:
                        break
            return results if results else [{"source": "biorxiv", "error": str(e)}]
        except Exception as e2:
            return [{"source": "biorxiv", "error": str(e2)}]


def search_github(query: str, limit: int) -> list[dict]:
    q = urllib.parse.quote(f"{query} in:readme,description")
    url = f"https://api.github.com/search/repositories?q={q}&per_page={limit}&sort=stars"
    try:
        req = urllib.request.Request(
            url,
            headers={
                "Accept": "application/vnd.github+json",
                "User-Agent": "chitta-research/1.0",
            }
        )
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            data = json.loads(r.read().decode())
        results = []
        for item in data.get("items", [])[:limit]:
            results.append({
                "source": "github",
                "title": item.get("full_name", ""),
                "summary": (item.get("description") or "")[:200],
                "url": item.get("html_url", ""),
                "published": (item.get("updated_at") or "")[:10],
                "stars": item.get("stargazers_count", 0),
                "language": item.get("language", ""),
            })
        return results
    except Exception as e:
        return [{"source": "github", "error": str(e)}]


def search_semantic_scholar(query: str, limit: int) -> list[dict]:
    """Semantic Scholar covers both CS and biology papers — useful complement to arXiv/bioRxiv."""
    q = urllib.parse.quote(query)
    url = (f"https://api.semanticscholar.org/graph/v1/paper/search"
           f"?query={q}&limit={limit}&fields=title,abstract,year,authors,externalIds,openAccessPdf")
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "chitta-research/1.0"}
        )
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            data = json.loads(r.read().decode())
        results = []
        for item in data.get("data", [])[:limit]:
            pdf_url = ""
            if item.get("openAccessPdf"):
                pdf_url = item["openAccessPdf"].get("url", "")
            doi = (item.get("externalIds") or {}).get("DOI", "")
            url_out = pdf_url or (f"https://doi.org/{doi}" if doi else "")
            results.append({
                "source": "semantic_scholar",
                "title": item.get("title", ""),
                "summary": (item.get("abstract") or "")[:300],
                "url": url_out,
                "published": str(item.get("year", "")),
                "authors": [a.get("name", "") for a in (item.get("authors") or [])[:3]],
            })
        return results
    except Exception as e:
        return [{"source": "semantic_scholar", "error": str(e)}]


def main():
    parser = argparse.ArgumentParser(description="Multi-source web research")
    parser.add_argument("--query", required=True)
    parser.add_argument("--sources", default="arxiv,semantic_scholar,github,biorxiv")
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()

    sources = [s.strip() for s in args.sources.split(",")]
    results = []

    for source in sources:
        if source == "arxiv":
            results.extend(search_arxiv(args.query, args.limit))
        elif source == "biorxiv":
            results.extend(search_biorxiv(args.query, args.limit))
        elif source == "github":
            results.extend(search_github(args.query, args.limit))
        elif source == "semantic_scholar":
            results.extend(search_semantic_scholar(args.query, args.limit))

    # Filter out error-only results if we have real results
    real = [r for r in results if "error" not in r]
    output = real if real else results

    print(json.dumps({
        "query": args.query,
        "sources_searched": sources,
        "result_count": len(output),
        "results": output,
    }, indent=2))


if __name__ == "__main__":
    main()
