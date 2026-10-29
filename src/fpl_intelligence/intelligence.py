"""Public specialist evidence and community leads with explicit influence paths."""

from __future__ import annotations

import asyncio
import re
import unicodedata
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import urljoin, urlparse
from xml.etree import ElementTree

import httpx
from bs4 import BeautifulSoup

from .ingestion import load_market
from .research import ALLOWED_DOMAINS, extract_with_fallback
from .settings import redact
from .storage import Store, atomic_json, now

SPECIALISTS = {"fantasyfootballscout.co.uk", "fantasyfootballgeek.co.uk", "fantasyfootballpundit.com"}
FEEDS = {
    "scout": ("https://www.fantasyfootballscout.co.uk/feed", "specialist"),
    "fpl_analytics": ("https://www.reddit.com/r/fplAnalytics/.rss?limit=15", "community_lead"),
    "fantasypl": ("https://www.reddit.com/r/FantasyPL/.rss?limit=15", "community_lead"),
}


def registered_url(url, community=False):
    p = urlparse(url)
    host = (p.hostname or "").removeprefix("www.")
    allowed = ALLOWED_DOMAINS | SPECIALISTS | ({"reddit.com"} if community else set())
    if p.scheme != "https" or p.username or p.password or p.port not in (None, 443) or host not in allowed:
        raise ValueError("URL is outside the registered public source boundary")
    if host == "reddit.com" and not any(
        p.path.startswith("/r/" + name + "/") for name in ("FantasyPL", "fplAnalytics")
    ):
        raise ValueError("Community source is outside the selected football communities")
    return url


async def fetch(client, url, community=False):
    for _ in range(3):
        registered_url(url, community)
        response = await client.get(url)
        if response.is_redirect:
            url = urljoin(url, response.headers["location"])
            continue
        response.raise_for_status()
        if len(response.content) > 2_000_000:
            raise ValueError("Source exceeds the byte budget")
        return response, url
    raise ValueError("Public source redirect limit exceeded")


def article_text(html, url):
    soup = BeautifulSoup(html, "html.parser")
    published = None
    modified = None
    for script in soup.select('script[type="application/ld+json"]'):
        try:
            body = __import__("json").loads(script.string or script.get_text())
            for node in body.get("@graph", [body]) if isinstance(body, dict) else body:
                if node.get("@type") in ("Article", "NewsArticle", "BlogPosting"):
                    published = node.get("datePublished")
                    modified = node.get("dateModified")
        except (ValueError, TypeError):
            pass
    root = soup.select_one(".entry-content") or soup.find("article")
    if root is None:
        raise ValueError("A public article body could not be isolated")
    for node in root.select(
        "script,style,noscript,form,.comments,.paywall-content,.members-only,.locked-content"
    ):
        node.decompose()
    links = [urljoin(url, a["href"]) for a in root.select("a[href]")]
    text = root.get_text(" ", strip=True)
    return {
        "url": url,
        "text": text[:60000],
        "published_at": published,
        "modified_at": modified,
        "links": links[:100],
        "known_at": now(),
        "source_tier": "specialist",
        "scope": "public article body only; comments and locked sections excluded",
    }


def feed_items(text):
    root = ElementTree.fromstring(text)
    atom = "{http://www.w3.org/2005/Atom}"
    result = []
    for node in root.findall(".//item") + root.findall(".//" + atom + "entry"):
        title = node.findtext("title") or node.findtext(atom + "title") or ""
        link = node.findtext("link")
        if not link:
            elem = node.find(atom + "link")
            link = elem.get("href") if elem is not None else None
        published = (
            node.findtext("pubDate") or node.findtext(atom + "published") or node.findtext(atom + "updated")
        )
        if published:
            try:
                published = parsedate_to_datetime(published).isoformat()
            except (TypeError, ValueError):
                try:
                    published = datetime.fromisoformat(published.replace("Z", "+00:00")).isoformat()
                except ValueError:
                    published = None
        originals = []
        content = node.findtext(atom + "content") or node.findtext("description") or ""
        for anchor in BeautifulSoup(content, "html.parser").select("a[href]"):
            candidate = anchor.get("href", "")
            try:
                registered_url(candidate)
            except ValueError:
                continue
            if candidate not in originals:
                originals.append(candidate)
        result.append(
            {"title": title, "url": link, "published_at": published, "original_urls": originals[:10]}
        )
    return result


async def refresh_intelligence(store: Store, limit=3):
    if not 1 <= limit <= 10:
        raise ValueError("Specialist article budget must be 1–10")
    players, market = load_market(store)
    health = []
    sources = []
    leads = []
    async with httpx.AsyncClient(
        timeout=25,
        follow_redirects=False,
        headers={"User-Agent": "FPL-Intelligence personal read-only football research"},
    ) as client:
        for identity, (url, kind) in FEEDS.items():
            try:
                response, final = await fetch(client, url, kind == "community_lead")
                items = feed_items(response.text)
                store.snapshot("intelligence_feed_" + identity, final, {"items": items})
                health.append(
                    {
                        "source": identity,
                        "url": final,
                        "status": "available",
                        "retrieved": len(items),
                        "usage": kind,
                    }
                )
                if kind == "community_lead":
                    for item in items:
                        lead = {
                            **item,
                            "source": identity,
                            "known_at": now(),
                            "numerical_effect": "None until its original evidence is verified",
                            "training_use": "No Reddit text is used to train model weights",
                        }
                        store.put("community_lead", lead)
                        leads.append(lead)
                    # Read only independently registered originals, never forum
                    # comments as model instructions or numerical labels.
                    original_urls = list(dict.fromkeys(u for item in items for u in item["original_urls"]))
                    for original in original_urls[:limit]:
                        try:
                            article, article_url = await fetch(client, original)
                            source = article_text(article.text, article_url)
                            source["discovered_via"] = identity
                            source["id"] = store.put("source_text", source)
                            sources.append(
                                {
                                    "url": article_url,
                                    "source_id": source["id"],
                                    "status": "original_retrieved",
                                    "numerical_effect": "Requires supported claim and validated mapping",
                                }
                            )
                        except (ValueError, httpx.HTTPError) as error:
                            sources.append(
                                {"url": original, "status": "unavailable", "reason": redact(error)[:160]}
                            )
                    continue
                selected = [
                    item
                    for item in items
                    if item["published_at"]
                    and datetime.fromisoformat(item["published_at"]) <= datetime.now(timezone.utc)
                ]
                for item in selected[:limit]:
                    try:
                        article, article_url = await fetch(client, item["url"])
                        source = article_text(article.text, article_url)
                        if not source["published_at"]:
                            source["published_at"] = item["published_at"]
                        source["id"] = store.put("source_text", source)
                        normalized = (
                            unicodedata.normalize("NFKD", source["text"].casefold())
                            .encode("ascii", "ignore")
                            .decode()
                        )
                        entities = []
                        for player in players.values():
                            aliases = {player.web_name, player.name, player.raw.get("second_name", "")}
                            names = [
                                unicodedata.normalize("NFKD", name.casefold())
                                .encode("ascii", "ignore")
                                .decode()
                                for name in aliases
                                if len(name) > 2
                            ]
                            if any(re.search(r"\b" + re.escape(name) + r"\b", normalized) for name in names):
                                entities.append(
                                    {
                                        "id": player.id,
                                        "name": player.name,
                                        "position": player.position,
                                        "club": player.team,
                                    }
                                )
                        if not entities:
                            sources.append({"url": article_url, "status": "no_resolved_players"})
                            continue
                        extraction = await extract_with_fallback(store, source, entities)
                        sources.append(
                            {
                                "url": article_url,
                                "source_id": source["id"],
                                "status": "extracted",
                                "source_tier": "specialist",
                                **extraction,
                            }
                        )
                    except (ValueError, httpx.HTTPError) as error:
                        sources.append(
                            {"url": item["url"], "status": "unavailable", "reason": redact(error)[:160]}
                        )
            except (ValueError, httpx.HTTPError, ElementTree.ParseError) as error:
                health.append(
                    {
                        "source": identity,
                        "url": url,
                        "status": "unavailable",
                        "reason": redact(error)[:160],
                        "usage": kind,
                    }
                )
    report = {
        "at": now(),
        "market_id": market["id"],
        "source_health": health,
        "specialist_sources": sources,
        "community_leads": len(leads),
        "claims": sum(len(r.get("claim_ids", [])) for r in sources),
        "influence_policy": "Community is a lead; supported specialist facts become timestamped context; speculative recommendations are not hard-coded point uplifts",
    }
    report["id"] = store.put("intelligence_refresh", report)
    atomic_json(Path("artifacts/evaluations/intelligence_coverage.json"), report)
    return report


async def refresh_sources(store: Store):
    from .elite import collect_elite
    from .workload import import_workload

    # Keep successful independent sources if another feed is unavailable.
    results = await asyncio.gather(
        collect_elite(store), import_workload(store), refresh_intelligence(store), return_exceptions=True
    )
    output = {}
    for kind, result in zip(("elite_cohort", "workload", "intelligence_refresh"), results):
        if isinstance(result, BaseException):
            output[kind] = {"status": "unavailable", "reason": redact(result)[:160]}
        else:
            output[kind] = {"status": "updated", "id": result["id"]}
    return output


def source_coverage(store: Store):
    fields = {
        "top_manager_history": (
            "retrieved_managers",
            "retrieved_manager_gameweeks",
            "complete_managers",
            "selection_event",
            "temporal_track",
            "at",
        ),
        "context_archive": ("retrieved_deadlines", "requested_deadlines", "temporal_track", "at"),
        "elite_cohort": (
            "retrieved_managers",
            "retrieved_manager_gameweeks",
            "cohort_definition",
            "feature_policy",
            "at",
        ),
        "workload": ("accepted_player_matches", "rejected_player_matches", "limits", "at"),
        "intelligence_refresh": ("source_health", "claims", "community_leads", "influence_policy", "at"),
        "fusion_features": ("coverage", "rows", "sources", "rules", "at"),
    }
    output = {}
    for kind, keys in fields.items():
        try:
            artifact = store.get(kind)
            output[kind] = {key: artifact[key] for key in keys if key in artifact}
        except ValueError:
            output[kind] = {"status": "not_collected"}
    return output
