from datetime import datetime, timedelta, timezone

from fpl_intelligence.research import Claim, named_players, save_claim
from fpl_intelligence.storage import Store, digest


def test_full_names_override_contained_surnames_and_ambiguous_first_names(market, tmp_path, monkeypatch):
    market[1] = market[1].model_copy(
        update={
            "name": "James Trafford",
            "web_name": "Trafford",
            "raw": {"first_name": "James", "second_name": "Trafford"},
        }
    )
    market[3] = market[3].model_copy(
        update={
            "name": "Reece James",
            "web_name": "James",
            "raw": {"first_name": "Reece", "second_name": "James"},
        }
    )
    assert named_players("James Trafford is available.", market) == {1}
    assert named_players("Reece James is available.", market) == {3}
    assert named_players("James is available.", market) == set()
    assert named_players("James Trafford and Reece James trained.", market) == {1, 3}
    monkeypatch.setattr("fpl_intelligence.ingestion.load_market", lambda *_: (market, {}))
    text = "James Trafford is available."
    known = datetime.now(timezone.utc)
    claim = Claim(
        player_id=1,
        claim_type="available",
        statement=text,
        evidence_span=text,
        source_url="https://www.mancity.com/news/test",
        source_hash=digest(text),
        known_at=known.isoformat(),
        expires_at=(known + timedelta(days=1)).isoformat(),
        status="confirmed",
        origin="test",
    )
    store = Store(tmp_path)
    identity = save_claim(store, claim, text)
    assert store.get("claim", identity)["player_id"] == 1


def test_unicode_names_and_word_boundaries_do_not_create_spurious_entities(market):
    for identity, name, web in [
        (1, "Martin Ødegaard", "Ødegaard"),
        (2, "Pascal Groß", "Groß"),
        (3, "Joshua King", "King"),
    ]:
        market[identity] = market[identity].model_copy(update={"name": name, "web_name": web})
    assert named_players("Martin Odegaard and Pascal Gross trained.", market) == {1, 2}
    assert named_players("Looking forward to the match.", market) == set()


def test_retrieval_isolates_article_and_keeps_its_publication_clock(tmp_path, monkeypatch):
    import asyncio

    from fpl_intelligence.research import retrieve

    html = """<html><nav>Reece James is ruled out.</nav><script type="application/ld+json">{"@type":"NewsArticle","datePublished":"2026-11-06T12:00:00+00:00"}</script><article><p>James Trafford is available.</p><div class="comments">unverified claim</div></article></html>"""

    async def get(*_, **__):
        return html

    monkeypatch.setattr("fpl_intelligence.transport.PublicTransport.get", get)
    source = asyncio.run(retrieve(Store(tmp_path), "https://www.mancity.com/news/test"))
    assert source["text"] == "James Trafford is available."
    assert source["published_at"] == "2026-11-06T12:00:00+00:00"


def test_unspecified_publication_zone_remains_a_range_and_stale_sources_fail_before_inference(
    tmp_path, monkeypatch
):
    import asyncio

    import pytest

    from fpl_intelligence.research import FreeCapability, publication_metadata, request_claims

    info = publication_metadata("2026-11-06T20:00:00")
    assert info["published_at"] is None
    assert info["publication_window"] == ["2026-11-06T06:00:00+00:00", "2026-11-07T10:00:00+00:00"]
    monkeypatch.setenv("GROQ_API_KEY", "")
    cap = FreeCapability(provider="groq", model="test", verified_until="2099-01-01T00:00:00Z")
    source = {"known_at": "2026-11-07T12:00:00+00:00", "published_at": "2026-11-06T20:00:00"}
    with pytest.raises(ValueError, match="key is not configured"):
        asyncio.run(request_claims(Store(tmp_path), source, cap, []))
    with pytest.raises(ValueError, match="stale"):
        asyncio.run(
            request_claims(Store(tmp_path), {**source, "published_at": "2026-11-01T20:00:00"}, cap, [])
        )


def test_club_article_body_excludes_related_stories_and_embedded_cards():
    from fpl_intelligence.intelligence import article_text

    html = """<article><h1>Current report</h1><section class="article-body"><p>Nico O’Reilly has a back problem.</p><section class="ff-article-content-card-wrapper">Haaland unavailable last month</section><aside>Unrelated transfer</aside></section><section class="article-section">Further reading: Foden ruled out</section><div class="most-read">Trafford injured</div></article>"""
    body = article_text(html, "https://www.mancity.com/news/test")
    assert body["text"] == "Nico O’Reilly has a back problem."
