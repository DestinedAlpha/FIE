from datetime import timedelta

import numpy as np
import pytest

from fpl_intelligence.context_archive import check_proof
from fpl_intelligence.fusion_data import sequence, source_features, tokens
from fpl_intelligence.intelligence import article_text, feed_items, registered_url
from fpl_intelligence.storage import Store, now
from fpl_intelligence.workload import time_value, valid_fixture, workload_features


def test_context_requires_both_capture_and_commit_before_deadline():
    row = {
        "deadline": "2026-10-31T11:00:00Z",
        "source_captured_at": "2026-10-30T00:00:00Z",
        "git_committed_at": "2026-10-30T00:01:00Z",
        "commit": "abc",
        "url": "https://raw.githubusercontent.com/Randdalf/fplcache/abc/cache/2026/10/30/0000.json.xz",
    }
    check_proof(row)
    for field in ("git_committed_at", "source_captured_at"):
        with pytest.raises(ValueError, match="before its deadline"):
            check_proof(row | {field: row["deadline"]})
    with pytest.raises(ValueError, match="pinned"):
        check_proof(row | {"url": "https://other.example/abc"})


def test_workload_completion_boundary_and_explicit_bench_zero():
    row = {
        "player_code": 7,
        "kickoff": "2026-10-28T20:00:00Z",
        "known_after": "2026-10-29T00:00:00Z",
        "minutes": 0,
        "extra_time": 0,
    }
    assert workload_features([row], 7, row["known_after"]) == [0] * 7
    after = (time_value(row["known_after"]) + timedelta(seconds=1)).isoformat()
    values = workload_features([row], 7, after)
    assert values[0] == 0 and values[2] > 0 and values[-1] == 1
    assert workload_features([row], 8, after) == [0] * 7
    assert not valid_fixture({"match_id": "cup-2026-10-28", "kickoff_time": "2026-09-05T20:00Z"})[0]
    assert valid_fixture({"match_id": "opaque-competition-id", "kickoff_time": "2025-11-26T20:00+00:00"})[0]
    assert not valid_fixture({"match_id": "cup-2026-10-28"})[0]


def test_sequence_masks_missing_stats_instead_of_observed_zero():
    seq, mask = sequence([{"minutes": 0, "total_points": None}, {"minutes": 90, "total_points": 8}])
    assert mask.sum() == 2 and seq.shape == (12, 32)
    assert seq[-2, 0] == 0 and seq[-2, 16] == 1
    assert seq[-2, 22] == 0 and seq[-1, 22] == 1
    assert tokens("hamstring injury").tolist() == tokens("HAMSTRING injury").tolist()


def test_zero_expert_ownership_is_observed_when_sample_exists():
    _, news, market, load, quality = source_features({"status": "a"}, {"sample": 100}, [], 7, now())
    assert market[3] == 0 and market[6] == 1 and market[7] == 1
    assert load[-1] == 0 and quality[-1] == 0 and news[-1] == 1


def test_public_article_isolates_locked_material_and_forum_originals():
    html = '<article><div class="entry-content">Public injury fact<div class="members-only">secret</div><div class="comments">noise</div></div></article>'
    source = article_text(html, "https://www.fantasyfootballscout.co.uk/test")
    assert source["text"] == "Public injury fact"
    rss = """<feed xmlns="http://www.w3.org/2005/Atom"><entry><title>Lead</title><link href="https://www.reddit.com/r/fplAnalytics/comments/1"/><content type="html">&lt;a href="https://www.arsenal.com/news/test"&gt;original&lt;/a&gt;&lt;a href="https://evil.example/"&gt;other&lt;/a&gt;</content></entry></feed>"""
    assert feed_items(rss)[0]["original_urls"] == ["https://www.arsenal.com/news/test"]
    with pytest.raises(ValueError):
        registered_url("https://www.reddit.com/r/unrelated/.rss", True)
    with pytest.raises(ValueError):
        registered_url("https://www.arsenal.com.evil.example/news")


def test_prior_qualified_cohort_is_lagged_and_ignores_realized_multipliers(tmp_path):
    from fpl_intelligence.elite import elite_features

    store = Store(tmp_path)
    store.put(
        "elite_cohort",
        {
            "season": "2026/27",
            "selections": [
                {
                    "event": 8,
                    "entry": 1,
                    "player_id": 7,
                    "intended_xi": True,
                    "captain": True,
                    "effective_multiplier": 0,
                },
                {
                    "event": 8,
                    "entry": 2,
                    "player_id": 7,
                    "intended_xi": False,
                    "captain": False,
                    "effective_multiplier": 3,
                },
            ],
        },
    )
    rows, _ = elite_features(store)
    assert ("2026/27", 8, 7) not in rows
    assert rows["2026/27", 9, 7]["ownership"] == 1
    assert rows["2026/27", 9, 7]["intended_xi"] == 0.5
    assert rows["2026/27", 9, 7]["captain"] == 0.5


def test_neural_missing_sources_are_finite_and_weights_load_safely(tmp_path):
    torch = pytest.importorskip("torch")
    from fpl_intelligence.fusion import architecture, neural_predict, standardized

    torch.set_num_threads(1)
    torch.manual_seed(47)
    n = 3
    arrays = {
        "tab": np.zeros((n, 26), dtype=np.float32),
        "sequence": np.zeros((n, 12, 32), dtype=np.float32),
        "sequence_mask": np.zeros((n, 12), dtype=bool),
        "text": np.zeros((n, 40), dtype=np.int64),
        "news": np.zeros((n, 13), dtype=np.float32),
        "market": np.zeros((n, 9), dtype=np.float32),
        "workload": np.zeros((n, 7), dtype=np.float32),
        "quality": np.zeros((n, 4), dtype=np.float32),
    }
    x, stats = standardized(arrays, np.arange(n))
    model = architecture()
    a, b, c = neural_predict(model, x)
    assert a.shape == (n, 4) and b.shape == (n, 3) and np.isfinite(a).all()
    path = tmp_path / "weights.pt"
    torch.save(model.state_dict(), path)
    loaded = architecture()
    loaded.load_state_dict(torch.load(path, weights_only=True, map_location="cpu"))
    for actual, expected in zip(neural_predict(loaded, x), (a, b, c)):
        np.testing.assert_allclose(actual, expected, atol=1e-7)
    changed = {k: v.copy() for k, v in arrays.items()}
    changed["tab"][2] = 10000
    # Earlier fit statistics are explicitly reused, never re-fitted on inference.
    _, after = standardized(changed, np.arange(n), stats)
    assert after == stats


def test_claim_identity_rejects_valid_wrong_ids_and_ambiguous_subjects(tmp_path, market, monkeypatch):
    from datetime import datetime, timezone

    from fpl_intelligence.research import Claim, save_claim
    from fpl_intelligence.storage import digest

    monkeypatch.setattr("fpl_intelligence.ingestion.load_market", lambda store: (market, {}))
    store = Store(tmp_path)
    known = datetime.now(timezone.utc)
    text = "Player 1 is available. Player 2 has a hamstring injury."
    base = Claim(
        player_id=1,
        claim_type="available",
        statement="Player 1 is available.",
        evidence_span="Player 1 is available.",
        source_url="https://www.arsenal.com/news/test",
        source_hash=digest(text),
        known_at=known.isoformat(),
        expires_at=(known + timedelta(days=2)).isoformat(),
        status="reported",
        origin="test",
    )
    save_claim(store, base, text)
    with pytest.raises(ValueError, match="unambiguously"):
        save_claim(store, base.model_copy(update={"player_id": 2}), text)
    with pytest.raises(ValueError, match="unambiguously"):
        save_claim(store, base.model_copy(update={"evidence_span": text}), text)
    with pytest.raises(ValueError, match="identity disagree"):
        save_claim(store, base.model_copy(update={"statement": "Player 2 is available."}), text)


def test_official_injury_span_expands_only_to_exact_named_registry_line(tmp_path, market, monkeypatch):
    from datetime import datetime, timezone

    from fpl_intelligence.research import Claim, save_claim
    from fpl_intelligence.storage import digest

    monkeypatch.setattr("fpl_intelligence.ingestion.load_market", lambda store: (market, {}))
    text = "Official registry\nPlayer 1: status=d; official news: Hip injury - 75% chance of playing\nPlayer 2: fit"
    known = datetime.now(timezone.utc)
    claim = Claim(
        player_id=1,
        claim_type="available",
        statement="Player 1 has a 75% chance of playing.",
        evidence_span="Hip injury - 75% chance of playing",
        source_url="https://fantasy.premierleague.com/api/bootstrap-static/",
        source_hash=digest(text),
        known_at=known.isoformat(),
        expires_at=(known + timedelta(days=2)).isoformat(),
        status="reported",
        origin="test",
    )
    store = Store(tmp_path)
    identity = save_claim(store, claim, text)
    assert store.get("claim", identity)["evidence_span"] == text.splitlines()[1]


def test_update_merges_research_without_replacing_account_or_active_model(tmp_path):
    import gzip
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "update", Path(__file__).resolve().parents[1] / "scripts/install_update.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = tmp_path / "new"
    target = tmp_path / "existing"
    source.mkdir()
    target.mkdir()
    new = Store(source / "data")
    existing = Store(target / "data")
    existing.put("state", {"bank": 3})
    selected = existing.put("decision_model", {"id": "existing active model"})
    new.put("state", {"bank": 999})
    new.put("decision_model", {"id": "unwanted replacement"})
    model = new.put("fusion_model", {"architecture": "verified candidate"})
    sourcefile = new.root / "models" / "candidate.pt"
    sourcefile.parent.mkdir()
    sourcefile.write_bytes(b"tensor-only-model")
    with new.db() as db:
        db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    (new.root / "state.sqlite.gz").write_bytes(gzip.compress((new.root / "state.sqlite").read_bytes()))
    assert module.merge_research(source, target) == 1
    assert existing.get("state")["bank"] == 3
    assert existing.get("decision_model")["id"] == selected
    assert existing.get("fusion_model")["id"] == model
    assert (existing.root / "models/candidate.pt").read_bytes() == b"tensor-only-model"


def test_fusion_recovery_mixes_joint_mass_without_double_injury_adjustment(tmp_path, market, monkeypatch):
    import fpl_intelligence.fusion as module

    store = Store(tmp_path)
    model = store.put("fusion_model", {"version": "test"})
    inputs = {"news": np.zeros((2, 13), dtype=np.float32), "text": np.zeros((2, 40), dtype=np.int64)}
    monkeypatch.setattr("fpl_intelligence.fusion_data.live_inputs", lambda *args: inputs)

    def predict(store, spec, values, kind):
        healthy = values["news"][:, 0] == 1
        p = np.zeros((2, 4))
        p[:, 0] = np.where(healthy, 0, 0.5)
        p[:, 3] = 1 - p[:, 0]
        c = np.zeros((2, 4))
        c[:, 3] = np.where(healthy, 6, 2)
        return p, c, p[:, 3]

    monkeypatch.setattr(module, "candidate_predict", predict)
    rows = [{"player_id": 1, "event": event, "availability": 0.5} for event in (9, 14)]
    spec = {"fusion_model_id": model, "fusion_kind": "tree"}
    p, c, _ = module.production_predictions(store, spec, rows, {}, market)
    means = (p * c).sum(1)
    assert means[0] == pytest.approx(1)
    assert means[1] == pytest.approx(np.exp(-1) * 1 + (1 - np.exp(-1)) * 6)
    assert all(r["availability"] == 1 for r in rows)
    scenario = {"players": [{"player_id": 1, "from_event": 14, "availability": 0.5}]}
    p, c, _ = module.production_predictions(store, spec, rows, {}, market, scenario)
    assert (p[1] * c[1]).sum() * rows[1]["availability"] == pytest.approx(3)


def test_fusion_promotion_cannot_skip_decision_or_uncertainty_gates(tmp_path):
    from fpl_intelligence.fusion import promote_fusion

    store = Store(tmp_path)
    identity = store.put("fusion_model", {"selected": "tree"})
    store.put("fusion_decisions", {"fusion_model_id": identity, "promotable": False})
    store.put("fusion_uncertainty", {"fusion_model_id": identity, "passed": True})
    with pytest.raises(ValueError, match="promotion gates"):
        promote_fusion(store)
    assert not store.list("decision_model")


def test_injury_concern_is_not_displayed_as_confirmed_future_absence(tmp_path, market, monkeypatch):
    from datetime import datetime, timezone

    from fpl_intelligence.research import Claim, save_claim
    from fpl_intelligence.storage import digest

    monkeypatch.setattr("fpl_intelligence.ingestion.load_market", lambda store: (market, {}))
    text = "Player 1 limped off at half time."
    known = datetime.now(timezone.utc)
    claim = Claim(
        player_id=1,
        claim_type="unavailable",
        statement="Player 1 is definitely out next weekend.",
        evidence_span=text,
        source_url="https://www.arsenal.com/news/test",
        source_hash=digest(text),
        known_at=known.isoformat(),
        expires_at=(known + timedelta(days=2)).isoformat(),
        status="reported",
        origin="test",
    )
    store = Store(tmp_path)
    identity = save_claim(store, claim, text)
    saved = store.get("claim", identity)
    assert saved["claim_type"] == "injury_concern"
    assert saved["statement"] == text


def test_future_training_uses_frozen_local_predeadline_context(tmp_path):
    from fpl_intelligence.context_archive import add_local_context, capture_live_context, load_context
    from fpl_intelligence.storage import now

    stamp = time_value(now())
    packet = {
        "events": [{"id": 9, "is_next": True, "deadline_time": (stamp + timedelta(days=1)).isoformat()}],
        "elements": [{"id": 1, "code": 11, "news": "First known context"}],
    }
    store = Store(tmp_path)
    capture_live_context(store, packet)
    entries = add_local_context(store, [])
    assert len(entries) == 1
    store.put("context_archive", {"entries": entries})
    packet["elements"][0]["news"] = "Later revision"
    capture_live_context(store, packet)
    assert load_context(store)["2026/27", 9, 11]["news"] == "First known context"
    packet["events"][0]["deadline_time"] = (stamp - timedelta(seconds=1)).isoformat()
    assert capture_live_context(store, packet) is None
