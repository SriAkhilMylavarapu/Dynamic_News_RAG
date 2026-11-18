from datetime import datetime, timedelta, timezone

from newsrag.retrieve import recency_decay, rrf


def test_rrf_rewards_agreement():
    scores = rrf([["a", "b", "c"], ["b", "a", "d"]])
    assert scores["a"] == scores["b"] > scores["c"] == scores["d"]


def test_recency_decay_half_life():
    now = datetime(2026, 11, 18, tzinfo=timezone.utc)
    assert recency_decay(now, now, 24) == 1.0
    assert abs(recency_decay(now - timedelta(hours=24), now, 24) - 0.5) < 1e-9
    assert recency_decay(now + timedelta(hours=1), now, 24) == 1.0  # clock skew doesn't boost


def test_convex_keeps_relevance_gaps():
    from newsrag.retrieve import convex, minmax

    assert minmax({"a": 3, "b": 1}) == {"a": 1.0, "b": 0.0}
    assert minmax({"a": 2}) == {"a": 1.0}
    fused = convex({"a": 10.0, "b": 9.0}, {"a": 0.8, "b": 0.1, "c": 0.0})
    assert fused["a"] == 1.0 and fused["b"] < 0.6 and fused["c"] == 0.0
