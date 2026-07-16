"""
Tests for app/relevance_gate.py.

Covers:
  - Cheap gate: no chunks, or top raw score below CHEAP_GATE_MIN_SCORE,
    fails immediately without ever calling the (expensive) reranker.
  - Real gate: cross-encoder score above/below RERANK_GATE_THRESHOLD
    correctly flips relevance_gate_passed.
  - rerank() sigmoid-normalizes raw logits and sorts by score desc.
  - Only chunks with score >= threshold matter for the pass/fail decision
    (driven by the top reranked chunk's score).

The cross-encoder model itself is never loaded for real -- conftest.py
stubs `sentence_transformers.CrossEncoder` at import time, and these
tests further monkeypatch `_get_model()` per-test to return fully
deterministic scores instead of the stub's fixed 0.0 output.
"""
import math

import pytest

from app.relevance_gate import (
    CHEAP_GATE_MIN_SCORE,
    RERANK_GATE_THRESHOLD,
    relevance_gate_node,
    rerank,
)


class _FakeModel:
    """Returns a caller-supplied raw logit for each (query, text) pair,
    matched up positionally with the `scores` list passed in."""

    def __init__(self, scores):
        self.scores = scores

    def predict(self, pairs):
        assert len(pairs) == len(self.scores)
        return self.scores


def _patch_model(monkeypatch, scores):
    monkeypatch.setattr(
        "app.relevance_gate._get_model", lambda: _FakeModel(scores)
    )


class TestCheapGate:
    def test_no_chunks_fails_immediately(self):
        state = {"message": "anything", "retrieved_chunks": []}
        result = relevance_gate_node(state)
        assert result["relevance_gate_passed"] is False
        assert result["relevance_score"] == 0.0
        assert result["reranked_chunks"] == []

    def test_missing_retrieved_chunks_key_fails_immediately(self):
        # retrieval_node always sets this, but the gate should be
        # defensive if it's absent for any reason.
        state = {"message": "anything"}
        result = relevance_gate_node(state)
        assert result["relevance_gate_passed"] is False

    def test_low_top_score_fails_without_reranking(self, monkeypatch):
        called = False

        def _boom():
            nonlocal called
            called = True
            raise AssertionError("reranker should not be called")

        monkeypatch.setattr("app.relevance_gate._get_model", _boom)

        state = {
            "message": "q",
            "retrieved_chunks": [
                {"source": "a.pdf", "text": "irrelevant", "score": CHEAP_GATE_MIN_SCORE - 0.01}
            ],
        }
        result = relevance_gate_node(state)
        assert result["relevance_gate_passed"] is False
        assert called is False  # cheap gate short-circuited before rerank

    def test_score_exactly_at_cheap_threshold_proceeds_to_rerank(self, monkeypatch):
        # top_raw_score < CHEAP_GATE_MIN_SCORE is the fail condition, so a
        # score exactly equal to the threshold should proceed to rerank.
        _patch_model(monkeypatch, [10.0])  # sigmoid(10) ~= 1.0, passes real gate
        state = {
            "message": "q",
            "retrieved_chunks": [
                {"source": "a.pdf", "text": "some text", "score": CHEAP_GATE_MIN_SCORE}
            ],
        }
        result = relevance_gate_node(state)
        assert result["relevance_gate_passed"] is True


class TestRealGate:
    def test_high_rerank_score_passes(self, monkeypatch):
        _patch_model(monkeypatch, [8.0])  # sigmoid(8) ~ 0.9997
        state = {
            "message": "what is the refund policy",
            "retrieved_chunks": [
                {"source": "refund-policy.pdf", "text": "refund text", "score": 0.5}
            ],
        }
        result = relevance_gate_node(state)
        assert result["relevance_gate_passed"] is True
        assert result["relevance_score"] == pytest.approx(1 / (1 + math.exp(-8.0)))
        assert len(result["reranked_chunks"]) == 1

    def test_low_rerank_score_fails(self, monkeypatch):
        _patch_model(monkeypatch, [-8.0])  # sigmoid(-8) ~ 0.0003
        state = {
            "message": "what is the refund policy",
            "retrieved_chunks": [
                {"source": "refund-policy.pdf", "text": "unrelated text", "score": 0.5}
            ],
        }
        result = relevance_gate_node(state)
        assert result["relevance_gate_passed"] is False

    def test_score_just_below_threshold_fails(self, monkeypatch):
        # Find a raw logit whose sigmoid lands just under RERANK_GATE_THRESHOLD.
        target = RERANK_GATE_THRESHOLD - 0.01
        raw_logit = -math.log(1 / target - 1)
        _patch_model(monkeypatch, [raw_logit])
        state = {
            "message": "q",
            "retrieved_chunks": [{"source": "a.pdf", "text": "t", "score": 0.5}],
        }
        result = relevance_gate_node(state)
        assert result["relevance_gate_passed"] is False

    def test_score_just_above_threshold_passes(self, monkeypatch):
        target = RERANK_GATE_THRESHOLD + 0.01
        raw_logit = -math.log(1 / target - 1)
        _patch_model(monkeypatch, [raw_logit])
        state = {
            "message": "q",
            "retrieved_chunks": [{"source": "a.pdf", "text": "t", "score": 0.5}],
        }
        result = relevance_gate_node(state)
        assert result["relevance_gate_passed"] is True

    def test_multiple_chunks_sorted_by_rerank_score_desc(self, monkeypatch):
        # Three chunks, deliberately fed in low->high logit order; rerank()
        # should return them sorted high->low.
        _patch_model(monkeypatch, [-2.0, 5.0, 0.0])
        chunks = [
            {"source": "low.pdf", "text": "t1", "score": 0.5},
            {"source": "high.pdf", "text": "t2", "score": 0.5},
            {"source": "mid.pdf", "text": "t3", "score": 0.5},
        ]
        reranked = rerank("q", chunks, top_n=3)
        sources = [c["source"] for c in reranked]
        assert sources == ["high.pdf", "mid.pdf", "low.pdf"]
        # scores strictly decreasing
        scores = [c["rerank_score"] for c in reranked]
        assert scores == sorted(scores, reverse=True)

    def test_rerank_truncates_to_top_n(self, monkeypatch):
        _patch_model(monkeypatch, [1.0, 2.0, 3.0, 4.0])
        chunks = [{"source": f"{i}.pdf", "text": "t", "score": 0.5} for i in range(4)]
        reranked = rerank("q", chunks, top_n=2)
        assert len(reranked) == 2

    def test_rerank_empty_chunks_returns_empty(self):
        assert rerank("q", []) == []
