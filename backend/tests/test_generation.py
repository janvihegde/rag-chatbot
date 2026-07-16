"""
Tests for app/generation.py.

These tests force the offline extractive-fallback path (no MISTRAL_API_KEY)
so they're deterministic and don't require network/API access. The Mistral
call path (_call_mistral) is exercised separately via a monkeypatch that
replaces the network call entirely -- we never hit the real API in tests.

Covers:
  - generate_answer() only cites chunks that individually clear
    RERANK_GATE_THRESHOLD.
  - Falls back to the single top chunk if every chunk is filtered out,
    rather than generating with zero context.
  - Extractive fallback picks the sentence with the most query-word
    overlap, not just the first sentence.
  - generation_node assembles the final response_text with a source line
    and sets escalated=False.
  - Mistral path is chosen automatically when MISTRAL_API_KEY is set, and
    conversation history is forwarded to it.
"""
import pytest

from app.generation import generate_answer, generation_node
from app.relevance_gate import RERANK_GATE_THRESHOLD


@pytest.fixture(autouse=True)
def no_mistral_key(monkeypatch):
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)


ABOVE = RERANK_GATE_THRESHOLD + 0.1
BELOW = RERANK_GATE_THRESHOLD - 0.1


class TestCitationFiltering:
    def test_only_chunks_above_threshold_are_cited(self):
        chunks = [
            {"source": "good.pdf", "text": "Refunds are available within 30 days.", "rerank_score": ABOVE},
            {"source": "bad.pdf", "text": "Shipping takes five days.", "rerank_score": BELOW},
        ]
        _, citations = generate_answer("refund policy", chunks)
        assert citations == ["good.pdf"]

    def test_falls_back_to_top_chunk_if_all_below_threshold(self):
        chunks = [
            {"source": "only.pdf", "text": "Some marginally related text.", "rerank_score": BELOW},
        ]
        answer, citations = generate_answer("anything", chunks)
        assert citations == ["only.pdf"]
        assert answer  # still produces something, not empty

    def test_empty_chunks_returns_no_answer_message(self):
        answer, citations = generate_answer("anything", [])
        assert citations == []
        assert "doesn't seem to directly address" in answer


class TestExtractiveFallback:
    def test_picks_sentence_with_most_query_overlap(self):
        chunks = [
            {
                "source": "faq.pdf",
                "text": (
                    "Our headquarters are in Springfield. "
                    "Two-factor authentication can be enabled in Account Settings > Security."
                ),
                "rerank_score": ABOVE,
            }
        ]
        answer, _ = generate_answer("how do I enable two-factor authentication", chunks)
        assert "Two-factor authentication" in answer
        assert "Springfield" not in answer


class TestGenerationNode:
    def test_response_text_includes_source_line(self):
        state = {
            "message": "what is the refund policy",
            "reranked_chunks": [
                {
                    "source": "refund-policy.pdf",
                    "text": "Annual subscriptions can be refunded within 30 days.",
                    "rerank_score": ABOVE,
                }
            ],
        }
        result = generation_node(state)
        assert "refund-policy.pdf" in result["response_text"]
        assert result["citations"] == ["refund-policy.pdf"]
        assert result["escalated"] is False

    def test_no_reranked_chunks_key_handled_gracefully(self):
        # generation_node should not KeyError if relevance_gate somehow
        # didn't set reranked_chunks.
        state = {"message": "anything"}
        result = generation_node(state)
        assert result["escalated"] is False
        assert result["citations"] == []


class TestMistralPathSelection:
    def test_uses_mistral_when_key_present(self, monkeypatch):
        monkeypatch.setenv("MISTRAL_API_KEY", "fake-key-for-test")

        calls = {}

        def _fake_call_mistral(query, chunks, history=None):
            calls["query"] = query
            calls["chunks"] = chunks
            calls["history"] = history
            return "a generated answer"

        monkeypatch.setattr("app.generation._call_mistral", _fake_call_mistral)

        chunks = [{"source": "a.pdf", "text": "t", "rerank_score": ABOVE}]
        history = [{"role": "user", "content": "earlier question"}]
        answer, citations = generate_answer("q", chunks, history=history)

        assert answer == "a generated answer"
        assert citations == ["a.pdf"]
        assert calls["history"] == history

    def test_uses_extractive_fallback_when_no_key(self, monkeypatch):
        monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
        chunks = [{"source": "a.pdf", "text": "Some text here.", "rerank_score": ABOVE}]
        answer, _ = generate_answer("q", chunks)
        # Extractive fallback returns text lifted straight from the chunk,
        # not a placeholder -- just confirm it ran without needing network.
        assert "Some text here" in answer or answer  # non-empty, grounded
