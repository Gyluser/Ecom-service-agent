"""验证线上 search_knowledge 的流水线装配，而不加载真实模型。"""

from app.agent.rag.chunker import Chunk
from app.agent.rag.backends.base import RetrievedChunk
from app.agent.tools import knowledge
from app.config.settings import settings


class _FakeDense:
    def __init__(self, **_kwargs):
        self.chunk = Chunk(
            chunk_id="policy::return",
            doc="policy",
            section="return",
            section_path=["policy", "return"],
            text="七天无理由退货规则",
        )

    def load(self):
        return None

    def list_chunks(self):
        return [self.chunk]

    def search(self, _query, top_k=3):
        return [RetrievedChunk(self.chunk, 0.9)][:top_k]


class _FakeHybrid:
    def __init__(self, dense, **_kwargs):
        self.dense = dense

    def search(self, query, top_k=3):
        return self.dense.search(query, top_k)


class _FakeReranker:
    def __init__(self, *_args, **_kwargs):
        pass


class _FakeReranked:
    def __init__(self, base, _reranker, **_kwargs):
        self.base = base

    def search(self, query, top_k=3):
        return self.base.search(query, top_k)


def test_search_knowledge_uses_hybrid_rerank_pipeline(monkeypatch) -> None:
    monkeypatch.setattr(knowledge, "KnowledgeRetriever", _FakeDense)
    monkeypatch.setattr(knowledge, "HybridRetriever", _FakeHybrid)
    monkeypatch.setattr(knowledge, "LocalCrossEncoderReranker", _FakeReranker)
    monkeypatch.setattr(knowledge, "RerankedRetriever", _FakeReranked)
    monkeypatch.setattr(knowledge, "_create_backend_from_settings", lambda: object())
    monkeypatch.setattr(knowledge, "Embedder", lambda **_kwargs: object())
    monkeypatch.setattr(settings, "rag_pipeline", "hybrid-rerank")
    monkeypatch.setattr(settings, "reranker_model_path", "local-reranker")
    knowledge.reset_retriever()

    result = knowledge.search_knowledge("七天无理由退货")

    assert result["success"] is True
    assert result["pipeline"] == "hybrid-rerank"
    assert result["results"][0]["chunk_id"] == "policy::return"


def test_search_knowledge_degrades_to_hybrid_when_reranker_fails(monkeypatch) -> None:
    monkeypatch.setattr(knowledge, "KnowledgeRetriever", _FakeDense)
    monkeypatch.setattr(knowledge, "HybridRetriever", _FakeHybrid)
    monkeypatch.setattr(knowledge, "_create_backend_from_settings", lambda: object())
    monkeypatch.setattr(knowledge, "Embedder", lambda **_kwargs: object())
    monkeypatch.setattr(settings, "rag_pipeline", "hybrid-rerank")
    monkeypatch.setattr(settings, "reranker_model_path", "missing-reranker")

    class _BrokenReranker:
        def __init__(self, *_args, **_kwargs):
            raise FileNotFoundError("missing")

    monkeypatch.setattr(knowledge, "LocalCrossEncoderReranker", _BrokenReranker)
    knowledge.reset_retriever()

    result = knowledge.search_knowledge("七天无理由退货")

    assert result["pipeline"] == "hybrid"
    assert "已降级" in result["warning"]
