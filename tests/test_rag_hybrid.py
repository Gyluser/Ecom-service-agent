"""Dense/BM25/RRF 的无模型单元测试。"""

from app.agent.rag.chunker import Chunk
from app.agent.rag.hybrid import BM25Index, reciprocal_rank_fusion
from app.agent.rag.backends.base import RetrievedChunk


def _chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc="test",
        section=chunk_id,
        text=text,
        section_path=["test", chunk_id],
    )


def test_bm25_prioritizes_exact_chinese_terms() -> None:
    chunks = [
        _chunk("generic", "普通商品支持七天无理由退货。"),
        _chunk("activated", "电子卡券一旦激活，不支持无理由退货。"),
    ]
    results = BM25Index(chunks).search("电子卡券激活后还能退吗", top_k=2)
    assert results[0].chunk.chunk_id == "activated"


def test_rrf_merges_results_by_chunk_id() -> None:
    first = _chunk("first", "first")
    second = _chunk("second", "second")
    merged = reciprocal_rank_fusion(
        [
            [RetrievedChunk(first, 0.9), RetrievedChunk(second, 0.8)],
            [RetrievedChunk(second, 0.9), RetrievedChunk(first, 0.8)],
        ],
        rrf_k=1,
    )
    assert {item.chunk.chunk_id for item in merged} == {"first", "second"}
    assert len(merged) == 2
