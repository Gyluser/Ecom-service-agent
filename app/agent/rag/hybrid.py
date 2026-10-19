"""Dense + BM25 混合检索。

BM25 只在当前已加载的知识库 chunk 上建立内存索引，不读取评测问题或答案。
混合结果使用 RRF 融合，避免直接比较 Dense 与 BM25 不同量纲的分数。
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

from app.agent.rag.backends.base import RetrievedChunk
from app.agent.rag.chunker import Chunk
from app.agent.rag.retriever import KnowledgeRetriever

_TOKEN_RE = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]+", re.IGNORECASE)


def tokenize_for_bm25(text: str) -> list[str]:
    """面向中英文客服文本的轻量 tokenization。

    中文使用重叠二元/三元字符片段，英文和数字保留连续词，避免新增分词依赖。
    """
    tokens: list[str] = []
    for piece in _TOKEN_RE.findall((text or "").lower()):
        if re.fullmatch(r"[\u4e00-\u9fff]+", piece):
            if len(piece) <= 3:
                tokens.append(piece)
            else:
                tokens.extend(piece[i : i + 2] for i in range(len(piece) - 1))
                tokens.extend(piece[i : i + 3] for i in range(len(piece) - 2))
        else:
            tokens.append(piece)
    return tokens


class BM25Index:
    """小规模知识库的纯 Python BM25 索引。"""

    def __init__(
        self,
        chunks: list[Chunk],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        self.chunks = list(chunks)
        self.k1 = k1
        self.b = b
        self._tokens = [tokenize_for_bm25(chunk.text) for chunk in self.chunks]
        self._term_frequency = [Counter(tokens) for tokens in self._tokens]
        self._document_frequency: Counter[str] = Counter()
        for tokens in self._tokens:
            self._document_frequency.update(set(tokens))
        self._avgdl = (
            sum(len(tokens) for tokens in self._tokens) / len(self._tokens)
            if self._tokens
            else 0.0
        )

    def search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        if not self.chunks or top_k <= 0:
            return []
        query_terms = tokenize_for_bm25(query)
        if not query_terms:
            return []

        document_count = len(self.chunks)
        query_frequency = Counter(query_terms)
        scored: list[tuple[int, float]] = []
        for index, term_frequency in enumerate(self._term_frequency):
            doc_length = len(self._tokens[index])
            score = 0.0
            for term, qtf in query_frequency.items():
                tf = term_frequency.get(term, 0)
                if not tf:
                    continue
                df = self._document_frequency.get(term, 0)
                idf = math.log1p(
                    (document_count - df + 0.5) / (df + 0.5)
                )
                denominator = tf + self.k1 * (
                    1.0 - self.b + self.b * doc_length / max(self._avgdl, 1.0)
                )
                score += idf * (tf * (self.k1 + 1.0) / denominator) * min(qtf, 2)
            scored.append((index, score))

        scored.sort(key=lambda item: (-item[1], self.chunks[item[0]].chunk_id))
        return [
            RetrievedChunk(chunk=self.chunks[index], score=score)
            for index, score in scored[:top_k]
        ]


def reciprocal_rank_fusion(
    result_lists: list[list[RetrievedChunk]],
    *,
    rrf_k: int = 60,
) -> list[RetrievedChunk]:
    """按 chunk_id 融合多路结果，RRF 分数越高越靠前。"""
    chunks: dict[str, Chunk] = {}
    scores: defaultdict[str, float] = defaultdict(float)
    for results in result_lists:
        for rank, result in enumerate(results, start=1):
            chunk_id = result.chunk.chunk_id
            chunks[chunk_id] = result.chunk
            scores[chunk_id] += 1.0 / (rrf_k + rank)

    ordered_ids = sorted(
        chunks,
        key=lambda chunk_id: (-scores[chunk_id], chunk_id),
    )
    return [
        RetrievedChunk(chunk=chunks[chunk_id], score=scores[chunk_id])
        for chunk_id in ordered_ids
    ]


class HybridRetriever:
    """在已有 Dense retriever 外包一层 BM25 + RRF。"""

    def __init__(
        self,
        dense_retriever: KnowledgeRetriever,
        chunks: list[Chunk] | None = None,
        *,
        candidate_k: int = 10,
        rrf_k: int = 60,
    ):
        self.dense_retriever = dense_retriever
        self.candidate_k = max(1, candidate_k)
        self.rrf_k = rrf_k
        self.bm25 = BM25Index(chunks or dense_retriever.list_chunks())

    def search(self, query: str, top_k: int = 3) -> list[RetrievedChunk]:
        candidate_k = max(top_k, self.candidate_k)
        dense_results = self.dense_retriever.search(query, top_k=candidate_k)
        sparse_results = self.bm25.search(query, top_k=candidate_k)
        fused = reciprocal_rank_fusion(
            [dense_results, sparse_results],
            rrf_k=self.rrf_k,
        )
        return fused[:top_k]
