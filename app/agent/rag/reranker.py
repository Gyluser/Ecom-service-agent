"""本地 Cross-Encoder reranker 适配器。

该模块只负责对已经召回的候选 chunk 打分，不负责初次召回；由知识工具组装进线上流水线。
模型必须通过本地路径提供，并强制 local_files_only，避免评测时隐式联网。
"""

from __future__ import annotations

from pathlib import Path

from app.agent.rag.backends.base import RetrievedChunk


class LocalCrossEncoderReranker:
    """使用 transformers SequenceClassification 模型对 query/chunk 对打分。"""

    def __init__(
        self,
        model_path: str | Path,
        *,
        device: str = "cpu",
        batch_size: int = 8,
        max_length: int = 512,
    ):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"reranker 模型目录不存在: {model_path}。"
                "请先准备本地 bge-reranker 模型。"
            )

        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "reranker 需要 torch 和 transformers，请先准备依赖。"
            ) from exc

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(
            str(model_path),
            local_files_only=True,
        )
        self._model = AutoModelForSequenceClassification.from_pretrained(
            str(model_path),
            local_files_only=True,
        )
        self._model.to(device)
        self._model.eval()
        self._device = device
        self._batch_size = batch_size
        self._max_length = max_length

    def rerank(
        self,
        query: str,
        candidates: list[RetrievedChunk],
        *,
        top_k: int,
    ) -> list[RetrievedChunk]:
        if not candidates:
            return []

        scored: list[tuple[RetrievedChunk, float]] = []
        for start in range(0, len(candidates), self._batch_size):
            batch = candidates[start : start + self._batch_size]
            encoded = self._tokenizer(
                [query] * len(batch),
                [item.chunk.text for item in batch],
                padding=True,
                truncation=True,
                max_length=self._max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(self._device) for key, value in encoded.items()}
            with self._torch.no_grad():
                logits = self._model(**encoded).logits
            values = logits.reshape(-1).detach().cpu().tolist()
            scored.extend(zip(batch, (float(value) for value in values)))

        scored.sort(key=lambda item: (-item[1], item[0].chunk.chunk_id))
        return [
            RetrievedChunk(chunk=item.chunk, score=score)
            for item, score in scored[:top_k]
        ]


class RerankedRetriever:
    """对任意已有 Retriever 的候选结果执行 Cross-Encoder 重排。"""

    def __init__(self, base_retriever, reranker: LocalCrossEncoderReranker, *, candidate_k: int = 10):
        self.base_retriever = base_retriever
        self.reranker = reranker
        self.candidate_k = max(1, candidate_k)

    def search(self, query: str, top_k: int = 3) -> list[RetrievedChunk]:
        candidates = self.base_retriever.search(
            query,
            top_k=max(top_k, self.candidate_k),
        )
        return self.reranker.rerank(query, candidates, top_k=top_k)
