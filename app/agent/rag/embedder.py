"""向量化封装：支持 OpenAI API 与本地 Sentence-Transformers 模型。

- 支持批量编码（list[str] → list[list[float]]）。
- 可配置 model 与 base_url（与 chat 模型共用一套 OpenAI 客户端配置）。
- 本地模式支持直接加载已下载的 BGE 等 Sentence-Transformers 模型。
- 返回原始 list[float]，由调用方决定如何持久化（这里用 json，不引入 numpy 依赖）。
"""

from typing import Iterable

from openai import OpenAI


class Embedder:
    """OpenAI 或本地 Sentence-Transformers Embeddings 同步封装。"""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str = "text-embedding-3-small",
        batch_size: int = 64,
        backend: str = "openai",
        model_path: str = "",
        device: str = "cpu",
    ):
        self._backend = backend.lower().strip()
        self._model = model
        self._batch_size = batch_size
        self._client = None
        self._local_model = None

        if self._backend == "openai":
            self._client = OpenAI(api_key=api_key, base_url=base_url)
        elif self._backend == "local":
            from sentence_transformers import SentenceTransformer

            source = model_path or model
            self._local_model = SentenceTransformer(
                source,
                device=device,
                local_files_only=True,
            )
        else:
            raise ValueError(
                f"不支持的 embedding backend: {backend}，可选值为 openai 或 local。"
            )

    @property
    def model(self) -> str:
        return self._model

    def encode(self, texts: Iterable[str]) -> list[list[float]]:
        """批量编码，自动按 batch_size 分批请求。"""
        texts = list(texts)
        if not texts:
            return []

        if self._backend == "local":
            vectors = self._local_model.encode(
                texts,
                batch_size=self._batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            return vectors.tolist()

        out: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            resp = self._client.embeddings.create(model=self._model, input=batch)
            out.extend(item.embedding for item in resp.data)
        return out

    def encode_one(self, text: str) -> list[float]:
        return self.encode([text])[0]
