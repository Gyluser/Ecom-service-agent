"""知识库检索工具：通过向量检索回答 FAQ、政策类问题。

与查订单/查物流这类「结构化数据查询」工具不同，
search_knowledge 面向非结构化文本（退换货政策、配送说明、FAQ 等），
返回 Top-K 命中片段及其来源，由 LLM 引用回答。

向量后端由 settings.rag_backend 切换：
- numpy ：手写余弦相似度，零依赖，教学透明
- chroma：嵌入式向量数据库，HNSW 索引，工程代表性

检索流水线由 settings.rag_pipeline 切换：
- dense：仅向量召回
- hybrid：Dense + BM25，经 RRF 融合
- hybrid-rerank：Hybrid 先召回候选，再由本地 Cross-Encoder 精排

为避免每次进程启动都重建索引或加载模型，单例缓存 Retriever。
"""

from pathlib import Path
from typing import Optional

from app.config.settings import settings
from app.agent.rag.backends import create_backend
from app.agent.rag.embedder import Embedder
from app.agent.rag.hybrid import HybridRetriever
from app.agent.rag.retriever import KnowledgeRetriever
from app.agent.rag.reranker import LocalCrossEncoderReranker, RerankedRetriever

_retriever: Optional[object] = None
_effective_pipeline: Optional[str] = None
_pipeline_warning: Optional[str] = None


def _create_backend_from_settings():
    """根据 settings.rag_backend 创建对应后端实例。"""
    name = settings.rag_backend.lower()
    if name == "numpy":
        return create_backend("numpy", index_path=Path(settings.kb_index_path))
    if name == "chroma":
        return create_backend(
            "chroma",
            persist_dir=Path(settings.chroma_persist_dir),
            collection_name=settings.chroma_collection,
        )
    raise ValueError(
        f"未知的 RAG 后端: {settings.rag_backend}（可选: numpy / chroma）"
    )


def _normalized_pipeline() -> str:
    pipeline = settings.rag_pipeline.strip().lower()
    if pipeline not in {"dense", "hybrid", "hybrid-rerank"}:
        raise ValueError(
            "未知的 RAG_PIPELINE: "
            f"{settings.rag_pipeline}（可选: dense / hybrid / hybrid-rerank）"
        )
    return pipeline


def _get_retriever():
    """创建线上检索流水线；reranker 不可用时安全降级为 Hybrid。"""
    global _retriever, _effective_pipeline, _pipeline_warning
    if _retriever is None:
        embedder = Embedder(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            model=settings.embedding_model,
            backend=settings.embedding_backend,
            model_path=settings.embedding_model_path,
            device=settings.embedding_device,
        )
        backend = _create_backend_from_settings()
        dense = KnowledgeRetriever(embedder=embedder, backend=backend)
        dense.load()

        pipeline = _normalized_pipeline()
        _pipeline_warning = None
        if pipeline == "dense":
            _retriever = dense
            _effective_pipeline = "dense"
        else:
            hybrid = HybridRetriever(
                dense,
                candidate_k=settings.rag_candidate_k,
                rrf_k=settings.rag_rrf_k,
            )
            if pipeline == "hybrid":
                _retriever = hybrid
                _effective_pipeline = "hybrid"
            else:
                try:
                    if not settings.reranker_model_path.strip():
                        raise ValueError("RERANKER_MODEL_PATH 未配置")
                    reranker = LocalCrossEncoderReranker(
                        settings.reranker_model_path,
                        device=settings.embedding_device,
                        batch_size=settings.reranker_batch_size,
                    )
                    _retriever = RerankedRetriever(
                        hybrid,
                        reranker,
                        candidate_k=settings.rag_candidate_k,
                    )
                    _effective_pipeline = "hybrid-rerank"
                except Exception as exc:
                    # 检索服务优先可用；同时把退化状态显式返回，避免误以为已精排。
                    _retriever = hybrid
                    _effective_pipeline = "hybrid"
                    _pipeline_warning = (
                        "hybrid-rerank 初始化失败，已降级为 hybrid："
                        f"{type(exc).__name__}: {exc}"
                    )
    return _retriever


def reset_retriever() -> None:
    """清空单例缓存（测试或切换后端时使用）。"""
    global _retriever, _effective_pipeline, _pipeline_warning
    _retriever = None
    _effective_pipeline = None
    _pipeline_warning = None


def search_knowledge(query: str, top_k: int = 3) -> dict:
    """检索退换货政策、配送说明、会员权益、FAQ 等知识库内容。

    Returns:
        {
          "success": bool,
          "backend": "numpy" | "chroma",
          "pipeline": "dense" | "hybrid" | "hybrid-rerank",
          "query": str,
          "results": [
            {
              "chunk_id": "...",
              "doc": "...",
              "section": "...",
              "section_path": ["..."],
              "parent_id": "...",
              "source": "...",
              "chunk_type": "...",
              "score": 0.83,
              "text": "..."
            },
            ...
          ],
          "error": "..."  # 仅失败时存在
        }
    """
    if not query or not query.strip():
        return {"success": False, "error": "query 不能为空", "query": query, "results": []}

    try:
        retriever = _get_retriever()
    except FileNotFoundError as e:
        return {
            "success": False,
            "error": str(e),
            "backend": settings.rag_backend,
            "query": query,
            "results": [],
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"知识库初始化失败: {e}",
            "backend": settings.rag_backend,
            "query": query,
            "results": [],
        }

    top_k = max(1, min(int(top_k or 3), 5))
    hits = retriever.search(query, top_k=top_k)

    payload = {
        "success": True,
        "backend": settings.rag_backend,
        "pipeline": _effective_pipeline,
        "query": query,
        "results": [
            {
                "chunk_id": h.chunk.chunk_id,
                "doc": h.chunk.doc,
                "section": h.chunk.section,
                "section_path": h.chunk.section_path,
                "parent_id": h.chunk.parent_id,
                "source": h.chunk.source,
                "chunk_type": h.chunk.chunk_type,
                "score": round(h.score, 4),
                "text": h.chunk.text,
            }
            for h in hits
        ],
    }
    if _pipeline_warning:
        payload["warning"] = _pipeline_warning
    return payload
