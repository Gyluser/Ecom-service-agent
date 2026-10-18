"""端到端测试：验证第 5 期 RAG 知识库检索（两套后端）。

测试场景（每套后端独立跑一遍）：
1. Markdown 切分正确性。
2. 索引构建（首次会调用 OpenAI Embeddings；已存在且模型一致则复用）。
3. KnowledgeRetriever Top-K 召回，与预期 doc 命中。
4. 可选的 Agent 端到端测试（默认不运行、不纳入 RAG 检索效果报告）。

前提：已配置 Embedding 后端；本地模式可直接使用已下载模型；chroma 后端需要 `pip install chromadb`。

用法：
  python tests/test_rag.py                # 跑两套后端的 RAG 检索效果
  python tests/test_rag.py --backend numpy
  python tests/test_rag.py --backend chroma
  python tests/test_rag.py --backend numpy --include-agent-e2e  # 可选调试项


. .\learn-claude-code\LCC\Scripts\Activate.ps1
cd .\ecom-service-agent
python tests\test_rag.py --backend numpy
"""

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.agent.chat import EcomAgent  # noqa: E402
from app.config.settings import settings  # noqa: E402
from app.agent.rag.backends import create_backend  # noqa: E402
from app.agent.rag.chunker import chunk_markdown_dir  # noqa: E402
from app.agent.rag.embedder import Embedder  # noqa: E402
from app.agent.rag.retriever import KnowledgeRetriever  # noqa: E402
from app.schemas.response import CustomerServiceResponse, IntentType  # noqa: E402
from app.agent.tools import knowledge as knowledge_tool  # noqa: E402

TEST_SESSION = str(ROOT / "app" / "sessions" / "test_rag_session.json")
RESULT_DIR = ROOT / "app" / "evaluation" / "rag_result"
EXPECTED_DOCS = {"退换货政策", "配送说明", "会员权益", "常见问题FAQ"}
RUN_REPORT: dict = {}
CURRENT_BACKEND_RESULT: dict | None = None


def _ok(msg: str):
    print(f"  ✅ {msg}")


def _fail(msg: str):
    if CURRENT_BACKEND_RESULT is not None:
        CURRENT_BACKEND_RESULT["status"] = "failed"
        CURRENT_BACKEND_RESULT.setdefault("errors", []).append(msg)
    print(f"  ❌ {msg}")
    sys.exit(1)


def _clean_session():
    Path(TEST_SESSION).unlink(missing_ok=True)


def _now_beijing() -> datetime:
    """返回北京时间，避免文件名依赖执行机器的系统时区。"""
    return datetime.now(timezone(timedelta(hours=8)))


def _save_report() -> Path:
    """保存本次 RAG 测试报告，不写入 API Key 等敏感配置。"""
    finished_at = _now_beijing()
    RUN_REPORT["finished_at"] = finished_at.isoformat()
    RUN_REPORT["summary"] = {
        item["backend"]: item["retrieval_summary"]
        for item in RUN_REPORT.get("backends", [])
        if "retrieval_summary" in item
    }
    RUN_REPORT["status"] = (
        "passed"
        if all(item.get("status") == "passed" for item in RUN_REPORT.get("backends", []))
        else "failed"
    )

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    filename = finished_at.strftime("rag_%Y_%m-%d_%H%M.json")
    output_path = RESULT_DIR / filename
    suffix = 1
    while output_path.exists():
        output_path = RESULT_DIR / finished_at.strftime(
            f"rag_%Y_%m-%d_%H%M_{suffix:02d}.json"
        )
        suffix += 1

    output_path.write_text(
        json.dumps(RUN_REPORT, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path


def _make_embedder() -> Embedder:
    return Embedder(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        model=settings.embedding_model,
        backend=settings.embedding_backend,
        model_path=settings.embedding_model_path,
        device=settings.embedding_device,
    )


def _make_backend(backend_name: str):
    if backend_name == "numpy":
        return create_backend("numpy", index_path=ROOT / settings.kb_index_path)
    if backend_name == "chroma":
        return create_backend(
            "chroma",
            persist_dir=ROOT / settings.chroma_persist_dir,
            collection_name=settings.chroma_collection,
        )
    raise ValueError(backend_name)


# ---------- 测试 1：切分 ----------
def test_chunker():
    print("\n  [1/5] Markdown 切分")
    chunks = chunk_markdown_dir(ROOT / settings.kb_dir)
    if not chunks:
        _fail("未切到任何 chunk")
    docs = {c.doc for c in chunks}
    missing = EXPECTED_DOCS - docs
    if missing:
        _fail(f"缺少文档: {missing}")
    for c in chunks:
        if not c.text.startswith("【"):
            _fail(f"chunk 文本未注入文档/章节标题: {c.chunk_id}")
    CURRENT_BACKEND_RESULT["chunker"] = {
        "status": "passed",
        "chunk_count": len(chunks),
        "documents": sorted(docs),
        "chunks": [
            {
                "chunk_id": c.chunk_id,
                "doc": c.doc,
                "section": c.section,
                "text_length": len(c.text),
            }
            for c in chunks
        ],
    }
    _ok(f"切分共 {len(chunks)} 个 chunk，覆盖文档 {docs}")


# ---------- 测试 2：构建/复用索引 ----------
def test_build_index(backend_name: str):
    print(f"\n  [2/5] 构建/加载向量索引（backend={backend_name}）")
    embedder = _make_embedder()
    backend = _make_backend(backend_name)

    try:
        backend.load()
        if backend.expected_embedding_model() == settings.embedding_model and backend.size() > 0:
            CURRENT_BACKEND_RESULT["index"] = {
                "status": "passed",
                "action": "reused",
                "size": backend.size(),
                "embedding_model": backend.expected_embedding_model(),
            }
            _ok(f"复用已有索引，{backend.size()} 条向量")
            return
    except FileNotFoundError:
        pass

    chunks = chunk_markdown_dir(ROOT / settings.kb_dir)
    vectors = embedder.encode([c.text for c in chunks])
    backend.upsert(
        chunks=chunks,
        vectors=vectors,
        embedding_model=settings.embedding_model,
    )
    CURRENT_BACKEND_RESULT["index"] = {
        "status": "passed",
        "action": "built",
        "size": backend.size(),
        "embedding_model": settings.embedding_model,
        "vector_dim": len(vectors[0]),
    }
    _ok(f"已构建索引：{backend.size()} chunks，维度 {len(vectors[0])}")


# ---------- 测试 3：Top-K 召回 ----------
def test_retriever_top_k(backend_name: str):
    print(f"\n  [3/5] KnowledgeRetriever Top-K 召回（backend={backend_name}）")
    embedder = _make_embedder()
    backend = _make_backend(backend_name)
    retriever = KnowledgeRetriever(embedder=embedder, backend=backend)
    retriever.load()

    cases = [
        ("七天无理由退货可以吗", "退换货政策"),
        ("钻石会员有哪些权益", "会员权益"),
        ("偏远地区还包邮吗", "配送说明"),
        ("忘记密码了怎么办", "常见问题FAQ"),
    ]
    retrieval_results = []
    for query, expected_doc in cases:
        hits = retriever.search(query, top_k=3)
        if not hits:
            _fail(f"query={query!r} 无召回")
        top = hits[0]
        expected_in_top_k = any(h.chunk.doc == expected_doc for h in hits)
        retrieval_results.append(
            {
                "query": query,
                "expected_doc": expected_doc,
                "top1": {
                    "chunk_id": top.chunk.chunk_id,
                    "doc": top.chunk.doc,
                    "section": top.chunk.section,
                    "score": round(top.score, 4),
                },
                "top_k": [
                    {
                        "chunk_id": h.chunk.chunk_id,
                        "doc": h.chunk.doc,
                        "section": h.chunk.section,
                        "score": round(h.score, 4),
                        "text": h.chunk.text,
                    }
                    for h in hits
                ],
                "top1_match": top.chunk.doc == expected_doc,
                "expected_in_top_k": expected_in_top_k,
            }
        )
        if top.chunk.doc != expected_doc:
            print(
                f"  ⚠️  query={query!r} top1 命中 {top.chunk.doc} / {top.chunk.section}"
                f"（期望 {expected_doc}），score={top.score:.3f}"
            )
            if not any(h.chunk.doc == expected_doc for h in hits):
                _fail(f"Top-3 都未命中预期文档 {expected_doc}")
        else:
            _ok(f"query={query!r} → {top.chunk.doc} / {top.chunk.section}（{top.score:.3f}）")
    CURRENT_BACKEND_RESULT["retrieval"] = {
        "status": "passed",
        "top_k": 3,
        "cases": retrieval_results,
    }

    ranks = []
    for case in retrieval_results:
        rank = next(
            (
                index
                for index, item in enumerate(case["top_k"], start=1)
                if item["doc"] == case["expected_doc"]
            ),
            None,
        )
        ranks.append(rank)

    top1_scores = [case["top1"]["score"] for case in retrieval_results]
    total = len(retrieval_results)
    CURRENT_BACKEND_RESULT["retrieval_summary"] = {
        "total_queries": total,
        "hit_at_1": sum(rank == 1 for rank in ranks) / total if total else 0.0,
        "hit_at_3": sum(rank is not None and rank <= 3 for rank in ranks) / total if total else 0.0,
        "mrr": sum(1 / rank for rank in ranks if rank is not None) / total if total else 0.0,
        "avg_top1_score": sum(top1_scores) / total if total else 0.0,
        "min_top1_score": min(top1_scores) if top1_scores else None,
        "max_top1_score": max(top1_scores) if top1_scores else None,
        "rank_of_expected_doc": ranks,
        "scope": "按预期文档判断，不等同于人工标注的chunk级相关性评估",
    }


# ---------- 可选测试：Agent 端到端 ----------
def test_agent_end_to_end(backend_name: str):
    print(f"\n  [可选] Agent 端到端：政策类问题触发 RAG（backend={backend_name}）")
    settings.rag_backend = backend_name
    knowledge_tool.reset_retriever()

    _clean_session()
    agent = EcomAgent(session_path=TEST_SESSION)
    agent.history_threshold = 30

    try:
        resp = agent.chat("我想了解一下你们的七天无理由退货是怎么算运费的？")
        if not isinstance(resp, CustomerServiceResponse):
            _fail("返回类型不正确")
        if resp.intent not in (
            IntentType.RETURN_REQUEST,
            IntentType.AFTER_SALE,
            IntentType.OTHER,
        ):
            print(f"  ⚠️  意图分类为 {resp.intent}（非退换货/售后，可接受）")
        if not resp.reply:
            _fail("reply 为空")

        triggered = any(
            m.get("role") == "assistant" and m.get("tool_calls")
            and any(tc["function"]["name"] == "search_knowledge" for tc in m["tool_calls"])
            for m in agent.raw_messages
        )
        if triggered:
            _ok("Agent 主动调用了 search_knowledge")
        else:
            print("  ⚠️  Agent 未触发 search_knowledge（模型可能直接回复）")

        keywords = ["运费", "顾客", "上门取件", "12 元", "12元"]
        if any(kw in resp.reply for kw in keywords):
            _ok("回复中包含政策原文关键信息")
        else:
            print(f"  ⚠️  回复中未匹配预期关键词，内容片段：{resp.reply[:120]}")
        CURRENT_BACKEND_RESULT["agent_end_to_end"] = {
            "status": "passed",
            "response_type": type(resp).__name__,
            "intent": resp.intent.value,
            "reply": resp.reply,
            "triggered_search_knowledge": triggered,
            "matched_keywords": [kw for kw in keywords if kw in resp.reply],
        }
    finally:
        agent.close()
        _clean_session()


def run_for_backend(backend_name: str, include_agent_e2e: bool = False):
    global CURRENT_BACKEND_RESULT
    print("\n" + "=" * 60)
    print(f"  Backend: {backend_name}")
    print("=" * 60)
    CURRENT_BACKEND_RESULT = {"backend": backend_name, "status": "running"}
    RUN_REPORT.setdefault("backends", []).append(CURRENT_BACKEND_RESULT)
    try:
        test_chunker()
        test_build_index(backend_name)
        test_retriever_top_k(backend_name)
        if include_agent_e2e:
            test_agent_end_to_end(backend_name)
        CURRENT_BACKEND_RESULT["status"] = "passed"
    finally:
        CURRENT_BACKEND_RESULT = None


def main():
    global RUN_REPORT
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["numpy", "chroma", "all"], default="all")
    parser.add_argument(
        "--include-agent-e2e",
        action="store_true",
        help="额外运行 Agent 端到端测试；默认只测试 RAG 检索效果",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  第 5 期 RAG · 端到端测试")
    print("=" * 60)

    backends = ["numpy", "chroma"] if args.backend == "all" else [args.backend]
    started_at = _now_beijing()
    RUN_REPORT = {
        "test": "ecom_rag",
        "started_at": started_at.isoformat(),
        "requested_backend": args.backend,
        "include_agent_e2e": args.include_agent_e2e,
        "configuration": {
            "embedding_backend": settings.embedding_backend,
            "embedding_model": settings.embedding_model,
            "embedding_device": settings.embedding_device,
            "kb_dir": settings.kb_dir,
            "kb_index_path": settings.kb_index_path,
            "chroma_persist_dir": settings.chroma_persist_dir,
            "chroma_collection": settings.chroma_collection,
        },
        "backends": [],
    }
    try:
        for name in backends:
            run_for_backend(name, include_agent_e2e=args.include_agent_e2e)
    finally:
        _clean_session()
        knowledge_tool.reset_retriever()
        output_path = _save_report()
        print(f"\n📄 RAG 测试结果已保存: {output_path}")

    print("\n" + "=" * 60)
    print(f"  🎉 全部测试通过（backends: {', '.join(backends)}）")
    print("=" * 60)


if __name__ == "__main__":
    main()
