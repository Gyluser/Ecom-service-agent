"""100 条评估集的 Retriever-only chunk 级 RAG 评测。

用法：
  ..\learn-claude-code\LCC\Scripts\python.exe tests\test_rag_chunk_eval.py --backend numpy

本脚本不调用 EcomAgent，不进入 ReAct/工具循环，只执行：
  测试集校验 → gold chunk 映射 → Top-10 检索 → Hit@1/3/5/10、
  Recall@1（仅单证据查询）、All-query Recall@1（辅助）与 Recall@3/5/10 统计。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.agent.rag.backends import create_backend  # noqa: E402
from app.agent.rag.chunker import CHUNKING_VERSION, Chunk, chunk_markdown_dir  # noqa: E402
from app.agent.rag.embedder import Embedder  # noqa: E402
from app.agent.rag.hybrid import HybridRetriever  # noqa: E402
from app.agent.rag.retriever import KnowledgeRetriever  # noqa: E402
from app.agent.rag.reranker import LocalCrossEncoderReranker, RerankedRetriever  # noqa: E402
from app.config.settings import settings  # noqa: E402

CASES_PATH = ROOT / "app" / "evaluation" / "rag" / "data" / "chunk_cases_100.json"
KB_DIR = ROOT / "app" / "evaluation" / "rag" / "rag_knowledge"
INDEX_PATH = ROOT / "app" / "evaluation" / "rag" / "rag_index" / "kb_index.json"
CHROMA_DIR = ROOT / "app" / "evaluation" / "rag" / "rag_index" / "chroma"
RESULT_DIR = ROOT / "app" / "evaluation" / "rag" / "rag_result"
TOP_K = 10
METRIC_KS = (1, 3, 5, 10)
HYBRID_CANDIDATE_K = 10
LEAKAGE_MARKERS = ("检索测试问题", "测试问题示例")
EXPECTED_SOURCE_COUNTS = {
    "project_original": 4,
    "tau_bench_adapted": 15,
    "tau_bench_expanded": 31,
    "tau_bench_expanded_100": 50,
}
EXPECTED_TYPE_COUNTS = {
    "direct": 4,
    "paraphrase": 41,
    "ambiguous": 10,
    "hard_negative": 30,
    "multi_chunk": 15,
}


def _now_beijing() -> datetime:
    return datetime.now(timezone(timedelta(hours=8)))


def _normalize(value: str) -> str:
    return re.sub(r"\s+", "", value or "")


def _load_cases() -> list[dict[str, Any]]:
    payload = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    cases = payload.get("cases")
    if not isinstance(cases, list):
        raise ValueError("chunk_cases_100.json 缺少 cases 列表")
    if len(cases) != 100:
        raise ValueError(f"测试集数量应为 100，实际为 {len(cases)}")

    ids = [case.get("id") for case in cases]
    if len(set(ids)) != len(ids):
        raise ValueError("测试集 id 存在重复")
    for case in cases:
        is_unanswerable = "unanswerable" in case.get("query_type", [])
        if is_unanswerable and case.get("answerable") is not False:
            raise ValueError(f"{case.get('id')} 的 unanswerable 用例必须 answerable=false")
        if not is_unanswerable and case.get("answerable") is not True:
            raise ValueError(f"{case.get('id')} 的可回答用例必须 answerable=true")

    source_counts = Counter(case["source_scope"] for case in cases)
    if dict(source_counts) != EXPECTED_SOURCE_COUNTS:
        raise ValueError(f"source_scope 分布错误：{dict(source_counts)}")

    type_counts = Counter(
        query_type
        for case in cases
        for query_type in case.get("query_type", [])
    )
    if dict(type_counts) != EXPECTED_TYPE_COUNTS:
        raise ValueError(f"query_type 分布错误：{dict(type_counts)}")

    multi_flag_count = sum(case.get("requires_multi_chunk", False) for case in cases)
    multi_gold_count = sum(len(case.get("gold_evidence", [])) > 1 for case in cases)
    if multi_flag_count != 15:
        raise ValueError(f"requires_multi_chunk=true 应恰好为 15 条，实际为 {multi_flag_count}")
    if multi_gold_count != 15:
        raise ValueError(f"包含多条 gold_evidence 的用例应恰好为 15 条，实际为 {multi_gold_count}")

    required_fields = {
        "id",
        "source_scope",
        "query",
        "query_type",
        "answerable",
        "expected_docs",
        "gold_evidence",
        "requires_multi_chunk",
        "notes",
    }
    for case in cases:
        missing = required_fields - set(case)
        if missing:
            raise ValueError(f"{case.get('id')} 缺少字段：{sorted(missing)}")
        is_unanswerable = "unanswerable" in case["query_type"]
        if is_unanswerable:
            if case["expected_docs"] or case["gold_evidence"]:
                raise ValueError(
                    f"{case['id']} 的无法回答用例不能包含 expected_docs 或 gold_evidence"
                )
            if case["requires_multi_chunk"]:
                raise ValueError(f"{case['id']} 的无法回答用例不能要求多 chunk")
            continue
        if not case["gold_evidence"]:
            raise ValueError(f"{case['id']} 没有 gold_evidence")
        is_multi_gold = len(case["gold_evidence"]) > 1
        is_multi_type = "multi_chunk" in case["query_type"]
        if case["requires_multi_chunk"] != is_multi_gold:
            raise ValueError(
                f"{case['id']} 的 requires_multi_chunk 与 gold_evidence 数量不一致"
            )
        if case["requires_multi_chunk"] != is_multi_type:
            raise ValueError(
                f"{case['id']} 的 requires_multi_chunk 与 query_type 不一致"
            )
        for evidence in case["gold_evidence"]:
            if not isinstance(evidence.get("section_path"), list):
                raise ValueError(f"{case['id']} 的 section_path 必须是列表")
            if evidence.get("relevance", 0) < 1:
                raise ValueError(f"{case['id']} 的 gold_evidence relevance 必须 >= 1")
    return cases


def _make_embedder() -> Embedder:
    return Embedder(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        model=settings.embedding_model,
        backend=settings.embedding_backend,
        model_path=settings.embedding_model_path,
        device=settings.embedding_device,
    )


def _chunk_signature(chunk: Chunk) -> tuple[Any, ...]:
    return (
        chunk.chunk_id,
        chunk.doc,
        tuple(chunk.section_path),
        chunk.text,
        chunk.chunking_version,
    )


def _index_is_current(chunks: list[Chunk]) -> bool:
    if not INDEX_PATH.exists():
        return False
    try:
        payload = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
        indexed = [Chunk(**item) for item in payload.get("chunks", [])]
        return (
            payload.get("embedding_model") == settings.embedding_model
            and [_chunk_signature(c) for c in indexed]
            == [_chunk_signature(c) for c in chunks]
            and bool(payload.get("vectors"))
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _make_backend(name: str):
    if name == "numpy":
        return create_backend("numpy", index_path=INDEX_PATH)
    if name == "chroma":
        return create_backend(
            "chroma",
            persist_dir=CHROMA_DIR,
            collection_name="ecom_rag_evaluation",
        )
    raise ValueError(f"未知 backend：{name}")


def _prepare_retriever(name: str, chunks: list[Chunk]) -> tuple[KnowledgeRetriever, str]:
    embedder = _make_embedder()
    backend = _make_backend(name)
    action = "reused"

    if name == "numpy":
        if not _index_is_current(chunks):
            vectors = embedder.encode([chunk.text for chunk in chunks])
            backend.upsert(
                chunks=chunks,
                vectors=vectors,
                embedding_model=settings.embedding_model,
            )
            action = "rebuilt"
    else:
        try:
            backend.load()
            if backend.size() != len(chunks):
                raise ValueError("Chroma 评估索引 chunk 数量与当前语料不一致")
        except (FileNotFoundError, ValueError):
            vectors = embedder.encode([chunk.text for chunk in chunks])
            backend.upsert(
                chunks=chunks,
                vectors=vectors,
                embedding_model=settings.embedding_model,
            )
            action = "rebuilt"

    retriever = KnowledgeRetriever(embedder=embedder, backend=backend)
    retriever.load()
    return retriever, action


def _match_evidence(chunks: list[Chunk], evidence: dict[str, Any]) -> list[Chunk]:
    anchor = _normalize(evidence["text_anchor"])
    return [
        chunk
        for chunk in chunks
        if chunk.doc == evidence["doc"]
        and chunk.section_path == evidence["section_path"]
        and anchor in _normalize(chunk.text)
    ]


def _validate_corpus_and_gold(
    cases: list[dict[str, Any]], chunks: list[Chunk]
) -> dict[str, set[str]]:
    for chunk in chunks:
        if any(marker in chunk.text for marker in LEAKAGE_MARKERS):
            raise ValueError(f"检测到评估语料泄露章节：{chunk.chunk_id}")

    gold_ids: dict[str, set[str]] = {}
    for case in cases:
        if not case["answerable"]:
            gold_ids[case["id"]] = set()
            continue
        case_gold: set[str] = set()
        for evidence in case["gold_evidence"]:
            matches = _match_evidence(chunks, evidence)
            if not matches:
                raise ValueError(
                    f"{case['id']} 的 gold_evidence 无法匹配 chunk："
                    f"{evidence['doc']} / {evidence['section_path']} / {evidence['text_anchor']}"
                )
            evidence["matched_chunk_ids"] = [chunk.chunk_id for chunk in matches]
            case_gold.update(chunk.chunk_id for chunk in matches)
        gold_ids[case["id"]] = case_gold
    return gold_ids


def _metrics_for_ranks(
    relevant: list[bool], gold_count: int, answerable: bool
) -> dict[str, Any]:
    if not answerable:
        metrics = {f"Recall@{k}": None for k in METRIC_KS}
        metrics.update({f"Hit@{k}": None for k in METRIC_KS})
        metrics["supplementary_metrics"] = {"All-query Recall@1": None}
        return metrics
    retrieved_relevant = {k: sum(relevant[:k]) for k in METRIC_KS}
    recalls = {
        k: retrieved_relevant[k] / gold_count if gold_count else 0.0
        for k in METRIC_KS
    }
    metrics: dict[str, Any] = {}
    for k in METRIC_KS:
        metrics[f"Recall@{k}"] = recalls[k]
    for k in METRIC_KS:
        metrics[f"Hit@{k}"] = 1.0 if retrieved_relevant[k] else 0.0
    metrics["supplementary_metrics"] = {"All-query Recall@1": recalls[1]}
    return metrics


def _average_metric(case_results: list[dict[str, Any]], key: str) -> float | None:
    if not case_results:
        return None
    return round(
        sum(item["metrics"][key] for item in case_results) / len(case_results),
        4,
    )


def _average_metrics(case_results: list[dict[str, Any]]) -> dict[str, Any]:
    answerable_results = [item for item in case_results if item["answerable"]]
    single_evidence_results = [
        item
        for item in answerable_results
        if not item["requires_multi_chunk"]
    ]
    metrics: dict[str, Any] = {}
    for k in METRIC_KS:
        if k == 1:
            metrics["Recall@1"] = _average_metric(
                single_evidence_results,
                "Recall@1",
            )
        else:
            metrics[f"Recall@{k}"] = _average_metric(
                answerable_results,
                f"Recall@{k}",
            )
    for k in METRIC_KS:
        metrics[f"Hit@{k}"] = _average_metric(answerable_results, f"Hit@{k}")
    metrics["supplementary_metrics"] = {
        "All-query Recall@1": round(
            sum(
                item["metrics"]["supplementary_metrics"]["All-query Recall@1"]
                for item in answerable_results
            )
            / len(answerable_results),
            4,
        )
        if answerable_results
        else None
    }
    return metrics


def _run_cases(
    cases: list[dict[str, Any]],
    gold_ids: dict[str, set[str]],
    retriever: KnowledgeRetriever,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for case in cases:
        hits = retriever.search(case["query"], top_k=TOP_K)
        deduped = []
        seen: set[str] = set()
        for hit in hits:
            if hit.chunk.chunk_id not in seen:
                deduped.append(hit)
                seen.add(hit.chunk.chunk_id)

        retrieved_chunks = []
        relevant_flags = []
        for rank, hit in enumerate(deduped, start=1):
            is_relevant = hit.chunk.chunk_id in gold_ids[case["id"]]
            relevant_flags.append(is_relevant)
            retrieved_chunks.append(
                {
                    "rank": rank,
                    "chunk_id": hit.chunk.chunk_id,
                    "doc": hit.chunk.doc,
                    "section": hit.chunk.section,
                    "section_path": hit.chunk.section_path,
                    "score": round(hit.score, 6),
                    "is_relevant": is_relevant,
                }
            )

        result = {
            **case,
            "retrieved_chunks": retrieved_chunks,
            "metrics": _metrics_for_ranks(
                relevant_flags,
                len(gold_ids[case["id"]]),
                case["answerable"],
            ),
        }
        results.append(result)
    return results


def _group_summary(results: list[dict[str, Any]], field: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        values = result.get(field, []) if field == "query_type" else [result.get(field)]
        for value in values:
            groups.setdefault(value, []).append(result)
    return {
        key: {"count": len(items), "metrics": _average_metrics(items)}
        for key, items in sorted(groups.items())
    }


def _save_report(report: dict[str, Any]) -> Path:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    now = _now_beijing()
    filename = now.strftime("chunk_eval_%Y_%m-%d_%H%M.json")
    output = RESULT_DIR / filename
    suffix = 1
    while output.exists():
        output = RESULT_DIR / now.strftime(f"chunk_eval_%Y_%m-%d_%H%M_{suffix:02d}.json")
        suffix += 1
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def _make_pipeline(
    pipeline_name: str,
    retriever: KnowledgeRetriever,
    chunks: list[Chunk],
    reranker_model: str | None,
):
    if pipeline_name == "baseline":
        return retriever

    hybrid = HybridRetriever(
        retriever,
        chunks,
        candidate_k=HYBRID_CANDIDATE_K,
    )
    if pipeline_name == "hybrid":
        return hybrid
    if pipeline_name == "hybrid-rerank":
        if not reranker_model:
            raise ValueError(
                "pipeline=hybrid-rerank 必须提供 --reranker-model 本地模型目录"
            )
        reranker = LocalCrossEncoderReranker(
            reranker_model,
            device=settings.embedding_device,
        )
        return RerankedRetriever(
            hybrid,
            reranker,
            candidate_k=HYBRID_CANDIDATE_K,
        )
    raise ValueError(f"未知 pipeline: {pipeline_name}")


def run(
    backend_name: str,
    pipeline_name: str = "baseline",
    reranker_model: str | None = None,
) -> Path:
    cases = _load_cases()
    chunks = chunk_markdown_dir(KB_DIR)
    if not chunks:
        raise ValueError("评估知识库没有生成 chunk")
    gold_ids = _validate_corpus_and_gold(cases, chunks)
    retriever, index_action = _prepare_retriever(backend_name, chunks)
    pipeline = _make_pipeline(pipeline_name, retriever, chunks, reranker_model)
    results = _run_cases(cases, gold_ids, pipeline)

    report = {
        "test": "ecom_rag_chunk_eval_100",
        "started_at": _now_beijing().isoformat(),
        "backend": backend_name,
        "pipeline": pipeline_name,
        "top_k": TOP_K,
        "agent_loop": False,
        "index_action": index_action,
        "configuration": {
            "embedding_backend": settings.embedding_backend,
            "embedding_model": settings.embedding_model,
            "embedding_device": settings.embedding_device,
            "chunking_version": CHUNKING_VERSION,
            "knowledge_dir": str(KB_DIR),
            "index_path": str(INDEX_PATH),
        },
        "validation": {
            "leakage_markers_absent": True,
            "gold_evidence_unmatched": [],
            "query_type_counts": dict(Counter(qt for case in cases for qt in case["query_type"])),
        },
        "summary": {
            "total_queries": len(results),
            "answerable_queries": sum(item["answerable"] for item in results),
            "unanswerable_queries": sum(not item["answerable"] for item in results),
            "metrics": _average_metrics(results),
        },
        "by_source_scope": _group_summary(results, "source_scope"),
        "by_query_type": _group_summary(results, "query_type"),
        "cases": results,
    }
    output = _save_report(report)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"结果已保存：{output}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="100 条 RAG chunk-level Retriever 评测")
    parser.add_argument("--backend", choices=["numpy", "chroma"], default="numpy")
    parser.add_argument(
        "--pipeline",
        choices=["baseline", "hybrid", "hybrid-rerank"],
        default="baseline",
    )
    parser.add_argument(
        "--reranker-model",
        default=None,
        help="hybrid-rerank 使用的本地 Cross-Encoder 模型目录",
    )
    args = parser.parse_args()
    run(args.backend, args.pipeline, args.reranker_model)


if __name__ == "__main__":
    main()
