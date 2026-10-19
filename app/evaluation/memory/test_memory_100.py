"""Evaluate the 100 newly generated tau2 Retail-derived Memory cases.

Unlike the original 20-case baseline runner, this evaluator calls the current
Memory Retriever with each case query and records the actual ranked evidence.
It is fully offline and writes results under app/evaluation/memory/result.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
DATASET = ROOT / "app" / "evaluation" / "data" / "cases_memory_100.json"
RESULT_DIR = ROOT / "app" / "evaluation" / "memory" / "result"
sys.path.insert(0, str(ROOT))

if "openai" not in sys.modules:
    import types

    openai_stub = types.ModuleType("openai")
    openai_stub.OpenAI = object
    sys.modules["openai"] = openai_stub

from app.agent.memory.long_term import LongTermMemory  # noqa: E402


def _load_dataset() -> dict[str, Any]:
    data = json.loads(DATASET.read_text(encoding="utf-8"))
    if data.get("evaluation_mode") != "memory_only":
        raise ValueError("Memory 数据集必须使用 evaluation_mode=memory_only")
    cases = data.get("cases")
    if not isinstance(cases, list) or len(cases) != 100:
        raise ValueError(f"Memory 数据集应有 100 条用例，实际为 {len(cases or [])}")
    required_counts = {
        "single_fact_recall": 30,
        "multi_fact_recall": 20,
        "no_relevant_memory": 10,
        "memory_update_conflict": 15,
        "user_isolation": 15,
        "cross_session_persistence": 5,
        "process_restart_persistence": 5,
    }
    actual_counts = defaultdict(int)
    for case in cases:
        actual_counts[case.get("category", "")] += 1
        if len(case.get("memory_records", [])) != 10:
            raise ValueError(f"{case.get('case_id')} 必须包含 10 条 Memory 记录")
    if dict(actual_counts) != required_counts:
        raise ValueError(f"类别数量不符合要求: {dict(actual_counts)}")
    return data


def _write_seed_memory(memory_dir: Path, user_id: str, records: list[dict[str, Any]]) -> None:
    memory_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "user_id": user_id,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "facts": [
            {
                "memory_id": record["memory_id"],
                "content": record["content"],
                "category": record.get("category", "other"),
                "created_at": record.get("created_at", ""),
                "source_session": record.get("source_session", ""),
            }
            for record in records
        ],
        "interaction_summaries": [],
    }
    (memory_dir / f"{user_id}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _retrieval_metrics(retrieved_ids: list[str], relevant_ids: list[str]) -> dict[str, float]:
    relevant = set(relevant_ids)
    metrics: dict[str, float] = {}
    for k in (1, 5, 10):
        top_k = retrieved_ids[:k]
        hits = len(set(top_k) & relevant)
        metrics[f"recall_at_{k}"] = round(hits / len(relevant), 4) if relevant else 0.0
        metrics[f"precision_at_{k}"] = round(hits / k, 4)
    metrics["mrr"] = round(
        next((1.0 / rank for rank, value in enumerate(retrieved_ids, 1) if value in relevant), 0.0),
        4,
    )
    metrics["returned_count"] = float(len(retrieved_ids))
    return metrics


def _answer_evidence(prompt: str, case: dict[str, Any]) -> float | None:
    if not case.get("answer_accuracy_evaluable", False):
        return None
    normalized = prompt.lower()
    required = case.get("answer_required_keywords", [])
    forbidden = case.get("answer_forbidden_keywords", [])
    return float(
        all(keyword.lower() in normalized for keyword in required)
        and all(keyword.lower() not in normalized for keyword in forbidden)
    )


def _restart_check(memory_dir: Path, user_id: str, expected: set[str]) -> bool:
    child_code = """
import json
import sys
import types
from pathlib import Path
root = Path(sys.argv[1])
memory_dir = Path(sys.argv[2])
user_id = sys.argv[3]
expected = set(json.loads(sys.argv[4]))
sys.path.insert(0, str(root))
stub = types.ModuleType("openai")
stub.OpenAI = object
sys.modules["openai"] = stub
from app.agent.memory.long_term import LongTermMemory
memory = LongTermMemory(user_id=user_id, memory_dir=str(memory_dir))
memory.load()
actual = {fact.content for fact in memory.facts}
raise SystemExit(0 if expected.issubset(actual) else 1)
"""
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            child_code,
            str(ROOT),
            str(memory_dir),
            user_id,
            json.dumps(sorted(expected), ensure_ascii=False),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.returncode == 0


def _case_result(case: dict[str, Any], memory_dir: Path) -> dict[str, Any]:
    user_id = case["user_id"]
    records = case["memory_records"]
    _write_seed_memory(memory_dir, user_id, records)
    memory = LongTermMemory(user_id=user_id, memory_dir=str(memory_dir))
    memory.load()

    matches = memory.retrieve(case["query"], top_k=10)
    retrieved_ids = [match.fact.memory_id for match in matches]
    relevant_ids = case["relevant_memory_ids"]
    metrics = (
        {"no_relevant_avoidance": float(not retrieved_ids)}
        if not relevant_ids
        else _retrieval_metrics(retrieved_ids, relevant_ids)
    )
    prompt = memory.build_prompt_section_for_query(case["query"], top_k=10) or ""
    answer_evidence = _answer_evidence(prompt, case)
    if answer_evidence is not None:
        metrics["answer_accuracy"] = answer_evidence

    expected_contents = {
        record["content"]
        for record in records
        if record["memory_id"] in set(relevant_ids)
    }
    reloaded = LongTermMemory(user_id=user_id, memory_dir=str(memory_dir))
    reloaded.load()
    persisted_contents = {fact.content for fact in reloaded.facts}
    metrics["cross_session_persistence"] = float(expected_contents.issubset(persisted_contents))

    update = case.get("update_expectation")
    if update:
        record_by_id = {record["memory_id"]: record for record in records}
        actual = {fact.content for fact in memory.facts}
        latest = record_by_id[update["latest_memory_id"]]["content"]
        stale = {
            record_by_id[memory_id]["content"]
            for memory_id in update.get("stale_memory_ids", [])
        }
        metrics["update_accuracy"] = float(latest in actual and not stale.intersection(actual))

    if case["category"] == "process_restart_persistence":
        metrics["process_restart_persistence"] = float(
            _restart_check(memory_dir, user_id, expected_contents)
        )

    result: dict[str, Any] = {
        "case_id": case["case_id"],
        "category": case["category"],
        "user_id": user_id,
        "query": case["query"],
        "retrieved_memory_ids": retrieved_ids,
        "retrieval": [
            {"memory_id": match.fact.memory_id, "score": match.score, "rank": match.rank}
            for match in matches
        ],
        "relevant_memory_ids": relevant_ids,
        "metrics": metrics,
        "prompt_fact_count": len(memory.facts),
        "reloaded_fact_count": len(reloaded.facts),
        "source_bench": case.get("source_bench", {}),
    }

    isolation = case.get("isolation_probe")
    if isolation:
        other_user_id = isolation["other_user_id"]
        _write_seed_memory(memory_dir, other_user_id, isolation["other_user_records"])
        other = LongTermMemory(user_id=other_user_id, memory_dir=str(memory_dir))
        other.load()
        other_prompt = other.build_prompt_section_for_query(case["query"]) or ""
        forbidden = isolation.get("forbidden_keywords", [])
        user_a_clean = not any(word.lower() in prompt.lower() for word in forbidden)
        user_b_clean = not any(
            record["content"].lower() in other_prompt.lower()
            for record in records
        )
        metrics["user_isolation_rate"] = round((user_a_clean + user_b_clean) / 2, 4)
        result["isolation_checks"] = {
            "user_a_does_not_see_user_b": user_a_clean,
            "user_b_does_not_see_user_a": user_b_clean,
        }
    return result


def _mean(results: list[dict[str, Any]], metric: str) -> float:
    values = [case["metrics"][metric] for case in results if metric in case["metrics"]]
    return round(sum(values) / len(values), 4) if values else 0.0


def main() -> Path:
    dataset = _load_dataset()
    case_results: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="ecom_memory_100_") as temp_dir:
        memory_dir = Path(temp_dir)
        for case in dataset["cases"]:
            case_results.append(_case_result(case, memory_dir))

    metric_names = (
        "recall_at_1",
        "recall_at_5",
        "recall_at_10",
        "precision_at_1",
        "precision_at_5",
        "precision_at_10",
        "mrr",
        "returned_count",
        "answer_accuracy",
        "cross_session_persistence",
        "user_isolation_rate",
        "no_relevant_avoidance",
        "update_accuracy",
        "process_restart_persistence",
    )
    summary_metrics = {
        metric: _mean(case_results, metric)
        for metric in metric_names
    }
    category_metrics: dict[str, dict[str, float]] = {}
    categories = sorted({case["category"] for case in case_results})
    for category in categories:
        category_cases = [case for case in case_results if case["category"] == category]
        category_metrics[category] = {
            metric: _mean(category_cases, metric)
            for metric in metric_names
            if any(metric in case["metrics"] for case in category_cases)
        }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = RESULT_DIR / f"memory_100_{timestamp}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "schema_version": "memory_100_result_v1",
        "evaluation_mode": "memory_only",
        "retrieval_behavior": "query_aware_top_k_local_retriever",
        "dataset": str(DATASET),
        "case_count": len(case_results),
        "repeat_count": 1,
        "metrics": summary_metrics,
        "category_metrics": category_metrics,
        "cases": case_results,
    }
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"result": str(output_path), "metrics": summary_metrics}, ensure_ascii=False, indent=2))
    return output_path


if __name__ == "__main__":
    main()
