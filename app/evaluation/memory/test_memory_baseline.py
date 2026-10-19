"""Run a deterministic baseline for the original Ecom Memory behavior.

Usage::

    python app/evaluation/memory/test_memory_baseline.py

The production Memory implementation is not changed here.  The baseline
models its current behavior: load all long-term facts for one user and expose
them in their persisted order.  Therefore Recall@k, Precision@k, and MRR are
measured against that order, not against a semantic retrieval algorithm.

Because this runner is offline, ``answer_accuracy`` means that all expected
answer facts are present in the original Memory prompt section.  It is an
evidence-coverage proxy, not an LLM final-answer score.
"""

from __future__ import annotations

import json
import sys
import subprocess
import tempfile
import types
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
DATASET = ROOT / "app" / "evaluation" / "data" / "cases_memory_20.json"
RESULT_DIR = ROOT / "app" / "evaluation" / "memory"
sys.path.insert(0, str(ROOT))

# The baseline only loads persisted Memory and never calls an LLM.  Keep this
# evaluator runnable in the lightweight Python environment by stubbing the
# type-only OpenAI import when the project dependency is unavailable.
if "openai" not in sys.modules:
    openai_stub = types.ModuleType("openai")
    openai_stub.OpenAI = object
    sys.modules["openai"] = openai_stub

from app.agent.memory.long_term import LongTermMemory  # noqa: E402


def _load_dataset() -> dict[str, Any]:
    data = json.loads(DATASET.read_text(encoding="utf-8"))
    if data.get("evaluation_mode") != "memory_only":
        raise ValueError("Memory 数据集必须使用 evaluation_mode=memory_only")
    cases = data.get("cases")
    if not isinstance(cases, list) or len(cases) != 20:
        raise ValueError(f"Memory 数据集应有 20 条用例，实际为 {len(cases or [])}")
    for case in cases:
        records = case.get("memory_records")
        if not isinstance(records, list) or len(records) < 10:
            raise ValueError(f"{case.get('case_id')} 至少需要 10 条 Memory 记录")
    return data


def _write_seed_memory(memory_dir: Path, user_id: str, records: list[dict]) -> None:
    memory_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "user_id": user_id,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "facts": [
            {
                "content": record["content"],
                "category": record.get("category", "other"),
                "created_at": record.get("created_at", ""),
                "source_session": record.get("source_session", ""),
            }
            for record in records
        ],
        "interaction_summaries": [],
    }
    path = memory_dir / f"{user_id}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_prompt(memory_dir: Path, user_id: str) -> tuple[LongTermMemory, str]:
    memory = LongTermMemory(user_id=user_id, memory_dir=str(memory_dir))
    memory.load()
    return memory, memory.build_prompt_section() or ""


def _retrieval_metrics(
    retrieved_ids: list[str],
    relevant_ids: list[str],
) -> dict[str, float]:
    relevant = set(relevant_ids)
    metrics: dict[str, float] = {}
    for k in (1, 5, 10):
        top_k = retrieved_ids[:k]
        hits = len(set(top_k) & relevant)
        metrics[f"recall_at_{k}"] = round(hits / len(relevant), 4) if relevant else 0.0
        metrics[f"precision_at_{k}"] = round(hits / k, 4) if k else 0.0

    reciprocal_rank = 0.0
    for rank, memory_id in enumerate(retrieved_ids, start=1):
        if memory_id in relevant:
            reciprocal_rank = 1.0 / rank
            break
    metrics["mrr"] = round(reciprocal_rank, 4)
    return metrics


def _answer_accuracy(
    prompt: str,
    required_keywords: list[str],
    forbidden_keywords: list[str],
) -> float:
    normalized = prompt.lower()
    required_ok = all(keyword.lower() in normalized for keyword in required_keywords)
    forbidden_ok = all(keyword.lower() not in normalized for keyword in forbidden_keywords)
    return float(required_ok and forbidden_ok)


def _process_restart_check(
    memory_dir: Path,
    user_id: str,
    expected_contents: set[str],
) -> bool:
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
            json.dumps(sorted(expected_contents), ensure_ascii=False),
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

    # This is the original Ecom order: all loaded facts, without query ranking.
    memory, prompt = _load_prompt(memory_dir, user_id)
    retrieved_ids = [record["memory_id"] for record in records]
    relevant_ids = case["relevant_memory_ids"]
    if relevant_ids:
        metrics = _retrieval_metrics(retrieved_ids, relevant_ids)
    else:
        # No-positive cases are excluded from Recall/Precision/MRR averages;
        # they use a separate false-positive avoidance check.
        metrics = {}
        metrics["no_relevant_avoidance"] = float(not retrieved_ids)

    if case.get("answer_accuracy_evaluable", True):
        metrics["answer_accuracy"] = _answer_accuracy(
            prompt,
            case.get("answer_required_keywords", []),
            case.get("answer_forbidden_keywords", []),
        )

    # Recreate the memory object to represent a new session/Agent instance.
    reloaded, reloaded_prompt = _load_prompt(memory_dir, user_id)
    expected_contents = {
        record["content"]
        for record in records
        if record["memory_id"] in set(case["relevant_memory_ids"])
    }
    persisted_contents = {fact.content for fact in reloaded.facts}
    metrics["cross_session_persistence"] = float(
        expected_contents.issubset(persisted_contents)
    )

    if case.get("process_restart_expected"):
        metrics["process_restart_persistence"] = float(
            _process_restart_check(memory_dir, user_id, expected_contents)
        )

    update_expectation = case.get("update_expectation")
    if update_expectation:
        record_by_id = {record["memory_id"]: record for record in records}
        latest_content = record_by_id[update_expectation["latest_memory_id"]]["content"]
        stale_contents = {
            record_by_id[memory_id]["content"]
            for memory_id in update_expectation.get("stale_memory_ids", [])
        }
        all_contents = {fact.content for fact in memory.facts}
        metrics["update_accuracy"] = float(
            latest_content in all_contents
            and not (stale_contents & all_contents)
        )

    result: dict[str, Any] = {
        "case_id": case["case_id"],
        "category": case.get("category", ""),
        "user_id": user_id,
        "query": case["query"],
        "retrieved_memory_ids": retrieved_ids,
        "relevant_memory_ids": case["relevant_memory_ids"],
        "metrics": metrics,
        "prompt_fact_count": len(memory.facts),
        "reloaded_fact_count": len(reloaded.facts),
        "answer_accuracy_mode": "offline_memory_evidence",
    }

    isolation = case.get("isolation_probe")
    if isolation:
        other_user_id = isolation["other_user_id"]
        other_records = isolation["other_user_records"]
        _write_seed_memory(memory_dir, other_user_id, other_records)
        _, other_prompt = _load_prompt(memory_dir, other_user_id)
        forbidden = isolation.get("forbidden_keywords", [])
        own_content = " ".join(record["content"] for record in records)
        user_a_clean = not any(word.lower() in prompt.lower() for word in forbidden)
        user_b_clean = not any(record["content"].lower() in other_prompt.lower() for record in records)
        metrics["user_isolation_rate"] = round((user_a_clean + user_b_clean) / 2, 4)
        result["isolation_checks"] = {
            "user_a_does_not_see_user_b": user_a_clean,
            "user_b_does_not_see_user_a": user_b_clean,
            "other_user_id": other_user_id,
            "own_fact_sample": own_content[:80],
        }

    return result


def _mean(results: list[dict[str, Any]], metric: str) -> float:
    values = [item["metrics"][metric] for item in results if metric in item["metrics"]]
    return round(sum(values) / len(values), 4) if values else 0.0


def main() -> Path:
    dataset = _load_dataset()
    case_results: list[dict[str, Any]] = []

    with tempfile.TemporaryDirectory(prefix="ecom_memory_baseline_") as temp_dir:
        memory_dir = Path(temp_dir)
        for case in dataset["cases"]:
            case_results.append(_case_result(case, memory_dir))

    summary_metrics = {
        "recall_at_1": _mean(case_results, "recall_at_1"),
        "recall_at_5": _mean(case_results, "recall_at_5"),
        "recall_at_10": _mean(case_results, "recall_at_10"),
        "precision_at_1": _mean(case_results, "precision_at_1"),
        "precision_at_5": _mean(case_results, "precision_at_5"),
        "precision_at_10": _mean(case_results, "precision_at_10"),
        "mrr": _mean(case_results, "mrr"),
        "answer_accuracy": _mean(case_results, "answer_accuracy"),
        "cross_session_persistence": _mean(case_results, "cross_session_persistence"),
        "user_isolation_rate": _mean(case_results, "user_isolation_rate"),
        "no_relevant_avoidance": _mean(case_results, "no_relevant_avoidance"),
        "update_accuracy": _mean(case_results, "update_accuracy"),
        "process_restart_persistence": _mean(case_results, "process_restart_persistence"),
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = RESULT_DIR / f"memory_baseline_{timestamp}.json"
    output = {
        "schema_version": "memory_baseline_result_v1",
        "evaluation_mode": "memory_only",
        "baseline_behavior": "original_full_memory_injection_order",
        "answer_accuracy_mode": "offline_memory_evidence",
        "case_count": len(case_results),
        "repeat_count": 1,
        "metrics": summary_metrics,
        "cases": case_results,
    }
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({"result": str(output_path), "metrics": summary_metrics}, ensure_ascii=False, indent=2))
    return output_path


if __name__ == "__main__":
    main()
