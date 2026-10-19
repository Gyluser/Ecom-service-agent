"""LCC-inspired Context compression evaluation.

运行：
    python tests/test_context_baseline.py

本测试专门使用独立 Context 数据集，不调用 agent.close()，并在每条用例中关闭
Memory。结果、session、transcript 和 tool result 统一写入 app/evaluation/context。
"""

from __future__ import annotations

import json
import contextlib
import io
import os
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DATASET = ROOT / "app" / "evaluation" / "data" / "cases_context_10.json"
RESULT_DIR = ROOT / "app" / "evaluation" / "context"
BASELINE_DIR = ROOT / "app" / "evaluation" / "result"


def _load_cases() -> list[dict[str, Any]]:
    data = json.loads(DATASET.read_text(encoding="utf-8"))
    if data.get("evaluation_mode") != "context_only":
        raise ValueError("Context 数据集 evaluation_mode 必须为 context_only")
    if data.get("memory_enabled") is not False:
        raise ValueError("Context 基线必须显式关闭 memory")
    cases = data.get("cases")
    if not isinstance(cases, list) or len(cases) != 10:
        raise ValueError(f"Context 数据集应有 10 条用例，实际为 {len(cases or [])}")
    return cases


def _contains_all(text: str, values: list[Any]) -> bool:
    """Match required keywords, allowing a list to represent alternatives."""
    return all(
        _expected_value_matches(text, value)
        for value in values
    )


def _expected_value_matches(text: str, value: Any) -> bool:
    if isinstance(value, list):
        return any(option in text for option in value)
    return bool(value) and value in text


def _text_matches(text: str, fact: dict[str, Any]) -> bool:
    candidates = [fact.get("content", ""), *fact.get("content_variants", [])]
    return any(candidate and candidate in text for candidate in candidates)


def _call_text(call: dict[str, Any]) -> str:
    return json.dumps(
        {
            "name": call.get("name"),
            "arguments": call.get("arguments", {}),
        },
        ensure_ascii=False,
    )


def _matches_arguments(call: dict[str, Any], expected: dict[str, Any]) -> bool:
    arguments = call.get("arguments", {})
    return all(arguments.get(key) == value for key, value in expected.items())


def _run_case_once(case: dict[str, Any], repeat_index: int, sandbox_root: Path) -> dict[str, Any]:
    from app.agent.chat import EcomAgent
    from app.config.settings import settings

    session_path = sandbox_root / "sessions" / f"{case['id']}_repeat_{repeat_index}.json"
    agent = None
    original_memory_enabled = settings.memory_enabled
    original_mcp_enabled = settings.mcp_enabled
    original_context_archive_dir = settings.context_archive_dir
    original_context_char_limit = settings.context_char_limit

    turn_calls: dict[int, list[dict[str, Any]]] = {}
    compression_events: list[dict[str, Any]] = []
    build_events: list[dict[str, Any]] = []
    llm_calls: list[dict[str, Any]] = []

    try:
        settings.memory_enabled = False
        settings.mcp_enabled = False
        settings.context_archive_dir = str(sandbox_root / "artifacts" / case["id"])
        # 评测使用较小的压力预算，确保 8~10 轮数据实际经过 LCC 管线；
        # 生产默认预算仍由 settings.context_char_limit=50000 控制。
        settings.context_char_limit = 18000
        agent = EcomAgent(session_path=str(session_path))

        if agent.memory_manager.memory_enabled:
            raise RuntimeError("Context 基线意外启用了 Memory")

        current_turn = {"index": 0}

        original_execute_tool = agent.tool_manager.execute_tool

        def observed_execute_tool(name: str, arguments: dict) -> str:
            result = original_execute_tool(name, arguments)
            turn_calls.setdefault(current_turn["index"], []).append({
                "name": name,
                "arguments": dict(arguments),
                "result": result,
            })
            return result

        agent.tool_manager.execute_tool = observed_execute_tool

        original_create = agent.client.chat.completions.create

        def observed_create(*args: Any, **kwargs: Any):
            messages = kwargs.get("messages", [])
            tool_definitions = kwargs.get("tools", []) or []
            tool_definition_chars = len(
                json.dumps(tool_definitions, ensure_ascii=False, default=str)
            )
            estimated_chars = agent.context_manager.estimate_chars(messages)
            call_record = {
                "purpose": getattr(agent.context_manager, "current_call_purpose", "react"),
                "message_count": len(messages),
                "estimated_chars": estimated_chars,
                "estimated_tokens": agent.context_manager.estimate_tokens(estimated_chars),
                "tool_definition_count": len(tool_definitions),
                "tool_definition_chars": tool_definition_chars,
                "tool_definition_tokens_est": agent.context_manager.estimate_tokens(
                    tool_definition_chars
                ) if tool_definitions else 0,
                "roles": [message.get("role") for message in messages],
                "tool_call_ids": [
                    tool_call.get("id")
                    for message in messages
                    if message.get("role") == "assistant"
                    for tool_call in message.get("tool_calls", [])
                ],
                "tool_result_ids": [
                    message.get("tool_call_id")
                    for message in messages
                    if message.get("role") == "tool"
                ],
                "latest_user_content": next(
                    (
                        message.get("content", "")
                        for message in reversed(messages)
                        if message.get("role") == "user"
                        and isinstance(message.get("content"), str)
                    ),
                    "",
                ),
            }
            try:
                response = original_create(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001 - 报告上下文错误类型
                call_record["error"] = f"{type(exc).__name__}: {exc}"
                call_record.update({
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                })
                llm_calls.append(call_record)
                raise
            usage = getattr(response, "usage", None)
            call_record.update({
                "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
                "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
                "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
            })
            llm_calls.append(call_record)
            return response

        agent.client.chat.completions.create = observed_create

        original_compress = agent._compress_history

        def observed_compress() -> None:
            before_summary = agent.summary or ""
            before_count = len(agent.raw_messages)
            original_compress()
            compression_events.append({
                "turn_index": current_turn["index"],
                "before_message_count": before_count,
                "after_message_count": len(agent.raw_messages),
                "before_summary": before_summary,
                "after_summary": agent.summary or "",
            })

        agent._compress_history = observed_compress

        original_build_messages = agent._build_messages

        def observed_build_messages() -> list[dict]:
            messages = original_build_messages()
            build_events.append({
                "turn_index": current_turn["index"],
                "message_count": len(messages),
                "summary_present": bool(agent.summary),
                "roles": [message.get("role") for message in messages],
            })
            return messages

        agent._build_messages = observed_build_messages

        turn_records: list[dict[str, Any]] = []
        for turn_index, user_input in enumerate(case["turns"], start=1):
            current_turn["index"] = turn_index
            turn_calls[turn_index] = []
            before_count = len(agent.raw_messages)
            before_summary = agent.summary or ""
            before_context_events = len(agent.context_manager.events)

            # 原始 Agent 会把 ReAct 中的“思考/动作/观察”打印到 stdout；
            # 基线需要保存结构化轨迹，避免终端输出淹没指标。
            with contextlib.redirect_stdout(io.StringIO()):
                response = agent.chat(user_input)
            turn_context_events = agent.context_manager.events[before_context_events:]
            for event in turn_context_events:
                event["turn_index"] = turn_index
            turn_records.append({
                "turn_index": turn_index,
                "user_input": user_input,
                "reply": response.reply,
                "intent": response.intent.value,
                "requires_human": response.requires_human,
                "tool_calls": turn_calls[turn_index],
                "before_message_count": before_count,
                "after_message_count": len(agent.raw_messages),
                "before_summary": before_summary,
                "after_summary": agent.summary or "",
                "context_events": turn_context_events,
            })

        compression_events = [
            item
            for event in agent.context_manager.events
            for item in event.get("compactions", [])
            if item.get("type") != "compact_history_skipped"
        ]
        return {
            "case_id": case["id"],
            "repeat_index": repeat_index,
            "turns": turn_records,
            "compression_events": compression_events,
            "build_events": build_events,
            "context_events": agent.context_manager.events,
            "llm_calls": llm_calls,
            "total_tokens": sum(int(item.get("total_tokens", 0) or 0) for item in llm_calls),
            "final_summary": agent.summary or "",
            "final_raw_message_count": len(agent.raw_messages),
            "archive_root": str(sandbox_root / "artifacts" / case["id"]),
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001 - 单条基线用例隔离
        return {
            "case_id": case["id"],
            "repeat_index": repeat_index,
            "turns": [],
            "compression_events": compression_events,
            "build_events": build_events,
            "context_events": agent.context_manager.events if agent else [],
            "llm_calls": llm_calls,
            "total_tokens": sum(int(item.get("total_tokens", 0) or 0) for item in llm_calls),
            "final_summary": "",
            "final_raw_message_count": 0,
            "archive_root": str(sandbox_root / "artifacts" / case["id"]),
            "error": f"{type(exc).__name__}: {exc}",
        }
    finally:
        if agent is not None:
            try:
                agent.tool_manager.close()
            except Exception:  # noqa: BLE001 - 清理失败不覆盖基线错误
                pass
        settings.memory_enabled = original_memory_enabled
        settings.mcp_enabled = original_mcp_enabled
        settings.context_archive_dir = original_context_archive_dir
        settings.context_char_limit = original_context_char_limit


def _flatten_compactions(run: dict[str, Any]) -> list[dict[str, Any]]:
    compactions: list[dict[str, Any]] = []
    for event in run.get("context_events", []):
        for item in event.get("compactions", []):
            if item.get("type") != "compact_history_skipped":
                compactions.append(item)
    return compactions


def _summary_is_structured(summary: str) -> bool:
    if not summary:
        return False
    try:
        data = json.loads(summary)
    except json.JSONDecodeError:
        return False
    return all(key in data for key in (
        "current_goal",
        "confirmed_facts",
        "completed_actions",
        "pending_questions",
        "next_action",
        "do_not_assume",
    ))


def _context_metrics(run: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    llm_calls = run.get("llm_calls", [])
    prepare_events = run.get("context_events", [])
    compactions = _flatten_compactions(run)
    summary_events = [
        item for item in compactions
        if item.get("type") in ("compact_history", "reactive_compact")
    ]
    archive_paths = [
        item.get("archive_path")
        for item in compactions
        if item.get("archive_path")
    ]
    archive_recoverable = all(
        Path(path).is_file() and Path(path).stat().st_size > 0
        for path in archive_paths
    ) if archive_paths else True

    budget = 18000
    budget_tolerance = 256
    pre_call_chars = [
        int(item.get("estimated_chars", 0))
        for item in llm_calls
        if item.get("purpose") == "react"
    ]
    context_budget_soft_overflows = sum(
        budget < value <= budget + budget_tolerance
        for value in pre_call_chars
    )
    context_budget_violations = sum(
        value > budget + budget_tolerance
        for value in pre_call_chars
    )
    tool_pair_results = [
        set(item.get("tool_call_ids", [])) == set(item.get("tool_result_ids", []))
        for item in llm_calls
        if item.get("purpose") == "react"
    ]
    tool_pair_integrity = (
        sum(tool_pair_results) / len(tool_pair_results)
        if tool_pair_results else 1.0
    )

    summary_validity_results = []
    for turn in run.get("turns", []):
        if turn.get("after_summary") and turn.get("after_summary") != turn.get("before_summary"):
            summary_validity_results.append(_summary_is_structured(turn["after_summary"]))
    summary_structure_validity = (
        sum(summary_validity_results) / len(summary_validity_results)
        if summary_validity_results else 1.0
    )

    active_request_results = []
    for turn in run.get("turns", []):
        active_request_results.append(any(
            call.get("purpose") == "react"
            and call.get("latest_user_content") == turn.get("user_input")
            for call in llm_calls
        ))
    active_request_preservation = (
        sum(active_request_results) / len(active_request_results)
        if active_request_results else 0.0
    )

    error_calls = [item for item in llm_calls if item.get("error")]
    prompt_tokens = sum(int(item.get("prompt_tokens", 0) or 0) for item in llm_calls)
    completion_tokens = sum(int(item.get("completion_tokens", 0) or 0) for item in llm_calls)
    react_calls = [item for item in llm_calls if item.get("purpose") == "react"]
    summary_calls = [item for item in llm_calls if item.get("purpose") == "context_summary"]
    tool_definition_tokens_est = sum(
        int(item.get("tool_definition_tokens_est", 0) or 0)
        for item in llm_calls
    )
    prompt_too_long_count = sum(
        any(marker in item.get("error", "").lower() for marker in (
            "prompt_too_long",
            "context_length_exceeded",
            "maximum context length",
            "too many tokens",
        ))
        for item in error_calls
    )
    reactive_retry_count = sum(
        int(event.get("force_reactive", False))
        for event in prepare_events
    )

    return {
        "pre_call_max_chars": max(pre_call_chars, default=0),
        "pre_call_max_tokens_est": max(
            [item.get("estimated_tokens", 0) for item in llm_calls if item.get("purpose") == "react"],
            default=0,
        ),
        "context_budget_chars": budget,
        "context_budget_tolerance_chars": budget_tolerance,
        "context_budget_soft_overflows": context_budget_soft_overflows,
        "context_budget_violations": context_budget_violations,
        "compaction_count": len(compactions),
        "compaction_types": sorted({item.get("type") for item in compactions}),
        "summary_compaction_count": len(summary_events),
        "summary_structure_validity": round(summary_structure_validity, 4),
        "tool_pair_integrity": round(tool_pair_integrity, 4),
        "active_request_preservation": round(active_request_preservation, 4),
        "archive_recoverability": archive_recoverable,
        "archive_count": len(archive_paths),
        "prompt_too_long_count": prompt_too_long_count,
        "reactive_retry_count": reactive_retry_count,
        "summary_api_calls": sum(item.get("purpose") == "context_summary" for item in llm_calls),
        "api_call_count": len(llm_calls),
        "error_call_count": len(error_calls),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "react_prompt_tokens": sum(int(item.get("prompt_tokens", 0) or 0) for item in react_calls),
        "summary_prompt_tokens": sum(int(item.get("prompt_tokens", 0) or 0) for item in summary_calls),
        "tool_definition_tokens_est": tool_definition_tokens_est,
        "avg_tokens_per_api_call": round(
            (prompt_tokens + completion_tokens) / len(llm_calls), 2
        ) if llm_calls else 0.0,
    }


def _evaluate_run(case: dict[str, Any], run: dict[str, Any]) -> dict[str, Any]:
    if run["error"]:
        return {
            "passed": False,
            "turn_assertion_pass_rate": 0.0,
            "tool_call_match_rate": 0.0,
            "fact_recall": 0.0,
            "stale_fact_avoidance": 0.0,
            "compaction_triggered": False,
            "compression_count": len(run.get("compression_events", [])),
            "final_keywords": 0.0,
            "context_quality_score": 0.0,
            "business_checks": 0.0,
            "total_tokens": run.get("total_tokens", 0),
            **_context_metrics(run, case),
            "error": run["error"],
        }

    turns = {item["turn_index"]: item for item in run["turns"]}
    assertions = case.get("turn_assertions", [])
    assertion_results: list[bool] = []
    tool_assertion_results: list[bool] = []

    for assertion in assertions:
        record = turns.get(assertion["turn_index"])
        if record is None:
            assertion_results.append(False)
            continue

        calls = record["tool_calls"]
        expected_tools = assertion.get("expected_tools", [])
        tools_ok = all(tool in [call["name"] for call in calls] for tool in expected_tools)

        expected_arguments = assertion.get("expected_arguments", {})
        arguments_ok = True
        if expected_arguments:
            arguments_ok = any(
                _matches_arguments(call, expected_arguments)
                for call in calls
            )

        forbidden_tools = assertion.get("forbidden_tools", [])
        forbidden_tools_ok = not any(call["name"] in forbidden_tools for call in calls)

        forbidden_arguments = assertion.get("forbidden_arguments", {})
        forbidden_arguments_ok = not any(
            _matches_arguments(call, forbidden_arguments)
            for call in calls
        ) if forbidden_arguments else True

        reply_ok = _contains_all(record["reply"], assertion.get("expected_keywords", []))
        tool_match = tools_ok and arguments_ok
        tool_assertion_results.append(tool_match)

        # Context 测试关注“模型是否保留并使用了正确上下文”。工具调用只作为诊断
        # 指标，除非单条断言显式设置 require_tools=true，否则不因工具调用时机
        # 不同而判定整轮失败。
        response_ok = forbidden_tools_ok and forbidden_arguments_ok and reply_ok
        if assertion.get("require_tools", False):
            assertion_results.append(tool_match and response_ok)
        else:
            assertion_results.append(response_ok)

    turn_assertion_pass_rate = (
        sum(assertion_results) / len(assertion_results) if assertion_results else 1.0
    )

    compaction = case.get("compaction", {})
    compaction_triggered = len(run["compression_events"]) >= int(
        compaction.get("min_events", 1) if compaction.get("required") else 0
    )
    fact_results: list[bool] = []
    stale_results: list[bool] = []
    for fact in case.get("context_facts", []):
        required_at = fact.get("required_at", [])
        fact_observed = False
        for turn_index in required_at:
            record = turns.get(turn_index)
            if not record:
                continue
            observable = record["reply"] + " " + " ".join(
                _call_text(call) + " " + call.get("result", "")
                for call in record["tool_calls"]
            )
            if _text_matches(observable, fact):
                fact_observed = True
                break
        if required_at:
            fact_results.append(fact_observed)

        cutoff = fact.get("must_not_use_after")
        if cutoff is not None:
            stale_used = False
            for turn_index, record in turns.items():
                if turn_index <= cutoff:
                    continue
                if any(_text_matches(_call_text(call), fact) for call in record["tool_calls"]):
                    stale_used = True
                    break
            stale_results.append(not stale_used)

    fact_recall = sum(fact_results) / len(fact_results) if fact_results else 1.0
    stale_fact_avoidance = (
        sum(stale_results) / len(stale_results) if stale_results else 1.0
    )

    final_record = turns.get(len(case["turns"]))
    final_reply = final_record["reply"] if final_record else ""
    final_keywords = case.get("expected_keywords", [])
    final_keyword_score = (
        sum(_expected_value_matches(final_reply, keyword) for keyword in final_keywords)
        / len(final_keywords)
        if final_keywords else 1.0
    )

    business_checks = case.get("expected_business_checks", [])
    business_results: list[bool] = []
    all_calls = [call for record in turns.values() for call in record["tool_calls"]]
    for check in business_checks:
        candidates = [call for call in all_calls if call["name"] == check.get("tool")]
        matched = any(
            _matches_arguments(call, check.get("arguments", {}))
            and _contains_all(call.get("result", ""), check.get("result_keywords", []))
            for call in candidates
        )
        business_results.append(matched)
    business_score = (
        sum(business_results) / len(business_results) if business_results else 1.0
    )

    context_metrics = _context_metrics(run, case)
    context_quality_score = round(
        0.45 * fact_recall
        + 0.35 * turn_assertion_pass_rate
        + 0.20 * final_keyword_score,
        4,
    )
    # Context 评测不把工具选择、摘要措辞和单个关键词的偶发差异当成硬失败。
    passed = all([
        context_quality_score >= 0.70,
        fact_recall >= 0.50,
        stale_fact_avoidance == 1.0,
        context_metrics["context_budget_violations"] == 0,
        context_metrics["tool_pair_integrity"] == 1.0,
        context_metrics["active_request_preservation"] == 1.0,
        context_metrics["archive_recoverability"],
        context_metrics["summary_structure_validity"] == 1.0,
        context_metrics["error_call_count"] == 0,
    ])

    return {
        "passed": passed,
        "turn_assertion_pass_rate": round(turn_assertion_pass_rate, 4),
        "tool_call_match_rate": round(
            sum(tool_assertion_results) / len(tool_assertion_results)
            if tool_assertion_results else 1.0,
            4,
        ),
        "fact_recall": round(fact_recall, 4),
        "stale_fact_avoidance": round(stale_fact_avoidance, 4),
        "compaction_triggered": compaction_triggered,
        "compression_count": len(run["compression_events"]),
        "final_keywords": round(final_keyword_score, 4),
        "context_quality_score": context_quality_score,
        "business_checks": round(business_score, 4),
        "total_tokens": run.get("total_tokens", 0),
        **context_metrics,
        "error": None,
    }


def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def _latest_baseline() -> tuple[Path | None, dict[str, Any]]:
    candidates = sorted(BASELINE_DIR.glob("context_baseline_*.json"))
    if not candidates:
        return None, {}
    path = candidates[-1]
    try:
        return path, json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return path, {}


def _comparison(current_summary: dict[str, Any]) -> dict[str, Any]:
    baseline_path, baseline = _latest_baseline()
    baseline_summary = baseline.get("summary", {})
    names = [
        "pass_rate",
        "avg_fact_recall",
        "avg_stale_fact_avoidance",
        "avg_turn_assertion_pass_rate",
        "avg_tool_call_match_rate",
        "avg_context_quality_score",
        "avg_total_prompt_tokens_per_case",
        "avg_total_completion_tokens_per_case",
        "avg_tokens_per_api_call",
        "avg_react_prompt_tokens_per_case",
        "avg_summary_prompt_tokens_per_case",
        "avg_tool_definition_tokens_est_per_case",
        "avg_pre_call_max_chars",
        "avg_context_budget_violations",
        "avg_summary_structure_validity",
        "avg_tool_pair_integrity",
        "avg_active_request_preservation",
        "avg_error_call_count",
    ]
    metrics = {}
    for name in names:
        current = current_summary.get(name)
        previous = baseline_summary.get(name)
        metrics[name] = {
            "before": previous,
            "after": current,
            "delta": round(current - previous, 4)
            if isinstance(current, (int, float)) and isinstance(previous, (int, float))
            else None,
        }
    return {
        "baseline_report": str(baseline_path) if baseline_path else None,
        "metrics": metrics,
        "interpretation": [
            "正向 delta 对事实召回、逐轮通过率、工具配对和当前请求保留通常是改善。",
            "负向 delta 对 token、峰值上下文和错误次数通常是改善。",
            "单次运行只能反映一次随机轨迹，不能替代 pass^k 稳定性结论。",
        ],
    }


def main() -> int:
    cases = _load_cases()
    repeats = 1
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sandbox_root = RESULT_DIR / "runtime" / timestamp
    sandbox_root.mkdir(parents=True, exist_ok=True)
    all_runs: list[dict[str, Any]] = []
    print("\n=== Context LCC run 1/1 ===")
    for case in cases:
        run = _run_case_once(case, 1, sandbox_root)
        evaluated = _evaluate_run(case, run)
        run["metrics"] = evaluated
        all_runs.append(run)
        status = "PASS" if evaluated["passed"] else "FAIL"
        print(
            f"{case['id']}: {status} | "
            f"fact={evaluated['fact_recall']:.2f} "
            f"turn={evaluated['turn_assertion_pass_rate']:.2f} "
            f"quality={evaluated['context_quality_score']:.2f} "
            f"tool_match={evaluated['tool_call_match_rate']:.2f} "
            f"compact={evaluated['compression_count']} "
            f"budget={evaluated['context_budget_violations']} "
            f"pairs={evaluated['tool_pair_integrity']:.2f} "
            f"tokens={evaluated['total_tokens']}"
            + (f" error={evaluated['error']}" if evaluated.get("error") else "")
        )

    by_case: dict[str, list[dict[str, Any]]] = {}
    for run in all_runs:
        by_case.setdefault(run["case_id"], []).append(run)

    case_summaries: list[dict[str, Any]] = []
    for case in cases:
        runs = by_case[case["id"]]
        metrics = [run["metrics"] for run in runs]
        case_summaries.append({
            "case_id": case["id"],
            "description": case["description"],
            "pass_rate": round(sum(item["passed"] for item in metrics) / len(metrics), 4),
            "pass_all_repeats": all(item["passed"] for item in metrics),
            "avg_fact_recall": _mean([item["fact_recall"] for item in metrics]),
            "avg_stale_fact_avoidance": _mean([item["stale_fact_avoidance"] for item in metrics]),
            "avg_turn_assertion_pass_rate": _mean([item["turn_assertion_pass_rate"] for item in metrics]),
            "avg_tool_call_match_rate": _mean([item["tool_call_match_rate"] for item in metrics]),
            "avg_context_quality_score": _mean([item["context_quality_score"] for item in metrics]),
            "avg_compression_count": _mean([float(item["compression_count"]) for item in metrics]),
            "avg_total_prompt_tokens": _mean([float(item["prompt_tokens"]) for item in metrics]),
            "avg_total_completion_tokens": _mean([float(item["completion_tokens"]) for item in metrics]),
            "avg_tokens_per_api_call": _mean([float(item["avg_tokens_per_api_call"]) for item in metrics]),
            "avg_react_prompt_tokens": _mean([float(item["react_prompt_tokens"]) for item in metrics]),
            "avg_summary_prompt_tokens": _mean([float(item["summary_prompt_tokens"]) for item in metrics]),
            "avg_tool_definition_tokens_est": _mean([float(item["tool_definition_tokens_est"]) for item in metrics]),
            "avg_pre_call_max_chars": _mean([float(item["pre_call_max_chars"]) for item in metrics]),
            "avg_context_budget_violations": _mean([float(item["context_budget_violations"]) for item in metrics]),
            "avg_summary_structure_validity": _mean([float(item["summary_structure_validity"]) for item in metrics]),
            "avg_tool_pair_integrity": _mean([float(item["tool_pair_integrity"]) for item in metrics]),
            "avg_active_request_preservation": _mean([float(item["active_request_preservation"]) for item in metrics]),
            "avg_error_call_count": _mean([float(item["error_call_count"]) for item in metrics]),
            "avg_api_call_count": _mean([float(item["api_call_count"]) for item in metrics]),
            "avg_summary_api_calls": _mean([float(item["summary_api_calls"]) for item in metrics]),
            "runs": metrics,
        })

    passed_cases = sum(item["pass_all_repeats"] for item in case_summaries)
    current_summary = {
        "case_count": len(cases),
        "repeat_count": repeats,
        "pass_all_repeats_cases": passed_cases,
        "pass_rate": round(passed_cases / len(cases), 4),
        "pass_k": round(passed_cases / len(cases), 4),
        "avg_fact_recall": _mean([item["avg_fact_recall"] for item in case_summaries]),
        "avg_stale_fact_avoidance": _mean([item["avg_stale_fact_avoidance"] for item in case_summaries]),
        "avg_turn_assertion_pass_rate": _mean([item["avg_turn_assertion_pass_rate"] for item in case_summaries]),
        "avg_tool_call_match_rate": _mean([item["avg_tool_call_match_rate"] for item in case_summaries]),
        "avg_context_quality_score": _mean([item["avg_context_quality_score"] for item in case_summaries]),
        "avg_total_prompt_tokens_per_case": _mean([item["avg_total_prompt_tokens"] for item in case_summaries]),
        "avg_total_completion_tokens_per_case": _mean([item["avg_total_completion_tokens"] for item in case_summaries]),
        "avg_tokens_per_api_call": _mean([item["avg_tokens_per_api_call"] for item in case_summaries]),
        "avg_react_prompt_tokens_per_case": _mean([item["avg_react_prompt_tokens"] for item in case_summaries]),
        "avg_summary_prompt_tokens_per_case": _mean([item["avg_summary_prompt_tokens"] for item in case_summaries]),
        "avg_tool_definition_tokens_est_per_case": _mean([item["avg_tool_definition_tokens_est"] for item in case_summaries]),
        "avg_pre_call_max_chars": _mean([item["avg_pre_call_max_chars"] for item in case_summaries]),
        "avg_context_budget_violations": _mean([item["avg_context_budget_violations"] for item in case_summaries]),
        "avg_summary_structure_validity": _mean([item["avg_summary_structure_validity"] for item in case_summaries]),
        "avg_tool_pair_integrity": _mean([item["avg_tool_pair_integrity"] for item in case_summaries]),
        "avg_active_request_preservation": _mean([item["avg_active_request_preservation"] for item in case_summaries]),
        "avg_error_call_count": _mean([item["avg_error_call_count"] for item in case_summaries]),
        "avg_api_call_count": _mean([item["avg_api_call_count"] for item in case_summaries]),
        "avg_summary_api_calls": _mean([item["avg_summary_api_calls"] for item in case_summaries]),
    }
    comparison = _comparison(current_summary)
    report = {
        "test": "ecom_context_lcc_single_run",
        "dataset": str(DATASET.relative_to(ROOT)),
        "evaluation_mode": "context_only",
        "memory_enabled": False,
        "repeats": repeats,
        "settings": {
            "context_char_limit": 18000,
            "context_target_ratio": 0.8,
            "context_max_messages": 50,
            "context_keep_recent_messages": 8,
            "model": os.environ.get("MODEL_NAME", "settings.model_name"),
        },
        "summary": current_summary,
        "comparison": comparison,
        "limitations": [
            "工具和订单来自本地 Ecom Mock，不代表 tau-bench 原始数据库状态。",
            "工具调用匹配率作为诊断指标，不要求在固定轮次调用工具；关键词和事实支持数据标注的同义变体。",
            "summary fidelity 使用数据标注的字符串证据，不等同于完整语义评估。",
            "本次只运行 1 次，不能估计 pass^k 稳定性。",
        ],
        "cases": case_summaries,
        "raw_runs": all_runs,
    }

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = RESULT_DIR / f"context_lcc_{timestamp}.json"
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== Context LCC summary ===")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"报告: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
