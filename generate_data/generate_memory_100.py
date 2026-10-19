"""Generate a 100-case Memory evaluation set from tau2-bench Retail tasks.

The source benchmark evaluates tool-agent-user interaction.  This generator
keeps the Retail entities and scenario difficulty, but turns them into offline
memory retrieval cases.  It does not copy tool actions as memory ground truth.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = REPO_ROOT.parent / "tau_retail_tasks.tmp.json"
OUTPUT_PATH = REPO_ROOT / "app" / "evaluation" / "data" / "cases_memory_100.json"

ORDER_RE = re.compile(r"#[A-Z]\d{7,}")
PRODUCT_RE = re.compile(r"\b\d{10}\b")
ZIP_RE = re.compile(r"\b\d{5}\b")

ACTION_ZH = {
    "find_user_id_by_name_zip": "根据姓名和邮编确认用户身份",
    "find_user_id_by_email": "根据邮箱确认用户身份",
    "get_order_details": "查询订单详情",
    "get_product_details": "查询商品详情",
    "get_user_details": "查询用户详情",
    "exchange_delivered_order_items": "申请已送达商品换货",
    "return_delivered_order_items": "申请已送达商品退货",
    "modify_pending_order_items": "修改待处理订单商品",
    "modify_pending_order_address": "修改待处理订单地址",
    "modify_pending_order_payment": "修改待处理订单支付方式",
    "cancel_pending_order": "取消待处理订单",
    "modify_user_address": "修改用户地址",
    "transfer_to_human_agents": "转人工处理",
    "calculate": "计算订单相关金额",
}


def _task_text(task: dict[str, Any]) -> str:
    instructions = task["user_scenario"]["instructions"]
    return instructions.get("reason_for_call") or "用户咨询零售订单服务。"


def _actions(task: dict[str, Any]) -> list[str]:
    return [item["name"] for item in task.get("evaluation_criteria", {}).get("actions", [])]


def _entities(task: dict[str, Any]) -> tuple[list[str], list[str], list[str]]:
    text = json.dumps(task, ensure_ascii=False)
    orders = list(dict.fromkeys(ORDER_RE.findall(text)))
    products = list(dict.fromkeys(PRODUCT_RE.findall(text)))
    zips = list(dict.fromkeys(ZIP_RE.findall(text)))
    return orders, products, zips


def _task_facts(task: dict[str, Any], source_index: int) -> list[dict[str, str]]:
    instructions = task["user_scenario"]["instructions"]
    reason = _task_text(task)
    orders, products, zips = _entities(task)
    actions = _actions(task)
    facts: list[dict[str, str]] = [
        {
            "content": f"Retail 任务 {source_index:03d} 中，用户的业务诉求是：{reason}",
            "category": "issue",
        },
        {
            "content": f"Retail 任务 {source_index:03d} 的身份信息（姓名、邮箱或邮编）是：{instructions.get('known_info') or '用户未提供额外身份信息'}。",
            "category": "identity",
        },
        {
            "content": f"Retail 任务 {source_index:03d} 中，用户暂时未知的信息是：{instructions.get('unknown_info') or '没有额外未知信息'}。",
            "category": "issue",
        },
    ]
    if orders:
        facts.append({
            "content": f"用户曾提到订单 {orders[0]}，该订单与当前零售售后诉求有关。",
            "category": "order",
        })
    if products:
        facts.append({
            "content": f"用户曾关注或处理商品 ID {products[0]}。",
            "category": "behavior",
        })
    if len(products) > 1:
        facts.append({
            "content": f"用户还关注过商品 ID {products[1]}，需要和其他商品区分。",
            "category": "behavior",
        })
    if zips:
        facts.append({
            "content": f"用户曾提供邮编 {zips[0]} 用于身份确认。",
            "category": "identity",
        })
    if actions:
        action_text = "、".join(ACTION_ZH.get(action, action) for action in dict.fromkeys(actions[:3]))
        facts.append({
            "content": f"该零售任务涉及的处理事项包括：{action_text}。",
            "category": "behavior",
        })
    facts.append({
        "content": f"用户在 Retail 场景中强调：{instructions.get('task_instructions') or '希望问题得到完整处理'}。",
        "category": "preference",
    })
    return facts


def _distractors(
    all_tasks: list[dict[str, Any]],
    current_index: int,
    count: int,
    excluded_contents: set[str] | None = None,
) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    excluded = set(excluded_contents or set())
    offset = 1
    attempts = 0
    while len(result) < count and attempts < len(all_tasks) * 12:
        task = all_tasks[(current_index + offset) % len(all_tasks)]
        facts = _task_facts(task, (current_index + offset) % len(all_tasks))
        candidate = facts[(offset + len(result)) % len(facts)]
        if candidate["content"] not in excluded:
            result.append(candidate)
            excluded.add(candidate["content"])
        offset += 1
        attempts += 1
    return result[:count]


def _record(case_number: int, number: int, fact: dict[str, str], date: str) -> dict[str, str]:
    return {
        "memory_id": f"m100_{case_number:03d}_{number:02d}",
        "content": fact["content"],
        "category": fact.get("category", "other"),
        "created_at": date,
        "source_session": f"tau2_retail_{case_number:03d}",
    }


def _fact_position(facts: list[dict[str, str]], marker: str, fallback: int = 0) -> int:
    """Locate the fact that actually contains the entity asked about."""
    for index, fact in enumerate(facts):
        if marker in fact["content"]:
            return index
    return fallback


def _style_variant(case_number: int, formal: str, colloquial: str, short: str) -> str:
    variants = (formal, colloquial, short, f"帮我回忆一下：{formal}")
    return variants[(case_number - 1) % len(variants)]


def _base_case(
    case_number: int,
    source_index: int,
    task: dict[str, Any],
    relevant_positions: list[int],
    query: str,
    all_tasks: list[dict[str, Any]],
) -> dict[str, Any]:
    facts = _task_facts(task, source_index)
    selected = [facts[position % len(facts)] for position in relevant_positions]
    distractors = _distractors(
        all_tasks,
        source_index,
        10 - len(selected),
        excluded_contents={fact["content"] for fact in selected},
    )
    records = selected + distractors
    base_date = datetime(2026, 1, 1) + timedelta(days=case_number)
    records = [
        _record(case_number, index, fact, (base_date + timedelta(days=index)).isoformat(timespec="seconds"))
        for index, fact in enumerate(records, start=1)
    ]
    return {
        "case_id": f"memory_100_{case_number:03d}",
        "category": "",
        "user_id": f"tau_memory_user_{case_number:03d}",
        "query": query,
        "relevant_memory_ids": [records[position - 1]["memory_id"] for position in range(1, len(selected) + 1)],
        "answer_required_keywords": (
            [selected[0]["content"].split("：", 1)[-1][:18]]
            if selected else []
        ),
        "answer_forbidden_keywords": [],
        "answer_accuracy_evaluable": False,
        "memory_records": records,
        "source_bench": {
            "dataset": "tau2-bench",
            "domain": "retail",
            "task_id": str(task.get("id", source_index)),
            "task_index": source_index,
            "difficulty_basis": "preserved multi-entity, conditional, and fallback wording from Retail task",
        },
    }


def _make_single(case_number: int, source_index: int, task: dict[str, Any], tasks: list[dict[str, Any]]) -> dict[str, Any]:
    facts = _task_facts(task, source_index)
    orders, products, _ = _entities(task)
    if orders:
        query = _style_variant(
            case_number,
            f"我之前那笔订单 {orders[0]} 相关的记忆是什么？",
            f"之前那个单 {orders[0]} 是怎么回事来着？",
            f"{orders[0]} 这单当时啥情况？",
        )
        relevant_position = _fact_position(facts, f"订单 {orders[0]}")
    elif products:
        query = _style_variant(
            case_number,
            f"我以前关注过的商品 {products[0]} 有什么记录？",
            f"之前那个商品 {products[0]} 我查过啥来着？",
            f"商品 {products[0]} 的那条记忆呢？",
        )
        relevant_position = _fact_position(facts, f"商品 ID {products[0]}")
    else:
        query = _style_variant(
            case_number,
            "我之前给客服留过哪些身份信息？",
            "我当时给过客服什么身份资料来着？",
            "之前留的身份信息是什么？",
        )
        relevant_position = _fact_position(facts, "身份信息")
    case = _base_case(case_number, source_index, task, [relevant_position], query, tasks)
    case["category"] = "single_fact_recall"
    return case


def _make_multi(case_number: int, source_index: int, task: dict[str, Any], tasks: list[dict[str, Any]]) -> dict[str, Any]:
    facts = _task_facts(task, source_index)
    orders, products, _ = _entities(task)
    if orders and products:
        query = _style_variant(
            case_number,
            f"关于订单 {orders[0]} 和商品 {products[0]}，历史记录里分别是什么？",
            f"订单 {orders[0]} 那次，商品 {products[0]} 到底怎么处理的？",
            f"把 {orders[0]} 和商品 {products[0]} 的记录一起找出来。",
        )
        relevant_positions = [
            _fact_position(facts, f"订单 {orders[0]}"),
            _fact_position(facts, f"商品 ID {products[0]}"),
        ]
    elif products:
        query = _style_variant(
            case_number,
            f"商品 {products[0]} 和当时的业务诉求分别是什么？",
            f"之前查商品 {products[0]} 的时候，我主要想解决啥？",
            f"找一下商品 {products[0]} 以及对应的诉求。",
        )
        relevant_positions = [_fact_position(facts, f"商品 ID {products[0]}"), 0]
    else:
        query = _style_variant(
            case_number,
            "把我之前的零售诉求和身份信息一起找出来。",
            "我之前找客服是为什么，身份资料又是什么？",
            "把诉求和身份信息一起回忆出来。",
        )
        relevant_positions = [0, _fact_position(facts, "身份信息")]
    case = _base_case(case_number, source_index, task, relevant_positions, query, tasks)
    case["category"] = "multi_fact_recall"
    return case


def _make_no_relevant(case_number: int, source_index: int, task: dict[str, Any], tasks: list[dict[str, Any]]) -> dict[str, Any]:
    query = _style_variant(
        case_number,
        "我以前咨询过咖啡机的库存和退货吗？",
        "我之前有没有问过咖啡机，记得吗？",
        "咖啡设备的库存和退货，我以前提过没有？",
    )
    case = _base_case(
        case_number,
        source_index,
        task,
        [],
        query,
        tasks,
    )
    case["category"] = "no_relevant_memory"
    case["relevant_memory_ids"] = []
    case["answer_required_keywords"] = []
    case["no_relevant_expected"] = True
    return case


def _make_update(case_number: int, source_index: int, task: dict[str, Any], tasks: list[dict[str, Any]]) -> dict[str, Any]:
    orders, _, _ = _entities(task)
    key = orders[0] if orders else f"任务 {source_index:03d}"
    stale = {"content": f"订单 {key} 之前状态为待处理。", "category": "issue"}
    latest = {"content": f"订单 {key} 当前状态为已完成。", "category": "issue"}
    case = _base_case(case_number, source_index, task, [0], f"{key} 现在的状态是什么？", tasks)
    base_date = datetime(2026, 4, 1) + timedelta(days=case_number)
    stale_record = _record(case_number, 1, stale, (base_date - timedelta(days=2)).isoformat(timespec="seconds"))
    latest_record = _record(case_number, 2, latest, base_date.isoformat(timespec="seconds"))
    case["memory_records"] = [stale_record, latest_record] + case["memory_records"][2:]
    case["relevant_memory_ids"] = [latest_record["memory_id"]]
    case["category"] = "memory_update_conflict"
    case["answer_required_keywords"] = ["已完成"]
    case["answer_forbidden_keywords"] = ["待处理"]
    case["answer_accuracy_evaluable"] = True
    case["update_expectation"] = {
        "latest_memory_id": latest_record["memory_id"],
        "stale_memory_ids": [stale_record["memory_id"]],
    }
    return case


def _make_isolation(case_number: int, source_index: int, task: dict[str, Any], tasks: list[dict[str, Any]]) -> dict[str, Any]:
    case = _make_single(case_number, source_index, task, tasks)
    case["category"] = "user_isolation"
    other_user_id = f"tau_memory_other_{case_number:03d}"
    case["isolation_probe"] = {
        "other_user_id": other_user_id,
        "other_user_records": [{
            "memory_id": f"other_{case_number:03d}_01",
            "content": "另一位用户偏好完全不同的紫色商品和自提配送。",
            "category": "preference",
            "created_at": "2026-06-01T09:00:00",
            "source_session": f"other_session_{case_number:03d}",
        }],
        "forbidden_keywords": ["紫色", "自提"],
    }
    return case


def _make_persistence(case_number: int, source_index: int, task: dict[str, Any], tasks: list[dict[str, Any]], process_restart: bool) -> dict[str, Any]:
    case = _make_single(case_number, source_index, task, tasks)
    case["category"] = "process_restart_persistence" if process_restart else "cross_session_persistence"
    case["process_restart_expected"] = process_restart
    return case


def build_dataset() -> dict[str, Any]:
    source = json.loads(SOURCE_PATH.read_text(encoding="utf-8"))
    if len(source) < 100:
        raise ValueError("tau2 Retail source must contain at least 100 tasks")
    tasks = source[:100]
    cases: list[dict[str, Any]] = []
    case_number = 1

    for index in range(30):
        cases.append(_make_single(case_number, index, tasks[index], tasks))
        case_number += 1
    for index in range(30, 50):
        cases.append(_make_multi(case_number, index, tasks[index], tasks))
        case_number += 1
    for index in range(50, 60):
        cases.append(_make_no_relevant(case_number, index, tasks[index], tasks))
        case_number += 1
    for index in range(60, 75):
        cases.append(_make_update(case_number, index, tasks[index], tasks))
        case_number += 1
    for index in range(75, 90):
        cases.append(_make_isolation(case_number, index, tasks[index], tasks))
        case_number += 1
    for index in range(90, 95):
        cases.append(_make_persistence(case_number, index, tasks[index], tasks, False))
        case_number += 1
    for index in range(95, 100):
        cases.append(_make_persistence(case_number, index, tasks[index], tasks, True))
        case_number += 1

    return {
        "schema_version": "memory_cases_v2",
        "evaluation_mode": "memory_only",
        "source": {
            "dataset": "tau2-bench",
            "domain": "retail",
            "source_file": "../tau_retail_tasks.tmp.json",
            "task_count_used": 100,
            "generation_note": "Retail scenarios adapted into offline memory retrieval cases; tool actions are not treated as memory labels.",
        },
        "case_count": len(cases),
        "category_counts": {
            category: sum(case["category"] == category for case in cases)
            for category in sorted({case["category"] for case in cases})
        },
        "cases": cases,
    }


def main() -> Path:
    dataset = build_dataset()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(dataset, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT_PATH), "case_count": dataset["case_count"], "category_counts": dataset["category_counts"]}, ensure_ascii=False, indent=2))
    return OUTPUT_PATH


if __name__ == "__main__":
    main()
