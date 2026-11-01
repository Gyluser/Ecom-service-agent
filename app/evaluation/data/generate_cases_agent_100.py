"""Generate the 100-case Ecom agent workflow evaluation dataset.

The first 25 cases are preserved verbatim.  The additional cases adapt the
tau2-bench Retail task families to this repository's local mock orders,
products, tools, and evaluation schema.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "app" / "evaluation" / "data" / "cases_agent_25.json"
TARGET = ROOT / "app" / "evaluation" / "data" / "cases_agent_100.json"


def case(
    number: int,
    description: str,
    turns: list[str],
    intent: str,
    keywords: list[str],
    tools: list[str],
    checks: list[dict] | None = None,
    *,
    human: bool = False,
    min_calls: int | None = None,
    source_task_id: str,
    source_actions: list[str],
    category: str = "agent_tool_workflow",
    notes: str = "将 tau2-bench Retail 任务改写为当前 Ecom Mock 数据和已注册工具。",
) -> dict:
    return {
        "id": f"agent_{number:02d}",
        "description": description,
        "turns": turns,
        "expected_intent": intent,
        "expected_keywords": keywords,
        "expected_requires_human": human,
        "expected_tools": tools,
        "min_tool_calls": len(tools) if min_calls is None else min_calls,
        "max_tokens": 6000,
        "category": category,
        "source_task_id": source_task_id,
        "mapping_notes": notes,
        "expected_business_checks": checks or [],
        "source_actions": source_actions,
    }


def order_check(tool: str, order_id: str, *keywords: str) -> dict:
    return {
        "tool": tool,
        "result_keywords": ["success", order_id, *keywords],
        "arguments": {"order_id": order_id},
    }


def list_check(*keywords: str) -> dict:
    return {"tool": "list_user_orders", "result_keywords": ["success", *keywords]}


def product_check(*keywords: str) -> dict:
    return {"tool": "query_product", "result_keywords": ["success", *keywords]}


def refund_check(order_id: str, *keywords: str) -> dict:
    return {
        "tool": "apply_refund",
        "result_keywords": ["success", "退款", *keywords],
        "arguments": {"order_id": order_id},
    }


def build_added_cases() -> list[dict]:
    added: list[dict] = []

    # Order-detail tasks: status, items, amount, and refund-state variants.
    order_rows = [
        ("ORD-20240115-001", "Nike", "899", "查询已发货订单金额", "请告诉我订单 ORD-20240115-001 的实付金额和商品", "order_query", "2", ["get_order_details"]),
        ("ORD-20240120-002", "AirPods", "待发货", "查询多商品待发货订单", "订单 ORD-20240120-002 里有哪些商品，当前发货状态是什么？", "order_query", "13", ["get_order_details"]),
        ("ORD-20240110-003", "小米", "已签收", "查询已签收订单详情", "请核对 ORD-20240110-003 是否已经签收，并告诉我买了什么", "order_query", "14", ["get_order_details", "return_delivered_order_items"]),
        ("ORD-20240118-004", "Levi", "审核中", "查询退款订单状态", "ORD-20240118-004 的售后退款现在是什么状态？", "return_request", "19", ["get_order_details"]),
        ("ORD-20240122-005", "戴森", "待发货", "查询待发货订单金额", "帮我看 ORD-20240122-005 的订单总额和发货状态", "order_query", "25", ["get_order_details"]),
    ]
    order_prompts = [
        "请核实 {oid} 的订单信息，我主要想知道商品和金额。",
        "帮我查一下 {oid}，看看这笔订单目前处于什么状态。",
        "订单 {oid} 的详情能发我一份吗？",
    ]
    refund_prompts = [
        "订单 {oid} 的退款进度到哪了？",
        "我想确认 {oid} 当前退款审核状态。",
    ]
    n = 26
    for row_index, (oid, kw1, kw2, desc, prompt, intent, source_id, actions) in enumerate(order_rows):
        for variant in range(3):
            if variant == 0:
                text = prompt
            elif intent == "return_request":
                text = refund_prompts[variant - 1].format(oid=oid)
            else:
                text = order_prompts[variant - 1].format(oid=oid)
            added.append(case(
                n, desc if variant == 0 else f"{desc}（改写{variant}）", [text], intent,
                [kw1, kw2], ["query_order"], [order_check("query_order", oid, kw1)],
                source_task_id=source_id, source_actions=actions,
            ))
            n += 1

    # Logistics tasks include shipped, delivered, and the not-yet-shipped boundary.
    logistics_rows = [
        ("ORD-20240115-001", "顺丰速运", "查询顺丰运输节点", "订单 ORD-20240115-001 最新物流到哪一步了？", "11"),
        ("ORD-20240110-003", "京东物流", "查询已签收物流", "ORD-20240110-003 的快递由哪家承运，是否已签收？", "14"),
        ("ORD-20240118-004", "圆通速递", "查询退款订单物流", "请查一下 ORD-20240118-004 的物流承运商和最新状态", "19"),
        ("ORD-20240120-002", "待发货", "待发货物流边界", "订单 ORD-20240120-002 还没有物流单号吗？", "13"),
        ("ORD-20240122-005", "待发货", "另一笔待发货物流边界", "戴森订单 ORD-20240122-005 现在能查到物流吗？", "25"),
    ]
    for oid, keyword, desc, prompt, source_id in logistics_rows:
        for suffix in ("", " 请给出当前配送进度。"):
            added.append(case(
                n, desc if not suffix else f"{desc}（追问）", [prompt + suffix], "order_query",
                [keyword], ["query_logistics"],
                [{"tool": "query_logistics", "result_keywords": ["尚未发货" if keyword == "待发货" else keyword], "arguments": {"order_id": oid}}],
                source_task_id=source_id, source_actions=["get_order_details", "return_delivered_order_items"],
            ))
            n += 1

    # Product consultation tasks cover every local product, including out-of-stock.
    product_rows = [
        ("Nike", ["Nike", "899"], "运动鞋价格库存咨询", "Nike Air Max 270 运动鞋现在多少钱，还有多少库存？", "31"),
        ("AirPods", ["AirPods", "1799"], "耳机参数咨询", "Apple AirPods Pro 2 的价格、降噪和续航信息是什么？", "32"),
        ("小米", ["小米", "5999"], "手机规格咨询", "请介绍小米14 Ultra 的价格、存储和屏幕规格。", "37"),
        ("Levi", ["Levi", "0"], "缺货牛仔裤咨询", "Levi's 501 经典牛仔裤目前是有货还是缺货？", "48"),
        ("戴森", ["戴森", "4299"], "吸尘器续航咨询", "戴森 V15 吸尘器价格和最长续航是多少？", "51"),
        ("保护壳", ["保护壳", "29.9"], "保护壳适配咨询", "AirPods 透明保护壳多少钱，适配哪一款耳机？", "53"),
    ]
    product_prompts = [
        "我想了解 {name} 的价格和库存，请查一下。",
        "帮我看看 {name} 的规格与售价。",
        "请查询 {name}，我想比较一下当前价格和商品信息。",
        "关于 {name}，请给我一份当前在售详情。",
    ]
    for name, keywords, desc, prompt, source_id in product_rows:
        for variant in range(4 if name in ("Nike", "AirPods", "小米") else 3):
            text = prompt if variant == 0 else product_prompts[variant - 1].format(name=name)
            added.append(case(
                n, desc if variant == 0 else f"{desc}（改写{variant}）", [text], "product_consult",
                keywords, ["query_product"], [product_check(name)],
                source_task_id=source_id, source_actions=["get_product_details"],
            ))
            n += 1

    # Refund/return tasks follow tau2's confirmation boundary but use the local
    # apply_refund behavior: pending orders are cancelled directly; other orders
    # create a refund request; refund_processing is query-only.
    refund_rows = [
        ("ORD-20240120-002", "订单尚未发货，想取消并退款", "我下单错了", "66"),
        ("ORD-20240122-005", "待发货订单退款", "暂时不需要吸尘器", "66"),
        ("ORD-20240110-003", "已签收订单退货退款", "商品不符合预期", "67"),
        ("ORD-20240115-001", "已发货订单售后申请", "现在不想要了", "67"),
        ("ORD-20240118-004", "退款处理中不重复申请", "", "68"),
    ]
    for oid, desc, reason, source_id in refund_rows:
        if oid == "ORD-20240118-004":
            turns = [f"订单 {oid} 已经申请退款，我只想确认现在的审核进度。"]
            added.append(case(
                n, desc, turns, "return_request", ["审核中"], ["query_order"],
                [order_check("query_order", oid, "审核中")], source_task_id=source_id,
                source_actions=["get_order_details", "return_delivered_order_items"],
            ))
            n += 1
            continue
        turns = [
            f"{desc}：订单 {oid}，原因是{reason}。",
            f"确认提交，原因就是{reason}。",
        ]
        added.append(case(
            n, desc, turns, "return_request", ["退款"], ["apply_refund"],
            [refund_check(oid)], source_task_id=source_id,
            source_actions=["get_order_details", "return_delivered_order_items"],
        ))
        n += 1
        # A second wording tests that confirmation remains a mutating-step gate.
        added.append(case(
            n, f"{desc}（确认改写）", [
                f"我想处理订单 {oid}，请先帮我申请退款，原因是{reason}。",
                "好的，确认申请退款。",
            ], "return_request", ["退款"], ["apply_refund"],
            [refund_check(oid)], source_task_id=source_id,
            source_actions=["get_order_details", "return_delivered_order_items"],
        ))
        n += 1

    # List-and-follow-up cases retain tau2's multi-step user workflow shape.
    added.extend([
        case(81, "列出订单后确认退款对象", [
            "我不知道具体订单号，先列出我的订单。",
            "就是 ORD-20240118-004，告诉我现在的退款状态。",
        ], "return_request", ["审核中"], ["list_user_orders", "query_order"],
        [list_check("ORD-20240118-004"), order_check("query_order", "ORD-20240118-004", "审核中")],
        source_task_id="73", source_actions=["find_user_id_by_name_zip", "get_order_details"]),
        case(82, "列出订单后确认物流对象", [
            "我不记得订单号，先把我的订单列出来。",
            "查 ORD-20240115-001 的配送进度。",
        ], "order_query", ["顺丰速运"], ["list_user_orders", "query_logistics"],
        [list_check("ORD-20240115-001"), {"tool": "query_logistics", "result_keywords": ["顺丰速运"], "arguments": {"order_id": "ORD-20240115-001"}}],
        source_task_id="55", source_actions=["find_user_id_by_email", "get_order_details"]),
        case(83, "列出订单后确认已签收商品", [
            "我想先看看自己的订单。",
            "请继续查询 ORD-20240110-003 的商品和状态。",
        ], "order_query", ["小米", "已签收"], ["list_user_orders", "query_order"],
        [list_check("ORD-20240110-003"), order_check("query_order", "ORD-20240110-003", "小米")],
        source_task_id="62", source_actions=["find_user_id_by_name_zip", "get_order_details"]),
        case(84, "查询订单列表后确认待发货商品", [
            "请列出我目前所有订单。",
            "其中 ORD-20240122-005 里买了什么？",
        ], "order_query", ["戴森", "待发货"], ["list_user_orders", "query_order"],
        [list_check("ORD-20240122-005"), order_check("query_order", "ORD-20240122-005", "戴森")],
        source_task_id="30", source_actions=["find_user_id_by_email", "get_order_details"]),
        case(85, "订单列表后确认耳机订单", [
            "我有几笔订单记不清了，先列出订单。",
            "请再查 ORD-20240120-002 的订单总额。",
        ], "order_query", ["AirPods", "1828"], ["list_user_orders", "query_order"],
        [list_check("ORD-20240120-002"), order_check("query_order", "ORD-20240120-002", "AirPods")],
        source_task_id="30", source_actions=["find_user_id_by_name_zip", "get_order_details"]),
    ])

    # Knowledge/policy tasks are supported by the local RAG tool and are kept
    # intentionally free of a DB assertion because retrieval ranking is not the
    # same as the tau2 structured database state.
    policy_rows = [
        ("七天无理由退货", "七天", "86", "退换货政策"),
        ("已发货商品能否仅退款", "退款", "87", "退款政策"),
        ("退货退款审核多久", "审核", "88", "退款流程"),
        ("退货需要满足什么条件", "二次销售", "89", "退货条件"),
        ("商品质量问题怎么处理", "质量", "90", "质量问题售后"),
        ("配送时效怎么计算", "配送", "91", "配送说明"),
        ("会员权益如何查询", "会员", "92", "会员FAQ"),
    ]
    for title, keyword, source_id, desc in policy_rows:
        policy_intent = (
            "return_request" if ("退" in title or "退款" in title)
            else "after_sale" if "质量" in title
            else "other"
        )
        added.append(case(
            n, desc, [f"请说明一下{title}的规则和办理方式。"], policy_intent,
            [keyword], ["search_knowledge"], [], source_task_id=source_id,
            source_actions=["search_policy", "communicate_info"], category="agent_policy_workflow",
        ))
        n += 1

    # Explicit unsupported/complaint boundaries: tau2 also evaluates what the
    # agent should not mutate. These cases mainly check final intent and handoff.
    for desc, text in [
        ("投诉商品质量并转人工", "商品质量太差了，我要投诉，请转人工客服。"),
        ("投诉退款处理并转人工", "退款一直没有结果，我要投诉并转人工。"),
        ("投诉物流异常并转人工", "物流长时间不更新，我要投诉，请安排人工处理。"),
        ("投诉服务态度并转人工", "客服服务态度很差，我要投诉并转人工。"),
    ]:
        added.append(case(
            n, desc, [text], "complaint", ["人工"], [], [], human=True, min_calls=0,
            source_task_id="10", source_actions=["communicate_info"], category="agent_handoff",
        ))
        n += 1

    for desc, text in [
        ("恢复已取消订单转人工", "请把已经取消的订单恢复配送，无法直接处理就转人工。"),
        ("撤销取消订单转人工", "我想撤销已取消订单，麻烦帮我转人工客服。"),
        ("恢复订单状态转人工", "订单取消后请恢复订单状态，转人工确认。"),
        ("取消后重新配送转人工", "我想恢复已取消的订单并重新配送，请转人工处理。"),
    ]:
        added.append(case(
            n, desc, [text], "after_sale", ["人工"], ["list_user_orders"],
            [list_check("ORD-")], human=True, min_calls=1,
            source_task_id="50", source_actions=["get_order_details", "communicate_info"], category="agent_handoff",
        ))
        n += 1

    # Keep the generated extension contiguous even though the task-family
    # blocks above are easier to read with their own local numbering.
    for number, item in enumerate(added, start=26):
        item["id"] = f"agent_{number:02d}"

    if len(added) != 75:
        raise AssertionError(f"expected 75 added cases, got {len(added)}")
    return added


def main() -> None:
    source_data = json.loads(SOURCE.read_text(encoding="utf-8"))
    original = copy.deepcopy(source_data["cases"])
    if len(original) != 25:
        raise AssertionError(f"source dataset changed: expected 25 cases, got {len(original)}")

    cases = original + build_added_cases()
    if len(cases) != 100:
        raise AssertionError(f"expected 100 cases, got {len(cases)}")
    if len({item["id"] for item in cases}) != 100:
        raise AssertionError("case ids are not unique")

    payload = {
        "dataset": "Ecom Agent Tool / Workflow adapted from tau2-bench Retail (100 cases)",
        "source": source_data["source"],
        "reference_repository": "https://github.com/sierra-research/tau2-bench",
        "mapping_policy": (
            "保留原 cases_agent_25.json 的 25 条；新增任务仅映射到当前 Ecom Mock 数据、"
            "已注册工具和本地知识库，不复用 tau2-bench 原始数据库状态。"
        ),
        "cases": cases,
    }
    TARGET.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(cases)} cases to {TARGET}")


if __name__ == "__main__":
    main()
