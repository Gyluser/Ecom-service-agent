"""将 τ-bench Retail 任务适配为当前 Ecom 的 Agent Tool / Workflow 测试集。

本脚本只生成评测数据，不修改 app/agent 主体。τ-bench 的订单、商品和工具
与 Ecom 不同，因此这里保留 source_task_id/source_actions 作为来源信息，
再把业务对象映射到 Ecom 的 Mock 订单和现有工具。

运行：
    python generate_data/generate_agent_cases.py --source <tau_tasks.json>
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / "app" / "evaluation" / "data" / "cases_agent_25.json"
DEFAULT_SOURCE_URL = (
    "https://huggingface.co/datasets/HuggingFaceH4/tau2-bench-data/"
    "resolve/main/domains/retail/tasks.json"
)


def _check(tool: str, keywords: list[str], arguments: dict | None = None) -> dict:
    item = {"tool": tool, "result_keywords": keywords}
    if arguments:
        item["arguments"] = arguments
    return item


def _case(
    case_id: str,
    source_task_id: str,
    description: str,
    turns: list[str],
    expected_intent: str,
    expected_keywords: list[str],
    expected_tools: list[str],
    mapping_notes: str,
    business_checks: list[dict] | None = None,
    requires_human: bool = False,
    min_tool_calls: int | None = None,
) -> dict:
    return {
        "id": case_id,
        "description": description,
        "turns": turns,
        "expected_intent": expected_intent,
        "expected_keywords": expected_keywords,
        "expected_requires_human": requires_human,
        "expected_tools": expected_tools,
        "min_tool_calls": min_tool_calls if min_tool_calls is not None else len(expected_tools),
        "max_tokens": 9000 if len(turns) > 1 else 6000,
        "category": "agent_tool_workflow",
        "source_task_id": source_task_id,
        "mapping_notes": mapping_notes,
        "expected_business_checks": business_checks or [],
    }


def build_cases() -> list[dict]:
    """返回 25 条适配当前 Ecom 工具边界的用例。"""
    return [
        _case("agent_01", "2", "查询已发货订单详情", ["请查询订单 ORD-20240115-001 的详细信息"], "order_query", ["Nike", "899"], ["query_order"], "将 Bench 中的订单详情查询映射为 query_order。", [_check("query_order", ["success", "ORD-20240115-001"], {"order_id": "ORD-20240115-001"})], min_tool_calls=1),
        _case("agent_02", "11", "查询已发货订单物流", ["订单 ORD-20240115-001 的物流到哪里了？"], "order_query", ["顺丰"], ["query_logistics"], "将 Bench 的订单跟踪需求映射为 query_logistics。", [_check("query_logistics", ["success", "顺丰速运"], {"order_id": "ORD-20240115-001"})], min_tool_calls=1),
        _case("agent_03", "13", "查询未发货订单物流边界", ["订单 ORD-20240120-002 现在到哪里了？"], "order_query", ["尚未发货"], ["query_logistics"], "保留 Bench 的物流查询意图，映射到 Ecom 的待发货订单边界。", [_check("query_logistics", ["尚未发货"], {"order_id": "ORD-20240120-002"})], min_tool_calls=1),
        _case("agent_04", "14", "查询已签收订单", ["帮我看一下订单 ORD-20240110-003 的状态和商品"], "order_query", ["小米", "5999"], ["query_order"], "将已完成订单查询映射为 query_order。", [_check("query_order", ["success", "ORD-20240110-003"], {"order_id": "ORD-20240110-003"})], min_tool_calls=1),
        _case("agent_05", "19", "查询退款处理中订单", ["订单 ORD-20240118-004 的退款进度怎么样了？"], "return_request", ["审核中"], ["query_order"], "Bench 的售后查询映射为查询退款处理中订单，不重复申请退款。", [_check("query_order", ["success", "ORD-20240118-004"], {"order_id": "ORD-20240118-004"})], min_tool_calls=1),
        _case("agent_06", "25", "查询待发货订单", ["请查询订单 ORD-20240122-005 目前是什么状态"], "order_query", ["戴森", "待发货"], ["query_order"], "将 Bench 的订单状态确认映射为 query_order。", [_check("query_order", ["success", "ORD-20240122-005"], {"order_id": "ORD-20240122-005"})], min_tool_calls=1),
        _case("agent_07", "30", "列出用户订单", ["我想看看我一共有哪些订单"], "order_query", ["ORD-20240115-001"], ["list_user_orders"], "Bench 中先定位用户再查订单的流程，适配为 Ecom 的 list_user_orders。", [_check("list_user_orders", ["success", "ORD-20240115-001"])], min_tool_calls=1),
        _case("agent_08", "31", "查询运动鞋商品", ["帮我查一下 Nike 运动鞋的价格和库存"], "product_consult", ["Nike", "899"], ["query_product"], "将 Bench 的商品详情查询映射为 query_product。", [_check("query_product", ["success", "Nike"])], min_tool_calls=1),
        _case("agent_09", "32", "查询耳机商品", ["Apple AirPods Pro 2 有什么参数，多少钱？"], "product_consult", ["AirPods", "1799"], ["query_product"], "将 Bench 的商品属性查询映射为 query_product。", [_check("query_product", ["success", "AirPods"])], min_tool_calls=1),
        _case("agent_10", "48", "查询缺货牛仔裤", ["Levi's 501 现在还有库存吗？"], "product_consult", ["Levi", "0"], ["query_product"], "将 Bench 的商品可用性问题映射到 Ecom 缺货商品。", [_check("query_product", ["success", "Levi"])], min_tool_calls=1),
        _case("agent_11", "51", "查询吸尘器商品", ["戴森 V15 吸尘器的价格、库存和续航是多少？"], "product_consult", ["戴森", "4299"], ["query_product"], "将 Bench 的商品筛选需求映射为现有商品查询。", [_check("query_product", ["success", "戴森"])], min_tool_calls=1),
        _case("agent_12", "53", "查询耳机保护壳", ["请查一下 AirPods 保护壳的价格和适配信息"], "product_consult", ["保护壳", "29.9"], ["query_product"], "将 Bench 的配件查询适配到 Ecom 的商品 Mock 数据。", [_check("query_product", ["success", "保护壳"])], min_tool_calls=1),
        _case("agent_13", "54", "多轮补充订单号查询", ["我想查一下我的订单", "订单号是 ORD-20240110-003，请告诉我买了什么"], "order_query", ["小米"], ["query_order"], "将 Bench 的多轮订单确认改写为先表达需求、后补充 Ecom 订单号。", [_check("query_order", ["success", "ORD-20240110-003"], {"order_id": "ORD-20240110-003"})], min_tool_calls=1),
        _case("agent_14", "55", "多轮补充物流查询", ["我想知道订单的配送进度", "订单号是 ORD-20240115-001"], "order_query", ["顺丰"], ["query_logistics"], "将 Bench 的物流任务适配为 Ecom 订单号补充场景。", [_check("query_logistics", ["success", "顺丰速运"], {"order_id": "ORD-20240115-001"})], min_tool_calls=1),
        _case("agent_15", "62", "订单号不存在时列出订单", ["我不确定订单号，只记得想查最近买过的东西"], "order_query", ["订单"], ["list_user_orders"], "Bench 中订单号不确定的场景映射为当前 Ecom 的订单列表兜底。", [_check("list_user_orders", ["success", "ORD-"])], min_tool_calls=1),
        _case("agent_16", "65", "已签收商品退款确认", ["订单 ORD-20240110-003 的手机我不想要了，想申请退款", "确认申请退款，原因是不想要了"], "return_request", ["退款"], ["apply_refund"], "将 Bench 的 return/exchange 分支统一适配为 Ecom 已有 apply_refund，并保留二次确认。", [_check("apply_refund", ["success", "退款"], {"order_id": "ORD-20240110-003"})], min_tool_calls=1),
        _case("agent_17", "66", "待发货订单取消并退款", ["订单 ORD-20240120-002 还没发货，我想取消并退款", "确认取消订单，原因是下单错误"], "return_request", ["取消", "退款"], ["apply_refund"], "将 Bench 的 cancel_pending_order 映射为 Ecom 对待发货订单调用 apply_refund。", [_check("apply_refund", ["success", "直接取消"], {"order_id": "ORD-20240120-002"})], min_tool_calls=1),
        _case("agent_18", "67", "已签收订单售后申请", ["我收到的订单 ORD-20240110-003 有问题，想退货退款", "确认提交，原因是商品不符合预期"], "return_request", ["退款申请"], ["apply_refund"], "将 Bench 的退货流程映射为 Ecom 已有退款申请流程。", [_check("apply_refund", ["success", "退款申请已提交"], {"order_id": "ORD-20240110-003"})], min_tool_calls=1),
        _case("agent_19", "68", "退款处理中不重复操作", ["订单 ORD-20240118-004 已经申请退款了，我只想确认当前进度"], "return_request", ["审核中"], ["query_order"], "针对 Ecom 已有退款处理中状态，禁止重复调用退款工具。", [_check("query_order", ["success", "ORD-20240118-004"], {"order_id": "ORD-20240118-004"})], min_tool_calls=1),
        _case("agent_20", "69", "商品咨询后查询订单", ["我之前买过 AirPods，先帮我确认一下这个商品", "订单号是 ORD-20240120-002，顺便看看订单状态"], "order_query", ["AirPods", "待发货"], ["query_order"], "商品查询是可选辅助步骤，不再强制要求 query_product；只校验最终订单查询和订单结果，允许 Agent 采用列订单后查详情等等价路径。", [_check("query_order", ["success", "ORD-20240120-002"], {"order_id": "ORD-20240120-002"})], min_tool_calls=1),
        _case("agent_21", "73", "查询全部订单后确认退款对象", ["我不知道具体订单号，先列出我的订单", "我要处理其中的订单 ORD-20240118-004，先告诉我退款状态"], "return_request", ["审核中"], ["list_user_orders", "query_order"], "将 Bench 的先定位用户订单再处理售后的流程适配为列表加详情查询。", [_check("list_user_orders", ["success", "ORD-20240118-004"]), _check("query_order", ["success", "ORD-20240118-004"], {"order_id": "ORD-20240118-004"})], min_tool_calls=2),
        _case("agent_22", "81", "退款前查询订单详情", ["我想退掉订单 ORD-20240122-005 的吸尘器，先帮我确认订单状态"], "return_request", ["戴森"], ["query_order"], "Bench 的取消前确认步骤适配为先查询订单，敏感退款操作留给确认后的下一轮。", [_check("query_order", ["success", "ORD-20240122-005"], {"order_id": "ORD-20240122-005"})], min_tool_calls=1),
        _case("agent_23", "10", "复杂退款支付方式超出能力边界转人工", ["我有两笔订单，想把退款分别退到另一张支付方式；如果系统不能处理，请帮我转人工客服"], "return_request", ["退款"], ["list_user_orders"], "保留 Bench 的跨支付方式退款场景；Ecom 没有支付方式切换工具，因此先列出订单并要求人工介入。", [_check("list_user_orders", ["success", "ORD-"])], requires_human=True, min_tool_calls=1),
        _case("agent_24", "50", "撤销已取消订单转人工", ["我想撤销之前已经取消的订单，让它尽快恢复配送；如果不能直接处理，请转人工"], "after_sale", ["人工"], [], "保留 Bench 的撤销订单场景；Ecom 没有恢复已取消订单的工具，只校验人工转接判断。", [], requires_human=True, min_tool_calls=None),
        _case("agent_25", "113", "没有订单号时列出全部订单", ["请把我所有订单列出来，我想确认哪个订单需要售后"], "order_query", ["ORD-20240115-001"], ["list_user_orders"], "将 Bench 的多订单处理入口映射到 Ecom 的 list_user_orders。", [_check("list_user_orders", ["success", "ORD-20240115-001"])], min_tool_calls=1),
    ]


def load_source(path: Path | None) -> dict[str, dict]:
    if path is None:
        with urllib.request.urlopen(DEFAULT_SOURCE_URL, timeout=60) as response:
            raw = response.read()
    else:
        raw = path.read_bytes()
    tasks = json.loads(raw.decode("utf-8"))
    return {str(task["id"]): task for task in tasks}


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 Ecom Agent Tool / Workflow 25 条测试集")
    parser.add_argument("--source", type=Path, default=None, help="τ-bench Retail tasks.json 路径")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="输出 JSON 路径")
    args = parser.parse_args()

    source = load_source(args.source)
    cases = build_cases()
    missing = [case["source_task_id"] for case in cases if case["source_task_id"] not in source]
    if missing:
        raise SystemExit(f"τ-bench source 缺少任务 ID: {missing}")

    for case in cases:
        task = source[case["source_task_id"]]
        case["source_actions"] = [item["name"] for item in task.get("evaluation_criteria", {}).get("actions", [])]

    payload = {
        "dataset": "Ecom Agent Tool / Workflow adapted from tau-bench Retail",
        "source": DEFAULT_SOURCE_URL,
        "mapping_policy": "每条任务映射到当前 Ecom Mock 数据和已注册工具；不模拟 τ-bench 原始数据库状态。",
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已生成 {len(cases)} 条用例: {args.output}")


if __name__ == "__main__":
    main()
