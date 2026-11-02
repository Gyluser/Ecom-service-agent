"""真实 LLM 的 10 条意图路由冒烟测试（5 条单意图 + 5 条多意图）。

每条用例创建独立的临时 session、Memory 和 Context 目录，不写入默认运行数据。
多意图请求以 ``expected_intent`` 表示用户的主诉求，同时用 ``required_tools``
检查 ReAct 是否完成各个明确的业务动作。
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.agent.chat import EcomAgent
from app.config.settings import settings
from app.schemas.response import IntentType


@dataclass(frozen=True)
class Case:
    case_id: str
    kind: str
    query: str
    expected_intent: IntentType
    required_tools: frozenset[str] = frozenset()


CASES = (
    Case("single_greeting", "single", "你好", IntentType.GREETING),
    Case(
        "single_product", "single", "这款耳机多少钱，还有没有库存？",
        IntentType.PRODUCT_CONSULT, frozenset({"query_product"}),
    ),
    Case(
        "single_order", "single", "请查询 ORD-20240115-001 的订单状态。",
        IntentType.ORDER_QUERY, frozenset({"query_order"}),
    ),
    Case(
        "single_promotion", "single", "满减活动的使用规则是什么？",
        IntentType.PROMOTION, frozenset({"search_knowledge"}),
    ),
    Case(
        "single_refund", "single", "确认直接退 ORD-20240110-003，质量有问题。",
        IntentType.RETURN_REQUEST, frozenset({"apply_refund"}),
    ),
    Case(
        "multi_logistics_refund", "multi",
        "ORD-20240115-001 的物流到哪里了？如果还没发货，我想退款。",
        IntentType.RETURN_REQUEST, frozenset({"query_logistics"}),
    ),
    Case(
        "multi_product_promotion", "multi",
        "这款耳机还有库存吗？能不能同时使用优惠券？",
        IntentType.PRODUCT_CONSULT, frozenset({"query_product", "search_knowledge"}),
    ),
    Case(
        "multi_after_sale_policy", "multi",
        "ORD-20240120-002 被取消了，怎么恢复订单？退款规则也请告诉我。",
        IntentType.AFTER_SALE, frozenset({"query_order", "search_knowledge"}),
    ),
    Case(
        "multi_complaint_refund", "multi",
        "服务太差我要投诉；同时确认直接退 ORD-20240110-003，原因是质量问题。",
        IntentType.COMPLAINT, frozenset({"query_order", "apply_refund"}),
    ),
    Case(
        "multi_order_logistics", "multi",
        "请查 ORD-20240115-001 买了什么、金额多少，以及物流轨迹。",
        IntentType.ORDER_QUERY, frozenset({"query_order", "query_logistics"}),
    ),
)


def _run_case(case: Case, root: Path) -> dict:
    agent = EcomAgent(
        session_path=str(root / "sessions" / f"{case.case_id}.json"),
        user_id=f"intent-routing-{case.case_id}",
        memory_dir=str(root / "memory"),
        context_archive_dir=str(root / "context" / case.case_id),
    )
    calls: list[str] = []
    original_execute = agent.tool_manager.execute_tool

    def tracked_execute(name: str, arguments: dict) -> str:
        calls.append(name)
        return original_execute(name, arguments)

    agent.tool_manager.execute_tool = tracked_execute
    try:
        response = agent.chat(case.query)
        routed = agent._current_intent_decision
        return {
            "case": case,
            "response_intent": response.intent,
            "routed_intent": routed.intent if routed else None,
            "tools": calls,
            "intent_ok": response.intent is case.expected_intent,
            "tools_ok": case.required_tools.issubset(calls),
            "reply_nonempty": bool(response.reply.strip()),
        }
    finally:
        # 不调用 agent.close()，避免将本次临时会话提炼并持久化为长期记忆。
        agent.tool_manager.close()


def main() -> None:
    if not settings.openai_api_key:
        raise RuntimeError("未配置 OPENAI_API_KEY，无法执行真实 LLM 冒烟测试。")

    print("真实 LLM 意图路由冒烟测试：10 条（5 单意图 + 5 多意图）")
    results = []
    with tempfile.TemporaryDirectory(prefix="intent-routing-live-") as temp_dir:
        root = Path(temp_dir)
        for index, case in enumerate(CASES, start=1):
            try:
                result = _run_case(case, root)
            except Exception as exc:  # noqa: BLE001 - 汇总真实运行异常
                result = {
                    "case": case, "response_intent": None, "routed_intent": None,
                    "tools": [], "intent_ok": False, "tools_ok": False,
                    "reply_nonempty": False, "error": f"{type(exc).__name__}: {exc}",
                }
            results.append(result)
            status = "PASS" if result["intent_ok"] and result["tools_ok"] and result["reply_nonempty"] else "FAIL"
            print(
                f"[{index:02d}] {status:<4} {case.case_id:<24} "
                f"intent={getattr(result['response_intent'], 'value', '-'):<16} "
                f"tools={result['tools']}"
                + (f" error={result['error']}" if result.get("error") else "")
            )

    intent_correct = sum(item["intent_ok"] for item in results)
    tools_correct = sum(item["tools_ok"] for item in results)
    normal_runs = sum(item["reply_nonempty"] for item in results)
    full_pass = sum(
        item["intent_ok"] and item["tools_ok"] and item["reply_nonempty"]
        for item in results
    )
    total = len(results)
    print("\n汇总")
    print(f"  正常返回率: {normal_runs}/{total} ({normal_runs / total:.0%})")
    print(f"  主意图准确率: {intent_correct}/{total} ({intent_correct / total:.0%})")
    print(f"  工具需求满足率: {tools_correct}/{total} ({tools_correct / total:.0%})")
    print(f"  全部通过: {full_pass}/{total} ({full_pass / total:.0%})")

    if full_pass != total:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
