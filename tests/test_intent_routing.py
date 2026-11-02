"""前置三路意图路由的离线回归测试，不请求真实模型或 MCP。"""

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.agent.chat import EcomAgent
from app.agent.intent_fusion import IntentDecision, IntentFusionClassifier
from app.schemas.response import IntentType


TOOL_NAMES = (
    "query_order", "list_user_orders", "query_logistics", "apply_refund",
    "query_product", "search_knowledge", "recall_user_memory",
)


def _tool_definitions():
    return [
        {"type": "function", "function": {"name": name}}
        for name in TOOL_NAMES
    ]


def _routing_agent(user_text: str, intent: IntentType) -> EcomAgent:
    """构造只覆盖工具筛选所需字段的 Agent，避免初始化外部依赖。"""
    agent = EcomAgent.__new__(EcomAgent)
    agent.raw_messages = [{"role": "user", "content": user_text}]
    agent._current_intent_decision = IntentDecision(intent, 0.9, {})
    agent._current_turn_extra_tools = set()
    agent._current_turn_called_tools = set()
    agent.tool_manager = SimpleNamespace(tool_definitions=_tool_definitions())
    return agent


def test_front_llm_is_called_and_fused():
    class FakeCompletions:
        def __init__(self):
            self.calls = []

        def create(self, **kwargs):
            self.calls.append(kwargs)
            content = '{"intent":"return_request","confidence":0.95}'
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
            )

    completions = FakeCompletions()
    agent = EcomAgent.__new__(EcomAgent)
    agent.client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions)
    )
    agent.model = "test-model"
    # 不配置 Embedding 时，它安全降级为零分；Pattern 和 LLM 仍共同参与融合。
    agent.intent_classifier = IntentFusionClassifier(
        llm_weight=0.30, embedding_weight=0.25, pattern_weight=0.45,
    )

    decision = agent._classify_current_intent("订单不要了，我要退款")

    assert len(completions.calls) == 1
    assert completions.calls[0]["temperature"] == 0
    assert decision.intent is IntentType.RETURN_REQUEST
    assert decision.scores["llm"][IntentType.RETURN_REQUEST.value] == 0.95
    assert decision.scores["pattern"][IntentType.RETURN_REQUEST.value] == 1.0


def test_intent_candidates_do_not_bypass_refund_confirmation():
    unconfirmed = _routing_agent(
        "ORD-20240115-001 我不想要了，想退款", IntentType.RETURN_REQUEST,
    )
    names = {
        item["function"]["name"]
        for item in unconfirmed._tool_definitions_for_current_turn()
    }
    assert {"query_order", "list_user_orders", "search_knowledge"} <= names
    assert "apply_refund" not in names

    confirmed = _routing_agent(
        "确认直接退 ORD-20240115-001，我要退款", IntentType.RETURN_REQUEST,
    )
    names = {
        item["function"]["name"]
        for item in confirmed._tool_definitions_for_current_turn()
    }
    assert "apply_refund" in names


def test_existing_dynamic_tool_supplement_is_retained():
    agent = _routing_agent("我想看看商品", IntentType.PRODUCT_CONSULT)
    agent._current_turn_called_tools = {"query_product"}
    agent._current_turn_extra_tools = {"list_user_orders"}

    names = {
        item["function"]["name"]
        for item in agent._tool_definitions_for_current_turn()
    }
    assert "query_product" not in names
    assert "list_user_orders" in names


def main():
    test_front_llm_is_called_and_fused()
    test_intent_candidates_do_not_bypass_refund_confirmation()
    test_existing_dynamic_tool_supplement_is_retained()
    print("intent routing: PASS")


if __name__ == "__main__":
    main()
