"""三路融合意图分类与四层评估指标的离线单元测试。"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.agent.intent_fusion import IntentFusionClassifier
from app.schemas.response import IntentType


class FakeEncoder:
    """固定向量，避免单元测试下载模型或请求 API。"""

    def encode(self, texts):
        vectors = []
        for text in texts:
            if "退款" in text or "退货" in text:
                vectors.append([1.0, 0.0])
            elif "订单" in text or "物流" in text:
                vectors.append([0.0, 1.0])
            else:
                vectors.append([0.5, 0.5])
        return vectors


def main():
    classifier = IntentFusionClassifier(encoder=FakeEncoder())

    refund = classifier.classify("订单不要了，我想申请退款", IntentType.ORDER_QUERY, 0.6)
    assert refund.intent is IntentType.RETURN_REQUEST, refund
    assert refund.scores["pattern"]["return_request"] == 1.0
    assert refund.scores["fused"]["return_request"] > refund.scores["fused"]["order_query"]

    order = classifier.classify("帮我查一下订单物流", IntentType.ORDER_QUERY, 0.9)
    assert order.intent is IntentType.ORDER_QUERY, order
    assert order.scores["embedding"]["order_query"] > 0.9

    order_detail = classifier.classify(
        "请核实 ORD-20240115-001 的订单信息，我想知道商品和金额。",
        IntentType.PRODUCT_CONSULT,
        0.9,
    )
    assert order_detail.intent is IntentType.ORDER_QUERY, order_detail

    refund_with_order = classifier.classify(
        "订单 ORD-20240115-001 我想申请退款。",
        IntentType.ORDER_QUERY,
        0.9,
    )
    assert refund_with_order.intent is IntentType.RETURN_REQUEST, refund_with_order

    dataset_path = ROOT / "app" / "evaluation" / "data" / "cases_agent_100.json"
    cases = json.loads(dataset_path.read_text(encoding="utf-8"))["cases"]
    valid = {intent.value for intent in IntentType}
    assert len(cases) == 100
    assert all(case.get("expected_intent") in valid for case in cases)
    print("intent fusion: PASS")


if __name__ == "__main__":
    main()
