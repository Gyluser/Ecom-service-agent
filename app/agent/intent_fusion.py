"""客服意图三路融合分类。

LLM 负责理解语义，Embedding 负责处理同义改写，Pattern 为明确业务词提供
可解释的高精度信号。该模块只产出分类结果，不能替代 Agent 的工具规划。
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Protocol

from app.schemas.response import IntentType


class Encoder(Protocol):
    def encode(self, texts: list[str]) -> list[list[float]]: ...


INTENT_EXAMPLES: dict[IntentType, tuple[str, ...]] = {
    IntentType.ORDER_QUERY: ("帮我查订单", "物流到哪里了", "订单状态是什么"),
    IntentType.RETURN_REQUEST: ("我要退货退款", "不想要了想退掉", "申请换货"),
    IntentType.PRODUCT_CONSULT: ("这个商品多少钱", "有没有库存", "商品规格是什么"),
    IntentType.COMPLAINT: ("我要投诉商家", "服务太差了", "我要举报"),
    IntentType.AFTER_SALE: ("商品坏了怎么维修", "保修怎么处理", "质量问题售后"),
    IntentType.PROMOTION: ("有什么优惠活动", "优惠券怎么用", "满减规则"),
    IntentType.ACCOUNT: ("账号登录不了", "怎么修改手机号", "账户安全问题"),
    IntentType.GREETING: ("你好", "有人吗", "在吗"),
    IntentType.OTHER: ("我有一个其他问题",),
}

PATTERNS: dict[IntentType, tuple[str, ...]] = {
    IntentType.COMPLAINT: (r"投诉", r"举报", r"消协", r"赔偿", r"欺骗", r"垃圾"),
    IntentType.RETURN_REQUEST: (r"退货", r"退款", r"换货", r"想退", r"不想要"),
    IntentType.AFTER_SALE: (r"维修", r"保修", r"质量问题", r"故障", r"恢复.*订单", r"撤销.*取消", r"重新配送"),
    IntentType.ORDER_QUERY: (r"订单状态", r"订单详情", r"查.{0,3}订单", r"查询.*订单", r"列出.*订单", r"所有订单", r"物流", r"快递", r"运单"),
    IntentType.PRODUCT_CONSULT: (r"商品价格", r"商品库存", r"商品规格", r"有没有货", r"多少钱"),
    IntentType.PROMOTION: (r"优惠券", r"优惠活动", r"满减", r"折扣"),
    IntentType.ACCOUNT: (r"账号", r"账户", r"登录", r"密码", r"手机号"),
    IntentType.GREETING: (r"^(你好|您好|嗨|hello|hi|在吗)[!！。？? ]*$",),
}


@dataclass(frozen=True)
class IntentDecision:
    intent: IntentType
    confidence: float
    scores: dict[str, dict[str, float]]


class IntentFusionClassifier:
    """将 LLM、Embedding 和 Pattern 分数按固定权重融合。"""

    def __init__(
        self,
        encoder: Encoder | None = None,
        llm_weight: float = 0.30,
        embedding_weight: float = 0.25,
        pattern_weight: float = 0.45,
    ):
        if min(llm_weight, embedding_weight, pattern_weight) < 0:
            raise ValueError("意图融合权重不能为负数")
        if not math.isclose(llm_weight + embedding_weight + pattern_weight, 1.0):
            raise ValueError("意图融合权重之和必须为 1.0")
        self.encoder = encoder
        self.weights = {
            "llm": llm_weight,
            "embedding": embedding_weight,
            "pattern": pattern_weight,
        }
        self._prototype_vectors: dict[IntentType, list[float]] | None = None

    def classify(
        self,
        text: str,
        llm_intent: IntentType | None,
        llm_confidence: float | None = None,
    ) -> IntentDecision:
        labels = list(IntentType)
        llm_scores = {label: 0.0 for label in labels}
        if llm_intent is not None:
            llm_scores[llm_intent] = self._bounded(llm_confidence, default=0.8)

        embedding_scores = self._embedding_scores(text, labels)
        pattern_scores = self._pattern_scores(text, labels)
        fused = {
            label: (
                self.weights["llm"] * llm_scores[label]
                + self.weights["embedding"] * embedding_scores[label]
                + self.weights["pattern"] * pattern_scores[label]
            )
            for label in labels
        }
        # 同分时优先相信 LLM；没有 LLM 时按枚举定义稳定选择，便于回归测试。
        intent = max(labels, key=lambda label: (fused[label], llm_scores[label]))
        scores = {
            "llm": {label.value: round(score, 4) for label, score in llm_scores.items()},
            "embedding": {label.value: round(score, 4) for label, score in embedding_scores.items()},
            "pattern": {label.value: round(score, 4) for label, score in pattern_scores.items()},
            "fused": {label.value: round(score, 4) for label, score in fused.items()},
        }
        return IntentDecision(intent=intent, confidence=round(fused[intent], 4), scores=scores)

    @staticmethod
    def _bounded(value: float | None, default: float) -> float:
        try:
            return min(1.0, max(0.0, float(value))) if value is not None else default
        except (TypeError, ValueError):
            return default

    def _pattern_scores(self, text: str, labels: list[IntentType]) -> dict[IntentType, float]:
        normalized = text.lower().strip()
        scores = {
            label: 1.0 if any(re.search(pattern, normalized, re.I) for pattern in PATTERNS.get(label, ())) else 0.0
            for label in labels
        }
        # 高风险目标常与“订单/商品”等动作词共现。规则通道以最终业务目标优先，
        # 防止查询动作覆盖投诉、退款或恢复已取消订单的真实意图。
        for priority in (IntentType.COMPLAINT, IntentType.RETURN_REQUEST, IntentType.AFTER_SALE):
            if scores[priority] > 0:
                return {label: 1.0 if label is priority else 0.0 for label in labels}
        # 带订单号的“商品/金额/签收”是订单明细查询，不是独立商品咨询。
        # 仅在前面的退款、投诉、售后高风险目标未命中时提升订单查询。
        has_order_id = bool(re.search(r"ord-\d{8}-\d{3}", normalized, re.I))
        order_detail_words = ("订单信息", "状态", "金额", "签收", "买了什么", "哪些商品", "核实", "商品")
        if has_order_id and any(word in normalized for word in order_detail_words):
            return {label: 1.0 if label is IntentType.ORDER_QUERY else 0.0 for label in labels}
        return scores

    def _embedding_scores(self, text: str, labels: list[IntentType]) -> dict[IntentType, float]:
        empty = {label: 0.0 for label in labels}
        if self.encoder is None or not text.strip():
            return empty
        try:
            if self._prototype_vectors is None:
                prototype_texts = ["；".join(INTENT_EXAMPLES[label]) for label in labels]
                vectors = self.encoder.encode(prototype_texts)
                self._prototype_vectors = dict(zip(labels, vectors, strict=True))
            query = self.encoder.encode([text])[0]
            # 余弦相似度映射到 [0, 1]，避免负相似度反向影响融合。
            return {
                label: max(0.0, (self._cosine(query, vector) + 1.0) / 2.0)
                for label, vector in self._prototype_vectors.items()
            }
        except Exception:  # Embedding 服务不可用时不阻断主 Agent。
            return empty

    @staticmethod
    def _cosine(left: list[float], right: list[float]) -> float:
        denominator = math.sqrt(sum(x * x for x in left)) * math.sqrt(sum(x * x for x in right))
        return sum(x * y for x, y in zip(left, right)) / denominator if denominator else 0.0
