"""本地长期记忆检索：精确键匹配 + 字符 n-gram TF-IDF。"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable


ORDER_ID_RE = re.compile(r"(?:#[A-Za-z]\d{7,}|[A-Za-z]{2,}-\d{8}-\d{3})")
_PUNCTUATION_RE = re.compile(r"[\s\W_]+", re.UNICODE)


@dataclass(frozen=True)
class MemoryMatch:
    """一次记忆召回结果，供 Prompt 和工具日志共同使用。"""

    fact: object
    score: float
    rank: int


def _normalized_text(value: str) -> str:
    return _PUNCTUATION_RE.sub("", (value or "").lower())


def _topic_key(text: str) -> str:
    """从中文记忆中提取稳定的主题键，用于冲突归并和精确召回。"""
    normalized = _normalized_text(text)
    order_match = ORDER_ID_RE.search(text or "")
    if order_match:
        order_id = order_match.group(0).lower()
        if any(word in normalized for word in ("状态", "已发货", "已签收", "待处理")):
            return f"order:{order_id}:status"
        if any(word in normalized for word in ("退款原因", "退款理由")):
            return f"order:{order_id}:refund_reason"
        if any(word in normalized for word in ("送达", "配送日期", "到达")):
            return f"order:{order_id}:delivery"
        if any(word in normalized for word in ("物流单号", "运单", "快递")):
            return f"order:{order_id}:logistics"
        return f"order:{order_id}:general"

    preference_cue = any(word in normalized for word in ("偏好", "喜欢", "更在意"))
    if any(word in normalized for word in ("颜色", "黑色", "白色", "蓝色", "红色", "外观")) and preference_cue:
        return "preference:color"
    if any(word in normalized for word in ("配送偏好", "快递", "顺丰", "圆通", "配送")) and preference_cue:
        return "preference:delivery"
    if any(word in normalized for word in (
        "商品类别", "品类", "家居用品", "数码产品", "电子产品", "家用电器",
    )) and preference_cue:
        return "preference:category"
    return "text:" + normalized[:48]


def _topic_keys(text: str, allow_generic_category: bool = False) -> set[str]:
    """提取查询与文档共享的主题标签；它比冲突归并键更宽。"""
    normalized = _normalized_text(text)
    keys: set[str] = set()
    preference_cue = any(word in normalized for word in ("偏好", "喜欢", "更在意"))
    order_match = ORDER_ID_RE.search(text or "")
    if order_match:
        order_id = order_match.group(0).lower()
        if any(word in normalized for word in ("状态", "已发货", "已签收", "待处理")):
            keys.add(f"order:{order_id}:status")
        if any(word in normalized for word in ("退款原因", "退款理由")):
            keys.add(f"order:{order_id}:refund_reason")
        if any(word in normalized for word in ("送达", "配送日期", "到达")):
            keys.add(f"order:{order_id}:delivery")
        if any(word in normalized for word in ("物流单号", "运单", "快递", "物流")):
            keys.add(f"order:{order_id}:logistics")
    if any(word in normalized for word in ("颜色", "黑色", "白色", "蓝色", "红色", "外观")):
        if preference_cue or "颜色" in normalized:
            keys.add("preference:color")
    if any(word in normalized for word in ("配送", "快递", "顺丰", "圆通")) and (preference_cue or "快递" in normalized):
        keys.add("preference:delivery")
    if any(word in normalized for word in (
        "商品类别", "品类", "家居用品", "数码产品", "电子产品", "家用电器",
    )) and (preference_cue or "品类" in normalized or "类别" in normalized):
        keys.add("preference:category")
    elif (
        allow_generic_category
        and "商品" in normalized
        and "偏好" in normalized
        and not any(word in normalized for word in ("颜色", "配送", "规格", "价格"))
    ):
        keys.add("preference:category")
    if any(word in normalized for word in ("物流", "快递", "运单", "到哪里", "到哪")):
        keys.add("issue:logistics")
    if any(word in normalized for word in ("退款", "退货")):
        keys.add("issue:refund")
    if any(word in normalized for word in ("库存", "有货")):
        keys.add("product:stock")
    if any(word in normalized for word in ("规格", "详细规格")):
        keys.add("preference:specification")
    if any(word in normalized for word in ("价格", "优惠", "预算")):
        keys.add("preference:price")
    if any(word in normalized for word in ("预计送达", "送到", "送达", "配送日期", "时效")):
        keys.add("issue:delivery")
    return keys


def derive_memory_key(content: str, category: str = "") -> str:
    """返回用于记忆归并/精确匹配的稳定键。"""
    return _topic_key(content)


def _char_ngrams(text: str) -> Counter[str]:
    normalized = _normalized_text(text)
    grams: Counter[str] = Counter()
    for size in (2, 3):
        for index in range(len(normalized) - size + 1):
            grams[normalized[index:index + size]] += 1
    return grams


def _timestamp(value: object) -> datetime:
    if not isinstance(value, str) or not value:
        return datetime.min
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        return datetime.min


class MemoryRetriever:
    """无外部依赖的轻量检索器，适合当前 JSON 规模的长期记忆。"""

    def __init__(self, min_score: float = 0.05):
        self.min_score = min_score

    def retrieve(
        self,
        query: str,
        facts: Iterable[object],
        top_k: int = 10,
    ) -> list[MemoryMatch]:
        candidates = [
            fact for fact in facts
            if getattr(fact, "status", "active") == "active"
            and getattr(fact, "content", "")
        ]
        if not candidates or top_k <= 0:
            return []

        if not query or not query.strip():
            return [
                MemoryMatch(fact=fact, score=0.0, rank=index)
                for index, fact in enumerate(candidates[:top_k], start=1)
            ]

        query_grams = _char_ngrams(query)
        query_key = derive_memory_key(query)
        query_keys = _topic_keys(query, allow_generic_category=True)
        strong_terms = ("咖啡机", "吸尘器", "AirPods", "手机")
        query_specific_terms = [term.lower() for term in strong_terms if term.lower() in query.lower()]
        document_grams = [_char_ngrams(getattr(fact, "content", "")) for fact in candidates]
        document_frequency: Counter[str] = Counter()
        for grams in document_grams:
            document_frequency.update(grams.keys())

        document_count = len(candidates)

        def weighted_vector(grams: Counter[str]) -> dict[str, float]:
            return {
                term: (1.0 + math.log(count)) * (
                    math.log((document_count + 1) / (document_frequency[term] + 1)) + 1.0
                )
                for term, count in grams.items()
            }

        query_vector = weighted_vector(query_grams)
        query_norm = math.sqrt(sum(value * value for value in query_vector.values()))
        scored: list[tuple[object, float, bool]] = []
        for fact, grams in zip(candidates, document_grams):
            vector = weighted_vector(grams)
            norm = math.sqrt(sum(value * value for value in vector.values()))
            dot = sum(query_vector.get(term, 0.0) * value for term, value in vector.items())
            score = dot / (query_norm * norm) if query_norm and norm else 0.0
            fact_key = getattr(fact, "key", "") or derive_memory_key(
                getattr(fact, "content", ""), getattr(fact, "category", "")
            )
            fact_content = getattr(fact, "content", "")
            fact_keys = _topic_keys(fact_content)
            if query_specific_terms and not any(
                term in fact_content.lower() for term in query_specific_terms
            ) and not any(key.startswith("order:") for key in query_keys):
                continue
            exact_key = (
                fact_key == query_key
                or (query_key.startswith("order:") and fact_key.startswith(query_key))
                or (query_key.startswith("preference:") and fact_key == query_key)
                or bool(query_keys & fact_keys)
            )
            if exact_key:
                score = max(score, 1.0)
            if score >= self.min_score or exact_key:
                scored.append((fact, score, exact_key))

        scored.sort(
            key=lambda item: (
                item[2],
                item[1],
                _timestamp(getattr(item[0], "updated_at", "") or getattr(item[0], "created_at", "")),
            ),
            reverse=True,
        )
        return [
            MemoryMatch(fact=fact, score=round(score, 6), rank=index)
            for index, (fact, score, _) in enumerate(scored[:top_k], start=1)
        ]
