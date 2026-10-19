"""本地长期记忆检索：精确键匹配 + 字符 n-gram TF-IDF。"""

from __future__ import annotations

import math
import re
import hashlib
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable


ORDER_ID_RE = re.compile(r"(?:#?[A-Za-z]\d{7,}|[A-Za-z]{2,}-\d{8}-\d{3})")
PRODUCT_ID_RE = re.compile(r"\b\d{10}\b")
_PUNCTUATION_RE = re.compile(r"[\s\W_]+", re.UNICODE)
_QUERY_CUES = (
    "我之前", "我以前", "之前", "以前", "那笔", "那个", "这单", "那次",
    "帮我", "回忆一下", "回忆出来", "查下", "找一下", "找出来", "关于", "把", "记录", "记忆", "相关", "什么",
    "怎么回事", "怎么处理", "是不是", "有没有", "提过没有", "吗", "的", "是",
    "我", "用户", "订单", "商品", "业务", "零售", "诉求", "身份信息", "身份资料",
    "处理", "问题", "事项", "咨询", "找客服", "给客服", "留过", "哪些", "为了",
    "历史", "分别", "一起", "现在", "又",
)


@dataclass(frozen=True)
class MemoryMatch:
    """一次记忆召回结果，供 Prompt 和工具日志共同使用。"""

    fact: object
    score: float
    rank: int


def _normalized_text(value: str) -> str:
    return _PUNCTUATION_RE.sub("", (value or "").lower())


def _order_ids(text: str) -> set[str]:
    return {
        match.group(0).lstrip("#").lower()
        for match in ORDER_ID_RE.finditer(text or "")
    }


def _product_ids(text: str) -> set[str]:
    return {match.group(0) for match in PRODUCT_ID_RE.finditer(text or "")}


def _residual_query_terms(text: str) -> set[str]:
    """Keep likely user/product nouns after removing generic recall wording."""
    residual = _normalized_text(text)
    residual = residual.replace("身份资料", "身份信息")
    for cue in sorted(_QUERY_CUES, key=len, reverse=True):
        residual = residual.replace(_normalized_text(cue), " ")
    return {
        term for term in re.findall(r"[\u4e00-\u9fff]{3,}", residual)
        if term
    }


def _topic_key(text: str) -> str:
    """从中文记忆中提取稳定的主题键，用于冲突归并和精确召回。"""
    normalized = _normalized_text(text)
    order_match = ORDER_ID_RE.search(text or "")
    if order_match:
        order_id = order_match.group(0).lstrip("#").lower()
        if any(word in normalized for word in (
            "状态", "已发货", "已签收", "待处理", "已完成", "已取消",
            "已退款", "处理中", "已送达",
        )):
            return f"order:{order_id}:status"
        if any(word in normalized for word in ("退款原因", "退款理由")):
            return f"order:{order_id}:refund_reason"
        if any(word in normalized for word in ("送达", "配送日期", "到达")):
            return f"order:{order_id}:delivery"
        if any(word in normalized for word in ("物流单号", "运单", "快递")):
            return f"order:{order_id}:logistics"
        digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:10]
        return f"order:{order_id}:general:{digest}"

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
        order_id = order_match.group(0).lstrip("#").lower()
        keys.add(f"order:{order_id}:entity")
        if any(word in normalized for word in (
            "状态", "已发货", "已签收", "待处理", "已完成", "已取消",
            "已退款", "处理中", "已送达",
        )):
            keys.add(f"order:{order_id}:status")
        if any(word in normalized for word in ("退款原因", "退款理由")):
            keys.add(f"order:{order_id}:refund_reason")
        if any(word in normalized for word in ("送达", "配送日期", "到达")):
            keys.add(f"order:{order_id}:delivery")
        if any(word in normalized for word in ("物流单号", "运单", "快递", "物流")):
            keys.add(f"order:{order_id}:logistics")
    for product_id in _product_ids(text):
        keys.add(f"product:{product_id}")
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
    if "换货" in normalized:
        keys.add("issue:exchange")
    if any(word in normalized for word in ("修改订单商品", "修改商品")):
        keys.add("issue:modify_items")
    if any(word in normalized for word in ("修改订单地址", "修改用户地址")):
        keys.add("issue:address")
    if "修改支付方式" in normalized:
        keys.add("issue:payment")
    if "取消订单" in normalized:
        keys.add("issue:cancel")
    if any(word in normalized for word in ("库存", "有货")):
        keys.add("product:stock")
    if any(word in normalized for word in ("规格", "详细规格")):
        keys.add("preference:specification")
    if any(word in normalized for word in ("价格", "优惠", "预算")):
        keys.add("preference:price")
    if any(word in normalized for word in ("预计送达", "送到", "送达", "配送日期", "时效")):
        keys.add("issue:delivery")
    if any(word in normalized for word in ("身份信息", "身份资料", "姓名", "邮箱", "邮编")):
        keys.add("identity:profile")
    if any(word in normalized for word in ("业务诉求", "诉求", "原因", "问题", "售后", "找客服")):
        keys.add("issue:request")
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
        query_order_ids = _order_ids(query)
        query_product_ids = _product_ids(query)
        document_grams = [_char_ngrams(getattr(fact, "content", "")) for fact in candidates]
        document_texts = [getattr(fact, "content", "").lower() for fact in candidates]
        residual_terms = _residual_query_terms(query) if not (
            query_order_ids or query_product_ids
        ) else set()
        if residual_terms and not any(
            term in content
            for term in residual_terms
            for content in document_texts
        ):
            return []
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
            fact_order_ids = _order_ids(fact_content)
            fact_product_ids = _product_ids(fact_content)
            order_hit = bool(query_order_ids & fact_order_ids)
            product_hit = bool(query_product_ids & fact_product_ids)
            if query_order_ids and query_product_ids:
                if not (order_hit or product_hit):
                    continue
            elif query_order_ids and not order_hit:
                continue
            elif query_product_ids and not product_hit:
                continue
            exact_key = (
                fact_key == query_key
                or (query_key.startswith("preference:") and fact_key == query_key)
                or bool(query_keys & fact_keys)
            )
            if exact_key:
                score += 0.5
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
