"""生成平衡版 100 条 Retriever-only RAG 测试集。

组成：保留原始 50 条，新增 50 条单 chunk 场景，其中 17 条为 hard_negative。
新增问法只改变自然语言表达；gold 章节和政策事实保持来自 Ecom 知识文档。
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
ORIGINAL_PATH = ROOT / "app" / "evaluation" / "rag" / "data" / "chunk_cases_50.json"
SCENARIO_CATALOG_PATH = ROOT / "app" / "evaluation" / "rag" / "data" / "chunk_cases_100.json"
OUTPUT_PATH = ROOT / "app" / "evaluation" / "rag" / "data" / "chunk_cases_100_1.json"

QUERY_REWRITES = {
    # 困难样本：保留条件冲突，但不将政策概念完全隐去。
    "tau100_002": "账户过去一年的消费已经达到最高档，送到偏远地点的大件仍被加配送费；会员说明和账单冲突时该按哪条算？",
    "tau100_003": "我注册刚满八天才准备下第一单，之前领到的新人券为什么不能用了？",
    "tau100_007": "积分余额明明够，为什么这次结账最多只能抵订单金额的一半，是每单有限制吗？",
    "tau100_015": "设备偶尔会出问题，申请质量退换货时除了照片，是不是还得上传故障演示视频？",
    "tau100_017": "订单还显示未发货，我根本没收到东西；售后只把钱退回来、不用寄货，这种情况该选哪个？",
    "tau100_018": "包裹拒收后已经退回去了，原路退款通常要等五到七个工作日吗？",
    "tau100_020": "收货地点不在常规国内配送区域，结算页还能继续，但这笔订单最终会安排派送吗？",
    "tau100_021": "不在市区的县城地址，商家口头说能次日到；这种时效承诺能当成确定安排吗？",
    "tau100_026": "订单已经发货才发现门牌号写错了，还能在平台上改收货地址吗？",
    # 普通样本：增加真实上下文和条件，但不改 gold 政策事实。
    "tau100_006": "到生日那阵子，账户会不会自动给点专属东西？",
    "tau100_005": "我是黄金会员，寄到新疆时订单把普通运费和偏远附加费分开算了，会员包邮能抵掉后面那笔吗？",
    "tau100_008": "衣服到手隔了一周多才发现尺码不合适，没剪吊牌也没下水；这种情况通常还能按不喜欢处理吗？",
    "tau100_009": "刻了名字的保温杯属于定制商品，即使没拆没用过，还能七天无理由退吗？",
    "tau100_010": "冷链海鲜这类鲜活易腐商品，即使还没拆封，也能按七天无理由退吗？",
    "tau100_014": "那件东西刚开始没看出问题，隔了半个月才发现不对；现在还来得及找商家处理吗？",
    "tau100_033": "晒单得到的积分分好几次进账，最早那一批已经放了很久；结账时它还会被算作可用额度吗？",
    "tau100_029": "账户里第一次购物的那笔额度一直没用，再结账时突然显示不可用；它的失效时间到底从哪一天开始算？",
    "tau100_034": "生日月拿到的会员权益在付款页没法用，账户等级和订单金额看着都够；是不是还有限制条件？",
    "tau100_037": "普通商品申请退货后，平台是先审核还是先等寄回，通常会在多长时间内处理完审核？",
    "tau100_038": "商家确认缺货而且商品还没发出时，我能不能直接申请仅退款？",
    "tau100_039": "包裹在代收点放了两天还没取，可能会被退回的话，我现在应该怎么处理？",
    "tau100_040": "物流卡在同一个状态三天没动静，我应该直接找谁处理这种配送异常？",
    "tau100_041": "电脑外箱已经破了，快递员催我先签收；贵重物品这种情况该怎么验货、要不要直接拒收？",
    "tau100_042": "付款超过半小时平台还没识别成功，联系客服时需要准备哪种支付凭证？",
    "tau100_043": "想买的那款一直买不到，页面隔阵子才冒出来一次；有没有办法不用每天反复点进去看？",
    "tau100_044": "刚买的鞋总觉得不太对，又怕一打开包装后面就说不清；这种情况我该从哪里反映？",
    "tau100_047": "账户还有没用完的优惠券和积分，我现在申请注销会不会被系统拦住？",
    "tau100_049": "我这台 2020 年的 MateBook D 15 重装过系统，老机型还能不能使用 SuperTurbo 加速功能？",
    "tau100_050": "电脑之前拿到非官方店处理过，后来又出故障；即使保修卡还在，官方售后还会按保修处理吗？",
}


def _read_cases(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = payload.get("cases")
    if not isinstance(cases, list):
        raise ValueError(f"{path.name} 缺少 cases")
    return cases


def build_payload() -> dict[str, Any]:
    original_cases = _read_cases(ORIGINAL_PATH)
    catalog_cases = _read_cases(SCENARIO_CATALOG_PATH)
    if len(original_cases) != 50 or len(catalog_cases) != 100:
        raise ValueError("输入数据集数量不符合预期")

    added_cases = [copy.deepcopy(case) for case in catalog_cases[50:]]
    if len(added_cases) != 50 or any(case["requires_multi_chunk"] for case in added_cases):
        raise ValueError("新增场景池必须恰好为 50 条单 chunk 用例")

    for index, case in enumerate(added_cases, start=1):
        original_id = case["id"]
        case["id"] = f"tau100_1_{index:03d}"
        case["source_scope"] = "tau_bench_balanced_100_1"
        case["query"] = QUERY_REWRITES.get(original_id, case["query"])
        case["notes"] = "基于官方政策 gold 章节的单 chunk 口语化场景；保持政策条件和证据粒度。"

    cases = copy.deepcopy(original_cases) + added_cases
    normalized_queries = ["".join(case["query"].split()) for case in cases]
    if len(cases) != 100 or len(set(case["id"] for case in cases)) != 100:
        raise ValueError("合并后用例数量或 id 不正确")
    if len(set(normalized_queries)) != 100:
        raise ValueError("生成后出现重复 query")
    if sum("hard_negative" in case["query_type"] for case in added_cases) != 17:
        raise ValueError("新增 hard_negative 必须恰好为 17 条")
    if any(case["requires_multi_chunk"] for case in added_cases):
        raise ValueError("新增样本必须全部为单 chunk")

    return {
        "dataset": "Ecom RAG chunk-level retrieval evaluation: balanced-100-1",
        "description": "保留原始 50 条；新增 50 条单 chunk 口语化场景，其中 17 条为困难样本。新增问题与原始困难样本保持相近的条件冲突与证据粒度。",
        "generation": {
            "method": "document_grounded_balanced_prompt_variation",
            "source_policy": "业务事实和 gold_evidence 沿用已核验的 Ecom 官方知识文档章节；chunk 仅为运行时检索单位。",
        },
        "composition": {
            "original_case_count": 50,
            "added_case_count": 50,
            "added_hard_negative_count": 17,
            "added_cases_single_chunk_only": True,
        },
        "cases": cases,
    }


def main() -> None:
    payload = build_payload()
    OUTPUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"已生成 {len(payload['cases'])} 条用例：{OUTPUT_PATH}")


if __name__ == "__main__":
    main()
