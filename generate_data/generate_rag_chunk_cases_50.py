"""生成 50 条 Ecom RAG chunk-level Retriever 测试数据。

构成：
- 保留 tests/test_rag.py 中最早的 4 条 Retriever 检索问题；
- 保留已有的 15 条 τ-bench Retail 风格场景化用例；
- 再新增 31 条不同场景的 τ-bench Retail 风格用例。

新增问题只借鉴 τ-bench 的自然口语、多目标和条件约束表达，业务事实与
gold_evidence 仍完全以当前 Ecom rag_knowledge 文档为准。

运行：
    python generate_data/generate_rag_chunk_cases_50.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = (
    ROOT / "app" / "evaluation" / "rag" / "data" / "chunk_cases_50.json"
)

TAU_BENCH_REPOSITORY = "https://github.com/sierra-research/tau-bench"
TAU_BENCH_TASKS = (
    "https://github.com/sierra-research/tau-bench/blob/main/"
    "tau_bench/envs/retail/tasks.py"
)


def _evidence(
    doc: str,
    section_path: list[str],
    text_anchor: str,
    relevance: int = 2,
) -> dict[str, Any]:
    return {
        "doc": doc,
        "section_path": section_path,
        "text_anchor": text_anchor,
        "relevance": relevance,
        "required": True,
    }


def _case(
    case_id: str,
    source_scope: str,
    query: str,
    query_type: str,
    expected_docs: list[str],
    gold_evidence: list[dict[str, Any]],
    requires_multi_chunk: bool,
    notes: str,
) -> dict[str, Any]:
    return {
        "id": case_id,
        "source_scope": source_scope,
        "query": query,
        "query_type": [query_type],
        "answerable": True,
        "expected_docs": expected_docs,
        "gold_evidence": gold_evidence,
        "requires_multi_chunk": requires_multi_chunk,
        "notes": notes,
    }


def build_project_original_cases() -> list[dict[str, Any]]:
    """保留 tests/test_rag.py 中原始的 4 条 Top-K 检索问题。"""

    return [
        _case(
            "project_original_001",
            "project_original",
            "七天无理由退货可以吗",
            "direct",
            ["退换货政策"],
            [
                _evidence(
                    "退换货政策",
                    ["并夕夕 退换货政策", "一、七天无理由退货", "1.1 适用范围"],
                    "签收商品之日起 7 天内",
                )
            ],
            False,
            "原样保留项目最早的 Retriever Top-K 测试问题。",
        ),
        _case(
            "project_original_002",
            "project_original",
            "钻石会员有哪些权益",
            "direct",
            ["会员权益"],
            [
                _evidence(
                    "会员权益",
                    ["并夕夕 会员权益", "二、各等级核心权益", "2.4 钻石会员"],
                    "全场包邮（含偏远地区）",
                )
            ],
            False,
            "原样保留项目最早的 Retriever Top-K 测试问题。",
        ),
        _case(
            "project_original_003",
            "project_original",
            "偏远地区还包邮吗",
            "direct",
            ["配送说明"],
            [
                _evidence(
                    "配送说明",
                    ["并夕夕 配送说明", "三、运费规则", "偏远地区与会员包邮例外"],
                    "不参与全场包邮",
                )
            ],
            False,
            "原样保留项目最早的 Retriever Top-K 测试问题。",
        ),
        _case(
            "project_original_004",
            "project_original",
            "忘记密码了怎么办",
            "direct",
            ["常见问题FAQ"],
            [
                _evidence(
                    "常见问题FAQ",
                    ["并夕夕 常见问题 FAQ", "四、账户与安全", "Q11：忘记密码怎么办？"],
                    "忘记密码",
                )
            ],
            False,
            "原样保留项目最早的 Retriever Top-K 测试问题。",
        ),
    ]


def build_tau_style_cases() -> list[dict[str, Any]]:
    """构造 15 条口语化、场景化、带约束的 Retriever 用例。"""

    return [
        {
            "id": "tau_style_001",
            "source_scope": "tau_bench_adapted",
            "query": "我那件衣服让店家绣了名字，收到后看着又不想要了，包装吊牌都没拆，这种还能直接退吗？",
            "query_type": ["hard_negative"],
            "answerable": True,
            "expected_docs": ["退换货政策"],
            "gold_evidence": [
                _evidence(
                    "退换货政策",
                    ["并夕夕 退换货政策", "一、七天无理由退货", "1.2 不适用商品"],
                    "个人定制类商品",
                )
            ],
            "requires_multi_chunk": False,
            "notes": "τ-bench 风格的具体退货场景；用日常描述表达定制商品例外。",
        },
        {
            "id": "tau_style_002",
            "source_scope": "tau_bench_adapted",
            "query": "买的电子卡券手滑激活了，可我一次都没用，这还能退不？",
            "query_type": ["hard_negative"],
            "answerable": True,
            "expected_docs": ["退换货政策"],
            "gold_evidence": [
                _evidence(
                    "退换货政策",
                    ["并夕夕 退换货政策", "一、七天无理由退货", "1.2 不适用商品"],
                    "已激活的软件序列号、电子卡券",
                )
            ],
            "requires_multi_chunk": False,
            "notes": "τ-bench 风格的简短追问；区分未使用与已激活两个条件。",
        },
        {
            "id": "tau_style_003",
            "source_scope": "tau_bench_adapted",
            "query": "手机送到的时候屏幕就裂了，我当时没注意已经签收；现在过去快一天了，我要补什么证明，还是当时就该拒收？",
            "query_type": ["ambiguous"],
            "answerable": True,
            "expected_docs": ["配送说明"],
            "gold_evidence": [
                _evidence(
                    "配送说明",
                    ["并夕夕 配送说明", "五、收货须知"],
                    "签收后发现外观破损",
                )
            ],
            "requires_multi_chunk": False,
            "notes": "τ-bench 风格的时序场景；同时给出签收前后两个状态。",
        },
        {
            "id": "tau_style_004",
            "source_scope": "tau_bench_adapted",
            "query": "一单里两样东西，杯子没问题我想留着，耳机到手就坏了；能只处理耳机吗，得拍什么、邮费算谁的？",
            "query_type": ["multi_chunk"],
            "answerable": True,
            "expected_docs": ["退换货政策"],
            "gold_evidence": [
                _evidence(
                    "退换货政策",
                    ["并夕夕 退换货政策", "二、质量问题退换货", "2.1 适用情形"],
                    "功能性故障",
                ),
                _evidence(
                    "退换货政策",
                    ["并夕夕 退换货政策", "二、质量问题退换货", "2.2 时效与运费"],
                    "运费由「并夕夕」承担",
                ),
                _evidence(
                    "退换货政策",
                    ["并夕夕 退换货政策", "二、质量问题退换货", "2.3 举证材料"],
                    "故障演示视频",
                ),
                _evidence(
                    "退换货政策",
                    ["并夕夕 退换货政策", "四、特殊场景", "4.3 部分退款"],
                    "只退其中一件或几件",
                ),
            ],
            "requires_multi_chunk": True,
            "notes": "τ-bench 风格的一次性多诉求；需要拼接资格、材料、运费和部分退款。",
        },
        {
            "id": "tau_style_005",
            "source_scope": "tau_bench_adapted",
            "query": "这快递也太离谱了，三天多没动静，刚更新又显示送到别人那儿了，我现在该让你们催还是找件？要报哪个号？",
            "query_type": ["paraphrase"],
            "answerable": True,
            "expected_docs": ["配送说明"],
            "gold_evidence": [
                _evidence(
                    "配送说明",
                    ["并夕夕 配送说明", "六、配送异常"],
                    "定位寻件",
                )
            ],
            "requires_multi_chunk": False,
            "notes": "参考 τ-bench 用户情绪和复合状态表达，问题本身不复述章节措辞。",
        },
        {
            "id": "tau_style_006",
            "source_scope": "tau_bench_adapted",
            "query": "地址填成公司了，货还没发，我能顺手把颜色也一起改了吗？要是已经出库了又怎么办？",
            "query_type": ["hard_negative"],
            "answerable": True,
            "expected_docs": ["常见问题FAQ"],
            "gold_evidence": [
                _evidence(
                    "常见问题FAQ",
                    ["并夕夕 常见问题 FAQ", "一、订单与支付", "Q3：能否修改订单信息？"],
                    "商品规格不可修改",
                )
            ],
            "requires_multi_chunk": False,
            "notes": "参考 τ-bench 的订单修改任务，把地址、规格和发货状态放在同一句里。",
        },
        {
            "id": "tau_style_007",
            "source_scope": "tau_bench_adapted",
            "query": "我钻石会员，买了台预售冰箱寄海南三沙。客服跟我说一定24小时发、还一分钱运费没有，这话靠谱吗？大概多久能到也一起说下。",
            "query_type": ["multi_chunk"],
            "answerable": True,
            "expected_docs": ["会员权益", "常见问题FAQ", "配送说明"],
            "gold_evidence": [
                _evidence(
                    "会员权益",
                    ["并夕夕 会员权益", "二、各等级核心权益", "2.4 钻石会员"],
                    "优先发货（24 小时内出库）",
                ),
                _evidence(
                    "常见问题FAQ",
                    ["并夕夕 常见问题 FAQ", "一、订单与支付", "Q1：下单后多久发货？"],
                    "预售商品按详情页公示发货时间为准",
                ),
                _evidence(
                    "配送说明",
                    ["并夕夕 配送说明", "一、配送范围"],
                    "海南三沙",
                ),
                _evidence(
                    "配送说明",
                    ["并夕夕 配送说明", "二、配送时效"],
                    "家具大件",
                ),
                _evidence(
                    "配送说明",
                    ["并夕夕 配送说明", "三、运费规则", "大件商品运费"],
                    "按体积重计费",
                ),
            ],
            "requires_multi_chunk": True,
            "notes": "τ-bench 风格的多条件核验；会员、预售、大件和偏远地区规则同时生效。",
        },
        {
            "id": "tau_style_008",
            "source_scope": "tau_bench_adapted",
            "query": "我黄金会员，在新疆喀什那边下了个98块的普通单。不是说会员免邮吗，怎么还提示要加钱，而且看着也要晚几天？",
            "query_type": ["multi_chunk"],
            "answerable": True,
            "expected_docs": ["会员权益", "配送说明"],
            "gold_evidence": [
                _evidence(
                    "会员权益",
                    ["并夕夕 会员权益", "二、各等级核心权益", "2.3 黄金会员"],
                    "不含偏远地区",
                ),
                _evidence(
                    "配送说明",
                    ["并夕夕 配送说明", "一、配送范围"],
                    "新疆喀什以西",
                ),
                _evidence(
                    "配送说明",
                    ["并夕夕 配送说明", "三、运费规则", "普通包邮规则"],
                    "不满则收取 8 元基础运费",
                ),
                _evidence(
                    "配送说明",
                    ["并夕夕 配送说明", "三、运费规则", "偏远地区与会员包邮例外"],
                    "不参与全场包邮",
                ),
            ],
            "requires_multi_chunk": True,
            "notes": "τ-bench 风格的规则冲突核验；金额、会员等级和目的地共同决定结果。",
        },
        {
            "id": "tau_style_009",
            "source_scope": "tau_bench_adapted",
            "query": "我刚退了一单，会员等级会不会跟着掉？当时是花呗分期付的，这期账单已经出了，退款到底退哪儿？",
            "query_type": ["multi_chunk"],
            "answerable": True,
            "expected_docs": ["会员权益", "常见问题FAQ"],
            "gold_evidence": [
                _evidence(
                    "会员权益",
                    ["并夕夕 会员权益", "五、常见问题", "退款是否会扣减升级所需金额？"],
                    "消费金额从累计中扣除",
                ),
                _evidence(
                    "常见问题FAQ",
                    ["并夕夕 常见问题 FAQ", "三、售后服务", "Q9：退款多久到账？"],
                    "退款抵扣下期账单",
                ),
            ],
            "requires_multi_chunk": True,
            "notes": "参考 τ-bench 同时处理账户状态和退款去向的多目标表达。",
        },
        {
            "id": "tau_style_010",
            "source_scope": "tau_bench_adapted",
            "query": "我登录不上，原来的手机号也早停了，偏偏账户里还冒出一笔不是我买的。现在先做什么，要交哪些东西，多久能处理？",
            "query_type": ["multi_chunk"],
            "answerable": True,
            "expected_docs": ["常见问题FAQ"],
            "gold_evidence": [
                _evidence(
                    "常见问题FAQ",
                    ["并夕夕 常见问题 FAQ", "四、账户与安全", "Q11：忘记密码怎么办？"],
                    "身份证照片+人脸认证申诉",
                ),
                _evidence(
                    "常见问题FAQ",
                    ["并夕夕 常见问题 FAQ", "四、账户与安全", "Q12：账户被盗如何处理？"],
                    "提供报警回执",
                ),
            ],
            "requires_multi_chunk": True,
            "notes": "τ-bench 风格的账户异常复合诉求；需要合并找回账户与盗号处置。",
        },
        {
            "id": "tau_style_011",
            "source_scope": "tau_bench_adapted",
            "query": "我想把号注销了，但里面还有没收货的单、积分和券。是不是得先清掉？要是过几天后悔了还能找回来吗？",
            "query_type": ["paraphrase"],
            "answerable": True,
            "expected_docs": ["常见问题FAQ"],
            "gold_evidence": [
                _evidence(
                    "常见问题FAQ",
                    ["并夕夕 常见问题 FAQ", "四、账户与安全", "Q13：能否注销账户？"],
                    "注销后数据保留 30 天",
                )
            ],
            "requires_multi_chunk": False,
            "notes": "参考 τ-bench 的自然多问句，加入注销前状态和反悔条件。",
        },
        {
            "id": "tau_style_012",
            "source_scope": "tau_bench_adapted",
            "query": "我是钻石会员，东西签收第10天突然降价了，不过现在这个价是限时秒杀。我还能补差价吗？",
            "query_type": ["hard_negative"],
            "answerable": True,
            "expected_docs": ["常见问题FAQ"],
            "gold_evidence": [
                _evidence(
                    "常见问题FAQ",
                    ["并夕夕 常见问题 FAQ", "二、商品与价格", "Q5：商品降价了能否退差价？"],
                    "限时秒杀",
                )
            ],
            "requires_multi_chunk": False,
            "notes": "τ-bench 风格的条件冲突：会员时限满足，但促销类型属于例外。",
        },
        {
            "id": "tau_style_013",
            "source_scope": "tau_bench_adapted",
            "query": "这双鞋我越看越像假的，盒子、吊牌和快递袋我都还留着。接下来在哪儿提交？真鉴定成假的大概怎么赔？",
            "query_type": ["paraphrase"],
            "answerable": True,
            "expected_docs": ["常见问题FAQ"],
            "gold_evidence": [
                _evidence(
                    "常见问题FAQ",
                    ["并夕夕 常见问题 FAQ", "二、商品与价格", "Q7：发现假货怎么办？"],
                    "赔付订单实付金额的 3 倍",
                )
            ],
            "requires_multi_chunk": False,
            "notes": "参考 τ-bench 用户先描述现状再提出多个连续问题的表达。",
        },
        {
            "id": "tau_style_014",
            "source_scope": "tau_bench_adapted",
            "query": "退货我已经寄出去了，结果快递说包裹找不到了。这个是平台赔还是我自己认，找客服前要先跟快递拿什么东西？",
            "query_type": ["paraphrase"],
            "answerable": True,
            "expected_docs": ["常见问题FAQ"],
            "gold_evidence": [
                _evidence(
                    "常见问题FAQ",
                    ["并夕夕 常见问题 FAQ", "三、售后服务", "Q8：寄回商品丢失了怎么办？"],
                    "快递公司出具的丢件证明",
                )
            ],
            "requires_multi_chunk": False,
            "notes": "τ-bench 风格的责任主体与后续动作组合询问。",
        },
        {
            "id": "tau_style_015",
            "source_scope": "tau_bench_adapted",
            "query": "我那台华为电脑在保修期里突然开不了机，本来想走品牌维修，但之前不小心进过水，我朋友还拆开看过。这个是寄你们仓库，还是找华为？大概率还能免费吗？",
            "query_type": ["multi_chunk"],
            "answerable": True,
            "expected_docs": ["常见问题FAQ", "华为硬件保修条款"],
            "gold_evidence": [
                _evidence(
                    "常见问题FAQ",
                    ["并夕夕 常见问题 FAQ", "三、售后服务", "Q10：商品保修期内坏了怎么办？"],
                    "品牌方授权售后处理",
                ),
                _evidence(
                    "华为硬件保修条款",
                    ["华为硬件保修条款（评估语料）", "不承担三包责任的常见情况"],
                    "进液、淋湿",
                ),
            ],
            "requires_multi_chunk": True,
            "notes": "τ-bench 风格的条件反转；保修期、故障原因和服务入口需要联合判断。",
        },
    ]


def build_additional_tau_style_cases() -> list[dict[str, Any]]:
    """新增 31 条与已有 15 条不重复的 τ-bench 风格问题。"""

    scope = "tau_bench_expanded"
    return [
        _case(
            "tau_extra_001",
            scope,
            "我手头只有国外发的信用卡，实在不行快递到了再付也行，你们这两种总有一个能用吧？",
            "hard_negative",
            ["常见问题FAQ"],
            [_evidence("常见问题FAQ", ["并夕夕 常见问题 FAQ", "一、订单与支付", "Q2：支持哪些支付方式？"], "暂不支持货到付款与境外信用卡")],
            False,
            "参考 τ-bench 的备选偏好表达；两个备选支付方式都属于限制项。",
        ),
        _case(
            "tau_extra_002",
            scope,
            "钱已经扣了二十多分钟，订单还是一副没付过的样子。我现在就找客服，还是再等等？真要找的话得截哪些东西？",
            "ambiguous",
            ["常见问题FAQ"],
            [_evidence("常见问题FAQ", ["并夕夕 常见问题 FAQ", "一、订单与支付", "Q4：订单显示「已支付」但平台未识别怎么办？"], "超过 30 分钟仍未同步")],
            False,
            "用临界时间和后续材料构成自然追问，不直接复述政策标题。",
        ),
        _case(
            "tau_extra_003",
            scope,
            "周一买的现货，到今天第四天了还趴着不动。我是继续催，还是干脆把这单退了？",
            "paraphrase",
            ["常见问题FAQ"],
            [_evidence("常见问题FAQ", ["并夕夕 常见问题 FAQ", "一、订单与支付", "Q1：下单后多久发货？"], "超过 72 小时仍未发货")],
            False,
            "参考 τ-bench 的用户目标表达，将时限换成生活化时间描述。",
        ),
        _case(
            "tau_extra_004",
            scope,
            "我买完才发现另一家同款便宜了，不过颜色不一样，而且那边还是限时秒杀。这个差价能给我补吗？",
            "hard_negative",
            ["常见问题FAQ"],
            [_evidence("常见问题FAQ", ["并夕夕 常见问题 FAQ", "二、商品与价格", "Q5：商品降价了能否退差价？"], "限时秒杀")],
            False,
            "多个相似条件同时出现，但店铺、规格和活动类型均影响价保。",
        ),
        _case(
            "tau_extra_005",
            scope,
            "这个东西每次点进去都卖光，我总不能一天刷几十遍吧。有货时能不能主动叫我一声？",
            "paraphrase",
            ["常见问题FAQ"],
            [_evidence("常见问题FAQ", ["并夕夕 常见问题 FAQ", "二、商品与价格", "Q6：商品页缺货怎么办？"], "到货提醒")],
            False,
            "用用户抱怨和隐含诉求表达补货通知需求。",
        ),
        _case(
            "tau_extra_006",
            scope,
            "家里的电视还在保修期，正常用着突然黑屏了，没人摔也没人拆。我要寄回你们仓库，还是直接找牌子的售后？",
            "paraphrase",
            ["常见问题FAQ"],
            [_evidence("常见问题FAQ", ["并夕夕 常见问题 FAQ", "三、售后服务", "Q10：商品保修期内坏了怎么办？"], "无需邮寄到「并夕夕」仓库")],
            False,
            "以真实设备故障场景询问售后入口。",
        ),
        _case(
            "tau_extra_007",
            scope,
            "我今天刚注册，顺便把生日也填了。新人那张券多久不用会没，生日那个月还有没有别的折扣？",
            "paraphrase",
            ["会员权益"],
            [_evidence("会员权益", ["并夕夕 会员权益", "二、各等级核心权益", "2.1 注册会员"], "新人专享 30 元无门槛券")],
            False,
            "同一会员等级内连续询问新人券和生日券。",
        ),
        _case(
            "tau_extra_008",
            scope,
            "我刚升到白银，这个月能拿什么券？以后要退东西叫人上门取，费用是不是能少点？",
            "paraphrase",
            ["会员权益"],
            [_evidence("会员权益", ["并夕夕 会员权益", "二、各等级核心权益", "2.2 白银会员"], "退换货上门取件费减免 50%")],
            False,
            "把优惠券和售后权益放进一次自然咨询。",
        ),
        _case(
            "tau_extra_009",
            scope,
            "黄金会员这个月到底发几张券？我不想排普通客服队伍，有没有快一点的入口？",
            "paraphrase",
            ["会员权益"],
            [_evidence("会员权益", ["并夕夕 会员权益", "二、各等级核心权益", "2.3 黄金会员"], "专属客服直通通道")],
            False,
            "参考 τ-bench 一次处理多个同类偏好的任务表达。",
        ),
        _case(
            "tau_extra_010",
            scope,
            "我去年买了不少东西，最近几个月没怎么下单，怎么等级突然变了？你们到底看哪段时间，又是哪天重新算？",
            "ambiguous",
            ["会员权益"],
            [_evidence("会员权益", ["并夕夕 会员权益", "一、会员等级体系"], "每月 1 号结算")],
            False,
            "用户只描述等级变化，需要召回滚动周期和结算规则。",
        ),
        _case(
            "tau_extra_011",
            scope,
            "我离黄金就差一点，能不能让客服先给我升上去？另外昨天那单要是退了，之前累计的钱会不会也往回扣？",
            "multi_chunk",
            ["会员权益"],
            [
                _evidence("会员权益", ["并夕夕 会员权益", "五、常见问题", "等级能否手动升级？"], "不可以"),
                _evidence("会员权益", ["并夕夕 会员权益", "五、常见问题", "退款是否会扣减升级所需金额？"], "消费金额从累计中扣除"),
            ],
            True,
            "两个会员状态问题分布在不同 FAQ chunk。",
        ),
        _case(
            "tau_extra_012",
            scope,
            "我买了300块东西又晒了图，这次大概攒多少分？结账时能不能一口气全抵掉，攒着不花会不会过期？",
            "paraphrase",
            ["会员权益"],
            [_evidence("会员权益", ["并夕夕 会员权益", "三、积分体系"], "单笔最多抵扣订单金额的 50%")],
            False,
            "把积分获取、抵扣上限和有效期放入同一句。",
        ),
        _case(
            "tau_extra_013",
            scope,
            "我只是黄金会员，生日那周也能领礼盒吗？当天那张88块的券我倒是能拿吧？",
            "hard_negative",
            ["会员权益"],
            [_evidence("会员权益", ["并夕夕 会员权益", "四、生日特权"], "限钻石会员")],
            False,
            "同一生日场景中，两项权益的会员等级限制不同。",
        ),
        _case(
            "tau_extra_014",
            scope,
            "这个会员是不是还得交年费？我想升黄金的话，最近一年至少得买到多少钱？",
            "multi_chunk",
            ["会员权益"],
            [
                _evidence("会员权益", ["并夕夕 会员权益", "五、常见问题", "会员费用怎么算？"], "完全免费"),
                _evidence("会员权益", ["并夕夕 会员权益", "一、会员等级体系"], "黄金会员 | 累计 5,000 元"),
            ],
            True,
            "费用和升级门槛来自两个会员 chunk。",
        ),
        _case(
            "tau_extra_015",
            scope,
            "我要寄到香港，普通下单页面一直不让我填。你们是完全不送，还是得换个入口？",
            "hard_negative",
            ["配送说明"],
            [_evidence("配送说明", ["并夕夕 配送说明", "一、配送范围"], "港澳台及海外暂不直接发货")],
            False,
            "用下单失败场景询问配送边界和替代入口。",
        ),
        _case(
            "tau_extra_016",
            scope,
            "我住省会，普通快递通常几天？页面上那个次日达在省会是真的都能选吗？",
            "paraphrase",
            ["配送说明"],
            [_evidence("配送说明", ["并夕夕 配送说明", "二、配送时效"], "省会城市")],
            False,
            "从地区和服务档位两个维度询问同一配送表格。",
        ),
        _case(
            "tau_extra_017",
            scope,
            "我订的是定做的柜子，客服却拿普通快递三五天那个表跟我算，这个时间能当准吗？",
            "hard_negative",
            ["配送说明"],
            [_evidence("配送说明", ["并夕夕 配送说明", "二、配送时效"], "按详情页公示时效为准")],
            False,
            "普通时效与定制大件的例外条件形成干扰。",
        ),
        _case(
            "tau_extra_018",
            scope,
            "我买了个80块的跑步机送青海，结算页又有基础运费、又有大件费、还提示偏远加价，这几笔分别是怎么来的？",
            "multi_chunk",
            ["配送说明"],
            [
                _evidence("配送说明", ["并夕夕 配送说明", "三、运费规则", "普通包邮规则"], "不满则收取 8 元基础运费"),
                _evidence("配送说明", ["并夕夕 配送说明", "三、运费规则", "大件商品运费"], "按体积重计费"),
                _evidence("配送说明", ["并夕夕 配送说明", "三、运费规则", "偏远地区与会员包邮例外"], "加收 20 元起"),
            ],
            True,
            "金额、商品体积和地区三项约束分别命中三个运费 chunk。",
        ),
        _case(
            "tau_extra_019",
            scope,
            "我这边只有邮政比较方便，下单时能不能别给我随机分，直接指定EMS？",
            "paraphrase",
            ["配送说明"],
            [_evidence("配送说明", ["并夕夕 配送说明", "四、合作快递"], "不支持顾客指定快递公司")],
            False,
            "以用户偏好询问是否能指定承运商。",
        ),
        _case(
            "tau_extra_020",
            scope,
            "我白天一直不在家，快递送不到会再来几次？要是最后丢到柜子里，我最晚什么时候得拿走？",
            "multi_chunk",
            ["配送说明"],
            [
                _evidence("配送说明", ["并夕夕 配送说明", "六、配送异常"], "再尝试投递 1-2 次"),
                _evidence("配送说明", ["并夕夕 配送说明", "五、收货须知"], "48 小时内取件"),
            ],
            True,
            "失败投递和代收点时限来自不同配送 chunk。",
        ),
        _case(
            "tau_extra_021",
            scope,
            "快递小哥就在门口，我看见电脑外箱已经撞瘪了。这种我是先签了再申请，还是现在就别收？",
            "paraphrase",
            ["配送说明"],
            [_evidence("配送说明", ["并夕夕 配送说明", "五、收货须知"], "破损请直接拒收")],
            False,
            "用即时决策场景表达贵重商品验货规则。",
        ),
        _case(
            "tau_extra_022",
            scope,
            "我住县里，明早急着用，今天下单能选当日达或者次日达吗？",
            "hard_negative",
            ["配送说明"],
            [_evidence("配送说明", ["并夕夕 配送说明", "二、配送时效"], "县级及以下")],
            False,
            "用户同时询问两个在县级地区不支持的时效服务。",
        ),
        _case(
            "tau_extra_023",
            scope,
            "东西正好第七天，没用过，盒子配件赠品也一个没少。我今天点退还来得及不？",
            "hard_negative",
            ["退换货政策"],
            [_evidence("退换货政策", ["并夕夕 退换货政策", "一、七天无理由退货", "1.1 适用范围"], "含第 7 天")],
            False,
            "使用边界日期和完整商品状态测试适用范围。",
        ),
        _case(
            "tau_extra_024",
            scope,
            "口红我只拆开试了一下颜色，没觉得合适；内衣也只是剪了封条没穿，这两样能一起退吗？",
            "hard_negative",
            ["退换货政策"],
            [_evidence("退换货政策", ["并夕夕 退换货政策", "一、七天无理由退货", "1.2 不适用商品"], "化妆品、内衣、贴身衣物（已拆封）")],
            False,
            "两个拆封后的健康相关商品共同触发无理由退货例外。",
        ),
        _case(
            "tau_extra_025",
            scope,
            "我是白银，鞋子没毛病就是不想要了，叫上门取件是不是还得从退款里扣钱？白银能给我省多少？",
            "multi_chunk",
            ["会员权益", "退换货政策"],
            [
                _evidence("会员权益", ["并夕夕 会员权益", "二、各等级核心权益", "2.2 白银会员"], "上门取件费减免 50%"),
                _evidence("退换货政策", ["并夕夕 退换货政策", "一、七天无理由退货", "1.3 运费规则"], "运费 12 元起"),
            ],
            True,
            "会员减免与无理由退货基础运费需要跨文档组合。",
        ),
        _case(
            "tau_extra_026",
            scope,
            "收到的耳机功能跟页面写的完全不是一回事，现在第14天。我想退的话时间还赶得上吗，要拍哪些东西，邮费谁出？",
            "multi_chunk",
            ["退换货政策"],
            [
                _evidence("退换货政策", ["并夕夕 退换货政策", "二、质量问题退换货", "2.1 适用情形"], "与详情页描述严重不符"),
                _evidence("退换货政策", ["并夕夕 退换货政策", "二、质量问题退换货", "2.2 时效与运费"], "签收后 15 天内"),
                _evidence("退换货政策", ["并夕夕 退换货政策", "二、质量问题退换货", "2.3 举证材料"], "商品外观照片"),
            ],
            True,
            "质量判定、截止时间、运费和材料分布在三个 chunk。",
        ),
        _case(
            "tau_extra_027",
            scope,
            "我昨晚给一个敏感商品提了退款，今天还没人看。普通东西不是一天内就审吗，这种是不是会慢一点？",
            "ambiguous",
            ["退换货政策"],
            [_evidence("退换货政策", ["并夕夕 退换货政策", "三、退款流程"], "敏感商品 48 小时")],
            False,
            "用等待时长和商品类别隐含指向审核规则。",
        ),
        _case(
            "tau_extra_028",
            scope,
            "包裹到门口我就拒收了，原来是银行卡付的。仓库退回去以后系统多久开始退，银行那边又可能卡几天？",
            "multi_chunk",
            ["退换货政策", "常见问题FAQ"],
            [
                _evidence("退换货政策", ["并夕夕 退换货政策", "四、特殊场景", "4.2 拒收退款"], "时效约 5-7 个工作日"),
                _evidence("常见问题FAQ", ["并夕夕 常见问题 FAQ", "三、售后服务", "Q9：退款多久到账？"], "银行卡")
            ],
            True,
            "拒收处理与支付渠道到账时效来自两个文档。",
        ),
        _case(
            "tau_extra_029",
            scope,
            "货压根没发出来，商家也说不用寄任何东西。我是不是只点退款就行，不用拿个空盒子退回去吧？",
            "hard_negative",
            ["退换货政策"],
            [_evidence("退换货政策", ["并夕夕 退换货政策", "四、特殊场景", "4.1 仅退款（不退货）"], "商品未发货")],
            False,
            "用反问方式表达无需退货的退款场景。",
        ),
        _case(
            "tau_extra_030",
            scope,
            "这台华为是在国外买的，现在人在国内，发票又找不到了，不过机器序列号还在。国内这边能不能按保修接，凭序列号够不够？",
            "multi_chunk",
            ["华为硬件保修条款"],
            [
                _evidence("华为硬件保修条款", ["华为硬件保修条款（评估语料）", "来源与适用范围"], "购买地所在国家或地区"),
                _evidence("华为硬件保修条款", ["华为硬件保修条款（评估语料）", "中国区保修政策"], "主机机身号显示的出厂日期"),
            ],
            True,
            "购买地区与缺失凭证的处理规则需要联合召回。",
        ),
        _case(
            "tau_extra_031",
            scope,
            "昨晚打雷后华为平板就开不了机了，我估计免费修够呛。要去门店的话怎么提前约，官方电话是多少？",
            "multi_chunk",
            ["华为硬件保修条款"],
            [
                _evidence("华为硬件保修条款", ["华为硬件保修条款（评估语料）", "不承担三包责任的常见情况"], "因雷击"),
                _evidence("华为硬件保修条款", ["华为硬件保修条款（评估语料）", "预约维修服务"], "申请预约服务"),
                _evidence("华为硬件保修条款", ["华为硬件保修条款（评估语料）", "客服信息"], "950800"),
            ],
            True,
            "损坏原因、预约方式和客服电话分布在三个相邻 chunk。",
        ),
    ]


def build_payload() -> dict[str, Any]:
    original_cases = build_project_original_cases()
    if len(original_cases) != 4:
        raise ValueError(f"项目原始用例应为 4 条，实际为 {len(original_cases)}")

    tau_style_cases = build_tau_style_cases()
    if len(tau_style_cases) != 15:
        raise ValueError(f"已有 τ-bench 风格用例应为 15 条，实际为 {len(tau_style_cases)}")

    additional_cases = build_additional_tau_style_cases()
    if len(additional_cases) != 31:
        raise ValueError(f"新增 τ-bench 风格用例应为 31 条，实际为 {len(additional_cases)}")

    cases = original_cases + tau_style_cases + additional_cases
    if len(cases) != 50:
        raise ValueError(f"合并后应为 50 条，实际为 {len(cases)}")

    ids = [case["id"] for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("用例 id 存在重复")
    queries = ["".join(case["query"].split()) for case in cases]
    if len(queries) != len(set(queries)):
        raise ValueError("用例 query 存在重复")

    return {
        "dataset": "Ecom RAG chunk-level retrieval evaluation: 4 project-original + 46 tau-bench-style",
        "description": "保留项目最早的 4 条检索问题、已有的 15 条 τ-bench 风格问题，再新增 31 条口语化、多目标和多约束场景。只评估 Retriever，不调用 Agent 循环。",
        "source_policy": "τ-bench 风格问题只借鉴自然表达、条件约束、偏好和多目标结构；所有业务事实与 gold_evidence 均以当前 Ecom rag_knowledge 为准，不照抄 chunk 标题或官方问答口吻。",
        "benchmark_reference": {
            "repository": TAU_BENCH_REPOSITORY,
            "retail_tasks": TAU_BENCH_TASKS,
        },
        "metrics_scope": "answerable-only Recall@1（仅35条单证据查询）、All-query Recall@1（辅助）、Hit@1/3/5/10、Recall@3/5/10",
        "cases": cases,
    }


def main() -> None:
    payload = build_payload()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"已生成 {len(payload['cases'])} 条用例：{OUTPUT_PATH}")


if __name__ == "__main__":
    main()
