"""评估执行器：编排 沙箱跑用例 → 双层评分 → 聚合报告（第9期）。

Evaluator 自己不碰 Agent，只负责：让 Sandbox 跑出 RunTrace，再用 metrics 对轨迹
打分（过程层 + 结果层，代码规则恒算、LLM judge 受 use_judge 控制），最后聚合。

通过规则分为两层：
1. 硬门槛：必需工具、意图、转人工、路由和事实忠实度不能低于关键指标阈值。
2. 软评分：结果质量、过程质量和 Token 成本按权重计算综合分。
Token 超预算会扣分，但不会单独否决一条用例。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from openai import OpenAI

from app.evaluation import metrics
from app.evaluation.dataset import EvalCase
from app.evaluation.sandbox import Sandbox


@dataclass
class EvalResult:
    """单条用例的评估结果。所有评分维度 None 表示该用例未指定（不计入聚合）。"""

    case_id: str
    description: str

    # ---------- 过程指标 ----------
    tool_accuracy: float | None = None
    business_state_accuracy: float | None = None
    tool_efficiency: float | None = None
    token_cost: int = 0
    token_pass: bool | None = None
    token_score: float | None = None
    process_soundness: float | None = None  # judge 1-5 归一化到 0-1
    route_match: float | None = None

    # ---------- 结果指标 ----------
    intent_match: float | None = None
    keyword_coverage: float | None = None
    requires_human_match: float | None = None
    answer_quality: float | None = None  # judge 1-5 归一化到 0-1
    faithfulness: float | None = None  # 1.0 忠实 / 0.0 幻觉

    # ---------- 汇总 ----------
    process_score: float | None = None
    result_score: float | None = None
    overall_score: float | None = None
    hard_gate_failures: list[str] = field(default_factory=list)
    passed: bool = False

    trace: dict = field(default_factory=dict)  # RunTrace 精简快照
    judge_reasons: dict = field(default_factory=dict)  # judge 文字理由
    error: str | None = None


def _avg(values: list[float | None]) -> float | None:
    """对非 None 项求均值；全为 None 返回 None。"""
    present = [v for v in values if v is not None]
    return sum(present) / len(present) if present else None


def _weighted_avg(values: list[tuple[float | None, float]]) -> float | None:
    """对非 None 项按权重求均值，并对剩余权重重新归一化。"""
    present = [(value, weight) for value, weight in values if value is not None]
    if not present:
        return None
    weight_sum = sum(weight for _, weight in present)
    if weight_sum <= 0:
        return None
    return sum(value * weight for value, weight in present) / weight_sum


class Evaluator:
    """评估器：对一批用例执行沙箱运行 + 双层评分 + 聚合。"""

    def __init__(
        self,
        sandbox: Sandbox,
        client: OpenAI,
        model: str,
        use_judge: bool = True,
        pass_threshold: float = 0.8,
        critical_threshold: float = 0.8,
        result_weight: float = 0.6,
        process_weight: float = 0.35,
        token_weight: float = 0.05,
    ):
        self.sandbox = sandbox
        self.client = client
        self.model = model
        self.use_judge = use_judge
        self.pass_threshold = pass_threshold
        self.critical_threshold = critical_threshold
        self.result_weight = result_weight
        self.process_weight = process_weight
        self.token_weight = token_weight

    def run_case(self, case: EvalCase) -> EvalResult:
        trace = self.sandbox.run(case)
        res = EvalResult(case_id=case.id, description=case.description, trace=trace.to_dict())

        if trace.error:
            res.error = trace.error
            return res

        try:
            resp = trace.final_response
            reply = resp.reply if resp else ""
            actual_intent = resp.intent.value if resp else ""
            actual_human = resp.requires_human if resp else False
            last_input = case.turns[-1]
            called = trace.tool_call_names

            res.token_cost = trace.total_tokens

            # ---------- 过程指标（代码规则）----------
            res.tool_accuracy = metrics.tool_accuracy(case.expected_tools, called)
            res.business_state_accuracy = metrics.business_state_accuracy(
                case.expected_business_checks, trace.tool_observations
            )
            res.tool_efficiency = metrics.tool_efficiency(case.min_tool_calls, trace.num_tool_calls)
            res.token_pass = metrics.token_cost_pass(trace.total_tokens, case.max_tokens)
            res.token_score = metrics.token_cost_score(trace.total_tokens, case.max_tokens)
            res.route_match = metrics.route_match(case.expected_route, trace.route)

            # ---------- 结果指标（代码规则）----------
            res.intent_match = metrics.intent_match(case.expected_intent, actual_intent)
            res.keyword_coverage = metrics.keyword_coverage(case.expected_keywords, reply)
            res.requires_human_match = metrics.requires_human_match(
                case.expected_requires_human, actual_human
            )

            # ---------- LLM judge ----------
            if self.use_judge:
                q_score, q_reason = metrics.judge_answer_quality(
                    self.client, self.model, last_input, reply, case.expected_keywords
                )
                res.answer_quality = q_score / 5.0 if q_score else None
                res.judge_reasons["answer_quality"] = q_reason

                f_score, f_reason = metrics.judge_faithfulness(
                    self.client, self.model, reply, trace.tool_observations
                )
                res.faithfulness = f_score
                res.judge_reasons["faithfulness"] = f_reason

                p_score, p_reason = metrics.judge_process_soundness(
                    self.client, self.model, last_input, called
                )
                res.process_soundness = p_score / 5.0 if p_score else None
                res.judge_reasons["process_soundness"] = p_reason

            # ---------- 汇总 ----------
            res.process_score = _avg([
                res.tool_accuracy, res.tool_efficiency,
                res.business_state_accuracy,
                res.process_soundness, res.route_match,
            ])
            res.result_score = _avg([
                res.intent_match, res.keyword_coverage,
                res.requires_human_match, res.answer_quality, res.faithfulness,
            ])
            res.overall_score = _weighted_avg([
                (res.result_score, self.result_weight),
                (res.process_score, self.process_weight),
                (res.token_score, self.token_weight),
            ])
            res.hard_gate_failures = self._hard_gate_failures(res)
            res.passed = self._decide_pass(res)

        except Exception as e:  # noqa: BLE001 —— 评分异常隔离到单条用例
            res.error = f"评分异常 {type(e).__name__}: {e}"

        return res

    def run_all(self, cases: list[EvalCase]) -> dict:
        results = [self.run_case(c) for c in cases]
        return self._aggregate(results)

    def _decide_pass(self, res: EvalResult) -> bool:
        """硬门槛通过且综合软评分达标，才算通过。"""
        if res.error:
            return False
        if res.hard_gate_failures:
            return False
        return res.overall_score is None or res.overall_score >= self.pass_threshold

    def _hard_gate_failures(self, res: EvalResult) -> list[str]:
        """返回关键指标未通过的名称；None 表示该用例不检查此维度。"""
        critical = {
            "tool_accuracy": res.tool_accuracy,
            "business_state_accuracy": res.business_state_accuracy,
            "intent_match": res.intent_match,
            "requires_human_match": res.requires_human_match,
            "route_match": res.route_match,
            "faithfulness": res.faithfulness,
        }
        return [
            name
            for name, value in critical.items()
            if value is not None and value < self.critical_threshold
        ]

    def _aggregate(self, results: list[EvalResult]) -> dict:
        total = len(results)
        passed = sum(1 for r in results if r.passed)
        total_tokens = sum(r.token_cost for r in results)
        summary = {
            "total": total,
            "passed": passed,
            "pass_rate": passed / total if total else 0.0,
            "task_success_rate": passed / total if total else 0.0,
            "avg_process_score": _avg([r.process_score for r in results]),
            "avg_result_score": _avg([r.result_score for r in results]),
            "total_tokens": total_tokens,
            "avg_tokens_per_case": total_tokens / total if total else 0,
        }
        return {
            "summary": summary,
            "cases": [self._result_to_dict(r) for r in results],
        }

    @staticmethod
    def _result_to_dict(r: EvalResult) -> dict:
        return {
            "case_id": r.case_id,
            "description": r.description,
            "passed": r.passed,
            "process": {
                "tool_accuracy": r.tool_accuracy,
                "business_state_accuracy": r.business_state_accuracy,
                "tool_efficiency": r.tool_efficiency,
                "token_cost": r.token_cost,
                "token_pass": r.token_pass,
                "token_score": r.token_score,
                "process_soundness": r.process_soundness,
                "route_match": r.route_match,
                "process_score": r.process_score,
            },
            "result": {
                "intent_match": r.intent_match,
                "keyword_coverage": r.keyword_coverage,
                "requires_human_match": r.requires_human_match,
                "answer_quality": r.answer_quality,
                "faithfulness": r.faithfulness,
                "result_score": r.result_score,
            },
            "overall_score": r.overall_score,
            "hard_gate_failures": r.hard_gate_failures,
            "judge_reasons": r.judge_reasons,
            "trace": r.trace,
            "error": r.error,
        }
