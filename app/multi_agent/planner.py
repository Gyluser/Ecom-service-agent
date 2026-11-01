"""多任务 Planner：仅识别可独立并行的客服任务，不执行任何工具。"""

from __future__ import annotations

import json
from dataclasses import dataclass

from openai import OpenAI


@dataclass(frozen=True)
class PlannedTask:
    agent: str
    query: str


class MultiTaskPlanner:
    """把一句话拆为可并行的售前/售后/投诉子任务。"""

    VALID_AGENTS = {"presale", "postsale", "complaint"}

    def __init__(self, client: OpenAI, model: str):
        self.client = client
        self.model = model

    def plan(self, user_input: str) -> list[PlannedTask]:
        prompt = """你是电商客服任务规划器。仅当用户一句话中包含两个或更多相互独立、
可以同时处理的任务时拆分；同一订单的“先查订单再退款”有依赖，不能拆分。

可用专家：presale（商品/库存/价格）、postsale（订单/物流/退款）、
complaint（投诉/转人工）。只输出 JSON：{"tasks":[{"agent":"...","query":"..."}]}。
不能安全拆分时输出 {"tasks":[]}。

用户消息：""" + user_input
        try:
            response = self.client.chat.completions.create(
                model=self.model, temperature=0.0, max_tokens=300,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = (response.choices[0].message.content or "").strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            data = json.loads(raw)
            seen: set[tuple[str, str]] = set()
            tasks: list[PlannedTask] = []
            for item in data.get("tasks", []):
                agent, query = item.get("agent"), item.get("query", "").strip()
                key = (agent, query)
                if agent in self.VALID_AGENTS and query and key not in seen:
                    tasks.append(PlannedTask(agent=agent, query=query))
                    seen.add(key)
            return tasks
        except (json.JSONDecodeError, AttributeError, TypeError):
            return []
