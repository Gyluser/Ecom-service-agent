"""意图路由器：分析用户消息，决定分发给哪个子 Agent。"""

from typing import List, Optional

from openai import OpenAI

from app.prompts.agents import ROUTER_PROMPT
from app.multi_agent.planner import PlannedTask

VALID_AGENTS = {"presale", "postsale", "complaint"}
DEFAULT_AGENT = "postsale"


class Router:
    """使用 LLM 对用户意图分类，路由到对应的子 Agent。"""

    def __init__(self, client: OpenAI, model: str):
        self.client = client
        self.model = model

    def route(self, user_input: str, history: Optional[List[dict]] = None) -> str:
        """返回子 Agent 标识: "presale" / "postsale" / "complaint"。"""
        recent_context = ""
        if history:
            recent = [
                m for m in history[-4:]
                if m.get("role") in ("user", "assistant")
            ]
            if recent:
                lines = []
                for m in recent:
                    role = "用户" if m["role"] == "user" else "客服"
                    content = m.get("content", "")
                    if content and len(content) < 200:
                        lines.append(f"{role}: {content}")
                if lines:
                    recent_context = "\n最近对话：\n" + "\n".join(lines) + "\n"

        prompt = ROUTER_PROMPT.format(user_input=user_input)
        if recent_context:
            prompt = recent_context + "\n" + prompt

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=10,
        )

        raw = (response.choices[0].message.content or "").strip().lower()

        for agent_key in VALID_AGENTS:
            if agent_key in raw:
                return agent_key

        return DEFAULT_AGENT

    def route_or_plan(self, user_input: str, history: Optional[List[dict]] = None) -> tuple[str | None, list[PlannedTask]]:
        """首轮 Router 同时决定单专家路由或多任务计划，不增加额外 LLM 调用。"""
        prompt = """你是电商客服路由器。输出严格 JSON。
若请求只有一个任务，输出 {"mode":"single","agent":"presale|postsale|complaint","tasks":[]}。
若有至少两个相互独立、可并行的任务，输出 {"mode":"parallel","agent":null,"tasks":[{"agent":"...","query":"..."}]}。
“先查订单再退款”等有依赖的动作必须输出 single；不确定时也输出 single。
专家：presale=商品/库存/价格，postsale=订单/物流/退款，complaint=投诉/转人工。
用户消息：""" + user_input
        try:
            response = self.client.chat.completions.create(
                model=self.model, temperature=0.0, max_tokens=300,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = (response.choices[0].message.content or "").strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            import json
            data = json.loads(raw)
            tasks = [
                PlannedTask(agent=item["agent"], query=item["query"].strip())
                for item in data.get("tasks", [])
                if item.get("agent") in VALID_AGENTS and isinstance(item.get("query"), str) and item["query"].strip()
            ]
            if data.get("mode") == "parallel" and len(tasks) >= 2:
                return None, tasks
            agent = data.get("agent")
            return (agent if agent in VALID_AGENTS else DEFAULT_AGENT), []
        except Exception:
            return self.route(user_input, history), []
