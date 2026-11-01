"""记忆查询工具：让 Agent 在 ReAct 循环中主动查询用户记忆。"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.agent.memory.manager import MemoryManager

_memory_manager: MemoryManager | None = None


def set_memory_manager(manager: MemoryManager) -> None:
    """由 Agent 初始化时调用，注入 MemoryManager 引用。"""
    global _memory_manager
    _memory_manager = manager


def recall_user_memory(query: str = "", memory_manager: MemoryManager | None = None) -> dict:
    """查询当前用户的记忆信息（长期记忆和短期记忆）。"""
    manager = memory_manager or _memory_manager
    if manager is None or not manager.memory_enabled:
        return {"success": False, "error": "记忆系统未启用"}

    long_term_facts = manager.recall(query)
    result: dict = {
        "success": True,
        "short_term_facts": manager.stm.facts,
        "long_term_facts": long_term_facts,
        "retrieved_memory_ids": [item["memory_id"] for item in long_term_facts],
        "retrieval": manager.last_retrieval,
    }

    if manager.ltm.interaction_summaries:
        result["recent_interactions"] = [
            s["summary"] for s in manager.ltm.interaction_summaries[-3:]
        ]

    return result
