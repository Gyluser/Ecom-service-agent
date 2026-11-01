"""Offline regression tests for the local CLI user/session storage layout."""

from __future__ import annotations

import tempfile
import unittest
import json
from datetime import datetime
from pathlib import Path

from app.agent.memory.long_term import LongTermMemory, MemoryFact
from app.agent.session_identity import build_local_session_context
from app.agent.storage import load_session, save_session
from app.agent.tools.manager import ToolManager


class LocalSessionIsolationTests(unittest.TestCase):
    def test_memory_tool_uses_its_own_agent_binding(self):
        class FakeMemoryManager:
            memory_enabled = True
            last_retrieval = []

            def __init__(self, fact: str):
                self.stm = type("STM", (), {"facts": [fact]})()
                self.ltm = type("LTM", (), {"interaction_summaries": []})()

            def recall(self, query: str):
                return []

        alice_tools = ToolManager()
        bob_tools = ToolManager()
        alice_tools.set_memory_manager(FakeMemoryManager("alice 的短期记忆"))
        bob_tools.set_memory_manager(FakeMemoryManager("bob 的短期记忆"))

        alice = json.loads(alice_tools.execute_tool("recall_user_memory", {}))
        bob = json.loads(bob_tools.execute_tool("recall_user_memory", {}))
        self.assertEqual(alice["short_term_facts"], ["alice 的短期记忆"])
        self.assertEqual(bob["short_term_facts"], ["bob 的短期记忆"])

    def test_user_memory_and_session_state_are_isolated(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            alice_first = build_local_session_context("alice", "20261027_100000_001-a1b2c3d4", root)
            alice_second = build_local_session_context("alice", "20261027_100100_002-b1c2d3e4", root)
            bob_first = build_local_session_context("bob", "20261027_100200_003-c1d2e3f4", root)

            self.assertEqual(alice_first.memory_dir, alice_second.memory_dir)
            self.assertNotEqual(alice_first.session_path, alice_second.session_path)
            self.assertNotEqual(alice_first.user_dir, bob_first.user_dir)

            memory = LongTermMemory(user_id="alice", memory_dir=str(alice_first.memory_dir))
            memory.add_facts([MemoryFact(
                content="用户偏好蓝色商品",
                category="preference",
                created_at=datetime.now().isoformat(timespec="seconds"),
                source_session=alice_first.session_id,
            )])
            memory.save()

            alice_reloaded = LongTermMemory(user_id="alice", memory_dir=str(alice_second.memory_dir))
            alice_reloaded.load()
            bob_reloaded = LongTermMemory(user_id="bob", memory_dir=str(bob_first.memory_dir))
            bob_reloaded.load()
            self.assertEqual([fact.content for fact in alice_reloaded.facts], ["用户偏好蓝色商品"])
            self.assertEqual(bob_reloaded.facts, [])

            save_session(str(alice_first.session_path), [{"role": "user", "content": "第一会话"}], None)
            save_session(str(alice_second.session_path), [{"role": "user", "content": "第二会话"}], None)
            self.assertEqual(load_session(str(alice_first.session_path))["messages"][0]["content"], "第一会话")
            self.assertEqual(load_session(str(alice_second.session_path))["messages"][0]["content"], "第二会话")


if __name__ == "__main__":
    unittest.main()
