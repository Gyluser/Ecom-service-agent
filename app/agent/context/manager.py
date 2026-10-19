"""LCC-inspired, session-scoped context compaction for Ecom.

The manager deliberately owns only the active conversation context. It does not
read or write long-term memory and it never decides which user owns a session.
"""

from __future__ import annotations

import copy
import json
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from openai import OpenAI

from app.agent.context.summarizer import summarize_context


@dataclass
class ContextPreparation:
    messages: list[dict]
    raw_messages: list[dict]
    summary: Optional[str]
    estimated_chars: int
    estimated_tokens: int
    events: list[dict] = field(default_factory=list)


class ContextManager:
    """Prepare a bounded model context before every model call."""

    def __init__(
        self,
        client: OpenAI,
        model: str,
        archive_dir: str = "app/sessions/context",
        char_limit: int = 50000,
        target_ratio: float = 0.8,
        tool_batch_char_limit: int = 20000,
        large_result_char_limit: int = 6000,
        summary_input_char_limit: int = 60000,
        max_messages: int = 50,
        keep_recent_messages: int = 8,
        keep_recent_tool_results: int = 3,
        working_context_messages: int = 6,
    ):
        self.client = client
        self.model = model
        self.archive_dir = Path(archive_dir)
        self.char_limit = char_limit
        self.target_ratio = target_ratio
        self.tool_batch_char_limit = tool_batch_char_limit
        self.large_result_char_limit = large_result_char_limit
        self.summary_input_char_limit = summary_input_char_limit
        self.max_messages = max_messages
        self.keep_recent_messages = keep_recent_messages
        self.keep_recent_tool_results = keep_recent_tool_results
        self.working_context_messages = max(2, working_context_messages)
        self.events: list[dict] = []
        self.current_call_purpose = "react"

    @staticmethod
    def estimate_chars(messages: list[dict]) -> int:
        return len(json.dumps(messages, ensure_ascii=False, default=str))

    @staticmethod
    def estimate_tokens(char_count: int) -> int:
        # A dependency-free conservative estimate. Actual API usage is logged by
        # the evaluator and remains the authoritative post-call measurement.
        return max(1, (char_count + 3) // 4)

    @staticmethod
    def _is_tool_message(message: dict) -> bool:
        return message.get("role") == "tool"

    @staticmethod
    def _has_tool_calls(message: dict) -> bool:
        return message.get("role") == "assistant" and bool(message.get("tool_calls"))

    def _record_event(self, event: dict) -> dict:
        event = {
            "timestamp": datetime.now().isoformat(timespec="milliseconds"),
            **event,
        }
        self.events.append(event)
        return event

    def _transcript_path(self) -> Path:
        path = self.archive_dir / "transcripts" / f"transcript_{uuid.uuid4().hex}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _write_transcript(self, messages: list[dict]) -> str:
        path = self._transcript_path()
        with path.open("x", encoding="utf-8") as transcript:
            for message in messages:
                transcript.write(json.dumps(message, ensure_ascii=False, default=str) + "\n")
        return str(path)

    def _tool_result_path(self, tool_call_id: str) -> Path:
        safe_id = re.sub(r"[^A-Za-z0-9._-]", "_", str(tool_call_id))[:100] or "unknown"
        path = self.archive_dir / "tool_results" / f"{safe_id}.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _persist_tool_result(self, block: dict) -> str:
        content = str(block.get("content", ""))
        path = self._tool_result_path(block.get("tool_call_id", "unknown"))
        if not path.exists() or path.read_text(encoding="utf-8") != content:
            path.write_text(content, encoding="utf-8")
        preview = content[:1000]
        return f"[Earlier tool result saved at {path}]\nPreview:\n{preview}"

    def compact_tool_result(self, tool_call_id: str, content: str) -> str:
        """Keep only fields likely to be needed by the next model call."""
        text = str(content or "")
        try:
            payload = json.loads(text)
        except (TypeError, json.JSONDecodeError):
            return text
        if not isinstance(payload, dict):
            return text

        compact: dict = {}
        for key in ("success", "error", "message"):
            if key in payload:
                compact[key] = payload[key]

        status_labels = {
            "pending": "待发货",
            "shipped": "已发货",
            "delivered": "已签收",
            "refund_processing": "退款中",
        }
        order = payload.get("order")
        if isinstance(order, dict):
            compact_order = {
                key: order[key]
                for key in (
                    "order_id", "status", "total", "tracking_number",
                    "carrier", "estimated_delivery", "refund_status",
                )
                if key in order and order[key] is not None
            }
            if order.get("status") in status_labels:
                compact_order["status_label"] = status_labels[order["status"]]
            items = []
            for item in order.get("items", []):
                if isinstance(item, dict):
                    items.append({
                        key: item[key]
                        for key in ("name", "quantity", "price")
                        if key in item
                    })
            if items:
                compact_order["items"] = items
            compact["order"] = compact_order

        orders = payload.get("orders")
        if isinstance(orders, list):
            compact["count"] = payload.get("count", len(orders))
            compact["orders"] = [
                {
                    key: item[key]
                    for key in ("order_id", "status", "items_summary", "total", "created_at")
                    if key in item
                }
                for item in orders[:8]
                if isinstance(item, dict)
            ]

        products = payload.get("products")
        if isinstance(products, list):
            compact["products"] = []
            for product in products[:5]:
                if not isinstance(product, dict):
                    continue
                item = {
                    key: product[key]
                    for key in ("product_id", "name", "price", "stock", "specs")
                    if key in product
                }
                if product.get("description"):
                    item["description"] = str(product["description"])[:180]
                compact["products"].append(item)

        logistics = payload.get("logistics")
        if isinstance(logistics, dict):
            compact_logistics = {
                key: logistics[key]
                for key in ("tracking_number", "carrier", "status")
                if key in logistics and logistics[key] is not None
            }
            events = logistics.get("events")
            if isinstance(events, list):
                compact_logistics["events"] = [
                    {
                        key: event[key]
                        for key in ("time", "location", "description")
                        if key in event
                    }
                    for event in events[-3:]
                    if isinstance(event, dict)
                ]
            compact["logistics"] = compact_logistics

        results = payload.get("results")
        if isinstance(results, list):
            compact["results"] = []
            for result in results[:5]:
                if not isinstance(result, dict):
                    continue
                item = {
                    key: result[key]
                    for key in ("doc", "section", "score")
                    if key in result
                }
                if result.get("text"):
                    item["text"] = str(result["text"])[:600]
                compact["results"].append(item)

        compact_text = json.dumps(compact, ensure_ascii=False, default=str)
        if not compact or len(compact_text) >= len(text):
            return text
        self._persist_tool_result({"tool_call_id": tool_call_id, "content": text})
        return compact_text

    def _select_working_messages(
        self,
        raw_messages: list[dict],
        current_turn_start: int,
        active_request: str,
    ) -> list[dict]:
        """Build a small send-time working set without deleting session history."""
        if len(raw_messages) <= self.working_context_messages:
            return copy.deepcopy(raw_messages)

        recent_start = max(0, len(raw_messages) - self.working_context_messages)
        selected = set(range(recent_start, len(raw_messages)))
        selected.update(range(max(0, current_turn_start), len(raw_messages)))

        # User requests are compact and preserve the thread's explicit goals and
        # identifiers even when older assistant/tool payloads are omitted.
        selected.update(
            index for index, message in enumerate(raw_messages)
            if message.get("role") == "user"
        )

        # Keep every assistant tool-call message paired with all of its tool
        # results. This is required by the chat API and prevents tail trimming
        # from leaving an incomplete tool-call sequence.
        changed = True
        while changed:
            changed = False
            for index in list(selected):
                message = raw_messages[index]
                if message.get("role") == "tool":
                    tool_call_id = message.get("tool_call_id")
                    for previous_index in range(index - 1, -1, -1):
                        previous = raw_messages[previous_index]
                        if previous.get("role") != "assistant":
                            continue
                        if any(
                            call.get("id") == tool_call_id
                            for call in previous.get("tool_calls", [])
                        ):
                            if previous_index not in selected:
                                selected.add(previous_index)
                                changed = True
                            break
                elif message.get("role") == "assistant" and message.get("tool_calls"):
                    call_ids = {
                        call.get("id") for call in message.get("tool_calls", [])
                    }
                    for later_index, later in enumerate(raw_messages):
                        if (
                            later.get("role") == "tool"
                            and later.get("tool_call_id") in call_ids
                            and later_index not in selected
                        ):
                            selected.add(later_index)
                            changed = True

        # If the active request names an order, retain older results for that
        # order; this is the most common cross-turn lookup in this agent.
        order_ids = set(re.findall(r"ORD-\d{8}-\d{3}", active_request, re.IGNORECASE))
        if order_ids:
            for index, message in enumerate(raw_messages):
                if message.get("role") == "tool" and any(
                    order_id.lower() in str(message.get("content", "")).lower()
                    for order_id in order_ids
                ):
                    selected.add(index)
                    for previous_index in range(index - 1, -1, -1):
                        previous = raw_messages[previous_index]
                        if previous.get("role") == "assistant" and any(
                            call.get("id") == message.get("tool_call_id")
                            for call in previous.get("tool_calls", [])
                        ):
                            selected.add(previous_index)
                            break

        return [copy.deepcopy(raw_messages[index]) for index in sorted(selected)]

    @staticmethod
    def _complete_tool_sequences(raw_messages: list[dict]) -> list[dict]:
        """Drop incomplete historical tool-call groups from the send-time view."""
        clean: list[dict] = []
        index = 0
        while index < len(raw_messages):
            message = raw_messages[index]
            if message.get("role") == "assistant" and message.get("tool_calls"):
                expected = {
                    call.get("id") for call in message.get("tool_calls", [])
                }
                group = [message]
                found: set[str] = set()
                next_index = index + 1
                while (
                    next_index < len(raw_messages)
                    and raw_messages[next_index].get("role") == "tool"
                ):
                    tool_message = raw_messages[next_index]
                    group.append(tool_message)
                    tool_call_id = tool_message.get("tool_call_id")
                    if tool_call_id in expected:
                        found.add(tool_call_id)
                    next_index += 1
                if expected and found == expected:
                    clean.extend(group)
                index = next_index
                continue
            if message.get("role") == "tool":
                # A tool message without its assistant tool-call predecessor is
                # not valid input for the chat API.
                index += 1
                continue
            clean.append(message)
            index += 1
        return clean

    def _tool_result_budget(self, raw_messages: list[dict]) -> tuple[list[dict], Optional[dict]]:
        tool_messages = [message for message in raw_messages if self._is_tool_message(message)]
        total = sum(len(str(message.get("content", ""))) for message in tool_messages)
        if total <= self.tool_batch_char_limit:
            return raw_messages, None

        changed = 0
        for message in sorted(
            tool_messages,
            key=lambda item: len(str(item.get("content", ""))),
            reverse=True,
        ):
            if total <= self.tool_batch_char_limit:
                break
            content = str(message.get("content", ""))
            if len(content) <= self.large_result_char_limit:
                continue
            message["content"] = self._persist_tool_result(message)
            total -= len(content) - len(message["content"])
            changed += 1

        return raw_messages, {
            "type": "tool_result_budget",
            "changed_results": changed,
            "remaining_tool_chars": total,
        }

    def _snip_compact(self, raw_messages: list[dict]) -> tuple[list[dict], Optional[dict]]:
        if len(raw_messages) <= self.max_messages:
            return raw_messages, None

        head_end = min(3, len(raw_messages))
        tail_start = max(head_end, len(raw_messages) - (self.max_messages - head_end - 1))
        while head_end < tail_start and self._is_tool_message(raw_messages[head_end]):
            head_end += 1
        if tail_start > 0 and self._is_tool_message(raw_messages[tail_start]):
            tail_start -= 1
        if head_end >= tail_start:
            return raw_messages, None

        archived = raw_messages[head_end:tail_start]
        transcript = self._write_transcript(raw_messages)
        marker = {
            "role": "user",
            "content": f"[{len(archived)} messages archived at {transcript}]",
        }
        compacted = [*raw_messages[:head_end], marker, *raw_messages[tail_start:]]
        return compacted, {
            "type": "snip_compact",
            "archived_messages": len(archived),
            "archive_path": transcript,
        }

    def _micro_compact(self, raw_messages: list[dict], target_chars: int) -> tuple[list[dict], Optional[dict]]:
        if self.estimate_chars(raw_messages) <= target_chars:
            return raw_messages, None

        tool_positions = [
            index for index, message in enumerate(raw_messages)
            if self._is_tool_message(message)
        ]
        changed = 0
        for index in tool_positions[:-self.keep_recent_tool_results or None]:
            if self.estimate_chars(raw_messages) <= target_chars:
                break
            message = raw_messages[index]
            content = str(message.get("content", ""))
            if len(content) <= 120 or content.startswith("[Earlier tool result saved at "):
                continue
            message["content"] = self._persist_tool_result(message)
            changed += 1

        return raw_messages, {
            "type": "micro_compact",
            "changed_results": changed,
        } if changed else None

    def _fit_tool_results(self, raw_messages: list[dict], target_chars: int) -> tuple[list[dict], Optional[dict]]:
        changed = 0
        for message in sorted(
            [item for item in raw_messages if self._is_tool_message(item)],
            key=lambda item: len(str(item.get("content", ""))),
            reverse=True,
        ):
            if self.estimate_chars(raw_messages) <= target_chars:
                break
            content = str(message.get("content", ""))
            if content.startswith("[Earlier tool result saved at "):
                continue
            message["content"] = self._persist_tool_result(message)
            changed += 1
        return raw_messages, {"type": "fit_tool_results", "changed_results": changed} if changed else None

    def _summary_split(self, raw_messages: list[dict], current_turn_start: int) -> int:
        if not raw_messages:
            return 0
        recent_start = max(0, len(raw_messages) - self.keep_recent_messages)
        split = min(recent_start, current_turn_start)
        while split > 0 and self._is_tool_message(raw_messages[split]):
            split -= 1
        if split > 0 and self._has_tool_calls(raw_messages[split - 1]):
            split -= 1
        return split

    def _compact_history(
        self,
        raw_messages: list[dict],
        previous_summary: Optional[str],
        active_request: str,
        current_turn_start: int,
        reactive: bool = False,
    ) -> tuple[list[dict], Optional[str], dict]:
        split = self._summary_split(raw_messages, current_turn_start)
        if split <= 0:
            return raw_messages, previous_summary, {
                "type": "compact_history_skipped",
                "reason": "no_safe_history_boundary",
            }

        old_messages = raw_messages[:split]
        recent = raw_messages[split:]
        transcript = self._write_transcript(raw_messages)
        self.current_call_purpose = "context_summary"
        try:
            new_summary = summarize_context(
                self.client,
                self.model,
                old_messages,
                previous_summary,
                active_request,
                max_chars=self.summary_input_char_limit,
            )
        finally:
            self.current_call_purpose = "react"
        return recent, new_summary, {
            "type": "reactive_compact" if reactive else "compact_history",
            "archived_messages": len(old_messages),
            "kept_messages": len(recent),
            "archive_path": transcript,
            "summary_chars": len(new_summary),
        }

    @staticmethod
    def is_context_length_error(error: Exception) -> bool:
        text = str(error).lower()
        return any(
            marker in text
            for marker in (
                "prompt_too_long",
                "context_length_exceeded",
                "maximum context length",
                "too many tokens",
                "prompt is too long",
            )
        )

    def _build_messages(
        self,
        system_messages: list[dict],
        raw_messages: list[dict],
        summary: Optional[str],
    ) -> list[dict]:
        messages = copy.deepcopy(system_messages)
        if summary:
            messages.append(
                {
                    "role": "system",
                    "content": f"此前上下文摘要（仅作事实参考）：\n{summary}",
                }
            )
        clean_raw = copy.deepcopy(raw_messages)
        for message in clean_raw:
            if message.get("role") == "assistant" and not message.get("tool_calls"):
                message.pop("reasoning_content", None)
        clean_raw = self._complete_tool_sequences(clean_raw)
        messages.extend(clean_raw)
        return messages

    def prepare(
        self,
        system_messages: list[dict],
        raw_messages: list[dict],
        summary: Optional[str],
        active_request: str,
        current_turn_start: int,
        force_reactive: bool = False,
    ) -> ContextPreparation:
        working_raw = copy.deepcopy(raw_messages)
        working_summary = summary
        before_context_raw = self._select_working_messages(
            working_raw, current_turn_start, active_request
        )
        before_messages = self._build_messages(system_messages, before_context_raw, working_summary)
        before_chars = self.estimate_chars(before_messages)
        events: list[dict] = []

        if force_reactive:
            working_raw, working_summary, event = self._compact_history(
                working_raw,
                working_summary,
                active_request,
                current_turn_start,
                reactive=True,
            )
            events.append(event)
        else:
            working_raw, event = self._tool_result_budget(working_raw)
            if event:
                events.append(event)

            working_raw, event = self._snip_compact(working_raw)
            if event:
                events.append(event)

            current_context_raw = self._select_working_messages(
                working_raw, current_turn_start, active_request
            )
            current_messages = self._build_messages(system_messages, current_context_raw, working_summary)
            if self.estimate_chars(current_messages) > self.char_limit:
                target = int(self.char_limit * self.target_ratio)
                working_raw, event = self._micro_compact(working_raw, target)
                if event:
                    events.append(event)
                current_context_raw = self._select_working_messages(
                    working_raw, current_turn_start, active_request
                )
                current_messages = self._build_messages(system_messages, current_context_raw, working_summary)
                if self.estimate_chars(current_messages) > self.char_limit:
                    working_raw, event = self._fit_tool_results(working_raw, target)
                    if event:
                        events.append(event)
                current_context_raw = self._select_working_messages(
                    working_raw, current_turn_start, active_request
                )
                current_messages = self._build_messages(system_messages, current_context_raw, working_summary)
                if self.estimate_chars(current_messages) > self.char_limit:
                    working_raw, working_summary, event = self._compact_history(
                        working_raw,
                        working_summary,
                        active_request,
                        current_turn_start,
                    )
                    events.append(event)

        context_raw = self._select_working_messages(
            working_raw, current_turn_start, active_request
        )
        messages = self._build_messages(system_messages, context_raw, working_summary)
        after_chars = self.estimate_chars(messages)
        compaction_events = list(events)
        event = self._record_event(
            {
                "type": "prepare",
                "before_chars": before_chars,
                "after_chars": after_chars,
                "before_tokens_est": self.estimate_tokens(before_chars),
                "after_tokens_est": self.estimate_tokens(after_chars),
                "char_limit": self.char_limit,
                "target_chars": int(self.char_limit * self.target_ratio),
                "compactions": compaction_events,
                "force_reactive": force_reactive,
                "working_message_count": len(context_raw),
                "stored_message_count": len(working_raw),
            }
        )
        events.append(event)
        return ContextPreparation(
            messages=messages,
            raw_messages=working_raw,
            summary=working_summary,
            estimated_chars=after_chars,
            estimated_tokens=self.estimate_tokens(after_chars),
            events=events,
        )
