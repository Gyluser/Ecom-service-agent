import json
import re
from typing import Optional

from openai import OpenAI

from app.agent.context import ContextManager
from app.agent.intent_fusion import IntentDecision, IntentFusionClassifier
from app.agent.rag.embedder import Embedder
from app.agent.storage import delete_session, load_session, save_session
from app.config.settings import settings
from app.prompts.customer_service import SYSTEM_PROMPT
from app.schemas.response import CustomerServiceResponse, IntentType
from app.agent.tools.manager import ToolManager


class EcomAgent:
    """电商客服 Agent —— 第八期：Skill 可复用能力模块"""

    def __init__(
        self,
        session_path: Optional[str] = None,
        user_id: Optional[str] = None,
        memory_dir: Optional[str] = None,
        context_archive_dir: Optional[str] = None,
        session_id: Optional[str] = None,
    ):
        self.client = OpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
        )
        self.model = settings.model_name
        self.temperature = settings.temperature
        self.user_id = user_id or settings.memory_user_id
        self.session_id = session_id
        self.session_path = session_path or settings.session_path
        self.history_threshold = settings.history_threshold
        self.history_keep_recent = settings.history_keep_recent
        self.max_react_steps = settings.max_react_steps
        self.context_manager = ContextManager(
            client=self.client,
            model=self.model,
            archive_dir=context_archive_dir or settings.context_archive_dir,
            char_limit=settings.context_char_limit,
            target_ratio=settings.context_target_ratio,
            tool_batch_char_limit=settings.context_tool_batch_char_limit,
            large_result_char_limit=settings.context_large_result_char_limit,
            summary_input_char_limit=settings.context_summary_input_char_limit,
            max_messages=settings.context_max_messages,
            keep_recent_messages=settings.context_keep_recent_messages,
            keep_recent_tool_results=settings.context_keep_recent_tool_results,
            working_context_messages=settings.context_working_messages,
        ) if settings.context_enabled else None

        self.tool_manager = ToolManager(
            use_mcp=settings.mcp_enabled,
            mcp_server_url=settings.mcp_server_url,
        )

        from app.agent.memory import MemoryManager
        self.memory_manager = MemoryManager(
            client=self.client,
            model=self.model,
            user_id=self.user_id,
            memory_dir=memory_dir or settings.memory_dir,
            memory_enabled=settings.memory_enabled,
            max_ltm_facts=settings.max_ltm_facts,
        )

        if settings.memory_enabled:
            self.tool_manager.set_memory_manager(self.memory_manager)

        from app.agent.skills import SkillManager
        self.skill_manager = SkillManager(
            skills_dir=settings.skills_dir,
            enabled=settings.skills_enabled,
        )
        if settings.skills_enabled:
            from app.agent.tools.skill_tool import set_skill_manager
            set_skill_manager(self.skill_manager)

        self.raw_messages: list[dict] = []
        self.summary: Optional[str] = None
        self._current_turn_called_tools: set[str] = set()
        self._current_turn_extra_tools: set[str] = set()
        self._current_turn_tools_blocked = False
        self._current_turn_tool_results: dict[tuple[str, str], str] = {}
        self._current_turn_start = 0
        self._active_request = ""
        # 对外的 intent 表示用户最终目标；该字段只记录本轮最近一次工具动作，
        # 不写入 CustomerServiceResponse，避免把“查订单”误当成“退款”等最终意图。
        self._current_turn_action: Optional[str] = None
        self.intent_classifier = self._build_intent_classifier()

        loaded = load_session(self.session_path)
        if loaded:
            self.summary = loaded["summary"]
            self.raw_messages = loaded["messages"]
            self._current_turn_start = len(self.raw_messages)
            if loaded.get("short_term_memory"):
                self.memory_manager.restore_stm(loaded["short_term_memory"])

    def _build_intent_classifier(self) -> IntentFusionClassifier:
        """构建三路意图分类器；Embedding 故障会在分类器内安全降级。"""
        encoder = None
        if settings.intent_embedding_enabled:
            try:
                encoder = Embedder(
                    api_key=settings.openai_api_key,
                    base_url=settings.openai_base_url,
                    model=settings.embedding_model,
                    backend=settings.embedding_backend,
                    model_path=settings.embedding_model_path,
                    device=settings.embedding_device,
                )
            except Exception:
                encoder = None
        return IntentFusionClassifier(
            encoder=encoder,
            llm_weight=settings.intent_llm_weight,
            embedding_weight=settings.intent_embedding_weight,
            pattern_weight=settings.intent_pattern_weight,
        )

    @property
    def history_size(self) -> int:
        return len(self.raw_messages)

    def chat(self, user_input: str) -> CustomerServiceResponse:
        """处理用户输入：ReAct 循环 → 结构化提取 → 返回结果"""
        self._current_turn_called_tools = set()
        self._current_turn_extra_tools = set()
        self._current_turn_tools_blocked = False
        self._current_turn_tool_results = {}
        self._current_turn_action = None
        self._current_turn_start = len(self.raw_messages)
        self._active_request = user_input
        self.raw_messages.append({"role": "user", "content": user_input})

        final_text = self._react_loop()

        result = self._parse_final_response(final_text)
        self.memory_manager.update_short_term(self.raw_messages[-6:])

        self.raw_messages.append(
            {"role": "assistant", "content": result.model_dump_json(ensure_ascii=False)}
        )

        # 保留旧测试/调用方显式覆盖 history_threshold 的兼容行为；正常生产
        # 路径不再在 chat 结束后按消息条数压缩，而是在每次模型调用前按预算压缩。
        if (
            len(self.raw_messages) > self.history_threshold
            and (
                self.context_manager is None
                or self.history_threshold != settings.history_threshold
            )
        ):
            self._compress_history()

        save_session(
            self.session_path, self.raw_messages, self.summary,
            short_term_memory=self.memory_manager.stm_to_dict(),
        )
        return result

    def reset(self):
        self.raw_messages = []
        self.summary = None
        self.memory_manager.reset_short_term()
        delete_session(self.session_path)

    def save(self) -> None:
        save_session(
            self.session_path, self.raw_messages, self.summary,
            short_term_memory=self.memory_manager.stm_to_dict(),
        )

    def close(self):
        self.memory_manager.consolidate_to_long_term(self.raw_messages, self.summary)
        self.tool_manager.close()

    def _react_loop(self) -> str:
        """ReAct 循环：调用 LLM → 执行工具 → 观察结果 → 重复，直到模型给出最终回答。"""
        for step in range(self.max_react_steps):
            messages = self._prepare_context_messages()
            tool_definitions = (
                []
                if self._current_turn_tools_blocked
                else self._tool_definitions_for_current_turn()
            )
            request_kwargs = {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
            }
            if tool_definitions:
                request_kwargs["tools"] = tool_definitions
            else:
                messages.append({
                    "role": "system",
                    "content": "当前没有可用工具，请直接根据已有信息回答，不要输出任何工具调用标记。",
                })
            response = self._create_with_context_retry(request_kwargs)
            choice = response.choices[0]
            assistant_msg = choice.message

            if assistant_msg.content:
                self._print_thought(assistant_msg.content)

            if not assistant_msg.tool_calls:
                content = assistant_msg.content or ""
                assistant_record = {"role": "assistant", "content": content}
                self.raw_messages.append(assistant_record)
                if self._contains_tool_marker(content):
                    self._current_turn_tools_blocked = True
                    continue
                return content

            allowed_names = {
                definition["function"]["name"]
                for definition in tool_definitions
            }
            executable_tool_calls = [
                tc for tc in assistant_msg.tool_calls
                if tc.function.name in allowed_names
            ]
            if not executable_tool_calls:
                assistant_record = {
                    "role": "assistant",
                    "content": assistant_msg.content or "",
                }
                self.raw_messages.append(assistant_record)
                self._current_turn_tools_blocked = True
                continue

            msg_dict = {"role": "assistant", "content": assistant_msg.content}
            reasoning_content = getattr(assistant_msg, "reasoning_content", None)
            if reasoning_content:
                # thinking 模式要求工具调用的 assistant 消息把该字段传回下一次请求。
                # ContextManager 只会在非工具 assistant 消息上移除它。
                msg_dict["reasoning_content"] = reasoning_content
            msg_dict["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in executable_tool_calls
            ]
            self.raw_messages.append(msg_dict)

            for tc in executable_tool_calls:
                func_name = tc.function.name
                func_args_json = tc.function.arguments
                func_args = json.loads(func_args_json)
                call_key = (func_name, func_args_json)
                self._current_turn_action = func_name

                self._print_action(func_name, func_args)
                if call_key in self._current_turn_tool_results:
                    # 只缓存完全相同的工具调用；不同参数或不同工具仍允许继续执行。
                    result_str = self._current_turn_tool_results[call_key]
                else:
                    result_str = self.tool_manager.execute_tool(func_name, func_args)
                    self._current_turn_tool_results[call_key] = result_str
                self._print_observation(result_str)
                self._current_turn_called_tools.add(func_name)
                try:
                    tool_result = json.loads(result_str)
                except (TypeError, json.JSONDecodeError):
                    tool_result = {}
                if (
                    func_name == "query_order"
                    and tool_result.get("success") is False
                ):
                    self._current_turn_extra_tools.add("list_user_orders")

                context_result = (
                    self.context_manager.compact_tool_result(tc.id, result_str)
                    if self.context_manager is not None
                    else result_str
                )
                self.raw_messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": context_result,
                })

        messages = self._prepare_context_messages()
        response = self._create_with_context_retry({
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        })
        content = response.choices[0].message.content or ""
        assistant_record = {"role": "assistant", "content": content}
        self.raw_messages.append(assistant_record)
        return content

    @staticmethod
    def _contains_tool_marker(content: str) -> bool:
        return any(marker in content for marker in (
            "DSML", "<tool_use", "<tool_result", "<｜｜",
        ))

    def _tool_definitions_for_current_turn(self) -> list[dict]:
        """按当前请求缩小工具集合，但保留多意图和多步骤任务的工具链。"""
        user_text = "\n".join(
            message.get("content", "")
            for message in self.raw_messages
            if message.get("role") == "user"
        )[-1200:]
        text = user_text.lower()
        order_id = re.search(r"ord-\d{8}-\d{3}", text, re.IGNORECASE)
        candidates: set[str] = set()

        greeting_words = ("你好", "您好", "嗨", "hello", "hi", "在吗")
        complaint_words = ("投诉", "消协", "举报", "赔偿", "欺骗", "垃圾")
        business_words = (
            "物流", "快递", "运单", "到哪", "配送", "退货", "退款", "换货",
            "退换", "商品", "库存", "价格", "多少钱", "规格", "有货", "订单",
            "下单", "买过", "政策", "规则", "流程", "会员", "包邮", "时效",
            "记得", "偏好", "会员信息", "以前",
        )

        # 纯问候/投诉不需要查询工具；如果同一条消息还包含业务诉求，
        # 则继续走下面的业务路由，避免把必要的查询工具一并屏蔽。
        pure_social = any(word in text for word in greeting_words + complaint_words)
        has_business_request = any(word in text for word in business_words)
        if pure_social and not has_business_request:
            candidates = set()
        else:
            has_logistics = any(
                word in text for word in ("物流", "快递", "运单", "到哪", "配送", "配送轨迹")
            )
            has_refund = any(word in text for word in ("退货", "退款", "换货", "退换"))
            has_product = any(
                word in text for word in ("商品", "库存", "价格", "多少钱", "规格", "有货")
            )
            has_order = any(word in text for word in ("订单", "下单", "买过"))
            has_policy = any(
                word in text for word in (
                    "政策", "规则", "流程", "会员", "包邮", "时效", "支持",
                    "能不能", "无理由", "七天", "faq",
                )
            )
            has_memory = any(word in text for word in ("记得", "偏好", "会员信息", "以前"))

            if has_logistics:
                candidates.add("query_logistics" if order_id else "list_user_orders")

            if has_refund:
                confirmed = any(
                    word in text for word in ("确认", "同意", "直接退", "马上退")
                )
                # 退款进度/退款状态属于查询，不应因为没有“确认退款”而没有工具。
                if order_id:
                    candidates.add("query_order")
                elif not confirmed:
                    candidates.add("list_user_orders")
                if confirmed and order_id:
                    candidates.add("apply_refund")
                if has_policy:
                    candidates.add("search_knowledge")

            if has_product:
                candidates.add("query_product")

            if has_order:
                candidates.add("query_order" if order_id else "list_user_orders")

            if has_policy:
                candidates.add("search_knowledge")

            if has_memory:
                candidates.add("recall_user_memory")

            if not candidates and not pure_social:
                candidates = {
                    definition["function"]["name"]
                    for definition in self.tool_manager.tool_definitions
                }

        candidates.update(self._current_turn_extra_tools)
        uncalled = candidates - self._current_turn_called_tools
        if self._current_turn_called_tools:
            candidates = uncalled
        return [
            definition
            for definition in self.tool_manager.tool_definitions
            if definition["function"]["name"] in candidates
        ]

    def _parse_final_response(self, text: str) -> CustomerServiceResponse:
        """直接解析最终回复，避免再发起一次结构化提取模型调用。"""
        data = self._extract_json_object(text)
        if data is not None:
            raw_intent = data.get("intent")
            model_intent = self._normalize_intent(raw_intent) if raw_intent else None
            decision = self._resolve_primary_intent(
                self._current_user_text(), model_intent, data.get("confidence")
            )
            data["intent"] = decision.intent.value
            data["confidence"] = decision.confidence
            try:
                return CustomerServiceResponse.model_validate(data)
            except Exception:
                pass
        return self._local_response_fallback(text, data)

    def _current_user_text(self) -> str:
        """提取最近用户消息，供最终意图判断使用，不把工具动作当成用户目标。"""
        return "\n".join(
            message.get("content", "")
            for message in self.raw_messages
            if message.get("role") == "user"
        )[-1200:]

    def _resolve_primary_intent(
        self,
        user_text: str,
        model_intent: str | None,
        model_confidence: object | None = None,
    ) -> IntentDecision:
        """用 LLM、Embedding、Pattern 三路分数输出用户最终目标。"""
        normalized = self._normalize_intent(model_intent) if model_intent else ""
        try:
            llm_intent = IntentType(normalized)
        except ValueError:
            llm_intent = None
        return self.intent_classifier.classify(user_text, llm_intent, model_confidence)

    @staticmethod
    def _extract_json_object(text: str) -> dict | None:
        """从纯 JSON、代码块或带少量前后缀的回复中提取对象。"""
        raw = text.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        candidates = [raw]
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            candidates.append(raw[start:end + 1])
        for candidate in candidates:
            try:
                data = json.loads(candidate)
            except (TypeError, json.JSONDecodeError):
                continue
            if isinstance(data, dict):
                return data
        return None

    @staticmethod
    def _normalize_intent(value: object) -> str:
        """兼容模型偶尔返回的中文意图标签。"""
        if not isinstance(value, str):
            return IntentType.OTHER.value
        mapping = {
            "订单查询": IntentType.ORDER_QUERY.value,
            "退货退款": IntentType.RETURN_REQUEST.value,
            "退货退款咨询": IntentType.RETURN_REQUEST.value,
            "退换货": IntentType.RETURN_REQUEST.value,
            "商品咨询": IntentType.PRODUCT_CONSULT.value,
            "投诉": IntentType.COMPLAINT.value,
            "售后": IntentType.AFTER_SALE.value,
            "售后服务": IntentType.AFTER_SALE.value,
            "优惠活动": IntentType.PROMOTION.value,
            "账户问题": IntentType.ACCOUNT.value,
            "问候": IntentType.GREETING.value,
            "其他": IntentType.OTHER.value,
        }
        return mapping.get(value.strip(), value.strip())

    def _local_response_fallback(
        self, text: str, data: dict | None = None,
    ) -> CustomerServiceResponse:
        """模型未遵守 JSON 格式时，用原文和用户输入做无额外请求的兜底。"""
        user_text = self._current_user_text()
        intent = IntentType.OTHER
        if data:
            normalized = self._normalize_intent(data.get("intent"))
            try:
                intent = IntentType(normalized)
            except ValueError:
                pass
        requires_human_hint = any(
            word in user_text
            for word in ("转人工", "人工客服", "人工处理", "超出能力", "无法处理")
        )
        intent = self._resolve_primary_intent(user_text, intent.value).intent
        if (
            intent == IntentType.OTHER
            and any(word in user_text for word in ("你好", "您好", "嗨", "hello", "hi", "在吗"))
        ):
            intent = IntentType.GREETING
        elif intent == IntentType.OTHER and any(
            word in user_text for word in ("售后", "维修", "保修", "质量问题")
        ):
            intent = IntentType.AFTER_SALE
        elif intent == IntentType.OTHER and any(
            word in user_text for word in ("商品", "库存", "价格", "多少钱", "规格", "有货")
        ):
            intent = IntentType.PRODUCT_CONSULT
        elif intent == IntentType.OTHER and any(
            word in user_text for word in ("物流", "快递", "订单", "下单", "买过", "运单", "到哪")
        ):
            intent = IntentType.ORDER_QUERY

        return CustomerServiceResponse(
            intent=intent,
            confidence=float(data.get("confidence", 0.6)) if data else 0.6,
            reply=(data.get("reply") if data and isinstance(data.get("reply"), str) else text),
            requires_human=(
                bool(data.get("requires_human"))
                if data and "requires_human" in data
                else requires_human_hint or intent == IntentType.COMPLAINT
            ),
            follow_up_question=(
                data.get("follow_up_question")
                if data and isinstance(data.get("follow_up_question"), str)
                else None
            ),
        )

    def _build_messages(self) -> list[dict]:
        messages = self._build_static_messages()
        if self.summary:
            messages.append(
                {
                    "role": "system",
                    "content": f"以下是此前对话的摘要，用于延续上下文记忆：\n{self.summary}",
                }
            )
        messages.extend(self.raw_messages)
        return messages

    def _build_static_messages(self) -> list[dict]:
        system_content = SYSTEM_PROMPT
        if self.skill_manager and self.skill_manager.enabled:
            system_content += self.skill_manager.build_catalog_prompt()

        messages: list[dict] = [
            {"role": "system", "content": system_content}
        ]
        messages.extend(
            self.memory_manager.build_memory_prompt_sections(
                query=self._active_request,
            )
        )
        return messages

    def _prepare_context_messages(self, force_reactive: bool = False) -> list[dict]:
        if self.context_manager is None:
            return self._build_messages()

        preparation = self.context_manager.prepare(
            system_messages=self._build_static_messages(),
            raw_messages=self.raw_messages,
            summary=self.summary,
            active_request=self._active_request,
            current_turn_start=self._current_turn_start,
            force_reactive=force_reactive,
        )
        self.raw_messages = preparation.raw_messages
        self.summary = preparation.summary

        # 压缩保留当前轮，因此重新定位当前轮起点，避免下一次压缩把当前请求
        # 当成旧历史归档。
        current_indices = [
            index for index, message in enumerate(self.raw_messages)
            if message.get("role") == "user"
            and message.get("content") == self._active_request
        ]
        if current_indices:
            self._current_turn_start = current_indices[-1]
        else:
            self._current_turn_start = max(0, len(self.raw_messages) - 1)
        return preparation.messages

    def _create_with_context_retry(self, request_kwargs: dict) -> object:
        retry_count = 0
        while True:
            try:
                return self.client.chat.completions.create(**request_kwargs)
            except Exception as exc:
                if (
                    self.context_manager is None
                    or not self.context_manager.is_context_length_error(exc)
                    or retry_count >= 1
                ):
                    raise
                retry_count += 1
                request_kwargs["messages"] = self._prepare_context_messages(
                    force_reactive=True
                )

    def _compress_history(self) -> None:
        """兼容旧调用方；实际压缩由每次模型调用前的 ContextManager 完成。"""
        if self.context_manager is None:
            return
        previous_keep = self.context_manager.keep_recent_messages
        self.context_manager.keep_recent_messages = max(2, self.history_keep_recent * 2)
        try:
            self._prepare_context_messages(force_reactive=True)
        finally:
            self.context_manager.keep_recent_messages = previous_keep

    def _print_thought(self, text: str) -> None:
        print(f"\n💭 [思考] {text}")

    def _print_action(self, func_name: str, func_args: dict) -> None:
        args_str = ", ".join(f"{k}={v!r}" for k, v in func_args.items())
        print(f"🔧 [调用工具] {func_name}({args_str})")

    def _print_observation(self, result: str) -> None:
        display = result if len(result) <= 300 else result[:300] + "..."
        print(f"📋 [工具结果] {display}")
