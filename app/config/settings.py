from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """项目配置，从 .env 文件读取"""

    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    model_name: str = "gpt-4o-mini"
    temperature: float = 0.7

    # ReAct 循环
    max_react_steps: int = 5

    # MCP 配置
    mcp_enabled: bool = False
    mcp_server_url: str = "http://127.0.0.1:9123/mcp"

    # RAG 配置（第5期）
    embedding_backend: str = "openai"  # openai / local
    embedding_model: str = "text-embedding-3-small"
    embedding_model_path: str = ""
    embedding_device: str = "cpu"
    # 意图分类：复用 RAG 的 Embedding 配置。关闭时仍保留 LLM + Pattern 两路。
    intent_embedding_enabled: bool = True
    intent_llm_weight: float = 0.30
    intent_embedding_weight: float = 0.25
    intent_pattern_weight: float = 0.45
    kb_dir: str = "app/agent/rag/knowledge"
    # 向量后端：numpy（手写余弦，教学透明，零依赖，默认）/ chroma（向量数据库，生产代表，需 pip install chromadb）
    rag_backend: str = "numpy"
    # 线上检索链路：dense / hybrid / hybrid-rerank。
    # hybrid = Dense + BM25 + RRF；hybrid-rerank 会在候选集上再执行 Cross-Encoder 精排。
    rag_pipeline: str = "hybrid-rerank"
    rag_candidate_k: int = 10
    rag_rrf_k: int = 60
    reranker_model_path: str = ""
    reranker_batch_size: int = 8
    # NumpyBackend 的 JSON 索引路径
    kb_index_path: str = "app/sessions/kb_index.json"
    # ChromaBackend 的持久化目录与 collection 名
    chroma_persist_dir: str = "app/sessions/chroma"
    chroma_collection: str = "ecom_kb"

    # Multi-Agent 配置（第6期）
    multi_agent_enabled: bool = False
    # 仅 MultiAgentOrchestrator 模式生效：识别到 >=2 个独立任务才并行调用专家。
    multi_task_enabled: bool = False
    # True 时复用 Multi-Agent 的首轮 Router 调用判断是否并行拆分；False 时完全不判断。
    multi_task_first_turn_enabled: bool = False

    # Memory 配置（第7期）
    memory_enabled: bool = True
    memory_dir: str = "app/sessions/memory"
    memory_user_id: str = "default"
    max_ltm_facts: int = 50

    # Skill 配置（第8期）
    skills_enabled: bool = True
    skills_dir: str = "app/agent/skills/definitions"

    # Evaluation 配置（第9期，离线评估工具，无聊天开关）
    eval_dataset_path: str = "app/evaluation/data/cases.json"
    eval_use_judge: bool = True  # 是否启用 LLM-as-judge（质量/幻觉/过程合理性）
    eval_pass_threshold: float = 0.8  # 综合软评分通过阈值
    eval_critical_threshold: float = 0.8  # 关键硬门槛阈值
    eval_result_weight: float = 0.6  # 结果质量权重
    eval_process_weight: float = 0.35  # 过程质量权重
    eval_token_weight: float = 0.05  # Token 成本权重

    # 多轮对话管理
    session_path: str = "app/sessions/session.json"
    history_threshold: int = 10  # 兼容旧配置；新的 Context 按预算触发
    history_keep_recent: int = 3  # 兼容旧配置；新的 Context 使用自己的安全边界

    # Session-scoped Context 压缩（参考 LCC s08，不属于长期 Memory）
    context_enabled: bool = True
    context_archive_dir: str = "app/sessions/context"
    context_char_limit: int = 50000
    context_target_ratio: float = 0.8
    context_tool_batch_char_limit: int = 20000
    context_large_result_char_limit: int = 6000
    context_summary_input_char_limit: int = 60000
    context_max_messages: int = 50
    context_keep_recent_messages: int = 8
    context_keep_recent_tool_results: int = 3
    context_working_messages: int = 6

    model_config = {"env_file": ".env"}


settings = Settings()
