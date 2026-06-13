import os
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()


class Configs(BaseSettings):

    PROJECT_ROOT: str = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    PROJECT_NAME: str = os.getenv("PROJECT_NAME", "Code Bot Reviewer")

    # data dir
    DATA_DIR: str = os.getenv(
        "DATA_DIR",
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"),
    )

    # date
    DATETIME_FORMAT: str = "%Y-%m-%dT%H:%M:%S"
    DATE_FORMAT: str = "%Y-%m-%d"

    # Neo4j (Required)
    APP_NEO4J_URL: str = os.getenv("APP_NEO4J_URL", "bolt://localhost:7687")
    APP_NEO4J_USER: str = os.getenv("APP_NEO4J_USER", "neo4j")
    APP_NEO4J_PASSWORD: str = os.getenv("APP_NEO4J_PASSWORD", "")
    APP_NEO4J_DATABASE: str = os.getenv("APP_NEO4J_DATABASE", "neo4j")

    def validate_neo4j_config(self) -> None:
        """Validate that required Neo4j configuration is present."""
        if not self.APP_NEO4J_PASSWORD:
            raise ValueError(
                "APP_NEO4J_PASSWORD environment variable is required. "
                "Please set it in your .env file or environment."
            )
        if not self.APP_NEO4J_URL:
            raise ValueError("APP_NEO4J_URL environment variable is required.")
        if not self.APP_NEO4J_USER:
            raise ValueError("APP_NEO4J_USER environment variable is required.")

    NEO4J_MAX_CONNECTION_LIFETIME: int = int(os.getenv("NEO4J_MAX_CONNECTION_LIFETIME", "30"))
    NEO4J_MAX_CONNECTION_POOL_SIZE: int = int(os.getenv("NEO4J_MAX_CONNECTION_POOL_SIZE", "50"))
    NEO4J_CONNECTION_TIMEOUT: float = float(os.getenv("NEO4J_CONNECTION_TIMEOUT", "30.0"))

    # GitHub
    GITHUB_TOKEN: str = os.getenv("GITHUB_TOKEN", "")

    # Embedding model for vector search (sentence-transformers, stored in Neo4j)
    EMBEDDING_ENABLED: bool = os.getenv("EMBEDDING_ENABLED", "true").lower() == "true"
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    EMBEDDING_VECTOR_DIM: int = int(os.getenv("EMBEDDING_VECTOR_DIM", "384"))

    # LLM Configuration
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "groq")  # ollama | groq | openai | together | vllm
    LLM_MODEL: str = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0"))
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    TOGETHER_API_KEY: str = os.getenv("TOGETHER_API_KEY", "")

    # Tri-role LLM config
    # fast_gate: classifier + evaluator (small/fast model — binary decisions only)
    FAST_LLM_PROVIDER: str = os.getenv("FAST_LLM_PROVIDER", "groq")
    FAST_LLM_MODEL: str = os.getenv("FAST_LLM_MODEL", "llama-3.1-8b-instant")
    # chat: agent loop + tool routing (needs strong reasoning; defaults to fast_gate config)
    CHAT_LLM_PROVIDER: str = os.getenv("CHAT_LLM_PROVIDER", "")  # empty = use FAST_LLM_PROVIDER
    CHAT_LLM_MODEL: str = os.getenv("CHAT_LLM_MODEL", "")         # empty = use FAST_LLM_MODEL
    # generation: Phase 3 patch writing (code-specialized model)
    GEN_LLM_PROVIDER: str = os.getenv("GEN_LLM_PROVIDER", "")   # empty = use LLM_PROVIDER
    GEN_LLM_MODEL: str = os.getenv("GEN_LLM_MODEL", "")          # empty = use LLM_MODEL
    # reason: Phase 2 fault analysis + Planner + Reflexion (reasoning-specialized model)
    # Recommended: deepseek-r1-distill-llama-70b on Groq (free, chain-of-thought trained)
    REASON_LLM_PROVIDER: str = os.getenv("REASON_LLM_PROVIDER", "")  # empty = fallback to GEN
    REASON_LLM_MODEL: str = os.getenv("REASON_LLM_MODEL", "")         # empty = fallback to GEN
    # vLLM endpoint config
    VLLM_FAST_BASE: str = os.getenv("VLLM_FAST_BASE", "")
    VLLM_CHAT_BASE: str = os.getenv("VLLM_CHAT_BASE", "")
    VLLM_GEN_BASE: str = os.getenv("VLLM_GEN_BASE", "")
    VLLM_REASON_BASE: str = os.getenv("VLLM_REASON_BASE", "")  # empty = shares VLLM_GEN_BASE
    VLLM_FALLBACK_TO_GROQ: bool = os.getenv("VLLM_FALLBACK_TO_GROQ", "true").lower() == "true"

    # Langfuse observability (opt-in — disabled by default)
    LANGFUSE_ENABLED: bool = os.getenv("LANGFUSE_ENABLED", "false").lower() == "true"
    LANGFUSE_PUBLIC_KEY: str = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    LANGFUSE_SECRET_KEY: str = os.getenv("LANGFUSE_SECRET_KEY", "")
    LANGFUSE_HOST: str = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

    # Execution-based verification gates
    EXECUTION_GATE_ENABLED: bool = os.getenv("EXECUTION_GATE_ENABLED", "true").lower() == "true"
    EXECUTION_GATE_TIMEOUT: int = int(os.getenv("EXECUTION_GATE_TIMEOUT", "300"))
    EXECUTION_GATE_TEST_CMD: str = os.getenv("EXECUTION_GATE_TEST_CMD", "python -m pytest --tb=short -q")

    # GitHub integration — post review results back to the PR
    GITHUB_POST_REVIEW_COMMENTS: bool = os.getenv("GITHUB_POST_REVIEW_COMMENTS", "false").lower() == "true"
    GITHUB_AUTO_FIX_PR: bool = os.getenv("GITHUB_AUTO_FIX_PR", "false").lower() == "true"

    # Meta-review: overall PR assessment after all files are reviewed
    # Uses META_REVIEW_LLM_PROVIDER/MODEL if set; falls back to reason LLM (free, Groq)
    # Set META_REVIEW_LLM_PROVIDER=openai + META_REVIEW_LLM_MODEL=gpt-4o to use GPT-4
    META_REVIEW_ENABLED: bool = os.getenv("META_REVIEW_ENABLED", "true").lower() == "true"
    META_REVIEW_LLM_PROVIDER: str = os.getenv("META_REVIEW_LLM_PROVIDER", "")
    META_REVIEW_LLM_MODEL: str = os.getenv("META_REVIEW_LLM_MODEL", "")

    # Reflexion settings
    REFLEXION_MAX_RETRIES: int = int(os.getenv("REFLEXION_MAX_RETRIES", "3"))

    # GraphRAG retrieval bounds (scalability)
    RETRIEVER_TOP_K: int = int(os.getenv("RETRIEVER_TOP_K", "20"))
    RETRIEVER_MAX_HOPS: int = int(os.getenv("RETRIEVER_MAX_HOPS", "3"))
    RETRIEVER_MAX_CONTEXT_NODES: int = int(os.getenv("RETRIEVER_MAX_CONTEXT_NODES", "500"))

    class Config:
        case_sensitive = True


configs = Configs()
