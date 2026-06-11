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

    # Weaviate
    WEAVIATE_COLLECTION_NAME: str = os.getenv("WEAVIATE_COLLECTION_NAME", "")

    # LLM Configuration
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "groq")  # ollama | groq | openai | together | vllm
    LLM_MODEL: str = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0"))
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    TOGETHER_API_KEY: str = os.getenv("TOGETHER_API_KEY", "")

    # Tri-backend LLM config (2 vLLM + 1 Groq fallback)
    # fast_gate role: classifier + evaluator (small/fast model)
    FAST_LLM_PROVIDER: str = os.getenv("FAST_LLM_PROVIDER", "groq")
    FAST_LLM_MODEL: str = os.getenv("FAST_LLM_MODEL", "llama-3.1-8b-instant")
    # generation role: Phase 1-2-3 + reflexion (large model)
    GEN_LLM_PROVIDER: str = os.getenv("GEN_LLM_PROVIDER", "")   # empty = use LLM_PROVIDER
    GEN_LLM_MODEL: str = os.getenv("GEN_LLM_MODEL", "")          # empty = use LLM_MODEL
    # vLLM dual-endpoint config
    VLLM_FAST_BASE: str = os.getenv("VLLM_FAST_BASE", "")
    VLLM_GEN_BASE: str = os.getenv("VLLM_GEN_BASE", "")

    # Reflexion settings
    REFLEXION_MAX_RETRIES: int = int(os.getenv("REFLEXION_MAX_RETRIES", "3"))

    # GraphRAG retrieval bounds (scalability)
    RETRIEVER_TOP_K: int = int(os.getenv("RETRIEVER_TOP_K", "20"))
    RETRIEVER_MAX_HOPS: int = int(os.getenv("RETRIEVER_MAX_HOPS", "3"))
    RETRIEVER_MAX_CONTEXT_NODES: int = int(os.getenv("RETRIEVER_MAX_CONTEXT_NODES", "500"))

    class Config:
        case_sensitive = True


configs = Configs()
