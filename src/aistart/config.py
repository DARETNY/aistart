from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    llm_base_url: str = "http://localhost:11434/v1"
    llm_api_key: str = "ollama"
    model: str = "llama3.2"
    embed_model: str = "nomic-embed-text"
    embed_doc_prefix: str = ""
    embed_query_prefix: str = ""
    max_tokens: int = 1024
    max_retries: int = 3
    timeout_s: float = 60.0
    price_in_per_mtok: float = 0.0
    price_out_per_mtok: float = 0.0


settings = Settings()
