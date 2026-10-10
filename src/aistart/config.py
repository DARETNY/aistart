from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    anthropic_api_key: str
    model: str = "claude-sonnet-5-5"
    max_tokens: int = 1024
    max_retries: int = 3
    timeout_s: float = 30.0
    # $ / 1M token. Güncel fiyatı docs'tan doğrula.
    price_in_per_mtok: float = 3.0
    price_out_per_mtok: float = 15.0


settings = Settings()
