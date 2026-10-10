import time

import anthropic
import structlog
from pydantic import BaseModel

from aistart.config import settings

log = structlog.get_logger()


class LLMResponse(BaseModel):
    text: str
    input_tokens: int
    output_tokens: int
    latency_s: float
    cost_usd: float


class LLMClient:
    def __init__(self) -> None:
        self._client = anthropic.Anthropic(
            api_key=settings.anthropic_api_key,
            max_retries=settings.max_retries,
            timeout=settings.timeout_s,
        )

    def complete(self, prompt: str, system: str | None = None) -> LLMResponse:
        start = time.perf_counter()
        kwargs = {"system": system} if system else {}
        msg = self._client.messages.create(
            model=settings.model,
            max_tokens=settings.max_tokens,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        latency = time.perf_counter() - start
        cost = (
            msg.usage.input_tokens * settings.price_in_per_mtok
            + msg.usage.output_tokens * settings.price_out_per_mtok
        ) / 1_000_000
        resp = LLMResponse(
            text=msg.content[0].text,
            input_tokens=msg.usage.input_tokens,
            output_tokens=msg.usage.output_tokens,
            latency_s=round(latency, 3),
            cost_usd=round(cost, 6),
        )
        log.info("llm_call", **resp.model_dump(exclude={"text"}))
        return resp
