import time

import structlog
from openai import OpenAI
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
        self._client = OpenAI(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            max_retries=settings.max_retries,
            timeout=settings.timeout_s,
        )

    @property
    def raw(self) -> OpenAI:
        return self._client

    def complete(
        self, prompt: str, system: str | None = None, temperature: float = 0.0
    ) -> LLMResponse:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        start = time.perf_counter()
        r = self._client.chat.completions.create(
            model=settings.model,
            max_tokens=settings.max_tokens,
            temperature=temperature,
            messages=messages,
        )
        latency = time.perf_counter() - start

        inp, out = r.usage.prompt_tokens, r.usage.completion_tokens
        cost = (
            inp * settings.price_in_per_mtok + out * settings.price_out_per_mtok
        ) / 1_000_000
        resp = LLMResponse(
            text=r.choices[0].message.content,
            input_tokens=inp,
            output_tokens=out,
            latency_s=round(latency, 3),
            cost_usd=round(cost, 6),
        )
        log.info("llm_call", **resp.model_dump(exclude={"text"}))
        return resp
