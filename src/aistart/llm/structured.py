import json
from typing import TypeVar

import structlog
from pydantic import BaseModel, ValidationError

from aistart.config import settings
from aistart.llm.client import LLMClient

log = structlog.get_logger()
T = TypeVar("T", bound=BaseModel)


def complete_structured(
    client: LLMClient,
    prompt: str,
    schema: type[T],
    system: str = "",
    max_attempts: int = 3,
) -> T:
    schema_json = json.dumps(schema.model_json_schema(), ensure_ascii=False)
    messages = [
        {
            "role": "system",
            "content": f"{system}\nSadece bu JSON şemasına uyan geçerli JSON döndür:\n{schema_json}",
        },
        {"role": "user", "content": prompt},
    ]
    for attempt in range(1, max_attempts + 1):
        r = client.raw.chat.completions.create(
            model=settings.model,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0,
        )
        raw = r.choices[0].message.content
        try:
            return schema.model_validate_json(raw)
        except ValidationError as e:
            log.warning("structured_retry", attempt=attempt, error=str(e)[:200])
            messages += [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": f"Geçersiz çıktı: {e}\nDüzelt, sadece JSON döndür."},
            ]
    raise ValueError(f"{max_attempts} denemede geçerli çıktı alınamadı")
