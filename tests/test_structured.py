from types import SimpleNamespace
from unittest.mock import patch

from pydantic import BaseModel

from aistart.llm.client import LLMClient
from aistart.llm.structured import complete_structured


class Person(BaseModel):
    name: str
    age: int


def resp(content: str):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def test_retries_on_invalid_json_then_succeeds():
    client = LLMClient()
    bad, good = resp('{"name": "Ali"}'), resp('{"name": "Ali", "age": 30}')
    with patch.object(
        client.raw.chat.completions, "create", side_effect=[bad, good]
    ) as m:
        p = complete_structured(client, "Ali 30 yaşında", Person)

    assert p == Person(name="Ali", age=30)
    assert m.call_count == 2
