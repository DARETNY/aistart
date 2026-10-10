from types import SimpleNamespace
from unittest.mock import patch

from aistart.llm.client import LLMClient


def fake_response():
    return SimpleNamespace(
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
        choices=[SimpleNamespace(message=SimpleNamespace(content="merhaba"))],
    )


def test_complete_parses_response():
    client = LLMClient()
    with patch.object(
        client._client.chat.completions, "create", return_value=fake_response()
    ):
        r = client.complete("selam")

    assert r.text == "merhaba"
    assert r.input_tokens == 10
    assert r.output_tokens == 5
    assert r.cost_usd == 0.0
