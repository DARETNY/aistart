from aistart.llm.client import LLMClient
from aistart.logging import setup_logging

setup_logging()
r = LLMClient().complete("RAG nedir? Tek cümleyle.")
print(r.text)
