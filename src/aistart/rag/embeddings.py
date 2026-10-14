import numpy as np

from aistart.config import settings
from aistart.llm.client import LLMClient


def embed_texts(
    client: LLMClient, texts: list[str], prefix: str = "", batch_size: int = 32
) -> np.ndarray:
    texts = [prefix + t for t in texts]
    vectors: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        r = client.raw.embeddings.create(model=settings.embed_model, input=batch)
        vectors.extend(d.embedding for d in r.data)
    return np.array(vectors, dtype=np.float32)
