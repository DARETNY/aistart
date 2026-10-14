import sys

from aistart.llm.client import LLMClient
from aistart.rag.embeddings import embed_texts
from aistart.rag.index import INDEX_DIR
from aistart.rag.store import VectorStore


def search(query: str, k: int = 3):
    store = VectorStore.load(INDEX_DIR)
    qvec = embed_texts(LLMClient(), [query])[0]
    return store.search(qvec, k)


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "Kaç gün izin hakkım var?"
    for chunk, score in search(q):
        print(f"\n[{score:.3f}] {chunk.id}\n{chunk.text[:200]}...")
