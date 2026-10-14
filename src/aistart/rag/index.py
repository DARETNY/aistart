from pathlib import Path

from aistart.llm.client import LLMClient
from aistart.rag.embeddings import embed_texts
from aistart.rag.ingest import ingest_dir
from aistart.rag.store import VectorStore

INDEX_DIR = Path("data/index")


def build_index() -> VectorStore:
    chunks = ingest_dir(Path("data/docs"))
    vectors = embed_texts(LLMClient(), [c.text for c in chunks])
    store = VectorStore(chunks, vectors)
    store.save(INDEX_DIR)
    print(f"{len(chunks)} chunk indekslendi, vektör boyutu: {vectors.shape[1]}")
    return store


if __name__ == "__main__":
    build_index()
