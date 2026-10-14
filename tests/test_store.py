import numpy as np

from aistart.rag.models import Chunk
from aistart.rag.store import VectorStore


def make_chunk(i: int) -> Chunk:
    return Chunk(id=f"d#{i}", source="d", index=i, text=f"metin {i}")


def test_search_returns_closest_first():
    chunks = [make_chunk(i) for i in range(3)]
    vectors = np.array([[1, 0], [0, 1], [0.9, 0.1]], dtype=np.float32)
    store = VectorStore(chunks, vectors)

    results = store.search(np.array([1.0, 0.0]), k=2)

    assert [c.id for c, _ in results] == ["d#0", "d#2"]
    assert results[0][1] > results[1][1]


def test_save_and_load_roundtrip(tmp_path):
    chunks = [make_chunk(0), make_chunk(1)]
    store = VectorStore(chunks, np.array([[1, 0], [0, 1]], dtype=np.float32))
    store.save(tmp_path)

    loaded = VectorStore.load(tmp_path)

    assert [c.id for c in loaded.chunks] == ["d#0", "d#1"]
    assert np.allclose(loaded.vectors, store.vectors)
