import json
from pathlib import Path

import numpy as np

from aistart.rag.models import Chunk


def _normalize(v: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.clip(norms, 1e-12, None)


class VectorStore:
    def __init__(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunk ve vektör sayısı eşleşmiyor")
        self.chunks = chunks
        self.vectors = _normalize(vectors)

    def search(self, query_vec: np.ndarray, k: int = 3) -> list[tuple[Chunk, float]]:
        scores = self.vectors @ _normalize(query_vec.reshape(-1))
        top = np.argsort(scores)[::-1][:k]
        return [(self.chunks[i], float(scores[i])) for i in top]

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "vectors.npy", self.vectors)
        data = [c.model_dump() for c in self.chunks]
        (directory / "chunks.json").write_text(
            json.dumps(data, ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def load(cls, directory: Path) -> "VectorStore":
        vectors = np.load(directory / "vectors.npy")
        raw = json.loads((directory / "chunks.json").read_text(encoding="utf-8"))
        return cls([Chunk(**c) for c in raw], vectors)
