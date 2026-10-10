from pathlib import Path

from aistart.rag.chunking import chunk_text
from aistart.rag.loader import list_documents, load_text
from aistart.rag.models import Chunk


def ingest_dir(directory: Path, max_chars: int = 800, overlap: int = 100) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in list_documents(directory):
        text = load_text(path)
        chunks.extend(chunk_text(text, source=path.name, max_chars=max_chars, overlap=overlap))
    return chunks


if __name__ == "__main__":
    result = ingest_dir(Path("data/docs"))
    print(f"{len(result)} chunk üretildi")
    for c in result[:3]:
        print(f"\n--- {c.id} ({len(c.text)} karakter) ---\n{c.text}")
