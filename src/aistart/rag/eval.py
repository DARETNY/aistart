from aistart.config import settings
from aistart.llm.client import LLMClient
from aistart.rag.embeddings import embed_texts
from aistart.rag.index import INDEX_DIR
from aistart.rag.store import VectorStore

DOC = "izin_politikasi.md"
CASES = [
    ("Kaç gün izin hakkım var?", f"{DOC}#0"),
    ("İzin talebini ne zaman bildirmeliyim?", f"{DOC}#0"),
    ("Altı yıldır çalışıyorum, kaç gün iznim olur?", f"{DOC}#0"),
    ("Haftada kaç gün evden çalışabilirim?", f"{DOC}#1"),
    ("Yeni başlayanlar evden çalışabilir mi?", f"{DOC}#1"),
    ("Masraf formunu ne zaman teslim etmeliyim?", f"{DOC}#2"),
    ("Şehir dışı görevde otel masrafını kim öder?", f"{DOC}#2"),
]


def run() -> float:
    store = VectorStore.load(INDEX_DIR)
    qvecs = embed_texts(
        LLMClient(), [q for q, _ in CASES], prefix=settings.embed_query_prefix
    )
    hits = 0
    for (question, expected), vec in zip(CASES, qvecs):
        top_chunk, score = store.search(vec, k=1)[0]
        ok = top_chunk.id == expected
        hits += ok
        mark = "OK  " if ok else "FAIL"
        print(f"{mark} [{score:.3f}] {question} -> {top_chunk.id} (beklenen {expected})")
    acc = hits / len(CASES)
    print(f"\nhit@1: {hits}/{len(CASES)} = {acc:.0%}")
    return acc


if __name__ == "__main__":
    run()
