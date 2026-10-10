import re

from aistart.rag.models import Chunk

_HEADING = re.compile(r"^#{1,6}\s+.+$", re.MULTILINE)
_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def _split_sections(text: str) -> list[tuple[str, str]]:
    matches = list(_HEADING.finditer(text))
    if not matches:
        return [("", text.strip())]
    sections = []
    pre = text[: matches[0].start()].strip()
    if pre:
        sections.append(("", pre))
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections.append((m.group().strip(), text[m.end() : end].strip()))
    return sections


def _split_long(piece: str, limit: int) -> list[str]:
    out = []
    for sent in _SENTENCE.split(piece):
        while len(sent) > limit:
            cut = sent.rfind(" ", 0, limit)
            if cut <= 0:
                cut = limit
            out.append(sent[:cut].strip())
            sent = sent[cut:].strip()
        if sent:
            out.append(sent)
    return out


def _pack(pieces: list[str], limit: int, overlap: int) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for piece in pieces:
        add = len(piece) + (2 if current else 0)
        if current and size + add > limit:
            chunks.append("\n\n".join(current))
            carry: list[str] = []
            carried = 0
            for p in reversed(current):
                if carried + len(p) > overlap:
                    break
                carry.insert(0, p)
                carried += len(p) + 2
            current = carry
            size = len("\n\n".join(current))
            if current and size + len(piece) + 2 > limit:
                current, size = [], 0
            add = len(piece) + (2 if current else 0)
        current.append(piece)
        size += add
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def chunk_text(
    text: str, source: str, max_chars: int = 800, overlap: int = 100
) -> list[Chunk]:
    if overlap >= max_chars:
        raise ValueError("overlap, max_chars'tan küçük olmalı")

    texts: list[str] = []
    for heading, body in _split_sections(text):
        if not body:
            continue
        limit = max(max_chars - (len(heading) + 2 if heading else 0), 100)
        pieces: list[str] = []
        for p in (p.strip() for p in body.split("\n\n") if p.strip()):
            pieces.extend(_split_long(p, limit) if len(p) > limit else [p])
        for body_text in _pack(pieces, limit, overlap):
            texts.append(f"{heading}\n\n{body_text}" if heading else body_text)

    return [
        Chunk(id=f"{source}#{i}", source=source, index=i, text=t)
        for i, t in enumerate(texts)
    ]
