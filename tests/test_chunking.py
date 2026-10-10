from aistart.rag.chunking import chunk_text


def test_heading_stays_with_content():
    text = "# A\n\nbir iki üç.\n\n# B\n\ndört beş altı."
    chunks = chunk_text(text, "t.md")
    assert chunks[0].text == "# A\n\nbir iki üç."
    assert chunks[1].text.startswith("# B")


def test_split_section_repeats_heading():
    sentences = " ".join(f"Cümle {i} burada." for i in range(60))
    chunks = chunk_text(f"# Başlık\n\n{sentences}", "t.md", max_chars=200, overlap=40)
    assert len(chunks) > 1
    assert all(c.text.startswith("# Başlık") for c in chunks)
    assert all(len(c.text) <= 200 for c in chunks)


def test_no_mid_word_cut():
    text = " ".join(["kelime"] * 500)
    chunks = chunk_text(text, "t.md", max_chars=100, overlap=20)
    assert len(chunks) > 1
    assert all(set(c.text.split()) == {"kelime"} for c in chunks)
