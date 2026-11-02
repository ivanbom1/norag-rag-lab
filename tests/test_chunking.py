from app.indexing import ChunkSettings, chunk_text


def paragraphs_in_order(span, chunk):
    position = 0
    for piece in chunk.split("\n\n"):
        piece = piece.strip()
        if not piece:
            continue
        position = span.find(piece, position)
        if position == -1:
            return False
        position += len(piece)
    return True


def test_offsets_are_exact_when_nothing_is_dropped():
    text = "\n\n".join(f"Paragraph number {i} is long enough to keep." for i in range(20))
    for chunk in chunk_text(text, ChunkSettings(chunk_size=100, overlap_percent=10)):
        assert text[chunk["start"]:chunk["end"]] == chunk["text"]


def test_short_paragraphs_are_dropped_but_offsets_still_match():
    text = "A first paragraph that is long enough.\n\nShort\n\nA second paragraph that is long enough."
    chunks = chunk_text(text, ChunkSettings(chunk_size=1000, overlap_percent=0))
    assert len(chunks) == 1
    assert "Short" not in chunks[0]["text"]
    assert chunks[0]["start"] == 0 and chunks[0]["end"] == len(text)


def test_leading_whitespace_and_extra_blank_lines():
    text = "\n\n   Indented paragraph that is long enough.\n\n\n\nAnother paragraph that is long enough.  "
    chunks = chunk_text(text, ChunkSettings(chunk_size=1000, overlap_percent=0))
    assert text[chunks[0]["start"]:].startswith("Indented")
    assert text[:chunks[0]["end"]].endswith("long enough.")
    assert paragraphs_in_order(text[chunks[0]["start"]:chunks[0]["end"]], chunks[0]["text"])


def test_consecutive_chunks_overlap():
    text = "\n\n".join("word " * 30 for _ in range(10))
    settings = ChunkSettings(chunk_size=200, overlap_percent=20)
    chunks = chunk_text(text, settings)
    overlap = settings.chunk_size * settings.overlap_percent // 100
    for previous, current in zip(chunks, chunks[1:]):
        assert previous["text"][-overlap:] == current["text"][:overlap]


def test_corpus_offsets_point_at_chunk_text():
    with open("corpus/monte-cristo-vol1.txt", encoding="utf-8") as f:
        text = f.read()
    for settings in (ChunkSettings(), ChunkSettings(chunk_size=300, overlap_percent=0)):
        for chunk in chunk_text(text, settings):
            assert paragraphs_in_order(text[chunk["start"]:chunk["end"]], chunk["text"])
