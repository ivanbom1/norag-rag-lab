import bisect
import os
from dataclasses import dataclass

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

CHROMA_PATH = "./chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
BATCH_SIZE = 5000

client = chromadb.PersistentClient(path=CHROMA_PATH)
embedding_function = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)


@dataclass(frozen=True)
class ChunkSettings:
    chunk_size: int = 800
    overlap_percent: int = 10
    min_paragraph_length: int = 20

    @property
    def collection_name(self):
        return f"chunks_{self.chunk_size}_{self.overlap_percent}_{self.min_paragraph_length}_minilm"


def chunk_text(text, settings):
    paragraphs = []
    # (position in joined text, position in original text, length) for each kept paragraph
    segments = []
    joined_position = 0
    original_position = 0
    for raw in text.split("\n\n"):
        paragraph = raw.strip()
        if len(paragraph) >= settings.min_paragraph_length:
            if paragraphs:
                joined_position += 2
            leading_whitespace = len(raw) - len(raw.lstrip())
            segments.append((joined_position, original_position + leading_whitespace, len(paragraph)))
            paragraphs.append(paragraph)
            joined_position += len(paragraph)
        original_position += len(raw) + 2

    if not paragraphs:
        return []

    joined = "\n\n".join(paragraphs)
    segment_starts = [segment[0] for segment in segments]

    def to_original(joined_index, is_end):
        i = bisect.bisect_right(segment_starts, joined_index - is_end) - 1
        segment_joined, segment_original, length = segments[i]
        return segment_original + min(joined_index - segment_joined, length)

    overlap = settings.chunk_size * settings.overlap_percent // 100
    step = settings.chunk_size - overlap
    chunks = []
    for start in range(0, len(joined), step):
        end = min(start + settings.chunk_size, len(joined))
        chunks.append({
            "text": joined[start:end],
            "start": to_original(start, is_end=False),
            "end": to_original(end, is_end=True),
        })
    return chunks


def get_collection(settings):
    return client.get_or_create_collection(name=settings.collection_name, embedding_function=embedding_function)


def index_document(path, settings):
    source = os.path.basename(path)
    with open(path, encoding="utf-8") as f:
        chunks = chunk_text(f.read(), settings)

    collection = get_collection(settings)
    collection.delete(where={"source": source})
    for batch_start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[batch_start:batch_start + BATCH_SIZE]
        collection.add(
            ids=[f"{source}#{batch_start + i}" for i in range(len(batch))],
            documents=[chunk["text"] for chunk in batch],
            metadatas=[
                {"source": source, "chunk_index": batch_start + i, "start": chunk["start"], "end": chunk["end"]}
                for i, chunk in enumerate(batch)
            ],
        )
    return len(chunks)


def delete_document(source):
    for collection in client.list_collections():
        if collection.name.startswith("chunks_"):
            collection.delete(where={"source": source})
