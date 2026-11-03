# norag-rag project

This workspace contains 2 Python applications to compare **NoRAG** (full-context prompting) versus **RAG** (retrieval-augmented generation) using **Google Gemini** via the **OpenAI SDK**.

## Applications

1. `ask_norag_google.py`
- NoRAG with Google Gemini via OpenAI SDK (`gemini-3.5-flash-lite`).
- Loads `docs/CrimeAndPunishment.txt`, injects a "needle" passage, and runs two direct large-context questions:
	- a baseline question about Marmeladov (present in the original book)
	- a needle question about Denis (the injected passage)

2. `ask_rag_google.py`
- RAG with Google Gemini for generation via OpenAI SDK.
- Uses local embeddings (`all-MiniLM-L6-v2`) + local ChromaDB (`chroma_db/`) to retrieve relevant chunks before asking the model.
- Default retrieval uses `n_results=5` for the question: `What does this say about Denis?`

## Project Structure

- `docs/CrimeAndPunishment.txt`: main source text used by the scripts.
- `docs/CrimeAndPunishment-shorter.txt`: shortened version of the main source text for quicker testing and rapid experimentation in Gemini online at https://gemini.google.com/app (See note below)
- `chroma_db/`: persistent local vector database (used by RAG scripts).
- `requirements.txt`: Python dependencies.

## Prerequisites

- Python 3.10+ (recommended)
- API key:
   - `GOOGLE_API_KEY` from [Google AI Studio](https://aistudio.google.com/app/apikey)


## Setup

1. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Configure environment variables:

```bash
cp .env.example .env
```

Then edit `.env` and set your real keys.

## Run Each Application

With your virtual environment activated:

### 1) NoRAG + Google

```bash
python ask_norag_google.py
```

### 2) RAG + Google

```bash
python ask_rag_google.py
```

## Notes

- Both applications use the OpenAI SDK to interact with Google Gemini via Google's OpenAI-compatible endpoint.
- The RAG app splits the text into fixed-size chunks with an overlap (`CHUNK_SIZE`, `OVERLAP_PERCENT` and `MIN_PARAGRAPH_LENGTH` at the top of the script), indexes them into `chroma_db/` and reuses them across runs.
- There is one index per document (`DOCUMENT_FILE`) and chunking configuration. When no index exists for the current document and settings, the script asks before building one (this takes a little while).
- Debug prompt printing can be enabled by passing `debug=True` in the NoRAG app.
- RAG app prints retrieved context for verification.
- Streaming output is enabled by default in both applications (controlled by `STREAM_ON` variable).

- Using the shortened version of the text (`docs/CrimeAndPunishment-shorter.txt`)
	- Go online at https://gemini.google.com/app
	- At the prompt, start writing "Here is a story: ",
	- Then paste the content of docs/CrimeAndPunishment-shorter.txt into the Gemini chat app and then, at the end of the prompt, ask "What can you tell me about Denis Amselemovitch?"
	- Observe the response from Gemini.

# Free books from Project Gutenberg
- https://www.gutenberg.org/
# Crime and Punishment by Fyodor Dostoevsky
- https://www.gutenberg.org/ebooks/2554
# Huckleberry Finn by Mark Twain
- https://www.gutenberg.org/ebooks/76

---

# V1 RAG web app (Monte Cristo)

A web app that answers questions about *The Count of Monte Cristo* with verified citations. It runs on FastAPI, ChromaDB, all-MiniLM-L6-v2 and Gemini.

## What was added

| Path | Purpose |
|---|---|
| `scripts/build_corpus.py` | Downloads Gutenberg #1184, cleans it, splits it into 5 volume files and plants test facts |
| `corpus/` | The 5 volume files; they are copied to `data/documents/` the first time the app runs |
| `app/indexing.py` | Chunking with character offsets back into the source file; one Chroma collection per chunk setting |
| `app/answering.py` | Retrieval, LLM call with a JSON schema, Pydantic validation and citation checks |
| `app/main.py` | API for documents, settings and questions; also serves the UI |
| `app/static/` | Plain HTML/JS frontend |
| `eval/` | 30-question set, evaluation script and the V1 baseline result |
| `tests/` | pytest tests for the chunk offsets |

## Run

```bash
pip install -r requirements.txt      # .env must contain GOOGLE_API_KEY
uvicorn app.main:app --reload        # http://localhost:8000, API docs at /docs
```

The first start indexes the corpus, which takes about 2 minutes.

## Test

```bash
python -m pytest tests                                   # offset tests, no LLM
python -m eval.evaluate                                  # retrieval hit@k and MRR, no LLM
python -m eval.evaluate --answers --no-context --delay 4 # + answer accuracy and an LLM-only comparison (~60 Gemini calls)
python -m eval.evaluate --chunk-size 400 --overlap 10    # compare another chunk setting
```

Manual checks in the UI:

- **Ask** *What jewel was in the purse that saved Morrel from ruin?*
  - It should answer **ruby**. The corpus was edited to say ruby, while the original book says diamond, so a ruby answer shows the app reads the corpus instead of relying on the model's memory.
  - Click `[C1]`, then **Show in document**: the cited text is highlighted.
- **Ask** *Who directed the 2002 film adaptation?* It should reply "Not found in the selected documents."
- **Restrict documents:** uncheck vol2 and vol3 and ask the purse question again. The answer should be "not found".
- **Upload a document:** upload a small `.md` file with an invented fact, ask about it, then delete it and ask again.
- **Change settings:** a top-k change applies instantly. A new chunk size re-indexes everything, which takes about 2 minutes.

## Baseline (800 chars, 10 % overlap, top-5)

These results are in `eval/results/chunks_800_10_20_minilm_k5.json`.

- **Retrieval:** hit@5 **0.44–0.48** and MRR ≈ 0.36. The range comes from repeated runs on the same index (see the last decision below). All 3 split questions fail retrieval.
- **Answers:** accuracy **0.60**. All 5 unanswerable questions return `not_found`.
- **Model without the documents:** Gemini, asked with no retrieved text, matched the corpus on 15 of the 25 answerable questions. It said "diamond" and "28 February" where the corpus says ruby and the 25th.

## Decisions

| Topic | Decision |
|---|---|
| Corpus | Monte Cristo, split into one file per volume, with planted facts so a grounded answer can be told apart from the model's memory. The plants are: new facts, one fact split across two volumes, and two deliberate contradictions with the original book (ruby instead of diamond, 25th instead of 24th of February). |
| Stack | FastAPI with a plain HTML/JS frontend, no build step |
| Vector store | One Chroma collection per (chunk size, overlap). Switching back to a setting that was already indexed is instant. |
| Distance | Chroma's default squared L2. The vectors are normalised, so the ranking is the same as cosine similarity. |
| Documents | Stored in `data/documents/`. Uploads must be `.md` or `.txt` in UTF-8. Deleting a document removes its chunks from every collection. |
| Parameter change | A new chunk size or overlap re-indexes before the request returns. A top-k change needs no re-indexing. |
| Response schema | `{answerable, answer, citations[{chunk_id, quote}]}`, validated with Pydantic |
| Citation check | The cited chunk must be one of the retrieved chunks, and the quote must appear in it, ignoring whitespace differences |
| Answer statuses | `answered`, `unverified` (the answer is shown with a warning), `not_found`, `invalid_response`, `llm_error`. A malformed response or an LLM error never crashes the app. |
| Relevance cutoff | None in V1. Distances were logged as data for V2. |
| References | Each reference points to a chunk with its file and character span. The UI opens the chunk and can also show it highlighted in the full document. |
| Eval evidence | Stored as exact quotes, converted to character spans when the eval runs, so it doesn't depend on the chunk size. Each question lists the required parts of its answer; each part can be found in any one of several passages, all located by searching the text. |
| Hit rule | A chunk counts as a hit if it overlaps the evidence span. The metrics are hit@k and MRR, broken down by question type. |
| Eval scope | Retrieval only by default (no LLM). The LLM checks are opt-in flags. Only the baseline result is committed. |
| HNSW | Chroma's search is approximate, so top-k can change from one run to the next. This was kept as it is and documented as a known failure. |
