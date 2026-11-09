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

# RAG web app (V2)

A web app that answers questions about two corpora, with verified references to the chunks it used. It uses FastAPI, ChromaDB, all-MiniLM-L6-v2 embeddings and Gemini.

| Corpus | Documents | Question set |
|---|---|---|
| `kestrel` | `kestrel-kb/docs/` (provided) | `kestrel-kb/eval/questions.starter.jsonl` (10) |
| `monte-cristo` | `corpus/` (*The Count of Monte Cristo* in 5 volumes, with planted facts) | `eval/questions/monte-cristo.jsonl` (33, changes in `eval/questions/CHANGES.md`) |

## Run the app

```bash
pip install -r requirements.txt      # .env must contain GOOGLE_API_KEY
uvicorn app.main:app --reload
```

Open http://localhost:8000; the API docs are at `/docs`. On the first start the app copies each corpus to `data/corpora/<corpus>/` and indexes it, which takes about 4 minutes.

The page has a **corpus selector** and the **document list** on the left, plus the chunk size, overlap and top-k settings. The main area has three tabs: **Ask**, **Evaluate** and **Words**.

## Run the evaluation

From the terminal, from the project root:

```bash
python -m eval.evaluate                                   # retrieval only, both corpora, 800 chars / 10 % overlap
python -m eval.evaluate --chunk-size 400 --overlap 10     # another configuration (indexes it first if needed)
python -m eval.evaluate --corpus kestrel --answers --delay 6   # also generate answers (LLM calls)
python -m eval.evaluate --no-context --delay 4            # no-context test: ask the LLM without retrieval
python -m eval.evaluate --retrievers dense                # one retriever only (dense alone = the V2 baseline)
python -m eval.evaluate --retrievers dense bm25           # several retrievers, fused with RRF (default)
python -m eval.evaluate --corpus kestrel --retrievers dense bm25 graph   # add the graph retriever (needs the graph built)
python -m eval.compare                                    # compare the saved runs of every configuration
```

Retrievers are switched on and off with `--retrievers`, or with the **Retrievers** checkboxes in the sidebar (used by Ask and Evaluate). Results go to `eval/results/<corpus>_800_10_20_<retrievers>.json`, for example `kestrel_800_10_20_dense+bm25.json`. Dense alone keeps the V2 name (`..._minilm.json`).

From the front-end, open the **Evaluate** tab, optionally tick **Also generate answers**, and click **Run evaluation**. It uses the corpus and settings currently selected.

**Measures.** A retrieved chunk is relevant if it comes from the evidence source and contains the evidence quote, ignoring whitespace. Both measures are computed for k = 1, 3, 5 and 10, overall and per question type. Unanswerable questions are not included.

- **HitRate@k:** at least one evidence quote is in the top k.
- **Recall@k:** the share of a question's evidence quotes found in the top k.

**Output** goes to `eval/results/`:

- **`<corpus>_<size>_<overlap>_20_minilm.json`:** the retrieval run, with the 10 retrieved chunks per question and the rank at which each evidence item was found.
- **`...-answers-<timestamp>.json`:** each answer next to the chunks it used, with `evidence_in_context` and empty `grade` / `failure` fields for hand grading. A new file is written each time, so grades are never overwritten.
- **`no-context-<corpus>.jsonl`:** the LLM's answer without retrieval and an LLM judge's verdict. An `override` field holds the human decision. Re-running only re-asks questions that failed, so overrides are kept.
- **`comparison.md`:** written by `eval.compare`: hit and recall per configuration, corpus and question type; cost per question; and the evidence ranks of every question in every configuration, with ▲ / ▼ when recall@5 moves against dense.

**Cost per question** is saved in every run: retrieval latency (per retriever and for the fusion), and with `--answers` the LLM calls, tokens and LLM latency. The CLI and the Evaluate tab show the averages.

## Build the concept graph (Kestrel)

```bash
python -m app.graph --corpus kestrel --workers 6     # gemini-3-flash-preview, 1 LLM call per chunk (42 at 800 / 10 %), cached
python -m app.graph --corpus kestrel --model gemini-3.5-flash-lite --delay 4   # another model
python -m app.graph --corpus kestrel --force         # extract every chunk again, ignoring the cache
python -m app.graph --corpus kestrel --sample 10     # 10 random triples to grade by hand, in triple-grades.json
```

An LLM extracts (entity, relation, entity) triples from every chunk, with a fixed schema (`ENTITY_TYPES` and `RELATIONS` in `app/graph.py`) and a short quote from the chunk for each triple. The prompt also gets the document headings around the chunk, because chunking drops short headings such as `### auth`. Rebuilding only calls the LLM for new, changed or failed chunks. The output goes to `graph/kestrel_800_10_20/`:

- **`extractions.jsonl`:** one row per chunk: the triples kept, the triples dropped by validation and why, and the cost of the call.
- **`graph.json`:** the merged entities (with every name they were found under and their chunks) and relations (with their chunks and quotes).
- **`build-report.json`:** entities and relations per type, merges, failed chunks, dropped triples, and the cost (LLM calls, tokens, time).

**Graph retriever** (`app/graph_retriever.py`, the **Graph** checkbox or `--retrievers ... graph`). It finds the entities named in the question (every word of the name must be in the question, and a plan also needs the word "plan"), walks up to 2 relations from them in both directions, and returns the chunks those relations were extracted from, closest first. Each chunk keeps the path that led to it, from the entity named in the question, and the Ask tab draws the paths of all retrieved chunks as one diagram (entities coloured by type, the entities named in the question on the left), and lists them under each retrieved chunk and in the reference panel (for example `KST-5031 —raised_by→ postroom, then postroom —owned_by→ Messaging`). It is only available where a `graph.json` exists for the corpus and chunk settings, and it is not on by default.

## Reach the Words tab

Open http://localhost:8000, pick a corpus, and click the **Words** tab. The counts are for the chunks of that corpus at the current chunk size. Words are lowercased with no stemming, and identifiers like `KST-2044` stay one word.

- **Question mode.** Type a question, or pick one from the question set, then click **Analyze words**. For each word you get its count and how many chunks contain it; for a picked question, you also see whether it appears in the chunks that hold the evidence. Words absent from the corpus are shown in red.
- **Corpus mode.** Shows the vocabulary (word, count, number of chunks), with a filter and an option to show stopwords. Click a word to list the chunks that contain it, and click a chunk to read it with the word highlighted.

## Tests

```bash
python -m pytest tests     # chunk offsets and tokenizer
```
