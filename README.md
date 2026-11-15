# Ask your documents: a RAG web app

Ask questions in plain language about a collection of documents, and get an answer with the exact passages it comes from. Every answer has a **Details** view showing what the app did to produce it: which searches ran, how long each step took, and how many tokens the language model used.

It comes with two collections:

| Collection | What it is | Documents |
|---|---|---|
| `kestrel` | The knowledge base of Kestrel Analytics, a made-up B2B software company (product guide, billing policies, error runbook) | `kestrel-kb/docs/` |
| `monte-cristo` | *The Count of Monte Cristo* in 5 volumes, with a few planted test facts | `corpus/` |

You can create your own collections and add `.md` or `.txt` documents from the page, then build a **concept graph** of any collection, explore it, and use it to answer questions.

Built with FastAPI, ChromaDB, `all-MiniLM-L6-v2` embeddings, BM25 (SQLite FTS5), a cross-encoder reranker and any language model with an OpenAI-compatible API (Google Gemini by default).

## What you need

- **Python 3.10 to 3.13** (`python3 --version`)
- **A language model with an OpenAI-compatible API.** A free Google Gemini key from [Google AI Studio](https://aistudio.google.com/app/apikey) works out of the box; any other provider or a local server (Ollama, LM Studio…) works too, see [Choose the language model](#choose-the-language-model).
- **An internet connection on the first start**, to download two small models (about 90 MB each) from Hugging Face
- About 2 GB of disk space for the Python packages

## Install and run

From a terminal:

```bash
git clone https://github.com/ivanbom1/norag-rag-lab.git
cd norag-rag-lab

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env               # Windows: copy .env.example .env
```

Open `.env` in a text editor and set `LLM_URL`, `LLM_API_KEY` and `LLM_MODEL` (see [Choose the language model](#choose-the-language-model); for Gemini, only the key needs changing). Then start the app:

```bash
uvicorn app.main:app
```

Open **http://localhost:8000** in your browser.

**The first start takes about 5 minutes.** The app copies both collections to `data/corpora/`, splits them into passages, indexes them and downloads the models. The page opens once the terminal shows `Application startup complete`. Later starts take a few seconds.

To stop the app, press `Ctrl+C` in the terminal. Next time, you only need:

```bash
cd norag-rag-lab
source .venv/bin/activate
uvicorn app.main:app
```

## Choose the language model

The app talks to the language model through the OpenAI API format, so any compatible server works. Set these in `.env` (or `.env.local`, which takes priority and is also kept out of git):

| Variable | What it is | Example |
|---|---|---|
| `LLM_URL` | base URL of the API | `https://generativelanguage.googleapis.com/v1beta/openai/` (Gemini), `https://api.openai.com/v1`, `http://localhost:11434/v1` (Ollama) |
| `LLM_API_KEY` | the API key; leave it empty for a local server that doesn't need one | |
| `LLM_MODEL` | the model that writes the answers | `gemini-3.5-flash-lite`, `gpt-4o-mini`, `llama3.2` |
| `LLM_GRAPH_MODEL` | optional: the model that builds concept graphs; defaults to `LLM_MODEL` | `gemini-3-flash-preview` |
| `LLM_PRICE_INPUT`, `LLM_PRICE_OUTPUT` | optional: USD per million tokens, to show the cost in Details (known for `gemini-3.5-flash-lite`, otherwise 0) | `0.15`, `0.60` |

Restart the app after changing them. **Advanced → Settings** shows the model and server in use. If a server doesn't support JSON-schema output, the app asks for plain JSON instead and reads it from the reply.

An older `.env` with only `GOOGLE_API_KEY` still works: the app then uses Gemini.

## Use the app

1. **Pick a collection** at the top left, or click **New collection** to make your own (lowercase letters, digits and dashes) and add its documents.
2. **Choose the documents** the answer may use. All of them are checked by default. Uncheck one to leave it out.
3. **Type your question** and press Enter.
4. Read the answer. Under **Sources**, click a source to read the passage it comes from, then **Show in document** to see it in its document.
5. Click **Details** to see what happened for this question (see below).

If the documents don't contain the answer, the app says so instead of making one up. If the language model can't be reached (no internet, wrong API key), the page tells you, and nothing else breaks.

**Add a document** with *Add a document* in the sidebar (`.md` or `.txt`). It is indexed straight away and can be used in the next question. **Delete** removes it from the collection.

**Concept graph.** The *Concept graph* section of the sidebar builds a graph of the concepts in the collection (people, organizations, places, ideas…) and the links between them:

1. Click **Build concept graph**. The language model reads each passage once, so a large collection takes a while; the page shows the progress and you can keep asking questions meanwhile.
2. When it's ready, **Use it to answer questions** is checked: the graph becomes one more search, next to meaning and keywords.
3. **Explore the graph** lists the concepts. Click one to see what it's linked to, open the passage each link comes from, or click **Ask about it**.
4. After you add or delete documents, the section says the graph is out of date: **Update concept graph** only reads the new passages.

## Details: what happened for each answer

Every answer has a **Details** button. It opens a table with one row per step of the request:

| Step | What it is |
|---|---|
| Search by meaning (`dense`) | finds passages whose meaning is close to the question (embeddings) |
| Search by keywords (`bm25`) | finds passages that share the question's words, good for codes like `KST-2044` |
| Search the concept graph (`graph`) | follows the links between concepts, when the collection has a graph and *Use it to answer questions* is checked |
| Combine the search results (`fusion`) | merges the lists into one (reciprocal rank fusion) |
| Rerank the passages (`rerank`) | reorders the best passages with a second model (Advanced, off by default) |
| Write the answer (`llm`) | the call to the language model, with its tokens and cost |

Each row shows its time, the tokens and cost of the LLM call, and how many passages it kept (hover to see their ids). **Raw trace** shows the full record.

Every question, every indexing run and every concept graph build is recorded as a trace in `traces/traces.sqlite3` on your machine, kept 30 days. Traces hold timings, token counts, chunk ids and the redacted question and answer, never the passage text, the prompts or API keys. The API lists them at `GET /api/traces` and returns one at `GET /api/traces/{id}`. Details in `REPORT_V4.md`, part A.

## Protecting users and data

Before anything is sent to the language model (questions, passages, graph extraction), email addresses, phone numbers, IBANs, card numbers and IP addresses are replaced by placeholders such as `[EMAIL_1]`. The real values are put back only in what you see on the page. Traces keep the placeholders. Names and postal addresses are not detected. Details in `REPORT_V4.md`, part C.

## Advanced mode

Click **Advanced** at the bottom of the sidebar for the tools we used to build and tune the app. The page remembers whether it was open.

- **Settings:** passage size (chunk size), overlap between passages, and how many passages go to the language model (top-k). A new passage size re-indexes the collection, which takes a few minutes.
- **Settings** also shows the language model in use.
- **Retrievers:** turn each search on or off. **Rerank the passages with a second model** turns reranking on (see `REPORT_V4.md`, part B, for why it is off by default).
- **Evaluate** tab: runs the question set of the collection and shows hit rate and recall (only `kestrel` and `monte-cristo` have one).
- **Words** tab: the vocabulary of the collection, and which words of a question appear in it.
- Under each answer, **How this answer was found** lists every passage searched and which search found it.

## Evaluate from the terminal

With the virtual environment active, from the project folder:

```bash
python -m eval.evaluate                                   # both collections, dense + BM25, 800 chars / 10 % overlap
python -m eval.evaluate --rerank                          # the same with reranking
python -m eval.evaluate --retrievers dense                # one retriever only (the V2 baseline)
python -m eval.evaluate --corpus kestrel --retrievers dense bm25 graph   # with the concept graph
python -m eval.evaluate --chunk-size 400 --overlap 10     # another passage size (indexes it first if needed)
python -m eval.evaluate --corpus kestrel --answers --delay 6   # also generate answers (LLM calls)
python -m eval.evaluate --no-context --delay 4            # ask the LLM without any retrieval
python -m eval.compare                                    # compare every saved run in eval/results/comparison.md
```

Question sets: `kestrel-kb/eval/questions.starter.jsonl` and `eval/questions/monte-cristo.jsonl` (our changes are listed in the `CHANGES.md` next to each).

**Measures** (k = 1, 3, 5, 10, overall and per question type; unanswerable questions are left out). A passage is relevant if it comes from the evidence document and contains the evidence quote.

- **HitRate@k:** at least one evidence quote is in the top k.
- **Recall@k:** the share of a question's evidence quotes found in the top k.

**Output** in `eval/results/`:

- `<corpus>_800_10_20_<retrievers>.json`, for example `kestrel_800_10_20_dense+bm25+rerank.json`: the 10 passages retrieved per question, the rank of each evidence quote, and the cost per question (time per step). Dense alone keeps the V2 name, `..._minilm.json`.
- `...-answers-<timestamp>.json`: each answer next to the passages it used, with empty `grade` / `failure` fields for grading by hand. A new file every time, so grades are never overwritten.
- `no-context-<corpus>.jsonl`: the LLM's answer without retrieval and an LLM judge's verdict, with an `override` field for the human decision.
- `comparison.md`: written by `eval.compare`.

## How the concept graph works

The language model extracts (entity, relation, entity) triples from every passage, each with a short quote from the passage. Entities with the same type and name are merged. The graph retriever finds the entities named in the question, walks up to 2 relations from them, and returns the passages those relations came from. The Ask tab draws the paths it followed.

- **`kestrel`** uses a fixed schema written for its knowledge base (error codes, services, teams, plans…, `ENTITY_TYPES` and `RELATIONS` in `app/graph.py`). Its graph at 800 / 10 % is already in the repo (`graph/kestrel_800_10_20/`).
- **Every other collection** uses a general schema: people, organizations, places, products, concepts, events and works, with free relation names like `works_for` or `located_in`.

Graphs are stored in `graph/<collection>_<chunk size>_<overlap>_20/` and cached per passage, so an update only calls the language model for new, changed or failed passages. One graph is built per collection and passage size. The page builds it in the background with 4 calls at a time (`GRAPH_WORKERS` and `GRAPH_DELAY` in `.env` change that, if your provider limits the rate).

From the terminal:

```bash
python -m app.graph --corpus my-notes --workers 6    # any collection
python -m app.graph --corpus kestrel --force         # extract every passage again
python -m app.graph --corpus kestrel --sample 10     # 10 random triples to grade by hand, in triple-grades.json
```

Details are in `REPORT_V3.md` and `REPORT_V4.md`.

## Tests

```bash
python -m pytest tests
```

The tests don't call the language model and don't need an API key.

## Troubleshooting

| Problem | Fix |
|---|---|
| "The answer service could not be reached" on every question | Check `LLM_URL`, `LLM_API_KEY` and `LLM_MODEL` in `.env` (Advanced → Settings shows what the app uses) and your connection, then restart the app. |
| "The language model returned a malformed answer" | The model didn't return the JSON the app asks for. Small local models do this more often: try a larger one. |
| Concept graph: "Every passage failed" | Usually a wrong key, model name or a rate limit. The message shows the first error. Fix it and click **Build** again: passages that worked are kept. |
| Concept graph build is slow or hits rate limits | Lower `GRAPH_WORKERS` (or raise `GRAPH_DELAY`) in `.env` and restart. A collection the size of `monte-cristo` has thousands of passages. |
| `uvicorn: command not found` or `No module named ...` | The virtual environment isn't active: `source .venv/bin/activate`. |
| `Address already in use` | Another app uses port 8000. Run `uvicorn app.main:app --port 8001` and open http://localhost:8001. |
| The first start seems stuck | Indexing and the model downloads take about 5 minutes. Wait for `Application startup complete`. |
| A warning about unauthenticated requests to the HF Hub | Harmless. Setting `HF_TOKEN` only raises Hugging Face's download limits. |
| Rerank row in red in Details | The reranking model couldn't be loaded (usually no internet on the first start). Answers still work, without reranking. Restart once online. |
| Start again from scratch | Stop the app and delete `data/`, `chroma_db/` and `bm25_db/`. The next start rebuilds them. |

## Project layout

| Path | What's there |
|---|---|
| `app/main.py` | the FastAPI app and its API (docs at http://localhost:8000/docs) |
| `app/answering.py` | retrieval, fusion, reranking and the LLM call |
| `app/llm.py` | the language model settings (`LLM_URL`, `LLM_API_KEY`, `LLM_MODEL`) and client |
| `app/indexing.py`, `app/bm25.py` | splitting into passages, ChromaDB and BM25 indexes |
| `app/graph.py`, `app/graph_retriever.py` | building and searching the concept graph |
| `app/graph_jobs.py` | building a graph in the background from the page, its status, and the explorer |
| `app/tracing.py`, `app/redaction.py` | traces, and personal data redaction |
| `app/reranking.py` | the cross-encoder reranker |
| `app/static/` | the web page |
| `eval/`, `app/evaluation.py` | evaluation and comparison |
| `tests/` | unit tests |
| `REPORT_V3.md`, `REPORT_V4.md` | results, decisions and known failures |
| `data/`, `chroma_db/`, `bm25_db/` | created on the first start, not in git |

## Earlier experiments (V1)

Two command-line scripts from the start of the course compare answering from the whole book (NoRAG) with retrieval (RAG), on *Crime and Punishment* (`docs/`, from [Project Gutenberg](https://www.gutenberg.org/ebooks/2554)):

```bash
python ask_norag_google.py     # the whole book in the prompt, with an injected "needle" passage
python ask_rag_google.py       # retrieve 5 passages first, then ask
```

They aren't needed for the web app.
