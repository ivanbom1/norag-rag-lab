# Ask your documents: a RAG web app

Ask questions in plain language about a collection of documents, and get an answer with the exact passages it comes from. Every answer has a **Details** view showing what the app did to produce it: which searches ran, how long each step took, and how many tokens the language model used.

It comes with two collections:

| Collection | What it is | Documents |
|---|---|---|
| `kestrel` | The knowledge base of Kestrel Analytics, a made-up B2B software company (product guide, billing policies, error runbook) | `kestrel-kb/docs/` |
| `monte-cristo` | *The Count of Monte Cristo* in 5 volumes, with a few planted test facts | `corpus/` |

You can add your own `.md` or `.txt` documents from the page.

Built with FastAPI, ChromaDB, `all-MiniLM-L6-v2` embeddings, BM25 (SQLite FTS5), a cross-encoder reranker and Gemini through the OpenAI SDK.

## What you need

- **Python 3.10 to 3.13** (`python3 --version`)
- **A Google Gemini API key**, free from [Google AI Studio](https://aistudio.google.com/app/apikey)
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

Open `.env` in a text editor and replace `your_google_api_key_here` with your key. Then start the app:

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

## Use the app

1. **Pick a collection** at the top left.
2. **Choose the documents** the answer may use. All of them are checked by default. Uncheck one to leave it out.
3. **Type your question** and press Enter.
4. Read the answer. Under **Sources**, click a source to read the passage it comes from, then **Show in document** to see it in its document.
5. Click **Details** to see what happened for this question (see below).

If the documents don't contain the answer, the app says so instead of making one up. If the language model can't be reached (no internet, wrong API key), the page tells you, and nothing else breaks.

**Add a document** with *Add a document* in the sidebar (`.md` or `.txt`). It is indexed straight away and can be used in the next question. **Delete** removes it from the collection.

## Details: what happened for each answer

Every answer has a **Details** button. It opens a table with one row per step of the request:

| Step | What it is |
|---|---|
| Search by meaning (`dense`) | finds passages whose meaning is close to the question (embeddings) |
| Search by keywords (`bm25`) | finds passages that share the question's words, good for codes like `KST-2044` |
| Search the concept graph (`graph`) | follows the links between concepts (Kestrel only, Advanced) |
| Combine the search results (`fusion`) | merges the lists into one (reciprocal rank fusion) |
| Rerank the passages (`rerank`) | reorders the best passages with a second model (Advanced, off by default) |
| Write the answer (`llm`) | the call to Gemini, with its tokens and cost |

Each row shows its time, the tokens and cost of the LLM call, and how many passages it kept (hover to see their ids). **Raw trace** shows the full record.

_To be completed (Part A): where traces are stored, how long they are kept, what they contain and what they leave out, and how to open the trace of an indexing run._

## Protecting users and data

_To be completed (Part C)._

## Advanced mode

Click **Advanced** at the bottom of the sidebar for the tools we used to build and tune the app. The page remembers whether it was open.

- **Settings:** passage size (chunk size), overlap between passages, and how many passages go to the language model (top-k). A new passage size re-indexes the collection, which takes a few minutes.
- **Retrievers:** turn each search on or off. **Rerank the passages with a second model** turns reranking on (see `REPORT.md`, part B, for why it is off by default).
- **Evaluate** tab: runs the question set of the collection and shows hit rate and recall.
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

## The concept graph (Kestrel)

The graph for Kestrel at 800 / 10 % is already in the repo (`graph/kestrel_800_10_20/`), so the **Graph** retriever works without building anything. To rebuild it (one LLM call per passage, cached):

```bash
python -m app.graph --corpus kestrel --workers 6     # only new, changed or failed passages call the LLM
python -m app.graph --corpus kestrel --force         # extract every passage again
python -m app.graph --corpus kestrel --sample 10     # 10 random triples to grade by hand, in triple-grades.json
```

An LLM extracts (entity, relation, entity) triples from every passage, with a fixed schema (`ENTITY_TYPES` and `RELATIONS` in `app/graph.py`). The graph retriever finds the entities named in the question, walks up to 2 relations from them, and returns the passages those relations came from. The Ask tab draws the paths it followed. Details are in `REPORT.md`.

## Tests

```bash
python -m pytest tests
```

The tests don't call the LLM, but `GOOGLE_API_KEY` must be set (the `.env` file is enough).

## Troubleshooting

| Problem | Fix |
|---|---|
| `OpenAIError: Missing credentials` when starting | `.env` is missing or has no key. Run `cp .env.example .env` and put your key in it. |
| "The answer service could not be reached" on every question | Check the key in `.env` and your internet connection, then restart the app. |
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
| `app/indexing.py`, `app/bm25.py` | splitting into passages, ChromaDB and BM25 indexes |
| `app/graph.py`, `app/graph_retriever.py` | building and searching the concept graph |
| `app/reranking.py` | the cross-encoder reranker |
| `app/static/` | the web page |
| `eval/`, `app/evaluation.py` | evaluation and comparison |
| `tests/` | unit tests |
| `REPORT.md` | results, decisions and known failures, V3 and V4 |
| `data/`, `chroma_db/`, `bm25_db/` | created on the first start, not in git |

## Earlier experiments (V1)

Two command-line scripts from the start of the course compare answering from the whole book (NoRAG) with retrieval (RAG), on *Crime and Punishment* (`docs/`, from [Project Gutenberg](https://www.gutenberg.org/ebooks/2554)):

```bash
python ask_norag_google.py     # the whole book in the prompt, with an injected "needle" passage
python ask_rag_google.py       # retrieve 5 passages first, then ask
```

They aren't needed for the web app.
