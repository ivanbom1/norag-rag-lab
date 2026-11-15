# REPORT - RAG V4

## A. Observability: decision log

### What do we trace?

There are two kinds of trace:

- **`question`:** one per question asked in the app (`/ask`).
- **`indexing`:** one per indexing run that actually indexed something: at startup when a document is missing, after the chunk settings change, and when a document is uploaded.

The evaluation CLI makes no traces, so evaluation runs don't fill the store.

A question trace holds:

- **Identity:** trace id, timestamp, total duration.
- **Configuration:** corpus, selected documents, chunk size, overlap, top-k, the retrievers switched on, rerank on or off.
- **Content:** the question and the answer, both redacted (Part C), and for each citation its chunk id and whether it verified.
- **Outcome:** `answered`, `not_found` (the app abstained), `unverified`, `invalid_response`, `llm_error`, or `error` when the request itself failed.

Its spans, in order, each with its duration:

- **One per retriever switched on** (`dense`, `bm25`, `graph`), with the chunk ids it returned in rank order. That shows which retriever found a chunk, and at what rank.
- **`fusion`:** the chunk ids kept after RRF.
- **`rerank`,** when on: the number of candidates, the chunk ids kept, and the error if the model failed.
- **`redact`:** how many values of each kind were replaced.
- **`llm`:** model, input and output tokens, cost in dollars, or the error.

An indexing trace holds the reason and the settings, with one span per document: `index_document` (chunks, and the time for chunking, embedding + Chroma, and BM25), or `copy_to_bm25` when the document was already in Chroma and only BM25 was missing.

**Never in a trace:** chunk text (the chunk id is enough to find it), prompts, API keys.

**Not traced yet:** the prompt version, the embedding model and RRF k, a fingerprint of the whole configuration, which model actually responded and its finish reason, and the retrieval scores.

### Where are they stored, and for how long?

In `traces/traces.sqlite3`: one row per trace, with the id, kind, timestamp, status and duration as columns and the rest of the trace, spans included, as JSON. The folder is gitignored.

Traces are kept 30 days. The app deletes older ones each time it starts (`RETENTION_DAYS` in `app/tracing.py`). A trace that can't be written doesn't break the request: the app prints a message and answers anyway.

### Who can read them? Do they leave the machine?

Anyone who can use the laptop, and anyone who can reach the app's `GET /api/traces` and `GET /api/traces/{id}`. There is no login, because the app runs locally for one user. Deploying it would need one.

Traces never leave the machine. Only the LLM call goes to Google, and its text is redacted first (Part C).

### Personal data in traces

The question and the answer are stored only after redaction, with placeholders such as `[EMAIL_1]`. The `redact` span stores counts per kind, never the values, and the mapping from placeholders to values only lives in memory for one request. Chunk text isn't stored at all.

Names are not removed (see Part C), so a question that names a person keeps that name in its trace.

### Cost in dollars

The cost comes from the token counts the API returns and the list price of gemini-3.5-flash-lite: $0.30 per million input tokens and $2.50 per million output tokens (ai.google.dev pricing page, November 2026). Our free-tier key isn't billed, so the figure is what the same call would cost on a paid key. For example, 365 input + 95 output tokens cost $0.000347.

Retrieval, reranking, embeddings and BM25 run locally and cost nothing in dollars. The graph extraction calls happen when the graph is built, not per question, and aren't traced.

### From traces to metrics

There is no metrics page yet. The traces hold what it would need, in one query over the `traces` table followed by Python:

- **Outcomes:** count of traces per `status`.
- **Latency:** p50 and p95 of `duration_ms`, overall and per span name. Percentiles show the slow questions that an average hides.
- **Cost:** the sum of the `llm` span's `cost`, per question and per day.

### Options we rejected

- **A JSON lines file:** easy to append to, but finding one trace by id, or computing any total, means reading the whole file.
- **Langfuse or Phoenix, self-hosted:** one more service to install, run and update (Docker), plus an SDK dependency. Too much for one user on a laptop.
- **A hosted tool (Langfuse Cloud, LangSmith):** a third party would receive the users' questions, chunk ids and answers.
- **OpenTelemetry with Jaeger or Grafana:** a whole stack to run for one app. We kept its vocabulary (trace, span, attributes, status) without the SDK.
- **A separate spans table:** only useful for SQL on spans, which only the metrics page would need.

## B. New feature: reranking

### What it does

After dense + BM25 fusion, the 30 best fused passages are scored again by a cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`), which reads the question and each passage together. The best top-k are kept. Code is in `app/reranking.py`, called from `retrieve_with_cost` in `app/answering.py`.

We picked a local cross-encoder over an LLM reranker: no API call, no tokens, no cost, and the passages don't leave the machine for this step. It comes with `sentence-transformers`, which we already use for the embeddings. The model (about 90 MB) is downloaded once and loaded when the server starts.

Switch: "Rerank the passages with a second model" in Advanced, `rerank` in the `/ask`, `/search` and `/evaluate` bodies, `--rerank` in `python -m eval.evaluate`. A reranked passage shows "rerank #n" next to its dense and BM25 ranks.

### Traced, cost

Reranking is its own step in the trace ("Rerank the passages"), with its time and the passages it kept. It makes no LLM call, so it adds no tokens and no money cost. It only adds time:

| corpus | retrieval without rerank | with rerank | of which rerank |
|---|---|---|---|
| kestrel | 13 ms | 284 ms | 268 ms |
| monte-cristo | 25 ms | 363 ms | 320 ms |

Our V3 answer run on kestrel (`kestrel_800_10_20_dense+bm25-answers-20261109-030321.json`) took 1.43 s per question for the LLM call, so reranking adds about 20 % to a question.

### Failures

If the model can't be loaded (no network on first start, missing package), the server still starts. The question is answered with the normal search order, and the Details panel shows the rerank step in red with the reason. Covered by `tests/test_reranking.py` (6 tests: ordering, ranks kept, 30 candidates, fallback, no call when off).

### Evaluation: on vs off

Same settings for both runs (800 / 10 %, dense + BM25, top 10), run one after the other on 14 November. Files: `eval/results/<corpus>_800_10_20_dense+bm25.json` and `..._dense+bm25+rerank.json`.

| corpus | rerank | hit@1 | hit@3 | hit@5 | hit@10 | recall@1 | recall@3 | recall@5 | recall@10 |
|---|---|---|---|---|---|---|---|---|---|
| kestrel (10) | off | 0.40 | 1.00 | 1.00 | 1.00 | 0.25 | 0.78 | 0.85 | 0.90 |
| kestrel (10) | on | **0.70** | 1.00 | 1.00 | 1.00 | **0.45** | **0.85** | 0.85 | 0.85 |
| monte-cristo (28) | off | 0.43 | 0.61 | 0.64 | 0.71 | 0.38 | 0.54 | 0.59 | 0.68 |
| monte-cristo (28) | on | **0.46** | **0.64** | 0.64 | 0.71 | **0.41** | **0.59** | 0.59 | 0.66 |

Reranking puts the right passage first more often (kestrel hit@1 0.40 → 0.70), but it does not change which passages are in the top 5: hit@5 and recall@5 are the same on both corpora.

What moved, question by question:

- Better: q01 (KST-2044, rank 3 → 1), q05 (lost phone, 2 → 1), q03 (card payment, three parts 5 → 2), d03 (prisoner number, 7 → 3), d10 (two words of wisdom, 3 → 1), s02 (postmark, 10 → 2), n03 (No. 27, missed → 10).
- Worse: q07 (annual saving): the 2025 billing policy now comes before the 2026 one (1 → 2). The cross-encoder judges the topic, not the year. h01 (supercargo's trick) drops out of the top 10 (9 → missed), h03 (poison) 5 → 9.
- Two-hop: the first hop gets better (q08 and q12 rank 2 → 1, n01 4 → 2), but the second hop is pushed out: q12 10 → missed, n01 3 → missed. The second hop doesn't look like the question, which is exactly what the cross-encoder scores. That's what the graph was for in V3.

With 10 and 28 questions, one question is 0.10 or 0.04, so only the kestrel hit@1 change is more than a question or two.

### Decision

Off by default, switch in Advanced. The app sends the top 5 to the LLM, and at k = 5 reranking changes the order but not the content, for +0.3 s on every question. It would matter with a smaller top-k (1 to 3), where hit@1 and hit@3 go up. It's also the wrong tool for two-hop and version questions, which are our hard cases.

## C. Responsible AI: personal data redaction (decision log)

### What it does

Before every call to the LLM, emails, phone numbers (international, French and North American formats), IBANs, card numbers and IP addresses are replaced by numbered placeholders such as `[EMAIL_1]`. A value gets the same placeholder every time it appears in a request, so the LLM can still refer to it. This covers the question, every chunk in the prompt, and the graph extraction calls.

The real values are put back in memory only:

- **The answer and the citations,** so the user sees the values and the citation quotes still verify.
- **The extracted triples,** so the graph stores the real values.

Traces get the redacted text. The stored indexes (Chroma and BM25) keep the original text, because they never leave the machine.

### How we detect, and why

We use regular expressions, with a checksum for IBANs (mod 97) and card numbers (Luhn), so order numbers and mistyped IBANs are left alone. Phone numbers are matched before IP addresses, so `06.99.00.56.78` is read as a phone number.

We don't detect names:

- **A local named-entity model (spaCy)** would also remove every character in Monte Cristo (Dantès, Mercédès…) and break the answers.
- **An LLM** would need the text sent to the provider to find the personal data, which is exactly what redaction is meant to avoid.

### How we checked it

- **`tests/test_redaction.py`:** 10 values of 5 kinds are replaced and restored. Look-alikes are left alone: dates, times, `KST-2044`, chunk ids, prices, version numbers, a 16-digit number that fails Luhn, and an IBAN with a wrong checksum. One test checks that a repeated value keeps its placeholder.
- **`tests/test_tracing.py`:** `ask()` with a fake LLM. Neither the prompt nor the stored trace contains the email, the user still gets it back, and the citation still verifies.
- **By hand:** we uploaded `samples/contacts-test.md` (fake data). The answer showed the real email and phone number, and the trace showed `[EMAIL_1]` and `[PHONE_1]` with `replaced: {EMAIL: 2, PHONE: 4, IBAN: 1, CARD: 1, IP: 1}`.

### Known gaps

- **Names and postal addresses** are not detected, so the LLM still sees names next to the placeholders.
- **Formats outside the patterns** get through: other countries' phone formats, or an email written as "name at domain".
- **Placeholders the LLM changes** can't be restored: if it writes `[EMAIL 1]` instead of `[EMAIL_1]`, the user sees the placeholder instead of the value.
