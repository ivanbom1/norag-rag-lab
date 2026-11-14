# REPORT - RAG V3

Team: Tristan and Ivan

All numbers: chunks 800 chars / 10 % overlap, min paragraph 20, MiniLM embeddings, top 10 retrieved, 20 candidates per retriever into the fusion. Source: `eval/results/comparison.md` (`python -m eval.compare`). Kestrel has 10 answerable questions (q01–q08, q11, q12), so one question = 0.10. Monte Cristo has 28. Chroma's HNSW search is approximate, so the dense numbers move by about ±0.04 between runs.

---

## 1. Results and default configuration

| corpus | config | hit@1 | hit@5 | hit@10 | recall@5 | recall@10 | two_hop recall@5 | retrieval / question |
|---|---|---|---|---|---|---|---|---|
| kestrel | dense | 0.50 | 0.80 | 1.00 | 0.70 | 0.85 | 0.33 | 29 ms |
| kestrel | bm25 | 0.70 | 0.90 | 0.90 | 0.68 | 0.80 | 0.50 | 3 ms |
| kestrel | graph | 0.20 | 0.30 | 0.40 | 0.20 | 0.35 | 0.17 | 2 ms |
| kestrel | **dense+bm25** | 0.40 | **1.00** | 1.00 | **0.85** | 0.90 | 0.50 | 27 ms |
| kestrel | dense+bm25+graph | 0.50 | 1.00 | 1.00 | 0.85 | 0.90 | 0.50 | 16 ms |
| monte-cristo | dense | 0.29 | 0.50 | 0.64 | 0.46 | 0.59 | 0.00 | 110 ms |
| monte-cristo | bm25 | 0.39 | 0.57 | 0.57 | 0.52 | 0.55 | 0.50 | 35 ms |
| monte-cristo | **dense+bm25** | 0.43 | **0.64** | 0.71 | **0.59** | 0.68 | 0.50 | 150 ms |

No retrieval configuration makes an LLM call. Answering adds 1 LLM call (about 1,050 prompt + 170 completion tokens, 1.3–2.1 s). There is no Monte Cristo graph: the graph is only built for Kestrel.

**Default: dense + bm25, fused with RRF. Graph is off.**

- dense + bm25 is the best or tied-best configuration on recall@5 on both corpora: Kestrel 0.70 → 0.85, Monte Cristo 0.46 → 0.59.
- The two retrievers fail on different questions, so fusion keeps the wins of both. BM25 fixes the identifiers (q01 KST-2044: dense rank 6 → fused rank 3). Dense covers the vocabulary questions BM25 misses (q05 "lost phone": BM25 finds nothing, fused rank 2).
- Fusion costs almost nothing: BM25 adds 2–45 ms and RRF 0.1 ms.
- Adding the graph changes no hit@5 or recall@k. It only moves hit@1 from 0.40 to 0.50, which is one question (q11). It also needs a paid-for LLM build per corpus and chunk setting, and it exists only for Kestrel. So it is a checkbox, not a default.
- Known cost of the default: hit@1 on Kestrel drops from 0.70 (bm25 alone) to 0.40. This matters little because answers use the top 5, and hit@5 is 1.00.

## 2. BM25

**Index.** Our own tables in one SQLite file, `bm25_db/bm25.sqlite3`. (We first used FTS5, then replaced it to use the course's BM25 formula; see the decision log.) The `chunks` table holds each chunk with its length in words. The `postings` table holds (word, chunk, count). Each corpus × chunk setting is its own index (`index_name`, like `kestrel_800_10_20`), the same split as the Chroma collections, so word statistics never mix across corpora. Scoring is the course `BM25Index` formula with k1 = 1.5 and b = 0.75. N, the average length and the document frequencies are read from the index at query time. The document filter (sidebar checkboxes) only filters the results, not the statistics. Code: `app/bm25.py`.

**Tokenization.** The same tokenizer as the Words tab (`app/tokens.py`): lowercase, no stemming, and letters/digits joined by `-`, `.` or `'` stay one token. So `KST-2044`, `203.0.113.10` and `captain’s` are single words. Stopwords are removed from the question only; the index keeps every word. A test (`vocabulary()` in `tests/test_bm25.py`) checks that the index tokenizes exactly like the Words tab.

**Updates.** `index_document()` writes the same chunks, with the same ids, to Chroma and BM25 in one pass. Re-indexing a document first deletes its old chunks and postings, then inserts the new ones, all in one transaction. Deleting a document removes it from both indexes. At startup, any document that exists only in Chroma is copied into BM25 from the stored chunks, without embedding it again. Because the statistics are computed per query, an update never needs a full rebuild.

**What it changed.** It wins on identifiers and exact wording: q01 KST-2044 (rank 1, dense 6), Monte Cristo d10, p02, s01, n01. It loses on paraphrase, because there is no stemming or synonyms: q05 (lost phone), h03, h07. Monte Cristo n03 ("prisoner No. 27") is still missed: "27" is in 12 chunks, and "prisoner", "château" and "d’if" are spread over dozens.

## 3. The concept graph (Kestrel)

### Schema

Fixed types in `app/graph.py`. Each relation allows exactly one subject type and one object type. The LLM response is constrained with a JSON schema, then each triple is validated on its own with Pydantic.

- 9 entity types: ErrorCode, Service, Team, Person, Channel, Plan, Role, Feature, Policy.
- 13 relations: `raised_by`, `escalates_to`, `concerns` (ErrorCode →); `owned_by`, `replaced`, `handles` (Service →); `on_call_channel`, `led_by`, `covered_by` (Team →); `includes` (Plan → Feature); `can_do` (Role → Feature); `supersedes`, `covers` (Policy →).

Every triple keeps its chunk and a word-for-word quote, which is checked against the chunk (`quote_verified`). The prompt also gets the headings around the chunk, because chunking drops short headings like `### auth`.

**Build result** (`graph/kestrel_800_10_20/build-report.json`): 42 chunks, 41 extracted and 1 failed. 222 triples, 0 dropped by validation. 129 entities and 172 relations.

| entities | n | relations | n |
|---|---|---|---|
| Feature | 76 | includes | 48 |
| ErrorCode | 12 | handles | 30 |
| Service | 11 | covers | 26 |
| Team, Channel, Person | 6 each | can_do | 25 |
| Plan | 5 | concerns | 14 |
| Role | 4 | owned_by | 9 |
| Policy | 3 | led_by, on_call_channel | 6 each |
| | | escalates_to | 3 |
| | | covered_by | 2 |
| | | raised_by, replaced, supersedes | 1 each |

### Entity resolution

Two names are the same entity if they have the same type and the same normalized name (`normalize_name`). Normalization applies NFKC, lowercases, unifies quotes and dashes, collapses spaces, trims punctuation and a leading "the". It also removes a trailing " team" for teams and " service" for services, and adds `#` to channels. There is no fuzzy or embedding matching.

The result was 10 merges, and every one is only a difference in letter case (`Audit log` / `audit log`, `Single sign-on` / `single sign-on`...). What this rule misses is in "Where it fails".

### Retrieval

`app/graph_retriever.py`, no LLM call at query time.

1. **Seeds:** the entities whose name, or one of their merged names, has all its words in the question. Plans also need the word "plan", because "Team" or "Free" are common words.
2. **Walk:** up to 2 relations from the seeds, in both directions.
3. **Return:** the chunks the reached relations were extracted from, closest hop first, then the chunks reached by the most paths. Each chunk keeps the paths that led to it, for the path view.

### Cost

- **Build:** one LLM call per chunk with `gemini-3-flash-preview`. In total 45 calls, 35,376 prompt + 12,037 completion tokens, so about 840 + 290 tokens per chunk. The cache (`extractions.jsonl`, keyed by chunk id and text hash) means a rebuild only calls the LLM for new, changed or failed chunks. The last run did 22 calls in 623 s wall time.
- **Rate limits:** the free tier allows 20 requests a day per model, so the build had to be done over two runs. `support-runbook-errors.md#10` still failed with a 429 error.
- **First try:** `gemini-3.5-flash-lite` (the answer model) produced only 23 triples, 2 error codes and 4 features (`first-build-flash-lite.json`). That is why we switched model.
- **Query:** 1.6–1.8 ms per question, 0 LLM calls.

### Extraction quality check

- **Quotes.** 30 of 222 quotes (14 %) are not found word for word in their chunk. All 30 are near misses, not invented text:
  - 21 drop the markdown bold (`Admin:` for `**Admin**:`);
  - 3 quote a heading from the prompt that is not in the chunk;
  - 6 have corrupted accents (see below).
- **Sample.** `python -m app.graph --sample 10` wrote 10 random triples (seed 42) to `triple-grades.json`. Our reading of them against their chunks:
  - 8 correct;
  - 1 partly right: "Free includes seats" comes from "the plan sets the number of seats";
  - 1 wrong: "Business includes role changes". It is the audit log that records role changes, and the audit log is what Business includes.

### Where it fails

Per question (graph alone, evidence ranks in its top 10): it finds evidence for q02, q04, q08 (hop 1) and q11 (both hops, ranks 6 and 8). It returns nothing for q01, q03, q05, q06, q07, q09 and q10.

- **Facts it never extracted.** KST-2044 is not in the graph. The extractor attributed the chunk's escalation sentence to KST-2040 and produced no triple for KST-2044, so q01 has no seed. There is only 1 `raised_by` relation for 12 error codes. Versions and prices (q06, q07) don't fit the schema at all.
- **Wrong granularity.** In q12, "automatic payment retries" was never extracted as a Feature of ledger, which got "payments", "invoices", "seats" and "subscriptions" instead. The only seed is `Feature:retries`, which leads to the policy, not to Payments Core.
- **Path too long.** q08 needs KST-5031 → postroom → Messaging → #oncall-messaging. That is 3 relations, and the walk stops at 2, so the channel chunk is never reached.
- **Resolution misses.** Singular and plural stay separate (`scheduled report` / `scheduled reports`, `payment method` / `payment methods`). Feature is a catch-all type with 76 of 129 entities. Nothing links `audit-log` (Service) and `Audit log` (Feature).
- **Accents corrupted.** The model's output replaced accented letters with a digit: Inès → `In8s`, Léa → `L9a`, Wójcik → `W3jcik`. The stored chunk text is correct, so this comes from the extraction output. We have not confirmed the cause. It breaks the quotes of 6 triples and any search for those people.
- **Noise.** Walking 2 hops from a general seed like `Feature:dashboards` pulls in billing chunks: q11 returns `billing-policy-2025.md#4` at rank 2. Unanswerable questions return nothing, which is correct.

### Features we added beyond the base graph

- The path view in the Ask tab, plus the path listed under each chunk and in the references.
- The extraction cache and incremental rebuild.
- Heading context in the extraction prompt.
- Quote check on every triple, and the build report with cost.
- Triple sampling for hand grading.
- `--model`, `--workers` and `--force` options.
- The graph is reloaded when `graph.json` changes, so a rebuild doesn't need an app restart.
- Two new two_hop questions (section 6).

## 4. What happened to the graph's chunks after fusion

Fusion is RRF with k = 60. Each retriever gives its top 20, and a chunk scores 1 / (60 + rank) in every list it is in. The graph returned chunks for 5 of the 10 answerable questions.

| question | graph chunks in fused top 5 | in fused top 10 | found only by the graph (top 10) | what changed vs dense+bm25 |
|---|---|---|---|---|
| q02 | 3 | 4 | 0 | `product-guide.md#4` enters the top 5 |
| q04 | 5 | 10 | 0 | `product-guide.md#4` enters the top 5 |
| q08 | 3 | 3 | 0 | `service-catalog.md#1` (graph #1, dense #16) enters the top 5 |
| q11 | 4 | 7 | 0 | hop-1 evidence goes to rank 1 (hit@1); `service-catalog.md#1` and `support-runbook-errors.md#9` enter the top 5 |
| q12 | 1 | 1 | 0 | nothing (the graph's one chunk was already rank 1) |

- **No chunk found only by the graph made the fused top 10, for any question.** A chunk that only one retriever finds scores at most 1/61 ≈ 0.016. The 10th fused chunk usually scores about 0.028–0.030, the score of a chunk found by both dense and bm25 near rank 10. So the graph can only re-rank chunks the others already found; it cannot add new ones.
- **q11 shows the cost.** The graph found the hop-2 evidence (`service-catalog.md#3`, the `#oncall-analytics` row) at graph rank 8. Its graph score of 1/68 ≈ 0.015 is half the score of the 10th fused chunk (0.029), so it was dropped. The answer then says Analytics Engine without the channel.
- **RRF rewards agreement, not strength.** `support-runbook-errors.md#9` entered q11's top 5 at rank 4 with ranks 15, 16 and 19 in the three lists. It beat chunks ranked 1–3 by a single retriever.
- **Net effect:** the same hit@5 and recall@k as dense+bm25. The graph helps where it agrees with the others (q11 hop 1 moves to rank 1). It is drowned out where it is the only one that is right.

What we would try next: give the graph its own guaranteed slots in the context (for example its top 2 hop-2 chunks), or weight its list in RRF. We have not run either.

## 5. Switches, attribution, path view

- **Switches:** the "Retrievers" checkboxes in the sidebar (used by Ask and Evaluate), or `--retrievers dense bm25 graph` in the CLI. At least one must be on, otherwise the API returns an error. Graph is only offered where a `graph.json` exists for the corpus and chunk setting.
- **Attribution:** every retrieved chunk and every citation shows something like "found by dense #9, bm25 #1". The rank is the chunk's rank in that retriever's own list of 20, not the final rank.
- **Path view:** the Ask tab draws the paths of all graph chunks as one diagram: entities coloured by type, the question's entities on the left, arrows in the relation's real direction. It also lists them as text, for example `KST-5031 —raised_by→ postroom, then postroom —owned_by→ Messaging`.

## 6. New two_hop questions

We added q11 (KST-3012 → Analytics Engine → channel) and q12 (payment retries → ledger → Payments Core → channel), so there are 3 two_hop questions with q08. They are logged in `kestrel-kb/eval/CHANGES.md`.

| | dense | bm25 | dense+bm25 | graph | dense+bm25+graph |
|---|---|---|---|---|---|
| q08 hop 1 / hop 2 | 1 / — | 2 / — | 2 / — | 2 / — | 1 / — |
| q11 hop 1 / hop 2 | 2 / — | 1 / — | 1 / — | 6 / 8 | 1 / — |
| q12 hop 1 / hop 2 | 8 / — | 1 / 10 | 2 / 10 | — / — | 3 / 10 |

Hop 1 is always found. Hop 2 is never in the top 5 in any configuration: two_hop recall@5 is at most 0.50.

## 7. Answers (Kestrel, top 5, not hand-graded yet)

The status counts out of 12 questions come from the app. The grade fields in the answer files are still empty.

| run | answered | unverified | not_found |
|---|---|---|---|
| dense (`...minilm-answers-20261109-030112`) | 2 | 5 | 5 |
| dense+bm25 (`...dense+bm25-answers-20261109-030321`) | 5 | 4 | 3 |
| dense+bm25+graph (`...dense+bm25+graph-answers-20261109-095110`) | 4 | 4 | 4 |

- **q01:** with BM25 it goes from `not_found` (evidence not in the context) to answered.
- **q11:** with dense+bm25 it answers `not_found`. With the graph it is `unverified`: it names Analytics Engine but not the channel, as explained in section 4.
- **q03:** same five chunks in both runs, but `answered` in one and `not_found` in the other. Generation varies between runs, so one run per configuration is not enough to compare answers.

## 8. Decision log

### Added or changed in V3

| Decision | Chosen | Alternatives considered | Reason |
|---|---|---|---|
| BM25 storage | own postings tables in SQLite, one file | FTS5 (first version), rank_bm25 in memory, bm25s, Elasticsearch | Stays on disk and comes with Python. We can apply the course formula and k1/b exactly. Deleting one document is easy (bm25s can't). Elasticsearch is too heavy for 2 corpora. |
| BM25 statistics | computed per query over the whole index | stored, rebuilt on change | Adding or deleting a document needs no rebuild. |
| BM25 tokenizer | the Words tab tokenizer, stopwords removed from the question only | stemming, separate tokenizer | Identifiers stay whole. The Words tab explains BM25's results exactly. |
| Fusion | RRF, k = 60, 20 candidates per retriever | score normalization + weighted sum | Dense distances and BM25 scores are not comparable. Ranks are. 20 candidates lets a chunk outside one retriever's top 10 still get in. |
| Default retrievers | dense + bm25 | dense only, all three | Section 1: best recall@5 on both corpora. The graph adds nothing at k = 5 and exists only for Kestrel. |
| Graph schema | fixed: 9 entity types, 13 typed relations | open extraction | Validation can drop off-schema triples, and the types match what the questions ask (error → service → team → channel). |
| Graph model | gemini-3-flash-preview | gemini-3.5-flash-lite | flash-lite: 23 triples; flash: 222. |
| Provenance | chunk + quote per triple, quote checked | chunk only | A triple can be checked, and the graph returns chunks, not facts. |
| Heading context in the prompt | yes | chunk text only | Chunking drops short headings, so a chunk does not say which error or section it is about. |
| Entity resolution | same type + normalized name | fuzzy or embedding match, LLM merge | Predictable and free. The misses are listed in section 3. |
| Graph walk | 2 hops, both directions, closest first | 1 hop, 3 hops, PageRank | Covers error → team. Needs 3 hops for q08. A 3-hop walk would pull in more noise; not tested. |
| Build cache | per chunk, keyed by text hash | rebuild everything | Free-tier limit of 20 calls a day. Rebuilds only pay for changed chunks. |
| Graph in fusion | same RRF weight as the others | own slots, higher weight | Simplest. Section 4 shows its limit. |

### From V2, unchanged

Corpora and folders, stack (FastAPI + HTML/JS, Gemini via the OpenAI SDK), one Chroma collection per corpus × setting, squared L2 distance, blocking re-index, JSON answer with verified citations, the 5 answer statuses, no relevance cutoff, relevance rule (the chunk contains the evidence quote), HitRate@k and Recall@k, shared CLI/UI eval runner, LLM judge + `override` for the no-context test, HNSW kept as is. See the V2 report (`git show 3ede37d:REPORT.md`).

## 9. Known failures

| Failure | What the user sees | Status in V3 / next step |
|---|---|---|
| Exact identifiers | V2: KST-2044 lost to KST-2040 | **Fixed by BM25** (fused rank 3). Monte Cristo n03 "No. 27" is still missed by every retriever. Next: phrase or number matching. |
| Paraphrase in BM25 | BM25 alone misses q05, h03, h07 | Covered by dense in the fusion. No stemming or synonyms. |
| Second hop of two_hop questions | The answer names the team but not the channel (q08, q11) | Not fixed. The graph finds it for q11 but fusion drops it (section 4). q08 needs 3 hops. Next: graph slots in the context, 3-hop walk. |
| Graph extraction gaps | No graph results for KST-2044, versions, prices; q12 seeds the wrong entity | Schema and granularity limits (section 3). Next: allow "retries"-style features per service, re-extract the failed chunk. |
| Corrupted accents in extraction | Inès → `In8s` in the graph and the path view | Cause not confirmed. Next: check the raw API response, and normalize names back to the chunk's spelling. |
| Graph build quota | `support-runbook-errors.md#10` has no triples | 429 from the free tier. Re-run the build (it uses the cache). |
| Abstaining with evidence in context | `not_found` for q03 and q07 | Still there. q07 is the 2025 vs 2026 policy conflict; the model refuses rather than picking the newer one. |
| Answers vary between runs | Same context, different status (q03) | Generation is not repeatable. Compare answers over several runs. |
| Wrong answers with verified citations | A real quote that doesn't answer the question | Still there. The check proves the quote exists, not that it answers. |
| Approximate search | Dense metrics vary by about ±0.04 | Kept and documented. |
| Rate limits | `llm_error` in the app | Free tier. The CLI has `--delay`. |
| Settings reset on restart | Chunk settings go back to the default | Not persisted. |
| MiniLM truncation | Long chunks are cut at 256 tokens for embedding | Matters for chunk sizes above about 1,000 chars. |
| Old copies | `data/documents/` and old `chunks_*` collections | Unused. Can be deleted. |

---

# REPORT - RAG V4

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
