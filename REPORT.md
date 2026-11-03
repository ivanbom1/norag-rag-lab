# REPORT — RAG project

<!--
SKELETON. The tables are generated from eval/results/ and the "Notes" are facts from the runs.
Every "✍️" marks text you write yourself. Delete the notes and these comments once written.
-->

## Team

✍️ Name, e-mail, (solo).

---

## 1. Corpora and question sets

| Corpus | Documents | Size | Chunks (800 / 10 %) | Question set |
|---|---|---|---|---|
| Kestrel KB | 5 `.md` files in `kestrel-kb/docs/` | ~29.7k chars | 42 | `kestrel-kb/eval/questions.starter.jsonl`, 10 questions (8 answerable) |
| Monte Cristo | 5 volume files in `corpus/` | ~2.6M chars | 3,613 | `eval/questions/monte-cristo.jsonl`, 33 questions (28 answerable) |

✍️ Why this corpus, and what makes its questions hard.

> Notes:
> - Planted facts (stationer, ledger, seal split over vol1/vol2) and two contradictions with the original book (25th vs 24th February; ruby vs diamond). They tell a grounded answer from the model's memory.
> - The V1 set was converted to Kestrel JSONL. All changes are logged in `eval/questions/CHANGES.md`: types mapped, quotes shortened, `alternatives` added.
> - Added in V2: n01 and n02 (`two_hop`) and n03 (`exact_identifier`). The set now covers 5 of Kestrel's 6 cases; `versions` is missing.

## 2. No-context test

| Corpus | Answerable questions | Answered correctly without retrieval (LLM judge) |
|---|---|---|
| Kestrel | 8 | 0 |
| Monte Cristo | 28 | 21: d01–d11, h01–h03, h05–h07, s03, n01–n03 |

Results are in `eval/results/no-context-<corpus>.jsonl`. The judge is a second Gemini call that compares the answer with `expected_answer`. Human corrections go in the `override` field.

✍️ Your review of the judge's verdicts. Did you override any?
✍️ Your decision for the 21 flagged questions: replace them, or keep them and explain why.

> Notes:
> - Not flagged: h04, p01, p02, s01, s02, c01, c02. These are the planted facts and the contradictions.
> - Without context the model said "28 February" and "a large diamond". The corpus says the 25th and a ruby.
> - It answered unanswerable questions from memory: u02 "Kevin Reynolds"; u04 invented a cook named "Pamphile".
> - The judge looks lenient on h02: "a heavy cannonball" was accepted where the expected answer is a thirty-six-pound shot.

## 3. Retrieval evaluation

**Measure.** A retrieved chunk is relevant if it comes from the evidence `source` and contains the evidence `quote`, ignoring whitespace (Kestrel's rule). Monte Cristo items can also list `alternatives`.

- **HitRate@k:** at least one evidence item is in the top k.
- **Recall@k:** the share of evidence items found in the top k.

Unanswerable questions are excluded.

✍️ Why these two measures.

### Results: both corpora × configurations, per type

Embedding all-MiniLM-L6-v2, squared L2 distance, top 10 retrieved; min paragraph 20 chars.

#### Kestrel

| config | type | n | hit@1 | hit@3 | hit@5 | hit@10 | recall@1 | recall@3 | recall@5 | recall@10 |
|---|---|---|---|---|---|---|---|---|---|---|
| 800 / 10 % | **overall** | 8 | 0.62 | 0.75 | 0.88 | 1.00 | 0.41 | 0.69 | 0.81 | 0.94 |
| 800 / 10 % | exact_identifier | 2 | 0.50 | 0.50 | 0.50 | 1.00 | 0.25 | 0.50 | 0.50 | 1.00 |
| 800 / 10 % | split_answer | 2 | 1.00 | 1.00 | 1.00 | 1.00 | 0.62 | 1.00 | 1.00 | 1.00 |
| 800 / 10 % | vocabulary | 1 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 800 / 10 % | versions | 2 | 0.00 | 0.50 | 1.00 | 1.00 | 0.00 | 0.50 | 1.00 | 1.00 |
| 800 / 10 % | two_hop | 1 | 1.00 | 1.00 | 1.00 | 1.00 | 0.50 | 0.50 | 0.50 | 0.50 |
| 400 / 10 % | **overall** | 8 | 0.50 | 0.75 | 0.75 | 0.75 | 0.28 | 0.59 | 0.66 | 0.72 |
| 400 / 10 % | exact_identifier | 2 | 0.50 | 0.50 | 0.50 | 0.50 | 0.25 | 0.50 | 0.50 | 0.50 |
| 400 / 10 % | split_answer | 2 | 0.50 | 1.00 | 1.00 | 1.00 | 0.12 | 0.38 | 0.62 | 0.88 |
| 400 / 10 % | vocabulary | 1 | 1.00 | 1.00 | 1.00 | 1.00 | 0.50 | 1.00 | 1.00 | 1.00 |
| 400 / 10 % | versions | 2 | 0.50 | 1.00 | 1.00 | 1.00 | 0.50 | 1.00 | 1.00 | 1.00 |
| 400 / 10 % | two_hop | 1 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

#### Monte Cristo

| config | type | n | hit@1 | hit@3 | hit@5 | hit@10 | recall@1 | recall@3 | recall@5 | recall@10 |
|---|---|---|---|---|---|---|---|---|---|---|
| 800 / 10 % | **overall** | 28 | 0.29 | 0.50 | 0.50 | 0.64 | 0.27 | 0.46 | 0.46 | 0.59 |
| 800 / 10 % | other | 15 | 0.40 | 0.53 | 0.53 | 0.60 | 0.40 | 0.53 | 0.53 | 0.60 |
| 800 / 10 % | vocabulary | 7 | 0.14 | 0.57 | 0.57 | 0.71 | 0.14 | 0.57 | 0.57 | 0.71 |
| 800 / 10 % | split_answer | 3 | 0.33 | 0.67 | 0.67 | 1.00 | 0.17 | 0.33 | 0.33 | 0.50 |
| 800 / 10 % | two_hop | 2 | 0.00 | 0.00 | 0.00 | 0.50 | 0.00 | 0.00 | 0.00 | 0.50 |
| 800 / 10 % | exact_identifier | 1 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 400 / 10 % | **overall** | 28 | 0.25 | 0.43 | 0.57 | 0.68 | 0.23 | 0.39 | 0.52 | 0.62 |
| 400 / 10 % | other | 15 | 0.33 | 0.53 | 0.67 | 0.80 | 0.33 | 0.53 | 0.67 | 0.80 |
| 400 / 10 % | vocabulary | 7 | 0.14 | 0.29 | 0.43 | 0.57 | 0.14 | 0.29 | 0.43 | 0.57 |
| 400 / 10 % | split_answer | 3 | 0.33 | 0.67 | 0.67 | 0.67 | 0.17 | 0.33 | 0.33 | 0.33 |
| 400 / 10 % | two_hop | 2 | 0.00 | 0.00 | 0.50 | 0.50 | 0.00 | 0.00 | 0.25 | 0.25 |
| 400 / 10 % | exact_identifier | 1 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

The source files are `eval/results/<corpus>_<size>_10_20_minilm.json`.

### What changed between configurations, and what moved

✍️ What you changed and why; what moved and your explanation.

> Notes:
> - The only change was the chunk size: 800 → 400 characters (overlap 10 %, top 10, same embedding).
> - **Kestrel:** hit@5 0.88 → 0.75 and recall@10 0.94 → 0.72. The two_hop question q08 is lost entirely, and split_answer recall@1 drops from 0.62 to 0.12.
> - **Monte Cristo:** hit@5 0.50 → 0.57. The `other` type improved (0.53 → 0.67), while vocabulary (0.57 → 0.43) and split_answer recall@10 (0.50 → 0.33) got worse.
> - Chroma's search is approximate, so repeated runs on the same index vary by about ±0.04 at hit@5. With 8 Kestrel questions, one question is worth 0.125.

## 4. Answer evaluation (hand-graded)

Kestrel set, best configuration (800 / 10 %), answers generated with top-5. File: `eval/results/kestrel_800_10_20_minilm-answers-20261103-115712.json`. Grades go in its `grade` and `failure` fields.

### Grading rules (written before grading)

✍️ Your definitions of correct / partial / wrong / unsupported, and of retrieval vs generation failure.

### Grades

| id | type | app status | evidence in the answer's top-5 | grade | failure |
|---|---|---|---|---|---|
| q01 | exact_identifier | not_found | no | ✍️ | ✍️ |
| q02 | exact_identifier | answered | yes | ✍️ | ✍️ |
| q03 | split_answer | not_found | yes | ✍️ | ✍️ |
| q04 | split_answer | answered | yes | ✍️ | ✍️ |
| q05 | vocabulary | answered | yes | ✍️ | ✍️ |
| q06 | versions | unverified | yes | ✍️ | ✍️ |
| q07 | versions | not_found | yes | ✍️ | ✍️ |
| q08 | two_hop | unverified | yes (hop 1 only) | ✍️ | ✍️ |
| q09 | not_in_corpus | not_found | — | ✍️ | ✍️ |
| q10 | not_in_corpus | not_found | — | ✍️ | ✍️ |

✍️ What the grades show.

## 5. Two misses diagnosed with the Words tab

### Kestrel: q01 "What does error KST-2044 mean and how do I fix it?"

✍️ What you saw, and your explanation.

> Notes (800 / 10 %, question mode):
> - The evidence chunk is `support-runbook-errors.md#4`, found at rank 6.
> - Rank 1 is `#3`, the chunk holding the neighbouring code **KST-2040** (timeout).
> - `kst-2044` occurs once in the corpus, in exactly one chunk, the evidence chunk.
> - "error" is in 9 of 42 chunks but not in the evidence chunk. "mean" and "fix" are absent from the corpus.

### Monte Cristo: n03 "Who was the prisoner No. 27 in the Château d’If?"

✍️ What you saw, and your explanation.

> Notes (800 / 10 %, question mode):
> - The evidence chunks are `vol1#330` and `vol5#452`. Neither is in the top 10.
> - "27" is in 12 of 3,613 chunks, including both evidence chunks. Its other uses are "No. 27, Rue du Helder" (Albert's address) and dates.
> - "prisoner" (90 chunks), "château" (48) and "d’if" (41) are spread everywhere. "château" and "d’if" are not in the evidence chunks.
> - The top 5 are generic prison passages (vol1#195, #534, #220, #582, vol2#10).

## 6. Decision log

| Decision | Chosen | Alternatives considered | Reason |
|---|---|---|---|
| Corpus | Monte Cristo, 5 volume files, planted facts + 2 contradictions | other books, one file | ✍️ |
| Stack | FastAPI + plain HTML/JS, Gemini via OpenAI SDK | framework frontend | ✍️ |
| Vector store layout | one Chroma collection per corpus × chunk setting | one collection with metadata filters | ✍️ |
| Distance | Chroma default squared L2 (ranking = cosine on normalized vectors) | cosine space | ✍️ |
| Re-index on settings change | blocking; already-indexed settings are instant | background job | ✍️ |
| LLM output | JSON schema + Pydantic `{answerable, answer, citations[{chunk_id, quote}]}`; quote must be in the cited chunk | free text | ✍️ |
| Statuses | answered / unverified (shown with warning) / not_found / invalid_response / llm_error | hide unverified answers | ✍️ |
| Relevance cutoff | none (distances logged) | distance threshold | ✍️ |
| Two corpora | corpus selector, `data/corpora/<name>/` | one folder + checkboxes | ✍️ |
| Relevance rule | chunk from source contains quote (Kestrel) | V1 span overlap | ✍️ |
| Measures | HitRate@k and Recall@k, k = 1, 3, 5, 10 | MRR, precision | ✍️ |
| Monte Cristo set | converted, change log, +3 questions | convert only | ✍️ |
| Eval runner | shared code for CLI and UI endpoint (in-process) | CLI over HTTP | ✍️ |
| No-context judging | LLM judge + human `override` | by hand only, keywords | ✍️ |
| Tokenizer (Words) | lowercase, IDs kept whole (`kst-2044`), no stemming, stopwords hidden | letters only, stemming | ✍️ |
| HNSW non-determinism | kept, documented | higher `ef_search`, exact search | ✍️ |

## 7. Known failures

✍️ For each failure: what the user sees, and what you would do in the next version.

> Notes:
> - **Exact identifiers.** KST-2044 loses to KST-2040; "No. 27" loses to generic prison text. The embedding doesn't weight the identifier.
> - **Wrong answers with verified citations.** The citation check proves a quote exists, not that it answers the question (V1: d06 "paralysis and madness").
> - **Abstaining with the evidence in context.** Kestrel q03 and q07 returned `not_found` although their evidence was retrieved.
> - **Approximate search.** Chroma HNSW top-k changes between processes, so the metric varies by about ±0.04.
> - **Rate limits.** The Gemini free tier returns 429; no-context runs were resumed. The app shows `llm_error`.
> - **Settings reset on restart.** Settings are not persisted.
> - **MiniLM truncation.** The 256-token input limit could truncate large chunk sizes.
> - **Old copies.** `data/documents/` and the old `chunks_*` Chroma collections from V1 are unused.
