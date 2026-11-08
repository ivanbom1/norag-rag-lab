# REPORT - RAG V3

Team: Tristan and Ivan

## 1. BM25

BM25 on both corpora. Index is a SQLite FTS5 file (bm25_db/bm25.sqlite3), one table per corpus and chunk setting. It gets built at the same time as Chroma, with the same chunk ids. Old Chroma indexes get copied over at startup. Delete removes from both.

We picked FTS5 because it comes with Python and stays on disk. Other options were rank_bm25 in memory (the naive one), bm25s (can't delete one doc) and Elasticsearch (too much for this).

Same tokenizer as the Words tab, so KST-2044 stays one word. Stopwords removed from the question. Default k1/b.

| hit@5 | dense | bm25 |
|---|---|---|
| kestrel | 0.80 | 0.90 |
| monte-cristo | 0.54 | 0.57 |

Kestrel numbers include q11 and q12. BM25 fixes q01 (KST-2044) but misses q05 (lost phone, different words). Still misses n03 (No. 27).

## 2. Graph extraction

## 3. Graph retriever

## 4. RRF fusion

RRF with k = 60, the usual value. Each retriever gives its top 20 to the fusion and then we keep the top k. So a chunk that is 12th for dense but 3rd for bm25 can still get in. Code is in app/fusion.py.

Default is dense + bm25.

## 5. On/off switches

Checkboxes in the sidebar ("Retrievers"), used by Ask and Evaluate. In the eval command it's --retrievers, e.g. --retrievers dense bm25. At least one has to be on, otherwise the API says no.

## 6. Attribution

Every retrieved chunk and every citation in an answer shows something like "found by dense #9, bm25 #1". The rank is the chunk's rank in that retriever's own list (out of 20), not the final rank.

## 7. Graph path view

## 8. Evaluation

## 9. New two_hop questions

Added q11 (KST-3012 -> Analytics Engine -> channel) and q12 (payment retries -> ledger -> Payments Core -> channel). Both are in kestrel-kb/eval/questions.starter.jsonl, change noted in kestrel-kb/eval/CHANGES.md. So 3 two_hop with q08.

With BM25 the first hop is found every time (rank 1 to 3) but the second hop mostly isn't. q08 and q11 never get it, q12 gets it at rank 7. That's what the graph should fix.

## 10. Extraction quality check
