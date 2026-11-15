# Kestrel Analytics — Knowledge Base (RAG project corpus)

Corpus for the RAG project of *AI Systems Integration & Operations* (BCS S5, EPITA).

Everything is **fictional**: Kestrel Analytics is the same invented B2B dashboard company as in the Support Triage dataset. Names, e-mail addresses (`*.example`), IP addresses (`203.0.113.x`, a documentation range) and internal channels are invented. Because no LLM has seen these documents during training, a correct answer has to come from retrieval.

## Files

| File | Audience | Content |
|---|---|---|
| `docs/product-guide.md` | customers | plans and limits, roles, invitations, sign-in, 2FA, SSO, connectors, reports, embedding, data deletion, support |
| `docs/billing-policy-2025.md` | customers, support | billing and refund policy effective 1 January 2025 (**superseded**) |
| `docs/billing-policy-2026.md` | customers, support | billing and refund policy effective 1 March 2026 (**current**) |
| `docs/support-runbook-errors.md` | support (internal) | every `KST-xxxx` error code: cause, action, escalation |
| `docs/service-catalog.md` | engineering (internal) | services, owning teams, on-call channels, escalation rules |
| `eval/questions.starter.jsonl` | students | 10 starter questions with evidence quotes |

About 30,000 characters in total: roughly 40 chunks at 800 characters.

## What is hidden in the corpus

The documents are written so that the six challenging cases from the course show up naturally:

| Case | Where |
|---|---|
| Exact identifier | `KST-xxxx` codes; neighbouring codes (KST-2040 / KST-2044) have very similar descriptions |
| Answer split across chunks | failed payments (several paragraphs), SSO (plans table + setup section) |
| Different vocabulary | users say "phone", "log in", "locked out"; the docs say "authenticator device", "sign in", "second factor" |
| Not in the corpus | prices, mobile apps… The docs are close enough to tempt the model |
| Two versions of a document | the two billing policies are nearly identical, with different numbers (refund window, discount, grace period, retries) |
| Two-hop | error code → service (runbook), then service → team and channel (service catalog) |


## Question format (JSONL, one question per line)

```json
{"id": "q06",
 "question": "Can I get a refund on my annual plan 45 days after it renewed?",
 "type": "versions",
 "answerable": true,
 "evidence": [
   {"source": "billing-policy-2026.md", "quote": "within 60 days of the purchase or renewal date"},
   {"source": "billing-policy-2026.md", "quote": "pro-rated from the date of the request"}
 ],
 "expected_answer": "Yes, under the current policy: within 60 days, pro-rated from the request date.",
 "notes": "The 2025 policy says 30 days."}
```

- `type`: one of `exact_identifier`, `split_answer`, `vocabulary`, `not_in_corpus`, `versions`, `two_hop`, `other`.
- `evidence`: the source file and a **short, exact quote** from it. A retrieved chunk is relevant if it comes from that source and contains the quote. Keep quotes short (one clause) so they are unlikely to be cut by a chunk boundary.
- Unanswerable questions: `"answerable": false` and an empty `evidence` list.
- `notes`: optional, why the question is hard.


## Your job

1. Use the **10 starter questions** as your Kestrel question set. They cover the six challenging cases, including 2 unanswerable questions. Your own corpus needs its own set of **at least 10 questions**, in the same format.
2. Run the **no-context test**: ask every question to the LLM without retrieval. Flag those it answers correctly.
3. Write the evaluation that calls your `/api/search` and measures the relevance of the retrieved chunks, with a measure of your choice, on the top k chunks for k = 1, 3, 5, 10, overall and per type.
4. Keep the question set **fixed** while you compare versions of your pipeline. Add questions; never edit old ones silently.