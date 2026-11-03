# Monte Cristo question set: changes

## V2 (conversion of `eval/questions.json` to `monte-cristo.jsonl`)

The V1 file is kept unchanged as the record of the V1 baseline.

- **Format.** Kestrel JSONL: `source`/`quote` instead of `file`/`quote`, plus `answerable`, `expected_answer` and `notes`. The V1 keyword lists (`expected`) were replaced by a written `expected_answer`.
- **Question text.** Unchanged for all 30 V1 questions.
- **Types.** Mapped to Kestrel's labels. The V1 type is kept in `notes`.
  - hard → `vocabulary`
  - split → `split_answer`
  - unanswerable → `not_in_corpus`
  - direct, planted, contradiction → `other`
- **Evidence alternatives.** V1 listed alternative locations for one evidence part, so several passages could each satisfy it. In V2 the first location is the evidence item, and the others go into an optional `alternatives` list on that item. A chunk matching the item or any of its alternatives is relevant to that item.
- **Relevance rule.** V1 counted a chunk as relevant when it overlapped the evidence span. V2 uses Kestrel's rule: the chunk comes from the source and contains the quote, ignoring whitespace differences. To keep quotes from being cut by chunk boundaries, these long quotes were shortened to the clause holding the answer:
  - d04, d05, d06, d07, d11, h02, h03, h04, h05, h06, c02: an alternative quote was shortened.
  - p01: one quote became two items (stationer, watermark), so the planted paragraph counts as two facts for recall.
  - p02, s01: quotes shortened to the answer clause.
- **Added in V2** (new ids, nothing removed):
  - n01 `two_hop`
  - n02 `two_hop`
  - n03 `exact_identifier`

  With these, the set covers 5 of Kestrel's 6 cases: exact_identifier, split_answer, vocabulary, not_in_corpus and two_hop. The missing case is versions.
