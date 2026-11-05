Corpus: 186 documents, 2902 chunks. Split `test`: 40 answerable and 12 unanswerable questions.

**Retrieval (document level)**

Not rerun for this generator.

**Answers (`gemini-3.5-flash-lite` generator, reranked retrieval)**

| Metric | Score |
|---|---:|
| Answer rate on answerable questions | 0.88 |
| Key fact recall, all answerable questions | 0.75 |
| Key fact recall, when it answered | 0.86 |
| Answers citing a correct document | 0.91 |
| Citation precision | 0.80 |
| Unanswerable questions correctly declined | 1.00 |
| Declines that were correct | 0.71 |
| Latency p50 / p95 (ms, gemini-3.5-flash-lite) | 3869.7 / 41448.8 |

Why answerable questions were declined: {'no_citation': 1, 'low_support': 1, 'model_declined': 3}
