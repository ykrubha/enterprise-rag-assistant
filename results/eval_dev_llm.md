Corpus: 186 documents, 2902 chunks. Split `dev`: 55 answerable and 18 unanswerable questions.

**Retrieval (document level)**

Not rerun for this generator.

**Answers (`gemini-3.5-flash-lite` generator, reranked retrieval)**

| Metric | Score |
|---|---:|
| Answer rate on answerable questions | 0.84 |
| Key fact recall, all answerable questions | 0.75 |
| Key fact recall, when it answered | 0.89 |
| Answers citing a correct document | 0.87 |
| Citation precision | 0.66 |
| Unanswerable questions correctly declined | 1.00 |
| Declines that were correct | 0.67 |
| Latency p50 / p95 (ms, gemini-3.5-flash-lite) | 13398.3 / 50045.9 |

Why answerable questions were declined: {'model_declined': 8, 'low_support': 1}
