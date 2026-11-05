Corpus: 186 documents, 2902 chunks. Split `test`: 40 answerable and 12 unanswerable questions.

**Retrieval (document level)**

| Method | Hit@1 | Hit@3 | Hit@5 | MRR@10 (95% CI) | nDCG@10 | Key fact in top 5 |
|---|---:|---:|---:|---:|---:|---:|
| BM25 | 0.42 | 0.62 | 0.72 | 0.54 (0.41-0.67) | 0.59 | 0.80 |
| Embeddings | 0.57 | 0.72 | 0.82 | 0.67 (0.54-0.79) | 0.70 | 0.78 |
| Hybrid (RRF) | 0.70 | 0.80 | 0.88 | 0.77 (0.65-0.87) | 0.78 | 0.78 |
| **Hybrid + reranker** | 0.55 | 0.85 | 0.90 | 0.70 (0.59-0.81) | 0.74 | 0.85 |

**Answers (`extractive` generator, reranked retrieval)**

| Metric | Score |
|---|---:|
| Answer rate on answerable questions | 0.97 |
| Key fact recall, all answerable questions | 0.60 |
| Key fact recall, when it answered | 0.62 |
| Answers citing a correct document | 0.72 |
| Citation precision | 0.48 |
| Unanswerable questions correctly declined | 0.75 |
| Declines that were correct | 0.90 |
| Latency p50 / p95 (ms, extractive) | 4409.6 / 6580.4 |
