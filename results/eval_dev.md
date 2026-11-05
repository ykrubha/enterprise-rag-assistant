Corpus: 186 documents, 2902 chunks. Split `dev`: 55 answerable and 18 unanswerable questions.

**Retrieval (document level)**

| Method | Hit@1 | Hit@3 | Hit@5 | MRR@10 (95% CI) | nDCG@10 | Key fact in top 5 |
|---|---:|---:|---:|---:|---:|---:|
| BM25 | 0.36 | 0.55 | 0.65 | 0.50 (0.39-0.61) | 0.56 | 0.69 |
| Embeddings | 0.49 | 0.71 | 0.75 | 0.60 (0.50-0.71) | 0.64 | 0.75 |
| Hybrid (RRF) | 0.55 | 0.71 | 0.78 | 0.64 (0.53-0.74) | 0.68 | 0.82 |
| **Hybrid + reranker** | 0.56 | 0.78 | 0.84 | 0.69 (0.59-0.79) | 0.73 | 0.84 |

**Answers (`extractive` generator, reranked retrieval)**

| Metric | Score |
|---|---:|
| Answer rate on answerable questions | 0.98 |
| Key fact recall, all answerable questions | 0.42 |
| Key fact recall, when it answered | 0.43 |
| Answers citing a correct document | 0.69 |
| Citation precision | 0.52 |
| Unanswerable questions correctly declined | 0.89 |
| Declines that were correct | 0.94 |
| Latency p50 / p95 (ms, extractive) | 4769.3 / 6839.1 |
