# RepoBench-R Retrieval Benchmark Report

**Dataset**: `tianyang/repobench_python_v1.1`  · **Paper**: [RepoBench arXiv:2306.03091](https://arxiv.org/abs/2306.03091)  · **Samples**: 1,000 per split

> **Metric definitions for RepoBench-R** (single gold snippet per sample):
> - **Exact Match (EM)**: top-1 retrieved snippet is the correct one. Mathematically = Recall@1.
> - **Recall@k**: correct snippet appears anywhere in top-k results.
> - **MRR**: Mean Reciprocal Rank = average of 1/rank(gold) across all samples.
> - EM = Recall@1 because each sample has exactly one gold snippet.

---

## Split: `cross_file_first`

| Retrieval Strategy | Exact Match (EM) | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **BM25 Sparse Search** | **33.3%** | **33.3%** | **84.6%** | **96.0%** | **99.8%** | **0.5957** |
| **Dense Semantic Search (all-MiniLM-L6-v2)** | **34.5%** | **34.5%** | **85.5%** | **96.4%** | **99.7%** | **0.6049** |
| **Hybrid RRF (BM25 + Dense)** | **33.2%** | **33.2%** | **85.2%** | **95.9%** | **99.8%** | **0.5982** |
| BM25 + GraphRAG (Ours) | 26.0% | 26.0% | 80.5% | 94.0% | 99.6% | 0.5430 |
| Hybrid + GraphRAG (Ours) | 19.4% | 19.4% | 79.1% | 93.7% | 99.7% | 0.5012 |

## Split: `cross_file_random`

| Retrieval Strategy | Exact Match (EM) | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **BM25 Sparse Search** | **43.3%** | **43.3%** | **88.3%** | **97.7%** | **99.8%** | **0.6604** |
| Dense Semantic Search (all-MiniLM-L6-v2) | 37.8% | 37.8% | 86.1% | 96.5% | 99.7% | 0.6257 |
| Hybrid RRF (BM25 + Dense) | 42.4% | 42.4% | 87.3% | 97.5% | 99.8% | 0.6528 |
| **BM25 + GraphRAG (Ours)** | **47.0%** | **47.0%** | **90.4%** | **97.4%** | **99.9%** | **0.6886** |
| **Hybrid + GraphRAG (Ours)** | **49.3%** | **49.3%** | **90.0%** | **96.5%** | **99.9%** | **0.6972** |

---

## Architecture Clarifications

### Dense Semantic Search
Vector-based retrieval using `all-MiniLM-L6-v2` sentence-transformer.  
Query and corpus snippets are embedded into 384-dim dense vectors; ranked by **cosine similarity**.
Unlike sparse (BM25) search, it captures semantic meaning without requiring exact keyword matches.

### BM25 + GraphRAG (Ours)
A two-stage retrieval pipeline:
1. **BM25** scores each candidate snippet vs the query (import_statement + last 5 lines of cropped_code).
2. **Code graph** is built from the RepoBench row fields:
   - `IMPORTS` edge: snippet identifier/module appears in `import_statement`
   - `CALLS` edge: snippet identifier (≥4 chars) appears in `cropped_code` body
   - `CO-FILE` edge: snippets sharing the same file path
3. **Spread-activation rerank**: BM25 scores propagate along graph edges.
   `final_score = α * bm25_norm + β * (direct_graph_signal + 0.3 * neighbour_bm25_avg)`
   This makes the ranking **genuinely distinct** from pure BM25 or Hybrid.

### Hybrid + GraphRAG (Ours)
Same as above but uses **RRF-fused (BM25+Dense)** scores as the base instead of raw BM25,
then applies the same spread-activation graph reranking.

### Why did previous GraphRAG ≈ Hybrid RRF?
The prior implementation used Hybrid's top-3 as seeds, expanded a few graph neighbours,
then **filled remaining slots back from the original Hybrid order** — producing nearly
identical rankings. The new implementation uses independent graph scoring (α/β weighted
spread-activation) that is mathematically decoupled from the Hybrid ranking.

---

## Academic Paper References

- **RepoBench**: Tianyang Liu, Canwen Xu, Julian McAuley — [arXiv:2306.03091](https://arxiv.org/abs/2306.03091)
- **SWE-bench**: Carlos E. Jimenez et al. — [arXiv:2310.06770](https://arxiv.org/abs/2310.06770)
- **MemGPT**: Charles Packer et al. — [arXiv:2310.08560](https://arxiv.org/abs/2310.08560)
- **MemTree**: Alireza Rezazadeh et al. — [arXiv:2410.14052](https://arxiv.org/abs/2410.14052)
- **MemORAI** — [arXiv:2605.01386](https://arxiv.org/abs/2605.01386)
- **Multi-Session Chat**: Jing Xu, Arthur Szlam, Jason Weston — [arXiv:2107.07567](https://arxiv.org/abs/2107.07567)
