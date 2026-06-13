"""
RepoBench-R Retrieval Benchmark
================================
Evaluates 5 retrieval strategies on tianyang/repobench_python_v1.1:
  1. BM25 Sparse Search
  2. Dense Semantic Search  (all-MiniLM-L6-v2 + cosine sim)
  3. Hybrid RRF             (BM25 + Dense fused via Reciprocal Rank Fusion)
  4. BM25 + GraphRAG        (BM25 seeds → graph spread-activation rerank)
  5. Hybrid + GraphRAG      (Hybrid seeds → graph spread-activation rerank)

Metric: Acc@k (= Recall@k when there is exactly 1 gold snippet).
Matches RepoBench paper (arXiv:2306.03091) standard evaluation.

Key design decisions:
  - Graph edges derived from (a) import path overlap, (b) identifier
    mentions in cropped_code, (c) co-file (same path) sharing.
    We use WORD-level matching with a minimum identifier length of 4
    to avoid false positives from short tokens like "os", "re", "io".
  - GraphRAG uses spread-activation: initial BM25 scores are
    propagated along graph edges and combined as a final score,
    producing a ranking that is genuinely different from BM25/Hybrid.
"""

import os
import re
import math
import json
from collections import defaultdict
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util

# ============================================================================
# BM25 Pure Python Implementation
# ============================================================================
class BM25:
    def __init__(self, corpus, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus)
        self.avgdl = sum(len(d) for d in corpus) / self.corpus_size if self.corpus_size > 0 else 1.0
        self.doc_freqs = []
        self.idf = {}
        self.doc_len = []

        dfs = defaultdict(int)
        for doc in corpus:
            self.doc_len.append(len(doc))
            freq_map = defaultdict(int)
            for word in doc:
                freq_map[word] += 1
            self.doc_freqs.append(freq_map)
            for word in freq_map:
                dfs[word] += 1

        for word, df in dfs.items():
            self.idf[word] = math.log(1 + (self.corpus_size - df + 0.5) / (df + 0.5))

    def get_scores(self, query):
        scores = []
        for i in range(self.corpus_size):
            score = 0.0
            freq_map = self.doc_freqs[i]
            dl = self.doc_len[i]
            for word in query:
                if word not in self.idf:
                    continue
                tf = freq_map.get(word, 0)
                idf = self.idf[word]
                score += idf * (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            scores.append(score)
        return scores


def tokenize_code(code_str):
    if not code_str:
        return []
    return [w.lower() for w in re.findall(r'\b[a-zA-Z_]\w{2,}\b', code_str)]


# ============================================================================
# Reciprocal Rank Fusion (RRF)
# ============================================================================
def reciprocal_rank_fusion(ranked_lists, k=60):
    """Fuse multiple ranked lists using RRF. ranked_lists = list of lists of indices."""
    scores = defaultdict(float)
    for ranked in ranked_lists:
        for rank, idx in enumerate(ranked):
            scores[idx] += 1.0 / (k + rank + 1)
    return sorted(scores.keys(), key=lambda x: scores[x], reverse=True)


# ============================================================================
# Code Dependency Graph (GraphRAG Style)
# ============================================================================
def build_code_graph(row):
    """
    Build a local code property graph from RepoBench row.

    Nodes: integer indices 0..N-1 for each snippet in context.
           'current' = the file being completed.
    Edges (undirected):
      - IMPORTS: snippet's module path segment appears in import_statement
      - CALLS:   snippet's identifier (≥4 chars) is used in cropped_code
      - CO-FILE: snippets share the same file path

    Returns:
        adj: dict[int -> set[int]], neighbour sets (no 'current' as int)
        graph_scores: dict[int -> float], raw graph signal for each snippet
                      (number of edges connecting it to 'current')
    """
    N = len(row["context"])
    adj = defaultdict(set)  # snippet_idx -> set of neighbour snippet_idxs
    graph_signal = defaultdict(float)  # snippet_idx -> edge weight to 'current'

    imports = (row.get("import_statement") or "").lower()
    cropped = (row.get("cropped_code") or "").lower()

    # Extract import module names (words from import statements)
    import_words = set(re.findall(r'\b[a-zA-Z_]\w{3,}\b', imports))
    # Extract word tokens from cropped code
    cropped_words = set(re.findall(r'\b[a-zA-Z_]\w{3,}\b', cropped))

    for idx, cand in enumerate(row["context"]):
        ident = (cand.get("identifier") or "").lower().strip()
        path = (cand.get("path") or "").lower().strip()

        # Derive module name from path: last component without extension
        # e.g. "src/utils/helpers.py" -> "helpers"
        path_module = ""
        if path:
            basename = path.split("/")[-1].split("\\")[-1]
            path_module = basename.replace(".py", "").strip()

        # --- Edge: IMPORTS (current -> snippet) ---
        # snippet's identifier is explicitly imported
        if ident and len(ident) >= 4 and ident in import_words:
            graph_signal[idx] += 2.0  # stronger signal
        # snippet's module (from path) appears in imports
        if path_module and len(path_module) >= 4 and path_module in import_words:
            graph_signal[idx] += 1.5

        # --- Edge: CALLS/USES (current -> snippet) ---
        # snippet's identifier is referenced in the current file body
        if ident and len(ident) >= 4 and ident in cropped_words:
            graph_signal[idx] += 1.5

    # --- Edge: CO-FILE (snippet <-> snippet) ---
    path_to_indices = defaultdict(list)
    for idx, cand in enumerate(row["context"]):
        path = (cand.get("path") or "").lower().strip()
        if path:
            path_to_indices[path].append(idx)

    for path, indices in path_to_indices.items():
        if len(indices) > 1:
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    adj[indices[i]].add(indices[j])
                    adj[indices[j]].add(indices[i])

    return adj, graph_signal


def graphrag_rerank(bm25_scores, adj, graph_signal, N, alpha=0.4, beta=0.6, top_seeds=5):
    """
    GraphRAG re-ranking via spreading activation.

    Algorithm:
      1. Normalise BM25 scores to [0, 1].
      2. For each candidate, compute a graph_bonus:
           - Direct bonus from graph_signal (imported / called by current file)
           - Neighbour bonus: if a neighbour of this node has high BM25 score,
             propagate some of that score here (co-file / adjacency)
      3. Final score = alpha * bm25_norm + beta * graph_bonus

    This produces a genuinely different ranking from pure BM25.
    """
    if N == 0:
        return []

    # 1. Normalise BM25
    max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1.0
    bm25_norm = [s / max_bm25 for s in bm25_scores]

    # 2. Graph bonus
    # Direct signal from current file
    max_signal = max(graph_signal.values()) if graph_signal else 1.0
    if max_signal == 0:
        max_signal = 1.0

    final_scores = []
    for idx in range(N):
        direct = graph_signal.get(idx, 0.0) / max_signal

        # Neighbour spreading: average BM25 norm of graph neighbours
        neighbours = adj.get(idx, set())
        neighbour_bonus = 0.0
        if neighbours:
            neighbour_bonus = sum(bm25_norm[n] for n in neighbours if n < N) / len(neighbours)

        graph_bonus = direct + 0.3 * neighbour_bonus
        final_scores.append(alpha * bm25_norm[idx] + beta * graph_bonus)

    ranked = sorted(range(N), key=lambda i: final_scores[i], reverse=True)
    return ranked


# ============================================================================
# Evaluator
# ============================================================================
def evaluate_retriever(rankings, gold_indices):
    """
    Compute Exact Match, Recall@k, and MRR for a single-gold retrieval task.

    Since each sample has exactly ONE correct snippet (gold_snippet_index),
    the metrics relate as follows:
      - Exact Match (EM)  = Recall@1 = fraction where gold is ranked #1
      - Recall@k          = fraction where gold appears in top-k results
      - MRR               = mean of 1/rank(gold) across all samples
    """
    em = r1 = r3 = r5 = r10 = 0
    mrr = 0.0
    total = len(rankings)

    for ranks, gold in zip(rankings, gold_indices):
        try:
            rank = ranks.index(gold) + 1  # 1-indexed
        except ValueError:
            rank = 99999

        if rank == 1:
            em  += 1   # Exact Match = top-1 is correct
            r1  += 1
        if rank <= 3:
            r3  += 1
        if rank <= 5:
            r5  += 1
        if rank <= 10:
            r10 += 1
        mrr += 1.0 / rank if rank <= 100 else 0.0

    return {
        "exact_match": em  / total,   # EM = Recall@1
        "recall_at_1": r1  / total,
        "recall_at_3": r3  / total,
        "recall_at_5": r5  / total,
        "recall_at_10": r10 / total,
        "mrr":         mrr / total,
    }


# ============================================================================
# Debug helper: show a few examples to verify graph quality
# ============================================================================
def debug_sample_graph(rows, n_show=5):
    print("\n=== DEBUG: Sample graph signal on first few rows ===")
    has_signal = 0
    for i, row in enumerate(rows[:50]):
        adj, graph_signal = build_code_graph(row)
        gold = row["gold_snippet_index"]
        gold_sig = graph_signal.get(gold, 0.0)
        total_signal = sum(graph_signal.values())
        if total_signal > 0:
            has_signal += 1
        if i < n_show:
            print(f"  Row {i}: gold={gold}, gold_graph_signal={gold_sig:.2f}, "
                  f"total_signal={total_signal:.2f}, "
                  f"imports={repr((row.get('import_statement') or '')[:80])}")
    print(f"  Rows with ANY graph signal (out of first 50): {has_signal}/50\n")


# ============================================================================
# Main evaluation loop
# ============================================================================
def run_evaluation_on_split(ds_split, model, limit=1000, split_name=""):
    rows = []
    for row in ds_split:
        rows.append(row)
        if len(rows) >= limit:
            break

    print(f"  Loaded {len(rows)} rows for {split_name}")
    debug_sample_graph(rows)

    bm25_rankings = []
    dense_rankings = []
    hybrid_rankings = []
    bm25_graphrag_rankings = []
    hybrid_graphrag_rankings = []
    gold_indices = []

    for i, row in enumerate(rows):
        if i % 100 == 0:
            print(f"  Processing row {i}/{len(rows)}...")

        gold_idx = row["gold_snippet_index"]
        gold_indices.append(gold_idx)

        corpus_snippets = [c["snippet"] for c in row["context"]]
        corpus_tokens = [tokenize_code(s) for s in corpus_snippets]
        N = len(corpus_snippets)

        # Query: imports + last 5 lines of cropped code
        imports_text = row.get("import_statement") or ""
        cropped = row.get("cropped_code") or ""
        last_lines = "\n".join(cropped.splitlines()[-5:])
        query_text = f"{imports_text}\n{last_lines}"
        query_tokens = tokenize_code(query_text)

        # ── 1. BM25 ──────────────────────────────────────────────
        bm25 = BM25(corpus_tokens)
        bm25_scores = bm25.get_scores(query_tokens)
        bm25_ranked = sorted(range(N), key=lambda x: bm25_scores[x], reverse=True)
        bm25_rankings.append(bm25_ranked)

        # ── 2. Dense Semantic ────────────────────────────────────
        c_embs = model.encode(corpus_snippets, convert_to_tensor=True, show_progress_bar=False)
        q_emb = model.encode(query_text, convert_to_tensor=True, show_progress_bar=False)
        dense_scores = util.cos_sim(q_emb, c_embs)[0].cpu().numpy().tolist()
        dense_ranked = sorted(range(N), key=lambda x: dense_scores[x], reverse=True)
        dense_rankings.append(dense_ranked)

        # ── 3. Hybrid RRF ────────────────────────────────────────
        hybrid_ranked = reciprocal_rank_fusion([bm25_ranked, dense_ranked])
        hybrid_rankings.append(hybrid_ranked)

        # ── 4. BM25 + GraphRAG ───────────────────────────────────
        adj, graph_signal = build_code_graph(row)
        bm25_graphrag_ranked = graphrag_rerank(
            bm25_scores, adj, graph_signal, N,
            alpha=0.4, beta=0.6
        )
        bm25_graphrag_rankings.append(bm25_graphrag_ranked)

        # ── 5. Hybrid + GraphRAG ─────────────────────────────────
        # Use RRF score as base, then apply graph rerank
        rrf_scores = defaultdict(float)
        for rank, idx in enumerate(bm25_ranked):
            rrf_scores[idx] += 1.0 / (60 + rank + 1)
        for rank, idx in enumerate(dense_ranked):
            rrf_scores[idx] += 1.0 / (60 + rank + 1)
        rrf_score_list = [rrf_scores[i] for i in range(N)]
        # Normalise RRF as base (same formula as graphrag_rerank, use rrf instead of bm25)
        hybrid_graphrag_ranked = graphrag_rerank(
            rrf_score_list, adj, graph_signal, N,
            alpha=0.4, beta=0.6
        )
        hybrid_graphrag_rankings.append(hybrid_graphrag_ranked)

    bm25_metrics = evaluate_retriever(bm25_rankings, gold_indices)
    dense_metrics = evaluate_retriever(dense_rankings, gold_indices)
    hybrid_metrics = evaluate_retriever(hybrid_rankings, gold_indices)
    bm25_graphrag_metrics = evaluate_retriever(bm25_graphrag_rankings, gold_indices)
    hybrid_graphrag_metrics = evaluate_retriever(hybrid_graphrag_rankings, gold_indices)

    return bm25_metrics, dense_metrics, hybrid_metrics, bm25_graphrag_metrics, hybrid_graphrag_metrics


def fmt(metrics):
    return (f"EM={metrics['exact_match']*100:.1f}%  "
            f"R@1={metrics['recall_at_1']*100:.1f}%  "
            f"R@3={metrics['recall_at_3']*100:.1f}%  "
            f"R@5={metrics['recall_at_5']*100:.1f}%  "
            f"R@10={metrics['recall_at_10']*100:.1f}%  "
            f"MRR={metrics['mrr']:.4f}")


def main():
    print("Loading tianyang/repobench_python_v1.1 ...")
    ds = load_dataset("tianyang/repobench_python_v1.1")

    print("Loading sentence-transformers all-MiniLM-L6-v2 ...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    limit = 1000
    results = {}

    for split in ["cross_file_first", "cross_file_random"]:
        print(f"\n{'='*60}")
        print(f"  Evaluating split: {split} ({limit} samples)")
        print(f"{'='*60}")
        metrics = run_evaluation_on_split(ds[split], model, limit, split_name=split)
        results[split] = metrics
        labels = ["BM25", "Dense", "Hybrid RRF", "BM25+GraphRAG", "Hybrid+GraphRAG"]
        for lbl, m in zip(labels, metrics):
            print(f"  {lbl:25s}: {fmt(m)}")

    # ── Generate Markdown Report ───────────────────────────────────────────
    def pct(v, bold=False):
        s = f"{v*100:.1f}%"
        return f"**{s}**" if bold else s

    def mrr_fmt(v, bold=False):
        s = f"{v:.4f}"
        return f"**{s}**" if bold else s

    def row_md(label, m, bold=False):
        lbl = f"**{label}**" if bold else label
        return (
            f"| {lbl} "
            f"| {pct(m['exact_match'], bold)} "
            f"| {pct(m['recall_at_1'], bold)} "
            f"| {pct(m['recall_at_3'], bold)} "
            f"| {pct(m['recall_at_5'], bold)} "
            f"| {pct(m['recall_at_10'], bold)} "
            f"| {mrr_fmt(m['mrr'], bold)} |"
        )

    report_lines = [
        "# RepoBench-R Retrieval Benchmark Report",
        "",
        "**Dataset**: `tianyang/repobench_python_v1.1`  "
        "· **Paper**: [RepoBench arXiv:2306.03091](https://arxiv.org/abs/2306.03091)  "
        f"· **Samples**: {limit:,} per split",
        "",
        "> **Metric definitions for RepoBench-R** (single gold snippet per sample):",
        "> - **Exact Match (EM)**: top-1 retrieved snippet is the correct one. Mathematically = Recall@1.",
        "> - **Recall@k**: correct snippet appears anywhere in top-k results.",
        "> - **MRR**: Mean Reciprocal Rank = average of 1/rank(gold) across all samples.",
        "> - EM = Recall@1 because each sample has exactly one gold snippet.",
        "",
        "---",
        "",
    ]

    for split in results:
        m_bm25, m_dense, m_hybrid, m_bm25g, m_hybg = results[split]
        all_metrics = [m_bm25, m_dense, m_hybrid, m_bm25g, m_hybg]
        labels = [
            "BM25 Sparse Search",
            "Dense Semantic Search (all-MiniLM-L6-v2)",
            "Hybrid RRF (BM25 + Dense)",
            "BM25 + GraphRAG (Ours)",
            "Hybrid + GraphRAG (Ours)",
        ]
        # Find best per column
        best_em  = max(m["exact_match"]   for m in all_metrics)
        best_r1  = max(m["recall_at_1"]   for m in all_metrics)
        best_r3  = max(m["recall_at_3"]   for m in all_metrics)
        best_r5  = max(m["recall_at_5"]   for m in all_metrics)
        best_r10 = max(m["recall_at_10"]  for m in all_metrics)
        best_mrr = max(m["mrr"]           for m in all_metrics)

        report_lines += [
            f"## Split: `{split}`",
            "",
            "| Retrieval Strategy | Exact Match (EM) | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
        for lbl, m in zip(labels, all_metrics):
            is_best = (
                m["exact_match"]  == best_em  or
                m["recall_at_1"]  == best_r1  or
                m["recall_at_3"]  == best_r3  or
                m["recall_at_5"]  == best_r5  or
                m["recall_at_10"] == best_r10 or
                m["mrr"]          == best_mrr
            )
            report_lines.append(row_md(lbl, m, bold=is_best))
        report_lines.append("")

    report_lines += [
        "---",
        "",
        "## Architecture Clarifications",
        "",
        "### Dense Semantic Search",
        "Vector-based retrieval using `all-MiniLM-L6-v2` sentence-transformer.  ",
        "Query and corpus snippets are embedded into 384-dim dense vectors; ranked by **cosine similarity**.",
        "Unlike sparse (BM25) search, it captures semantic meaning without requiring exact keyword matches.",
        "",
        "### BM25 + GraphRAG (Ours)",
        "A two-stage retrieval pipeline:",
        "1. **BM25** scores each candidate snippet vs the query (import_statement + last 5 lines of cropped_code).",
        "2. **Code graph** is built from the RepoBench row fields:",
        "   - `IMPORTS` edge: snippet identifier/module appears in `import_statement`",
        "   - `CALLS` edge: snippet identifier (≥4 chars) appears in `cropped_code` body",
        "   - `CO-FILE` edge: snippets sharing the same file path",
        "3. **Spread-activation rerank**: BM25 scores propagate along graph edges.",
        "   `final_score = α * bm25_norm + β * (direct_graph_signal + 0.3 * neighbour_bm25_avg)`",
        "   This makes the ranking **genuinely distinct** from pure BM25 or Hybrid.",
        "",
        "### Hybrid + GraphRAG (Ours)",
        "Same as above but uses **RRF-fused (BM25+Dense)** scores as the base instead of raw BM25,",
        "then applies the same spread-activation graph reranking.",
        "",
        "### Why did previous GraphRAG ≈ Hybrid RRF?",
        "The prior implementation used Hybrid's top-3 as seeds, expanded a few graph neighbours,",
        "then **filled remaining slots back from the original Hybrid order** — producing nearly",
        "identical rankings. The new implementation uses independent graph scoring (α/β weighted",
        "spread-activation) that is mathematically decoupled from the Hybrid ranking.",
        "",
        "---",
        "",
        "## Academic Paper References",
        "",
        "- **RepoBench**: Tianyang Liu, Canwen Xu, Julian McAuley — [arXiv:2306.03091](https://arxiv.org/abs/2306.03091)",
        "- **SWE-bench**: Carlos E. Jimenez et al. — [arXiv:2310.06770](https://arxiv.org/abs/2310.06770)",
        "- **MemGPT**: Charles Packer et al. — [arXiv:2310.08560](https://arxiv.org/abs/2310.08560)",
        "- **MemTree**: Alireza Rezazadeh et al. — [arXiv:2410.14052](https://arxiv.org/abs/2410.14052)",
        "- **MemORAI** — [arXiv:2605.01386](https://arxiv.org/abs/2605.01386)",
        "- **Multi-Session Chat**: Jing Xu, Arthur Szlam, Jason Weston — [arXiv:2107.07567](https://arxiv.org/abs/2107.07567)",
    ]

    report_path = "/Users/manhnguyen/Project/CodeReviewBot/benchmark/repobench_retrieval_report.md"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines) + "\n")

    print(f"\n✅ Report saved to {report_path}")


if __name__ == "__main__":
    main()
