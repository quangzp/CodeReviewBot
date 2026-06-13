# Comparative Analysis: Conversational Memory Architectures

This analysis evaluates the architectural design and resource footprints of **MemGPT**, **MemORAI**, **MemTree**, and **Double Graph Memory** for persistent context storage and retrieval in conversational agents.

---

## 1. Architectural Feature Comparison

| Capabilities / Feature | MemGPT | MemORAI | MemTree | Double Graph Memory (Ours) |
| :--- | :---: | :---: | :---: | :---: |
| **Storage Representation** | Flat Text Logs | Multi-relational Graph | Hierarchical Tree of Summaries | Relational Code/Behavior Graph |
| **Write/Update Trigger** | Explicit LLM Tool Call | Turn-level Provenance pipeline | Tree consolidation traversal | Implicit Cypher ETL (Deterministic) |
| **Retrieval Strategy** | Semantic keyword search | Query-Adaptive Subgraph | Hierarchical Tree Walk | Proactive Cypher graph queries |
| **Provenance Tracking** | No (Flat list only) | **Yes** (Turn-level mapping) | No (Summaries merge details) | No (Aggregated evidence stats) |
| **Abstraction Level** | Raw chat history | Subgraph of relevant nodes | Hierarchical summaries | Aggregated profile entities |
| **LLM Reasoning Overhead** | High (Requires planning) | None (Adaptive database Rank) | High (Requires summarizations) | None (Deterministic execution) |

---

## 2. Simulated Footprint Metrics (5-Turn Developer Dialogue)

| Metrics (Simulated) | MemGPT | MemTree | MemORAI | Double Graph Memory |
| :--- | :---: | :---: | :---: | :---: |
| **Context Window Size (Tokens)** | ~111 | ~9 | ~76 | **~19** |
| **LLM Calls for Memory Updates** | 1 | 7 | 0 (Etl pipelines) | **0 (Deterministic updates)** |
| **Database Write Operations** | 5 | 5 | 20 | **16** |
| **Database Read Operations** | 1 (Semantic) | 1 (Tree Walk) | 2 (PageRank) | **1 (Cypher Index lookup)** |

---

## 3. Evaluation on Memory Task Types

Memory systems in software development are queried for two distinct categories: **Conversation History** (linear dialog tracking) and **Coder Behavior Profiling** (behavioral habits synthesis).

| Memory Task Type | MemGPT | MemTree | MemORAI | Double Graph Memory (Ours) |
| :--- | :---: | :---: | :---: | :---: |
| **Conversation History Retrieval**<br>*(e.g., retrieving details of a past crash dialog)* | **Excellent (Exact turn logs retrieved)** | **Good (Tree path retrieved)** | **Excellent (Turn-level provenance node links)** | **Poor (Only global behavior profiles stored; raw chats discarded)** |
| **Coder Behavior Profiling**<br>*(e.g., aggregating frequency & confidence of buggy patterns)* | **Poor (Requires semantic aggregation of unstructured logs)** | **Moderate (Summaries lose exact frequency counts)** | **Good (Relational patterns stored with node attributes)** | **Excellent (Aggregated Cypher relations track confidence & counts)** |

### A. Conversation History Retrieval Analysis
*   **MemGPT / MemORAI**: Excel in retrieving linear conversational snapshots. Since they retain raw turn logs linked to timestamps or turn IDs, queries like "What did the developer say about checkout.py crashing?" return the exact turn contents with minimal noise.
*   **Double Graph**: Fails at exact dialog retrieval because it discards raw conversational texts in favor of high-level profiling. It cannot reconstruct the original conversation turns.

### B. Coder Behavior Profiling Analysis
*   **Double Graph Memory**: Excels at behavior tracking. By utilizing direct Cypher indexing, the graph aggregates relationship weights (`confidence`, `evidence_count`) deterministically across multiple repositories and sessions. No LLM tokens are wasted on synthesis, and the output is mathematically exact.
*   **MemGPT**: Requires loading large volumes of raw history logs into the LLM context and asking the LLM to count and summarize the developer's bugs. This results in heavy token consumption and risk of hallucinated pattern analysis.
*   **MemTree**: Summaries abstract details over time, but loss of exact turn counts occurs due to information compression at ancestor nodes.

---

## 4. Structural Comparison Details

### A. MemGPT: Virtual Paged Memory
*   **Strengths**: Good for mimicking OS pagination where context is loaded dynamically based on immediate necessity.
*   **Weaknesses**: High token consumption because it retrieves raw text logs without hierarchy, causing context pollution as session lengths grow. Manual write updates rely entirely on LLM tool-calling capability, which can hallucinate or fail.

### B. MemTree: Dynamic Summary Tree
*   **Strengths**: Beautiful hierarchical structure where leaf nodes hold raw details and higher-level nodes hold abstracts. Walking the tree isolates search scopes.
*   **Weaknesses**: High LLM overhead. Every turn requires updating tree ancestors and generating new LLM summaries, resulting in expensive continuous LLM API usage.

### C. MemORAI: Provenance-Enriched Adaptive Subgraph
*   **Strengths**: Multi-relational graph structure that maps raw turns to structural nodes. Supports **Turn-level Provenance**, meaning it is audit-friendly. The Query-Adaptive Subrank PageRank provides extremely precise contexts.
*   **Weaknesses**: Complex graph updates. Relational semantic filtering is computationally expensive for high-volume transactions.

### D. Double Graph Memory: Cypher-Driven Entity Graphs
*   **Strengths**: Combines the Code Property Graph (CPG) with Developer Behavioral Profiles in a global Neo4j instance. Updates are completely deterministic (via background ETLs), requiring **zero LLM reasoning tokens** for updates. Prompt context stays constant and small (e.g. ~70-80 tokens) because Cypher aggregates history count and confidence scores before injection.
*   **Weaknesses**: Lacks strict turn-by-turn conversational provenance tracking (aggregates facts into global profiles instead of linking turn IDs to nodes).

