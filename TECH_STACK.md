# Tech Stack — GraphRAG Code Review Bot

A complete reference for every technology used in this project, organized by layer.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Backend API](#2-backend-api)
3. [AI & Agent Layer](#3-ai--agent-layer)
4. [Harness Engineering Pipeline](#4-harness-engineering-pipeline)
5. [Graph Database — Neo4j](#5-graph-database--neo4j)
6. [Vector Database — Weaviate](#6-vector-database--weaviate)
7. [Memory Layer — Graphiti](#7-memory-layer--graphiti)
8. [Embeddings](#8-embeddings)
9. [GitHub Integration](#9-github-integration)
10. [Risk Scoring](#10-risk-scoring)
11. [Frontend](#11-frontend)
12. [SWE-bench Evaluation](#12-swe-bench-evaluation)
13. [Storage & Persistence](#13-storage--persistence)
14. [LLM Models & Providers](#14-llm-models--providers)
15. [Configuration & Environment](#15-configuration--environment)
16. [Dependency Map](#16-dependency-map)

---

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Frontend (React + Vite)                  │
│         Chat UI · Dashboard · Projects · Reviews · Devs         │
└────────────────────────────┬────────────────────────────────────┘
                             │ HTTP / SSE
┌────────────────────────────▼────────────────────────────────────┐
│                      FastAPI Backend                            │
│   /api/chat (SSE)  ·  /api/reviews  ·  /api/projects           │
│   /api/github/webhook  ·  /api/developers                       │
└────┬──────────┬──────────────┬───────────────────┬─────────────┘
     │          │              │                   │
     ▼          ▼              ▼                   ▼
 Agent Loop  Reviewer    Project Indexer     GitHub App
 (LangGraph) (Pipeline)  (GitPython)        (PyGithub)
     │          │
     ▼          ▼
┌─────────────────────────────┐
│   Harness Engineering       │
│  Classifier → Generator     │
│  → Evaluator → Reflexion    │
└────────┬────────────────────┘
         │
   ┌─────┴──────┐
   ▼            ▼
 Neo4j       Weaviate
 (GraphRAG)  (Vector)
   │
   ▼
 Graphiti
 (Memory)
```

---

## 2. Backend API

### FastAPI
- **Version:** `>=0.100.0`
- **Role:** Main HTTP server — REST endpoints + Server-Sent Events (SSE) for streaming agent responses
- **Key features used:**
  - `StreamingResponse` with `text/event-stream` for real-time agent output
  - `BackgroundTasks` for async project indexing
  - `lifespan` context manager for startup/shutdown (DB init, Graphiti teardown)
  - Dependency injection via `Depends()`

### Uvicorn
- **Version:** `>=0.23.0`
- **Role:** ASGI server that runs FastAPI
- **Run command:** `uvicorn api.main:app --reload --port 8000`

### Pydantic v2
- **Version:** `>=2.0.0`
- **Role:** Data validation and serialization for all API request/response models
- **Used in:** `api/models.py` — `ReviewRequest`, `ProjectCreate`, `ChatMessage`, etc.

### Pydantic Settings
- **Version:** `>=2.0.0`
- **Role:** Loads environment variables into typed `Configs` class (`src_bot/config/config.py`)
- **Pattern:** Single `configs` singleton imported across the entire codebase

### python-dotenv
- **Version:** `>=1.0.0`
- **Role:** Loads `.env` file into environment at startup

---

## 3. AI & Agent Layer

### LangGraph
- **Version:** `>=0.2.0`
- **Role:** Orchestrates the stateful agent graph — manages conversation state, tool calls, and multi-step reasoning
- **Used in:** `api/agent/loop.py` — the main chat agent loop
- **Graph shape:** Pipeline (Intent Classifier → Planner → Generator → Evaluator)

### LangChain Core
- **Version:** `>=0.3.0`
- **Role:** Base abstractions — `BaseChatModel`, `HumanMessage`, `AIMessage`, tool schemas
- **Used in:** LLM wrappers, prompt templates, message history

### LangChain Groq
- **Version:** `>=0.2.0`
- **Role:** LangChain adapter for the Groq API
- **Used in:** `src_bot/llm/router.py` — primary LLM provider

### LangChain Ollama
- **Version:** `>=0.2.0`
- **Role:** LangChain adapter for local Ollama models (fallback when `LLM_PROVIDER=ollama`)

### LangChain OpenAI
- **Version:** `>=0.2.0`
- **Role:** LangChain adapter for OpenAI (fallback when `LLM_PROVIDER=openai`)

---

## 4. Harness Engineering Pipeline

This is the core innovation of the project, inspired by the Harness Engineering discipline (Alice.io, 2026). It wraps the LLM with production-grade governance.

### Stage 0 — Intent Classifier (`api/agent/classifier.py`)
- **Model:** `llama-3.1-8b-instant` (Groq, fast & free)
- **Role:** Classifies every user message BEFORE the main LLM sees it
- **Output:** `IN_SCOPE` / `OUT_SCOPE` / `CLARIFY`
- **Fail mode:** Falls back to `IN_SCOPE` on any error (fail open — never block the user on classifier failure)
- **Implementation note:** Uses `.replace("REPLACE_MESSAGE", ...)` instead of `.format()` to avoid Python KeyError on JSON examples in the prompt

### Stage 1 — Planner (inside `api/agent/loop.py`)
- **Model:** `llama-3.3-70b-versatile` (Groq, 70B reasoning)
- **Role:** Analyzes the bug report, reads graph context, identifies root cause files
- **ACI Design:** Injects code WITH line numbers (`" 42 | def foo():..."`) so the LLM targets exact lines — prevents off-by-one patch errors

### Stage 2 — Generator (`src_bot/reflexion/patch_generator.py`)
- **Role:** Produces a unified diff patch from the planner's analysis
- **ACI Design:**
  - Code shown with line numbers (Agent-Computer Interface)
  - Patch normalizer (`_normalize_patch`) fuzzy-matches deletion lines against actual file content and rebuilds correct `@@` hunk headers
  - Blank lines in hunks converted to single space `" "` (git apply requirement)

### Stage 3 — Evaluator (`api/agent/evaluator.py`)
- **Model:** `llama-3.1-8b-instant` (fast, free — adds minimal latency)
- **Role:** Independently scores the patch 1–5. Patches scoring < 3 are rejected and trigger another reflexion attempt with the evaluator's feedback as context
- **Score rubric:**
  - `5` — Directly fixes the bug, minimal change, no regressions
  - `4` — Fixes bug but slightly over-engineered
  - `3` — Partially addresses, may miss edge cases
  - `2` — Wrong approach
  - `1` — Completely wrong, introduces regressions
- **Fail mode:** Returns `score=3` on any API error (fail open)

### Reflexion Loop (`src_bot/reflexion/reflector.py`)
- **Role:** On patch rejection, generates self-critique combining: bug description + failed patch + test output + graph context + previous reflections
- **Max retries:** Configured via `REFLEXION_MAX_RETRIES` (default: 3)
- **`--no-reflexion` flag:** Sets `max_retries=1` — saves ~3x tokens for SWE-bench runs
- **Implementation note:** Uses direct `llm.invoke(filled_prompt)` instead of LangChain pipe (`prompt | llm`) to avoid `_RateLimitedLLM` proxy incompatibility

### Frustration Detection (`api/reviewer.py`)
- **Rule:** If `len(new_patch) < len(last_patch) * 0.6` → the agent is confused (patch is shrinking)
- **Response:** Widens the graph context window (`RETRIEVER_TOP_K` increases) and injects a hint to look at adjacent files

### Rate-Limited LLM Router (`src_bot/llm/router.py`)
- **Role:** Wraps the LLM with rate limiting to stay within Groq free-tier limits
- **Supports:** `groq` · `ollama` · `openai` · `together` (selected via `LLM_PROVIDER` env var)

---

## 5. Graph Database — Neo4j

- **Client:** `neo4j>=5.0.0` (official Python driver)
- **Role:** Stores the code knowledge graph — functions, classes, files, imports, call edges, commit history
- **Connection:** Bolt protocol (`bolt://localhost:7687`)
- **Config:**
  - `NEO4J_MAX_CONNECTION_POOL_SIZE=50`
  - `NEO4J_MAX_CONNECTION_LIFETIME=30`
  - `NEO4J_CONNECTION_TIMEOUT=30.0`

### GraphRAG Retrieval (`src_bot/graph_retriever.py`)
- **Pattern:** Multi-hop graph traversal starting from changed files in the PR
- **Config bounds:**
  - `RETRIEVER_TOP_K=20` — top-K nodes per hop
  - `RETRIEVER_MAX_HOPS=3` — depth of traversal
  - `RETRIEVER_MAX_CONTEXT_NODES=500` — total context cap (prevents prompt overflow)
- **Output:** Structured code context injected into the Planner prompt

### Neo4j Service (`src_bot/neo4jdb/`)
- `neo4j_db.py` — connection pool management, session factory
- `neo4j_service.py` — high-level CRUD: ingest files, functions, call edges, commit nodes

---

## 6. Vector Database — Weaviate

- **Client:** `weaviate-client>=4.0.0` (v4 API)
- **Role:** Semantic search over code chunks — used during project indexing to find relevant code sections by meaning, not just graph topology
- **Collection:** Configured via `WEAVIATE_COLLECTION_NAME` env var
- **Used in:** `ingest_weaviate.py`, `migrate_weaviate.py` (ingestion scripts)

---

## 7. Memory Layer — Graphiti

- **Package:** `graphiti-core` (version pinned to `0.29.1` in the project)
- **Role:** Cross-session episodic memory — records every PR review as an episode, auto-extracts developer patterns and bug recurrence as graph entities/relationships
- **Used in:** `src_bot/memory/graphiti_store.py`

### Why Graphiti instead of a plain vector store
Graphiti builds a **temporal knowledge graph** from episodes. It extracts entities (developers, files, bug types) and relationships (who fixed what, recurring patterns) automatically via LLM. Plain vector stores only do semantic similarity; Graphiti also captures *how things are connected over time*.

### Free Setup (no OpenAI required)
```python
# LLM: GroqClient (built into graphiti-core)
llm_config = LLMConfig(
    api_key=configs.GROQ_API_KEY,
    model="llama-3.3-70b-versatile",      # entity extraction
    small_model="llama-3.1-8b-instant",   # classification
)
llm_client = GroqClient(config=llm_config)

# Embedder: local SentenceTransformer (free, no API)
embedder = SentenceTransformerEmbedder(model_name="all-MiniLM-L6-v2")

graphiti = Graphiti(
    uri=NEO4J_URL, user=NEO4J_USER, password=NEO4J_PASSWORD,
    llm_client=llm_client, embedder=embedder
)
```

### Memory operations
- `record_review()` — ingests a completed review as an episode (developer login, repo, PR number, summary, bug patterns)
- `build_memory_context()` — semantic search for past patterns relevant to the current developer + repo

---

## 8. Embeddings

### SentenceTransformers (`sentence-transformers>=2.2.0`)
- **Model:** `all-MiniLM-L6-v2` (22M parameters, runs locally, free)
- **Role:** Generates 384-dimension embeddings for Graphiti memory and Weaviate ingestion
- **Custom wrapper:** `src_bot/memory/embedder.py` implements `EmbedderClient` interface required by `graphiti-core`

```python
class SentenceTransformerEmbedder(EmbedderClient):
    async def create(self, input_data) -> list[float]:       # single embed
    async def create_batch(self, input_data_list) -> list[list[float]]:  # batch
```

---

## 9. GitHub Integration

### PyGithub (`PyGithub>=2.0.0`)
- **Role:** Fetches PR metadata, diff, file list, commit history, contributor info
- **Used in:** `api/reviewer.py`, `src_bot/risk/git_enrichment.py`

### GitHub App (`api/github_app.py`)
- **Role:** Receives push webhooks for incremental re-indexing when new commits land
- **Endpoint:** `POST /api/github/webhook`
- **Verification:** HMAC-SHA256 signature check on incoming payloads

### unidiff (`unidiff>=0.7.0`)
- **Role:** Parses unified diff format from GitHub PR diffs into structured `PatchedFile` objects
- **Used in:** Diff analysis during review to identify changed lines and files

### Requests (`requests>=2.31.0`)
- **Role:** HTTP client for GitHub API calls that bypass PyGithub (raw diff fetch, rate limit checks)

---

## 10. Risk Scoring

### CHID-Based Scorer (`src_bot/risk/scorer.py`)
Based on the CHID paper (Springer 2025) — "Change-Impact Driven" risk model.

**6 risk dimensions** (weights in `config/risk_weights.yaml`):

| Dimension | Weight | What it measures |
|---|---|---|
| `blast_radius` | 0.20 | How many other files import the changed file |
| `coverage_gap` | 0.25 | Files changed without test coverage |
| `bug_frequency` | 0.15 | Historical bug density in changed files |
| `contributor_churn` | 0.15 | How many different developers touched the file |
| `new_contributor` | 0.15 | Whether the PR author is new to this file |
| `pr_size` | 0.10 | Lines changed (large PRs are riskier) |

**Decision thresholds:**
- Score `< 0.30` → auto-approve
- Score `0.30–0.55` → review recommended
- Score `> 0.55` → request changes

### GitPython (`GitPython>=3.1.0`)
- **Role:** Mines local git history for commit frequency, contributor churn, bug-fix commits (by commit message pattern)
- **Used in:** `src_bot/risk/git_enrichment.py`

### PyYAML (`pyyaml>=6.0`)
- **Role:** Loads `config/risk_weights.yaml` at runtime — allows tuning weights without code changes

---

## 11. Frontend

### React 18 (`react@^18.3.1`)
- **Role:** UI component library
- **Key pattern:** Functional components with hooks (`useState`, `useEffect`, `useRef`, `useContext`)

### TypeScript (`typescript@^5.5.3`)
- **Role:** Type safety across all frontend code
- **Config:** `tsconfig.json` with strict mode, path aliases

### Vite (`vite@^5.4.2`)
- **Role:** Build tool and dev server (HMR)
- **Plugin:** `@vitejs/plugin-react` for JSX transform
- **Dev server:** `http://localhost:5173`
- **API proxy:** Vite proxies `/api/*` to `http://localhost:8000` in dev mode

### TailwindCSS (`tailwindcss@^3.4.11`)
- **Role:** Utility-first CSS framework
- **Config:** `tailwind.config.js` with custom dark theme colors

### React Router DOM (`react-router-dom@^6.26.2`)
- **Role:** Client-side routing
- **Routes:** `/` (Dashboard) · `/projects` · `/projects/:id` · `/reviews/:id` · `/chat` · `/developers`

### react-diff-viewer-continued (`^3.4.0`)
- **Role:** Renders unified diff patches as side-by-side or inline code diff view in `PatchViewer.tsx`

### Key Frontend Pages

| Page | File | Description |
|---|---|---|
| Dashboard | `pages/Dashboard.tsx` | Overview stats, recent reviews |
| Projects | `pages/Projects.tsx` | List + create projects (repos) |
| Project Detail | `pages/ProjectDetail.tsx` | Files, indexing status, risk map |
| New Review | `pages/NewReview.tsx` | Submit PR URL for review |
| Review Detail | `pages/ReviewDetail.tsx` | Review result, patch viewer, fix attempts |
| Chat | `pages/Chat.tsx` | ChatGPT-style agent interface with SSE streaming |
| Developers | `pages/Developers.tsx` | Developer stats from memory graph |

### Server-Sent Events (SSE) Streaming
The Chat page uses the browser `EventSource` API (or `fetch` with `ReadableStream`) to stream agent responses token-by-token from `GET /api/chat` as the agent thinks and calls tools.

---

## 12. SWE-bench Evaluation

### SWE-bench Lite
- **Dataset:** `princeton-nlp/SWE-bench_Lite` (HuggingFace) — 300 tasks across 12 Python repos (Django, Sympy, Astropy, etc.)
- **Purpose:** Measures patch correctness — whether the generated patch fixes the failing test suite

### Local SWE-bench Runner (`run_swebench_v2.py`)
Key engineering improvements over a naive runner:

| Feature | Implementation |
|---|---|
| **ACI line numbers** | `_add_line_numbers()` prefixes every code line with `" N | "` |
| **Patch normalizer** | `_normalize_patch()` uses `SequenceMatcher` (threshold 0.6) to fuzzy-match deletion lines against actual file, rebuilds correct `@@` headers |
| **Blank line fix** | Empty lines in hunks converted to `" "` (space) — git apply requirement |
| **`--recount` flag** | `git apply --check --recount` tolerates shifted line numbers |
| **`patch` fallback** | Falls back to `patch -p1 --fuzz=3 --dry-run` if `git apply` fails |
| **Evaluator wired** | Score < 3 → reject patch, retry |
| **Frustration detect** | Patch shrinks < 60% of previous → widen context |
| **`--no-reflexion`** | `max_retries=1` — saves ~3x tokens, needed for free-tier rate limits |

### Official SWE-bench Harness
- **Package:** `swebench` (pip)
- **Requires:** Docker Desktop running
- **Command:** `python -m swebench.harness.run_evaluation --predictions_path predictions.jsonl --run_id my_run`
- **Note:** Run from `/tmp` to avoid local `swebench/` folder shadowing the pip package

### HuggingFace Datasets (`datasets>=2.14.0`)
- **Role:** Downloads and caches SWE-bench Lite dataset
- **Cache:** `~/.cache/huggingface/datasets/`

---

## 13. Storage & Persistence

### SQLite + aiosqlite (`aiosqlite>=0.19.0`)
- **File:** `data/reviews.db`
- **Role:** Stores reviews, projects, chat sessions, fix attempts
- **Pattern:** Fully async — `async with aiosqlite.connect(DB_PATH) as db:`
- **Tables:** `projects`, `reviews`, `fix_attempts`, `chat_sessions`, `chat_messages`
- **Note:** `data/reviews.db` is git-ignored (contains runtime user data)

### Project Clones (Filesystem)
- **Location:** `data/projects/{project_id}/` (persistent across server restarts)
- **Moved from:** `/tmp/` (was wiped on server reboot — caused `RuntimeError: Project clone not found`)
- **Managed by:** `api/project_indexer.py`
- **Note:** `data/projects/` is git-ignored (contains full repo clones, can be GBs)

### Neo4j (Graph Persistence)
- Graph data persists in Neo4j's own storage — survives server restarts
- Connection pooled via `neo4j>=5.0.0` driver

---

## 14. LLM Models & Providers

### Primary: Groq API (Free Tier)
| Model | Role | Context | Speed |
|---|---|---|---|
| `llama-3.3-70b-versatile` | Main reasoning (Planner + Generator) | 128K tokens | ~200 tok/s |
| `llama-3.1-8b-instant` | Classifier + Evaluator | 128K tokens | ~750 tok/s |

**Free tier limits:**
- `llama-3.1-8b-instant`: ~500K tokens/day
- `llama-3.3-70b-versatile`: ~100K tokens/day

**Why Groq:** Zero cost, extremely fast inference (GroqChip), sufficient quality for code tasks at 70B scale.

### Fallback Options (configured via `.env`)

| Provider | `LLM_PROVIDER` | Best for |
|---|---|---|
| Ollama | `ollama` | Fully offline, no API limits |
| OpenAI | `openai` | Highest quality (GPT-4o) |
| Together.ai | `together` | Alternative cloud inference |

### Dual-LLM Harness Design
A key Harness Engineering pattern: use a **small fast model** for high-frequency cheap tasks (classification, evaluation) and a **large model** for reasoning tasks (planning, generation). This keeps latency low and token costs minimal.

```
User message
    │
    ▼
llama-3.1-8b-instant  ← Intent Classifier (fast, cheap)
    │ IN_SCOPE
    ▼
llama-3.3-70b-versatile  ← Planner + Generator (smart, heavy)
    │ patch
    ▼
llama-3.1-8b-instant  ← Evaluator (fast, cheap)
    │ score ≥ 3
    ▼
  Accept
```

---

## 15. Configuration & Environment

All configuration lives in `.env` (git-ignored). Copy `.env-example` to get started:

```bash
cp .env-example .env
```

### Required Variables

| Variable | Example | Description |
|---|---|---|
| `APP_NEO4J_URL` | `bolt://localhost:7687` | Neo4j connection string |
| `APP_NEO4J_USER` | `neo4j` | Neo4j username |
| `APP_NEO4J_PASSWORD` | `your_password` | Neo4j password |
| `APP_NEO4J_DATABASE` | `neo4j` | Neo4j database name |
| `GROQ_API_KEY` | `gsk_...` | Groq API key (free at console.groq.com) |
| `WEAVIATE_COLLECTION_NAME` | `CodeBotCollection` | Weaviate collection |

### Optional Variables

| Variable | Default | Description |
|---|---|---|
| `GITHUB_TOKEN` | _(empty)_ | GitHub PAT — needed for private repos; public repos work without it |
| `LLM_PROVIDER` | `groq` | `groq` · `ollama` · `openai` · `together` |
| `LLM_MODEL` | `llama-3.3-70b-versatile` | Main reasoning model |
| `LLM_TEMPERATURE` | `0` | Deterministic output for code generation |
| `REFLEXION_MAX_RETRIES` | `3` | Max patch retry attempts |
| `RETRIEVER_TOP_K` | `20` | Graph nodes retrieved per hop |
| `RETRIEVER_MAX_HOPS` | `3` | Graph traversal depth |
| `RETRIEVER_MAX_CONTEXT_NODES` | `500` | Max nodes injected into prompt |

---

## 16. Dependency Map

```
CodeReviewBot
├── Backend
│   ├── fastapi + uvicorn          → HTTP server + SSE streaming
│   ├── pydantic v2                → Data validation
│   └── python-dotenv              → Env loading
│
├── AI & Agent
│   ├── langgraph                  → Agent state machine
│   ├── langchain-core             → LLM abstractions
│   ├── langchain-groq             → Groq adapter (primary)
│   ├── langchain-ollama           → Local model adapter (fallback)
│   └── langchain-openai           → OpenAI adapter (fallback)
│
├── Harness Pipeline
│   ├── Classifier (8b-instant)    → Intent gate
│   ├── Planner (70b-versatile)    → Root cause analysis
│   ├── Generator (70b-versatile)  → Patch creation
│   ├── Evaluator (8b-instant)     → Patch scoring
│   └── Reflector                  → Self-critique on rejection
│
├── Databases
│   ├── neo4j                      → Code knowledge graph (GraphRAG)
│   ├── weaviate-client            → Semantic vector search
│   └── aiosqlite                  → Reviews/sessions persistence
│
├── Memory
│   ├── graphiti-core              → Episodic cross-session memory
│   └── sentence-transformers      → Local embeddings (all-MiniLM-L6-v2)
│
├── GitHub
│   ├── PyGithub                   → PR data, contributor stats
│   ├── unidiff                    → Diff parsing
│   └── requests                   → Raw GitHub API calls
│
├── Risk
│   ├── GitPython                  → Git history mining
│   └── pyyaml                     → Weight config loading
│
├── Frontend
│   ├── react@18                   → UI components
│   ├── typescript@5               → Type safety
│   ├── vite@5                     → Build + dev server
│   ├── tailwindcss@3              → Utility CSS
│   ├── react-router-dom@6         → Client routing
│   └── react-diff-viewer-continued → Patch diff renderer
│
└── SWE-bench
    ├── datasets (HuggingFace)     → SWE-bench Lite dataset
    ├── swebench (pip)             → Official harness (Docker)
    └── run_swebench_v2.py         → Custom runner with ACI + normalizer
```

---

*Last updated: 2026-05-30 | Branch: `chatbot_dev`*
