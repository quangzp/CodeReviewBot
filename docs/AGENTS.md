# AGENTS.md — GraphRAG Code Review Bot

## Identity

**GraphRAG Code Review Bot** — automates PR reviews, bug fixes, and code refactoring using Neo4j Code Property Graph, Behavioral Memory Graph, and tri-role LLM architecture.

## Scope

Python codebases indexed via AST into Neo4j. Three core capabilities:
- **review_pr** — analyze a GitHub pull request, generate patches per changed file
- **fix_bug** — fix a described bug in an indexed project without a PR
- **refactor_code** — refactor code based on a natural language description

## Architecture

```
User Message
    ↓
[classifier.py]   Intent gate: IN_SCOPE / OUT_SCOPE / CLARIFY  (fast_gate role — 8B)
    ↓ (IN_SCOPE)
[loop.py]         Agent loop (max 4 tool-call steps per turn)  (chat role — 32B)
    ↓
[tools.py]        Tool dispatch via <tool_call>{JSON}</tool_call> regex parsing
    ↓
  ┌─────────────────────────────────────────────────────────┐
  │ review_pr   → [reviewer.py]  → 3-phase pipeline        │
  │ fix_bug     → [fix_bug.py]   → 3-phase pipeline        │  (generation role — 32B)
  │ refactor    → [refactor.py]  → 3-phase pipeline        │
  └─────────────────────────────────────────────────────────┘
    ↓
[run_swebench_v2.py]
  phase1_localize_file    (BM25 + Neo4j vector + RRF)
  phase2_localize_fault   (LLM + graph context)
  [planner.py]            PlannerContract (root_cause + acceptance_criteria)
  phase3_generate_patch   (LLM + PlannerContract context)
    ↓
[verification/pipeline.py]
  Gate 1: ast_gate.py   — ast.parse() in-memory (~100ms, free)
  Gate 2: test_gate.py  — git apply → pytest → revert (skipped if no tests)
  Gate 3: evaluator.py  — LLM score 1-5, contract-aware (fail if < 3)
    ↓ (if fail)
[reflexion/reflector.py]  generate_reflection() → verbal self-reflection
  → retry Phase 3 (max retries: low=2, medium=3, high=5 based on risk)
    ↓
[memory/memory_neo4j.py]  Memory read (before) + write (after)
  → (Developer)-[:TENDS_TO]->(BugPattern) edges updated via MemoryStore
```

## Tools (7 total, defined in api/agent/tools.py)

| Tool | Required Args | Description |
|------|--------------|-------------|
| `add_project` | `repo_url` | Onboard GitHub repo, clone + build Neo4j graph |
| `list_projects` | — | List all indexed projects with stats |
| `project_status` | `repo_name` | Get status, node/edge counts |
| `review_pr` | `pr_url` | Run 3-phase pipeline on a pull request |
| `fix_bug` | `repo_name`, `bug_description` | Fix bug without PR |
| `refactor_code` | `repo_name`, `refactor_description` | Refactor without PR (optional: `file_path`) |
| `recent_reviews` | — | List recent reviews (optional: `repo_name`, `limit`) |

## LLM — Tri-role Architecture

All LLM calls route through `get_llm(role=...)` in `src_bot/llm/router.py`.

| Role | Default model | Provider | Used for |
|------|--------------|----------|---------|
| `fast_gate` | `llama-3.1-8b-instant` | Groq (free) | Intent classifier, patch evaluator (binary decisions) |
| `chat` | `llama-3.3-70b-versatile` | Groq or vLLM | Agent loop, tool routing, orchestration |
| `generation` | `llama-3.3-70b-versatile` | Groq or vLLM | Phase 1/2/3, Planner, Fix bug, Refactor |

- **Rate limit protection**: Auto-retry with exponential backoff on Groq 429 errors
- **vLLM fallback**: If vLLM server down → auto-fallback to Groq
- **Recommended self-hosted**: `Qwen2.5-Coder-32B` for generation, `DeepSeek-R1-32B` for chat

## Key Directories

```
api/                  FastAPI backend + agent infrastructure
api/agent/            loop.py, tools.py, classifier.py, planner.py, evaluator.py,
                      fix_bug.py, refactor.py, code_context.py, smell_detector.py
src_bot/memory/       store.py (MemoryStore — Neo4j TENDS_TO edges)
                      schema.py, extractor.py, retriever.py, memory_neo4j.py
src_bot/reflexion/    reflector.py, evaluator.py, patch_generator.py
src_bot/risk/         scorer.py (CHID formula), git_enrichment.py
src_bot/verification/ ast_gate.py, test_gate.py, pipeline.py
src_bot/observability/ langfuse_ctx.py (langfuse_trace + langfuse_span), langfuse_handler.py
src_bot/llm/          router.py — tri-role LLM routing
src_bot/config/       config.py — Pydantic settings (all env vars)
swebench/             neo4j_ingest.py (AST→Neo4j) + runner.py (SWE-bench)
frontend/             React 18 + TypeScript + TailwindCSS
data/projects/        Persistent git clones (one per indexed repo)
tests/                126 backend tests + 21 frontend (Vitest) tests
```

## Dev Setup

```bash
cp .env-example .env          # Fill in GROQ_API_KEY, APP_NEO4J_PASSWORD, GITHUB_TOKEN
bash init.sh                  # Check deps, init SQLite DB, verify connectivity
uvicorn api.main:app --reload --port 8000   # Backend
cd frontend && npm install && npm run dev   # Frontend (port 3000)
```

## Guardrails

- **Fail-open classifier**: Returns IN_SCOPE on error (never blocks user)
- **Fail-open evaluator**: Returns score=3 on error (never blocks patching)
- **Fail-open Langfuse**: All observability errors silently swallowed — pipeline never breaks
- **Fail-open Planner**: JSON parse error → minimal contract with `root_cause=fault_description`
- **Rate-limit retry**: Auto-retry on Groq 429 with exponential backoff
- **Scope gate**: Classifier rejects non-code-review requests before LLM
- **Max steps**: Agent loop caps at 4 tool-call steps per turn
- **Risk-adaptive retries**: low=2, medium=3, high=5 (from CHID score)

## Testing

```bash
# Backend (126 tests)
pytest tests/ -v --tb=short
pytest tests/ --cov --cov-report=term-missing

# Frontend (21 tests)
cd frontend && npm test
```

Test files: `test_reviewer.py`, `test_risk_scorer.py`, `test_reflector.py`, `test_memory_store.py`, `test_api_endpoints.py`, `test_pipeline_integration.py`, `test_evaluator.py`, `test_classifier.py`, `test_smell_detector.py`, `test_code_context.py`, `test_smoke.py`

## Environment Variables

| Variable | Required | Default | Purpose |
|----------|----------|---------|---------|
| `APP_NEO4J_URL` | Yes | bolt://localhost:7687 | Neo4j connection |
| `APP_NEO4J_PASSWORD` | Yes | — | Neo4j password |
| `GROQ_API_KEY` | Yes | — | Groq API key (free at groq.com) |
| `GITHUB_TOKEN` | Yes | — | GitHub PAT for PR data |
| `FAST_LLM_PROVIDER` | No | groq | Provider for fast_gate role |
| `FAST_LLM_MODEL` | No | llama-3.1-8b-instant | Model for fast_gate |
| `GEN_LLM_PROVIDER` | No | groq | Provider for generation role |
| `GEN_LLM_MODEL` | No | llama-3.3-70b-versatile | Model for generation |
| `CHAT_LLM_PROVIDER` | No | groq | Provider for chat role |
| `VLLM_GEN_BASE` | No | — | vLLM server URL (if self-hosted) |
| `LANGFUSE_ENABLED` | No | false | Enable LLM call tracing |
| `EXECUTION_GATE_ENABLED` | No | true | Enable test execution gate |
| `REFLEXION_MAX_RETRIES` | No | 3 | Max patch retry attempts |
