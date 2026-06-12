# GraphRAG Code Review Bot — System Overview

> Hệ thống tự động review PR, phát hiện bug và đề xuất patch bằng AI, sử dụng knowledge graph + LLM + Reflexion loop.

---

## 1. Vấn đề giải quyết

Việc review code thủ công tốn thời gian, dễ bỏ sót bug tinh tế, và không học từ lịch sử lỗi của team. Bot này giải quyết bằng cách:

- Hiểu **cấu trúc code** qua knowledge graph (ai gọi ai, ai import gì)
- **Định vị lỗi** chính xác thay vì đọc toàn bộ file
- **Tự generate patch** và kiểm tra lại trước khi đề xuất
- **Học từ lịch sử** — nhớ pattern bug của từng developer

---

## 2. Kiến trúc tổng thể

```
┌─────────────────────────────────────────────────────────────────┐
│                        Frontend (React)                         │
│  Chat UI │ Projects │ Reviews │ Developers │ Live SSE stream    │
└──────────────────────────┬──────────────────────────────────────┘
                           │ REST + SSE
┌──────────────────────────▼──────────────────────────────────────┐
│                     FastAPI Backend                             │
│  /api/projects  /api/reviews  /api/chat  /webhook  /health     │
└──────┬──────────┬──────────────┬──────────────────┬────────────┘
       │          │              │                  │
  ┌────▼───┐ ┌───▼────┐  ┌─────▼──────┐  ┌────────▼───────┐
  │ Project │ │ Review │  │  Chat Agent │  │ GitHub Webhook │
  │ Indexer │ │Pipeline│  │   (loop)    │  │ (push / PR)    │
  └────┬───┘ └───┬────┘  └─────┬──────┘  └────────────────┘
       │         │              │
┌──────▼─────────▼──────────────▼──────────────────────────────┐
│                      Core Engine                              │
│  LLM Router  │  GraphRAG Retriever  │  Reflexion Loop        │
│  Planner     │  Verification Gates  │  Risk Scorer           │
│  Langfuse    │  MemoryStore         │  Smell Detector        │
└──────┬───────────────┬──────────────┬─────────────────────┘
       │               │              │
  ┌────▼────────────────▼──────────────────────┐
  │                  Neo4j                     │
  │  ┌─────────────────┐  ┌─────────────────┐ │
  │  │ Code Graph      │  │ Vector Index    │ │
  │  │ (AST: Module /  │  │ (native Neo4j   │ │
  │  │  Class /        │  │  all-MiniLM     │ │
  │  │  Function)      │  │  384-dim)       │ │
  │  ├─────────────────┤  └─────────────────┘ │
  │  │ Behavioral      │                      │
  │  │ Memory Graph    │                      │
  │  │ (Developer /    │                      │
  │  │  BugPattern /   │                      │
  │  │  Review)        │                      │
  │  └─────────────────┘                      │
  └────────────────────────────────────────────┘
       │              │
  ┌────▼────┐   ┌─────▼──────┐
  │ SQLite  │   │ Disk clone │
  │(metadata│   │ data/      │
  │ + chat) │   │ projects/  │
  └─────────┘   └────────────┘
```

---

## 3. Hai Graph trong Neo4j

Neo4j lưu **hai knowledge graph độc lập** phân biệt bằng node labels khác nhau:

### 3a. Code Property Graph (AST)
Được tạo khi onboard repo — toàn bộ Python source phân tích bằng AST:

| Node | Properties |
|------|-----------|
| `Module`   | path, content, line_count |
| `Class`    | name, file_path, docstring |
| `Function` | name, qualified_name, body, line_count, lineno |

| Relationship | Ý nghĩa |
|-------------|---------|
| `DEFINES`   | Module chứa Class/Function |
| `CALLS`     | Function A gọi Function B |
| `IMPORTS`   | Module A import Module B |
| `INHERITS`  | Class A kế thừa Class B |
| `CONTAINS`  | Class chứa Method |

**Dùng để làm gì:** GraphRAG retriever dùng **BM25 full-text search + Neo4j native vector index (all-MiniLM-L6-v2, 384-dim)** + graph traversal để lấy context cho LLM — ai gọi function bị lỗi, blast radius, dependency chain. Kết quả merge bằng **Reciprocal Rank Fusion (RRF)**.

### 3b. Behavioral Memory Graph
Được cập nhật sau mỗi review — học hành vi developer theo thời gian. Xem chi tiết tại **mục 7**.

---

## 4. Pipeline xử lý PR (3 Phase)

```
PR changed files
      │
      ▼
Phase 1 — File Localization
  Hybrid search: BM25 + vector cosine (Neo4j native), merge bằng RRF
  → Tìm file và function liên quan đến bug
      │
      ▼
Phase 2 — Fault Localization
  LLM phân tích code + graph context
  → Xác định dòng code lỗi, nguyên nhân gốc rễ
      │
      ▼
[NEW] Planner — PlannerContract
  LLM tạo kế hoạch có cấu trúc:
  - root_cause, approach, must_preserve
  - acceptance_criteria, files_to_modify, risk_notes
      │
      ▼
Phase 3 — Patch Generation
  LLM viết unified diff patch
  Có ngữ cảnh từ PlannerContract
      │
      ▼
Verification Pipeline (3 gates)
  Gate 1: AST parse — syntax valid?   (free, ~100ms)
  Gate 2: Test execution — tests pass? (uses existing pytest)
  Gate 3: LLM evaluate — quality 1-5? (contract-aware)
      │
      ▼
Reflexion Loop (max 3 retries)
  Nếu gate fail → Reflector viết "verbal self-reflection"
  → Phân tích tại sao fail → Feed vào Phase 3 retry
      │
      ▼
Final patch + risk score
```

---

## 5. LLM Router — Tri-role Architecture

Tất cả LLM calls đi qua 1 điểm duy nhất: `get_llm(role=...)`.

| Role | Model | Provider | Dùng cho |
|------|-------|----------|---------|
| `fast_gate` | llama-3.1-8b-instant | Groq API | Intent classifier, patch evaluator |
| `chat` | DeepSeek-R1-Distill-Qwen-32B | vLLM (self-hosted) | Agent loop, tool routing, orchestration |
| `generation` | Qwen2.5-Coder-32B-Instruct | vLLM (self-hosted) | Phase 1-2-3, Planner, Fix bug, Refactor, Review |

**Phân công lý do:**
- `fast_gate` — quyết định đơn giản (binary intent + score 1-5), không cần model lớn
- `chat` — orchestrator: hiểu ngữ cảnh hội thoại, quyết định gọi tool nào với đúng arguments; dùng DeepSeek-R1 vì cần reasoning mạnh
- `generation` — toàn bộ coding tasks; Qwen2.5-Coder chuyên biệt code, tốt hơn general model ở patch generation

**Multi-backend hỗ trợ:** Ollama (local) · Groq (free API) · OpenAI · Together.ai · vLLM (self-hosted)

**Rate limit protection:** Tự động retry với exponential backoff (429 errors)

**vLLM fallback:** Nếu vLLM server down → tự động fallback sang Groq

**Config mẫu (`.env`):**
```bash
# fast_gate — Groq free tier
FAST_LLM_PROVIDER=groq
FAST_LLM_MODEL=llama-3.1-8b-instant

# chat — DeepSeek-R1 tự host (orchestration)
CHAT_LLM_PROVIDER=vllm
CHAT_LLM_MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B
VLLM_CHAT_BASE=http://localhost:8001/v1

# generation — Qwen2.5-Coder tự host (code tasks)
GEN_LLM_PROVIDER=vllm
GEN_LLM_MODEL=Qwen/Qwen2.5-Coder-32B-Instruct
VLLM_GEN_BASE=http://localhost:8000/v1
```

---

## 6. Chat Agent Layer

Người dùng chat bằng ngôn ngữ tự nhiên. Agent quyết định tool nào cần gọi:

```
User message
     │
     ▼
Intent Classifier (fast_gate — Groq 8B)
  IN_SCOPE / OUT_SCOPE / CLARIFY
     │ (chỉ IN_SCOPE đi tiếp)
     ▼
Agent Loop (chat — DeepSeek-R1-32B, self-hosted)
  Orchestrator: chọn tool + format arguments
  add_project · list_projects · project_status
  review_pr · fix_bug · refactor_code · recent_reviews
     │
     ▼
Tool execution (generation — Qwen2.5-Coder-32B, self-hosted)
  Planner → Phase 1/2/3 → Verification → Reflexion
     │
     ▼
Persisted vào SQLite (session_id → messages)
```

**Tính năng:** Session persistence (ChatGPT-style) · Tool call rendering (diff viewer, project card) · Max 4 tool-call steps/turn

---

## 7. Behavioral Memory Graph (Custom — trong Neo4j)

Đây là graph thứ hai sống trong cùng Neo4j với Code Graph, nhưng hoàn toàn tách biệt về label và mục đích. Được thiết kế theo kiến trúc của Graphiti (temporal knowledge graph) nhưng implement thuần Neo4j, không cần thư viện ngoài.

### Schema

```
Nodes:
  (:Developer  { login, name, first_seen_at, last_seen_at, pr_count })
  (:BugPattern { id, name, description, severity, occurrence_count })
  (:Review     { id, repo_name, pr_number, pr_url, reviewed_at, summary })
  (:Module     { path, project_id, recent_bug_count, last_bug_at })

Relationships:
  (Developer)-[:AUTHORED]         ->(Review)
  (Review)   -[:TOUCHED]          ->(Module)
  (Review)   -[:CONTAINS_PATTERN] ->(BugPattern)
  (Developer)-[:TENDS_TO {
      confidence,        ← 0.0–1.0, tăng 0.2 mỗi lần quan sát
      evidence_count,    ← số PR đã thấy pattern này
      last_observed_at,
      invalidated_at     ← NULL = còn hiệu lực (bi-temporal)
  }]->(BugPattern)
```

### Luồng hoạt động

```
Trước mỗi review:
  MemoryStore.get_developer_profile(login)
  → Lấy các TENDS_TO edges có confidence cao (evidence >= 3)
  → Inject vào prompt: "Developer này hay gây: missing-null-check (80%), timezone-bug (60%)"

Sau mỗi review:
  LLM extractor trích xuất bug_patterns[] từ kết quả review
  → record_review(ReviewFact)
  → Upsert Developer, Review, Module nodes
  → _update_tendency(): tăng confidence trên TENDS_TO edge
  → Nếu evidence_count >= 3: edge được xét là "đủ tin cậy"
```

### Điểm khác biệt so với lưu trữ thông thường

| Đặc điểm | Giá trị |
|----------|---------|
| **Bi-temporal** | `invalidated_at` cho phép "quên" pattern khi dev đã sửa thói quen |
| **Confidence scoring** | Tích lũy bằng chứng qua nhiều PR, không phải boolean |
| **Graph traversal** | Query "Developer → TENDS_TO → BugPattern" trong O(1) |
| **Module hotspot** | Biết file nào hay bị bug, kết hợp với Risk Scorer |

**API:** `GET /api/developers` · `GET /api/developers/{login}`

---

## 8. Risk Scoring

Mỗi file review được tính risk score từ 6 tín hiệu (công thức từ CHID paper – Springer 2025):

| Tín hiệu | Trọng số | Nguồn |
|----------|----------|-------|
| Blast radius (# functions bị ảnh hưởng) | 20% | Neo4j CALLS graph |
| Coverage gap ratio | 25% | Git history |
| Bug frequency của file | 15% | Git log |
| Contributor churn | 15% | Git log |
| New contributor flag | 15% | GitHub API |
| PR size (lines changed) | 10% | Diff |

**Ngưỡng:** `< 0.30` = low · `0.30-0.55` = medium · `> 0.55` = high

---

## 9. Observability — Langfuse

Mọi LLM call được trace tự động qua Langfuse. Kể từ phiên bản hiện tại, mỗi **phase** trong pipeline được bọc bởi một **named span** riêng:

```
Trace (= 1 pipeline run: PR review / chat fix / swebench task)
  └── Span: phase1_file_localization     { file, metadata }
  └── Span: phase2_fault_localization    { file, metadata }
  └── Span: phase2_5_planner             { file, risk_level }
  └── Span: phase3_patch_generation      { file, attempt }  ← lặp lại mỗi retry
       └── Generation (= 1 LLM call, auto-created by LangChain callback)
            metadata: model, tokens, latency, prompt, response
```

**Trace IDs:** review_id · `fix-{project_id}` · `chat-{uuid}` · `swebench-{instance_id}`

**Bật:** `LANGFUSE_ENABLED=true` trong `.env` (opt-in, không ảnh hưởng pipeline nếu tắt)

**Triển khai:** `src_bot/observability/langfuse_ctx.py` có hai primitives:
- `langfuse_trace(trace_id, ...)` — context manager bọc toàn bộ pipeline run
- `langfuse_span(name, metadata=...)` — context manager bọc từng phase (no-op khi disabled)

---

## 10. GitHub Integration

| Feature | Mô tả |
|---------|-------|
| **GitHub Token** | Đọc PR data (title, files, base commit) |
| **GitHub OAuth** | Login UI qua GitHub account |
| **Webhook** | `pull_request` event → tự trigger review khi PR mở |
| **Webhook push** | `push` to default branch → incremental re-index |
| **Webhook secret** | HMAC-SHA256 verification |

---

## 11. Persistence

| Data | Storage | Mục đích |
|------|---------|---------|
| **Code Property Graph** | Neo4j (labels: Module/Class/Function) | AST nodes + CALLS/IMPORTS/INHERITS edges, BM25 + vector index |
| **Behavioral Memory Graph** | Neo4j (labels: Developer/BugPattern/Review/Module) | TENDS_TO edges với confidence scoring, developer profiles |
| Projects/Reviews/Sessions | SQLite (`data/reviews.db`) | Metadata, chat history |
| Repo clones | Disk (`data/projects/`) | Source code để đọc file khi review |
| PR working dirs | `/tmp/` | Per-review checkout (git clone --shared), tự xóa sau xong |

---

## 12. Tech Stack

| Layer | Technology |
|-------|-----------|
| **Backend** | Python 3.9+, FastAPI, uvicorn |
| **Frontend** | React 18, TypeScript, TailwindCSS, Vite |
| **LLM Framework** | LangChain (ChatGroq, ChatOllama, ChatOpenAI) |
| **LLM — fast_gate** | Groq API · llama-3.1-8b-instant |
| **LLM — chat** | vLLM (self-hosted) · DeepSeek-R1-Distill-Qwen-32B |
| **LLM — generation** | vLLM (self-hosted) · Qwen2.5-Coder-32B-Instruct |
| **Graph DB** | Neo4j 5.x (Code Graph + Memory Graph + Vector Index) |
| **Embedding** | sentence-transformers · all-MiniLM-L6-v2 (384-dim) |
| **Observability** | Langfuse (opt-in) |
| **Persistence** | SQLite (aiosqlite) |
| **GitHub** | PyGithub, requests, HMAC webhook |
| **Testing** | pytest, httpx.AsyncClient + ASGITransport |

---

## 13. Cấu trúc thư mục

```
CodeReviewBot/
├── api/
│   ├── main.py              # FastAPI routes: /api/projects, /reviews, /chat, /webhook
│   ├── reviewer.py          # PR review pipeline (SSE stream, phase spans)
│   ├── project_indexer.py   # Clone + AST ingest vào Neo4j
│   ├── database.py          # SQLite CRUD (aiosqlite)
│   ├── models.py            # Pydantic schemas
│   ├── github_app.py        # OAuth + webhook HMAC
│   └── agent/
│       ├── loop.py          # Chat agent turn loop (max 4 steps)
│       ├── tools.py         # 7 tools (add_project, review_pr, fix_bug…)
│       ├── classifier.py    # Intent classifier (fast_gate role)
│       ├── planner.py       # PlannerContract generator
│       ├── fix_bug.py       # Bug fix pipeline
│       ├── refactor.py      # Refactor pipeline + smell context
│       ├── evaluator.py     # LLM patch evaluator (contract-aware)
│       ├── smell_detector.py # Code smell detection (long method, god class…)
│       └── code_context.py  # CodeContext builder cho chat queries
├── src_bot/
│   ├── llm/router.py        # Tri-role LLM router (fast_gate/chat/generation)
│   ├── config/config.py     # Tất cả env vars (Pydantic Settings)
│   ├── graph_retriever.py   # Hybrid search: BM25 + Neo4j vector (RRF)
│   ├── memory/
│   │   ├── store.py         # MemoryStore — Neo4j TENDS_TO edges
│   │   ├── schema.py        # Developer, BugPattern, ReviewFact models
│   │   ├── extractor.py     # LLM extractor: facts từ review output
│   │   ├── retriever.py     # Developer profile lookup trước review
│   │   └── memory_neo4j.py  # Async wrapper
│   ├── reflexion/
│   │   ├── reflector.py     # generate_reflection() — verbal self-reflection
│   │   ├── evaluator.py     # Patch evaluator + test runner
│   │   └── patch_generator.py
│   ├── risk/
│   │   ├── scorer.py        # RiskScorer: 6-signal CHID formula
│   │   └── git_enrichment.py # bug_frequency, contributor_churn từ git log
│   ├── observability/
│   │   ├── langfuse_ctx.py  # langfuse_trace() + langfuse_span() context managers
│   │   └── langfuse_handler.py # LangChain CallbackHandler factory
│   └── verification/
│       ├── ast_gate.py      # Gate 1: AST parse in-memory
│       ├── test_gate.py     # Gate 2: git apply → pytest → revert
│       └── pipeline.py      # Orchestrate 3 gates cheapest-first
├── swebench/
│   ├── neo4j_ingest.py      # AST parser → Neo4j (Module/Class/Function + edges)
│   └── runner.py            # SWE-bench task runner
├── frontend/src/
│   ├── pages/               # Dashboard, Projects, Reviews, Chat, Developers
│   ├── components/          # StatusBadge, RiskBadge, PatchViewer…
│   └── api/                 # Fetch wrappers + SSE client
├── tests/                   # 126 backend tests
│   ├── test_reviewer.py     # Integration: full SSE pipeline
│   ├── test_api_endpoints.py # API endpoint tests
│   ├── test_risk_scorer.py  # Unit: RiskScorer CHID formula
│   ├── test_reflector.py    # Unit: generate_reflection prompt
│   ├── test_memory_store.py # Unit: MemoryStore với mock Neo4j driver
│   ├── test_pipeline_integration.py
│   ├── test_evaluator.py
│   ├── test_classifier.py
│   ├── test_smell_detector.py
│   ├── test_code_context.py
│   └── test_smoke.py
├── config/risk_weights.yaml  # Trọng số risk scoring (tunable)
├── run_swebench_v2.py        # Phase 1/2/3 implementations + SWE-bench harness
├── run_ingesting.py          # Batch ingest script
├── docker-compose.yml        # Neo4j + Backend + Frontend
├── .github/workflows/ci.yml  # CI: lint + test --cov + npm build + npm test
├── pyproject.toml            # ruff + pytest + coverage config
├── init.sh                   # One-command setup script
└── .env-example              # Template env vars với giải thích đầy đủ
```

---

## 14. Cách chạy

```bash
# 1. Bật Neo4j (Docker)
docker run -d --name neo4j -p 7687:7687 -e NEO4J_AUTH=neo4j/pass neo4j:5.20

# 2. Bật vLLM servers (2 GPU A100)
# GPU 0 — Qwen2.5-Coder-32B (generation)
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-Coder-32B-Instruct \
  --dtype float16 --max-model-len 16384 --port 8000

# GPU 1 — DeepSeek-R1-32B (chat/orchestration)
CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
  --dtype float16 --max-model-len 16384 --port 8001

# 3. Setup
cp .env-example .env   # điền GROQ_API_KEY, GITHUB_TOKEN, Neo4j password
bash init.sh           # install deps + init SQLite DB

# 4. Chạy
uvicorn api.main:app --reload --port 8000   # Backend
cd frontend && npm run dev                   # Frontend → http://localhost:5173

# 5. Sử dụng
# Vào UI → nhập GitHub repo URL → hệ thống tự clone + index
# Sau khi indexed → nhập PR URL để review
# Hoặc dùng Chat để "Fix bug in owner/repo: ..."
```

---

## 15. Điểm khác biệt so với review bot thông thường

| Tính năng | Bot thông thường | Bot này |
|-----------|-----------------|---------|
| Hiểu code | Đọc file thô | Knowledge graph (CALLS, IMPORTS, INHERITS) |
| Định vị lỗi | Đọc toàn bộ diff | 3-phase pipeline (file → function → line) |
| Chất lượng patch | Không kiểm tra | AST gate + test execution + LLM evaluate |
| Retry khi sai | Không | Reflexion loop với verbal self-reflection |
| Nhớ lịch sử | Không | Behavioral Memory Graph — TENDS_TO edges với confidence |
| Debug LLM | Không | Langfuse trace mọi call |
| Kế hoạch trước khi code | Không | PlannerContract (root cause + acceptance criteria) |
