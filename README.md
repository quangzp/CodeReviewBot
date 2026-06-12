# GraphRAG Code Review Bot

> **Tự động review PR, phát hiện bug và generate patch** bằng Neo4j knowledge graph + LLM + Reflexion loop.
> Không hallucinate như LLM thuần túy vì hiểu cấu trúc code thực sự — ai gọi ai, ai import gì, blast radius.

---

## Tính năng

| Khả năng | Mô tả |
|----------|-------|
| **Review PR** | Phân tích từng file thay đổi, generate patch với unified diff |
| **Fix bug** | Nhận mô tả lỗi bằng ngôn ngữ tự nhiên → tạo patch |
| **Refactor** | Detect code smell + tự động refactor theo mô tả |
| **Chat agent** | Giao tiếp tự nhiên, agent tự chọn tool phù hợp |
| **Developer memory** | Nhớ pattern bug của từng dev qua nhiều PR |
| **Risk scoring** | Đánh giá mức độ rủi ro từng file (CHID formula) |
| **Verification gates** | AST parse → test execution → LLM evaluate trước khi đề xuất patch |
| **Reflexion loop** | Tự phân tích thất bại → retry thông minh hơn |

---

## Kiến trúc tổng thể

```
┌──────────────────────────────────────────────────┐
│              Frontend (React + TypeScript)        │
│  Chat · Projects · Reviews · Developers · SSE     │
└─────────────────────┬────────────────────────────┘
                      │ REST + Server-Sent Events
┌─────────────────────▼────────────────────────────┐
│           FastAPI Backend  :8000                  │
│  /api/projects  /api/reviews  /api/chat           │
│  /api/developers  /webhook  /health               │
└──────┬───────────┬───────────────┬───────────────┘
       │           │               │
  Project      Review          Chat Agent
  Indexer     Pipeline          (loop.py)
  (AST→Neo4j) (3-phase)     Intent→Tool→Execute
       │           │               │
┌──────▼───────────▼───────────────▼───────────────┐
│                  Core Engine                      │
│                                                   │
│  LLM Router (tri-role)   GraphRAG Retriever       │
│  ┌──────────────────┐    BM25 + Neo4j vector      │
│  │ fast_gate  8B    │    + graph traversal (RRF)  │
│  │ chat       32B   │                             │
│  │ generation 32B   │    Planner (PlannerContract) │
│  └──────────────────┘    Verification (3 gates)   │
│                          Reflexion (self-reflect)  │
│                          Risk Scorer (CHID)        │
│                          Langfuse (tracing)        │
└──────────────────┬────────────────────────────────┘
                   │
        ┌──────────▼──────────┐
        │       Neo4j 5.x     │
        │                     │
        │  Code Property Graph│  ← AST: Module/Class/Function
        │  (CALLS, IMPORTS,   │    CALLS, IMPORTS, INHERITS
        │   INHERITS, DEFINES)│    + BM25 + vector index
        │                     │
        │  Behavioral Memory  │  ← Developer/BugPattern/Review
        │  (Developer→TENDS_TO│    confidence scoring
        │   →BugPattern)      │    bi-temporal edges
        └──────────┬──────────┘
                   │
        ┌──────────▼──────────┐
        │  SQLite  │  Disk    │
        │ metadata │ clones   │
        │ + chat   │ data/    │
        └──────────┴──────────┘
```

---

## Cài đặt nhanh — Docker Compose

**Yêu cầu:** Docker Desktop, GPU (optional cho vLLM)

```bash
# 1. Clone và cấu hình
git clone https://github.com/<your-org>/CodeReviewBot.git
cd CodeReviewBot
cp .env-example .env
# Chỉnh sửa .env: điền GROQ_API_KEY, GITHUB_TOKEN, APP_NEO4J_PASSWORD

# 2. Khởi chạy toàn bộ stack
docker compose up -d

# Services chạy:
#   neo4j    → http://localhost:7474   (user: neo4j / pass: trong .env)
#   backend  → http://localhost:8000
#   frontend → http://localhost:80
```

**Kiểm tra:**
```bash
curl http://localhost:8000/health
# {"status":"ok","neo4j":"connected","db":"ok"}
```

---

## Cài đặt thủ công — Local Dev

### Yêu cầu

- Python 3.9+
- Node 20+
- Neo4j 5.x (Docker hoặc Neo4j Desktop)

### Backend

```bash
# 1. Virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. Cài dependencies
pip install -r requirements.txt

# 3. Biến môi trường
cp .env-example .env
# Điền tối thiểu: APP_NEO4J_PASSWORD, GROQ_API_KEY, GITHUB_TOKEN

# 4. Khởi Neo4j (Docker)
docker run -d --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/<your_password> \
  neo4j:5.20

# 5. Chạy backend
uvicorn api.main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
# → http://localhost:3000
```

### vLLM (self-hosted, cần GPU — tuỳ chọn)

```bash
# GPU 0 — Qwen2.5-Coder-32B (generation role)
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-Coder-32B-Instruct \
  --dtype float16 --max-model-len 16384 --port 8000

# GPU 1 — DeepSeek-R1-32B (chat/orchestration role)
CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
  --dtype float16 --max-model-len 16384 --port 8001
```

> **Không có GPU?** Dùng Groq free tier cho cả 3 roles — chỉ cần `GROQ_API_KEY`.
> Xem `.env-example` phần LLM Role Architecture.

---

## Biến môi trường quan trọng

| Biến | Bắt buộc | Mặc định | Mô tả |
|------|----------|---------|-------|
| `APP_NEO4J_PASSWORD` | ✓ | — | Mật khẩu Neo4j |
| `GROQ_API_KEY` | ✓ | — | Groq API key (lấy free tại groq.com) |
| `GITHUB_TOKEN` | ✓ | — | GitHub PAT để đọc PR data |
| `FAST_LLM_PROVIDER` | | `groq` | Provider cho role fast_gate |
| `FAST_LLM_MODEL` | | `llama-3.1-8b-instant` | Model fast_gate |
| `GEN_LLM_PROVIDER` | | `groq` | Provider cho role generation |
| `GEN_LLM_MODEL` | | `llama-3.3-70b-versatile` | Model generation |
| `CHAT_LLM_PROVIDER` | | `groq` | Provider cho role chat |
| `VLLM_GEN_BASE` | | — | URL vLLM server (nếu dùng self-hosted) |
| `LANGFUSE_ENABLED` | | `false` | Bật tracing LLM calls |
| `EXECUTION_GATE_ENABLED` | | `true` | Bật chạy test thật trước khi đề xuất patch |
| `REFLEXION_MAX_RETRIES` | | `3` | Số lần retry tối đa trong Reflexion loop |

Xem đầy đủ tại [`.env-example`](.env-example).

---

## Cách sử dụng

### 1. Onboard repo

```
UI: Projects → Add Project → Nhập GitHub repo URL → Submit
Bot tự động: git clone → AST parse → import vào Neo4j graph
```

### 2. Review Pull Request

```
UI: Reviews → New Review → Nhập PR URL (github.com/owner/repo/pull/123)
Bot: 3-phase pipeline → patch suggestions hiện real-time qua SSE stream
```

### 3. Chat agent

```
UI: Chat
Ví dụ:
  "Fix bug in owner/repo: NullPointerException in auth/login.py line 42"
  "Refactor the UserService class to use dependency injection"
  "Review PR https://github.com/owner/repo/pull/99"
```

### 4. GitHub Webhook (tự động)

```bash
# Cấu hình webhook tại GitHub repo → Settings → Webhooks:
URL: https://<your-domain>/webhook
Content type: application/json
Events: Pull requests, Pushes
Secret: <GITHUB_WEBHOOK_SECRET trong .env>
```

---

## Cấu trúc dự án

```
CodeReviewBot/
│
├── api/                         # FastAPI application
│   ├── main.py                  # Routes: /api/projects, /api/reviews, /api/chat, /webhook
│   ├── reviewer.py              # PR review pipeline (SSE stream)
│   ├── project_indexer.py       # Clone repo + AST ingest vào Neo4j
│   ├── database.py              # SQLite CRUD (aiosqlite)
│   ├── models.py                # Pydantic schemas (ProjectRecord, ReviewRecord…)
│   ├── github_app.py            # GitHub OAuth + webhook HMAC verification
│   └── agent/
│       ├── loop.py              # Chat agent turn loop (max 4 tool steps)
│       ├── tools.py             # 7 tools: add_project, review_pr, fix_bug, refactor…
│       ├── classifier.py        # Intent gate (fast_gate role — 8B model)
│       ├── planner.py           # PlannerContract: root_cause + acceptance_criteria
│       ├── evaluator.py         # LLM patch evaluator, contract-aware scoring
│       ├── fix_bug.py           # Bug fix pipeline
│       ├── refactor.py          # Refactor pipeline + smell detection
│       ├── smell_detector.py    # Code smell detection (long method, god class…)
│       └── code_context.py      # Build graph context cho chat queries
│
├── src_bot/                     # Core engine
│   ├── config/config.py         # Tất cả env vars (Pydantic Settings)
│   ├── llm/router.py            # Tri-role LLM router: fast_gate / chat / generation
│   ├── graph_retriever.py       # Hybrid search: BM25 + Neo4j vector + RRF merge
│   ├── memory/
│   │   ├── store.py             # MemoryStore: Neo4j TENDS_TO edges với confidence
│   │   ├── schema.py            # Developer, BugPattern, ReviewFact Pydantic models
│   │   ├── extractor.py         # LLM extractor: ReviewFact từ review output
│   │   ├── retriever.py         # Lấy developer profile trước mỗi review
│   │   └── memory_neo4j.py      # Async wrapper cho MemoryStore
│   ├── reflexion/
│   │   ├── reflector.py         # generate_reflection(): verbal self-reflection
│   │   ├── evaluator.py         # Patch evaluator + test runner (git apply → pytest)
│   │   └── patch_generator.py   # Patch generator helpers
│   ├── risk/
│   │   ├── scorer.py            # RiskScorer: 6-signal CHID formula
│   │   └── git_enrichment.py    # Bug frequency, contributor churn từ git log
│   ├── verification/
│   │   ├── ast_gate.py          # Gate 1: ast.parse() in-memory (free, ~100ms)
│   │   ├── test_gate.py         # Gate 2: git apply → pytest → revert
│   │   └── pipeline.py          # Orchestrate 3 gates cheapest-first
│   └── observability/
│       ├── langfuse_ctx.py      # ContextVar trace context + langfuse_span()
│       └── langfuse_handler.py  # LangChain CallbackHandler factory
│
├── swebench/                    # SWE-bench evaluation
│   ├── neo4j_ingest.py          # AST parser → Neo4j (Module/Class/Function nodes)
│   └── runner.py                # SWE-bench task runner
│
├── frontend/                    # React SPA
│   └── src/
│       ├── pages/               # Dashboard, Projects, Reviews, Chat, Developers
│       ├── components/          # StatusBadge, RiskBadge, PatchViewer, PipelineProgress
│       └── api/                 # Fetch wrappers + SSE client
│
├── tests/                       # pytest suite (147 tests total)
│   ├── test_reviewer.py         # 8 integration tests: full SSE pipeline với mocked LLM
│   ├── test_api_endpoints.py    # 16 API endpoint tests
│   ├── test_pipeline_integration.py  # 10 pipeline integration tests
│   ├── test_risk_scorer.py      # 14 unit tests: RiskScorer CHID formula
│   ├── test_reflector.py        # 10 unit tests: generate_reflection prompt content
│   ├── test_memory_store.py     # 17 unit tests: MemoryStore với mock Neo4j driver
│   ├── test_evaluator.py        # 7 tests: patch evaluator
│   ├── test_classifier.py       # 6 tests: intent classifier
│   ├── test_smoke.py            # 20 import + schema smoke tests
│   ├── test_code_context.py     # 6 tests: CodeContext builder
│   └── test_smell_detector.py   # 12 tests: smell detection
│
├── config/risk_weights.yaml     # Trọng số risk scoring (có thể chỉnh)
├── run_swebench_v2.py           # Phase 1/2/3 implementations + SWE-bench harness
├── run_ingesting.py             # Batch ingest script
├── docker-compose.yml           # Neo4j + Backend + Frontend
├── Dockerfile / Dockerfile.frontend
├── .github/workflows/ci.yml     # CI: lint + test (--cov) + frontend build + npm test
├── pyproject.toml               # ruff + pytest + coverage config
└── .env-example                 # Template biến môi trường
```

---

## Testing

```bash
# Backend — 126 tests
python -m pytest tests/ -v

# Với coverage report
python -m pytest tests/ --cov --cov-report=term-missing

# Frontend — 21 tests (Vitest)
cd frontend && npm test
```

**CI/CD:** Mỗi push lên `main` / `chatbot_dev` tự chạy:
1. Ruff lint
2. `pytest tests/ --cov` (fail nếu coverage < 35%)
3. `npm run build`
4. `npm test` (Vitest component tests)

---

## Pipeline 3-Phase

```
PR changed files
      │
      ▼
Phase 1 — File Localization
  BM25 full-text + Neo4j vector cosine (all-MiniLM-L6-v2, 384-dim)
  Merge bằng Reciprocal Rank Fusion (RRF)
  → Tìm đúng file liên quan đến bug
      │
      ▼
Phase 2 — Fault Localization
  LLM đọc file content + graph context (CALLS/IMPORTS neighbors)
  → Xác định function bị lỗi + nguyên nhân gốc rễ
      │
      ▼
Phase 2.5 — Planner
  LLM tạo PlannerContract:
    root_cause · approach · must_preserve
    acceptance_criteria · files_to_modify · risk_notes
      │
      ▼
Phase 3 — Patch Generation
  LLM viết unified diff patch (có ngữ cảnh từ PlannerContract)
      │
      ▼
Verification Pipeline
  Gate 1 — AST:   ast.parse() in-memory       (~100ms, free)
  Gate 2 — Tests: git apply → pytest → revert (tắt nếu không có tests)
  Gate 3 — LLM:   evaluate 1-5, contract-aware (tốn token nhất)
      │
      ▼ (nếu fail)
Reflexion Loop  (max 2-5 retries tuỳ risk level)
  Reflector phân tích tại sao fail → verbal self-reflection
  → Feed reflection vào Phase 3 retry
      │
      ▼
Final patch + risk score + SSE stream về frontend
```

---

## LLM Router — Tri-role

| Role | Model khuyến nghị | Provider | Dùng cho |
|------|-------------------|----------|---------|
| `fast_gate` | llama-3.1-8b-instant | Groq (free) | Intent classifier, patch evaluator (quyết định binary) |
| `chat` | DeepSeek-R1-Distill-Qwen-32B | vLLM self-hosted | Agent loop, tool routing, orchestration |
| `generation` | Qwen2.5-Coder-32B-Instruct | vLLM self-hosted | Phase 1/2/3, Planner, Fix bug, Refactor |

**Fallback:** Nếu vLLM down → tự động dùng Groq. Nếu chỉ có Groq → cả 3 roles dùng Groq 70B.

---

## Observability — Langfuse

Khi bật `LANGFUSE_ENABLED=true`, mọi LLM call được trace tự động:

```
Trace (1 pipeline run)
  └── Span: phase1_file_localization
  └── Span: phase2_fault_localization
  └── Span: phase2_5_planner
  └── Span: phase3_patch_generation  ← repeated per retry
       └── Generation (LLM call) — tokens, latency, prompt, response
```

Dashboard tại [cloud.langfuse.com](https://cloud.langfuse.com).

---

## API Reference

| Method | Endpoint | Mô tả |
|--------|----------|-------|
| `GET`  | `/health` | Health check + Neo4j/DB status |
| `POST` | `/api/projects` | Onboard repo mới |
| `GET`  | `/api/projects` | Danh sách projects |
| `GET`  | `/api/projects/{id}/stream` | SSE stream tiến trình indexing |
| `POST` | `/api/projects/{id}/reindex` | Re-index sau khi push |
| `POST` | `/api/reviews` | Tạo review mới |
| `GET`  | `/api/reviews/{id}/stream` | SSE stream kết quả review real-time |
| `GET`  | `/api/developers` | Danh sách developers đã được observe |
| `GET`  | `/api/developers/{login}` | Profile + bug patterns của 1 developer |
| `POST` | `/api/chat` | Gửi message trong chat session |
| `POST` | `/webhook` | GitHub webhook receiver |
| `GET`  | `/auth/login` | Bắt đầu GitHub OAuth flow |

---

## Tài liệu chi tiết

| Tài liệu | Nội dung |
|----------|---------|
| [`docs/SYSTEM_OVERVIEW.md`](docs/SYSTEM_OVERVIEW.md) | Kiến trúc sâu: 2 Neo4j graphs, memory schema, risk scoring, Langfuse |
| [`docs/AGENTS.md`](docs/AGENTS.md) | Hướng dẫn cho AI coding agents làm việc trên repo này |
| [`PIPELINE.md`](PIPELINE.md) | Chi tiết từng bước pipeline 3-phase |
| [`TECH_STACK.md`](TECH_STACK.md) | Tech stack và lý do lựa chọn |
| [`.env-example`](.env-example) | Template + giải thích mọi biến môi trường |

---

## Đóng góp

```bash
# 1. Tạo branch từ chatbot_dev
git checkout chatbot_dev
git checkout -b feature/your-feature

# 2. Code + test
python -m pytest tests/ -v
cd frontend && npm test

# 3. Lint
pip install ruff && ruff check . --select E,F,W,I --ignore E501,F401

# 4. Push và tạo PR → chatbot_dev
git push origin feature/your-feature
```

**Quy tắc:**
- Mỗi feature mới phải có test (ưu tiên unit test trước integration test)
- Không commit `.env` (đã gitignore)
- Không hardcode API key trong code — dùng `configs.*` từ `src_bot/config/config.py`
- Mọi LLM call phải đi qua `get_llm(role=...)` trong `src_bot/llm/router.py`
