# Pipeline — GraphRAG Code Review Bot

Toàn bộ luồng xử lý từ khi user gửi request đến khi trả về kết quả review / patch.

---

## Table of Contents

1. [Tổng quan Pipeline](#1-tổng-quan-pipeline)
2. [Stage 0 — Intent Classifier](#2-stage-0--intent-classifier)
3. [Stage 1 — PR Review Pipeline](#3-stage-1--pr-review-pipeline)
4. [Stage 2 — Planner](#4-stage-2--planner)
5. [Stage 3 — Generator](#5-stage-3--generator)
6. [Stage 4 — Evaluator](#6-stage-4--evaluator)
7. [Stage 5 — Reflexion Loop](#7-stage-5--reflexion-loop)
8. [Memory Pipeline](#8-memory-pipeline)
9. [Project Indexing Pipeline](#9-project-indexing-pipeline)
10. [SWE-bench Evaluation Pipeline](#10-swe-bench-evaluation-pipeline)
11. [Luồng dữ liệu đầy đủ (End-to-End)](#11-luồng-dữ-liệu-đầy-đủ-end-to-end)

---

## 1. Tổng quan Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          USER REQUEST                                       │
│              (Chat message hoặc PR URL gửi lên API)                        │
└──────────────────────────────┬──────────────────────────────────────────────┘
                               │
                               ▼
              ┌────────────────────────────────┐
              │    STAGE 0: Intent Classifier   │  ← llama-3.1-8b-instant
              │   IN_SCOPE / OUT_SCOPE / CLARIFY│     (Groq, ~50ms)
              └────────────┬───────────────────┘
                           │
          ┌────────────────┼────────────────┐
          │                │                │
          ▼                ▼                ▼
     OUT_SCOPE          CLARIFY          IN_SCOPE
    Hard refusal     Ask for info      Continue ✓
    (no LLM cost)    (no LLM cost)
                                           │
                               ┌───────────▼───────────┐
                               │  STAGE 1: PR Reviewer  │
                               │  Risk Score + Diff     │
                               └───────────┬───────────┘
                                           │
                               ┌───────────▼───────────┐
                               │  STAGE 2: Planner      │  ← llama-3.3-70b
                               │  GraphRAG Context      │     -versatile
                               │  Root Cause Analysis   │
                               └───────────┬───────────┘
                                           │
                               ┌───────────▼───────────┐
                               │  STAGE 3: Generator    │  ← llama-3.3-70b
                               │  ACI (line numbers)    │     -versatile
                               │  Unified Diff Patch    │
                               └───────────┬───────────┘
                                           │
                               ┌───────────▼───────────┐
                               │  STAGE 4: Evaluator    │  ← llama-3.1-8b
                               │  Score patch 1–5       │     -instant
                               └───────────┬───────────┘
                                           │
                          ┌────────────────┼─────────────────┐
                          │                                   │
                    score < 3                           score ≥ 3
                          │                                   │
                          ▼                                   ▼
              ┌───────────────────────┐          ┌─────────────────────┐
              │  STAGE 5: Reflexion   │          │   ACCEPT PATCH ✓    │
              │  Self-critique        │          │   Stream to user    │
              │  Retry (max 3 lần)    │          │   Save to DB        │
              └─────────┬─────────────┘          └─────────────────────┘
                        │ retry
                        └──────────► STAGE 2 (next attempt)
```

---

## 2. Stage 0 — Intent Classifier

**Mục đích:** Chặn mọi câu hỏi nằm ngoài phạm vi (không phải review PR / fix bug) TRƯỚC KHI gọi LLM lớn. Tiết kiệm token, bảo vệ scope.

**File:** `api/agent/classifier.py`

**Model:** `llama-3.1-8b-instant` (Groq) — fast, free, ~50ms latency

```
User message
      │
      ▼
┌─────────────────────────────────────────────────────┐
│              CLASSIFIER PROMPT                      │
│                                                     │
│  "Classify if this is about:                        │
│   - PR review / code review → IN_SCOPE              │
│   - Bug fixing → IN_SCOPE                           │
│   - Anything else → OUT_SCOPE                       │
│   - Missing info (no repo/PR) → CLARIFY"            │
│                                                     │
│  Reply JSON: {"intent": "...", "missing_info": "..."}│
└─────────────────────────────────────────────────────┘
      │
      ├── IN_SCOPE  ──────────────► Continue to Stage 1
      │
      ├── OUT_SCOPE ──────────────► Return hard refusal:
      │                             "I only help with PR review
      │                              and bug fixing."
      │
      └── CLARIFY  ──────────────► Return clarification request:
                                   "I need more info: {missing_info}"
```

**Fail-safe:** Nếu classifier lỗi (API timeout, parse error) → mặc định `IN_SCOPE` (fail open — không bao giờ block user vì lỗi classifier).

**Kỹ thuật:** Dùng `.replace("REPLACE_MESSAGE", message)` thay vì `.format()` để tránh `KeyError` khi prompt chứa `{}` trong ví dụ JSON.

---

## 3. Stage 1 — PR Review Pipeline

**Mục đích:** Phân tích PR, tính risk score, lấy context từ graph, gửi nhận xét review.

**File:** `api/reviewer.py`

```
PR URL (ví dụ: github.com/user/repo/pull/42)
      │
      ▼
┌─────────────────┐
│  1. Fetch PR    │  PyGithub → title, description, diff, changed files
│     from GitHub │  unidiff  → parse unified diff → list[PatchedFile]
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  2. Risk Score  │  CHID model (6 dimensions):
│     (CHID)      │  blast_radius × 0.20
│                 │  coverage_gap × 0.25
│                 │  bug_frequency × 0.15
│                 │  contributor_churn × 0.15
│                 │  new_contributor × 0.15
│                 │  pr_size × 0.10
│                 │  ─────────────────────
│                 │  score < 0.30 → LOW
│                 │  score 0.30–0.55 → MEDIUM
│                 │  score > 0.55 → HIGH
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  3. Check       │  data/projects/{project_id}/ tồn tại không?
│     Workspace   │  Nếu không → raise RuntimeError (user phải reindex)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  4. GraphRAG    │  Neo4j multi-hop traversal từ changed files:
│     Context     │  TOP_K=20, MAX_HOPS=3, MAX_NODES=500
│                 │  → code context (functions, classes, callers)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  5. Memory      │  Graphiti semantic search:
│     Context     │  build_memory_context(author, repo)
│                 │  → past patterns của developer này
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  6. Stage 2:    │  Gọi Planner → Generator → Evaluator
│     Planner     │  (chi tiết ở các stage tiếp theo)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  7. Record      │  graphiti.record_review(author, repo, pr, summary)
│     to Memory   │  → lưu vào Graphiti cho lần sau
└────────┬────────┘
         │
         ▼
   Return ReviewResult
   { summary, patch, risk_score, fix_attempts, status }
```

---

## 4. Stage 2 — Planner

**Mục đích:** Phân tích bug, xác định root cause, chỉ ra đúng file + function cần sửa.

**Model:** `llama-3.3-70b-versatile` (Groq)

**ACI Design (Agent-Computer Interface):**
Code được inject vào prompt CÓ số dòng để LLM target đúng line:

```
 289 | def process_request(self, data):
 290 |     if data is None:
 291 |         return {}
 292 |     result = self._transform(data)   ← LLM biết đây là line 292
 293 |     return result
```

```
PLANNER PROMPT
      │
      ├── Bug description (từ PR / issue)
      ├── Changed files list
      ├── GraphRAG context (functions, call graph, 500 nodes max)
      ├── Memory context (past patterns của developer)
      └── Code với line numbers (ACI)
      │
      ▼
LLM OUTPUT:
  - Root cause analysis
  - File(s) cần sửa
  - Specific lines liên quan
  - Approach đề xuất
```

---

## 5. Stage 3 — Generator

**Mục đích:** Tạo unified diff patch từ phân tích của Planner.

**Model:** `llama-3.3-70b-versatile` (Groq)

**File:** `src_bot/reflexion/patch_generator.py`

```
GENERATOR PROMPT
      │
      ├── Planner output (root cause + approach)
      ├── Code với line numbers
      └── "Respond ONLY with a ```diff ... ``` block"
      │
      ▼
LLM OUTPUT (raw):
  ```diff
  --- a/src/module.py
  +++ b/src/module.py
  @@ -289,7 +289,7 @@
       if data is None:
           return {}
  -    result = self._transform(data)
  +    result = self._transform(data or {})
       return result
  ```
      │
      ▼
┌────────────────────────────────────────┐
│         PATCH POST-PROCESSING          │
│                                        │
│  1. Extract diff block                 │
│     (strip markdown fences)            │
│                                        │
│  2. Fix blank lines                    │
│     "" → " " (single space)            │
│     (git apply requirement)            │
│                                        │
│  3. _normalize_patch()                 │
│     SequenceMatcher(threshold=0.6)     │
│     fuzzy-match deletion lines vs      │
│     actual file content                │
│     → rebuild correct @@ headers      │
│     → use actual file lines for "-"   │
│                                        │
│  4. git apply --check --recount        │
│     fallback: patch -p1 --fuzz=3       │
└────────────────────────────────────────┘
      │
      ▼
   Clean patch ready for Evaluator
```

---

## 6. Stage 4 — Evaluator

**Mục đích:** LLM thứ hai độc lập chấm điểm patch. Reject nếu score < 3.

**Model:** `llama-3.1-8b-instant` (Groq, fast & free)

**File:** `api/agent/evaluator.py`

```
Input: (bug_description[:500], patch[:1500])
      │
      ▼
┌─────────────────────────────────────────────┐
│              EVALUATOR PROMPT               │
│                                             │
│  "Score this patch 1-5:                     │
│   5 = Directly fixes bug, minimal, no regressions
│   4 = Fixes but over-engineered             │
│   3 = Partially addresses, may miss cases   │
│   2 = Wrong approach                        │
│   1 = Completely wrong / regressions        │
│                                             │
│   Reply ONLY: {"score": N, "reason": "..."}"│
└─────────────────────────────────────────────┘
      │
      ▼
  Parse JSON response
      │
      ├── score ≥ 3 ──────────────► ACCEPT → return patch
      │
      └── score < 3 ──────────────► REJECT → Reflexion Stage
                                    with reason as feedback

FAIL-SAFE: Nếu API lỗi → return score=3 (fail open)
```

**Frustration Detection** (trong `api/reviewer.py`):
```python
if last_patch and len(new_patch) < len(last_patch) * 0.6:
    # Patch đang co lại → agent bị confused
    # → Tăng RETRIEVER_TOP_K, inject hint nhìn file xung quanh
```

---

## 7. Stage 5 — Reflexion Loop

**Mục đích:** Tự phê bình patch thất bại, sinh insight mới, retry với context tốt hơn.

**File:** `src_bot/reflexion/reflector.py`

```
Rejected patch (score < 3)
      │
      ▼
┌─────────────────────────────────────────────┐
│           REFLECTION PROMPT                 │
│                                             │
│  Bug description: ...                       │
│  Failed patch: ...                          │
│  Evaluator feedback: "score 2/5: Wrong      │
│    approach, doesn't fix root cause"        │
│  Graph context: ...                         │
│  Previous reflections: [attempt 1: ...,    │
│                          attempt 2: ...]    │
│                                             │
│  → Generate self-critique + new approach   │
└─────────────────────────────────────────────┘
      │
      ▼
  Reflection text
      │
      ▼
┌─────────────────────────────────────────────┐
│           RETRY LOOP                        │
│                                             │
│  attempt 1 → Generate → Evaluate           │
│  attempt 2 → Reflect → Generate → Evaluate │
│  attempt 3 → Reflect → Generate → Evaluate │
│  attempt N → Give up, return best patch    │
│                                             │
│  Max retries: REFLEXION_MAX_RETRIES=3       │
│  --no-reflexion flag: max_retries=1         │
│  (saves ~3x tokens cho SWE-bench runs)      │
└─────────────────────────────────────────────┘
```

---

## 8. Memory Pipeline

**Mục đích:** Ghi nhớ pattern của từng developer qua các PR, inject vào review sau.

**Stack:** Graphiti-core + GroqClient + SentenceTransformer (all free)

```
┌──────────────────────────────────────────────────────────┐
│                    WRITE PIPELINE                        │
│              (sau khi review hoàn tất)                   │
└──────────────────────────────────────────────────────────┘

Review completed
      │
      ▼
graphiti.record_review(
    developer_login = "john_doe",
    repo_name       = "myorg/myrepo",
    pr_number       = 42,
    summary         = "Fixed null pointer in auth module",
    bug_patterns    = ["null check", "auth", "edge case"]
)
      │
      ▼
Graphiti Episode Ingestion:
  1. LLM extracts entities:
     - Developer: john_doe
     - File: auth/middleware.py
     - Bug type: NullPointerException
     - Fix pattern: null guard
  2. LLM extracts relationships:
     - john_doe --FIXED_BUG_IN--> auth/middleware.py
     - auth/middleware.py --HAS_PATTERN--> null_check
  3. Embedder (all-MiniLM-L6-v2) embeds episode text
  4. Store in Neo4j (same DB as code graph)


┌──────────────────────────────────────────────────────────┐
│                    READ PIPELINE                         │
│                 (khi bắt đầu review mới)                 │
└──────────────────────────────────────────────────────────┘

New review request (john_doe, myorg/myrepo)
      │
      ▼
graphiti.build_memory_context(
    developer_login = "john_doe",
    repo_name       = "myorg/myrepo"
)
      │
      ▼
Semantic search trên graph:
  Query: "john_doe myorg/myrepo recent patterns"
  → Top-K episodes retrieved
  → Format thành text context
      │
      ▼
Inject vào Planner prompt:
  "Developer history: john_doe has previously
   fixed 3 null-pointer bugs in auth module.
   Common pattern: adding None checks before
   attribute access."
```

---

## 9. Project Indexing Pipeline

**Mục đích:** Clone repo, phân tích code, build knowledge graph trong Neo4j + Weaviate.

**File:** `api/project_indexer.py`

```
POST /api/projects/{id}/index
      │
      ▼
┌─────────────────────────────────────────────┐
│  1. Clone Repository                        │
│     git clone {repo_url}                    │
│     destination: data/projects/{project_id} │
│     (persistent — survives server restart)  │
└────────────┬────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────┐
│  2. Parse Source Files                      │
│     Walk all .py / .js / .ts / etc files   │
│     AST parsing → extract:                 │
│       - Functions & methods                 │
│       - Classes                             │
│       - Import statements                   │
│       - Function call edges                 │
└────────────┬────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────┐
│  3. Ingest to Neo4j                         │
│     Nodes: File, Function, Class, Module    │
│     Edges: IMPORTS, CALLS, DEFINES,         │
│            CONTAINS, INHERITS               │
│     Commit history: MODIFIED_BY             │
└────────────┬────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────┐
│  4. Ingest to Weaviate                      │
│     Chunk code by function/class            │
│     Embed with all-MiniLM-L6-v2            │
│     Store: {file, function, code, embedding}│
└────────────┬────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────┐
│  5. Git History Enrichment                  │
│     GitPython → commit log                  │
│     Extract: bug-fix commits (by message)   │
│     Count: file change frequency            │
│     Track: contributor churn per file       │
└────────────┬────────────────────────────────┘
             │
             ▼
   Project status → "indexed" ✓
   SSE stream progress to frontend
```

---

## 10. SWE-bench Evaluation Pipeline

**Mục đích:** Chạy pipeline trên 300 tasks SWE-bench Lite, đo tỷ lệ patch đúng.

**File:** `run_swebench_v2.py`

```
SWE-bench Lite dataset (princeton-nlp/SWE-bench_Lite)
300 tasks × {repo, issue, fail_to_pass_tests}
      │
      ▼
┌─────────────────────────────────────────────┐
│  For each task:                             │
│                                             │
│  1. Clone target repo at base commit        │
│  2. Read issue description                  │
│  3. Inject to Reviewer pipeline             │
│     (Stage 1 → 2 → 3 → 4 → 5)              │
│  4. Apply patch to repo                     │
│  5. Write to predictions.jsonl              │
└────────────┬────────────────────────────────┘
             │
             ▼
predictions.jsonl format:
{
  "instance_id": "django__django-12345",
  "model_patch": "--- a/file.py\n+++ b/file.py\n...",
  "model_name_or_path": "codereviewbot-v2"
}
             │
             ▼
┌─────────────────────────────────────────────┐
│  Official SWE-bench Harness (Docker)        │
│                                             │
│  python -m swebench.harness.run_evaluation  │
│    --predictions_path predictions.jsonl     │
│    --run_id my_run                          │
│                                             │
│  Harness:                                   │
│  - Applies patch in Docker container        │
│  - Runs FAIL_TO_PASS tests                 │
│  - Reports resolved / total                 │
└────────────┬────────────────────────────────┘
             │
             ▼
   Score: resolved_count / 300
   (e.g. 15/300 = 5% resolve rate)
```

**Key flags:**
```bash
# Tắt reflexion để tiết kiệm token (3x faster)
python run_swebench_v2.py --no-reflexion --max-tasks 50

# Chạy đầy đủ 300 tasks
python run_swebench_v2.py --no-reflexion
```

---

## 11. Luồng dữ liệu đầy đủ (End-to-End)

```
USER
 │  "Review PR github.com/org/repo/pull/42"
 │
 ▼
[FRONTEND - React]
 │  POST /api/chat  {message, session_id}
 │  ← SSE stream response
 │
 ▼
[FASTAPI - api/main.py]
 │  Route to agent loop
 │
 ▼
[STAGE 0 - Classifier - api/agent/classifier.py]
 │  llama-3.1-8b-instant → IN_SCOPE ✓
 │
 ▼
[AGENT LOOP - api/agent/loop.py]
 │  Parse PR URL from message
 │  Call tool: create_review(repo, pr_number)
 │
 ▼
[TOOL LAYER - api/agent/tools.py]
 │  Validate workspace exists
 │  Call: api/reviewer.py
 │
 ▼
[STAGE 1 - PR REVIEWER - api/reviewer.py]
 │
 ├─► PyGithub → fetch PR diff + metadata
 │
 ├─► CHID Risk Scorer → score = 0.62 (HIGH)
 │
 ├─► Neo4j GraphRAG → 847 nodes, trimmed to 500
 │
 ├─► Graphiti Memory → "Developer fixed 2 null bugs before"
 │
 ├─► STAGE 2: Planner [llama-3.3-70b-versatile]
 │      Input:  bug desc + graph context + memory + code w/ line numbers
 │      Output: "Root cause at line 292 in auth/middleware.py,
 │               _transform() not handling None input"
 │
 ├─► STAGE 3: Generator [llama-3.3-70b-versatile]
 │      Input:  planner output + code w/ line numbers
 │      Output: raw diff block
 │      Post-process:
 │        → Extract diff
 │        → Fix blank lines
 │        → _normalize_patch() (fuzzy match + rebuild @@ headers)
 │        → git apply --check --recount ✓
 │
 ├─► STAGE 4: Evaluator [llama-3.1-8b-instant]
 │      Score: 4/5 ✓ "Directly fixes null handling, minimal change"
 │
 ├─► Patch ACCEPTED (score ≥ 3)
 │
 └─► Graphiti record_review() → save to memory graph
 │
 ▼
[FASTAPI - SSE Stream]
 │  yield {"type": "tool_result", "patch": "...", "score": 4}
 │  yield {"type": "message", "text": "Review complete. Risk: HIGH..."}
 │  yield {"type": "done"}
 │
 ▼
[FRONTEND - Chat.tsx]
 │  Render: PipelineProgress, PatchViewer, RiskBadge
 │
 ▼
USER SEES:
  ✅ Risk: HIGH (0.62)
  ✅ Patch (4/5): null guard added at line 292
  ✅ Summary: "auth/middleware.py missing None check..."
```

---

## Tóm tắt Models theo Stage

| Stage | Model | Provider | Vai trò | Latency |
|---|---|---|---|---|
| 0 — Classifier | `llama-3.1-8b-instant` | Groq | Gate intent | ~50ms |
| 2 — Planner | `llama-3.3-70b-versatile` | Groq | Root cause | ~2–5s |
| 3 — Generator | `llama-3.3-70b-versatile` | Groq | Patch creation | ~3–8s |
| 4 — Evaluator | `llama-3.1-8b-instant` | Groq | Score patch | ~100ms |
| 5 — Reflector | `llama-3.3-70b-versatile` | Groq | Self-critique | ~2–5s |
| Memory extract | `llama-3.1-8b-instant` | Groq | Entity/rel extract | background |

---

## Harness Engineering — 4 nguyên tắc áp dụng

| Nguyên tắc | Áp dụng trong project |
|---|---|
| **Tools & Permissions** | Intent classifier gate → chặn out-of-scope trước LLM lớn |
| **Memory & Recovery** | Graphiti episodic memory → inject developer history vào mỗi review |
| **Multi-Agent Topology** | Pipeline shape: Classifier → Planner → Generator → Evaluator → Reflector |
| **Observability** | Evaluator score + frustration detection → feedback loop cho retry |

*"Anytime an agent makes a mistake, engineer a solution so it never makes that mistake again."*
— Harness Engineering, Alice.io 2026

---

*Last updated: 2026-05-30 | Branch: `chatbot_dev`*
