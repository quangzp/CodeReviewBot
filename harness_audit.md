# 🔍 Harness Engineering Audit — CodeReviewBot

> Đối chiếu source code với chuẩn Anthropic (Nov 2025 + Mar 2026)

---

## Tóm tắt điểm quan trọng từ 2 bài Anthropic

### Bài 1: "Effective harnesses for long-running agents" (Nov 2025)
Vấn đề cốt lõi với long-running agents:
- **Context window hết** → mất trạng thái, agent phải đoán lại
- **Agent khai báo "xong" quá sớm** → bỏ sót features
- **Agent không test end-to-end** → bug ẩn qua code review

Giải pháp Anthropic:
1. **Initializer Agent** → setup môi trường, viết `init.sh`, `claude-progress.txt`, `feature_list.json`
2. **Coding Agent** → làm từng feature một, commit git, update progress file
3. **Browser automation testing** → test như user thật (Puppeteer MCP)

### Bài 2: "Harness design for long-running apps" (Mar 2026)
Bổ sung thêm:
- **Self-evaluation problem** → agent luôn khen output của mình → cần **Evaluator tách biệt**
- **Context anxiety** → agent vội "wrap up" khi context gần đầy → cần **Context Reset**
- **GAN-inspired loop** → Generator + Evaluator độc lập → feedback loop chất lượng cao
- **Sprint Contract** → Generator và Evaluator thỏa thuận "done" trước khi code

**3-agent architecture** (bài 2):
```
Planner → phân tích spec → feature list
Generator → code từng sprint, self-evaluate
Evaluator (Playwright) → test live app → grade → feedback → retry
```

---

## Phân tích Source Code

### ✅ Đã triển khai đúng theo Anthropic

---

#### ✅ 1. Intent Gate (Stage 0) — ĐÚNG HOÀN TOÀN

**File:** [`api/agent/classifier.py`](file:///Users/manhnguyen/Project/CodeReviewBot/api/agent/classifier.py)

```python
# loop.py line 207-224
intent, missing = await loop.run_in_executor(
    None, lambda: classify(user_message, history=history)
)
if intent == Intent.OUT_SCOPE:
    yield {"type": "message", "text": OUT_SCOPE_REPLY}
    return
```

**Đúng với Anthropic:** Classifier chạy trước mọi thứ, dùng model nhỏ (`8b-instant`), fail-open (nếu lỗi → `IN_SCOPE`). Đây chính xác là "gate" concept trong bài 2.

---

#### ✅ 2. Planner → Generator → Evaluator Pipeline — ĐÚNG (Spirit)

Project có đúng 3-agent pipeline:
- `Classifier (8b)` → gate intent
- `Planner (70b)` → root cause + acceptance criteria  
- `Generator (70b)` → unified diff patch
- `Evaluator (8b)` → score 1-5, reject nếu < 3

Đây chính xác là GAN-inspired loop mà bài 2 mô tả. **Evaluator là agent tách biệt**, không để Generator tự đánh giá mình.

---

#### ✅ 3. Reflexion Loop — ĐÚNG

**File:** [`src_bot/reflexion/reflector.py`](file:///Users/manhnguyen/Project/CodeReviewBot/src_bot/reflexion/reflector.py)

Khi patch bị reject → Reflector generate self-critique → inject vào lần retry tiếp theo. Đây chính xác là **"incremental progress + leave clean state"** pattern.

---

#### ✅ 4. Risk-based context adjustment — ĐÚNG

```python
# reviewer.py
if last_patch and len(new_patch) < len(last_patch) * 0.6:
    # Frustration detection → widen RETRIEVER_TOP_K
```

Đây là form của "frustration detection" — tương đương với bài 2's observation rằng agents cần feedback loop khi đi lạc hướng.

---

#### ✅ 5. Structured context injection (ACI) — ĐÚNG

Inject code với line numbers: `" 292 | def process_request(...)"`  
Đây là **Agent-Computer Interface** mà bài 2 nhấn mạnh giúp model target đúng dòng code.

---

### ⚠️ Triển khai nhưng chưa đủ chuẩn

---

#### ⚠️ 6. Sprint Contract (PlannerContract) — THIẾU NEGOTIATION

**File:** [`api/agent/planner.py`](file:///Users/manhnguyen/Project/CodeReviewBot/api/agent/planner.py)

Project có `PlannerContract` (root_cause, approach, acceptance_criteria, risk_notes), nhưng:

**Anthropic chuẩn (bài 2):**
> "Before each sprint, the generator and evaluator **negotiated** a sprint contract: agreeing on what 'done' looked like... The generator proposed what it would build, and the **evaluator reviewed that proposal** before any code was written. The two iterated until they agreed."

**Project hiện tại:** Planner tự tạo contract → truyền thẳng cho Generator, không có vòng lặp negotiation với Evaluator. Contract là **unilateral** (1 chiều), không phải **bilateral negotiation**.

**Mức độ:** ⚠️ Partial — tinh thần đúng nhưng thiếu feedback loop trong contract phase.

---

#### ⚠️ 7. Testing Gate — CÓ NHƯNG BỊ TẮT

**File:** [`src_bot/verification/test_gate.py`](file:///Users/manhnguyen/Project/CodeReviewBot/src_bot/verification/test_gate.py)

```python
# test_gate.py: git apply → pytest → revert
# Nhưng bị tắt nếu không có test suite
EXECUTION_GATE_ENABLED=true  # default
```

**Anthropic chuẩn (bài 1):** Test bằng browser automation như user thật (Puppeteer MCP).

**Project hiện tại:** Chỉ chạy pytest (unit test), không có E2E browser testing. Bài 2 của Anthropic nhấn mạnh Evaluator dùng **Playwright MCP** để navigate live app trước khi grade — project này không làm điều này.

**Mức độ:** ⚠️ Partial — có test gate nhưng thiếu E2E verification.

---

#### ⚠️ 8. Feature List / Progress Tracking — THIẾU

**Anthropic chuẩn (bài 1):**
```json
{
  "description": "User can open a new chat, type a query, and see AI response",
  "passes": false
}
```
Agent chỉ mark `"passes": true` sau khi test thực sự.

**Project hiện tại:** Có `feature_list.json` trong root folder, nhưng đây là **file tĩnh dùng cho SWE-bench harness**, không phải progress tracker động cho agent. Không có cơ chế agent update status feature sau mỗi sprint.

**Mức độ:** ⚠️ File tồn tại nhưng không được dùng đúng mục đích.

---

### ❌ Thiếu hoàn toàn

---

#### ❌ 9. Context Reset / Session Handoff — KHÔNG CÓ

**Anthropic chuẩn (bài 1 + 2):**
> "Context resets—clearing the context window entirely and starting a fresh agent, combined with a **structured handoff** that carries the previous agent's state and the next steps."

**Project hiện tại:**
```python
# loop.py line 233
for h in history[-10:]:  # last 10 turns — đơn giản là cắt bớt history
```

Agent chỉ giữ 10 turns cuối — đây **không phải** context reset. Không có:
- Progress file được cập nhật giữa các session
- Structured handoff artifact
- Context compaction

Khi session mới bắt đầu, agent không có cách nào biết được trạng thái của session trước. Đây là **điểm yếu lớn nhất** so với chuẩn Anthropic.

**Mức độ:** ❌ Hoàn toàn thiếu.

---

#### ❌ 10. Initializer Agent Pattern — KHÔNG CÓ

**Anthropic chuẩn (bài 1):**
> "Initializer agent: sets up `init.sh`, `claude-progress.txt`, and an initial git commit."

**Project hiện tại:** Không có agent nào chạy một lần duy nhất để setup môi trường trước khi coding agents hoạt động. `project_indexer.py` chỉ clone và ingest vào Neo4j, không phải Initializer Agent theo nghĩa Anthropic.

---

## Câu hỏi 2: Có dùng 2 Graph chưa?

### Thực trạng: **Cùng một Neo4j instance, 2 logical graph bằng cách phân biệt node labels**

```
╔══════════════════════════════════════════════════════════════╗
║                    Neo4j 5.x (single instance)               ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  📊 Code Property Graph (GraphRAG)                          ║
║  Labels: Module, Class, Function, Commit                     ║
║  Edges:  IMPORTS, CALLS, DEFINES, INHERITS, MODIFIED_BY     ║
║  Purpose: AST structure — ai gọi ai, blast radius            ║
║                                                              ║
║  🧠 Behavioral Memory Graph                                  ║
║  Labels: Developer, BugPattern, Review, Module, Topic        ║
║  Edges:  TENDS_TO, AUTHORED, CONTAINS_PATTERN, TOUCHED      ║
║          INTERESTED_IN                                       ║
║  Purpose: Developer patterns — ai hay mắc lỗi gì            ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
```

**Từ `store.py` line 4:**
> "Schema (lives in the **same Neo4j** as GraphRAG, but with **different labels**)"

### Đánh giá: ✅ Đúng concept, ⚠️ có rủi ro thiết kế

**Ưu điểm:**
- Concept đúng: 2 graph có mục đích khác nhau, node labels phân biệt rõ ràng
- Memory Graph có `Module` node overlap với Code Graph → có thể cross-query (`Review` → `TOUCHED` → `Module` → `CALLS` → tìm blast radius)
- Không cần managed 2 DB connection pool

**Nhược điểm / Rủi ro:**
- `Module` node bị dùng ở **cả 2 graph** (`Module` trong Code Graph vs `Module {path, project_id}` trong Memory) → có thể conflict nếu không có phân biệt rõ `project_id`
- Không có **database isolation** — nếu Memory Graph query bị chậm sẽ ảnh hưởng Code Graph query
- Khó scale riêng từng graph

### Còn thiếu: **Conversation History Graph**

Câu hỏi của bạn hỏi về "graph lưu lịch sử hội thoại". Hiện tại:

| Storage | Nơi lưu | Kiểu |
|---------|---------|------|
| Chat messages | `SQLite: chat_messages table` | Relational |
| Chat sessions | `SQLite: chat_sessions table` | Relational |
| Developer patterns | `Neo4j: Memory Graph` | Graph |

**Lịch sử hội thoại được lưu trong SQLite, KHÔNG phải trong Neo4j graph.** Khi agent cần context: chỉ load `history[-10:]` từ SQLite, không có graph-based retrieval.

**Chuẩn Anthropic cho conversation context:**
- `claude-progress.txt` → file text đơn giản, mỗi session append vào
- `feature_list.json` → JSON structured tracking

Project dùng SQLite + Neo4j Memory Graph cho developer patterns, nhưng **không có graph-native conversation history** (e.g., `(:Session)-[:CONTAINS]->(:Message)-[:REFERENCES]->(:Review)`).

---

## Bảng đánh giá tổng hợp

| Concept Anthropic | Có trong project? | Mức độ |
|---|---|---|
| Intent Gate (classifier before main LLM) | ✅ Classifier 8b-instant | Đầy đủ |
| Small model for gate, large for generation | ✅ 8b vs 70b | Đầy đủ |
| Planner → Generator → Evaluator | ✅ Pipeline 5-stage | Đầy đủ |
| Evaluator tách biệt (không self-evaluate) | ✅ Evaluator riêng | Đầy đủ |
| Reflexion / retry loop | ✅ max 3 retries | Đầy đủ |
| Fail-open on gate errors | ✅ fallback IN_SCOPE | Đầy đủ |
| ACI (line numbers in code) | ✅ `" 42 | def foo()"` | Đầy đủ |
| Sprint Contract negotiation | ⚠️ Unilateral chỉ | Thiếu Evaluator review |
| E2E browser/live testing | ⚠️ Chỉ pytest | Thiếu Playwright-style |
| Feature progress tracking | ⚠️ File tồn tại nhưng static | Không dùng đúng |
| Context Reset between sessions | ❌ Không có | Hoàn toàn thiếu |
| Initializer Agent | ❌ Không có | Hoàn toàn thiếu |
| Structured session handoff | ❌ Không có | Hoàn toàn thiếu |
| Code Graph (GraphRAG) | ✅ Neo4j Code Property Graph | Đầy đủ |
| Memory/Pattern Graph | ✅ Neo4j Behavioral Graph | Đầy đủ |
| Conversation History Graph | ❌ SQLite only | Không có graph |
| Dual-graph trong cùng Neo4j | ⚠️ Same DB, diff labels | Rủi ro thiết kế |

---

## Gợi ý cải tiến ưu tiên cao

### P0 — Context Reset & Session Handoff
```python
# Cần thêm: SessionHandoffWriter
class SessionHandoff:
    def write(self, session_id: str, state: dict):
        """Ghi progress file trước khi session kết thúc"""
        # data/sessions/{session_id}/handoff.json
        
    def read(self, session_id: str) -> dict:
        """Load state khi session mới bắt đầu"""
```

### P1 — Sprint Contract Negotiation
Thêm bước Evaluator review PlannerContract trước khi Generator chạy:
```
Planner → draft contract → Evaluator validates → approved? → Generator
                                              ↑ reject → Planner revise
```

### P2 — Feature Progress Tracking
Convert `feature_list.json` thành living document:
```json
{
  "features": [
    {"id": "F001", "desc": "...", "passes": false, "last_attempt": null}
  ]
}
```
Agent update `"passes": true` chỉ sau khi test pass.

### P3 — Tách Neo4j sang 2 database riêng (nếu scale)
```
NEO4J_CODE_URI=bolt://localhost:7687/code_graph
NEO4J_MEMORY_URI=bolt://localhost:7687/memory_graph
```
Hoặc dùng **named databases** trong Neo4j Enterprise.

---

*Audit dựa trên: Anthropic Engineering Blog (Nov 2025, Mar 2026) + source code review*
