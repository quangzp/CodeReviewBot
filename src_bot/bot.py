from __future__ import annotations

"""
GraphRAG Code Review Bot — LangGraph pipeline.

Pipeline:
    parse -> retrieve -> review -> [if bug detected] -> generate_patch
                                                            -> evaluate
                                                            -> [if fail] -> reflect -> retry (max 3)

Combines:
    - GraphRAG (Neo4j + Weaviate) for context-aware retrieval
    - Few-Shot CoT prompting for structured reasoning
    - Reflexion loop for iterative bug fixing
"""

from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate

from src_bot.graph_retriever import CustomGraphRAGRetriever
from src_bot.llm.router import get_llm
from src_bot.config.config import configs
from src_bot.reflexion.patch_generator import generate_patch
from src_bot.reflexion.evaluator import evaluate_patch, EvalResult
from src_bot.reflexion.reflector import generate_reflection
from src_bot.risk.scorer import RiskScorer, RiskInput, RiskResult


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class CodeReviewState(TypedDict):
    # Input
    pr_diff: str
    issue_text: str  # For SWE-bench: the issue description
    repo_path: str   # For SWE-bench: local path to cloned repo
    test_command: str # For SWE-bench: custom test command

    # Retrieval
    changed_files: List[str]
    context_data: List[str]

    # Risk scoring
    risk_score: float
    risk_level: str
    risk_explanation: str

    # Review
    final_review: str
    bug_detected: bool
    bug_description: str

    # Reflexion loop
    generated_patch: str
    test_passed: bool
    test_output: str
    reflections: List[str]  # episodic memory
    retry_count: int


# ---------------------------------------------------------------------------
# Bot
# ---------------------------------------------------------------------------

class GraphRAGBot:
    def __init__(self):
        self.app = None
        self.retriever = None
        self.llm = None

    def initialize(self):
        """Initialize retriever, LLM, risk scorer, and compile the LangGraph workflow."""
        self.retriever = CustomGraphRAGRetriever()
        self.llm = get_llm(
            provider=configs.LLM_PROVIDER,
            model=configs.LLM_MODEL,
            temperature=configs.LLM_TEMPERATURE,
        )
        self.risk_scorer = RiskScorer()

        workflow = StateGraph(CodeReviewState)

        # Nodes
        workflow.add_node("parse", self.parse_diff_node)
        workflow.add_node("retrieve", self.retrieve_node)
        workflow.add_node("risk_score", self.risk_score_node)
        workflow.add_node("review", self.review_node)
        workflow.add_node("generate_patch", self.generate_patch_node)
        workflow.add_node("evaluate", self.evaluate_node)
        workflow.add_node("reflect", self.reflect_node)

        # Edges: parse -> retrieve -> risk_score -> review
        workflow.set_entry_point("parse")
        workflow.add_edge("parse", "retrieve")
        workflow.add_edge("retrieve", "risk_score")
        workflow.add_edge("risk_score", "review")

        # After review: fix bug or done
        workflow.add_conditional_edges(
            "review",
            self._should_fix,
            {"fix": "generate_patch", "done": END},
        )

        # After patch generation: evaluate with tests
        workflow.add_edge("generate_patch", "evaluate")

        # After evaluation: done, reflect, or give up
        workflow.add_conditional_edges(
            "evaluate",
            self._should_retry,
            {"done": END, "reflect": "reflect", "give_up": END},
        )
        workflow.add_edge("reflect", "generate_patch")

        self.app = workflow.compile()

    def invoke(self, inputs: dict) -> dict:
        """Run the full pipeline on the given inputs."""
        if not self.app:
            raise Exception("Bot not initialized! Call initialize() first.")

        # Set defaults for optional fields
        defaults = {
            "issue_text": "",
            "repo_path": "",
            "test_command": "python -m pytest --tb=short -q",
            "changed_files": [],
            "context_data": [],
            "risk_score": 0.0,
            "risk_level": "",
            "risk_explanation": "",
            "final_review": "",
            "bug_detected": False,
            "bug_description": "",
            "generated_patch": "",
            "test_passed": False,
            "test_output": "",
            "reflections": [],
            "retry_count": 0,
        }
        defaults.update(inputs)
        return self.app.invoke(defaults)

    def close(self):
        if self.retriever:
            self.retriever.close()

    # ------------------------------------------------------------------
    # Node: Parse diff
    # ------------------------------------------------------------------
    def parse_diff_node(self, state: CodeReviewState) -> dict:
        """Parse the PR diff into searchable queries."""
        print("--- STEP 1: PARSING DIFF ---")
        diff = state["pr_diff"]
        issue = state.get("issue_text", "")

        # Use issue text as primary query if available (SWE-bench mode),
        # otherwise fall back to the diff itself
        queries = []
        if issue:
            queries.append(issue[:500])  # cap for embedding model
        if diff:
            queries.append(diff[:500])

        return {"changed_files": queries if queries else [diff]}

    # ------------------------------------------------------------------
    # Node: Retrieve graph context
    # ------------------------------------------------------------------
    def retrieve_node(self, state: CodeReviewState) -> dict:
        """Phase 1+2: Weaviate semantic search -> Neo4j graph expansion."""
        print("--- STEP 2: RETRIEVING GRAPH CONTEXT ---")
        queries = state["changed_files"]
        collected_context = []

        for query in queries:
            results = self.retriever.search(
                query_text=query,
                top_k=configs.RETRIEVER_TOP_K,
            )
            collected_context.extend(results)

        return {"context_data": collected_context}

    # ------------------------------------------------------------------
    # Node: Risk scoring
    # ------------------------------------------------------------------
    def risk_score_node(self, state: CodeReviewState) -> dict:
        """Compute a risk score based on graph context signals."""
        print("--- STEP 3: COMPUTING RISK SCORE ---")

        context_data = state.get("context_data", [])

        # Count blast radius from retrieved graph context
        blast_radius_size = len(context_data)

        # Count lines changed in the diff
        diff = state.get("pr_diff", "")
        added = sum(1 for line in diff.split("\n") if line.startswith("+") and not line.startswith("+++"))
        removed = sum(1 for line in diff.split("\n") if line.startswith("-") and not line.startswith("---"))
        pr_size_lines = added + removed

        risk_input = RiskInput(
            blast_radius_size=blast_radius_size,
            coverage_gap_ratio=0.0,  # TODO: integrate coverage mapper
            bug_frequency=0,         # TODO: read from Neo4j node properties
            contributor_churn=0,     # TODO: read from Neo4j node properties
            is_new_contributor=False, # TODO: check git history
            pr_size_lines=pr_size_lines,
        )

        result: RiskResult = self.risk_scorer.score(risk_input)

        return {
            "risk_score": result.score,
            "risk_level": result.level,
            "risk_explanation": result.explanation(),
        }

    # ------------------------------------------------------------------
    # Node: Review (with Few-Shot CoT)
    # ------------------------------------------------------------------
    def review_node(self, state: CodeReviewState) -> dict:
        """Generate a code review using graph context + Few-Shot CoT."""
        print("--- STEP 3: GENERATING REVIEW ---")

        system_prompt = """You are a Senior Code Reviewer & Security Auditor.

TASK: Review the code diff using the SYSTEM CONTEXT from the knowledge graph.

SYSTEM CONTEXT (from Knowledge Graph):
{graph_context}

RISK ASSESSMENT:
{risk_info}

PULL REQUEST DIFF:
{pr_diff}

Think step by step (Chain-of-Thought):
Step 1: Identify what changed in the diff.
Step 2: Use the graph context to understand which functions call this code and what depends on it.
Step 3: Consider the risk level — HIGH risk changes need extra scrutiny on callers and test coverage.
Step 4: Check for bugs — logic errors, missing edge cases, security issues.
Step 5: Check if the change could break any callers or downstream dependencies.

REVIEW FORMAT:
1. **Summary**: What changed and why.
2. **Risk Level**: {risk_level} — explain why based on blast radius and change size.
3. **Impact Analysis**: Which callers/dependencies are affected (from graph context).
4. **Bugs Found**: List any bugs with [BUG] prefix. If no bugs, say "No bugs found."
5. **Security Warnings**: Any security concerns.
6. **Suggestions**: Improvement recommendations.

IMPORTANT: If you find a bug, start the Bugs Found section with exactly "[BUG]" so the system can detect it.
"""

        context_str = "\n\n".join(state["context_data"]) if state["context_data"] else "No graph context available."
        risk_info = state.get("risk_explanation", "Risk assessment not available.")
        risk_level = state.get("risk_level", "unknown").upper()
        content = None
        bug_detected = False
        bug_description = ""

        prompt = ChatPromptTemplate.from_template(system_prompt)
        chain = prompt | self.llm
        response = chain.invoke({
            "graph_context": context_str,
            "risk_info": risk_info,
            "risk_level": risk_level,
            "pr_diff": state["pr_diff"],
        })
        content = response.content

        # Detect if review found bugs
        if content and "[BUG]" in content.upper():
            bug_detected = True
            # Extract bug description for the patch generator
            bug_description = content

        return {
            "final_review": content,
            "bug_detected": bug_detected,
            "bug_description": bug_description,
        }

    # ------------------------------------------------------------------
    # Node: Generate patch (Reflexion Actor)
    # ------------------------------------------------------------------
    def generate_patch_node(self, state: CodeReviewState) -> dict:
        """Generate a fix patch using graph context + reflections from memory."""
        retry = state.get("retry_count", 0)
        print(f"--- STEP 4: GENERATING PATCH (attempt {retry + 1}) ---")

        context_str = "\n\n".join(state["context_data"]) if state["context_data"] else ""

        patch = generate_patch(
            llm=self.llm,
            bug_description=state["bug_description"],
            file_content=state["pr_diff"],
            graph_context=context_str,
            reflections=state.get("reflections", []),
        )

        return {
            "generated_patch": patch,
            "retry_count": retry + 1,
        }

    # ------------------------------------------------------------------
    # Node: Evaluate patch (Reflexion Evaluator)
    # ------------------------------------------------------------------
    def evaluate_node(self, state: CodeReviewState) -> dict:
        """Apply the generated patch and run tests to verify it."""
        print("--- STEP 5: EVALUATING PATCH ---")

        repo_path = state.get("repo_path", "")
        patch = state.get("generated_patch", "")
        test_command = state.get("test_command", "python -m pytest --tb=short -q")

        if not repo_path or not patch:
            # No repo path or patch — skip evaluation (review-only mode)
            return {"test_passed": False, "test_output": ""}

        result: EvalResult = evaluate_patch(
            patch_diff=patch,
            repo_path=repo_path,
            test_command=test_command,
            timeout=300,
        )

        print(f"  -> {result.summary}")

        return {
            "test_passed": result.passed,
            "test_output": f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}",
        }

    # ------------------------------------------------------------------
    # Node: Reflect (Reflexion Self-Reflection)
    # ------------------------------------------------------------------
    def reflect_node(self, state: CodeReviewState) -> dict:
        """Generate a verbal self-reflection after a failed patch attempt."""
        print("--- STEP 5: REFLECTING ON FAILURE ---")

        context_str = "\n\n".join(state["context_data"]) if state["context_data"] else ""
        test_output = state.get("test_output", "No test output available.")

        reflection = generate_reflection(
            llm=self.llm,
            bug_description=state["bug_description"],
            failed_patch=state["generated_patch"],
            test_output=test_output,
            graph_context=context_str,
            previous_reflections=state.get("reflections", []),
        )

        # Append to episodic memory (bounded to last 3 reflections)
        reflections = list(state.get("reflections", []))
        reflections.append(reflection)
        if len(reflections) > 3:
            reflections = reflections[-3:]

        return {"reflections": reflections}

    # ------------------------------------------------------------------
    # Conditional edges
    # ------------------------------------------------------------------
    def _should_fix(self, state: CodeReviewState) -> str:
        """After review: fix the bug or finish."""
        if state.get("bug_detected", False):
            return "fix"
        return "done"

    def _should_retry(self, state: CodeReviewState) -> str:
        """After evaluation: done if tests pass, reflect if they fail, give up after max retries."""
        if state.get("test_passed", False):
            print("--- TESTS PASSED! Patch is valid. ---")
            return "done"

        retry_count = state.get("retry_count", 0)
        max_retries = configs.REFLEXION_MAX_RETRIES

        if retry_count >= max_retries:
            print(f"--- MAX RETRIES ({max_retries}) REACHED — giving up ---")
            return "give_up"

        # If we have test output (tests were run and failed), reflect and retry
        if state.get("test_output"):
            return "reflect"

        # No test runner configured (no repo_path) — return the patch as-is
        return "done"


# Singleton instance (initialized in api/main.py lifespan)
bot_instance = GraphRAGBot()
