"""
Multi-gate verification pipeline.

Gate order (cheapest first — skip expensive gates on early failure):
  1. AST parse  (FREE, ~0ms)  — always on
  2. Test exec  (seconds)     — optional, auto-detects test presence
  3. LLM eval   (tokens)      — always on, contract-aware when contract given

Usage:
    from src_bot.verification.pipeline import verify_patch

    result = verify_patch(patch, repo_dir, bug_description, contract=contract)
    if not result.passed:
        last_error = result.rejection_reason  # feed into reflector
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from src_bot.verification.ast_gate import ASTGateResult, check_ast
from src_bot.verification.test_gate import TestGateResult, check_tests


@dataclass
class VerificationResult:
    ast_result: ASTGateResult
    test_result: Optional[TestGateResult] = None
    llm_score: int = 3
    llm_reason: str = "not evaluated"

    @property
    def passed(self) -> bool:
        if not self.ast_result.passed:
            return False
        if (
            self.test_result is not None
            and not self.test_result.passed
            and not self.test_result.skipped
        ):
            return False
        if self.llm_score < 3:
            return False
        return True

    @property
    def rejection_reason(self) -> str:
        """Human-readable reason for rejection — feeds into the reflexion loop."""
        if not self.ast_result.passed:
            line_info = f" (line {self.ast_result.error_line})" if self.ast_result.error_line else ""
            return (
                f"[AST SYNTAX ERROR{line_info}] {self.ast_result.error}\n\n"
                "The patch produces a Python syntax error. Fix the syntax before retrying."
            )
        if (
            self.test_result is not None
            and not self.test_result.passed
            and not self.test_result.skipped
        ):
            output = ""
            if self.test_result.stdout:
                output += f"\nTest output:\n{self.test_result.stdout[-800:]}"
            if self.test_result.stderr:
                output += f"\nStderr:\n{self.test_result.stderr[-400:]}"
            return (
                f"[TEST FAILURE] Tests failed after applying patch.{output}\n\n"
                "Fix the logic so the existing tests pass."
            )
        if self.llm_score < 3:
            return (
                f"[EVALUATOR REJECTION score={self.llm_score}/5]\n"
                f"{self.llm_reason}\n\n"
                "The patch applied cleanly but failed quality review. "
                "Focus on improving logic, not syntax."
            )
        return ""


def verify_patch(
    patch_diff: str,
    repo_dir: Path,
    bug_description: str,
    fault_description: str = "",
    contract: Optional[object] = None,
    mode: str = "bug_fix",
    test_enabled: bool = True,
    test_command: str = "python -m pytest --tb=short -q",
    test_timeout: int = 300,
) -> VerificationResult:
    """Run multi-gate verification.

    Gate 1: AST parse (always on, free)
    Gate 2: Test execution (optional, uses existing reflexion/evaluator.py runner)
    Gate 3: LLM quality check (always on, contract-aware when contract provided)

    If a gate fails, subsequent more-expensive gates are skipped.
    """
    # Gate 1: AST parse
    ast_result = check_ast(patch_diff, repo_dir)
    if not ast_result.passed:
        return VerificationResult(ast_result=ast_result)

    # Gate 2: Test execution
    test_result = check_tests(
        patch_diff=patch_diff,
        repo_dir=repo_dir,
        test_command=test_command,
        timeout=test_timeout,
        enabled=test_enabled,
    )
    if not test_result.passed and not test_result.skipped:
        return VerificationResult(ast_result=ast_result, test_result=test_result)

    # Gate 3: LLM quality evaluation
    # Default to 0 (not 3) so an LLM exception does not silently pass the gate.
    # "evaluation unavailable" ≠ "approved" — the orchestrator handles 0-score
    # patches via accept_best/reanalyze/abort rather than auto-approving them.
    llm_score = 0
    llm_reason = "LLM evaluation unavailable"
    try:
        from api.agent.evaluator import evaluate_patch_with_contract, evaluate_patch
        description = bug_description
        if fault_description:
            description = f"{bug_description[:300]}\nFault: {fault_description}"

        if contract is not None:
            llm_score, llm_reason = evaluate_patch_with_contract(
                description, patch_diff, contract, mode
            )
        else:
            llm_score, llm_reason = evaluate_patch(description, patch_diff, mode)
    except Exception as _eval_exc:
        import logging as _logging
        _logging.getLogger(__name__).warning(
            "[verify] LLM evaluator failed (non-fatal, score=0): %s", _eval_exc
        )

    return VerificationResult(
        ast_result=ast_result,
        test_result=test_result,
        llm_score=llm_score,
        llm_reason=llm_reason,
    )
