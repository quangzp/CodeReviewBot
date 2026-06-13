"""
Smoke tests — verify all key modules import correctly and are well-formed.
These run without Neo4j or Groq being available.
"""
import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestImports:
    """Verify all key modules can be imported without errors."""

    def test_import_agent_loop(self):
        from api.agent.loop import run_agent_turn
        assert callable(run_agent_turn)

    def test_import_agent_tools(self):
        from api.agent.tools import build_tool_registry, TOOL_DESCRIPTIONS
        assert callable(build_tool_registry)
        assert len(TOOL_DESCRIPTIONS) == 10  # updated: get_review_detail added

    def test_import_classifier(self):
        from api.agent.classifier import classify, Intent
        assert Intent.IN_SCOPE.value == "IN_SCOPE"
        assert Intent.OUT_SCOPE.value == "OUT_SCOPE"
        assert Intent.CLARIFY.value == "CLARIFY"

    def test_import_evaluator(self):
        from api.agent.evaluator import evaluate_patch
        assert callable(evaluate_patch)

    def test_import_code_context(self):
        from api.agent.code_context import CodeContext, build_code_context
        assert callable(build_code_context)

    def test_import_smell_detector(self):
        from api.agent.smell_detector import detect_all_smells, SmellResult
        assert callable(detect_all_smells)

    def test_import_memory_schema(self):
        from src_bot.memory.schema import ReviewFact, DeveloperProfile
        assert ReviewFact is not None

    def test_import_memory_extractor(self):
        from src_bot.memory.extractor import extract_facts_from_review
        assert callable(extract_facts_from_review)

    def test_import_models(self):
        from api.models import ProjectRecord, ReviewRecord, FileReviewResult
        p = ProjectRecord(
            id="test",
            repo_name="t/t",
            repo_url="https://github.com/t/t",
            created_at="2025-01-01",
            updated_at="2025-01-01",
        )
        assert p.status == "pending"

    def test_import_config(self):
        from src_bot.config.config import configs
        assert configs.LLM_PROVIDER in ("groq", "ollama", "openai", "together", "vllm")
        assert configs.FAST_LLM_PROVIDER in ("groq", "ollama", "openai", "together", "vllm")

    def test_import_neo4j_ingest(self):
        from swebench.neo4j_ingest import ingest_repo_to_neo4j, bm25_search, CodeVisitor
        assert callable(ingest_repo_to_neo4j)
        assert callable(bm25_search)

    def test_import_fix_bug(self):
        from api.agent.fix_bug import fix_bug_in_project
        assert callable(fix_bug_in_project)

    def test_import_refactor(self):
        from api.agent.refactor import refactor_code_in_project, detect_refactoring_targets
        assert callable(refactor_code_in_project)
        assert callable(detect_refactoring_targets)


class TestToolDescriptions:
    """Verify tool catalog is well-formed."""

    def test_all_tools_have_required_fields(self):
        from api.agent.tools import TOOL_DESCRIPTIONS
        for t in TOOL_DESCRIPTIONS:
            assert "name" in t, f"Tool missing 'name': {t}"
            assert "description" in t, f"Tool missing 'description': {t}"
            assert "parameters" in t, f"Tool missing 'parameters': {t}"
            assert len(t["description"]) > 20, f"Tool description too short: {t['name']}"

    def test_all_tools_have_object_parameters(self):
        from api.agent.tools import TOOL_DESCRIPTIONS
        for t in TOOL_DESCRIPTIONS:
            assert t["parameters"]["type"] == "object", \
                f"Tool {t['name']} parameters type is not 'object'"

    def test_expected_tools_present(self):
        from api.agent.tools import TOOL_DESCRIPTIONS
        names = {t["name"] for t in TOOL_DESCRIPTIONS}
        expected = {"add_project", "list_projects", "project_status",
                    "review_pr", "fix_bug", "refactor_code", "recent_reviews",
                    "explore_project", "apply_review_fixes", "get_review_detail"}
        assert expected == names


class TestCodeVisitorInheritance:
    """Verify the F007 CodeVisitor changes work correctly."""

    def test_visitor_has_inherits_attribute(self):
        from swebench.neo4j_ingest import CodeVisitor
        v = CodeVisitor("test.py", "class A: pass")
        assert hasattr(v, "inherits")
        assert isinstance(v.inherits, list)

    def test_visitor_collects_inherits(self):
        import ast
        from swebench.neo4j_ingest import CodeVisitor
        source = "class Parent: pass\nclass Child(Parent): pass\n"
        v = CodeVisitor("test.py", source)
        tree = ast.parse(source)
        v.visit(tree)
        assert ("Child", "Parent") in v.inherits

    def test_node_has_line_count(self):
        import ast
        from swebench.neo4j_ingest import CodeVisitor
        source = "def foo():\n    x = 1\n    return x\n"
        v = CodeVisitor("test.py", source)
        tree = ast.parse(source)
        v.visit(tree)
        func_nodes = [n for n in v.nodes if n["name"] == "foo"]
        assert len(func_nodes) == 1
        assert "line_count" in func_nodes[0]
        assert func_nodes[0]["line_count"] >= 1

    def test_class_has_method_count(self):
        import ast
        from swebench.neo4j_ingest import CodeVisitor
        source = (
            "class MyClass:\n"
            "    def a(self): pass\n"
            "    def b(self): pass\n"
            "    def c(self): pass\n"
        )
        v = CodeVisitor("test.py", source)
        tree = ast.parse(source)
        v.visit(tree)
        class_nodes = [n for n in v.nodes if n["name"] == "MyClass"]
        assert len(class_nodes) == 1
        assert class_nodes[0]["method_count"] == 3
