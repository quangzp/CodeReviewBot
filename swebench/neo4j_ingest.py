"""
Neo4j-only ingestion — parses Python repos into a code property graph.

Creates:
  Nodes: Module, Class, Function (with name, file_path, content, docstring)
  Relationships: DEFINES, CALLS, IMPORTS, CONTAINS

Also creates a full-text BM25 index for Phase 1 retrieval (replacing Weaviate).
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import sys
import textwrap
from pathlib import Path
from typing import Optional

from neo4j import GraphDatabase


# ---------------------------------------------------------------------------
# AST visitors
# ---------------------------------------------------------------------------

class CodeVisitor(ast.NodeVisitor):
    """Extract functions, classes, and imports from a Python file."""

    def __init__(self, file_path: str, source: str):
        self.file_path = file_path
        self.source_lines = source.splitlines()
        self.nodes = []       # list of dicts: {type, name, file_path, content, docstring, lineno}
        self.calls = []       # list of (caller_name, callee_name)
        self.imports = []     # list of module names imported
        self._current_class = None
        self._current_func = None

    def _get_source(self, node) -> str:
        try:
            return ast.get_source_segment("\n".join(self.source_lines), node) or ""
        except Exception:
            # Fallback: grab lines by lineno
            start = node.lineno - 1
            end = getattr(node, "end_lineno", start + 20)
            return "\n".join(self.source_lines[start:end])

    def _get_docstring(self, node) -> str:
        try:
            return ast.get_docstring(node) or ""
        except Exception:
            return ""

    def visit_ClassDef(self, node):
        prev_class = self._current_class
        self._current_class = node.name
        content = self._get_source(node)
        self.nodes.append({
            "type": "Class",
            "name": node.name,
            "qualified_name": node.name,
            "file_path": self.file_path,
            "content": content[:2000],
            "docstring": self._get_docstring(node),
            "lineno": node.lineno,
        })
        self.generic_visit(node)
        self._current_class = prev_class

    def visit_FunctionDef(self, node):
        self._visit_func(node)

    def visit_AsyncFunctionDef(self, node):
        self._visit_func(node)

    def _visit_func(self, node):
        prev_func = self._current_func
        qualified = f"{self._current_class}.{node.name}" if self._current_class else node.name
        self._current_func = qualified

        content = self._get_source(node)
        self.nodes.append({
            "type": "Method" if self._current_class else "Function",
            "name": node.name,
            "qualified_name": qualified,
            "file_path": self.file_path,
            "content": content[:2000],
            "docstring": self._get_docstring(node),
            "lineno": node.lineno,
        })

        # Collect call relationships
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                callee = _extract_call_name(child)
                if callee:
                    self.calls.append((qualified, callee))

        self.generic_visit(node)
        self._current_func = prev_func

    def visit_Import(self, node):
        for alias in node.names:
            self.imports.append(alias.name)

    def visit_ImportFrom(self, node):
        if node.module:
            self.imports.append(node.module)


def _extract_call_name(node: ast.Call) -> Optional[str]:
    if isinstance(node.func, ast.Name):
        return node.func.id
    elif isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

def ingest_repo_to_neo4j(
    repo_dir: Path,
    project_id: str,
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
    max_files: int = 500,
) -> dict:
    """
    Parse a Python repo and ingest into Neo4j as a code property graph.

    Returns summary: {files, nodes, relationships}
    """
    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

    with driver.session() as session:
        # Ensure full-text index exists
        _ensure_indexes(session)

        # Clear existing nodes for this project
        session.run(
            "MATCH (n {project_id: $pid}) DETACH DELETE n",
            pid=project_id
        )

        # Collect all Python files (skip tests for faster ingestion)
        all_py = list(repo_dir.rglob("*.py"))
        # Exclude pycache and git, but KEEP source files even if near test dirs
        # Only exclude files whose name starts with test_ or ends with _test.py
        py_files = [
            f for f in all_py
            if "__pycache__" not in str(f)
            and "/.git/" not in str(f)
            and not f.name.startswith("test_")
            and not f.name.endswith("_test.py")
            and "/tests/" not in str(f)
            and "/test/" not in str(f)
        ][:max_files]

        total_nodes = 0
        total_rels = 0
        file_count = 0

        for fpath in py_files:
            try:
                source = fpath.read_text(errors="ignore")
                if len(source) > 100_000:
                    source = source[:100_000]  # cap large files

                rel_path = str(fpath.relative_to(repo_dir))

                # Create Module node
                session.run("""
                    MERGE (m:Module {file_path: $fp, project_id: $pid})
                    SET m.name = $name, m.content = $content
                """, fp=rel_path, pid=project_id,
                    name=fpath.stem,
                    content=source[:500])

                # Parse AST
                try:
                    tree = ast.parse(source)
                except SyntaxError:
                    continue

                visitor = CodeVisitor(rel_path, source)
                visitor.visit(tree)

                # Ingest code nodes
                for node_data in visitor.nodes:
                    node_id = hashlib.md5(
                        f"{project_id}:{rel_path}:{node_data['qualified_name']}".encode()
                    ).hexdigest()

                    session.run(f"""
                        MERGE (n:CodeNode {{node_id: $nid}})
                        SET n.type = $type,
                            n.name = $name,
                            n.qualified_name = $qname,
                            n.file_path = $fp,
                            n.content = $content,
                            n.docstring = $doc,
                            n.lineno = $lineno,
                            n.project_id = $pid
                        WITH n
                        MATCH (m:Module {{file_path: $fp, project_id: $pid}})
                        MERGE (m)-[:DEFINES]->(n)
                    """, nid=node_id, type=node_data["type"],
                        name=node_data["name"],
                        qname=node_data["qualified_name"],
                        fp=rel_path,
                        content=node_data["content"],
                        doc=node_data["docstring"],
                        lineno=node_data["lineno"],
                        pid=project_id)
                    total_nodes += 1

                # Ingest call relationships (best-effort name matching)
                for caller, callee in visitor.calls[:50]:  # cap per file
                    session.run("""
                        MATCH (a:CodeNode {qualified_name: $caller, project_id: $pid})
                        MATCH (b:CodeNode {name: $callee, project_id: $pid})
                        WHERE a <> b
                        MERGE (a)-[:CALLS]->(b)
                    """, caller=caller, callee=callee, pid=project_id)
                    total_rels += 1

                file_count += 1

            except Exception as e:
                continue

        print(f"  Ingested: {file_count} files, {total_nodes} nodes, {total_rels} call relationships")

    driver.close()
    return {"files": file_count, "nodes": total_nodes, "relationships": total_rels}


def _ensure_indexes(session):
    """Create full-text BM25 index and constraints if not present."""
    # Full-text index for BM25 search (Phase 1)
    try:
        session.run("""
            CREATE FULLTEXT INDEX codeSearch IF NOT EXISTS
            FOR (n:CodeNode)
            ON EACH [n.name, n.content, n.docstring, n.file_path]
        """)
    except Exception:
        pass

    # Also index Modules
    try:
        session.run("""
            CREATE FULLTEXT INDEX moduleSearch IF NOT EXISTS
            FOR (m:Module)
            ON EACH [m.name, m.content, m.file_path]
        """)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# BM25 retrieval
# ---------------------------------------------------------------------------

def bm25_search(
    session,
    query: str,
    project_id: str,
    top_k: int = 20,
) -> list[dict]:
    """
    Phase 1: Full-text BM25 search over code nodes.
    Returns list of node dicts sorted by relevance score.
    """
    # Sanitize query for Lucene syntax
    safe_query = _sanitize_lucene(query)

    results = session.run("""
        CALL db.index.fulltext.queryNodes('codeSearch', $search_query)
        YIELD node, score
        WHERE node.project_id = $pid
        RETURN node.node_id AS node_id,
               node.name AS name,
               node.qualified_name AS qualified_name,
               node.type AS type,
               node.file_path AS file_path,
               node.content AS content,
               node.docstring AS docstring,
               node.lineno AS lineno,
               score
        ORDER BY score DESC
        LIMIT $top_k
    """, search_query=safe_query, pid=project_id, top_k=top_k)

    return [dict(r) for r in results]


def graph_expand(
    session,
    node_ids: list[str],
    project_id: str,
    max_hops: int = 2,
    max_results: int = 30,
) -> list[dict]:
    """
    Phase 2: Graph traversal from seed nodes.
    Expands CALLS and DEFINES relationships up to max_hops.
    Returns related nodes with relationship context.
    """
    results = session.run("""
        MATCH (seed:CodeNode)
        WHERE seed.node_id IN $node_ids AND seed.project_id = $pid
        CALL apoc.path.subgraphNodes(seed, {
            relationshipFilter: 'CALLS|DEFINES',
            maxLevel: $hops,
            limit: $limit
        })
        YIELD node
        RETURN node.name AS name,
               node.qualified_name AS qualified_name,
               node.type AS type,
               node.file_path AS file_path,
               node.content AS content,
               node.docstring AS docstring
    """, node_ids=node_ids, pid=project_id, hops=max_hops, limit=max_results)

    return [dict(r) for r in results]


def graph_expand_simple(
    session,
    node_ids: list[str],
    project_id: str,
    max_results: int = 30,
) -> list[dict]:
    """
    Simple graph expansion without APOC — uses direct CALLS traversal.
    Works without APOC plugin installed.
    """
    results = session.run("""
        MATCH (seed:CodeNode)
        WHERE seed.node_id IN $node_ids AND seed.project_id = $pid
        OPTIONAL MATCH (seed)-[:CALLS]->(callee:CodeNode {project_id: $pid})
        OPTIONAL MATCH (caller:CodeNode {project_id: $pid})-[:CALLS]->(seed)
        OPTIONAL MATCH (seed)<-[:DEFINES]-(mod:Module {project_id: $pid})
        WITH seed, collect(DISTINCT callee) AS callees,
             collect(DISTINCT caller) AS callers,
             collect(DISTINCT mod) AS modules
        RETURN seed.name AS name,
               seed.qualified_name AS qualified_name,
               seed.type AS type,
               seed.file_path AS file_path,
               seed.content AS content,
               seed.docstring AS docstring,
               [c IN callees | c.name] AS calls,
               [c IN callers | c.name] AS called_by
        LIMIT $limit
    """, node_ids=node_ids, pid=project_id, limit=max_results)

    return [dict(r) for r in results]


def _sanitize_lucene(query: str) -> str:
    """Escape special Lucene characters and extract key terms."""
    import re
    # Extract meaningful words (3+ chars, alphanumeric/underscore)
    words = re.findall(r'\b\w{3,}\b', query)
    # Remove common English stop words
    stop = {"the","this","that","for","and","are","has","have","was","with",
            "from","but","not","been","when","what","how","should","could",
            "would","which","their","there","here","will","than","also"}
    words = [w for w in words if w.lower() not in stop]
    # Take top 10 unique words
    seen, result = set(), []
    for w in words:
        if w not in seen:
            seen.add(w)
            result.append(w)
        if len(result) >= 10:
            break
    return " ".join(result) if result else "code"


# ---------------------------------------------------------------------------
# CLI for standalone ingestion
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Ingest a repo into Neo4j")
    parser.add_argument("repo_dir")
    parser.add_argument("project_id")
    parser.add_argument("--uri", default="bolt://localhost:7687")
    parser.add_argument("--user", default="neo4j")
    parser.add_argument("--password", default="CodeReview2024!")
    args = parser.parse_args()

    result = ingest_repo_to_neo4j(
        Path(args.repo_dir), args.project_id,
        args.uri, args.user, args.password
    )
    print(f"Done: {result}")
