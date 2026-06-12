"""
Neo4j-only ingestion — parses Python repos into a code property graph.

Creates:
  Nodes: Module, Class, Function (with name, file_path, content, docstring)
  Relationships: DEFINES, CALLS, IMPORTS, CONTAINS

Retrieval (Phase 1):
  - BM25 fulltext via Neo4j FULLTEXT INDEX (always available)
  - Vector cosine via Neo4j VECTOR INDEX + sentence-transformers embeddings
  - Hybrid: BM25 + vector merged with Reciprocal Rank Fusion (RRF)
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import sys
import textwrap
from pathlib import Path
from typing import Any, Optional

from neo4j import GraphDatabase


DEFAULT_BATCH_SIZE = int(os.getenv("NEO4J_INGEST_BATCH_SIZE", "500"))


def _repo_rel_path(path: Path, repo_dir: Path) -> str:
    """Return a repository-relative path using POSIX separators for diffs/Neo4j."""
    return path.relative_to(repo_dir).as_posix()


def _batched(items: list[dict], batch_size: int = DEFAULT_BATCH_SIZE):
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def _insert_modules(session, modules: list[dict]) -> None:
    for batch in _batched(modules):
        result = session.run("""
            UNWIND $rows AS row
            MERGE (m:Module {file_path: row.file_path, project_id: row.project_id})
            SET m.name = row.name,
                m.content = row.content
        """, rows=batch)
        result.consume()


def _insert_code_nodes(session, nodes: list[dict]) -> int:
    total_created = 0
    for batch in _batched(nodes):
        result = session.run("""
            UNWIND $rows AS row
            MERGE (n:CodeNode {node_id: row.node_id})
            SET n.type = row.type,
                n.name = row.name,
                n.qualified_name = row.qualified_name,
                n.file_path = row.file_path,
                n.content = row.content,
                n.docstring = row.docstring,
                n.lineno = row.lineno,
                n.project_id = row.project_id
            WITH n, row
            MATCH (m:Module {file_path: row.file_path, project_id: row.project_id})
            MERGE (m)-[:DEFINES]->(n)
        """, rows=batch)
        summary = result.consume()
        total_created += summary.counters.nodes_created
    return total_created


def _insert_calls(session, calls: list[dict]) -> int:
    total_created = 0
    for batch in _batched(calls, batch_size=DEFAULT_BATCH_SIZE * 2):
        result = session.run("""
            UNWIND $rows AS row
            MATCH (a:CodeNode {
                qualified_name: row.caller,
                project_id: row.project_id
            })
            MATCH (b:CodeNode {
                name: row.callee,
                project_id: row.project_id
            })
            WHERE a <> b
            MERGE (a)-[:CALLS]->(b)
        """, rows=batch)
        summary = result.consume()
        total_created += summary.counters.relationships_created
    return total_created


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
        py_files = []
        for f in all_py:
            path_text = f.as_posix()
            if (
                "__pycache__" in path_text
                or "/.git/" in path_text
                or f.name.startswith("test_")
                or f.name.endswith("_test.py")
                or "/tests/" in path_text
                or "/test/" in path_text
            ):
                continue
            py_files.append(f)
            if len(py_files) >= max_files:
                break

        modules: list[dict] = []
        nodes: list[dict] = []
        calls: list[dict] = []
        file_count = 0

        for fpath in py_files:
            try:
                source = fpath.read_text(errors="ignore")
                if len(source) > 100_000:
                    source = source[:100_000]  # cap large files

                rel_path = _repo_rel_path(fpath, repo_dir)

                modules.append({
                    "file_path": rel_path,
                    "project_id": project_id,
                    "name": fpath.stem,
                    "content": source[:500],
                })

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
                    nodes.append({
                        "node_id": node_id,
                        "type": node_data["type"],
                        "name": node_data["name"],
                        "qualified_name": node_data["qualified_name"],
                        "file_path": rel_path,
                        "content": node_data["content"],
                        "docstring": node_data["docstring"],
                        "lineno": node_data["lineno"],
                        "project_id": project_id,
                    })

                # Ingest call relationships (best-effort name matching)
                for caller, callee in visitor.calls[:50]:  # cap per file
                    calls.append({
                        "caller": caller,
                        "callee": callee,
                        "project_id": project_id,
                    })

                file_count += 1

            except Exception as e:
                continue

        _insert_modules(session, modules)
        total_nodes = _insert_code_nodes(session, nodes)
        total_rels = _insert_calls(session, calls)

        print(f"  Ingested: {file_count} files, {total_nodes} nodes, {total_rels} call relationships")

    driver.close()
    return {"files": file_count, "nodes": total_nodes, "relationships": total_rels}


def _ensure_indexes(session):
    """Create full-text BM25 index and constraints if not present."""
    try:
        session.run("""
            CREATE CONSTRAINT codeNodeId IF NOT EXISTS
            FOR (n:CodeNode)
            REQUIRE n.node_id IS UNIQUE
        """)
    except Exception:
        pass

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

    try:
        session.run("""
            CREATE INDEX codeNodeQualified IF NOT EXISTS
            FOR (n:CodeNode)
            ON (n.project_id, n.qualified_name)
        """)
    except Exception:
        pass

    try:
        session.run("""
            CREATE INDEX codeNodeName IF NOT EXISTS
            FOR (n:CodeNode)
            ON (n.project_id, n.name)
        """)
    except Exception:
        pass

    try:
        session.run("""
            CREATE INDEX codeNodeFilePath IF NOT EXISTS
            FOR (n:CodeNode)
            ON (n.project_id, n.file_path)
        """)
    except Exception:
        pass

    try:
        session.run("""
            CREATE INDEX moduleFilePath IF NOT EXISTS
            FOR (m:Module)
            ON (m.project_id, m.file_path)
        """)
    except Exception:
        pass

    # Vector index for semantic search — dimension must match EMBEDDING_MODEL output
    # Default: all-MiniLM-L6-v2 → 384 dims. Change EMBEDDING_VECTOR_DIM if using another model.
    try:
        dim = int(os.getenv("EMBEDDING_VECTOR_DIM", "384"))
        session.run(f"""
            CREATE VECTOR INDEX codeNodeEmbedding IF NOT EXISTS
            FOR (n:CodeNode) ON (n.embedding)
            OPTIONS {{indexConfig: {{
                `vector.dimensions`: {dim},
                `vector.similarity_function`: 'cosine'
            }}}}
        """)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Embedding model — lazy singleton, loaded once per process
# ---------------------------------------------------------------------------

_embedding_model_cache: dict[str, Any] = {}


def _get_embedding_model(model_name: str = "all-MiniLM-L6-v2"):
    """Lazy-load SentenceTransformer; cached per model name."""
    if model_name not in _embedding_model_cache:
        from sentence_transformers import SentenceTransformer
        _embedding_model_cache[model_name] = SentenceTransformer(model_name, device="cpu")
    return _embedding_model_cache[model_name]


def store_embeddings(
    session,
    project_id: str,
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 128,
) -> int:
    """
    Generate embeddings for all CodeNodes in a project and store as n.embedding.
    Idempotent — skips nodes that already have embeddings.
    Returns the number of nodes embedded.
    """
    result = session.run("""
        MATCH (n:CodeNode {project_id: $pid})
        WHERE n.embedding IS NULL
        RETURN n.node_id AS node_id,
               COALESCE(n.name, '') AS name,
               COALESCE(n.docstring, '') AS docstring,
               COALESCE(n.content, '') AS content
    """, pid=project_id)

    rows = [
        (r["node_id"], f"{r['name']} {r['docstring']} {r['content']}"[:2000])
        for r in result
    ]
    if not rows:
        return 0

    model = _get_embedding_model(model_name)
    texts = [text for _, text in rows]
    embeddings = model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                              show_progress_bar=False)

    write_rows = [
        {"node_id": node_id, "embedding": emb.tolist()}
        for (node_id, _), emb in zip(rows, embeddings)
    ]
    for batch in _batched(write_rows):
        session.run("""
            UNWIND $rows AS row
            MATCH (n:CodeNode {node_id: row.node_id})
            SET n.embedding = row.embedding
        """, rows=batch)

    return len(rows)


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


def vector_search(
    session,
    query_embedding: list[float],
    project_id: str,
    top_k: int = 20,
) -> list[dict]:
    """
    Vector cosine similarity search over CodeNode.embedding.
    Returns [] gracefully if the index or embeddings don't exist yet.
    """
    try:
        results = session.run("""
            CALL db.index.vector.queryNodes('codeNodeEmbedding', $top_k, $embedding)
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
        """, embedding=query_embedding, pid=project_id, top_k=top_k)
        return [dict(r) for r in results]
    except Exception:
        return []


def hybrid_search(
    session,
    query: str,
    project_id: str,
    top_k: int = 20,
    model_name: str = "all-MiniLM-L6-v2",
) -> list[dict]:
    """
    Hybrid Phase 1 retrieval: BM25 fulltext + vector cosine, merged via RRF.

    Falls back to BM25-only if sentence-transformers isn't installed or
    embeddings haven't been generated for this project yet.

    RRF formula: score = Σ 1/(k + rank_i), k=60.
    Rank-based fusion means BM25 and vector scores are naturally comparable.
    """
    bm25_results = bm25_search(session, query, project_id, top_k=top_k * 2)

    try:
        model = _get_embedding_model(model_name)
        q_emb = model.encode(query, normalize_embeddings=True).tolist()
        vec_results = vector_search(session, q_emb, project_id, top_k=top_k * 2)
    except Exception:
        return bm25_results[:top_k]

    if not vec_results:
        return bm25_results[:top_k]

    RRF_K = 60
    rrf_scores: dict[str, float] = {}
    node_data: dict[str, dict] = {}

    for rank, node in enumerate(bm25_results):
        nid = node.get("node_id", "")
        if not nid:
            continue
        rrf_scores[nid] = rrf_scores.get(nid, 0.0) + 1.0 / (RRF_K + rank + 1)
        node_data[nid] = node

    for rank, node in enumerate(vec_results):
        nid = node.get("node_id", "")
        if not nid:
            continue
        rrf_scores[nid] = rrf_scores.get(nid, 0.0) + 1.0 / (RRF_K + rank + 1)
        if nid not in node_data:
            node_data[nid] = node

    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [
        {**node_data[nid], "score": score}
        for nid, score in ranked[:top_k]
        if nid in node_data
    ]


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
