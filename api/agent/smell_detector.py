"""
Graph-based code smell detector.

Uses the Neo4j Code Property Graph (built by swebench/neo4j_ingest.py) to
identify structural issues in indexed projects:

  - God classes    — too many methods or too many lines
  - Long methods   — high line count
  - Feature envy   — method calls more other-class methods than its own
  - Dead code      — functions with zero callers in the graph

Each detector returns a list of SmellResult. Run detect_all_smells() to
get a combined, severity-sorted list.

Requires F007 (neo4j_ingest improvements): CodeNode.line_count,
CodeNode.method_count, and INHERITS edges must exist in the graph.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Thresholds (tunable)
# ---------------------------------------------------------------------------
GOD_CLASS_METHOD_THRESHOLD = 10
GOD_CLASS_LINE_THRESHOLD = 300
LONG_METHOD_LINE_THRESHOLD = 50


@dataclass
class SmellResult:
    """A detected code smell."""
    smell_type: str           # "god_class" | "long_method" | "feature_envy" | "dead_code"
    file_path: str
    name: str                 # function / class name
    severity: str             # "low" | "medium" | "high"
    description: str
    metric_value: Optional[float] = None
    threshold: float = 0.0


# ---------------------------------------------------------------------------
# Individual detectors
# ---------------------------------------------------------------------------

def detect_god_classes(neo4j_session, project_id: str) -> list[SmellResult]:
    """Find classes with too many methods or too many lines."""
    try:
        results = neo4j_session.run("""
            MATCH (n:CodeNode {type: 'Class', project_id: $pid})
            WHERE n.method_count > $method_thresh
               OR n.line_count > $line_thresh
            RETURN n.name AS name, n.file_path AS file_path,
                   n.method_count AS method_count, n.line_count AS line_count,
                   n.qualified_name AS qualified_name
            ORDER BY n.method_count DESC
            LIMIT 20
        """, pid=project_id,
            method_thresh=GOD_CLASS_METHOD_THRESHOLD,
            line_thresh=GOD_CLASS_LINE_THRESHOLD)

        smells = []
        for r in results:
            mc = r["method_count"] or 0
            lc = r["line_count"] or 0
            if mc > GOD_CLASS_METHOD_THRESHOLD * 2 or lc > GOD_CLASS_LINE_THRESHOLD * 2:
                severity = "high"
            else:
                severity = "medium"
            smells.append(SmellResult(
                smell_type="god_class",
                file_path=r["file_path"] or "",
                name=r["name"] or "",
                severity=severity,
                description=(
                    f"Class '{r['name']}' has {mc} methods and {lc} lines. "
                    "Consider splitting into smaller, focused classes."
                ),
                metric_value=float(mc),
                threshold=float(GOD_CLASS_METHOD_THRESHOLD),
            ))
        return smells
    except Exception:
        return []


def detect_long_methods(neo4j_session, project_id: str) -> list[SmellResult]:
    """Find functions/methods with too many lines."""
    try:
        results = neo4j_session.run("""
            MATCH (n:CodeNode {project_id: $pid})
            WHERE n.type IN ['Function', 'Method']
              AND n.line_count > $thresh
            RETURN n.name AS name, n.file_path AS file_path,
                   n.line_count AS line_count,
                   n.qualified_name AS qualified_name
            ORDER BY n.line_count DESC
            LIMIT 20
        """, pid=project_id, thresh=LONG_METHOD_LINE_THRESHOLD)

        smells = []
        for r in results:
            lc = r["line_count"] or 0
            if lc > LONG_METHOD_LINE_THRESHOLD * 3:
                severity = "high"
            elif lc > LONG_METHOD_LINE_THRESHOLD * 2:
                severity = "medium"
            else:
                severity = "low"
            smells.append(SmellResult(
                smell_type="long_method",
                file_path=r["file_path"] or "",
                name=r["qualified_name"] or r["name"] or "",
                severity=severity,
                description=(
                    f"Function '{r['name']}' is {lc} lines long. "
                    "Consider extracting sub-functions."
                ),
                metric_value=float(lc),
                threshold=float(LONG_METHOD_LINE_THRESHOLD),
            ))
        return smells
    except Exception:
        return []


def detect_dead_code(neo4j_session, project_id: str) -> list[SmellResult]:
    """Find functions with zero callers (excluding dunder, main, setUp, tearDown)."""
    try:
        results = neo4j_session.run("""
            MATCH (n:CodeNode {project_id: $pid})
            WHERE n.type IN ['Function', 'Method']
              AND NOT n.name STARTS WITH '_'
              AND NOT n.name IN ['main', 'setUp', 'tearDown', 'setUpClass', 'tearDownClass']
            OPTIONAL MATCH (caller:CodeNode {project_id: $pid})-[:CALLS]->(n)
            WITH n, count(caller) AS caller_count
            WHERE caller_count = 0
            RETURN n.name AS name, n.file_path AS file_path,
                   n.qualified_name AS qualified_name,
                   n.line_count AS line_count
            ORDER BY n.line_count DESC
            LIMIT 30
        """, pid=project_id)

        smells = []
        for r in results:
            smells.append(SmellResult(
                smell_type="dead_code",
                file_path=r["file_path"] or "",
                name=r["qualified_name"] or r["name"] or "",
                severity="low",
                description=(
                    f"Function '{r['name']}' has no callers in the codebase. "
                    "May be dead code or only called externally."
                ),
                metric_value=0.0,
            ))
        return smells
    except Exception:
        return []


def detect_feature_envy(neo4j_session, project_id: str) -> list[SmellResult]:
    """Find methods that call more methods from OTHER classes than their own."""
    try:
        results = neo4j_session.run("""
            MATCH (m:CodeNode {type: 'Method', project_id: $pid})-[:CALLS]->(callee:CodeNode {project_id: $pid})
            WHERE callee.type = 'Method'
            WITH m,
                 CASE WHEN split(m.qualified_name, '.')[0] = split(callee.qualified_name, '.')[0]
                      THEN 'same_class' ELSE 'other_class' END AS call_type
            WITH m,
                 sum(CASE WHEN call_type = 'same_class' THEN 1 ELSE 0 END) AS own_calls,
                 sum(CASE WHEN call_type = 'other_class' THEN 1 ELSE 0 END) AS other_calls
            WHERE other_calls > own_calls AND other_calls >= 3
            RETURN m.name AS name, m.file_path AS file_path,
                   m.qualified_name AS qualified_name,
                   own_calls, other_calls
            ORDER BY other_calls DESC
            LIMIT 15
        """, pid=project_id)

        smells = []
        for r in results:
            smells.append(SmellResult(
                smell_type="feature_envy",
                file_path=r["file_path"] or "",
                name=r["qualified_name"] or r["name"] or "",
                severity="medium",
                description=(
                    f"Method '{r['name']}' calls {r['other_calls']} methods from other classes "
                    f"but only {r['own_calls']} from its own. Consider moving this method."
                ),
                metric_value=float(r["other_calls"]),
            ))
        return smells
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Combined entry point
# ---------------------------------------------------------------------------

def detect_all_smells(
    neo4j_session,
    project_id: str,
    file_path: str = "",
) -> list[SmellResult]:
    """
    Run all detectors and return a combined, severity-sorted list.

    Args:
        neo4j_session: Active Neo4j session.
        project_id: The project identifier (e.g. "owner_repo").
        file_path: Optional. If given, filter results to that file only.

    Returns:
        List of SmellResult sorted by severity (high → medium → low).
    """
    all_smells: list[SmellResult] = []
    all_smells.extend(detect_god_classes(neo4j_session, project_id))
    all_smells.extend(detect_long_methods(neo4j_session, project_id))
    all_smells.extend(detect_dead_code(neo4j_session, project_id))
    all_smells.extend(detect_feature_envy(neo4j_session, project_id))

    if file_path:
        all_smells = [s for s in all_smells if s.file_path == file_path]

    severity_order = {"high": 0, "medium": 1, "low": 2}
    all_smells.sort(key=lambda s: severity_order.get(s.severity, 3))

    return all_smells
