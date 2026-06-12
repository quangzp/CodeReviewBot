"""
Neo4j storage layer for institutional memory.

Schema (lives in the same Neo4j as GraphRAG, but with different labels):

    (:Developer {login, name, first_seen_at, last_seen_at, pr_count})
    (:BugPattern {id, name, description, severity, occurrence_count, ...})
    (:Module {path, project_id, recent_bug_count, last_bug_at})
    (:Review {id, repo_name, pr_number, pr_url, reviewed_at, summary})

Relationships:
    (Developer)-[:AUTHORED]->(Review)
    (Review)-[:TOUCHED]->(Module)
    (Review)-[:CONTAINS_PATTERN {confidence}]->(BugPattern)
    (Developer)-[:TENDS_TO {confidence, evidence_count, last_observed_at, invalidated_at}]->(BugPattern)

The (Developer)-[:TENDS_TO]->(BugPattern) edge is the heart of the system —
it's the learned fact that personalizes future reviews.
"""
from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Optional

from neo4j import Driver

from src_bot.memory.schema import (
    Developer,
    BugPattern,
    ModuleStats,
    DeveloperProfile,
    PatternWithEvidence,
    ReviewFact,
)


# Threshold: how many observations before we believe a dev "tends to" make a pattern
PATTERN_THRESHOLD = 3
# Confidence growth per evidence
CONFIDENCE_PER_EVIDENCE = 0.20


class MemoryStore:
    """Neo4j-backed institutional memory."""

    def __init__(self, driver: Driver):
        self.driver = driver

    # -----------------------------------------------------------------
    # Schema setup
    # -----------------------------------------------------------------
    def ensure_indexes(self):
        """Create indexes + constraints. Idempotent."""
        with self.driver.session() as s:
            s.run("CREATE CONSTRAINT developer_login IF NOT EXISTS FOR (d:Developer) REQUIRE d.login IS UNIQUE")
            s.run("CREATE CONSTRAINT pattern_id IF NOT EXISTS FOR (p:BugPattern) REQUIRE p.id IS UNIQUE")
            s.run("CREATE CONSTRAINT review_id IF NOT EXISTS FOR (r:Review) REQUIRE r.id IS UNIQUE")
            s.run("CREATE CONSTRAINT topic_id IF NOT EXISTS FOR (t:Topic) REQUIRE t.id IS UNIQUE")
            s.run("CREATE INDEX module_path IF NOT EXISTS FOR (m:Module) ON (m.path, m.project_id)")

    # -----------------------------------------------------------------
    # Write path — called by the fact extractor after each review
    # -----------------------------------------------------------------
    def record_review(self, fact: ReviewFact, review_id: str):
        """
        Record a completed review, then update developer profiles.

        This is the main write entry-point. After this:
        - The developer's profile reflects the new evidence
        - Module hotspot stats are updated
        - If pattern threshold is crossed, a (Developer)-[:TENDS_TO]->(BugPattern) edge is created
        """
        now = datetime.now(timezone.utc).isoformat()
        with self.driver.session() as s:
            # 1. Upsert Developer
            s.run("""
                MERGE (d:Developer {login: $login})
                ON CREATE SET d.first_seen_at = $now,
                              d.pr_count = 0
                SET d.last_seen_at = $now,
                    d.pr_count = COALESCE(d.pr_count, 0) + 1
            """, login=fact.developer_login, now=now)

            # 2. Create the Review node
            s.run("""
                MERGE (r:Review {id: $review_id})
                SET r.repo_name = $repo,
                    r.pr_number = $pr_number,
                    r.pr_url = $pr_url,
                    r.reviewed_at = $reviewed_at,
                    r.summary = $summary

                WITH r
                MATCH (d:Developer {login: $login})
                MERGE (d)-[:AUTHORED]->(r)
            """,
                review_id=review_id,
                repo=fact.repo_name,
                pr_number=fact.pr_number,
                pr_url=fact.pr_url,
                reviewed_at=fact.reviewed_at.isoformat(),
                summary=fact.summary[:500],
                login=fact.developer_login,
            )

            # 3. Touch modules (update hotspot stats)
            project_id = fact.repo_name.replace("/", "_")
            for path in fact.files_touched:
                has_bug = bool(fact.bug_patterns)
                s.run("""
                    MERGE (m:Module {path: $path, project_id: $project_id})
                    ON CREATE SET m.recent_bug_count = 0, m.total_bug_count = 0
                    SET m.total_bug_count = COALESCE(m.total_bug_count, 0) + CASE WHEN $has_bug THEN 1 ELSE 0 END,
                        m.last_bug_at = CASE WHEN $has_bug THEN $now ELSE m.last_bug_at END

                    WITH m
                    MATCH (r:Review {id: $review_id})
                    MERGE (r)-[:TOUCHED]->(m)
                """, path=path, project_id=project_id, has_bug=has_bug, now=now, review_id=review_id)

            # 4. Link review to bug patterns + update developer tendencies
            for pattern_id in fact.bug_patterns:
                # Upsert pattern
                s.run("""
                    MERGE (p:BugPattern {id: $pid})
                    ON CREATE SET p.name = $name,
                                  p.first_seen_at = $now,
                                  p.occurrence_count = 0
                    SET p.last_seen_at = $now,
                        p.occurrence_count = COALESCE(p.occurrence_count, 0) + 1
                """, pid=pattern_id, name=pattern_id.replace("-", " ").title(), now=now)

                # Review contains pattern
                s.run("""
                    MATCH (r:Review {id: $review_id}), (p:BugPattern {id: $pid})
                    MERGE (r)-[:CONTAINS_PATTERN]->(p)
                """, review_id=review_id, pid=pattern_id)

                # Update developer's TENDS_TO edge (the core learning step)
                self._update_tendency(s, fact.developer_login, pattern_id, now)

    def _update_tendency(self, session, login: str, pattern_id: str, now: str):
        """
        Update (or create) the (Developer)-[:TENDS_TO]->(BugPattern) edge.
        Confidence grows with each piece of evidence.
        """
        session.run("""
            MATCH (d:Developer {login: $login}), (p:BugPattern {id: $pid})
            MERGE (d)-[t:TENDS_TO]->(p)
            ON CREATE SET t.evidence_count = 1,
                          t.confidence = $start_conf,
                          t.first_observed_at = $now,
                          t.invalidated_at = NULL
            ON MATCH SET t.evidence_count = COALESCE(t.evidence_count, 0) + 1,
                         t.confidence = CASE
                             WHEN COALESCE(t.confidence, 0) + $delta > 1.0 THEN 1.0
                             ELSE COALESCE(t.confidence, 0) + $delta
                         END,
                         t.invalidated_at = NULL
            SET t.last_observed_at = $now
        """,
            login=login,
            pid=pattern_id,
            start_conf=CONFIDENCE_PER_EVIDENCE,
            delta=CONFIDENCE_PER_EVIDENCE,
            now=now,
        )

    # -----------------------------------------------------------------
    # Read path — called by the retriever before each review
    # -----------------------------------------------------------------
    def get_developer_profile(self, login: str) -> Optional[DeveloperProfile]:
        """Get the full profile for a developer, including learned patterns."""
        with self.driver.session() as s:
            result = s.run("""
                MATCH (d:Developer {login: $login})
                OPTIONAL MATCH (d)-[t:TENDS_TO]->(p:BugPattern)
                WHERE t.invalidated_at IS NULL AND t.evidence_count >= $threshold
                RETURN d,
                       collect({pattern: p, edge: t}) AS patterns_data
            """, login=login, threshold=PATTERN_THRESHOLD).single()

            if not result or not result["d"]:
                return None

            d = result["d"]
            dev = Developer(
                login=d["login"],
                name=d.get("name"),
                avatar_url=d.get("avatar_url"),
                first_seen_at=_parse_dt(d.get("first_seen_at")),
                last_seen_at=_parse_dt(d.get("last_seen_at")),
                pr_count=d.get("pr_count", 0),
            )

            patterns = []
            for pd in result["patterns_data"]:
                if not pd.get("pattern"):
                    continue
                p = pd["pattern"]
                t = pd["edge"]
                patterns.append(PatternWithEvidence(
                    pattern=BugPattern(
                        id=p["id"],
                        name=p.get("name", p["id"]),
                        description=p.get("description", ""),
                        severity=p.get("severity", "medium"),
                        occurrence_count=p.get("occurrence_count", 0),
                        first_seen_at=_parse_dt(p.get("first_seen_at")),
                        last_seen_at=_parse_dt(p.get("last_seen_at")),
                    ),
                    confidence=t.get("confidence", 0.0),
                    evidence_count=t.get("evidence_count", 0),
                    last_observed_at=_parse_dt(t.get("last_observed_at")),
                    is_active=t.get("invalidated_at") is None,
                ))

            # Sort by confidence descending
            patterns.sort(key=lambda x: x.confidence, reverse=True)

            return DeveloperProfile(
                developer=dev,
                patterns=patterns,
            )

    def get_module_hotspot(self, path: str, project_id: str, months: int = 6) -> ModuleStats:
        """Get bug history for a module."""
        cutoff = (datetime.now(timezone.utc) - timedelta(days=30 * months)).isoformat()
        with self.driver.session() as s:
            result = s.run("""
                MATCH (m:Module {path: $path, project_id: $project_id})
                OPTIONAL MATCH (r:Review)-[:TOUCHED]->(m)
                WHERE r.reviewed_at > $cutoff
                  AND EXISTS { MATCH (r)-[:CONTAINS_PATTERN]->(:BugPattern) }
                RETURN m,
                       count(DISTINCT r) AS recent_bugs
            """, path=path, project_id=project_id, cutoff=cutoff).single()

            if not result or not result["m"]:
                return ModuleStats(path=path, project_id=project_id)

            m = result["m"]
            return ModuleStats(
                path=path,
                project_id=project_id,
                recent_bug_count=result["recent_bugs"] or 0,
                total_bug_count=m.get("total_bug_count", 0),
                last_bug_at=_parse_dt(m.get("last_bug_at")),
            )

    def list_developers(self, limit: int = 100) -> list[Developer]:
        """List all known developers, most recent first."""
        with self.driver.session() as s:
            result = s.run("""
                MATCH (d:Developer)
                RETURN d
                ORDER BY d.last_seen_at DESC
                LIMIT $limit
            """, limit=limit)
            devs = []
            for r in result:
                d = r["d"]
                devs.append(Developer(
                    login=d["login"],
                    name=d.get("name"),
                    avatar_url=d.get("avatar_url"),
                    first_seen_at=_parse_dt(d.get("first_seen_at")),
                    last_seen_at=_parse_dt(d.get("last_seen_at")),
                    pr_count=d.get("pr_count", 0),
                ))
            return devs

    def get_recent_reviews_for_developer(self, login: str, limit: int = 10) -> list[dict]:
        """Get a developer's recent reviews with bug patterns."""
        with self.driver.session() as s:
            result = s.run("""
                MATCH (d:Developer {login: $login})-[:AUTHORED]->(r:Review)
                OPTIONAL MATCH (r)-[:CONTAINS_PATTERN]->(p:BugPattern)
                RETURN r, collect(p.id) AS patterns
                ORDER BY r.reviewed_at DESC
                LIMIT $limit
            """, login=login, limit=limit)
            reviews = []
            for row in result:
                r = row["r"]
                reviews.append({
                    "id": r["id"],
                    "pr_url": r.get("pr_url"),
                    "pr_number": r.get("pr_number"),
                    "repo_name": r.get("repo_name"),
                    "summary": r.get("summary"),
                    "reviewed_at": r.get("reviewed_at"),
                    "patterns": [p for p in row["patterns"] if p],
                })
            return reviews


    # -----------------------------------------------------------------
    # Topic interest tracking — chat-driven learning
    # -----------------------------------------------------------------
    def record_topic_interest(
        self,
        developer_login: str,
        topic_id: str,
        topic_name: str,
        category: str,
    ) -> None:
        """
        Increment (Developer)-[:INTERESTED_IN]->(Topic) counter.
        Called after every chat fix/refactor to track what the user asks about.
        """
        now = datetime.now(timezone.utc).isoformat()
        with self.driver.session() as s:
            s.run("""
                MERGE (t:Topic {id: $tid})
                ON CREATE SET t.name = $name, t.category = $category,
                              t.interest_count = 0, t.first_seen_at = $now
                SET t.interest_count = COALESCE(t.interest_count, 0) + 1,
                    t.last_seen_at = $now
            """, tid=topic_id, name=topic_name, category=category, now=now)

            s.run("""
                MERGE (d:Developer {login: $login})
                ON CREATE SET d.first_seen_at = $now, d.pr_count = 0
                SET d.last_seen_at = $now
            """, login=developer_login, now=now)

            s.run("""
                MATCH (d:Developer {login: $login}), (t:Topic {id: $tid})
                MERGE (d)-[e:INTERESTED_IN]->(t)
                ON CREATE SET e.count = 1, e.first_seen_at = $now
                ON MATCH SET e.count = COALESCE(e.count, 0) + 1
                SET e.last_seen_at = $now
            """, login=developer_login, tid=topic_id, now=now)

    def get_developer_topics(self, login: str, limit: int = 10) -> list[dict]:
        """Get the topics a developer has most frequently asked about in chat."""
        with self.driver.session() as s:
            result = s.run("""
                MATCH (d:Developer {login: $login})-[e:INTERESTED_IN]->(t:Topic)
                RETURN t.id AS id, t.name AS name, t.category AS category,
                       e.count AS count, e.last_seen_at AS last_seen_at
                ORDER BY e.count DESC
                LIMIT $limit
            """, login=login, limit=limit)
            return [
                {
                    "id": row["id"],
                    "name": row["name"],
                    "category": row["category"],
                    "count": row["count"] or 0,
                    "last_seen_at": row["last_seen_at"],
                }
                for row in result
            ]


def _parse_dt(value) -> datetime:
    """Lenient datetime parser. Returns epoch on failure."""
    if not value:
        return datetime.fromtimestamp(0, tz=timezone.utc)
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except Exception:
        return datetime.fromtimestamp(0, tz=timezone.utc)
