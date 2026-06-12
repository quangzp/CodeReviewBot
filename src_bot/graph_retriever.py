from __future__ import annotations

"""
Two-phase bounded GraphRAG retriever — Neo4j only (no Weaviate).

Phase 1 (Broad): Neo4j hybrid search (BM25 + vector cosine via RRF) → top-k seed nodes
Phase 2 (Narrow): Neo4j graph traversal → 2-3 hop expansion per seed

Context size is O(k * hop_fanout) regardless of repo size.
For a 10,000-file repo: 20 seeds × ~25 nodes/seed = ~500 nodes max.
"""

from typing import List, Optional
from neo4j import GraphDatabase
from src_bot.neo4jdb.neo4j_service import Neo4jService
from src_bot.config.config import configs
from swebench.neo4j_ingest import hybrid_search


class CustomGraphRAGRetriever:
    def __init__(self):
        self.neo4j_service = Neo4jService()
        self._driver = GraphDatabase.driver(
            configs.APP_NEO4J_URL,
            auth=(configs.APP_NEO4J_USER, configs.APP_NEO4J_PASSWORD),
        )

    def close(self):
        self.neo4j_service.db.driver.close()
        self._driver.close()

    def search(self, query_text: str, top_k: int = None, project_id: str = "") -> List[str]:
        """
        Two-phase hybrid search: Neo4j BM25+vector → graph expansion.

        Args:
            query_text: natural-language or code query
            top_k: number of seed nodes (default from config)
            project_id: project_id to scope the search

        Returns list of formatted context strings for LLM consumption.
        """
        top_k = top_k or configs.RETRIEVER_TOP_K
        max_hops = configs.RETRIEVER_MAX_HOPS

        with self._driver.session() as session:
            seed_nodes = hybrid_search(
                session,
                query_text,
                project_id,
                top_k=top_k,
                model_name=configs.EMBEDDING_MODEL,
            )

        if not seed_nodes:
            return []

        final_context = []
        total_nodes = 0

        for item in seed_nodes:
            if total_nodes >= configs.RETRIEVER_MAX_CONTEXT_NODES:
                break

            node_id = item.get("node_id")
            if not node_id:
                continue

            graph_data = self.neo4j_service.get_node_by_ast_hash(node_id)
            if not graph_data:
                continue

            related_nodes = self.neo4j_service.get_related_nodes(
                [graph_data],
                max_level=max_hops,
                max_results=50,
            )
            relationship_data = self.neo4j_service.extract_relationships(related_nodes)

            context_str = self._format_context(item, relationship_data)
            if context_str:
                final_context.append(context_str)
                total_nodes += len(related_nodes) + 1

        return final_context

    def _format_context(
        self, seed_node: dict, relationship_data: List[dict]
    ) -> Optional[str]:
        parts = []

        name = seed_node.get("name", "unknown")
        node_type = seed_node.get("type", "unknown")
        file_path = seed_node.get("file_path", "unknown")
        content = seed_node.get("content", "")

        parts.append(f"## {node_type}: {name}")
        parts.append(f"File: {file_path}")
        if content:
            parts.append(f"```\n{content}\n```")

        if relationship_data:
            parts.append("\nRelated code from knowledge graph:")
            for rel in relationship_data:
                rel_type = rel.get("relationship_type", "UNKNOWN")
                from_labels = ", ".join(rel.get("from_labels", []))
                to_labels = ", ".join(rel.get("to_labels", []))
                from_content = rel.get("from_content", "")
                to_content = rel.get("to_content", "")

                parts.append(f"- [{rel_type}] ({from_labels}) -> ({to_labels})")
                if from_content:
                    parts.append(f"  From:\n  ```\n{from_content}\n  ```")
                if to_content:
                    parts.append(f"  To:\n  ```\n{to_content}\n  ```")

        return "\n".join(parts) if parts else None
