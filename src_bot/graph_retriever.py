from __future__ import annotations

"""
Two-phase bounded GraphRAG retriever.

Phase 1 (Broad): Weaviate semantic + keyword hybrid search -> top-k seed nodes
Phase 2 (Narrow): Neo4j graph traversal -> 2-3 hop expansion per seed

This ensures O(k * hop_fanout) context size regardless of repo size.
For a 10,000-file Django repo: 20 seeds * ~25 nodes/seed = ~500 nodes max.
"""

import weaviate
from typing import List, Optional
from src_bot.neo4jdb.neo4j_service import Neo4jService
from src_bot.config.config import configs
from sentence_transformers import SentenceTransformer


class CustomGraphRAGRetriever:
    def __init__(self):
        self.neo4j_service = Neo4jService()
        self.model = SentenceTransformer(
            "microsoft/codebert-base",
            device="cpu",
        )
        self.weaviate_client = weaviate.connect_to_local()
        self.weaviate_collection = configs.WEAVIATE_COLLECTION_NAME

    def close(self):
        """Close connections when done."""
        self.neo4j_service.db.driver.close()
        self.weaviate_client.close()

    def search(self, query_text: str, top_k: int = None) -> List[str]:
        """
        Two-phase hybrid search: Vector (Weaviate) -> Graph (Neo4j).

        Phase 1: Weaviate finds top-k semantically similar code chunks.
        Phase 2: For each seed, Neo4j expands local graph context (bounded).

        Returns list of formatted context strings for LLM consumption.
        """
        top_k = top_k or configs.RETRIEVER_TOP_K
        max_hops = configs.RETRIEVER_MAX_HOPS

        # Phase 1: Semantic search in Weaviate (fast, bounded)
        seed_nodes = self._weaviate_search(query_text, top_k=top_k)

        if not seed_nodes:
            return []

        # Phase 2: Graph expansion from seed nodes (bounded per seed)
        final_context = []
        total_nodes = 0

        for item in seed_nodes:
            if total_nodes >= configs.RETRIEVER_MAX_CONTEXT_NODES:
                break

            ast_hash = item.get("ast_hash")
            if not ast_hash:
                continue

            graph_data = self.neo4j_service.get_node_by_ast_hash(ast_hash)
            if not graph_data:
                continue

            # Bounded traversal: max_hops per seed (default 3, not 7/20)
            related_nodes = self.neo4j_service.get_related_nodes(
                [graph_data],
                max_level=max_hops,
                max_results=50,  # cap per seed
            )
            relationship_data = self.neo4j_service.extract_relationships(related_nodes)

            # Format the seed node info + relationships
            context_str = self._format_context(item, relationship_data)
            if context_str:
                final_context.append(context_str)
                total_nodes += len(related_nodes) + 1

        return final_context

    def _weaviate_search(self, query_text: str, top_k: int) -> List[dict]:
        """Phase 1: Weaviate hybrid search (vector + keyword)."""
        query_embedding = self.model.encode(query_text, normalize_embeddings=True)
        collection = self.weaviate_client.collections.use(self.weaviate_collection)
        response = collection.query.hybrid(
            query=query_text,
            vector=query_embedding.tolist(),
            alpha=0.5,
            limit=top_k,
            return_properties=["ast_hash", "name", "content", "file_path", "node_type"],
        )

        results = []
        for obj in response.objects:
            results.append({
                "ast_hash": obj.properties.get("ast_hash"),
                "name": obj.properties.get("name"),
                "content": obj.properties.get("content"),
                "file_path": obj.properties.get("file_path"),
                "node_type": obj.properties.get("node_type"),
            })
        return results

    def _format_context(
        self, seed_node: dict, relationship_data: List[dict]
    ) -> Optional[str]:
        """Format a seed node and its graph relationships for LLM context."""
        parts = []

        # Seed node info
        name = seed_node.get("name", "unknown")
        node_type = seed_node.get("node_type", "unknown")
        file_path = seed_node.get("file_path", "unknown")
        content = seed_node.get("content", "")

        parts.append(f"## {node_type}: {name}")
        parts.append(f"File: {file_path}")
        if content:
            parts.append(f"```\n{content}\n```")

        # Graph relationships
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
