import weaviate
from typing import List
from src_bot.neo4jdb.neo4j_service import Neo4jService
from src_bot.config.config import configs
from sentence_transformers import SentenceTransformer

class CustomGraphRAGRetriever:
    def __init__(self):
        self.neo4j_service = Neo4jService()
        self.model = SentenceTransformer(
            "microsoft/codebert-base",
            device= "cpu"
        )
        self.weaviate_client = weaviate.connect_to_local()
        self.weaviate_collection = configs.WEAVIATE_COLLECTION_NAME
        self.cypher_query = """
        MATCH (startNode) WHERE startNode.ast_hash = $weaviate_id
        
        OPTIONAL MATCH (startNode:EndpointNode)-[:CALL]->(implMethod:MethodNode)
        OPTIONAL MATCH (caller:MethodNode)-[:CALL]->(startNode:MethodNode)
        OPTIONAL MATCH (startNode)-[:CALL]->(callee:MethodNode)
        OPTIONAL MATCH (usedClass:ClassNode)<-[:USE]-(startNode:ClassNode)
        OPTIONAL MATCH (startNode)-[:USE]->(usedClass:ClassNode)
        OPTIONAL MATCH (startNode)<-[:CALL]-(parentEndpoint:EndpointNode)
        
        RETURN 
            labels(startNode) as node_type,
            startNode.name as name,
            startNode.content as code,
            collect(DISTINCT usedClass.name) as uses_class, 
            collect(DISTINCT caller.name) as called_by,
            collect(DISTINCT callee.name) as calls_to,
            collect(DISTINCT parentEndpoint.url) as triggers_endpoint
        """
    def close(self):
        """Đóng kết nối khi không dùng nữa"""
        self.neo4j_service.db.driver.close()
        self.weaviate_client.close()

    def search(self, query_text: str, top_k: int = 3) -> List[str]:
        """
        Thực hiện tìm kiếm lai: Vector (Weaviate) -> Graph (Neo4j)
        """
        query_embedding = self.model.encode(query_text, normalize_embeddings=True)
        collection = self.weaviate_client.collections.use(self.weaviate_collection)
        response = collection.query.hybrid(
            query=query_text,
            vector=query_embedding.tolist(),
            alpha=0.5,
            limit=top_k,
            return_properties=["ast_hash","name","content","file_path","node_type"]       # lấy field cần in
        )
        results = []
        for i, obj in enumerate(response.objects):
            results.append({
                "ast_hash": obj.properties.get("ast_hash"),
                "name": obj.properties.get("name"),
                "content": obj.properties.get("content"),
                "file_path": obj.properties.get("file_path"),
                "node_type": obj.properties.get("node_type"),
            })
        
        final_context = []
        for item in results:
            ast_hash = item.get("ast_hash")
            graph_data = self.neo4j_service.get_node_by_ast_hash(ast_hash)
            if graph_data:
                related_nodes = self.neo4j_service.get_related_nodes([graph_data], max_level=7)
                relationship_data = self.neo4j_service.extract_relationships(related_nodes)
                context_str = self._format_context(relationship_data)
                if context_str:
                    final_context.append(context_str)
        
        return final_context

    def _format_context(self, relationship_data) -> str:
        if not relationship_data or len(relationship_data) == 0:
            return None
        
        description = "Based on the code graph, here are some related code relationships:\n"
        for rel in relationship_data:
            description += f"- Relationship Type: {rel['relationship_type']}\n"
            description += f"  From ({', '.join(rel['from_labels'])}):\n"
            description += f"  ```\n{rel['from_content']}\n```\n"
            description += f"  To ({', '.join(rel['to_labels'])}):\n"
            description += f"  ```\n{rel['to_content']}\n```\n"

        return description
    