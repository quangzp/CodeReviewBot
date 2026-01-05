import weaviate
from typing import List
from src_bot.neo4jdb.neo4j_db import Neo4jDB
from src_bot.config.config import configs
from sentence_transformers import SentenceTransformer

class CustomGraphRAGRetriever:
    def __init__(self):
        self.neo4j_db = Neo4jDB()
        self.driver = self.neo4j_db.driver
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
        self.driver.close()
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
        with self.driver.session() as session:
            for item in results:
                ast_hash = item.get("ast_hash")
                graph_data = session.run(self.cypher_query, weaviate_id=ast_hash).single()
                #print(item)
                #print(graph_data)
                if graph_data:
                     context_str = self._format_context(item, graph_data)
                     final_context.append(context_str)
        
        return final_context

    def _format_context(self, vector_data, graph_data) -> str:
        node_type = graph_data["node_type"][0] if len(graph_data["node_type"]) else "Unknown"
        name = graph_data["name"]
        
        description = f"\n--- FOUND CONTEXT: {node_type} '{name}' ---\n"
        
        if node_type == "EndpointNode":
            description += f"Logic xử lý: {graph_data['triggers_endpoint']}\n"
            description += f"Nội dung code đầy đủ:\n```\n{graph_data['code']}\n```\n"
            
        elif node_type == "MethodNode":
            description += f"Được gọi bởi (Callers): {graph_data['called_by']}\n"
            description += f"Gọi đến (Callees): {graph_data['calls_to']}\n"
            description += f"Thuộc API (Triggered by): {graph_data['triggers_endpoint']}\n"
            description += f"Nội dung code đầy đủ:\n```\n{graph_data['code']}\n```\n"
            
        elif node_type == "ClassNode":
            description += f"Nội dung code đầy đủ:\n```\n{graph_data['code']}\n```\n"
            
        return description
    