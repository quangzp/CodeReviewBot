import weaviate
from typing import List
from neo4jdb.neo4j_db import Neo4jDB

class CustomGraphRAGRetriever:
    def __init__(self):
        self.neo4j_db = Neo4jDB()
        self.driver = self.neo4j_db.driver
        self.weaviate_client = weaviate.connect_to_local()
        self.cypher_query = """
        MATCH (startNode) WHERE startNode.id = $weaviate_id
        
        // Case 1: Nếu là Endpoint -> Tìm Method implement và Config liên quan
        OPTIONAL MATCH (startNode:EndpointNode)-[:IMPLEMENT]->(implMethod:MethodNode)
        OPTIONAL MATCH (implMethod)-[:USE]->(usedConfig:Configuration)
        
        // Case 2: Nếu là Method -> Tìm Caller, Callee, Config và Endpoint cha
        OPTIONAL MATCH (caller:MethodNode)-[:CALL]->(startNode:MethodNode)
        OPTIONAL MATCH (startNode)-[:CALL]->(callee:MethodNode)
        OPTIONAL MATCH (startNode)-[:USE]->(methodConfig:ConfigurationNode)
        OPTIONAL MATCH (startNode)<-[:IMPLEMENT]-(parentEndpoint:EndpointNode)
        
        // Case 3: Nếu là Configuration -> Impact Analysis (Ai đang dùng nó?)
        OPTIONAL MATCH (startNode:ConfigurationNode)<-[:USE]-(userMethod:MethodNode)
        
        RETURN 
            labels(startNode) as type,
            startNode.name as name,
            startNode.code as code,
            // Gom nhóm thông tin
            collect(DISTINCT implMethod.name) as implements_logic,
            collect(DISTINCT usedConfig.key + '='+ usedConfig.value) as uses_config, // Lấy cả value config
            collect(DISTINCT methodConfig.key + '='+ methodConfig.value) as method_configs,
            collect(DISTINCT caller.name) as called_by,
            collect(DISTINCT callee.name) as calls_to,
            collect(DISTINCT parentEndpoint.url) as triggers_endpoint,
            collect(DISTINCT userMethod.name) as impacts_methods
        """
    def close(self):
        """Đóng kết nối khi không dùng nữa"""
        self.driver.close()
        self.weaviate_client.close()

    def search(self, query_text: str, top_k: int = 3) -> List[str]:
        """
        Thực hiện tìm kiếm lai: Vector (Weaviate) -> Graph (Neo4j)
        """
        response = (
            self.weaviate_client.query
            .get("CodeChunk", ["neo4j_id", "content", "functionName", "node_type"])
            .with_hybrid(query=query_text)
            .with_limit(top_k)
            .do()
        )
        
        results = response.get("data", {}).get("Get", {}).get("CodeChunk", [])
        final_context = []

       
        with self.driver.session() as session:
            for item in results:
                neo4j_id = item.get("neo4j_id")
               
                graph_data = session.run(self.cypher_query, weaviate_id=neo4j_id).single()
                
                if graph_data:
                    context_str = self._format_context(item, graph_data)
                    final_context.append(context_str)
        
        return final_context

    def _format_context(self, vector_data, graph_data) -> str:
        """Helper để tạo văn bản mô tả ngữ cảnh dễ hiểu cho LLM"""
        node_type = graph_data["type"][0] if graph_data["type"] else "Unknown"
        name = graph_data["name"]
        
        description = f"\n--- FOUND CONTEXT: {node_type} '{name}' ---\n"
        
        if node_type == "Endpoint":
            description += f"Logic xử lý (Implementation): {graph_data['implements_logic']}\n"
            description += f"Cấu hình liên quan: {graph_data['uses_config']}\n"
            
        elif node_type == "Method":
            description += f"Được gọi bởi (Callers): {graph_data['called_by']}\n"
            description += f"Gọi đến (Callees): {graph_data['calls_to']}\n"
            description += f"Thuộc API (Triggered by): {graph_data['triggers_endpoint']}\n"
            description += f"Sử dụng Config: {graph_data['method_configs']}\n"
            description += f"Nội dung code đầy đủ:\n```\n{graph_data['code']}\n```\n"
            
        elif node_type == "Configuration":
            description += f"Giá trị cấu hình: {vector_data['content']}\n"
            description += f"ẢNH HƯỞNG ĐẾN (Impact Analysis): Các hàm {graph_data['impacts_methods']} đang sử dụng config này.\n"
            
        return description
    