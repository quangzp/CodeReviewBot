from neo4jdb.neo4j_db import Neo4jDB

db = Neo4jDB()

def run_query(query, params=None):
    with db.driver.session() as session:
        result = session.run(query, params)
        return [record.data() for record in result]

q = """
    MATCH (m:MethodNode) 
RETURN m.ast_hash as id, 'Method' as type, m.name as name, m.content as content, m.file_path as source
UNION ALL
//2. Lấy Endpoints (Giả sử có property url và method)
MATCH (e:EndpointNode)
RETURN e.ast_hash as id, 'Endpoint' as type, e.name as name, 'API Endpoint: ' + e.endpoint as content, 'N/A' as source
UNION ALL
//3. Lấy Configurations
MATCH (c:ConfigurationNode)
 RETURN c.ast_hash as id, 'Configuration' as type, c.name as name, c.content as content, c.file_path as source
 UNION ALL
// 4. Lấy Classes
MATCH (cl:ClassNode)
RETURN cl.ast_hash as id, 'Class' as type, cl.name as name, 'Class definition for ' + cl.name as content, cl.file_path as source
"""

data = run_query(q)
import json
with open("test_data_output.json", "w", encoding="utf-8") as f:
    f.write(json.dumps(data, indent=2, ensure_ascii=False))

print(len(data))