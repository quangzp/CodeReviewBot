import json
import weaviate
from weaviate.util import generate_uuid5
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from src_bot.neo4jdb.neo4j_db import Neo4jDB

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

chunks = run_query(q)

model = SentenceTransformer(
    "microsoft/codebert-base",
    device="cpu"
)

texts = ["passage: " + chunk["content"] for chunk in chunks]

print("Generating embeddings...")
embeddings = model.encode(
    texts,
    batch_size=64,         
    show_progress_bar=True,
    normalize_embeddings=True
)

# Gắn embeddings
for i, chunk in enumerate(chunks):
    chunk["embedding"] = embeddings[i].tolist()

client = weaviate.connect_to_local()

COLLECTION = "CodeBotCollection"

collection = client.collections.use(COLLECTION)

# Batch ingest + progress bar
with collection.batch.fixed_size(batch_size=10) as batch:
    for chunk in tqdm(
        chunks,
        desc="📥 Ingesting chunks",
        unit="chunk",
        ncols=100,
    ):
        properties = {
            "name": chunk["name"],
            "content": chunk["content"],
            "file_path": chunk["source"],
            "node_type": chunk["type"],
            "ast_hash": chunk["id"],
        }

        uuid = generate_uuid5(chunk["id"])

        batch.add_object(
            properties=properties,
            uuid=uuid,
            vector=chunk["embedding"],
        )

print(f"✅ Ingested {len(chunks)} documents into Weaviate")

# Kiểm tra total count (v4)
agg = collection.aggregate.over_all(total_count=True)
print(f"Total documents in DB: {agg.total_count}")

# Kiểm tra lỗi batch
failed = collection.batch.failed_objects
if failed:
    print(f"⚠️ Failed objects: {len(failed)}")
    print("First failed object:", failed[0])

client.close()