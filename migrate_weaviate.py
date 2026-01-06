import weaviate
import os
from weaviate.classes.config import Configure, Property, DataType
from weaviate.util import generate_uuid5
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from src_bot.neo4jdb.neo4j_db import Neo4jDB

def init_weaviate(client : weaviate.WeaviateClient = None, collection_name: str = None):
    COLLECTION = collection_name
    
    if not COLLECTION:
        raise ValueError("WEAVIATE_COLLECTION_NAME is not set in environment variables.")

    try:
        client.collections.delete(COLLECTION)
        print(f"Đã xóa {COLLECTION}")

        client.collections.create(
        name=COLLECTION,
        vector_config=Configure.Vectors.self_provided(), #tự cung cấp cho db embedding
        properties=[
            Property(
                name="name",
                data_type=DataType.TEXT,
                index_filterable=True,
                index_searchable=True,
            ),
            Property(
                name="content",
                data_type=DataType.TEXT,
                index_filterable=True,
                index_searchable=True,
            ),
            Property(
                name="file_path",
                data_type=DataType.TEXT,
                index_filterable=True,
                index_searchable=True,
            ),
            Property(
                name="node_type",
                data_type=DataType.TEXT,
                index_filterable=True,
                index_searchable=False,
            ),
            Property(
                name="ast_hash",
                data_type=DataType.TEXT,
                index_filterable=True,
                index_searchable=False,
            ),
        ],
    )

        print("✅ Collection created")
    except Exception:
        pass
    finally:
        client.close()

def ingest_to_weaviate(client : weaviate.WeaviateClient = None, collection_name: str = None):

    if not collection_name:
        raise ValueError("WEAVIATE_COLLECTION_NAME is not set in environment variables.")

    db = Neo4jDB()

    def run_query(query, params=None):
        with db.driver.session() as session:
            result = session.run(query, params)
            return [record.data() for record in result]

    q = """
        MATCH (m:MethodNode) 
    RETURN m.ast_hash as id, 'Method' as type, m.name as name, m.content as content, m.file_path as source
    UNION ALL

    MATCH (e:EndpointNode)
    RETURN e.ast_hash as id, 'Endpoint' as type, e.name as name, 'API Endpoint: ' + e.endpoint as content, 'N/A' as source
    UNION ALL

    MATCH (c:ConfigurationNode)
    RETURN c.ast_hash as id, 'Configuration' as type, c.name as name, c.content as content, c.file_path as source
    UNION ALL

    MATCH (cl:ClassNode)
    RETURN cl.ast_hash as id, 'Class' as type, cl.name as name, 'Class definition for ' + cl.name as content, cl.file_path as source
    """

    chunks = run_query(q)

    model = SentenceTransformer(
        "microsoft/codebert-base",
        device="cpu"
    )

    texts = [chunk["content"] for chunk in chunks]

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

    collection = client.collections.use(collection_name)

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

if __name__ == "__main__":
    client = weaviate.connect_to_local()
    collection_name = os.getenv("WEAVIATE_COLLECTION_NAME", None)
    init_weaviate(client, collection_name)
    ingest_to_weaviate(client, collection_name)