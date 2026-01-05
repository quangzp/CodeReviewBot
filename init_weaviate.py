import weaviate
from weaviate.classes.config import Configure, Property, DataType

client = weaviate.connect_to_local()

COLLECTION = "CodeBotCollection"

# Xóa collection cũ nếu có
try:
    client.collections.delete(COLLECTION)
    print("đã xóa")
except Exception:
    pass

# Tạo collection mới
# "vectorizer": "none" (v3)  ⇢  v4 dùng self_provided (bring your own vectors)
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

# (khuyến nghị) đóng client khi xong việc
client.close()