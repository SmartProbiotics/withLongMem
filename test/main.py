import torch
from milvus_model.hybrid import BGEM3EmbeddingFunction
ef = BGEM3EmbeddingFunction(model_name_or_path="./test", use_fp16=False, device="cpu")
dense_dim = ef.dim["dense"]
chunk=["这是一个爱好测试文本。","测试的篮球游戏视频。","奔跑","我喜欢到处打篮球"]
chunks_embeddings = ef(chunk)
print("chunks_embeddings:", chunks_embeddings)
def similarity(embedding1, embedding2):
    """计算两个向量的余弦相似度"""
    embedding1 = torch.tensor(embedding1)
    embedding2 = torch.tensor(embedding2)
    cos_sim = torch.nn.functional.cosine_similarity(embedding1, embedding2, dim=0)
    return cos_sim.item()

print("相似度测试：")
for i in range(len(chunks_embeddings['dense'])):
    for j in range(i + 1, len(chunks_embeddings['dense'])):
        sim = similarity(chunks_embeddings['dense'][i], chunks_embeddings['dense'][j])
        print(f"文本 {chunk[i]} 和文本 {chunk[j]} 的相似度: {sim:.4f}")
print(dense_dim)
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)

# 连接到本地 Milvus Lite 数据库
connections.connect(uri="./milvus_demo.db")
field=[
    FieldSchema(name="id", dtype=DataType.VARCHAR,is_primary=True, auto_id=True, max_length=64),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=512),
    FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
    FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=dense_dim),
]
schema=CollectionSchema(fields=field)
col_name = "hybrid_demo"
if utility.has_collection(col_name):
    Collection(col_name).drop()
col = Collection(col_name, schema, consistency_level="Strong")

sparse_index = {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"}
col.create_index("sparse_vector", sparse_index)
dense_index = {"index_type": "AUTOINDEX", "metric_type": "IP"}
col.create_index("dense_vector", dense_index)
col.load()
for i in range(0, len(chunk), 50):
    batched_entities = [
        chunk[i : i + 50],
        chunks_embeddings["sparse"][i : i + 50],
        chunks_embeddings["dense"][i : i + 50],
    ]
    col.insert(batched_entities)
print("Number of entities inserted:", col.num_entities)
query = '爱好运动'

query_embeddings = ef([query])
print("query_embeddings:", query_embeddings)

# 执行混合向量搜索（同时考虑密集和稀疏向量）
print("\n=== 密集向量搜索结果 ===")
dense_search_params = {
    "metric_type": "IP",
    "params": {"nprobe": 10}
}

# 密集向量搜索
dense_results = col.search(
    data=[query_embeddings["dense"][0]],
    anns_field="dense_vector",
    param=dense_search_params,
    limit=3,
    output_fields=["text"]
)

for hits in dense_results:
    for rank, hit in enumerate(hits):
        print(f"第{rank+1}个匹配结果（密集向量）：")
        print(f"  文本: {hit.entity.get('text')}")
        print(f"  相似度分数: {hit.score:.4f}")

print("\n=== 稀疏向量搜索结果 ===")
sparse_search_params = {
    "metric_type": "IP",
    "params": {}
}

# 稀疏向量搜索
sparse_results = col.search(
    data=[query_embeddings["sparse"]],
    anns_field="sparse_vector",
    param=sparse_search_params,
    limit=3,
    output_fields=["text"]
)

for hits in sparse_results:
    for rank, hit in enumerate(hits):
        print(f"第{rank+1}个匹配结果（稀疏向量）：")
        print(f"  文本: {hit.entity.get('text')}")
        print(f"  相似度分数: {hit.score:.4f}")

print("\n=== 混合向量搜索结果（加权组合） ===")
# 获取两种向量的搜索结果，然后手动组合
# 这里我们使用简单的加权平均方法
dense_weight = 0.7
sparse_weight = 0.3

from pymilvus import (
    AnnSearchRequest,
    WeightedRanker,
)

def hybrid_search(
    col,
    query_dense_embedding,
    query_sparse_embedding,
    sparse_weight=1.0,
    dense_weight=1.0,
    limit=10,
):
    dense_search_params = {"metric_type": "IP", "params": {}}
    dense_req = AnnSearchRequest(
        [query_dense_embedding], "dense_vector", dense_search_params, limit=limit
    )
    sparse_search_params = {"metric_type": "IP", "params": {}}
    sparse_req = AnnSearchRequest(
        [query_sparse_embedding], "sparse_vector", sparse_search_params, limit=limit
    )
    rerank = WeightedRanker(sparse_weight, dense_weight)
    res = col.hybrid_search(
        [sparse_req, dense_req], rerank=rerank, limit=limit, output_fields=["text"]
    )[0]
    return [hit.get("text") for hit in res]

print(hybrid_search(
    col,
    query_embeddings["dense"][0],
    query_embeddings["sparse"]._getrow(0),
    sparse_weight=0.7,
    dense_weight=1.0,
))
