from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    AnnSearchRequest,
    WeightedRanker,
)
from src.memoryBlock import MemoryBlock
from src.embedding import ef
LONG_TERM_MEM="long_term_memory"
class MyMilvus:
    def __init__(self):
        # 连接到本地 Milvus Lite 数据库
        connections.connect(
            uri="./milvus_demo.db",
            # 配置keepalive参数避免too_many_pings错误
            keepalive_time=30,  # keepalive时间间隔（秒）
            keepalive_timeout=10,  # keepalive超时时间（秒）
            keepalive_permit_without_calls=True  # 允许无调用时发送keepalive
        )
    def convert_sparse_vector(self, sparse_vectors: list[dict]) -> dict:
        """将稀疏向量列表转换为 {index: value} 格式的字典"""
        return [{item["index"]: float(item["value"]) for item in sparse_list} for sparse_list in sparse_vectors]
    
    def load_collection(self, collection_name: str = "my_collection"):
        if utility.has_collection(collection_name):
            collection = Collection(collection_name)
            collection.load()
            print(f"✅ 已加载集合: {collection_name}")
        else:
            print(f"❌ 集合 {collection_name} 不存在，请先插入数据创建集合")
    def insert_data(self, data: list[str], title: list[str], source: list[str], collection_name: str = "my_collection"):
        if not utility.has_collection(collection_name):
            # 定义字段
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, auto_id=True, max_length=64),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=512),
                FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=512),
                FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=128),
                FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
                FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=512),
            ]
            # 创建集合
            schema = CollectionSchema(fields, description="DashScope 嵌入向量集合")
            collection = Collection(collection_name, schema, consistency_level="Strong")

            # 创建索引
            sparse_index = {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"}
            collection.create_index("sparse_vector", sparse_index)

            dense_index = {"index_type": "AUTOINDEX", "metric_type": "IP"}
            collection.create_index("dense_vector", dense_index)

            # 加载集合
            collection.load()
        else:
            collection = Collection(collection_name)
            collection.load()
            print(f"✅ 已加载集合: {collection_name}")
        dense_dim=ef.dimension
        print(f"📊 密集向量维度: {dense_dim}")
        contain=[d+t for d,t in zip(data,title)]
        chunks_embeddings = ef(contain, text_type="document")
        if not chunks_embeddings["dense"]:
            print("❌ 文档嵌入生成失败")
        print(f"✅ 成功生成 {len(chunks_embeddings['dense'])} 个文档嵌入向量")
        print("\n💾 插入数据到 Milvus...")
        sparse_vectors = self.convert_sparse_vector(chunks_embeddings["sparse"]) #转换稀疏向量格式为 {index: value}
        entities = [
            data,                    # text 字段
            title,                   # title 字段
            source,                  # source 字段
            sparse_vectors,  # sparse_vector 字段
            chunks_embeddings["dense"],     # dense_vector 字段
        ]
        insert_result = collection.insert(entities)
        print(f"✅ 成功插入 {len(insert_result.primary_keys)} 条数据")
        print(f"📊 集合中总数据量: {collection.num_entities}")
    
    def dense_search(self, query: str, top_k: int = 5, collection_name: str = "my_collection"):
        if not utility.has_collection(collection_name):
            print(f"❌ 集合 {collection_name} 不存在，请先插入数据创建集合")
            return
        collection = Collection(collection_name)
        collection.load()
        query_embeddings = ef([query], text_type="query")
        if not query_embeddings["dense"]:
            print("❌ 查询嵌入生成失败")
        search_params = {
            "metric_type": "IP",
            "params": {"nprobe": 10}
        }
        results = collection.search(
            data=[query_embeddings["dense"][0]],
            anns_field="dense_vector",
            param=search_params,
            limit=top_k,
            output_fields=["text","title","source"]
        )
        print(f"✅ 密集向量搜索结果 (Top {top_k}):")
        for hits in results:
            for rank, hit in enumerate(hits):
                print(f"  第{rank+1}名: {hit.entity.get('text')} (分数: {hit.score:.4f})")
        return results
    def sparse_search(self, query: str, top_k: int = 5, collection_name: str = "my_collection"):
        if not utility.has_collection(collection_name):
            print(f"❌ 集合 {collection_name} 不存在，请先插入数据创建集合")
            return
        collection = Collection(collection_name)
        collection.load()
        query_embeddings = ef([query], text_type="query")
        if not query_embeddings["sparse"]:
            print("❌ 查询嵌入生成失败")
        search_params = {
            "metric_type": "IP",
            "params": {"drop_ratio_search": 0.2} 
        }
        query_sparse_vector = self.convert_sparse_vector(query_embeddings["sparse"]) #转换查询稀疏向量格式为 {index: value}
        results = collection.search(
            data=query_sparse_vector,
            anns_field="sparse_vector",
            param=search_params,
            limit=top_k,
            output_fields=["text","title","source"]
        )
        print(f"✅ 稀疏向量搜索结果 (Top {top_k}):")
        for hits in results:
            for rank, hit in enumerate(hits):
                print(f"  第{rank+1}名: {hit.entity.get('text')} (分数: {hit.score:.4f})")
        return results
    
    def hybrid_search(self, query: str, top_k: int = 5, collection_name: str = "my_collection",dense_alpha: float = 0.5,sparse_alpha: float = 0.5):
        """返回值类型：
            [Hit(score=0.8234,entity={'text': '...', 'title': '...'}),...]
            """
        # 确保权重参数是浮点数
        dense_alpha = float(dense_alpha) if isinstance(dense_alpha, (int, float)) else 0.5
        sparse_alpha = float(sparse_alpha) if isinstance(sparse_alpha, (int, float)) else 0.5
        
        if not utility.has_collection(collection_name):
            print(f"❌ 集合 {collection_name} 不存在，请先插入数据创建集合")
            return []
        collection = Collection(collection_name)
        collection.load()
        query_embeddings = ef([query], text_type="query")
        if not query_embeddings["dense"] or not query_embeddings["sparse"]:
            print("❌ 查询嵌入生成失败")
            return []
        # 确保向量数据类型正确 - 转换为浮点数
        # 你的稠密转换非常稳健，保持不变
        dense_vectors = [[float(val) for val in vector] for vector in query_embeddings["dense"]]
        sparse_vectors = self.convert_sparse_vector(query_embeddings["sparse"])
        
        dense_req = AnnSearchRequest(
            dense_vectors,  
            "dense_vector",
            {
                "metric_type": "IP",
                "params": {"nprobe": 10} # 稠密向量使用 nprobe
            },
            limit=top_k # 建议直接和最终的 top_k 保持一致
        )
        
        sparse_req = AnnSearchRequest(
            sparse_vectors,  
            "sparse_vector",
            {
                "metric_type": "IP",
                # 稀疏向量丢弃长尾低权重特征的参数，0.2代表丢弃20%。也可以直接给空字典 {}
                "params": {"drop_ratio_search": 0.2} 
            },
            limit=top_k
        )
        
        rerank = WeightedRanker(sparse_alpha, dense_alpha)  
        
        hybrid_results = collection.hybrid_search(
            [sparse_req, dense_req], 
            rerank=rerank,
            limit=top_k,
            output_fields=["text", "title","source"]
        )[0]
        print(f"✅ 混合向量搜索结果 (Top {top_k}):")
        for rank, hit in enumerate(hybrid_results):
            print(f"  第{rank+1}名: {hit.entity.get('text')} (分数: {hit.score:.4f})")
        return hybrid_results
    def result_to_blocks(self, result):
        ret=[]
        # result 直接是 Hit 对象列表，不需要双重循环
        for hit in result:
            text   = hit.entity.get('text')
            title  = hit.entity.get('title')
            source = hit.entity.get('source') or '未知来源'
            block  = MemoryBlock(text=text, metadata={"title": title, "source": source})
            ret.append(block)
        return ret
    
    def insert_block(self, memory_block:MemoryBlock, collection_name: str = "my_collection"):
        """ :Memory Block:
            'text': "内存块文本",
            'metadata': 
                'title': "内存块标题",
                'source': "内存块来源"
        """
        self.insert_data([memory_block.text], [memory_block.metadata['title']], [memory_block.metadata['source']], collection_name=collection_name)
db=MyMilvus()

#测试代码
if __name__ == "__main__":
    query="测试保存到a.txt"
    # db.insert_block(MemoryBlock(text=query,metadata={"title":"查询","source":"用户输入"}),collection_name=LONG_TERM_MEM)
    results=db.hybrid_search(query,10,LONG_TERM_MEM,1.0,0.2)
    blocks=db.result_to_blocks(results)
    for block in blocks:
        print(block.text)
        print(block.metadata["title"])
        print(block.metadata["source"])
        print("============")