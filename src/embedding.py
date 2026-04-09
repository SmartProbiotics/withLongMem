import os
import torch
import numpy as np
from typing import List, Dict, Any
import dashscope
from dashscope import TextEmbedding
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 获取 DashScope API Key
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
if not DASHSCOPE_API_KEY:
    print("❌ 未找到 DASHSCOPE_API_KEY 环境变量")
    print("💡 请在 .env 文件中添加: DASHSCOPE_API_KEY=your-api-key")
    print("🔗 获取 API Key: https://help.aliyun.com/document_detail/611472.html")
    exit(1)

dashscope.api_key = DASHSCOPE_API_KEY
dashscope.base_http_api_url = 'https://dashscope.aliyuncs.com/api/v1'


class DashScopeEmbeddingV2:
    """阿里云 DashScope 文本嵌入模型 v2"""
    
    def __init__(self, dense_model_name="text-embedding-v4", sparse_model_name="text-embedding-v4", dimension=512):
        """
        初始化 DashScope 嵌入模型
        
        Args:
            dense_model_name: 密集模型名称，默认 text-embedding-v4
            sparse_model_name: 稀疏模型名称，默认 ops-text-sparse-embedding-001
            dimension: 向量维度，v2 模型支持 512 维
        """
        self.dense_model_name = dense_model_name
        self.sparse_model_name = sparse_model_name
        self.dimension = dimension
        print(f"✅ 初始化 DashScope 嵌入模型: {dense_model_name}, {sparse_model_name}")
        print(f"📊 向量维度: {dimension}")
    
    def __call__(self, texts: List[str], text_type: str = "document") -> Dict[str, List]:
        """使函数可调用，兼容原有接口"""
        return self.encode(texts, text_type)
    
    def encode(self, texts: List[str], text_type: str = "document") -> Dict[str, List]:
        """
        将文本编码为嵌入向量
        Args:
            texts: 文本列表
            text_type: 文本类型，"document" 或 "query"
            
        Returns:
            包含密集向量和稀疏向量的字典
            {
            "dense": dense_vectors,:list of [0.123, 0.456, ...],
            "sparse": sparse_vectors:list of [{'index': 105464, 'token': '这是一个', 'value': 2.1484}, {'index': 103059, 'token': '爱好', 'value': 2.1836}, {'index': 81705, 'token': '测试', 'value': 1.999}, {'index': 108704, 'token': '文本', 'value': 1.6162}, {'index': 1773, 'token': '。', 'value': 1.0615}]
            }
        """
        # 调用 DashScope API
        
        # 提取密集向量
        dense_vectors = []
        sparse_vectors = []
        # 生成简化版稀疏向量
        response=TextEmbedding.call(
            model=self.sparse_model_name,
            input=texts,
            text_type=text_type,
            output_type="sparse",
            dimension=self.dimension
        )
        print(response.output)
        for embedding_data in response.output["embeddings"]:
            sparse_vector = embedding_data["sparse_embedding"]
            sparse_vectors.append(sparse_vector)
            print("sparse_vector:"+str(sparse_vector))
        # print(response.output)
        response = TextEmbedding.call(
            model=self.dense_model_name,
            input=texts,
            text_type=text_type,
            output_type="dense",
            dimension=self.dimension
        )
        
        for embedding_data in response.output["embeddings"]:
            # 密集向量
            dense_vector = embedding_data["embedding"]
            dense_vectors.append(dense_vector)
            
        return {
            "dense": dense_vectors,
            "sparse": sparse_vectors
        }
print("\n📦 初始化 DashScope 嵌入模型...")
ef = DashScopeEmbeddingV2(dense_model_name="text-embedding-v4",sparse_model_name= "text-embedding-v4", dimension=512)
