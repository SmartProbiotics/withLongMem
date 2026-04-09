from src.memoryBlock import MemoryBlock
from langchain_core.tools import tool
from typing import List
import re
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage


def generate_tool(memory_block_list: list[MemoryBlock]) -> List:
    """根据内存块列表生成动态工具函数列表"""
    tools = []
    
    for i, block in enumerate(memory_block_list):
        title = block.metadata.get('title', f'内存块{i+1}')
        content = block.text
        source = block.metadata.get('source', '未知来源')
        
        # 关键修复：使用立即执行函数表达式(IIFE)避免闭包问题
        def create_memory_tool_factory(memory_content, memory_title, memory_source, index):
            @tool
            def memory_tool() -> str:
                """获取内存块内容"""
                return f"标题: {memory_title}\n来源: {memory_source}\n内容: {memory_content}"

            
            # 生成安全的工具名称
            safe_name = f"get_memory_{index}"
            
            # 关键：正确设置LangChain工具的属性
            memory_tool.name = safe_name
            memory_tool.description = f"如果有必要获取更多关于'{memory_title}'的内存信息，来源：{memory_source}"
            
            return memory_tool
        
        # 立即调用工厂函数，确保每个工具使用独立的变量
        tool_instance = create_memory_tool_factory(content, title, source, i)
        tools.append(tool_instance)
    
    return tools

# 测试函数
def test_tool_generation():
    """测试工具生成是否正确"""
    print("=== 测试工具生成 ===\n")
    
    # 创建测试内存块
    test_blocks = [
        MemoryBlock(
            text="Python是一种高级编程语言。",
            metadata={"title": "Python介绍", "source": "教程"}
        ),
        MemoryBlock(
            text="机器学习让计算机从数据中学习。",
            metadata={"title": "机器学习", "source": "教材"}
        ),
        MemoryBlock(
            text="深度学习使用神经网络。",
            metadata={"title": "深度学习", "source": "论文"}
        )
    ]
    
    # 生成工具
    tools = generate_tool(test_blocks)
    print(tools)
    print(tools[0].invoke({}))
    # return 
    # LLM 初始化
    from src.graph_builder import llm
    llm_with_tools = llm.bind_tools(tools)
    # 测试工具调用
    response = llm_with_tools.invoke("我给你提供了哪些工具，利用工具查询关于'机器学习'的内存信息？")
    print(response)


if __name__ == "__main__":
    # 运行测试
    test_tool_generation()
