from typing import Annotated, TypedDict, Sequence
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from src.tools import get_text, write_text, execute_command, restart_agent
import os
from dotenv import load_dotenv
from datetime import datetime
from src.memory_module import AgentState, save_long_term_memory
from src.graph_builder import build_graph
# 加载环境变量
load_dotenv()

# ==========================================
# 1. 初始化 LLM
# ==========================================
llm = ChatOpenAI(
    model="qwen-plus-2025-07-28", 
    api_key=os.getenv("OPENAI_API_KEY"), # 从环境变量获取API密钥
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# ==========================================
# 2. 构建图
# ==========================================
app, memory = build_graph()
config = {"configurable": {"thread_id": "1"}}

# ==========================================
# 3. 运行测试
# ==========================================
if __name__ == "__main__":
    print("=== LangGraph 反思流程启动 ===")
    try:
        with open("./Figure.txt", "r", encoding="utf-8") as f:
            self_portrait = f.read()
        consent = input("是否允许我读取自我画像以初始化记忆？(y/n): ")
        if consent.lower() in ['y', 'yes', '是', '允许']:
            initial_memory = [AIMessage(content=f"自我画像：\n{self_portrait}")]
            result=app.invoke({"messages": initial_memory}, config=config)
            print(result)
            print("✅ 自我画像已加载到记忆中")
        else:
            initial_memory = []
            print("❌ 自我画像未加载")
    except FileNotFoundError:
        initial_memory = []
        print("⚠️ 未找到Figure.txt文件，跳过自我画像加载")
        pass
    except Exception as e:
        initial_memory = []
        print(f"⚠️ 加载自我画像时出现错误: {e}")
    
    while True:
        # 询问用户是否允许读取自我画像
        print("=== LangGraph 反思流程启动 ===")
        # 测试用例：故意让它读一个不存在的文件，触发报错和反思
        user_text = input("\n👤 用户: ") 
        if user_text.lower() in ['exit', 'quit', 'q']:
            print("再见！")
            break
            
        # ✅ 将其包装成字典，Key 必须是 "messages"
        # 且值应该是一个列表，包含一个 HumanMessage
        inputs = {"messages": [HumanMessage(content=user_text)]}
        

        # # 尝试从内存中加载历史记录
        # initial_memory = memory.get(config)
        # if initial_memory:
        #     print("  🧠 发现历史记忆，正在加载...")
        #     # 将历史消息与当前用户输入合并
        #     inputs["messages"] = initial_memory.values['messages'] + inputs["messages"]
        # else:
        #     print("  ✨ 未发现历史记忆，开始新的对话。")
        # 启动图的流式输出
        for output in app.stream(inputs, config=config):
            # 打印状态流转，方便你理解底层逻辑
            for key, value in output.items():
                # print(f"  [{key}] -> {value}")
                pass # 具体的打印逻辑我写在 node 函数里了，这里保持简洁
        
        print("\n=== 最终结果 ===")
        try:
            final_state = app.get_state(config={"configurable": {"thread_id": "1"}})
            print(final_state.values["messages"][-1].content)
            
            # 在每次对话循环结束后，保存长期记忆
            save_long_term_memory(final_state.values, user_text)
        except Exception as e:
            print(f"获取最终结果时出错: {e}")