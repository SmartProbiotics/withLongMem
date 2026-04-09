from typing import Annotated, TypedDict, Sequence
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from src.tools import basic_tools
from src.memory_module import my_reducer, AgentState, critic_node, memory_compression_node
from src.dynamic_tool import generate_tool
from src.myMilvus import LONG_TERM_MEM, db
# 加载环境变量
load_dotenv()
workflow = StateGraph(AgentState)

all_tools=[]
# LLM 初始化
llm = ChatOpenAI(
    model="qwen-plus-2025-07-28", 
    api_key=os.getenv("OPENAI_API_KEY"), # 从环境变量获取API密钥
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
llm_with_tools = llm.bind_tools(basic_tools)



def agent_node(state: AgentState):
    """大脑节点：负责思考和决定下一步"""
    print("\n[节点执行] 👉 Agent 正在思考...")
    def extract_query_from_messages(messages):
        """从消息列表中提取用户查询"""
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage) and hasattr(msg, 'content'):
                return msg.content
        return ""  # 默认返回空字符串
    # 提取查询文本而不是直接传递消息列表
    query_text = extract_query_from_messages(state["messages"])
    if query_text:
        retrive_mem = db.hybrid_search(query_text, top_k=3, collection_name=LONG_TERM_MEM, dense_alpha=1.0, sparse_alpha=0.2)
        memory_blocks = db.result_to_blocks(retrive_mem)
        dynamic_tools = generate_tool(memory_blocks)
        print("生成动态工具：")
        for tool in dynamic_tools:
            print(f"  🛠️ {tool.name}: {tool.description}")
        import sys
        mod = sys.modules[__name__]
        mod.all_tools = basic_tools + dynamic_tools
    else:
        # 如果没有查询文本，只使用基础工具
        import sys
        mod = sys.modules[__name__]
        mod.all_tools = basic_tools
    
    # 1. 重建 LLM 绑定
    llm_with_tools = llm.bind_tools(all_tools)
    # 2. 把最新工具列表暂存到 state，供 build_graph 里统一刷新
    state["__dynamic_tools__"] = all_tools
    
    response = llm_with_tools.invoke(state["messages"])
    print(f"LLM 收到的消息:", state["messages"])
    print(f"  🤖 Agent 思考结果: {response.content}")
    return {"messages": [response]}

def tool_node(state: AgentState):
    """工具节点：负责调用工具"""
    print("\n[节点执行] 👉 工具节点 正在调用工具...")
    from langchain_core.messages import ToolMessage

    tool_calls = state["messages"][-1].tool_calls
    results = []
    print("[调试] 本次 all_tools 列表：", [t.name for t in all_tools])
    print("[调试] 本次 tool_calls：", [tc["name"] for tc in tool_calls])

    for tool_call in tool_calls:
        # 1. 从当前 state 里拿最新工具列表（含动态生成的）
        # all_tools = state.get("__dynamic_tools__", all_tools)
        for tool in all_tools:
            if tool.name == tool_call["name"]:
                # 2. 调用并包装成 ToolMessage
                print(f"调用工具: {tool.name}")
                args = tool_call.get("arguments") or tool_call.get("args") or {}
                output = tool.invoke(args)
                if not isinstance(output, str):
                    output = str(output)
                tool_call_id = tool_call.get("id") or tool_call.get("call_id") or ""
                results.append(ToolMessage(content=output, tool_call_id=tool_call_id))
                break
    print(f"  🤖 工具节点 调用工具结果: {results}")
    return {"messages": results}


def should_continue(state: AgentState):
    """判断 Agent 思考完之后，是去调工具，还是结束对话"""
    last_message = state["messages"][-1]
    # 如果 LLM 决定调用工具
    if last_message.tool_calls:
        return "continue"      # 保持和边字典键一致
    # 否则直接结束
    return "end"

# 图构建函数
def build_graph():
    # 添加节点
    workflow.add_node("agent", agent_node)
    # 先放一个默认 ToolNode，保证图编译通过；后续由 agent_node 动态覆盖
    # 用 lambda 让节点每次从 state 里拿最新 ToolNode
    workflow.add_node("tools", tool_node)
    workflow.add_node("critic", critic_node) # 我们的反思节点
    workflow.add_node("memory_compressor", memory_compression_node) # 我们的记忆压缩节点

    # 设置起点
    workflow.set_entry_point("agent")

    # 添加条件边：Agent 之后，根据 should_continue 的返回值决定去哪
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "continue": "tools",   # 键和返回值保持一致
            "end": END
        }
    )

    # 工具执行完后，必须去 Critic 接受审查
    workflow.add_edge("tools", "critic")
    # Critic 审查完后，先去压缩记忆，再回到 Agent 重新规划
    workflow.add_edge("critic", "memory_compressor")
    workflow.add_edge("memory_compressor", "agent")

    # 编译成可执行的应用
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    return app, memory