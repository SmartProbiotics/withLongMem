from typing import Annotated, TypedDict, List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
import operator
from pydantic import BaseModel, Field
from datetime import datetime
import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from src.memoryBlock import MemoryBlock
from src.myMilvus import LONG_TERM_MEM, db
from datetime import datetime as dt

# 加载环境变量
load_dotenv()

# LLM 初始化（供记忆模块使用）
llm = ChatOpenAI(
    model="qwen-plus-2025-07-28", 
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

def my_reducer(left: list, right: list) -> list:
    """
    逻辑：检查新消息列表（right）的第一条消息
    是否携带了 'reset=True' 的元数据标签。
    """
    # 检查 metadata (additional_kwargs) 里的逻辑标记
    if right and right[0].additional_kwargs.get("reset") is True:
        print("--- 🔄 检测到重置信号，正在替换全局记忆 ---")
        return right 
        
    return (left or []) + right

class AgentState(TypedDict):
    messages: Annotated[list, my_reducer]

class feed_back(BaseModel):
    success: bool = Field(..., description="工具执行是否成功")
    content: str=Field("", description="如果执行失败，这里是错误信息；如果成功，可以留空")

def memory_compression_node(state: dict):
    if len(state["messages"])< 10:
        print("\n[节点执行] 🧠 Memory Compressor 不需要压缩。")
        return {"messages": []}
    print("\n[节点执行] 🧠 Memory Compressor 正在进行扁平化整理...")
    
    # 1. 构造一个让 LLM 总结的提示词
    history_summary_prompt = (
        "请将以下对话历史总结为一段简短的【任务快照】。"
        "必须包含：1. 已经检查过的文件；2. 运行过的命令及其结果（特别是报错）；3. 当前待解决的问题。"
        "不需要保留对话格式，只需提供核心事实。"
    )
    
    try:
        # 这里不需要 with_structured_output，直接让它回一段话即可
        summary_response = llm.invoke([
            SystemMessage(content=history_summary_prompt),
            HumanMessage(content=str(state["messages"]))
        ])
        
        # 2. 构造一条全新的、干净的消息
        clean_snapshot = SystemMessage(
            content=f"【项目历史档案】：\n{summary_response.content}",
            additional_kwargs={"reset": True} # 配合之前的 Reducer 清空旧记忆
        )
        
        # 3. 加上用户最后那条指令，确保 Agent 知道现在要干嘛
        last_user_msg = [m for m in state["messages"] if isinstance(m, HumanMessage)][-1]

        print("  ✅ 记忆已扁平化，移除了所有可能导致报错的 ToolMessage 结构。")
        print(f"  🧾 新的记忆内容:\n{clean_snapshot.content}\n  🧾 最后用户指令:\n{last_user_msg.content}")
        return {"messages": [clean_snapshot, last_user_msg]}

    except Exception as e:
        # 备选方案：如果压缩崩了，直接截断并强行转换类型
        print(f"  ❌ 压缩失败，执行安全截断...")
        fallback_msg = HumanMessage(
            content="[系统自动截断] 之前的对话太长已清理，请继续当前任务。",
            additional_kwargs={"reset": True}
        )
        return {"messages": [fallback_msg]}

def save_long_term_memory(state: dict, user_input: str,collection_name:str=LONG_TERM_MEM) -> None:
    """将精炼的有价值的记忆文本写入memory.txt"""
    try:
        # 从状态中获取最后的AI响应
        all_messages = state["messages"]
        last_ai_response = ""
        for msg in reversed(all_messages):
            if hasattr(msg, 'content'):
                last_ai_response = msg.content
                break
        
        # 定义专用的记忆列表模型
        class MemoryBlockList(BaseModel):
            memories: List[MemoryBlock]
        
        # 让LLM生成结构化记忆摘要
        memory_summary_prompt = f"""请将以下对话历史分割为多个的长期记忆条目，每个条目包含标题和内容：
        
        用户输入：{user_input}
        AI回应：{last_ai_response}
        
        要求：
        1. 每个条目字数控制在300字左右，标题不超过50字
        2. 包含有价值的信息和对应条目的标题
        3. 只能陈述记忆核心现象和事实，过滤掉思考过程
        """
        
        # 使用正确的Pydantic模型
        structured_llm = llm.with_structured_output(MemoryBlockList)
        result = structured_llm.invoke([HumanMessage(content=memory_summary_prompt)])
        #追加到milvus
        
        current_time = dt.now().strftime("%Y-%m-%d %H:%M:%S")
        for block in result.memories:
            block.metadata['source'] =collection_name+current_time
            db.insert_block(block,collection_name=LONG_TERM_MEM)

        # 读取现有记忆并追加新的记忆
        memory_file_path = "./memory.txt"
        existing_memory = ""
        try:
            with open(memory_file_path, 'r', encoding='utf-8') as f:
                existing_memory = f.read()
        except FileNotFoundError:
            pass
            
        # 格式化记忆条目
        new_memory_entry = ""
        for block in result.memories:
            new_memory_entry += f"\n\n=== Title: {block.metadata['title']} ===\n{block.text}\n=== END ===\n"
        final_memory = existing_memory + new_memory_entry

        # 写入memory.txt文件
        with open(memory_file_path, 'w', encoding='utf-8') as f:
            f.write(final_memory)
        

        print(f"  💾 长期记忆已保存到 {memory_file_path}")

    except Exception as e:
        print(f"  ❌ 长期记忆保存失败: {str(e)}")

def critic_node(state: dict):
    """反思节点：专门检查工具执行的结果"""
    print("\n[节点执行] 🧐 Critic 正在审查执行结果...")
    last_message = state["messages"][-1]

    # 如果最后一条消息不是 ToolMessage，则无需审查
    if not isinstance(last_message, ToolMessage):
        print("  🤔 非工具执行结果，跳过审查。")
        return {"messages": []}

    # 为审查员创建一个提示
    critic_prompt_message = HumanMessage(
        content=f"你是一个工具执行结果的审查员。请根据以下工具执行的输出，判断操作是否成功。如果成功，调用 feed_back 工具并设置 success=True。如果失败，设置 success=False 并将错误信息填入 content 字段。如果输出中包含 '未找到'、'error'、'失败'、'超时' 等关键词，通常意味着失败。\n\n工具输出:\n```\n{last_message.content}\n```"
    )

    # 绑定 Pydantic 模型作为工具，并强制调用它
    critic_llm = llm.bind_tools([feed_back], tool_choice="feed_back")

    # 调用审查员 LLM
    response = critic_llm.invoke([critic_prompt_message])

    # LLM 的响应应该是一条包含工具调用的 AI 消息
    if not response.tool_calls:
        # 作为备用方案，如果 LLM 没有按预期调用工具，我们默认操作是成功的
        print("  🤖 Critic 未能调用 feedback 工具，默认操作成功。")
        return {"messages": []}

    # 解析工具调用的参数
    tool_call = response.tool_calls[0]
    feedback_args = tool_call['args']
    
    if feedback_args.get("success", False):
        print("  ✅ 执行正常，无需额外反思。")
        return {"messages": []} 
    else:
        error_content = feedback_args.get("content", "未知错误")
        print(f"  ⚠️ 发现错误！注入反思建议: {error_content}")
        return {"messages": [HumanMessage(content=f"【系统反思提示】：刚才的工具执行报错了。错误信息是：'{error_content}'。请不要重复相同的操作！请仔细分析错误原因，并尝试换一种方式（例如检查文件名是否正确）。")]}