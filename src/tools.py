from langchain_core.tools import tool
import subprocess
import chardet
from src.myMilvus import LONG_TERM_MEM, db

@tool
def get_text(filename: str) -> str:
    """需要读取文件内容调用这个函数"""
    encodings = ['utf-8', 'utf-8-sig', 'gbk', 'cp936', 'cp1252', 'latin-1', 'utf-16']
    try:
        with open(filename, 'rb') as f:
            raw = f.read()
        guess = chardet.detect(raw)
        charset = guess.get('encoding')
        if charset:
            try:
                return raw.decode(charset)
            except Exception:
                pass
    except FileNotFoundError:
        return f"文件 {filename} 未找到。"
    except Exception as e:
        return f"读取文件时发生未知错误: {e}"

    for enc in encodings:
        try:
            with open(filename, 'r', encoding=enc) as f:
                return f.read()
        except Exception:
            continue

    return f"无法用常见编码解码 {filename}"

@tool
def write_text(filename: str, content: str) -> str:
    """需要写入文件调用这个函数,执行完成后会写入并保存"""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"文件 {filename} 已成功写入。"
    except Exception as e:
        return f"写入文件时发生错误: {e}"

@tool(description="执行指令，当前为Linux环境，我已授权你执行任何操作，包括pip等可以直接执行")
def execute_command(command: str) -> str:
    """执行终端指令，如ls ./*"""
    print(f"准备执行命令: {command}")
    confirmation = input("输入 'yes' 确认执行命令: ")
    if confirmation.lower() == 'yes':
        try:
            result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=30)
            if result.stderr:
                return f"stderr: {result.stderr}"
            return f"stdout: {result.stdout}"
        except subprocess.TimeoutExpired:
            return "命令执行超时。"
        except Exception as e:
            return f"执行命令时发生错误: {e}"
    else:
        return "命令执行已取消。"
    

# 添加重启工具函数
@tool(description="重启代理程序，执行后你必须迅速退出当前程序，程序将在20秒后重启。")
def restart_agent():
    """重启代理程序，在20秒后重新启动主程序"""
    import subprocess
    import sys
    try:
        # 使用subprocess启动批处理文件，不阻塞当前进程
        subprocess.Popen(['start', 'cmd', '/c', 'restart_agent.bat'], shell=True)
        return "重启指令已发送，程序将在20秒后重启。当前进程即将关闭。"
    except Exception as e:
        return f"重启失败: {str(e)}"

@tool(description="提供搜索语句，进行milvus数据库混合检索历史记忆")
def long_mem_retrive(query:str):
    results = db.hybrid_search(query,top_k=3,collection_name=LONG_TERM_MEM,sparse_alpha=1.0,dense_alpha=0.2)
    ret=""
    for result in results:
        ret+="1."+result.entity.get('text')+"\n\n"       
    return ret

# 工具定义
basic_tools = [get_text, write_text, execute_command, restart_agent, long_mem_retrive]