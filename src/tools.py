import os
from typing import Any, Dict, List

from langchain_core.tools import tool
import subprocess
import chardet
from langchain_openai import ChatOpenAI
from src.myMilvus import LONG_TERM_MEM, db
from deepdoc.pdf_parse import LOCAL_PDF_FILE, process_pdf_document
from src.memoryBlock import MemoryBlock
from dotenv import load_dotenv
load_dotenv()
llm = ChatOpenAI(
    model="qwen-plus-2025-07-28", 
    api_key=os.getenv("OPENAI_API_KEY"), # 从环境变量获取API密钥
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    max_tokens=2048,  # 设置最大token长度为2048
    temperature=0.1,  # 降低随机性，使标题生成更稳定
)
@tool
def import_pdf(filename: str) -> str:
    """需要导入PDF文件调用这个函数"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    file_path = os.path.join(project_root, "user_document", filename)
    lang = "english"
    output_file = os.path.join(project_root, "user_document", "result.json")
    
    try:
        # 处理PDF文档
        result = process_pdf_document(input_file=file_path, output_file=output_file, lang=lang)
        
        # 检查处理结果
        if not isinstance(result, dict):
            return f"处理结果格式错误: 期望dict，得到{type(result)}"
            
        if result.get("status") != "success":
            error_msg = result.get("error", "未知错误")
            return f"PDF处理失败: {error_msg}"
        
        # 处理sections
        sections: List[Dict[str, Any]] = result.get("sections", [])
        if not isinstance(sections, list):
            return f"sections格式错误: 期望list，得到{type(sections)}"
            
        processed_count = 0
        sections=sections[0:20]# 只处理前20个段落
        # 使用tqdm显示进度
        try:
            from tqdm import tqdm
            section_iterator = tqdm(enumerate(sections), total=len(sections), desc="处理段落")
        except ImportError:
            section_iterator = enumerate(sections)
            print(f"开始处理 {len(sections)} 个段落...")
        
        for i, section in section_iterator:
        # for i, section in enumerate(sections[0:20]):# 只处理前20个段落
            if not isinstance(section, dict):
                continue
                
            page = section.get("page")
            content = section.get("content")
            
            if page is None or content is None:
                continue
                
            if not isinstance(page, int) or not isinstance(content, str):
                continue
                
            try:
                # 创建内存块
                from src.memoryBlock import MemoryBlock
                
                # 生成标题
                from langchain_core.messages import HumanMessage, SystemMessage
                import time
                
                title = f"页面{page}"  # 默认标题
                try:
                    # 严格限制内容长度，避免API调用失败
                    max_content_length = 150  # 进一步减少到150字符
                    truncated_content = content[:max_content_length] if len(content) > max_content_length else content
                    
                    # 使用极简的prompt模板
                    system_msg = "生成标题"
                    user_msg = f"标题(≤10字)：{truncated_content}"
                    
                    # 确保总长度不超过300字符（严格限制）
                    if len(user_msg) > 300:
                        available_length = 280  # 给模板留出空间
                        truncated_content = truncated_content[:available_length]
                        user_msg = f"标题(≤10字)：{truncated_content}"
                    
                    # 添加重试机制
                    max_retries = 3
                    for retry in range(max_retries):
                        try:
                            response = llm.invoke([
                                SystemMessage(content=system_msg),
                                HumanMessage(content=user_msg)
                            ])
                            if hasattr(response, 'content') and response.content:
                                title = str(response.content).strip()
                                if len(title) > 10:
                                    title = title[:10]  # 确保标题长度不超过10个字符
                                break
                        except Exception as api_error:
                            if "HTTP" in str(api_error) and retry < max_retries - 1:
                                wait_time = (retry + 1) * 2  # 指数退避
                                print(f"  ⚠️  API调用失败 (重试{retry+1}/{max_retries})，等待{wait_time}秒后重试: {api_error}")
                                time.sleep(wait_time)
                            else:
                                raise api_error
                
                except Exception as title_error:
                    print(f"  ⚠️  生成标题失败，使用默认标题: {title_error}")
                    title = f"页面{page}"
                
                # 创建metadata字典
                metadata = {}
                if isinstance(title, str) and title.strip():
                    metadata["title"] = title.strip()
                else:
                    metadata["title"] = f"页面{page}"
                
                # 添加必需的source字段
                metadata["source"] = f"PDF页面{page}"
                
                # 初始化MemoryBlock，提供所有必需字段
                # 截断内容以适应Milvus数据库的512字符限制
                max_text_length = 500  # 留出空间给页面号前缀
                page_prefix = f"页面号{page} "
                available_content_length = max_text_length - len(page_prefix)
                
                if len(content) > available_content_length:
                    truncated_content = content[:available_content_length]
                else:
                    truncated_content = content
                
                memory = MemoryBlock(
                    text=f"{page_prefix}{truncated_content}",
                    metadata=metadata
                )
                
                # 插入数据库
                db.insert_block(memory, collection_name=LOCAL_PDF_FILE)
                processed_count += 1
                
                # 更新进度条描述
                if hasattr(section_iterator, 'set_description'):
                    section_iterator.set_description(f"处理段落 (已处理{processed_count}个)")
                
            except Exception as e:
                error_msg = f"处理第{i}个段落时出错: {e}"
                print(error_msg)
                if hasattr(section_iterator, 'write'):
                    section_iterator.write(error_msg)
                continue
        
        # 处理表格和图片
        tables_or_images: List[Dict[str, Any]] = result.get("tables_or_images", [])
        if isinstance(tables_or_images, list):
            for table in tables_or_images:
                if isinstance(table, dict):
                    # 这里可以添加表格/图片处理逻辑
                    pass
        
        return f"成功导入文件 {filename}，处理了{processed_count}个段落"
        
    except FileNotFoundError:
        return f"文件不存在: {filename}"
    except Exception as e:
        return f"导入文件 {filename} 时发生错误: {type(e).__name__}: {e}"

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

@tool(description="提供搜索语句，进行milvus数据库混合检索历史记忆，需要提供询问以及哪个数据库str:enum{'LONG_TERM_MEM','LOCAL_PDF_FILE'}")
def long_mem_retrive(query:str,collection_name:str):
    if collection_name=="LONG_TERM_MEM":
        results = db.hybrid_search(query,top_k=3,collection_name=LONG_TERM_MEM,sparse_alpha=0.2,dense_alpha=0.8)
    elif collection_name=="LOCAL_PDF_FILE":
        results = db.hybrid_search(query,top_k=3,collection_name=LOCAL_PDF_FILE,sparse_alpha=0.2,dense_alpha=0.8)
    else:
        raise ValueError(f"未知的数据库名称: {collection_name}")
    ret=""
    for result in results:
        ret+="1."+result.entity.get('text')+"\n\n"       
    return ret

# 工具定义
basic_tools = [get_text, write_text, execute_command, long_mem_retrive, import_pdf]