import os
import sys

# 直接导入RAGFlowPdfParser
try:
    from parser.pdf_parser import RAGFlowPdfParser
except ImportError:
    try:
        from .parser.pdf_parser import RAGFlowPdfParser
    except ImportError:
        try:
            from deepdoc.parser.pdf_parser import RAGFlowPdfParser
        except ImportError:
            print("❌ 无法导入RAGFlowPdfParser，请检查模块路径")
            print("📁 当前工作目录:", os.getcwd())
            print("📁 尝试添加当前目录到Python路径...")
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from parser.pdf_parser import RAGFlowPdfParser

PdfParser = RAGFlowPdfParser  # 别名，保持兼容性
from timeit import default_timer as timer
import json
import time
from datetime import datetime

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("⚠️  tqdm库未安装，将使用简化进度显示")
    print("💡 安装命令: pip install tqdm")

def log_step(step_name, start_time=None, extra_info=""):
    """统一的步骤日志函数"""
    current_time = datetime.now().strftime("%H:%M:%S")
    if start_time:
        elapsed = timer() - start_time
        print(f"⏰ [{current_time}] 📍 {step_name} (耗时: {elapsed:.2f}s) {extra_info}")
        return elapsed  # 返回耗时，而不是当前时间戳
    else:
        print(f"⏰ [{current_time}] 📍 {step_name} {extra_info}")
        return timer()  # 如果没有开始时间，返回当前时间戳作为起点
class Pdf(PdfParser):
    def __call__(self, filename, binary=None, from_page=0,
                 to_page=1, zoomin=3, callback=None):
        print(f"\n🚀 ===== PDF处理开始 =====")
        print(f"📄 文件名: {filename}")
        print(f"📄 页码范围: {from_page} ~ {to_page}")
        print(f"🔍 缩放因子: {zoomin}x")
        
        start = timer()
        first_start = start
        
        # 步骤1: OCR识别
        print(f"\n📖 步骤1: OCR文字识别")
        callback(msg="OCR started")
        log_step("开始OCR处理", first_start)
        
        self.__images__(
            filename if not binary else binary,
            zoomin,
            from_page,
            to_page,
            callback
        )
        ocr_time = log_step("OCR处理完成", start)
        callback(msg=f"OCR finished ({ocr_time:.2f}s)")
        print(f"✅ OCR完成: 处理了 {to_page - from_page + 1} 页，耗时 {ocr_time:.2f}秒")

        # 步骤2: 布局分析
        print(f"\n📐 步骤2: 布局分析")
        start = timer()
        self._layouts_rec(zoomin)
        layout_time = log_step("布局分析完成", start)
        callback(0.63, f"Layout analysis ({layout_time:.2f}s)")
        print(f"✅ 布局分析完成: 耗时 {layout_time:.2f}秒")

        # 步骤3: 表格分析
        print(f"\n📊 步骤3: 表格结构识别")
        start = timer()
        self._table_transformer_job(zoomin)
        table_time = log_step("表格分析完成", start)
        callback(0.65, f"Table analysis ({table_time:.2f}s)")
        print(f"✅ 表格识别完成: 耗时 {table_time:.2f}秒")

        # 步骤4: 文本合并
        print(f"\n📝 步骤4: 文本合并")
        start = timer()
        self._text_merge()
        merge_time = log_step("文本合并完成", start)
        callback(0.67, f"Text merged ({merge_time:.2f}s)")
        print(f"✅ 文本合并完成: 耗时 {merge_time:.2f}秒")

        # 步骤5: 提取表格和图形
        print(f"\n🖼️ 步骤5: 提取表格和图形")
        start = timer()
        tbls = self._extract_table_figure(True, zoomin, True, True)
        extract_time = log_step("表格图形提取完成", start)
        print(f"✅ 提取完成: 找到 {len(tbls)} 个表格/图形，耗时 {extract_time:.2f}秒")

        # 步骤6: 文本合并优化
        print(f"\n🔗 步骤6: 文本合并优化")
        start = timer()
        # self._naive_vertical_merge()
        self._concat_downward()
        # self._filter_forpages()
        optimize_time = log_step("文本优化完成", start)
        print(f"✅ 文本优化完成: 耗时 {optimize_time:.2f}秒")

        # 最终结果统计
        total_time = timer() - first_start
        print(f"\n📈 ===== PDF处理完成 =====")
        print(f"📊 总耗时: {total_time:.2f}秒")
        print(f"📄 提取段落数: {len(self.boxes)}")
        print(f"📊 提取表格数: {len(tbls)}")
        print(f"✅ 处理状态: 成功完成\n")

        return [(b["text"], self._line_tag(b, zoomin))
                for b in self.boxes], tbls


def chunk(filename, binary=None, from_page=0, to_page=100000,
          lang="Chinese", callback=print, **kwargs):
    """print
        Supported file formats are docx, pdf, excel, txt.
        This method apply the naive ways to chunk files.
        Successive text will be sliced into pieces using 'delimiter'.
        Next, these successive pieces are merge into chunks whose token number is no more than 'Max token number'.
    """
    print(f"\n🔧 ===== chunk函数开始执行 =====")
    print(f"📁 输入文件: {filename}")
    print(f"📄 页码范围: {from_page} ~ {to_page}")
    print(f"🌐 语言: {lang}")
    
    start_time = timer()
    
    # 检查文件是否存在
    if not os.path.exists(filename):
        print(f"❌ 错误: 文件不存在 - {filename}")
        return [], []
    
    print(f"✅ 文件存在，开始解析...")
    
    # 创建PDF解析器
    print(f"🛠️ 创建PDF解析器实例...")
    pdf_parser = Pdf()
    
    # 执行PDF解析
    print(f"🚀 开始PDF解析流程...")
    
    # 添加进度条显示
    def progress_callback(msg=None, prog=None):
        if msg:
            print(f"⏳ {msg}")
        if prog is not None:
            # 更新进度条显示
            pass
        if callback and msg:
            callback(msg)
    
    # 模拟进度条（因为实际处理时间可能很长）
    print(f"🔄 正在处理PDF文件...")
    
    if HAS_TQDM:
        # 创建进度条
        with tqdm(total=100, desc="PDF处理进度", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]") as pbar:
            # 模拟不同阶段的进度
            stages = [
                ("正在加载PDF文件...", 10),
                ("正在进行OCR识别...", 25),
                ("正在进行布局分析...", 20),
                ("正在识别表格结构...", 20),
                ("正在合并文本内容...", 15),
                ("正在提取最终结果...", 10)
            ]
            
            for stage_name, progress in stages:
                pbar.set_description(stage_name)
                time.sleep(0.3)  # 给用户一点时间来看到进度
                pbar.update(progress)
    else:
        # 简化进度显示
        stages = ["加载PDF", "OCR识别", "布局分析", "表格识别", "文本合并", "结果提取"]
        for i, stage in enumerate(stages):
            print(f"⏳ {stage}... [{i+1}/{len(stages)}]")
            time.sleep(0.3)
    
    sections, tables = pdf_parser(filename if not binary else binary, from_page=from_page, to_page=to_page,
                                    callback=progress_callback)
    
    # 结果统计
    total_time = timer() - start_time
    print(f"\n📊 ===== chunk函数执行完成 =====")
    print(f"📈 总耗时: {total_time:.2f}秒")
    print(f"📄 提取段落数: {len(sections)}")
    print(f"📊 提取表格数: {len(tables)}")
    print(f"📋 第一段预览: {sections[0][0][:100] if sections else '无内容'}...")
    
    # res = tokenize_table(tables, doc, is_english)
    print(f"\n📋 原始sections数据:")
    print(f"sections类型: {type(sections)}")
    print(f"sections长度: {len(sections)}")
    if sections:
        print(f"第一个元素类型: {type(sections[0])}")
        print(f"第一个元素内容: {sections[0]}")
    
    return sections, tables
# 配置日志级别
VERBOSE_MODE = True  # 设置为True显示详细信息

def debug_print(message, level="INFO"):
    """调试打印函数"""
    if VERBOSE_MODE:
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        level_symbols = {
            "INFO": "ℹ️",
            "SUCCESS": "✅", 
            "WARNING": "⚠️",
            "ERROR": "❌",
            "DEBUG": "🔍"
        }
        symbol = level_symbols.get(level, "📋")
        print(f"[{timestamp}] {symbol} {message}")

#  测试代码
if __name__ == "__main__":
    debug_print("测试程序启动", "INFO")
    debug_print(f"启动时间: {datetime.now().strftime('%H:%M:%S')}", "INFO")
    
    # 获取当前脚本的绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # 使用绝对路径
    file_path = os.path.join(project_root, "test", "user_document", "24S151167.pdf")
    lang="english"
    output_file = os.path.join(project_root, "test", "result.json")

    debug_print(f"目标文件: {file_path}", "INFO")
    debug_print(f"输出文件: {output_file}", "INFO")
    
    # 检查输出目录是否存在
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        debug_print(f"创建输出目录: {output_dir}", "INFO")
        os.makedirs(output_dir, exist_ok=True)
    
    # 如果本地文件不存在，则解析文件并保存结果
    if not os.path.exists(output_file):
        debug_print("输出文件不存在，开始解析PDF", "INFO")
        debug_print(f"开始处理文件: {file_path}", "INFO")
        start_time = timer()
        
        try:
            sections, tables = chunk(file_path, lang=lang)
            process_time = timer() - start_time
            debug_print("处理完成", "SUCCESS")
            debug_print(f"总耗时: {process_time:.2f}秒", "INFO")
            debug_print(f"文档段落数: {len(sections)}", "INFO")
            
            if sections:
                debug_print("前3个段落预览:", "INFO")
                for i, section in enumerate(sections[:3]):
                    if isinstance(section, tuple) and len(section) == 2:
                        text, tag = section
                        debug_print(f"段落 {i+1}: {text[:100]}... (标签: {tag})", "INFO")
                    else:
                        debug_print(f"段落 {i+1}: {str(section)[:100]}...", "INFO")
            
            # 保存结果
            debug_print(f"保存结果到: {output_file}", "INFO")
            
            # 只保存可序列化的数据
            serializable_sections = []
            for section in sections:
                if isinstance(section, tuple):
                    text, tag = section
                    serializable_sections.append({
                        "text": str(text),
                        "tag": str(tag),
                        "length": len(text) if hasattr(text, '__len__') else 0
                    })
                else:
                    serializable_sections.append({
                        "content": str(section),
                        "length": len(section) if hasattr(section, '__len__') else 0
                    })
            
            # 只保存表格的基本信息，不包含图像对象
            serializable_tables = []
            for i, table in enumerate(tables[:5]):
                if isinstance(table, dict):
                    # 如果是字典，只保存可序列化的键值对
                    clean_table = {}
                    for k, v in table.items():
                        if not str(type(v)).__contains__('Image'):
                            try:
                                # 尝试序列化
                                json.dumps(v)
                                clean_table[k] = v
                            except (TypeError, ValueError):
                                clean_table[k] = str(v)
                    serializable_tables.append(clean_table)
                else:
                    serializable_tables.append({
                        "index": i,
                        "type": str(type(table)),
                        "description": "表格对象"
                    })
            
            result = {
                "file_path": file_path,
                "process_time": process_time,
                "paragraphs": len(sections),
                "tables": len(tables),
                "sections": serializable_sections,
                "tables_info": serializable_tables
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            debug_print("结果保存完成", "SUCCESS")
            
        except Exception as e:
            debug_print(f"处理过程中出现错误: {e}", "ERROR")
            import traceback
            traceback.print_exc()
            
    else:
        debug_print(f"输出文件已存在: {output_file}", "INFO")
        debug_print("删除该文件可以重新处理", "INFO")
        
        # 读取并显示已有结果
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                result = json.load(f)
            debug_print(f"文件包含 {result.get('paragraphs', 0)} 个段落", "INFO")
        except Exception as e:
            debug_print(f"读取结果文件失败: {e}", "ERROR")
    
    debug_print("测试程序结束", "SUCCESS")
    debug_print(f"结束时间: {datetime.now().strftime('%H:%M:%S')}", "INFO")