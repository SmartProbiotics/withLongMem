import os
import sys
import re
from deepdoc.parser.pdf_parser import RAGFlowPdfParser
LOCAL_PDF_FILE="LocalPDF"
PdfParser = RAGFlowPdfParser  # 别名，保持兼容性
from timeit import default_timer as timer
import json
import time
from datetime import datetime
from tqdm import tqdm
from http import HTTPStatus
from dashscope import Application, MultiModalConversation
import tempfile
import io
from dotenv import load_dotenv
load_dotenv()


# 环境变量配置说明:
# 要使用千问模型识别表格图片，需要配置以下环境变量:
# DASHSCOPE_API_KEY: 千问API密钥
# QWEN_APP_ID 或 YOUR_APP_ID: 千问应用ID


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
    def __init__(self):
        super().__init__()
        self.zoomin = 3
    def __call__(self, filename, binary=None, from_page=0,
                 to_page=1, zoomin=3, callback=None):
        print(f"\n🚀 ===== PDF处理开始 =====")
        print(f"📄 文件名: {filename}")
        print(f"📄 页码范围: {from_page} ~ {to_page}")
        print(f"🔍 缩放因子: {zoomin}x")
        self.zoomin = zoomin
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

        #步骤7：超短文本二次检测 - 改进版：连续块检测和OCR重试
        print(f"\n🔍 步骤7: 超短文本二次检测(含连续块分析和OCR重试)")
        start_time = timer()
        
        # 第一步：识别连续短文本块
        short_text_blocks=self._get_short_text_blocks()

        # 保存可疑连续块图片
        if short_text_blocks:
            merged_boxes = self._rebuild_boxes(short_text_blocks)
            # if merged_boxes:
            #     self._save_figures(merged_boxes)
            
            # 第二步：为每个merged块生成Figure类型，从boxes删除并添加到tbls
            for block in merged_boxes:
                if not block: continue
                # 获取合并后的box（已在_rebuild_boxes中更新到self.boxes）
                merged_box = self.boxes[block[0]]
                page_num = merged_box['page_number']
                x0, y0, x1, y1 = merged_box['x0'], merged_box['top'], merged_box['x1'], merged_box['bottom']
                merged_text = merged_box['text']
                
                # 创建表格数据 - 匹配实际的tbls格式
                # tbls中的每个元素是 (图片, 表格数据) 的元组
                
                # 1. 首先裁剪出表格图片
                page_idx = page_num - 1
                if page_idx < len(self.page_images):
                    page_height = self.page_cum_height[page_idx] if hasattr(self, 'page_cum_height') else 0
                    
                    # 扩展边界
                    margin_x, margin_y = (x1-x0)*0.02, (y1-y0)*0.02  # 小边距
                    left = max(0, int((x0 - margin_x) * self.zoomin))
                    top = max(0, int((y0 - margin_y - page_height) * self.zoomin))
                    right = int((x1 + margin_x) * self.zoomin)
                    bottom = int((y1 + margin_y - page_height) * self.zoomin)
                    
                    # 确保裁剪区域有效
                    img_width, img_height = self.page_images[page_idx].size
                    left = min(left, img_width - 1)
                    top = min(top, img_height - 1)
                    right = min(right, img_width)
                    bottom = min(bottom, img_height)
                    
                    if right > left and bottom > top:
                        table_img = self.page_images[page_idx].crop((left, top, right, bottom))
                        
                        # 2. 创建表格数据（文本描述）
                        table_data = [f"合并的短文本块表格: {merged_text[:100]}..."]
                        
                        # 3. 添加到tbls - 格式为 (图片, 表格数据)
                        tbls.append((table_img, table_data))
                        print(f"  添加表格: 页码{page_num}, 边界({x0:.1f},{y0:.1f},{x1:.1f},{y1:.1f}), 文本长度{len(merged_text)}")
                    else:
                        print(f"  警告: 无效的表格裁剪区域")
                else:
                    print(f"  警告: 页码{page_num}超出范围")
    
        detect_time = timer() - start_time
        print(f"✅ 超短文本二次检测完成，耗时{detect_time:.2f}秒")
        # 最终结果统计
        total_time = timer() - first_start
        print(f"\n📈 ===== PDF处理完成 =====")
        print(f"📊 总耗时: {total_time:.2f}秒")
        print(f"📄 提取段落数: {len(self.boxes)}")
        print(f"📊 提取表格数: {len(tbls)}")
        print(f"✅ 处理状态: 成功完成\n")

        return [(b["text"], self._line_tag(b, zoomin))
                for b in self.boxes], tbls
    def _get_short_text_blocks(self):
        short_text_blocks = []
        current_block = []
        for i, b in enumerate(tqdm(self.boxes, desc="极大连续短文本块检测")):
            if len(b["text"]) < 20:  #文本长度<20
                current_block.append(i)
            else:
                # 遇到长文本，检查当前块是否为极大连续短文本块
                if len(current_block) >5:  # >=5
                    short_text_blocks.append(current_block.copy())
                    print(f"发现极大连续短文本块: 索引{current_block[0]}-{current_block[-1]}, 长度{len(current_block)}, token长度<{10}")
                current_block = []
        # 处理末尾的剩余块
        if len(current_block) >= 5:
            short_text_blocks.append(current_block.copy())
            print(f"发现末尾极大连续短文本块: 索引{current_block[0]}-{current_block[-1]}, 长度{len(current_block)}")
        # short_text_blocks 融合，如果box存在交叉关系，则融合为一个新的box，极小且恰好可以框住这两
        def boxes_overlap(box1, box2):
            return (box1['page_number'] == box2['page_number'] and
                    box1['x0'] < box2['x1'] and box1['x1'] > box2['x0'] and
                    box1['top'] < box2['bottom'] and box1['bottom'] > box2['top'])
        # 合并有交叉关系的块
        merged_blocks = []
        for block in short_text_blocks:
            if not merged_blocks:
                merged_blocks.append(block)
                continue
            # 检查是否与已合并的块有交叉
            should_merge = False
            for merged in merged_blocks:
                # 检查当前块与已合并块中的任意box是否有交叉
                for idx1 in block:
                    for idx2 in merged:
                        if boxes_overlap(self.boxes[idx1], self.boxes[idx2]):
                            # 合并两个块
                            merged.extend(block)
                            should_merge = True
                            break
                    if should_merge:
                        break
                if should_merge:
                    break
            
            if not should_merge:
                merged_blocks.append(block)
        return merged_blocks
    def _save_figures(self, short_text_blocks):
        os.makedirs("suspicious_blocks", exist_ok=True)
        for idx, block in enumerate(short_text_blocks):
            page = min(self.boxes[i]['page_number'] for i in block) - 1
            if page < len(self.page_images):
                # 参照pdf_parser.py的坐标转换逻辑
                page_height = self.page_cum_height[page] if hasattr(self, 'page_cum_height') else 0
                x0 = min(self.boxes[i]['x0'] for i in block)
                y0 = min(self.boxes[i]['top'] for i in block) - page_height
                x1 = max(self.boxes[i]['x1'] for i in block)
                y1 = max(self.boxes[i]['bottom'] for i in block) - page_height
                # 添加边距并确保坐标有效
                margin_x, margin_y = (x1-x0)*0.1, (y1-y0)*0.1
                left = max(0, int((x0 - margin_x) * self.zoomin))
                top = max(0, int((y0 - margin_y) * self.zoomin))
                right = int((x1 + margin_x) * self.zoomin)
                bottom = int((y1 + margin_y) * self.zoomin)
                # 确保裁剪区域在图片范围内
                img_width, img_height = self.page_images[page].size
                left = min(left, img_width - 1)
                top = min(top, img_height - 1)
                right = min(right, img_width)
                bottom = min(bottom, img_height)
                if right > left and bottom > top:
                    img = self.page_images[page].crop((left, top, right, bottom))
                    name = f"block_{idx+1}_p{page+1}_{datetime.now().strftime('%m%d_%H%M%S')}.png"
                    img.save(os.path.join("suspicious_blocks", name))
                    print(f"  保存可疑块: {name} (坐标: {left},{top}-{right},{bottom})")
                else:
                    print(f"  跳过无效裁剪区域: {left},{top}-{right},{bottom}")

    def _rebuild_boxes(self, short_text_blocks):
        """
        根据short_text_blocks重建boxes：
        1. 先把short_text_blocks融合成一个大块
        2. 从box列表中删除原来的小box，替换成融合后的大块
        3. 用XGBoost判断大块旁边的box是否能加入这个大块
        4. 迭代更新box列表
        
        返回：融合后的新box信息列表，用于后续图片保存
        """
        if not short_text_blocks:
            return []
        
        print(f"🔧 开始重建boxes，处理{len(short_text_blocks)}个短文本块...")
        
        # 步骤1: 收集所有短文本块中的box索引
        short_block_indices = set()
        for block in short_text_blocks:
            short_block_indices.update(block)
        
        if not short_block_indices:
            return []
        
        # 步骤2: 创建融合后的大块
        short_boxes = [self.boxes[i] for i in short_block_indices]
        merged_x0 = min(b['x0'] for b in short_boxes)
        merged_y0 = min(b['top'] for b in short_boxes)
        merged_x1 = max(b['x1'] for b in short_boxes)
        merged_y1 = max(b['bottom'] for b in short_boxes)
        merged_text = ' '.join(b['text'] for b in short_boxes)
        page_num = short_boxes[0]['page_number']
        
        # 创建融合后的box
        merged_box = {
            'text': merged_text,
            'x0': merged_x0,
            'y0': merged_y0,
            'x1': merged_x1,
            'y1': merged_y1,
            'top': merged_y0,
            'bottom': merged_y1,
            'page_number': page_num,
            'layout_type': 'text'
        }
        
        print(f"  融合后的大块: 边界({merged_x0:.1f}, {merged_y0:.1f}, {merged_x1:.1f}, {merged_y1:.1f})")
        print(f"  融合文本长度: {len(merged_text)}字符")
        
        # 步骤3: 删除原来的短文本box，添加融合后的大块
        # 按索引降序删除，避免索引变化问题
        for idx in sorted(short_block_indices, reverse=True):
            del self.boxes[idx]
        
        # 添加融合后的大块到box列表
        self.boxes.append(merged_box)
        merged_idx = len(self.boxes) - 1
        
        print(f"  已替换: 删除{len(short_block_indices)}个小box，添加1个融合后的大box")
        print(f"  当前box总数: {len(self.boxes)}")
        
        # 步骤4: 迭代使用XGBoost判断附近的box是否能加入这个大块
        if not hasattr(self, 'updown_cnt_mdl') or self.updown_cnt_mdl is None:
            print(f"⚠️  没有XGBoost模型，跳过迭代融合")
            return
            
        proximity_threshold = 50  # 像素单位的接近阈值
        iteration = 0
        max_iterations = 10  # 最大迭代次数，避免无限循环
        
        while iteration < max_iterations:
            iteration += 1
            print(f"  第{iteration}轮迭代融合...")
            
            # 获取当前大块的边界（可能在之前的迭代中已更新）
            current_merged = self.boxes[merged_idx]
            current_x0, current_y0 = current_merged['x0'], current_merged['top']
            current_x1, current_y1 = current_merged['x1'], current_merged['bottom']
            current_page = current_merged['page_number']
            
            # 扩展搜索范围
            expanded_x0 = current_x0 - proximity_threshold
            expanded_y0 = current_y0 - proximity_threshold
            expanded_x1 = current_x1 + proximity_threshold
            expanded_y1 = current_y1 + proximity_threshold
            
            # 找到可能可以加入的box
            candidates = []
            for i, box in enumerate(self.boxes):
                if i == merged_idx:  # 跳过自己
                    continue
                if box['page_number'] != current_page:  # 只处理同一页的
                    continue
                    
                # 检查是否与扩展后的区域相交
                if (box['x1'] >= expanded_x0 and box['x0'] <= expanded_x1 and
                    box['bottom'] >= expanded_y0 and box['top'] <= expanded_y1):
                    candidates.append(i)
            
            print(f"    找到{len(candidates)}个候选box")
            
            if not candidates:
                print(f"    没有更多候选box，迭代结束")
                break
            
            # 对每个候选box使用XGBoost判断是否融合
            merged_any = False
            for candidate_idx in candidates[:]:
                candidate_box = self.boxes[candidate_idx]
                
                # 提取特征
                fea = self._box_merge_features(current_merged, candidate_box)
                
                # 判断候选box是否是短文本（文本长度小于20）
                is_short_text = len(candidate_box['text']) < 20
                
                print(f"    box{candidate_idx}文本长度: {len(candidate_box['text'])}, 是否短文本: {is_short_text}")
                
                if is_short_text:  # 如果是短文本就直接融合
                    # 扩展当前大块的边界
                    new_x0 = min(current_x0, candidate_box['x0'])
                    new_y0 = min(current_y0, candidate_box['top'])
                    new_x1 = max(current_x1, candidate_box['x1'])
                    new_y1 = max(current_y1, candidate_box['bottom'])
                    new_text = current_merged['text'] + ' ' + candidate_box['text']
                    
                    # 更新融合后的大块
                    self.boxes[merged_idx] = {
                        'text': new_text,
                        'x0': new_x0,
                        'y0': new_y0,
                        'x1': new_x1,
                        'y1': new_y1,
                        'top': new_y0,
                        'bottom': new_y1,
                        'page_number': current_page,
                        'layout_type': 'text'
                    }
                    
                    # 删除被融合的box
                    del self.boxes[candidate_idx]
                    
                    # 如果删除的box在merged_idx之后，需要调整merged_idx
                    if candidate_idx < merged_idx:
                        merged_idx -= 1
                    
                    merged_any = True
                    print(f"    ✅ 已融合box{candidate_idx}，新边界({new_x0:.1f}, {new_y0:.1f}, {new_x1:.1f}, {new_y1:.1f})")
                    break  # 每次只融合一个，然后重新开始
            
            if not merged_any:
                print(f"    本轮没有融合任何box，迭代结束")
                break
        
        print(f"✅ boxes重建完成，最终box总数: {len(self.boxes)}")
        return [[merged_idx]] 

    def _box_merge_features(self, box1, box2):
        """提取两个box融合判断的特征"""
        # 垂直距离
        y_dis = abs(box1['top'] - box2['bottom'])
        
        # 水平对齐度
        x_align = abs(box1['x0'] - box2['x0']) + abs(box1['x1'] - box2['x1'])
        
        # 高度相似度
        h1, h2 = box1['bottom'] - box1['top'], box2['bottom'] - box2['top']
        h_sim = min(h1, h2) / max(h1, h2) if max(h1, h2) > 0 else 1.0
        
        # 宽度相似度
        w1, w2 = box1['x1'] - box1['x0'], box2['x1'] - box2['x0']
        w_sim = min(w1, w2) / max(w1, w2) if max(w1, w2) > 0 else 1.0
        
        # 文本长度比
        len_ratio = len(box1['text']) / max(len(box2['text']), 1)
        
        # 文本特征
        box1_end_with_punct = bool(re.search(r'[。！？.!?]$', box1['text']))
        box2_start_with_punct = bool(re.search(r'^[。！？.!?，,]', box2['text']))
        
        # 布局类型一致性
        layout_same = box1.get('layout_type', 'text') == box2.get('layout_type', 'text')
        
        return [
            y_dis / 50.0,  # 归一化垂直距离
            x_align / 100.0,  # 归一化水平对齐度
            h_sim,
            w_sim,
            len_ratio,
            1.0 if box1_end_with_punct else 0.0,
            1.0 if box2_start_with_punct else 0.0,
            1.0 if layout_same else 0.0,
            len(box1['text']) / 50.0,  # 归一化文本长度
            len(box2['text']) / 50.0
        ]


def chunk(filename, binary=None, from_page=0, to_page=100000,
          lang="Chinese", callback=print, **kwargs):
    if not os.path.exists(filename):
        print(f"❌ 错误: 文件不存在 - {filename}")
        return [], []
    pdf_parser = Pdf()
    # 添加进度条
    def progress_callback(msg=None, prog=None):
        if msg:
            print(f"⏳ {msg}")
        if prog is not None:
            # 更新进度条显示
            pass
        if callback and msg:
            callback(msg)
    sections, tables = pdf_parser(filename if not binary else binary, from_page=from_page, to_page=to_page,
                                    callback=progress_callback)
    # 结果统计
    if VERBOSE_MODE:
        print(f"\n📊 ===== chunk函数执行完成 =====")
        print(f"📄 提取段落数: {len(sections)}")
        print(f"📊 提取表格数: {len(tables)}")
        if sections:
            print(f"📋 第一段预览: {sections[0][0][:100] if sections else '无内容'}...")
        
        # 打印表格对象的详细信息
        if tables:
            print(f"\n📋 ===== 表格对象详细信息 =====")
            for i, table in enumerate(tables):
                print(f"\n表格 {i+1}:")
                print(f"  类型: {type(table)}")
                
                if isinstance(table, dict):
                    print(f"  字典键: {list(table.keys())}")
                    for key, value in table.items():
                        print(f"    {key}: {type(value)} - {str(value)[:200]}{'...' if len(str(value)) > 200 else ''}")
                elif isinstance(table, list):
                    print(f"  列表长度: {len(table)}")
                    if len(table) > 0:
                        print(f"  第一个元素: {type(table[0])} - {str(table[0])[:200]}{'...' if len(str(table[0])) > 200 else ''}")
                else:
                    print(f"  内容: {str(table)[:500]}{'...' if len(str(table)) > 500 else ''}")
    
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
            "ERROR": "❌"
        }
        symbol = level_symbols.get(level, "ℹ️")
        print(f"[{timestamp}] {symbol} {message}")

def recognize_table_image(table_tuple, api_key=None, app_id=None):
    """
    使用千问模型识别表格图片内容
    
    Args:
        table_tuple: 表格元组对象，格式为 [(PIL.Image, 文本列表, 位置信息)]
        api_key: 千问API密钥，如果不提供则从环境变量获取
    
    Returns:
        str: 图片内容描述文本
    """
    try:
        # 提取表格元组中的图像对象
        if not isinstance(table_tuple, tuple) or len(table_tuple) < 1:
            return "错误: 表格元组格式不正确"
        
        # 获取PIL图像对象
        pil_image = table_tuple[0][0]
        if not hasattr(pil_image, 'mode') or not hasattr(pil_image, 'size'):
            return "错误: 表格元组第一个元素不是有效的PIL图像对象"
        
        # 将PIL图像转换为临时文件
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            pil_image.save(tmp_file.name, 'PNG')
            image_path = tmp_file.name
        
        try:
            # 调用千问模型识别图片内容 - 使用新的MultiModalConversation接口
            debug_print(f"开始调用千问模型识别图片: {pil_image.size[0]}x{pil_image.size[1]} 像素", "INFO")
            
            response = MultiModalConversation.call(
                model='qwen-vl-max',
                messages=[{
                    'role': 'user',
                    'content': [
                        {'image': image_path},  # 直接传本地路径
                        {'text': '请详细描述这个图片中的内容，包括可能的主题、图表、文字、结构等信息。'}
                    ]
                }],
                api_key=api_key
            )
            
            if response.status_code == HTTPStatus.OK:
                description = response.output.choices[0].message.content[0]['text']
                debug_print(f"千问模型识别成功: {description[:100]}...", "SUCCESS")
                return description
            else:
                error_msg = f"千问模型调用失败: {response.message}"
                debug_print(error_msg, "ERROR")
                return error_msg
                
        finally:
            # 清理临时文件
            if os.path.exists(image_path):
                os.unlink(image_path)
                
    except Exception as e:
        error_msg = f"识别表格图片时出错: {str(e)}"
        debug_print(error_msg, "ERROR")
        return error_msg

def process_pdf_document(input_file: str, output_file: str, lang: str = "Chinese", verbose: bool = True) -> dict:
    """
    PDF文档处理接口函数
    
    Args:
        input_file: 输入PDF文件路径
        output_file: 输出JSON文件路径
        lang: 文档语言 ("Chinese" 或 "english")
        verbose: 是否显示详细日志
    
    Returns:
        dict: 处理结果，包含文档内容、表格信息等
        {
            "status": "success" | "failed",
            "file_path": str,                    # 输入文件路径
            "output_file": str,                  # 输出文件路径
            "process_time": float,               # 处理时间(秒)
            "paragraphs": int,                   # 段落数量
            "tables": int,                       # 表格数量
            "sections": [{"page": int, "content": str}],  # 文档段落
            "tables_or_images": [{               # 表格/图片信息
                "page": int,                     # 所在页码
                "image_path": str,               # 图片路径
                "description": str,              # 描述信息
                "type": str                      # 类型("table"或"image")
            }],
            "language": str                      # 文档语言
        }
    """
    global VERBOSE_MODE
    VERBOSE_MODE = verbose
    
    # 检查输入文件
    if not os.path.exists(input_file):
        error_msg = f"输入文件不存在: {input_file}"
        debug_print(error_msg, "ERROR")
        return {"error": error_msg, "status": "failed"}
    
    # 检查输出目录
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        debug_print(f"创建输出目录: {output_dir}", "INFO")
        os.makedirs(output_dir, exist_ok=True)
    
    start_time = timer()
    
    try:
        # 调用chunk函数处理PDF
        debug_print("开始PDF解析...", "INFO")
        sections, tables = chunk(input_file, lang=lang,from_page=0,to_page=999)
        process_time = timer() - start_time
        # section 二次融合：如果一个section的token数量<50,则融合到下一个section（支持连续融合）
        
        # 导入tokenize方法
        from deepdoc.rag_tokenizer import tokenize
        
        debug_print("开始section二次融合...", "INFO")
        merged_sections = []
        i = 0
        while i < len(sections):
            section = sections[i]
            if isinstance(section, tuple):
                text, tag = section
                # 使用提供的tokenize方法进行准确的token计数
                tokenized_text = tokenize(text)
                estimated_tokens = len(tokenized_text.split())
                
                # 如果当前section token数<50，开始连续融合
                if estimated_tokens < 50 and i < len(sections) - 1:
                    merged_text = text
                    merged_count = 1  # 记录融合了几个section
                    current_tokens = estimated_tokens
                    
                    # 连续融合，直到token数≥50或没有更多section
                    while current_tokens < 50 and i + merged_count < len(sections):
                        next_section = sections[i + merged_count]
                        if isinstance(next_section, tuple):
                            next_text, next_tag = next_section
                            merged_text += "\n" + next_text
                            current_tokens = len(tokenize(merged_text).split())
                            merged_count += 1
                        else:
                            break  # 遇到非tuple格式，停止融合
                    
                    # 添加融合后的section（保留第一个section的tag信息）
                    merged_sections.append((merged_text, tag))
                    if merged_count > 1:
                        debug_print(f"连续融合section {i+1}到{i+merged_count}，最终token数：{current_tokens}", "INFO")
                    i += merged_count  # 跳过已融合的sections
                    continue
                
                # 不是短section，直接添加
                merged_sections.append(section)
            else:
                # 非tuple格式的section，直接添加
                merged_sections.append(section)
            i += 1
        
        # 用融合后的sections替换原始sections
        original_count = len(sections)
        sections = merged_sections
        debug_print(f"section二次融合完成: {original_count}个section融合为{len(sections)}个section", "SUCCESS")
        
        debug_print(f"PDF解析完成: {process_time:.2f}秒, {len(sections)}段落, {len(tables)}表格", "SUCCESS")
        
        # 准备可序列化的结果 - 提取纯文本内容，使用@@后面的数字作为页面号
        serializable_sections = []
        for section in sections:
            if isinstance(section, tuple):
                text, tag = section
                # 从tag中提取@@后面的数字作为页面号
                page_num = 1  # 默认页面号
                if isinstance(tag, str) and '@@' in tag:
                    try:
                        # 解析格式如：@@1	52.7	124.0	422.3	434.0##
                        tag_parts = tag.split('\t')
                        if tag_parts and tag_parts[0].startswith('@@'):
                            page_num = int(tag_parts[0][2:])  # 去掉@@取数字
                    except (ValueError, IndexError):
                        pass  # 使用默认页面号
                
                serializable_sections.append({
                    "page": page_num,
                    "content": str(text),  # 只使用纯文本内容
                    "tag": str(tag),
                    "length": len(text) if hasattr(text, '__len__') else 0
                })
            else:
                serializable_sections.append({
                    "page": 1,  # 默认页面号
                    "content": str(section),  # 只使用纯文本内容
                    "length": len(section) if hasattr(section, '__len__') else 0
                })
        
        # 处理表格数据
        serializable_tables = []
        for i, table in enumerate(tables[:10]):  # 最多保存10个表格
            if isinstance(table, dict):
                clean_table = {}
                for k, v in table.items():
                    if not str(type(v)).__contains__('Image'):
                        try:
                            json.dumps(v)  # 测试是否可序列化
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
        
        # 构建结果字典 - 符合指定格式
        result = {
            "status": "success",
            "file_path": input_file,
            "output_file": output_file,
            "process_time": process_time,
            "paragraphs": len(sections),
            "tables": len(tables),
            "sections": serializable_sections,
            "tables_or_images": [
                {
                    "page": table.get("page", 0) if isinstance(table, dict) else 0,
                    "image_path": table.get("image_path", "") if isinstance(table, dict) else "",
                    "description": table.get("description", str(table)) if isinstance(table, dict) else str(table),
                    "type": table.get("type", "table") if isinstance(table, dict) else "table"
                }
                for table in tables
            ],
            "language": lang
        }
        
        # 保存到输出文件
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        debug_print(f"结果已保存: {output_file}", "SUCCESS")
        
        # 返回结果
        return result
        
    except Exception as e:
        error_msg = f"PDF处理失败: {str(e)}"
        debug_print(error_msg, "ERROR")
        
        return {
            "error": error_msg,
            "status": "failed",
            "file_path": input_file,
            "process_time": timer() - start_time
        }

#  测试代码
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    file_path = os.path.join(project_root,  "user_document", "24S151167.pdf")
    lang="english"
    output_file = os.path.join(project_root, "user_document", "result.json")
    
    print(f"\n📚 ===== PDF表格检测工具使用示例 =====")
    print(f"📁 输入文件: {file_path}")
    print(f"📄 输出文件: {output_file}")
    print(f"🌐 语言: {lang}")
    print(f"📸 提取表格图片将保存到: {os.path.join(os.path.dirname(output_file), 'detected_tables')}")
    print(f"🔍 可疑连续块图片将保存到: {os.path.join(os.path.dirname(output_file), 'suspicious_continuous_blocks')}")
    print(f"\n开始处理...")
    
    result=process_pdf_document(file_path, output_file,lang=lang)
    
    # 打印结果摘要
    if result.get("status") == "success":
        print(f"\n✅ 处理成功！")
        print(f"📊 解析段落数: {result.get('paragraphs', 0)}")
        print(f"📊 提取表格数: {result.get('tables', 0)}")
        print(f"📸 提取表格图片数: {len(result.get('extracted_table_images', []))}")
        print(f"📸 检测表格块数: {len(result.get('detected_table_images', []))}")
        print(f"🔍 可疑连续块数: {len(result.get('suspicious_continuous_block_images', []))}")
            
    else:
        print(f"\n❌ 处理失败: {result.get('error', '未知错误')}")
    
    print(f"\n📋 详细结果已保存到: {output_file}")