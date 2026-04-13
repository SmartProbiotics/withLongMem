# DeepDoc Test 使用说明

## 功能概述

这个测试程序提供了PDF文档处理功能，包括：
- PDF文本提取和分段
- 表格识别和结构分析
- 图片内容识别（使用千问模型）
- 多种命令行参数支持

## 使用方法

### 基本使用

```bash
# 运行完整的PDF处理流程
cd /root/withLongMem && python -m deepdoc.test
```

### 命令行参数

```bash
# 显示帮助信息
python -m deepdoc.test --help

# 仅测试表格图片识别功能
python -m deepdoc.test --test-table-only

# 测试接口函数
python -m deepdoc.test --test-interface
```

### 环境变量配置

要使用千问模型识别表格图片内容，需要配置以下环境变量：

```bash
export DASHSCOPE_API_KEY="你的千问API密钥"
export QWEN_APP_ID="你的千问应用ID"
```

## 主要函数

### process_pdf_document(input_file, output_file, lang="Chinese", verbose=True)

PDF文档处理接口函数

**参数：**
- `input_file`: 输入PDF文件路径
- `output_file`: 输出JSON文件路径
- `lang`: 文档语言 ("Chinese" 或 "english")
- `verbose`: 是否显示详细日志

**返回值：**
- 处理结果字典，包含文档内容、表格信息等

### recognize_table_image(table_tuple, api_key=None, app_id=None)

使用千问模型识别表格图片内容

**参数：**
- `table_tuple`: 表格元组对象，格式为 (PIL.Image, 文本列表, 位置信息)
- `api_key`: 千问API密钥（可选，会从环境变量获取）
- `app_id`: 千问应用ID（可选，会从环境变量获取）

**返回值：**
- 图片内容描述文本

## 输出结果

程序会在 `/root/withLongMem/test/result.json` 中保存处理结果，包括：
- 文档基本信息（路径、处理时间等）
- 提取的段落内容
- 表格结构信息
- 第一个表格的图片描述（如果启用了千问模型）

## 注意事项

1. 首次运行需要下载模型文件，可能需要较长时间
2. 千问模型需要有效的API密钥才能使用
3. 处理大型PDF文件可能需要较长时间
4. 程序会自动使用GPU加速（如果可用）

## 示例输出

```json
{
  "status": "success",
  "file_path": "/root/withLongMem/test/user_document/24S151167.pdf",
  "process_time": 87.6,
  "paragraphs": 271,
  "tables": 13,
  "sections": [...],
  "tables_info": [...],
  "first_table_description": "图片描述内容..."
}
```