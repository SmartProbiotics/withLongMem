# withLongMem
> 一个面向新手的 **LangGraph + Milvus Lite** 长期记忆 Demo  
> 让 LLM 拥有“过目不忘”的本地向量记忆，支持 **稀疏 + 稠密混合检索** 与 **动态工具注入**。

---

## 🧠 一句话原理
1. 用户每轮输入 → 稠密 WW& 稀疏向量 → 存入 **本地 Milvus Lite**；  
2. 下轮提问时先 **混合检索** 最相关记忆块；  
3. 把记忆块 **动态封装成工具** 注入 LangGraph，让 Agent 自行决定是否调用；  
4. 提供 **寿命计数** `deny_to_die`，到期自动淘汰记忆，防止无限膨胀。

---

## 📁 项目结构（≈ 3 k 行有效代码）
```
withLongMem/
├── src/
│   ├── graph_builder.py      # LangGraph 状态图定义 & 动态工具注入
│   ├── myMilvus.py           # Milvus Lite 增删改查 + 混合检索 + 寿命管理
│   ├── memory_module.py      # 记忆压缩 / 长期存储 / 结构化 LLM 输出
│   ├── dynamic_tool.py       # 把 MemoryBlock 动态变成 LangChain Tool
│   ├── embedding.py          # DashScope 稀疏 & 稠密向量生成
│   ├── memoryBlock.py        # 记忆块数据模型
│   └── tools.py              # 基础工具（读写文件、执行命令等）
├── test/
│   └── main.py               # 单测脚本（可直接跑）
├── StateGraphDemo.py         # 演示对话循环
├── milvus_demo.db            # 本地向量数据库（Git 可忽略）
└── .env                      # 放 API_KEY（Git 可忽略）
```

---

## 🚀 1 分钟跑起来
```bash
# 1. 装依赖
pip install -r requirements.txt   # 见文末清单

# 2. 写密钥
echo "OPENAI_API_KEY=sk-your-dashscope-key" >> .env
echo "DASHSCOPE_API_KEY=sk-your-dashscope-key" >> .env

# 3. 运行演示
python StateGraphDemo.py
```

---

## 🔍 核心能力一览
| 功能 | 文件 | 备注 |
|----|------|------|
| **本地向量库** | `myMilvus.py` | 零依赖 Milvus Lite，开箱即用，支持 **稀疏 + 稠密混合检索** |
| **动态工具** | `dynamic_tool.py` | 记忆块 → 自动生成 `get_memory_0/1/2...` 工具，**运行时注入** LangGraph |
| **记忆寿命** | `deny_to_die` 字段 | 每调用一次 `-1`，≤0 自动删除，防止记忆无限膨胀 |
| **结构化输出** | `memory_module.py` | 用 Pydantic 约束 LLM，保证 **标题 / 摘要 / 来源** 字段稳定 |

---

## 🎞️ 典型交互流程
```text
用户：把“项目地址 /root/foo”记一下。
Agent：✅ 已保存长期记忆《项目路径》。

用户：我上次让你记的目录是啥？
Agent：🔍 调用 get_memory_0 → 返回“/root/foo”。
```

---

## 🛠️ 关键 API（可直接 import）
```python
from src.myMilvus import db, LONG_TERM_MEM
from src.memory_module import save_long_term_memory

# 存记忆
save_long_term_memory("项目地址 /root/foo", source="用户口头")

# 查记忆
hits = db.hybrid_search("项目地址", top_k=3, collection_name=LONG_TERM_MEM)
blocks = db.result_to_blocks(hits)

# 寿命管理
for b in blocks:
    db.block_to_die(b)   # 每调用一次寿命 -1
```

---

## 🔐 安全 & 最佳实践
- **禁止硬编码密钥**——已统一用 `os.getenv` 读取；  
- **`.env` 与 `*.db` 已加入 `.gitignore`**，防止误提交；  
- **支持离线**——可设置 `HF_HUB_OFFLINE=1` 运行，无公网也能用；  
- **单文件数据库**——`milvus_demo.db` 可随项目拷贝，部署只需复制这一个文件。

---

## 📦 主要依赖（requirements.txt 精简版）
```
langchain>=0.2
langgraph>=0.1
pymilvus>=2.4          # Milvus Lite 内置
openai>=1.0              # DashScope 兼容 OpenAI 接口
dashscope>=1.15
python-dotenv
```

---

## 🧪 单元测试
```bash
cd test
python main.py   # 自动写入 → 检索 → 寿命扣减 → 过期删除 全流程
```

---

## 🏷️ 适合人群 & 场景
- **LLM 初学者**：把“向量数据库 +  Agent 记忆”一次跑通；  
- **本地知识库**：个人笔记、项目文档秒变可检索记忆；  
- **低资源环境**：树莓派、笔记本都能跑，**纯 CPU** 无显卡亦可；  
- **GitHub 首项目**：代码量 < 3 k 行，注释少、逻辑直，方便二开/PR。

---

## 🤝 二次开发 Tips
1. 想换 Embedding？改 `src/embedding.py` 即可；  
2. 想换 LLM？把 `ChatOpenAI` 换成任意 `langchain_core.language_models.BaseChatModel` 子类；  
3. 想加字段？在 `myMilvus.insert_data` 里按顺序追加即可，**顺序必须对齐**；  
4. 想调寿命？改 `DENY_TO_DIE = 10` 或动态传参都行。

---

## 📄 许可证
MIT · 随便用、随便改、随便商用，**记得自己保管好 API_KEY**。

---

> 如果对你有帮助，点个 ⭐ 让仓库活下去 ~