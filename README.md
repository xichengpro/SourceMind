# SourceMind - 学术论文智能分析助手

SourceMind 是一个基于 **LangGraph** 和 **Streamlit** 构建的智能学术论文分析工具。它能够自动化地从 PDF 文件或 Arxiv 链接中提取核心信息，进行深度研读，并生成结构化的分析报告。

## 🌟 核心功能

* **多源输入支持**：支持直接输入 Arxiv 论文链接或上传本地 PDF 文件。

* **并行化深度分析**：
  * **📝 智能翻译**：提供摘要、引言、结论的精准翻译。
  * **🔑 核心要点提取**：自动识别并提取论文的关键贡献和创新点。
  * **📊 实验结果分析**：提取实验设置、数据集、对比方法及结果数据。
  * **📖 术语解释**：自动识别专业术语并提供上下文相关的解释。
  * **🔍 相关工作搜索**：集成网络搜索（Tavily/Exa/Google），自动关联相关研究背景，包括 **GitHub 代码搜索**，自动查找官方实现与社区复现。
  * **👥 圆桌讨论 / 对话评审**：
    *   **对话模式**（默认）：模拟初学者与专家的问答，辅助快速理解。
    *   **圆桌模式**（高级）：模拟一场多智能体（主持人、作者、方法论专家、应用实践者）之间的学术辩论，全方位剖析论文优缺点。
    *   **严格约束**：所有角色（Methodology Critic, Industry Practitioner）均受到严格的 Prompt 约束，**严禁捏造数据、参考文献或虚假案例**，确保讨论的学术严谨性。
    *   **实时控制**：支持实时流式显示讨论内容，并可随时**手动停止**讨论。
  

* **💬 互动问答 (Q&A)**：分析完成后，支持用户基于论文内容进行自由提问（Human-in-the-loop），获取精准解答。
  * *注：为保证流程专注，系统在分析或讨论进行中会暂时禁用提问功能。*

* **📜 历史记录与本地存储**：
  * **自动保存**：所有分析结果自动保存到本地数据库 (`history_data/`)。
  * **便捷浏览**：提供历史记录列表，支持按时间排序和关键词搜索。
  * **一键回溯**：随时查看过往分析报告，无需重新消耗 Token。
  * **数据导出**：支持将分析结果导出为 JSON 格式，方便二次开发或备份。

* **👁️ 视觉解析模式 (VLM)**：支持使用多模态大模型（如 GPT-4o, Claude 3.5 Sonnet）逐页解析 PDF，完美还原公式、图表和复杂排版。

* **📑 综合报告生成**：汇总所有分析维度，生成 Markdown 格式的研读报告。
  * **实时预览**：报告生成完成后立即在界面展示，无需等待后续讨论流程。
  * **下载支持**：支持下载最终报告及圆桌讨论记录（Markdown 格式）。

* **🤖 灵活的模型配置**：
  * 支持 **OpenAI**, **Anthropic**, **OpenRouter** 及 **自定义 OpenAI 兼容接口**（如 Ollama, vLLM）。
  * **细粒度控制**：可为翻译、搜索总结、视觉解析、对话评审等不同任务单独配置特定的模型。

* **🔭 全链路可观测性**：集成 **Langfuse** **LangSmith**，支持对 LangGraph 工作流及内部 LLM 调用的完整追踪与可视化。

## 🛠️ 技术栈

* **Workflow Orchestration**: [LangGraph](https://github.com/langchain-ai/langgraph)
* **LLM Integration**: [LangChain](https://github.com/langchain-ai/langchain)
* **User Interface**: [Streamlit](https://streamlit.io/)
* **Observability**: [Langfuse](https://langfuse.com/), [LangSmith](https://smith.langchain.com/)
* **PDF Parsing**: [PyMuPDF4LLM](https://github.com/pymupdf/PyMuPDF) / Custom VLM Parser

## 🚀 快速开始

### 1. 环境要求

* Python >= 3.9
* 建议使用 `uv` 或 `pip` 管理依赖

### 2. 安装依赖

```bash
# 克隆项目
git clone https://github.com/yourusername/SourceMind.git
cd SourceMind

# 安装依赖
pip install -r requirements.txt
# 或者使用 uv
uv sync
```

### 3. 配置环境

复制示例配置文件并重命名为 `.env`：

```bash
cp .env.example .env
```

编辑 `.env` 文件，配置你的 API Key：

```ini
# 核心模型配置 (必填)
LLM_PROVIDER=OpenAI
OPENAI_API_KEY=sk-proj-...
OPENAI_MODEL_NAME=gpt-4o

# Langfuse 追踪配置 (可选，推荐)
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com

# LangSmith 追踪配置 (可选，推荐)
LANGSMITH_TRACING_V2=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY=lsv2_...
LANGSMITH_PROJECT=SourceMind

# 网络搜索配置 (可选，用于相关工作搜索)
TAVILY_API_KEY=tvly-...
EXA_API_KEY=exa-...
SERPAPI_API_KEY=serp-...
```

### 4. 启动应用

```bash
streamlit run app.py
```

浏览器自动打开 `http://localhost:8501` 即可使用。

## ⚙️ 详细配置说明

### 模型配置 (Model Configuration)

SourceMind 支持为不同任务配置不同的模型，以平衡成本和效果。你可以在界面侧边栏或 `.env` 文件中进行配置。

* **核心模型**：用于生成最终报告、提取要点、问答等通用任务。
* **翻译专用模型**：建议使用上下文窗口大且翻译能力强的模型（如 `claude-3-5-sonnet`）。
* **评审/圆桌讨论模型**：用于驱动圆桌讨论中的主持人与评审员，建议使用推理能力强的模型。
* **视觉解析 (VLM) 模型**：仅在开启 VLM 模式时使用，必须支持视觉输入（如 `gpt-4o`, `claude-3-5-sonnet`）。

### 可观测性 (Langfuse & LangSmith)

本项目支持双重可观测性配置：

本项目深度集成了 Langfuse & LangSmith。配置好 `LANGFUSE_*` `LANGSMITH_*`  环境变量后：
1. **自动追踪**：每次点击“开始分析”，Langfuse 后台会自动生成一条名为 `SourceMind Analysis` 或 `LangGraph` 的 Trace。
2. **完整视图**：该 Trace 包含从图执行开始，到每一个节点（Node），再到每一次底层 LLM 调用的完整层级结构。

## 📂 项目结构

```
SourceMind/
├── app.py              # Streamlit 主程序入口
├── src/
│   ├── graph.py        # LangGraph 工作流定义
│   ├── nodes.py        # 各个分析节点的具体实现 (含 Prompt 逻辑)
│   ├── loader.py       # PDF 加载与解析逻辑 (含 VLM 实现)
│   ├── model_utils.py  # 模型实例化、回调管理与配置
│   ├── prompts.py      # Prompt 模板管理
│   └── state.py        # LangGraph 状态定义
├── .env.example        # 环境变量示例
└── requirements.txt    # 项目依赖
```

## 📝 License

[MIT License](LICENSE)
