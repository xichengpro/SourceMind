import streamlit as st
import os
import time
from dotenv import load_dotenv
from src.graph import create_graph
from src.nodes import review_dialogue_node
from src.model_utils import get_llm
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# Safe import for Langfuse decorators
try:
    from langfuse import observe
except ImportError:
    # Dummy decorator if langfuse is not installed
    def observe(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

# Load environment variables
load_dotenv()

# Configure LangSmith Tracing if enabled in env
if os.getenv("LANGSMITH_TRACING") == "true":
    os.environ["LANGSMITH_TRACING"] = "true"
    # Ensure endpoint is set, default to standard if not
    if not os.getenv("LANGSMITH_ENDPOINT"):
        os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
    # Project defaults to "default" if not set, but user might have set it in .env
    if not os.getenv("LANGSMITH_PROJECT"):
        os.environ["LANGSMITH_PROJECT"] = "SourceMind"

# Configure Langfuse Tracing if enabled in env
if os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"):
    try:
        from langfuse.langchain import CallbackHandler
        st.session_state.langfuse_handler = CallbackHandler()
    except ImportError:
        print("Langfuse not installed. Skipping Langfuse configuration.")
        
st.set_page_config(
    page_title="学术论文分析助手--SourceMind",
    page_icon="📚",
    layout="wide"
)

def save_uploaded_file(uploaded_file):
    """Save uploaded file to a temporary directory."""
    if not os.path.exists("temp"):
        os.makedirs("temp")
    
    file_path = os.path.join("temp", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def render_model_config_ui(section_title, env_prefix, key_prefix, checkbox_label="启用独立配置"):
    """
    Render model configuration UI for a specific section.
    
    Args:
        section_title: Title of the configuration section.
        env_prefix: Prefix for environment variables (e.g., "TRANSLATION_" or "").
                    Final var names will be like {env_prefix}LLM_PROVIDER.
        key_prefix: Prefix for streamlit widget keys to ensure uniqueness.
        checkbox_label: Label for the enable checkbox (if applicable).
    """
    st.subheader(section_title)
    
    # Checkbox to enable if it's an optional section (heuristic based on prefix)
    is_enabled = True
    if env_prefix:
        is_enabled = st.checkbox(checkbox_label, key=f"{key_prefix}_enable", value=False)
        if not is_enabled:
            # Clear provider env var if disabled
            provider_env_key = f"{env_prefix}LLM_PROVIDER"
            if provider_env_key in os.environ:
                del os.environ[provider_env_key]
            return

    provider = st.selectbox(
        "选择模型提供商",
        ["OpenAI", "Anthropic", "OpenRouter", "自定义 (OpenAI 兼容)"],
        key=f"{key_prefix}_provider"
    )
    
    os.environ[f"{env_prefix}LLM_PROVIDER"] = provider
    
    if provider == "OpenAI":
        api_key = st.text_input("OpenAI API Key", type="password", key=f"{key_prefix}_openai_key", value=os.getenv(f"{env_prefix}OPENAI_API_KEY", ""))
        if api_key:
            os.environ[f"{env_prefix}OPENAI_API_KEY"] = api_key
        
        model_options = ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo", "o1-preview", "o1-mini"]
        selected_model = st.selectbox("选择模型", model_options, index=0, key=f"{key_prefix}_openai_model")
        if selected_model:
            os.environ[f"{env_prefix}OPENAI_MODEL_NAME"] = selected_model
            
    elif provider == "Anthropic":
        api_key = st.text_input("Anthropic API Key", type="password", key=f"{key_prefix}_anthropic_key", value=os.getenv(f"{env_prefix}ANTHROPIC_API_KEY", ""))
        if api_key:
            os.environ[f"{env_prefix}ANTHROPIC_API_KEY"] = api_key
            
        model_options = ["claude-3-5-sonnet-20240620", "claude-3-opus-20240229", "claude-3-sonnet-20240229"]
        selected_model = st.selectbox("选择模型", model_options, index=0, key=f"{key_prefix}_anthropic_model")
        if selected_model:
            os.environ[f"{env_prefix}ANTHROPIC_MODEL_NAME"] = selected_model

    elif provider == "OpenRouter":
        api_key = st.text_input("OpenRouter API Key", type="password", key=f"{key_prefix}_openrouter_key", value=os.getenv(f"{env_prefix}OPENROUTER_API_KEY", ""))
        if api_key:
            os.environ[f"{env_prefix}OPENROUTER_API_KEY"] = api_key
            # Also set OPENAI_API_KEY as the underlying client uses it
            os.environ[f"{env_prefix}OPENAI_API_KEY"] = api_key
        
        # OpenRouter base URL is usually https://openrouter.ai/api/v1
        os.environ[f"{env_prefix}OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"
        
        model_name = st.text_input("OpenRouter Model Name (例如: google/gemini-pro-1.5)", value=os.getenv(f"{env_prefix}OPENROUTER_MODEL_NAME", "openai/gpt-4o"), key=f"{key_prefix}_openrouter_model")
        if model_name:
            os.environ[f"{env_prefix}OPENROUTER_MODEL_NAME"] = model_name
            # Also set OPENAI_MODEL_NAME as the underlying client uses it
            os.environ[f"{env_prefix}OPENAI_MODEL_NAME"] = model_name

    elif provider == "自定义 (OpenAI 兼容)":
        api_base = st.text_input("API Base URL", key=f"{key_prefix}_custom_base", value=os.getenv(f"{env_prefix}OPENAI_API_BASE", ""))
        api_key = st.text_input("API Key", type="password", key=f"{key_prefix}_custom_key", value=os.getenv(f"{env_prefix}OPENAI_API_KEY", ""))
        model_name = st.text_input("模型名称", key=f"{key_prefix}_custom_model", value=os.getenv(f"{env_prefix}OPENAI_MODEL_NAME", ""))
        
        if api_base: os.environ[f"{env_prefix}OPENAI_API_BASE"] = api_base
        if api_key: os.environ[f"{env_prefix}OPENAI_API_KEY"] = api_key
        if model_name: os.environ[f"{env_prefix}OPENAI_MODEL_NAME"] = model_name

@observe(name="SourceMind Analysis")
def run_analysis_stream(app, inputs, config=None):
    """Run the graph stream with observability."""
    for output in app.stream(inputs, config=config):
        yield output

def main():
    st.title("📚 学术论文分析助手--SourceMind")
    st.markdown("""
    本助手可以分析 **Arxiv 论文** 或 **上传的 PDF 文件**。
    核心功能包括：
    - 📝 **论文翻译** (摘要、引言、结论)
    - 🔑 **核心要点提取**
    - 📊 **实验结果分析**
    - 📖 **专业术语解释**
    - 📑 **生成综合报告**
    """)
    
    # Sidebar for configuration
    with st.sidebar:
        with st.expander("大模型配置", expanded=True):
            # Core Model Configuration
            render_model_config_ui("核心模型", "", "core")
            st.divider()
            
            # Translation Dedicated Model Configuration
            render_model_config_ui("翻译专用模型 (可选,默认使用核心模型)", "TRANSLATION_", "trans")
            st.divider()

            # Related Work Dedicated Model Configuration
            render_model_config_ui("网络搜索结果处理模型 (可选,默认使用核心模型)", "RELATED_WORK_", "rel_work")
            st.divider()

            # Dialogue Review Model Configuration
            render_model_config_ui("评审/圆桌讨论模型 (可选,默认使用核心模型)", "REVIEW_", "review")
            st.divider()

            # VLM Dedicated Model Configuration
            render_model_config_ui("视觉解析 (VLM) 模型 (可选,默认使用PyMuPDF4LLM)", "VLM_", "vlm", checkbox_label="启用视觉解析模式 (VLM)")
            st.divider()

        # Web Search Configuration
        with st.expander("网络搜索配置", expanded=True):
            tavily_key = st.text_input("Tavily API Key (用于搜索相关工作)", type="password", value=os.getenv("TAVILY_API_KEY", ""))
            if tavily_key:
                os.environ["TAVILY_API_KEY"] = tavily_key
                
            exa_key = st.text_input("Exa API Key (可选，增强搜索)", type="password", value=os.getenv("EXA_API_KEY", ""))
            if exa_key:
                os.environ["EXA_API_KEY"] = exa_key
                
            serp_key = st.text_input("SerpAPI Key (可选，Google 搜索)", type="password", value=os.getenv("SERPAPI_API_KEY", ""))
            if serp_key:
                os.environ["SERPAPI_API_KEY"] = serp_key
                
            if not tavily_key and not exa_key and not serp_key:
                st.caption("如果没有提供 Key，将跳过网络搜索步骤。")
        
        st.markdown("---")
        # st.markdown("基于 **LangGraph** & **LangChain** 构建")

    # Input section
    st.header("选择输入源")
    input_type = st.radio("请选择:", ["Arxiv 链接", "上传 PDF"], key="input_type_radio")
    
    source = None
    
    if input_type == "Arxiv 链接":
        url = st.text_input("请输入 Arxiv 链接 (例如 https://arxiv.org/abs/2310.00000)", key="arxiv_url_input")
        if url:
            source = url
            
    else:
        uploaded_file = st.file_uploader("上传 PDF 文件", type=["pdf"], key="pdf_file_uploader")
        if uploaded_file:
            source = save_uploaded_file(uploaded_file)
            st.success(f"文件已上传: {uploaded_file.name}")
            
    # Advanced Options
    with st.expander("高级选项"):
        enable_full_translation = st.checkbox("开启全文翻译（含术语一致性优化）", help="开启后将逐段翻译全文，并自动提取术语表以保证上下文一致性。速度较慢，消耗 Token 较多。\n\n 默认会翻译摘要、引言、结论。", value=False)

        enable_round_table = st.checkbox("开启圆桌讨论 (Round Table Discussion)", help="开启后，将模拟一场多智能体（主持人、作者、专家、实践者）之间的学术辩论。\n\n 默认会在报告生成后进行模拟的“初学者-作者”对话。", value=False)
        
        # VLM parsing is now controlled via the sidebar configuration
        enable_vlm_parsing = st.session_state.get("vlm_enable", False)

        enable_vlm_parsing = st.session_state.get("vlm_enable", False)

        if enable_full_translation:
            st.info("已开启全文翻译（含术语一致性优化）")
        else:
            st.caption("默认翻译摘要、引言、结论。")

        if enable_round_table:
            st.info("已开启圆桌讨论 (Round Table Discussion)")
        else:
            st.caption("默认在报告生成后进行模拟的“初学者-作者”对话。")

    # --- Session State Management ---
    if "analysis_running" not in st.session_state:
        st.session_state.analysis_running = False
    if "analysis_result" not in st.session_state:
        st.session_state.analysis_result = None
    if "analysis_error" not in st.session_state:
        st.session_state.analysis_error = None
    if "execution_logs" not in st.session_state:
        st.session_state.execution_logs = []
    if "round_table_running" not in st.session_state:
        st.session_state.round_table_running = False

    def start_analysis():
        # Validate inputs
        input_type = st.session_state.get("input_type_radio")
        valid = False
        if input_type == "Arxiv 链接":
            if st.session_state.get("arxiv_url_input"):
                valid = True
        else:
            if st.session_state.get("pdf_file_uploader"):
                valid = True
        
        if not valid:
            st.session_state.analysis_error = "请提供有效的输入源。"
            st.session_state.analysis_running = False
            return

        st.session_state.analysis_running = True
        st.session_state.analysis_result = None
        st.session_state.analysis_error = None
        st.session_state.execution_logs = []

    def stop_analysis():
        st.session_state.analysis_running = False
        
    def handle_human_qa():
        """Handle human Q&A interaction."""
        if not st.session_state.analysis_result:
            return
            
        doc_content = st.session_state.analysis_result.get("doc_content", "")
        if not doc_content:
            st.error("无法获取论文内容，请先运行分析。")
            return
            
        user_question = st.session_state.get("human_qa_input", "")
        if not user_question:
            return
            
        # Display user question immediately
        if "qa_history" not in st.session_state:
            st.session_state.qa_history = []
        
        st.session_state.qa_history.append({"role": "user", "content": user_question})
        
        # Generate answer
        with st.spinner("思考中..."):
            try:
                llm = get_llm()
                # Use a QA prompt that has access to the full document content
                qa_prompt = ChatPromptTemplate.from_template("""
                你是一位精通这篇论文的学术助手。
                请根据以下论文内容，回答用户的提问。
                
                论文内容摘要/片段：
                {doc_content}
                
                用户提问：{question}
                
                回答要求：
                1. 准确、客观，基于论文内容。
                2. 如果论文中没有提到，请明确告知。
                """)
                
                # Limit context size to avoid token limits, though modern models handle large context
                # Taking first 50k chars is a safe heuristic for now
                chain = qa_prompt | llm | StrOutputParser()
                answer = chain.invoke({
                    "doc_content": doc_content[:50000], 
                    "question": user_question
                })
                
                st.session_state.qa_history.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                st.error(f"回答失败: {str(e)}")
                
        # Clear input
        st.session_state.human_qa_input = ""

    # --- Analysis Button Section ---
    if not st.session_state.analysis_running:
        st.button("🚀 开始分析", type="primary", on_click=start_analysis)
        if st.session_state.analysis_error:
            st.error(st.session_state.analysis_error)
    else:
        st.button("🛑 停止分析", type="secondary", on_click=stop_analysis)

    # --- Execution & Result View ---
    
    # Determine if we should show the results area
    show_results = st.session_state.analysis_running or st.session_state.analysis_result is not None
    
    if show_results:
        st.markdown("---")
        if st.session_state.analysis_running:
            st.subheader("🚀 正在分析... (结果实时生成中)")
            # Progress bar and status area
            progress_bar = st.progress(0)
            status_container = st.empty()
        elif st.session_state.get("analysis_error"):
            st.subheader("❌ 分析失败")
            status_container = st.empty() # Placeholder
            progress_bar = st.empty()
        else:
            st.header("✅ 分析完成")
            status_container = st.empty() # Placeholder
            progress_bar = st.empty()

        # Initialize Tabs immediately so they are visible during analysis
        tab_names = [
            "论文翻译", 
            "论文要点", 
            "论文实验", 
            "专业术语",
            "提取的图表",
            "相关搜索",
            "研读报告", 
            "评审/圆桌讨论"
        ]
        tabs = st.tabs(tab_names)
        
        # Map state keys to tabs for easy updating
        # Note: Some state keys map to specific tabs
        tab_map = {
            "translation": tabs[0],
            "key_points": tabs[1],
            "experiments": tabs[2],
            "terms": tabs[3],
            "figures": tabs[4],
            "related_work_search": tabs[5],
            "final_report": tabs[6],
            "review_dialogue": tabs[7]
        }

    # --- Logic when Analysis is Running ---
    if st.session_state.analysis_running:
        if not source:
            # Fallback check (should be caught by callback, but safe to keep)
            st.error("请提供有效的输入源。")
            st.session_state.analysis_running = False
        
        elif not os.environ.get("OPENAI_API_KEY"):
            st.error("请在侧边栏设置您的 OpenAI API Key。")
            st.session_state.analysis_running = False

        else:
            try:
                # Create graph
                app = create_graph()
                
                # Configure Round Table Streaming Container
                # This allows nodes.py to write directly to the Round Table tab
                st.session_state.stream_container = tab_map["review_dialogue"]
                
                # Progress definition
                steps_config = {
                    "load_paper": {"running": "📥 正在加载论文...", "done": "✅ 论文加载完成", "weight": 10},
                    "translate": {"running": "🌐 正在翻译论文...", "done": "✅ 翻译任务完成", "weight": 30},
                    "extract_key_points": {"running": "🔑 正在提取核心要点...", "done": "✅ 核心要点提取完成", "weight": 10},
                    "extract_experiments": {"running": "📊 正在提取实验数据...", "done": "✅ 实验数据提取完成", "weight": 10},
                    "explain_terms": {"running": "📖 正在解释专业术语...", "done": "✅ 专业术语解释完成", "weight": 10},
                    "related_work_search": {"running": "🔍 正在搜索相关工作...", "done": "✅ 相关工作搜索完成", "weight": 15},
                    "generate_report": {"running": "📑 正在生成研读报告...", "done": "✅ 最终报告生成完成", "weight": 15},
                    "review_dialogue": {"running": "👥 正在进行评审/圆桌讨论...", "done": "✅ 评审/圆桌讨论完成", "weight": 10}
                }
                
                step_status = {key: 'pending' for key in steps_config}
                step_timing = {key: {'start': None, 'end': None, 'duration': None} for key in steps_config}
                step_status['load_paper'] = 'running'
                step_timing['load_paper']['start'] = time.time()
                
                current_progress = 0
                completed_nodes = set()
                
                def format_duration(seconds):
                    return f"{seconds:.1f}秒" if seconds < 60 else f"{seconds/60:.1f}分钟"

                def render_logs():
                    with status_container.container():
                        # st.info("🚀 分析进度:")
                        cols = st.columns(4) # Grid layout for status
                        idx = 0
                        for key, config in steps_config.items():
                            status = step_status[key]
                            with cols[idx % 4]:
                                if status == 'running':
                                    st.info(config['running'])
                                elif status == 'done':
                                    duration = step_timing[key]['duration']
                                    d_text = f" ({format_duration(duration)})" if duration else ""
                                    st.caption(f"{config['done']}{d_text}")
                            idx += 1
                
                render_logs()
                
                final_state = {}
                run_config = {}
                if "langfuse_handler" in st.session_state:
                    run_config["callbacks"] = [st.session_state.langfuse_handler]

                # Run stream
                for output in run_analysis_stream(app, {
                    "source": source,
                    "is_full_translation": enable_full_translation,
                    "use_vlm_parsing": enable_vlm_parsing,
                    "enable_round_table": enable_round_table
                }, config=run_config):
                    for node_name, state_update in output.items():
                        final_state.update(state_update)
                        
                        # Real-time Tab Update with Error Handling
                        try:
                            # Identify which tab corresponds to this node update
                            if "final_report" in state_update:
                                tab_map["final_report"].markdown(state_update["final_report"])
                            if "translation" in state_update:
                                tab_map["translation"].markdown(state_update["translation"])
                            if "key_points" in state_update:
                                tab_map["key_points"].markdown(state_update["key_points"])
                            if "experiments" in state_update:
                                tab_map["experiments"].markdown(state_update["experiments"])
                            if "terms" in state_update:
                                tab_map["terms"].markdown(state_update["terms"])
                            if "figures" in state_update and state_update["figures"]:
                                with tab_map["figures"]:
                                    st.write(f"共提取到 {len(state_update['figures'])} 张图表")
                                    for img in state_update['figures']:
                                        st.image(img, caption=os.path.basename(img))
                            if "related_work_search" in state_update:
                                tab_map["related_work_search"].markdown(state_update["related_work_search"])
                        except Exception as update_err:
                            # Log error but do not crash the main loop
                            print(f"Error updating tabs: {update_err}")
                            # Optional: show a small warning in status
                            st.warning(f"部分结果显示更新失败: {update_err}")

                        # Update Progress logic (same as before)
                        if node_name in steps_config:
                            step_status[node_name] = 'done'
                            end_time = time.time()
                            step_timing[node_name]['end'] = end_time
                            if step_timing[node_name]['start']:
                                step_timing[node_name]['duration'] = end_time - step_timing[node_name]['start']
                            
                            if node_name == 'load_paper':
                                for next_step in ['translate', 'extract_key_points', 'extract_experiments', 'explain_terms', 'related_work_search']:
                                    step_status[next_step] = 'running'
                                    step_timing[next_step]['start'] = time.time()
                            
                            if node_name not in completed_nodes:
                                completed_nodes.add(node_name)
                                current_progress += steps_config[node_name]['weight']
                                progress_bar.progress(min(current_progress, 95))
                        
                        parallel_steps = ['translate', 'extract_key_points', 'extract_experiments', 'explain_terms', 'related_work_search']
                        if all(step_status[s] == 'done' for s in parallel_steps):
                            if step_status['generate_report'] != 'running' and step_status['generate_report'] != 'done':
                                step_status['generate_report'] = 'running'
                                step_timing['generate_report']['start'] = time.time()
                        
                        if step_status['generate_report'] == 'done':
                             if step_status['review_dialogue'] != 'running' and step_status['review_dialogue'] != 'done':
                                step_status['review_dialogue'] = 'running'
                                step_timing['review_dialogue']['start'] = time.time()
                        
                        render_logs()

                # Finalize
                step_status['review_dialogue'] = 'done'
                progress_bar.progress(100)
                st.success("🎉 全部分析流程结束！")
                
                # Cleanup stream container
                if "stream_container" in st.session_state:
                    del st.session_state.stream_container

                # Store result
                st.session_state.analysis_result = final_state
                st.session_state.analysis_running = False
                st.rerun()
                
            except Exception as e:
                st.error(f"发生错误: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
                st.session_state.analysis_running = False
                st.session_state.analysis_error = f"分析过程中发生错误: {str(e)}"
                if "stream_container" in st.session_state:
                    del st.session_state.stream_container
                st.rerun()

    # --- Logic when Analysis is Done (Static Display) ---
    if st.session_state.analysis_result and not st.session_state.analysis_running:
        final_state = st.session_state.analysis_result
        
        # Fill Tabs with final content
        with tab_map["final_report"]:
            report = final_state.get("final_report", "")
            if report:
                st.markdown(report)
                st.download_button("下载报告", report, "analysis_report.md", "text/markdown")
            else:
                st.info("报告生成失败")
                
        # Related Work Tab
        with tab_map["related_work_search"]:
            st.markdown(final_state.get("related_work_search", "暂无内容或未配置搜索 Key"))

        with tab_map["review_dialogue"]:
            content = final_state.get("review_dialogue", "")
            if content:
                st.markdown(content)
                st.download_button("下载讨论记录", content, "review_dialogue.md", "text/markdown")
            else:
                st.info("暂无对话记录")
            
            st.divider()
            # Post-hoc Round Table Button
            
            def start_round_table():
                st.session_state.round_table_running = True
            
            def stop_round_table():
                st.session_state.round_table_running = False
                if "stream_container" in st.session_state:
                    del st.session_state.stream_container

            if not st.session_state.round_table_running:
                st.button("🎙️ 开始圆桌讨论 (Round Table)", help="点击开启多智能体圆桌辩论，结果将实时流式显示", on_click=start_round_table)
            else:
                st.button("🛑 停止圆桌讨论", type="secondary", on_click=stop_round_table)
                
                with st.spinner("正在召集专家进行圆桌讨论..."):
                    # Setup streaming
                    st.session_state.stream_container = tab_map["review_dialogue"]
                    
                    # Prepare state
                    state_for_node = final_state.copy()
                    state_for_node["enable_round_table"] = True
                    
                    try:
                        # Run node
                        update = review_dialogue_node(state_for_node)
                        
                        # Only update if still running (not stopped by user)
                        if st.session_state.round_table_running:
                            # Update final state
                            final_state.update(update)
                            st.session_state.analysis_result = final_state
                            st.session_state.round_table_running = False
                            
                            # Cleanup
                            if "stream_container" in st.session_state:
                                del st.session_state.stream_container
                                
                            st.rerun()
                    except Exception as e:
                        st.error(f"圆桌讨论运行失败: {e}")
                        st.session_state.round_table_running = False
                        if "stream_container" in st.session_state:
                            del st.session_state.stream_container

        with tab_map["translation"]:
            st.markdown(final_state.get("translation", "暂无内容"))
            
        with tab_map["key_points"]:
            st.markdown(final_state.get("key_points", "暂无内容"))
            
        with tab_map["experiments"]:
            st.markdown(final_state.get("experiments", "暂无内容"))
            
        with tab_map["terms"]:
            st.markdown(final_state.get("terms", "暂无内容"))
            
        with tab_map["figures"]:
            figures = final_state.get("figures", [])
            if figures:
                st.write(f"共提取到 {len(figures)} 张图表")
                for img in figures:
                    st.image(img, caption=os.path.basename(img))
            else:
                st.info("未提取到图表")

        # --- Human Q&A Section ---
        # Disable Q&A during ANY running process (Analysis or Round Table)
        is_busy = st.session_state.get("round_table_running", False) or st.session_state.get("analysis_running", False)
        
        if not is_busy:
            st.markdown("---")
            st.header("💬 向论文提问")
            
            if "qa_history" not in st.session_state:
                st.session_state.qa_history = []
                
            for msg in st.session_state.qa_history:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
            
            st.text_input("请输入您的问题...", key="human_qa_input", on_change=handle_human_qa)
        else:
            st.info("💡 系统正在繁忙（分析中或圆桌讨论中），暂时无法提问。请等待当前任务结束。")


if __name__ == "__main__":
    main()
