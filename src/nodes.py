import os
import streamlit as st
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser

from src.state import AgentState
from src.loader import load_paper
from src.model_utils import get_llm, get_translation_llm, get_related_work_llm, get_review_llm
from src.prompts import (
    TRANSLATION_PROMPT,
    KEY_POINTS_PROMPT,
    EXPERIMENTS_PROMPT,
    TERMS_PROMPT,
    REPORT_PROMPT,
    GLOSSARY_PROMPT,
    FULL_TRANSLATION_PROMPT,
    RELATED_WORK_PROMPT,
    MODERATOR_PROMPT,
    CRITIC_PROMPT,
    PRACTITIONER_PROMPT,
    AUTHOR_PROMPT,
    READER_PROMPT,
    SIMPLE_AUTHOR_PROMPT
)

# Load environment variables
load_dotenv()

def get_exa_search_results(query: str) -> str:
    """Perform a web search using Exa."""
    try:
        from langchain_exa import ExaSearchResults
        
        if not os.getenv("EXA_API_KEY"):
            return None 
            
        # ExaSearchResults expects api_key param or env var EXA_API_KEY
        # Note: newer langchain-exa versions might use 'exa_api_key' arg
        tool = ExaSearchResults(num_results=3) 
        
        # The tool.invoke input format can vary. Usually string or dict.
        # For Tavily it was dict with "query". For Exa let's try similar.
        # But if it fails, we catch exception.
        results = tool.invoke(query) # Exa tool often takes just string or dict
        
        # If results is already a string, return it
        if isinstance(results, str):
            return results
            
        # If it's a list of results
        formatted_results = []
        if isinstance(results, list):
            for res in results:
                # Handle Document objects or dicts
                if hasattr(res, 'page_content'): # Document object
                    url = res.metadata.get('url') or res.metadata.get('source', 'Link')
                    formatted_results.append(f"- **{url}**: {res.page_content}")
                elif isinstance(res, dict):
                    url = res.get('url', 'Link')
                    content = res.get('text') or res.get('content') or str(res)
                    formatted_results.append(f"- **{url}**: {content}")
                else:
                    formatted_results.append(str(res))
            return "\n\n".join(formatted_results)
            
        return str(results)

    except ImportError:
        return "Exa Search dependency missing. Install with 'uv add langchain-exa'"
    except Exception as e:
        return f"Exa Search failed: {str(e)}"

def get_tavily_search_results(query: str) -> str:
    """Perform a web search using Tavily."""
    try:
        from langchain_tavily import TavilySearch
        
        if not os.getenv("TAVILY_API_KEY"):
            return "Tavily API Key not found. Cannot perform web search."
            
        tool = TavilySearch(max_results=3, search_depth="advanced")
        # TavilySearch returns a dictionary with 'results' key containing the list of results
        response = tool.invoke({"query": query})
        
        # Check if response is a dict and has 'results' key
        if isinstance(response, dict) and "results" in response:
            results = response["results"]
        else:
            # Fallback or empty if structure is unexpected
            results = []
        
        # Format results
        formatted_results = []
        for res in results:
            # Ensure res is a dictionary before accessing fields
            if isinstance(res, dict):
                formatted_results.append(f"- **{res.get('url', 'No URL')}**: {res.get('content', 'No content')}")
            
        return "\n\n".join(formatted_results)
    except ImportError:
        return "Tavily Search dependency missing. Install with 'uv add langchain-tavily'"
    except Exception as e:
        return f"Search failed: {str(e)}"

def get_serp_search_results(query: str) -> str:
    """Perform a web search using SerpAPI."""
    try:
        from langchain_community.utilities import SerpAPIWrapper
        
        if not os.getenv("SERPAPI_API_KEY"):
            return "SerpAPI Key not found. Cannot perform web search."
            
        search = SerpAPIWrapper()
        results = search.run(query)
        
        # SerpAPIWrapper.run usually returns a string directly if it finds a snippet,
        # or we might want to use .results() for structured data if run() is too simple.
        # However, run() is the standard interface. Let's return it as is or format it.
        # For better consistency with others, let's try to wrap it if it's just a string.
        
        return f"### SerpAPI Search Results\n{results}"
        
    except ImportError:
        return "SerpAPI dependency missing. Install with 'uv add google-search-results'"
    except Exception as e:
        return f"SerpAPI Search failed: {str(e)}"

# LLM Helper functions have been moved to src/model_utils.py

def load_paper_node(state: AgentState) -> AgentState:
    """Node to load paper content."""
    source = state["source"]
    use_vlm = state.get("use_vlm_parsing", False)
    try:
        text, metadata, figures = load_paper(source, use_vlm=use_vlm)
        return {"doc_content": text, "metadata": metadata, "figures": figures}
    except Exception as e:
        return {"doc_content": f"Error loading paper: {str(e)}", "metadata": {}, "figures": []}

def translate_node(state: AgentState) -> AgentState:
    """Node to translate paper content."""
    text = state.get("doc_content", "")
    if not text:
        return {"translation": "No content to translate."}
    
    # Check if full translation is requested
    is_full_translation = state.get("is_full_translation", False)
    
    if is_full_translation:
        # Full Translation Logic with Glossary Consistency
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            
            llm = get_translation_llm()
            
            # Step 1: Extract Glossary from the first part of the text (e.g., first 10k chars)
            # This ensures we capture key terms from Abstract, Intro, Method
            glossary_chain = GLOSSARY_PROMPT | llm | StrOutputParser()
            glossary = glossary_chain.invoke({"text": text[:10000]})
            
            # Step 2: Split text into chunks
            # Use a reasonable chunk size to fit in context and allow parallel processing
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=4000,
                chunk_overlap=200
            )
            chunks = text_splitter.split_text(text)
            
            # Step 3: Batch Translate with Glossary
            translation_chain = FULL_TRANSLATION_PROMPT | llm | StrOutputParser()
            
            # Prepare inputs for batch processing
            batch_inputs = [{"text": chunk, "glossary": glossary} for chunk in chunks]
            
            # Run batch translation (LangChain automatically handles parallelism if supported)
            # Adjust max_concurrency if needed to avoid rate limits
            translated_chunks = translation_chain.batch(batch_inputs, config={"max_concurrency": 5})
            
            # Step 4: Merge results
            full_translation = "\n\n".join(translated_chunks)
            
            # Prepend the glossary for reference
            final_result = f"### 术语对照表 (Glossary)\n{glossary}\n\n---\n\n### 全文翻译\n\n{full_translation}"
            
            return {"translation": final_result}
            
        except ImportError:
             return {"translation": "Error: langchain-text-splitters not installed. Please install it to use full translation."}
        except Exception as e:
             return {"translation": f"Full translation failed: {str(e)}. Partial result might be available."}

    else:
        # Default: Summary Translation
        chain = TRANSLATION_PROMPT | get_translation_llm() | StrOutputParser()
    # Increase context limit to 100k chars (approx 25k tokens) to accommodate Markdown formatting
    # Modern models (GPT-4o, Claude 3.5) support 128k+ tokens, so this is safe.
    result = chain.invoke({"text": text[:100000]}) 
    return {"translation": result}

def extract_key_points_node(state: AgentState) -> AgentState:
    """Node to extract key points."""
    text = state.get("doc_content", "")
    if not text:
        return {"key_points": "No content to analyze."}
    
    chain = KEY_POINTS_PROMPT | get_llm() | StrOutputParser()
    result = chain.invoke({"text": text[:100000]})
    return {"key_points": result}

def extract_experiments_node(state: AgentState) -> AgentState:
    """Node to extract experiments."""
    text = state.get("doc_content", "")
    if not text:
        return {"experiments": "No content to analyze."}
    
    chain = EXPERIMENTS_PROMPT | get_llm() | StrOutputParser()
    result = chain.invoke({"text": text[:100000]})
    return {"experiments": result}

def explain_terms_node(state: AgentState) -> AgentState:
    """Node to explain terms."""
    text = state.get("doc_content", "")
    if not text:
        return {"terms": "No content to analyze."}
    
    chain = TERMS_PROMPT | get_llm() | StrOutputParser()
    result = chain.invoke({"text": text[:100000]})
    return {"terms": result}

def related_work_search_node(state: AgentState) -> AgentState:
    """Node to search for related work and existing analysis."""
    metadata = state.get("metadata", {})
    title = metadata.get("Title", "")
    
    # If title is missing, try to extract from text (simple heuristic) or skip
    if not title:
        text = state.get("doc_content", "")
        if text:
            # Assume first line might be title
            title = text.split("\n")[0][:100]
        else:
            return {"related_work_search": "No title or content to search for."}
    
    search_query = f"analysis review of paper '{title}'"
    search_query_zh = f"'{title}' 论文解读 深度分析 评价"
    github_query = f"site:github.com '{title}' code implementation"
    
    combined_results = []
    
    # 1. Try Exa Search
    exa_res = get_exa_search_results(search_query)
    if exa_res and "dependency missing" not in exa_res and "Search failed" not in exa_res:
        combined_results.append(f"### Exa Search Results (English)\n{exa_res}")
    
    # Exa Chinese Search
    exa_res_zh = get_exa_search_results(search_query_zh)
    if exa_res_zh and "dependency missing" not in exa_res_zh and "Search failed" not in exa_res_zh:
         combined_results.append(f"### Exa Search Results (Chinese)\n{exa_res_zh}")

    # Exa GitHub Search
    exa_res_gh = get_exa_search_results(github_query)
    if exa_res_gh and "dependency missing" not in exa_res_gh and "Search failed" not in exa_res_gh:
         combined_results.append(f"### Exa GitHub Search Results\n{exa_res_gh}")

    # 2. Try Tavily Search
    tavily_res = get_tavily_search_results(search_query)
    if tavily_res and "Tavily API Key not found" not in tavily_res and "dependency missing" not in tavily_res and "Search failed" not in tavily_res:
        combined_results.append(f"### Tavily Search Results (English)\n{tavily_res}")
    
    # Tavily Chinese Search
    tavily_res_zh = get_tavily_search_results(search_query_zh)
    if tavily_res_zh and "Tavily API Key not found" not in tavily_res_zh and "dependency missing" not in tavily_res_zh and "Search failed" not in tavily_res_zh:
         combined_results.append(f"### Tavily Search Results (Chinese)\n{tavily_res_zh}")

    # Tavily GitHub Search
    tavily_res_gh = get_tavily_search_results(github_query)
    if tavily_res_gh and "Tavily API Key not found" not in tavily_res_gh and "dependency missing" not in tavily_res_gh and "Search failed" not in tavily_res_gh:
         combined_results.append(f"### Tavily GitHub Search Results\n{tavily_res_gh}")

    # 3. Try SerpAPI Search
    serp_res = get_serp_search_results(search_query)
    if serp_res and "SerpAPI Key not found" not in serp_res and "dependency missing" not in serp_res and "Search failed" not in serp_res:
        combined_results.append(serp_res.replace("### SerpAPI Search Results", "### SerpAPI Search Results (English)"))
        
    # SerpAPI Chinese Search
    serp_res_zh = get_serp_search_results(search_query_zh)
    if serp_res_zh and "SerpAPI Key not found" not in serp_res_zh and "dependency missing" not in serp_res_zh and "Search failed" not in serp_res_zh:
         combined_results.append(serp_res_zh.replace("### SerpAPI Search Results", "### SerpAPI Search Results (Chinese)"))

    # SerpAPI GitHub Search
    serp_res_gh = get_serp_search_results(github_query)
    if serp_res_gh and "SerpAPI Key not found" not in serp_res_gh and "dependency missing" not in serp_res_gh and "Search failed" not in serp_res_gh:
         combined_results.append(serp_res_gh.replace("### SerpAPI Search Results", "### SerpAPI GitHub Search Results"))
        
    if not combined_results:
        # Check why we failed to give better feedback
        missing_keys = []
        if not os.getenv("EXA_API_KEY"):
            missing_keys.append("Exa")
        if not os.getenv("TAVILY_API_KEY"):
             missing_keys.append("Tavily")
        if not os.getenv("SERPAPI_API_KEY"):
             missing_keys.append("SerpAPI")
        
        if len(missing_keys) == 3:
             return {"related_work_search": "No search results found. Please configure Tavily, Exa, or SerpAPI Key."}
        elif combined_results == []:
             # Keys existed but search returned nothing or failed
             return {"related_work_search": f"Search executed but returned no results. (Tavily: {str(tavily_res)[:50]}...)"}
    
    raw_search_results = "\n\n".join(combined_results)
    
    # Process results with LLM to summarize/extract
    try:
        chain = RELATED_WORK_PROMPT | get_related_work_llm() | StrOutputParser()
        processed_results = chain.invoke({
            "title": title,
            "search_results": raw_search_results
        })
        return {"related_work_search": processed_results}
    except Exception as e:
        # Fallback to raw results if LLM processing fails
        return {"related_work_search": f"Error processing search results: {str(e)}\n\nRaw Results:\n{raw_search_results}"}

def generate_report_node(state: AgentState) -> AgentState:
    """Node to generate final report."""
    chain = REPORT_PROMPT | get_llm() | StrOutputParser()
    result = chain.invoke({
        "source": state.get("source", "Unknown"),
        "translation": state.get("translation", "N/A"),
        "key_points": state.get("key_points", "N/A"),
        "experiments": state.get("experiments", "N/A"),
        "terms": state.get("terms", "N/A"),
        "related_work": state.get("related_work_search", "N/A")
    })
    return {"final_report": result}

def review_dialogue_node(state: AgentState) -> AgentState:
    """
    Node to simulate a Multi-Agent Round Table Discussion.
    Roles: Moderator, Author, Critic (Reviewer A), Practitioner (Reviewer B).
    """
    report = state.get("final_report", "")
    doc_content = state.get("doc_content", "")
    metadata = state.get("metadata", {})
    title = metadata.get("Title", "Untitled Paper")
    
    # Check if Round Table is enabled
    enable_round_table = state.get("enable_round_table", True)
    
    if not report:
        return {"review_dialogue": "无法进行对话评审：未生成最终报告。"}

    # Initialize Agents
    author_llm = get_llm()
    review_llm = get_review_llm()
    
    dialogue_history = []
    
    # Helper for streaming output to Streamlit UI
    def stream_msg(content):
        if "stream_container" in st.session_state:
            container = st.session_state.stream_container
            # Use markdown for rendering
            container.markdown(content)
            container.markdown("---")

    if enable_round_table:
        # --- Multi-Agent Round Table Mode ---
        
        # --- Phase 1: Opening ---
        # Moderator opens the session
        stream_msg("### 🟢 会议开始 (Opening)")
        moderator_input_1 = f"会议开始。请简要开场，介绍论文《{title}》的核心贡献（基于摘要），并介绍嘉宾：论文作者、方法论专家（评审员 A）和应用实践者（评审员 B）。"
        moderator_open = (MODERATOR_PROMPT | review_llm | StrOutputParser()).invoke({
            "title": title,
            "input_text": moderator_input_1,
            "status_description": "会议刚开始，需要进行开场介绍。"
        })
        msg = f"**🎓 主持人 (Moderator):**\n{moderator_open}"
        dialogue_history.append(msg)
        stream_msg(msg)
        
        # --- Phase 2: Round 1 (Methodology) ---
        # Critic (Reviewer A) asks question
        stream_msg("### 1️⃣ 第一轮：方法论探讨 (Round 1/3)")
        critic_input = f"主持人邀请你（方法论专家）发言。请基于研读报告，针对论文的理论推导、算法或实验严谨性提出一个尖锐的问题。\n\n研读报告片段：\n{report[:10000]}"
        critic_q = (CRITIC_PROMPT | review_llm | StrOutputParser()).invoke({
            "report_content": report[:10000],
            "input_text": critic_input
        })
        msg = f"**⚔️ 方法论专家 (Critic):**\n{critic_q}"
        dialogue_history.append(msg)
        stream_msg(msg)
        
        # Author answers
        author_a1 = (AUTHOR_PROMPT | author_llm | StrOutputParser()).invoke({
            "doc_content": doc_content[:50000],
            "input_text": f"方法论专家提出了质疑：{critic_q}\n请基于论文内容进行有力反驳或解释。"
        })
        msg = f"**🛡️ 论文作者 (Author):**\n{author_a1}"
        dialogue_history.append(msg)
        stream_msg(msg)
        
        # --- Phase 3: Round 2 (Practicality) ---
        # Practitioner (Reviewer B) asks question
        stream_msg("### 2️⃣ 第二轮：落地应用质疑 (Round 2/3)")
        practitioner_input = f"主持人邀请你（应用实践者）发言。作者刚刚回答了方法论问题。请基于你的视角，针对落地的成本、难度或实际价值提出质疑。\n\n研读报告片段：\n{report[:10000]}"
        practitioner_q = (PRACTITIONER_PROMPT | review_llm | StrOutputParser()).invoke({
            "report_content": report[:10000],
            "input_text": practitioner_input
        })
        msg = f"**🛠️ 应用实践者 (Practitioner):**\n{practitioner_q}"
        dialogue_history.append(msg)
        stream_msg(msg)
        
        # Author answers
        author_a2 = (AUTHOR_PROMPT | author_llm | StrOutputParser()).invoke({
            "doc_content": doc_content[:50000],
            "input_text": f"应用实践者提出了质疑：{practitioner_q}\n请基于论文内容进行回应，重点谈实际应用价值和成本。"
        })
        msg = f"**🛡️ 论文作者 (Author):**\n{author_a2}"
        dialogue_history.append(msg)
        stream_msg(msg)
        
        # --- Phase 4: Round 3 (Follow-up / Deep Dive) ---
        # Moderator selects a follow-up
        stream_msg("### 3️⃣ 第三轮：深度追问与总结 (Round 3/3)")
        moderator_input_2 = f"前两轮已结束。\n方法论专家问了：{critic_q}\n应用实践者问了：{practitioner_q}\n\n请总结争议点，并指定其中一位评审员（专家或实践者）进行深入追问。"
        moderator_followup_inst = (MODERATOR_PROMPT | review_llm | StrOutputParser()).invoke({
            "title": title,
            "input_text": moderator_input_2,
            "status_description": "进入自由辩论环节，需要指定一位评审员追问。"
        })
        msg = f"**🎓 主持人 (Moderator):**\n{moderator_followup_inst}"
        dialogue_history.append(msg)
        stream_msg(msg)
        
        # Critic asks final tough question
        critic_input_2 = f"主持人让你追问。作者之前的回答如下：\n1. {author_a1}\n2. {author_a2}\n\n请抓住其中一个逻辑漏洞或模糊点，进行终极追问。"
        critic_q2 = (CRITIC_PROMPT | review_llm | StrOutputParser()).invoke({
            "report_content": report[:10000],
            "input_text": critic_input_2
        })
        msg = f"**⚔️ 方法论专家 (Critic - 追问):**\n{critic_q2}"
        dialogue_history.append(msg)
        stream_msg(msg)
        
        # Author final response
        author_a3 = (AUTHOR_PROMPT | author_llm | StrOutputParser()).invoke({
            "doc_content": doc_content[:50000],
            "input_text": f"方法论专家进行了追问：{critic_q2}\n这是最后的回应机会，请做出精彩的总结性回答。"
        })
        msg = f"**🛡️ 论文作者 (Author):**\n{author_a3}"
        dialogue_history.append(msg)
        stream_msg(msg)
        
        # --- Phase 5: Closing ---
        # Moderator summary
        stream_msg("### 🏁 会议结束 (Closing)")
        moderator_input_3 = f"辩论结束。作者最后的回答是：{author_a3}\n\n请综合各方观点，对论文进行多维度技术总结（如创新点、工程可行性、算法完备性），并给出最终的“技术推荐等级”（如：强烈推荐、值得尝试、仅供参考）。"
        moderator_close = (MODERATOR_PROMPT | review_llm | StrOutputParser()).invoke({
            "title": title,
            "input_text": moderator_input_3,
            "status_description": "会议结束，需要进行总结和打分。"
        })
        msg = f"**🎓 主持人 (Moderator - 总结):**\n{moderator_close}"
        dialogue_history.append(msg)
        stream_msg(msg)

    else:
        # --- Fallback: Simple Reader-Author Dialogue ---
        
        # --- Round 1 ---
        stream_msg("### 1️⃣ 第一轮问答 (Round 1/5)")
        reader_input_1 = f"我已经阅读了这份关于论文的报告。请基于报告内容，提出你最想问作者的一个核心问题，或者指出你觉得最难理解的一个概念。\n\n报告内容：\n{report[:10000]}"
        reader_q1 = (READER_PROMPT | review_llm | StrOutputParser()).invoke({"input_text": reader_input_1})
        msg = f"**👤 Reader (Q1):**\n{reader_q1}"
        dialogue_history.append(msg)
        stream_msg(msg)
        
        author_a1 = (SIMPLE_AUTHOR_PROMPT | author_llm | StrOutputParser()).invoke({
            "doc_content": doc_content[:50000],
            "input_text": f"读者提问：{reader_q1}"
        })
        msg = f"**🎓 Author (A1):**\n{author_a1}"
        dialogue_history.append(msg)
        stream_msg(msg)
        
        # --- Round 2 ---
        stream_msg("### 2️⃣ 第二轮问答 (Round 2/5)")
        reader_input_2 = f"作者刚刚回答了你的第一个问题。\n作者回答：{author_a1}\n\n请基于此追问一个更深入或具体的问题。"
        reader_q2 = (READER_PROMPT | review_llm | StrOutputParser()).invoke({"input_text": reader_input_2})
        msg = f"**👤 Reader (Q2):**\n{reader_q2}"
        dialogue_history.append(msg)
        stream_msg(msg)
        
        author_a2 = (SIMPLE_AUTHOR_PROMPT | author_llm | StrOutputParser()).invoke({
            "doc_content": doc_content[:50000],
            "input_text": f"读者追问：{reader_q2}"
        })
        msg = f"**🎓 Author (A2):**\n{author_a2}"
        dialogue_history.append(msg)
        stream_msg(msg)

        # --- Round 3 ---
        stream_msg("### 3️⃣ 第三轮问答 (Round 3/5)")
        reader_input_3 = f"作者刚刚回答了你的第二个问题。\n作者回答：{author_a2}\n\n请基于此继续追问，或者询问该研究的局限性/应用场景。"
        reader_q3 = (READER_PROMPT | review_llm | StrOutputParser()).invoke({"input_text": reader_input_3})
        msg = f"**👤 Reader (Q3):**\n{reader_q3}"
        dialogue_history.append(msg)
        stream_msg(msg)
        
        author_a3 = (SIMPLE_AUTHOR_PROMPT | author_llm | StrOutputParser()).invoke({
            "doc_content": doc_content[:50000],
            "input_text": f"读者追问：{reader_q3}"
        })
        msg = f"**🎓 Author (A3):**\n{author_a3}"
        dialogue_history.append(msg)
        stream_msg(msg)

        # --- Round 4 ---
        stream_msg("### 4️⃣ 第四轮问答 (Round 4/5)")
        reader_input_4 = f"作者刚刚回答了你的第三个问题。\n作者回答：{author_a3}\n\n请基于此继续追问，例如关于未来发展方向或者潜在的缺陷。"
        reader_q4 = (READER_PROMPT | review_llm | StrOutputParser()).invoke({"input_text": reader_input_4})
        msg = f"**👤 Reader (Q4):**\n{reader_q4}"
        dialogue_history.append(msg)
        stream_msg(msg)
        
        author_a4 = (SIMPLE_AUTHOR_PROMPT | author_llm | StrOutputParser()).invoke({
            "doc_content": doc_content[:50000],
            "input_text": f"读者追问：{reader_q4}"
        })
        msg = f"**🎓 Author (A4):**\n{author_a4}"
        dialogue_history.append(msg)
        stream_msg(msg)
        
        # --- Round 5 ---
        stream_msg("### 5️⃣ 最终点评 (Round 5/5)")
        reader_input_5 = f"作者已经回答了你的所有问题。\n作者回答：{author_a4}\n\n请总结你对这篇论文的理解，并对这份报告的易读性（1-10分）和论文的启发性（1-10分）进行打分和点评。"
        reader_feedback = (READER_PROMPT | review_llm | StrOutputParser()).invoke({"input_text": reader_input_5})
        msg = f"**👤 Reader (Final Feedback):**\n{reader_feedback}"
        dialogue_history.append(msg)
        stream_msg(msg)
    
    # Format the full dialogue
    formatted_dialogue = "\n\n---\n\n".join(dialogue_history)
    
    return {"review_dialogue": formatted_dialogue}
