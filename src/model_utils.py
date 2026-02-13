import os
import streamlit as st
from langchain_openai import ChatOpenAI

def get_llm():
    """Get LLM instance based on environment configuration."""
    provider = os.getenv("LLM_PROVIDER", "OpenAI")
    
    # Check for Langfuse callback
    callbacks = []
    if "langfuse_handler" in st.session_state:
        callbacks.append(st.session_state.langfuse_handler)
    
    if provider == "Anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError:
             raise ImportError(
                "langchain-anthropic is not installed. "
                "Please install it using 'pip install langchain-anthropic' or 'uv add langchain-anthropic'"
            )
        model_name = os.getenv("ANTHROPIC_MODEL_NAME", "claude-3-5-sonnet-20240620")
        api_key = os.getenv("ANTHROPIC_API_KEY")
        return ChatAnthropic(model=model_name, temperature=0, api_key=api_key, callbacks=callbacks)
        
    elif provider in ["OpenAI", "OpenRouter", "自定义 (OpenAI 兼容)"]:
        model_name = os.getenv("OPENAI_MODEL_NAME", "gpt-4o")
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_API_BASE")
        
        return ChatOpenAI(
            model=model_name, 
            temperature=0, 
            api_key=api_key,
            base_url=base_url,
            callbacks=callbacks
        )
    
    else:
        # Default fallback
        return ChatOpenAI(model="gpt-4o", temperature=0, callbacks=callbacks)

def get_translation_llm():
    """Get LLM instance for translation tasks, potentially using a dedicated configuration."""
    # Check if a dedicated translation provider is configured
    provider = os.getenv("TRANSLATION_LLM_PROVIDER")
    
    # Check for Langfuse callback
    callbacks = []
    if "langfuse_handler" in st.session_state:
        callbacks.append(st.session_state.langfuse_handler)
    
    # If not configured, fallback to the main LLM
    if not provider or provider == "None":
        return get_llm()
        
    if provider == "Anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError:
             raise ImportError(
                "langchain-anthropic is not installed."
            )
        model_name = os.getenv("TRANSLATION_ANTHROPIC_MODEL_NAME", "claude-3-sonnet-20240229")
        api_key = os.getenv("TRANSLATION_ANTHROPIC_API_KEY")
        return ChatAnthropic(model=model_name, temperature=0, api_key=api_key, callbacks=callbacks)
        
    elif provider in ["OpenAI", "OpenRouter", "自定义 (OpenAI 兼容)"]:
        model_name = os.getenv("TRANSLATION_OPENAI_MODEL_NAME", "gpt-4o")
        api_key = os.getenv("TRANSLATION_OPENAI_API_KEY")
        base_url = os.getenv("TRANSLATION_OPENAI_API_BASE")
        
        return ChatOpenAI(
            model=model_name, 
            temperature=0, 
            api_key=api_key,
            base_url=base_url,
            callbacks=callbacks
        )
    
    return get_llm()

def get_review_llm():
    """Get LLM instance for dialogue review tasks."""
    # Check if a dedicated review provider is configured
    provider = os.getenv("REVIEW_LLM_PROVIDER")
    
    # Check for Langfuse callback
    callbacks = []
    if "langfuse_handler" in st.session_state:
        callbacks.append(st.session_state.langfuse_handler)
    
    # If not configured, fallback to the main LLM
    if not provider or provider == "None":
        return get_llm()
        
    if provider == "Anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError:
             raise ImportError(
                "langchain-anthropic is not installed."
            )
        model_name = os.getenv("REVIEW_ANTHROPIC_MODEL_NAME", "claude-3-5-sonnet-20240620")
        api_key = os.getenv("REVIEW_ANTHROPIC_API_KEY")
        return ChatAnthropic(model=model_name, temperature=0, api_key=api_key, callbacks=callbacks)
        
    elif provider in ["OpenAI", "OpenRouter", "自定义 (OpenAI 兼容)"]:
        model_name = os.getenv("REVIEW_OPENAI_MODEL_NAME", "gpt-4o")
        api_key = os.getenv("REVIEW_OPENAI_API_KEY")
        base_url = os.getenv("REVIEW_OPENAI_API_BASE")
        
        return ChatOpenAI(
            model=model_name, 
            temperature=0, 
            api_key=api_key,
            base_url=base_url,
            callbacks=callbacks
        )
    
    return get_llm()

def get_vlm_llm():
    """Get LLM instance for VLM (Visual Language Model) tasks."""
    # Check if a dedicated VLM provider is configured
    provider = os.getenv("VLM_LLM_PROVIDER")
    
    # Check for Langfuse callback
    callbacks = []
    if "langfuse_handler" in st.session_state:
        callbacks.append(st.session_state.langfuse_handler)
    
    # If not configured, fallback to the main LLM
    if not provider or provider == "None":
        return get_llm()
        
    if provider == "Anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError:
             raise ImportError(
                "langchain-anthropic is not installed."
            )
        model_name = os.getenv("VLM_ANTHROPIC_MODEL_NAME", "claude-3-5-sonnet-20240620")
        api_key = os.getenv("VLM_ANTHROPIC_API_KEY")
        return ChatAnthropic(model=model_name, temperature=0, api_key=api_key, callbacks=callbacks)
        
    elif provider in ["OpenAI", "OpenRouter", "自定义 (OpenAI 兼容)"]:
        model_name = os.getenv("VLM_OPENAI_MODEL_NAME", "gpt-4o")
        api_key = os.getenv("VLM_OPENAI_API_KEY")
        base_url = os.getenv("VLM_OPENAI_API_BASE")
        
        return ChatOpenAI(
            model=model_name, 
            temperature=0, 
            api_key=api_key,
            base_url=base_url,
            callbacks=callbacks
        )
    
    return get_llm()

def get_related_work_llm():
    """Get LLM instance for related work processing tasks."""
    # Check if a dedicated related work provider is configured
    provider = os.getenv("RELATED_WORK_LLM_PROVIDER")
    
    # Check for Langfuse callback
    callbacks = []
    if "langfuse_handler" in st.session_state:
        callbacks.append(st.session_state.langfuse_handler)
    
    # If not configured, fallback to the main LLM
    if not provider or provider == "None":
        return get_llm()
        
    if provider == "Anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError:
             raise ImportError(
                "langchain-anthropic is not installed."
            )
        model_name = os.getenv("RELATED_WORK_ANTHROPIC_MODEL_NAME", "claude-3-sonnet-20240229")
        api_key = os.getenv("RELATED_WORK_ANTHROPIC_API_KEY")
        return ChatAnthropic(model=model_name, temperature=0, api_key=api_key, callbacks=callbacks)
        
    elif provider in ["OpenAI", "OpenRouter", "自定义 (OpenAI 兼容)"]:
        model_name = os.getenv("RELATED_WORK_OPENAI_MODEL_NAME", "gpt-4o")
        api_key = os.getenv("RELATED_WORK_OPENAI_API_KEY")
        base_url = os.getenv("RELATED_WORK_OPENAI_API_BASE")
        
        return ChatOpenAI(
            model=model_name, 
            temperature=0, 
            api_key=api_key,
            base_url=base_url,
            callbacks=callbacks
        )
    
    return get_llm()
