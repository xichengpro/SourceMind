from langgraph.graph import StateGraph, END

from src.state import AgentState
from src.nodes import (
    load_paper_node,
    translate_node,
    extract_key_points_node,
    extract_experiments_node,
    explain_terms_node,
    related_work_search_node,
    generate_report_node,
    review_dialogue_node
)

def create_graph():
    # Initialize the graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("load_paper", load_paper_node)
    workflow.add_node("translate", translate_node)
    workflow.add_node("extract_key_points", extract_key_points_node)
    workflow.add_node("extract_experiments", extract_experiments_node)
    workflow.add_node("explain_terms", explain_terms_node)
    workflow.add_node("related_work_search", related_work_search_node)
    workflow.add_node("generate_report", generate_report_node)
    workflow.add_node("review_dialogue", review_dialogue_node)
    
    # Define edges
    # Start -> Load
    workflow.set_entry_point("load_paper")
    
    # Load -> Parallel Analysis Nodes
    workflow.add_edge("load_paper", "translate")
    workflow.add_edge("load_paper", "extract_key_points")
    workflow.add_edge("load_paper", "extract_experiments")
    workflow.add_edge("load_paper", "explain_terms")
    workflow.add_edge("load_paper", "related_work_search")
    
    # Analysis Nodes -> Report
    # We need to make sure 'generate_report' waits for all analysis nodes.
    # In LangGraph, if a node has multiple incoming edges, it might be triggered multiple times 
    # OR we need to synchronize. 
    # However, the standard way to fan-out and fan-in is usually implicit or requires a barrier.
    # But here, let's just connect all of them to report.
    # NOTE: In current LangGraph, if we just add edges, the next node might run as soon as ONE is ready 
    # if not properly handled, or it waits for all if it's a join.
    # Actually, the best way for a "Fan-In" in LangGraph is to let them all go to the next step.
    # But to ensure 'generate_report' receives ALL updates, we usually rely on the fact that
    # the state is shared. But we need to ensure flow control.
    
    # Simple approach: Linearize or explicit fan-in. 
    # Since LangGraph execution is step-based.
    # Let's try connecting all to generate_report.
    
    workflow.add_edge("translate", "generate_report")
    workflow.add_edge("extract_key_points", "generate_report")
    workflow.add_edge("extract_experiments", "generate_report")
    workflow.add_edge("explain_terms", "generate_report")
    workflow.add_edge("related_work_search", "generate_report")
    
    # Report -> Review Dialogue
    workflow.add_edge("generate_report", "review_dialogue")
    
    # Review Dialogue -> End
    workflow.add_edge("review_dialogue", END)
    
    # Compile the graph
    app = workflow.compile()
    return app
