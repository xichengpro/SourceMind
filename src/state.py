from typing import TypedDict, Optional, Dict, Any

class AgentState(TypedDict):
    source: str
    doc_content: str
    metadata: Dict[str, Any]
    translation: Optional[str]
    key_points: Optional[str]
    experiments: Optional[str]
    terms: Optional[str]
    related_work_search: Optional[str]
    final_report: Optional[str]
    is_full_translation: Optional[bool]
    use_vlm_parsing: Optional[bool] # Toggle for VLM-based PDF parsing
    figures: Optional[list[str]] # List of paths to extracted figure images
    review_dialogue: Optional[str] # Dialogue between reader and author models
    enable_round_table: Optional[bool] # Toggle for Multi-Agent Round Table Discussion
