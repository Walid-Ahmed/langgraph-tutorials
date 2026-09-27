from typing import TypedDict, List


class AgentState(TypedDict):
    question: str
    refined_question: str        # planner rewrites for clarity
    doc_sections: List[str]      # relevant sections from local text file
    web_results: str             # raw Google Serper search results
    synthesis: str               # combined doc + web analysis
    final_report: str            # polished final answer
    session_id: str
    steps_taken: List[str]       # audit trail of nodes that ran
