from langgraph.graph import StateGraph, START, END

from scoping.states import AgentState, AgentInputState
from scoping.scoping import clarify_with_user, write_research_brief

# ===== GRAPH CONSTRUCTION =====

# Build the scoping workflow
deep_researcher_builder = StateGraph(AgentState, input_schema=AgentInputState)

# Add workflow nodes
deep_researcher_builder.add_node("clarify_with_user", clarify_with_user)
deep_researcher_builder.add_node("write_research_brief", write_research_brief)

# Add workflow edges
deep_researcher_builder.add_edge(START, "clarify_with_user")
deep_researcher_builder.add_edge("write_research_brief", END)

from langgraph.checkpoint.memory import MemorySaver

# Compile the workflow with memory checkpointer
memory = MemorySaver()
scope_research = deep_researcher_builder.compile(checkpointer=memory, name="Scoping_Agent")