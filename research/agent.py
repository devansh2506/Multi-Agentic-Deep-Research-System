import os
from pydantic import BaseModel, Field
from typing_extensions import Literal

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, filter_messages
from langchain.chat_models import init_chat_model
from tavily import TavilyClient
from langgraph.checkpoint.memory import MemorySaver

from research.tools import tavily_search, think_tool
from research.prompts import (
    research_agent_prompt,
    compress_research_system_prompt
)
from research.states import ResearcherState, ResearcherOutputState
from research.functions import get_today_str

# ===== CONFIGURATION =====



# 2. FIXED: Re-instantiate global tool lists and registry maps for the nodes to read
tools = [tavily_search, think_tool]
tools_by_name = {tool.name: tool for tool in tools}

# 3. Flagship Orchestration Models (Keep the heavy 70B model for complex strategic reasoning)
model = init_chat_model(model="gemini-2.5-flash", model_provider="google_genai", temperature=0.0)
model_with_tools = model.bind_tools(tools)  # Cleanly binds using our initialized tools list

# Gemini 1.5 Flash has a 1 million token context window, perfect for compressing massive amounts of research notes!
compress_model = init_chat_model(model="gemini-1.5-flash", model_provider="google_genai", temperature=0.0)

# 4. Updated the decommissioned 8B model string to the active 'llama-3.1-8b-instant' architecture
summarization_model = init_chat_model(model="gemini-2.5-flash", model_provider="google_genai", temperature=0.0)

tavily_client = TavilyClient()
print("✅ Configuration updated! Tool lookup maps generated and successfully switched to llama-3.1-8b-instant.")


# ===== AGENT NODES =====

def llm_call(state: ResearcherState):
    """Analyze current state and decide on next actions.

    The model analyzes the current conversation state and decides whether to:
    1. Call search tools to gather more information
    2. Provide a final answer based on gathered information

    Returns updated state with the model's response.
    """
    # noinspection PyTypeChecker
    return {
        "researcher_messages": [
            model_with_tools.invoke(
                [SystemMessage(content=research_agent_prompt.format(date=get_today_str()))] + state["researcher_messages"]
            )
        ]
    }


def tool_node(state: ResearcherState):
    """Execute all tool calls from the previous LLM response.

    Executes all tool calls from the previous LLM responses.
    Returns updated state with tool execution results and increments iteration counter.
    """
    tool_calls = state["researcher_messages"][-1].tool_calls

    # Execute all tool calls
    observations = []
    for tool_call in tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observations.append(tool.invoke(tool_call["args"]))

    # Create tool message outputs
    tool_outputs = [
        ToolMessage(
            content=observation,
            name=tool_call["name"],
            tool_call_id=tool_call["id"]
        ) for observation, tool_call in zip(observations, tool_calls)
    ]

    # FIXED: Increment the loop iteration counter to prevent infinite search loops
    current_iterations = state.get("tool_call_iterations", 0)

    return {
        "researcher_messages": tool_outputs,
        "tool_call_iterations": current_iterations + 1
    }


def compress_research(state: ResearcherState) -> dict:
    """Compress research findings into a concise summary.

    Takes all the research messages and tool outputs and create
    a compressed summary suitable for the supervisor's decision-making.
    """
    system_message = compress_research_system_prompt.format(date=get_today_str())

    messages = (
            [SystemMessage(content=system_message)]
            + state.get("researcher_messages", [])
    )

    response = compress_model.invoke(messages)

    # Extract raw notes from tool and AI messages
    raw_notes = [
        str(m.content) for m in filter_messages(
            state["researcher_messages"],
            include_types=["tool", "ai"]
        )
    ]

    return {
        "compressed_research": str(response.content),
        "raw_notes": ["\n".join(raw_notes)]
    }


# ===== ROUTING LOGIC =====

def should_continue(state: ResearcherState) -> Literal["tool_node", "compress_research"]:
    """Determine whether to continue research or provide final answer.

    Determines whether the agent should continue the research loop or provide
    a final answer based on whether the LLM made tool calls or hit the safety limit.
    """
    # FIXED: Hard programmatic stop if the agent hits the 5 loop limit budget
    if state.get("tool_call_iterations", 0) >= 5:
        print("Programmatic Budget Limit Reached: Forcing compression step.")
        return "compress_research"

    messages = state["researcher_messages"]
    last_message = messages[-1]

    # If the LLM makes a tool call, continue to tool execution
    if last_message.tool_calls:
        return "tool_node"

    # Otherwise, we have a final answer
    return "compress_research"


# ===== GRAPH CONSTRUCTION =====

# Build the agent workflow
agent_builder = StateGraph(ResearcherState, output_schema=ResearcherOutputState)

# Add nodes to the graph
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)
agent_builder.add_node("compress_research", compress_research)

# Add edges to connect nodes
agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    {
        "tool_node": "tool_node",  # Continue research loop
        "compress_research": "compress_research",  # Provide final answer
    },
)
agent_builder.add_edge("tool_node", "llm_call")  # Loop back for more research
agent_builder.add_edge("compress_research", END)

# Compile the agent with memory
memory = MemorySaver()
researcher_agent = agent_builder.compile(checkpointer=memory)
print("Researcher graph successfully built and compiled using Gemini models!")