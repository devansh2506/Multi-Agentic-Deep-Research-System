import os
from datetime import datetime
from typing_extensions import Literal

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage, get_buffer_string
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command

from scoping.states import AgentState
from scoping.output_schemas import ClarifyWithUser, ResearchQuestion
from scoping.prompts import clarify_with_user_instructions, transform_messages_into_research_topic_prompt


# ===== UTILITY FUNCTIONS =====

def get_today_str() -> str:
    day = str(datetime.now().day)
    return datetime.now().strftime(f"%a %b {day}, %Y")


# ===== CONFIGURATION =====


# 2. Initialize Groq Cloud using Llama 3.3 (blazing fast + handles structured output perfectly)
model = init_chat_model(model="llama-3.3-70b-versatile", model_provider="groq", temperature=0.0)


# ===== WORKFLOW NODES =====

def clarify_with_user(state: AgentState) -> Command[Literal["write_research_brief", "__end__"]]:
    """
    Determine if the user's request contains sufficient information to proceed with research.
    """
    structured_output_model = model.with_structured_output(ClarifyWithUser, method="json_mode")

    response = structured_output_model.invoke([
        HumanMessage(content=clarify_with_user_instructions.format(
            messages=get_buffer_string(state["messages"]),
            date=get_today_str()
        ))
    ])

    if response.need_clarification:
        current_count = state.get("clarification_count", 0)
        
        # If the agent has already asked 5 questions, force the brief generation
        if current_count >= 5:
            return Command(
                goto="write_research_brief"
            )
            
        return Command(
            goto=END,
            update={
                "messages": [AIMessage(content=response.question)],
                "clarification_count": 1
            }
        )
    else:
        return Command(
            goto="write_research_brief"
        )


def write_research_brief(state: AgentState):
    """
    Transform the conversation history into a comprehensive research brief.
    """
    structured_output_model = model.with_structured_output(ResearchQuestion, method="json_mode")

    response = structured_output_model.invoke([
        HumanMessage(content=transform_messages_into_research_topic_prompt.format(
            messages=get_buffer_string(state["messages"]),
            date=get_today_str()
        ))
    ])

    return {
        "research_brief": response.research_brief
    }