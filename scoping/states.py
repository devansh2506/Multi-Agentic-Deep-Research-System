import operator
from typing_extensions import Optional, Annotated, List, Sequence

from langchain_core.messages import BaseMessage
from langgraph.graph import MessagesState
from langgraph.graph.message import add_messages

class AgentInputState(MessagesState):
    pass

class AgentState(MessagesState):

    # Research brief generated from user conversation history
    research_brief: Optional[str]
    # Number of times the agent has asked a clarifying question
    clarification_count: Annotated[int, operator.add]