from typing import Optional
from pydantic import BaseModel, Field


class ClarifyWithUser(BaseModel):
    """Schema for user clarification decision and questions."""

    need_clarification: bool = Field(
        description="Whether the user needs to be asked a clarifying question.",
    )
    question: Optional[str] = Field(
        default="",
        description="A question to ask the user to clarify the report scope",
    )



class ResearchQuestion(BaseModel):
    """Schema for structured research brief generation."""

    research_brief: str = Field(
        description="A research question that will be used to guide the research.",
    )