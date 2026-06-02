from pydantic import BaseModel, Field
from typing_extensions import Optional


# ===== STRUCTURED OUTPUT SCHEMAS =====

class Summary(BaseModel):
    """Schema for webpage content summarization."""

    summary: Optional[str] = Field(default="", description="Concise summary of the webpage content")
    key_excerpts: Optional[str] = Field(default="", description="Important quotes and excerpts from the content")