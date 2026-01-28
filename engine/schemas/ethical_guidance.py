from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from .base import BaseSchema

class EthicalGuidance(BaseSchema):
    """
    Specifies the moral reasoning framework or set of ethical principles
    that the AI agent should follow. Acts as the 'moral compass'.
    """
    id: str = Field(..., description="Name of the ethical framework or guidance profile (e.g., Utilitarian, Deontological).")
    description: str = Field(..., description="Concise summary of the guidance or framework.")
    principles: Optional[List[str]] = Field(default_factory=list, description="Specific ethical principles emphasized (e.g., beneficence, autonomy).")
    prompt_template: Optional[str] = Field(None, description="System message template to prime the LLM with this ethical stance.")
    evaluation_focus: Optional[List[str]] = Field(default_factory=list, description="Aspects to scrutinize during evaluation under this guidance.")

    class Config:
        extra = 'forbid'
