from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from .base import BaseSchema

class Identity(BaseSchema):
    """
    Defines the characteristics of the persona or context for an ethical scenario.
    Represents user profiles, demographic/cultural contexts, or fictional roles.
    """
    id: str = Field(..., description="A unique name or key for the identity.")
    description: str = Field(..., description="Description of this identity context (background, values, specifics).")
    attributes: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Set of attribute keys detailing the identity (e.g., age_group, culture).")
    moral_preferences: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Specific moral or ethical inclinations of this identity (optional).")
    notes: Optional[str] = Field(None, description="Additional notes or guidance for using this identity.")

    class Config:
        extra = 'forbid' # Disallow extra fields not defined in the schema
