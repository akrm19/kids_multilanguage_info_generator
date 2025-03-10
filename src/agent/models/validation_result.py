from typing import Optional

from pydantic import BaseModel, Field

class ValidationResult(BaseModel):
    is_valid: bool = Field(description="Whether the input is valid or not")
    message: Optional[str] = Field(None, description="A message describing the validation result")