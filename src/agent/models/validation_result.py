from typing import Optional

from pydantic import BaseModel, Field

class ValidationResult(BaseModel):
    """Results from the validation of the input topic and languages."""

    is_valid: bool = Field(description="Whether the input is valid or not")
    reason_for_failed_validation: Optional[str] = Field(None, description="A string containing the reasons why the validation is invalid")