from pydantic import BaseModel
from typing import Optional

class BaseIssue(BaseModel):
    type: str
    file: Optional[str] = None

class GeminiIssue(BaseModel):
    severity: str = "Medium"
    impact: str
    solution: str

class EnrichedIssue(BaseIssue, GeminiIssue):
    weight: int
