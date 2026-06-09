from pydantic import BaseModel
from typing import Optional

class ChatActionRequest(BaseModel):
    session_id: str           
    action_type: str     
    message: Optional[str] = None
    code: Optional[str] = None
    problem_description: Optional[str] = None