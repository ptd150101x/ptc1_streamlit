from enum import Enum
from pydantic import BaseModel, Extra
from typing import Optional, List, Dict

class ChatMessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "model"
    SYSTEM = "system"


class ChatMessage(BaseModel):
    role: ChatMessageRole
    content: str



class ChatLogicInputData(BaseModel, extra=Extra.forbid):
    user_id: Optional[str] = None
    user_name: str = ""
    thread_id: Optional[str] = None
    model: str = "gemini-2.0-flash-lite"
    content: str = ""
    histories: List[ChatMessage] = []
    summary: str = ""
    language: str = ""
    timestamp: Optional[int] = None
    metadata: Dict = {}
    customer_id: str = "ptc1"
    thinking: bool = False
    category: str = ""