from typing import List
from fastapi.responses import StreamingResponse
from services.chatbot.chatbot_ai_service import AI_Chatbot_Service
from fastapi import APIRouter
from schemas.api_schema import ChatLogicInputData
from pydantic import BaseModel
from schemas.document_basemodel import RelevantDocumentBaseModel
chatbot_router = APIRouter()

@chatbot_router.post("/chat")
async def send_message(user_data: ChatLogicInputData):
    ai_chatbot_service = AI_Chatbot_Service(
        model=user_data.model,
        thinking=user_data.thinking
    )
    
    async def stream_response():
        async for chunk in ai_chatbot_service.create_response(user_data=user_data):
            yield chunk
            print(chunk)
            print("AAAAAAAAAAAA")
    return StreamingResponse(stream_response(), media_type="application/json")

class ExtractReferencesRequest(BaseModel):
    relevant_documents: List[RelevantDocumentBaseModel]
    question: str


@chatbot_router.post("/extract_references")
async def extract_references(request: ExtractReferencesRequest):
    ai_chatbot_service = AI_Chatbot_Service(
        model="gemini-2.0-flash-lite",
    )
    extract_references = await ai_chatbot_service.create_references(
        relevant_documents=request.relevant_documents,
        question=request.question
    )
    return extract_references