from services.chatbot.generator import GeminiGenerator
from services.chatbot.gemini_api_keys import GEMINI_API_KEYS
from services.chatbot.chat_generator import GeminiChatGenerator
from services.chatbot.detect_language import DetectLanguage
from services.chatbot.document_retriever import DocumentRetriever
from services.chatbot.single_query import SingleQuery
from services.chatbot.answer_generator import AnswerGenerator
from services.chatbot.extract_references import ExtractReferences
from langfuse.decorators import observe
from schemas.api_schema import ChatLogicInputData
from utils.monitor_log import logger
from utils.default_response import OVERLOAD_MESSAGE
import traceback
from utils.connect_langfuse import connect_langfuse
from models.database import get_db
from typing import List, AsyncGenerator
from schemas.document import RelevantDocument

connect_langfuse()

class AI_Chatbot_Service:
    def __init__(self,
                model: str = "gemini-2.0-flash-lite",
                thinking: bool = False
                ):
        self.thinking = thinking
        self.gemini_generator = GeminiGenerator(
            api_keys=GEMINI_API_KEYS,
            model=model if model != "gemini-2.0-flash-thinking-exp-01-21" else "gemini-2.0-flash-lite"
        )
        self.gemini_chat_generator = GeminiChatGenerator(
            api_keys=GEMINI_API_KEYS,
            model=model
        )
        self.detect_language = DetectLanguage(
            generator=self.gemini_generator
        )
        self.document_retriever = DocumentRetriever(session=next(get_db()))
        self.single_query = SingleQuery(
            generator=self.gemini_generator
        )
        self.answer_generator = AnswerGenerator(
            chat_generator=self.gemini_chat_generator
        )
        self.extract_references = ExtractReferences(
            generator=self.gemini_generator
        )


    @observe(name="AI_Chatbot_Service_Answer", as_type="service")
    async def create_response(self,
                            user_data: ChatLogicInputData,
                            ) -> AsyncGenerator[str, None]:
        try:
            user_message = user_data.content
            category = user_data.category
            summary_history = user_data.summary
            customer_id = user_data.customer_id
            language_response = await self.detect_language.run(
                question=user_message
            )
            language = language_response.get("language", "")

            relevant_documents = []
            seen_ids = set()
            single_query = (await self.single_query.run(
                user_data=user_data,
                question=user_message,
                )).get("rewrite_prompt", "")
            print(single_query)
            if single_query:
                child_prompts = [single_query]
            else:
                child_prompts = [user_message]

            # for query in child_prompts:
            #     documents = self.document_retriever.run(
            #         query_text=query,
            #         threshold=0.2,
            #         category=category
            #         )
            #     for doc in documents['final_rerank']:
            #         if doc['id'] not in seen_ids:
            #             relevant_documents.append(doc)
            #             seen_ids.add(doc['id'])

            # if not relevant_documents:
            #     documents = self.document_retriever.run(
            #         query_text=user_message,
            #         threshold=0.2,
            #         category=category
            #         )
            #     for doc in documents['final_rerank']:
            #         if doc['id'] not in seen_ids:
            #             relevant_documents.append(doc)
            #             seen_ids.add(doc['id'])

            # relevant_documents = sorted(relevant_documents, key=lambda doc: doc['cross_score'], reverse=False) if relevant_documents else []

            async for chunk in self.answer_generator.run(
                messages=user_data.histories,
                relevant_documents=relevant_documents,
                summary_history=summary_history,
                original_query=user_message,
                language=language,
                thinking=self.thinking
            ):
                yield chunk
            return
        except Exception as e:
            logger.error(f"Error in main create_response: {e}")
            logger.error(f"Traceback:\n{traceback.format_exc()}")
            logger.info("No data found")
            yield {
                "status": 500,
                "answer": OVERLOAD_MESSAGE,
                "references": relevant_documents
            }
    @observe(name="AI_Chatbot_Service_References", as_type="service")
    async def create_references(self,
                        relevant_documents: List[RelevantDocument],
                        question: str,
                        ):
        return await self.extract_references.run(
            relevant_documents=relevant_documents,
            question=question
        )
