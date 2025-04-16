import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from langfuse.decorators import observe
from schemas.api_schema import ChatMessage
from typing import List
from schemas.document import RelevantDocument
from services.chatbot.chat_generator import ChatGenerator
from datetime import datetime
import pytz
from pydantic import BaseModel, Field
from typing import List, AsyncGenerator
import asyncio



# Đặt múi giờ thành múi giờ Việt Nam
timezone = pytz.timezone('Asia/Ho_Chi_Minh')

class ChatResponseWithContext(BaseModel):
    reasoning_answer: str = Field(
        description="Dựa vào tài liệu được cung cấp và bối cảnh trò chuyện, Viết các phân tích của bạn, giải thích tại sao bạn lại trả lời như vậy"
    )
    final_answer: str = Field(
        description="Đưa ra câu trả lời logic, tự nhiên và sâu sắc. Trả lời khéo léo nếu không có đủ thông tin để trả lời đầu vào của người dùng"
    )


system_prompt_template_with_context = """
# NGÀY VÀ GIỜ
Ngày và giờ hiện tại là {current_time}.

# VAI TRÒ CỦA BẠN
- Bạn là một nhân viên chăm sóc khách hàng của Công ty Truyền tải điện 1 - Công ty Truyền tải điện 1, tên tiếng Anh là: Power Transmission Company N01 (viết tắt PTC1), là một công ty chuyên sản xuất, kinh doanh vật tư thiết bị điện phục vụ sản xuất và truyền tải điện năng của Việt Nam, trực thuộc Tổng Công ty Truyền tải điện Quốc Gia.
- Nhiệm vụ của bạn là trả lời các vấn đề liên quan đến tài liệu nội bộ của Công ty Truyền tải điện 1
- Bạn cần trả lời khách hàng một cách lịch sự, chính xác và chuyên nghiệp. Trong mọi trường hợp, hãy giữ thái độ lịch sự, tôn trọng và chuyên nghiệp khi giao tiếp với khách hàng. Sử dụng các cụm từ như 'Tôi hiểu thắc mắc của bạn', 'Xin vui lòng cho tôi biết thêm thông tin'. Tránh sử dụng các biểu cảm cảm xúc, các từ ngữ không chính thức hoặc các câu hỏi mang tính cá nhân.
- Bạn là chuyên gia xuất sắc trong việc hiểu ý định của người dùng và trọng tâm của đầu vào người dùng, và cung cấp câu trả lời tối ưu nhất cho nhu cầu của người dùng từ các tài liệu bạn được cung cấp.

# NHIỆM VỤ
Nhiệm vụ của bạn là trả lời đầu vào của người dùng bằng cách sử dụng tài liệu được cung cấp, được đặt trong thẻ XML <RETRIEVED CONTEXT> dưới đây:
```
TÀI LIỆU ĐƯỢC CUNG CẤP:
<RETRIEVED CONTEXT>
{context}
</RETRIEVED CONTEXT>
```

# PIPELINE
Suy nghĩ thật kĩ và thực hiện từng bước theo flow Mermaid dưới đây:
    A[Bắt đầu] --> C[Dựa vào tài liệu được cung cấp và bối cảnh trò chuyện, nhận định xem có đủ thông tin để trả lời đầu vào của người dùng hay không?]

    C -->|Không đủ thông tin| D[Tạo câu trả lời:
    - Cần tạo câu trả lời khéo léo, vì không đủ thông tin để trả lời    
    - NO YAPPING
    - NO GREETING
    - Không sử dụng câu hỏi mở
    - Ngôn ngữ của câu trả lời là: {language}]
    
    D --> I[Gửi câu trả lời]

    C -->|Đủ thông tin| E[Phân tích tài liệu được cung cấp cùng với các nội dung dưới đây:
    - Đầu vào của người dùng
    - Lịch sử trò chuyện
    - Bản tóm tắt lịch sử trò chuyện]
    
    E --> F[Chọn lọc ra những nội dung liên quan nhất đến đầu vào của người dùng]
    
    F --> G[Tạo câu trả lời:
    - NO YAPPING
    - NO GREETING
    - Logic, Tự nhiên, Sâu sắc
    - Không sử dụng câu hỏi mở
    - Dùng ít nhất 4 câu
    - Dùng Markdown để định dạng câu trả lời
    - Ngôn ngữ của câu trả lời là: {language}]
    
    G --> I[Gửi câu trả lời]
    
    I --> J[Kết thúc]

# BỐI CẢNH TRÒ CHUYỆN
1. Lịch sử trò chuyện:
```
{pqa}
```

2. Tóm tắt lịch sử trò chuyện:
```
{summary_history}
```

3. Đầu vào của người dùng
```
{original_query}
```
"""




class ChatResponseWithNoContext(BaseModel):
    reasoning_answer: str = Field(
        description="Dựa vào bối cảnh trò chuyện, Viết các phân tích của bạn, giải thích tại sao bạn lại trả lời như vậy"
    )
    final_answer: str = Field(
        description="Dựa vào bối cảnh trò chuyện, đưa ra câu trả lời logic, tự nhiên và sâu sắc. Trả lời khéo léo nếu không đủ thông tin để trả lời đầu vào của người dùng"
    )



system_prompt_template_no_context = """\
# NGÀY VÀ GIỜ
Ngày và giờ hiện tại là {current_time}.

# THÔNG TIN VỀ Công ty Truyền tải điện 1
Công ty Truyền tải điện 1, tên tiếng Anh là: Power Transmission Company N01 (viết tắt PTC1), là một công ty chuyên sản xuất, kinh doanh vật tư thiết bị điện phục vụ sản xuất và truyền tải điện năng của Việt Nam, trực thuộc Tổng Công ty Truyền tải điện Quốc Gia.

# VAI TRÒ & NHIỆM VỤ
- Bạn là trợ lý AI chuyên nghiệp của nền tảng Công ty Truyền tải điện 1
- Nhiệm vụ: Hỗ trợ người dùng giải quyết các vấn đề liên quan đến Công ty Truyền tải điện 1
- Cam kết: Cung cấp thông tin chính xác, hữu ích và thân thiện.

# PIPELINE
Suy nghĩ thật kĩ và thực hiện từng bước theo flow Mermaid dưới đây:
    A[Bắt đầu] --> C[Dựa vào bối cảnh trò chuyện, nhận định xem có đủ thông tin để trả lời đầu vào của người dùng hay không?]

    C -->|Không đủ thông tin| I[Gửi câu trả lời khéo léo cho người dùng:
    - Dẫn dắt họ đến với các thông tin của Công ty Truyền tải điện 1 đã được nêu ở trên
    - Không sử dụng câu hỏi mở
    - Ngôn ngữ của câu trả lời là: {language}]
    
    C -->|Đủ thông tin| E[Phân tích bối cảnh trò chuyện bao gồm các nội dung dưới đây:
    - Đầu vào của người dùng
    - Lịch sử trò chuyện
    - Bản tóm tắt lịch sử trò chuyện]
    
    E --> F[Chọn lọc ra những nội dung liên quan nhất đến đầu vào của người dùng]
    
    F --> G[Tạo câu trả lời:
    - Logic
    - Tự nhiên
    - Sâu sắc
    - Không sử dụng câu hỏi mở
    - Dùng ít nhất 4 câu
    - Ngôn ngữ của câu trả lời là: {language}]
    
    G --> I[Gửi câu trả lời]
    
    I --> J[Kết thúc]


# BỐI CẢNH TRÒ CHUYỆN
1. Lịch sử trò chuyện:
```
{pqa}
```

2. Tóm tắt lịch sử trò chuyện:
```
{summary_history}
```

3. Đầu vào của người dùng
```
{original_query}
```
"""


class AnswerGenerator:
    def __init__(
        self,
        chat_generator: ChatGenerator,
    ) -> None:
        self.chat_generator = chat_generator


    def run(self,
            messages: List[ChatMessage],
            relevant_documents: List[dict],
            summary_history: str,
            original_query: str,
            language: str,
            thinking: bool = False
            ) -> AsyncGenerator[str, None]:
        if len(relevant_documents) == 0:
            return self.runNoContext(messages=messages,
                                    summary_history=summary_history,
                                    original_query=original_query,
                                    language=language,
                                    thinking=thinking
                                    )
        return self.runWithContext(messages=messages,
                                relevant_documents=relevant_documents,
                                summary_history=summary_history,
                                original_query=original_query,
                                language=language,
                                thinking=thinking
                                )
                                


    @observe(name="AnswerGeneratorWithContext")
    async def runWithContext(self,
                            messages: List[ChatMessage],
                            relevant_documents: List[RelevantDocument],
                            summary_history: str,
                            original_query: str,
                            language: str,
                            thinking: bool = False
                            ) -> AsyncGenerator[str, None]:
        # Thêm cấu hình Retry
        retry_settings = {
            "max_retries": 5,  # Số lần thử lại tối đa
            "initial_delay": 1,  # Thời gian chờ ban đầu (giây)
            "max_delay": 60,  # Thời gian chờ tối đa (giây)
            "multiplier": 2,  # Hệ số tăng thời gian chờ
        }

        current_time = datetime.now(timezone).strftime("%A, %Y-%m-%d %H:%M:%S")
        taken_messages = messages[-5:-1]
        pqa: str = "\n".join(map(lambda message: f"{message.role.value if hasattr(message.role, 'value') else message.role}: {message.content}", taken_messages))


        if thinking:
            for attempt in range(retry_settings["max_retries"]):
                try:
                    async for chunk in self.chat_generator.run(
                        messages=messages, 
                        system_prompt=system_prompt_template_with_context.format(
                            current_time=current_time, 
                            context=context,
                            pqa=pqa,
                            summary_history=summary_history,
                            original_query=original_query,
                            language=language,
                            ), 
                        temperature=0.5,
                        ):
                        yield chunk
                    return
                except Exception as e:
                    if attempt < retry_settings["max_retries"] - 1:
                        delay = min(retry_settings["initial_delay"] * (retry_settings["multiplier"] ** attempt), retry_settings["max_delay"])
                        await asyncio.sleep(delay)
                    else:
                        raise e



        else:
            relevant_documents = [RelevantDocument(
                id=doc['id'],
                page_content=doc['page_content'],
                tables=doc['tables'],
                images=doc['images'],
                references=doc['references'],
                category=doc['category'],
                url=doc['url'],
                score=doc['score'],
                cross_score=doc['cross_score'],
            ) for doc in relevant_documents]
            
            

            def format_document(index, doc):
                doc_start = f"\t<Document {index}>\n"
                url_line = f"\t\Chunk_ID: {doc.id}\n"
                content_line = f"\t\t{doc.page_content}\n"
                doc_end = f"\t</Document {index}>"
                return doc_start + url_line + content_line + doc_end

            context: str = "\n".join(
                format_document(i, doc) 
                for i, doc in enumerate(relevant_documents, 1)
            )

            # Thực hiện yêu cầu với Retry
            for attempt in range(retry_settings["max_retries"]):
                try:
                    async for chunk in self.chat_generator.run(
                        messages=messages, 
                        system_prompt=system_prompt_template_with_context.format(
                            current_time=current_time, 
                            context=context,
                            pqa=pqa,
                            summary_history=summary_history,
                            original_query=original_query,
                            language=language,
                            ), 
                        temperature=0.5,
                        ):
                        yield chunk
                    return
                except Exception as e:
                    if attempt < retry_settings["max_retries"] - 1:
                        delay = min(retry_settings["initial_delay"] * (retry_settings["multiplier"] ** attempt), retry_settings["max_delay"])
                        await asyncio.sleep(delay)
                    else:
                        raise e



    @observe(name="FormatAnswer") 
    def format_answer(self, answer: str) -> str:
        answer = answer.replace("```", "")
        # Thay thế ".  ", "!  ", "?  ", ",  " bằng một dấu cách
        # answer = re.sub(r'([.!?,])\s{2}', ' ', answer)
        # # Thay thế ký tự xuống dòng \n bằng một dòng mới
        # answer = answer.replace('\\n', '\n')
        items = [item.strip() for item in answer.split('\n') if item.strip()]
        return {
            "answer": answer.replace('\\', ''),
            "items": items
        }



    @observe(name="AnswerGeneratorNoContext")
    async def runNoContext(self,
                            messages: List[ChatMessage], 
                            summary_history: str, 
                            original_query: str,
                            language: str,
                            thinking: bool = False
                            ) -> AsyncGenerator[str, None]:
        current_time = datetime.now(timezone).strftime("%A, %Y-%m-%d %H:%M:%S")
        taken_messages = messages[-11:-1]
        pqa: str = "\n".join(map(lambda message: f"{message.role.value if hasattr(message.role, 'value') else message.role}: {message.content}", taken_messages))

        # Thêm cấu hình Retry
        retry_settings = {
            "max_retries": 5,  # Số lần thử lại tối đa
            "initial_delay": 1,  # Thời gian chờ ban đầu (giây)
            "max_delay": 60,  # Thời gian chờ tối đa (giây)
            "multiplier": 2,  # Hệ số tăng thời gian chờ
        }
        if thinking:
            for attempt in range(retry_settings["max_retries"]):
                try:
                    async for chunk in self.chat_generator.run(
                        messages=messages, 
                        system_prompt=system_prompt_template_no_context.format(
                            summary_history=summary_history, 
                            original_query=original_query,
                            pqa=pqa,
                            current_time=current_time,
                            language=language,
                            ), 
                        temperature=0.5,
                        ):
                        yield chunk
                    return
                except Exception as e:
                    if attempt < retry_settings["max_retries"] - 1:
                        delay = min(retry_settings["initial_delay"] * (retry_settings["multiplier"] ** attempt), retry_settings["max_delay"])
                        await asyncio.sleep(delay)
                    else:
                        raise e
        else:
            # Thực hiện yêu cầu với Retry
            for attempt in range(retry_settings["max_retries"]):
                try:
                    async for chunk in self.chat_generator.run(
                        messages=messages, 
                        system_prompt=system_prompt_template_no_context.format(
                            summary_history=summary_history, 
                            original_query=original_query,
                            pqa=pqa,
                            current_time=current_time,
                            language=language
                        ), 
                        temperature=0.5,
                        ):
                        yield chunk
                    return
                except Exception as e:
                    if attempt < retry_settings["max_retries"] - 1:
                        # Tính toán thời gian chờ với max_delay
                        delay = min(retry_settings["initial_delay"] * (retry_settings["multiplier"] ** attempt), retry_settings["max_delay"])
                        await asyncio.sleep(delay)
                    else:
                        raise e

