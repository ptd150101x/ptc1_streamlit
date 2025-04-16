import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langfuse.decorators import observe
from .generator import Generator
import asyncio
from pydantic import BaseModel, Field
import traceback
from utils.monitor_log import logger
from utils.default_response import OVERLOAD_MESSAGE
import json
from typing import List
from schemas.document import RelevantDocument


class References(BaseModel):
    """Model định dạng thông tin tham chiếu"""
    chunk_id: str = Field(description="ID của chunk tài liệu đã được sử dụng để trả lời")
    score: float = Field(description="Điểm số đánh giá mức độ liên quan của chunk tài liệu")
    
    
    def dict(self):
        return {
            "chunk_id": self.chunk_id,
            "score": self.score
        }

class DocumentReferences(BaseModel):
    """Model định dạng danh sách các tham chiếu từ tài liệu"""
    references: list[References] = Field(
        default_factory=list,
        description="Danh sách các tham chiếu từ các chunks tài liệu được sử dụng"
    )
    
    def dict(self):
        return {
            "references": [ref.dict() for ref in self.references]
        }



system_prompt = """\
# NHIỆM VỤ
- Nhiệm vụ của bạn là đánh giá độ liên quan của các chunks tài liệu được cung cấp so với đầu vào của người dùng, và chấm điểm độ liên quan này theo thang điểm từ 0 đến 1
- Điểm 0 nghĩa là không hề liên quan, trong khi điểm 1 nghĩa là hoàn toàn liên quan

# HƯỚNG DẪN
- Vui lòng sử dụng các hướng dẫn dưới đây để đảm bảo độ nhất quán trong việc chấm điểm, và đặc biệt không cung cấp bất kỳ giải thích nào
1. Bắt đầu bằng cách đọc kỹ đầu vào của người dùng
2. Tiếp theo, đọc đoạn tài liệu được cung cấp
3. So sánh nội dung của đoạn tài liệu với đầu vào của người dùng
4. Gán một điểm đánh giá độ liên quan theo thang điểm từ 0 đến 1:
    - 0: Không hề liên quan
    - 0.25: Ít liên quan; có kết nối nhỏ hoặc gián tiếp
    - 0.5: Tạm liên quan; có một số kết nối rõ ràng
    - 0.75: Liên quan nhiều; phần lớn liên quan và có mối quan hệ chặt chẽ
    - 1: Hoàn toàn liên quan; trực tiếp và hoàn toàn phù hợp

# TÀI LIỆU ĐƯỢC CUNG CẤP:
```
{context}
```

# ĐẦU VÀO CỦA NGƯỜI DÙNG:
```
{question}
```
"""


class ExtractReferences:
    def __init__(
        self,
        generator: Generator,
        max_retries: int = 20,
        retry_delay: float = 2.0
    ) -> None:
        self.generator = generator
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    @observe(name="ExtractReferences")
    async def run(self,
                relevant_documents: List[RelevantDocument],
                question: str,
                ) -> str:
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

        for attempt in range(self.max_retries):
            try:
                response = await self.generator.run(
                            prompt = system_prompt.format(
                            context=context,
                            question=question,
                            ),
                            temperature = 0.4,
                            response_model=DocumentReferences,
                )
                response_json = json.loads(response)
                return response_json

            except Exception as e:
                logger.warning(f"Lỗi khi gọi ExtractReferences (lần thử {attempt + 1}/{self.max_retries}): {str(e)}")
                logger.error(traceback.format_exc())
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (2 ** attempt))
                else:
                    logger.error("Đã hết số lần thử lại. Không thể tăng cường.")
                    return OVERLOAD_MESSAGE