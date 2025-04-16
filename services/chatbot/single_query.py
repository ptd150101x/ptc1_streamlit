import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from langfuse.decorators import observe
from schemas.api_schema import ChatLogicInputData
from .generator import Generator
from datetime import datetime
import pytz
import asyncio
from pydantic import BaseModel, Field
import traceback
from utils.monitor_log import logger
from utils.default_response import OVERLOAD_MESSAGE
import json


# Đặt múi giờ thành múi giờ Việt Nam
timezone = pytz.timezone('Asia/Ho_Chi_Minh')

class RewritePrompt(BaseModel):
   analysis: str = Field(description="Dựa vào bối cảnh trò chuyện, Viết các phân tích của bạn, giải thích tại sao bạn lại trả lời như vậy")
   rewrited_prompt: str = Field(description="Đầu vào của người dùng được viết lại (nếu cần thiết)")


system_prompt = """\
# NGÀY VÀ GIỜ
Ngày và giờ hiện tại là {current_time}.

# THÔNG TIN VỀ Công ty Truyền tải điện 1
Công ty Truyền tải điện 1, tên tiếng Anh là: Power Transmission Company N01 (viết tắt PTC1), là một công ty chuyên sản xuất, kinh doanh vật tư thiết bị điện phục vụ sản xuất và truyền tải điện năng của Việt Nam, trực thuộc Tổng Công ty Truyền tải điện Quốc Gia.

# NHIỆM VỤ
Phân tích và viết lại đầu vào của người dùng (nếu cần thiết) thành một prompt độc lập để truy xuất tài liệu từ cơ sở dữ liệu vector

## QUY TRÌNH THỰC HIỆN
Suy nghĩ từng bước và thực hiện theo từng bước dưới đây:
1. Phân tích bối cảnh trò chuyện:
   - Thứ tự ưu tiên: Đầu vào của người dùng > Lịch sử trò chuyện > Tóm tắt lịch sử trò chuyện
   - Xác định các thực thể, mệnh đề, các mối quan hệ, và ý định chính

2. Viết lại prompt độc lập
   - Chỉ cần viết lại khi ý định của người dùng không được thể hiện rõ ràng trong bối cảnh trò chuyện
   - Hạn chế sử dụng ngữ cảnh trước đó nếu không cần thiết
   - Sử dụng ngôn ngữ đầu vào của người dùng
   - Không sử dụng chủ ngữ, vị ngữ
   - Loại bỏ thông tin dư thừa, tập trung vào nội dung chính và ý định của người dùng
   
# VÍ DỤ
Ví dụ 1:
```
USER: Tích hợp Shopee với Getfly CRM có lợi ích gì?
GETFLY PRO: Việc tích hợp Shopee giúp đồng bộ đơn hàng và quản lý dễ dàng trên Getfly CRM.
USER: Còn Google Drive thì sao?
```

USER's Standalone Prompt:
```
Lợi ích của việc kết nối Google Drive với Getfly CRM là gì?
```

Ví dụ 2:
```
USER: Getfly CRM có hỗ trợ gì cho bộ phận tài chính không?
GETFLY PRO: Có, Getfly CRM có tính năng Tài chính kế toán giúp thiết lập định khoản, quản lý ngân sách, và báo cáo tài chính.
USER: Không, ý tôi là phiên bản web cơ?
```

USER's Standalone Prompt:
```
Getfly CRM phiên bản web có hỗ trợ gì cho bộ phận tài chính không?
```

# BỐI CẢNH TRÒ CHUYỆN:
Tóm tắt lịch sử trò chuyện:
```
<summary history>
{summary_history}
</summary_history>
```

Lịch sử trò chuyện:
```
<chat history>
{chat_history}
</chat history>
```

Đầu vào của người dùng:
```
<user's input>
{question}
</user's input>
```
"""




class SingleQuery:
   def __init__(
      self,
      generator: Generator,
      max_retries: int = 5,
      retry_delay: float = 2.0
   ) -> None:
      self.generator = generator
      self.max_retries = max_retries
      self.retry_delay = retry_delay

   @observe(name="SingleQuery")
   async def run(self,
                  user_data: ChatLogicInputData,
                  question: str,
                  rewrite_prompt: str = "",
                  thinking: bool = False
                  ) -> str:
      if len(user_data.histories) < 11:
         taken_messages = user_data.histories  # Lấy tất cả nếu ít hơn 5
      else:
         taken_messages = user_data.histories[-11:-1]

      # Giả sử taken_messages là danh sách các tin nhắn trong chat history
      chat_history: str = "\n".join(map(lambda message: f"{message.role.value if hasattr(message.role, 'value') else message.role}: {message.content}", taken_messages))
      current_time = datetime.now(timezone).strftime("%A, %Y-%m-%d %H:%M:%S")
      summary_history: str = user_data.summary

      if thinking:
         for attempt in range(self.max_retries):
            try:
               response = await self.generator.run(
                        prompt = system_prompt.format(
                           current_time=current_time,
                           summary_history=summary_history,
                           chat_history=chat_history, 
                           question=question,
                           ),
                        temperature = 0.5,
               )
               return result

            except Exception as e:
               logger.warning(f"Lỗi khi gọi Enrichment (lần thử {attempt + 1}/{self.max_retries}): {str(e)}")
               logger.error(traceback.format_exc())
      else:
         for attempt in range(self.max_retries):
            try:
               response = await self.generator.run(
                        prompt = system_prompt.format(
                           current_time=current_time,
                           summary_history=summary_history,
                           chat_history=chat_history, 
                           question=question,
                           ),
                        temperature = 0.5,
                        response_model=RewritePrompt,
               )
               response_json = response.dict()
               result = {
                  "analysis": response_json.get("analysis", ""),
                  "rewrite_prompt": response_json.get("rewrited_prompt", ""),
               }
               return result

            except Exception as e:
               logger.warning(f"Lỗi khi gọi Enrichment (lần thử {attempt + 1}/{self.max_retries}): {str(e)}")
               logger.error(traceback.format_exc())
               if attempt < self.max_retries - 1:
                  await asyncio.sleep(self.retry_delay * (2 ** attempt))
               else:
                  logger.error("Đã hết số lần thử lại. Không thể tăng cường.")
                  return OVERLOAD_MESSAGE