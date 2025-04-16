import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from schemas.api_schema import ChatLogicInputData
import streamlit as st
import requests
import base64
import os
import json
import redis
import time
import uuid
from utils.monitor_log import logger
from api_url import API_URL
from services.chatbot.chatbot_ai_service import AI_Chatbot_Service





# Kết nối Redis
redis_host = os.getenv("REDIS_HOST", "localhost")
redis_port = int(os.getenv("REDIS_PORT", 6379))
redis_password = os.getenv("REDIS_PASSWORD", "123456")
redis_client = redis.Redis(
    host=redis_host,
    port=redis_port,
    password=redis_password,
    decode_responses=True
)

# Chuyển đổi base64 thành file JPEG
def save_base64_as_image(base64_str, output_file):
    with open(output_file, "wb") as f:
        f.write(base64.b64decode(base64_str))

# Hàm để lấy thread từ Redis
def get_thread_names(user_id="default_user"):
    thread_ids = redis_client.smembers(f"user_threads:{user_id}")
    thread_names = {}
    
    # Debug: In ra danh sách thread_ids
    logger.info(f"DEBUG - Thread IDs: {thread_ids}")
    
    for thread_id in thread_ids:
        thread_info = redis_client.hgetall(f"thread:{thread_id}")
        if thread_info:
            thread_names[thread_id] = thread_info.get("name", f"Thread {thread_id}")
    
    # Debug: In ra thread_names
    logger.info(f"DEBUG - Thread Names: {thread_names}")
    
    # Nếu không có thread nào, tạo thread mặc định
    if not thread_names:
        default_thread_id = "default"
        redis_client.hset(f"thread:{default_thread_id}", mapping={
            "name": "Cuộc hội thoại mặc định",
            "created_at": str(time.time())
        })
        redis_client.sadd(f"user_threads:{user_id}", default_thread_id)
        
        # Thêm tin nhắn chào mừng
        welcome_message = {
            "role": "assistant",
            "content": "Xin chào! Tôi là trợ lý của PTC1. Tôi có thể giúp gì cho bạn?"
        }
        redis_client.rpush(f"messages:{default_thread_id}", json.dumps(welcome_message))
        
        thread_names[default_thread_id] = "Cuộc hội thoại mặc định"
    
    return thread_names

# Hàm để lấy tin nhắn của thread từ Redis
def get_thread_messages(thread_id):
    messages_json = redis_client.lrange(f"messages:{thread_id}", 0, -1)
    
    # Debug: In ra số lượng tin nhắn JSON nhận được
    logger.info(f"DEBUG - Số lượng tin nhắn cho thread {thread_id}: {len(messages_json)}")
    
    messages = [json.loads(msg) for msg in messages_json]
    
    # Nếu không có tin nhắn, thêm tin nhắn chào mừng
    if not messages:
        logger.info(f"DEBUG - Không tìm thấy tin nhắn cho thread {thread_id}, thêm tin nhắn chào mừng")
        welcome_message = {
            "role": "assistant",
            "content": "Xin chào! Tôi là trợ lý của PTC1. Tôi có thể giúp gì cho bạn?"
        }
        redis_client.rpush(f"messages:{thread_id}", json.dumps(welcome_message))
        messages = [welcome_message]
    
    return messages

# Hàm để thêm tin nhắn vào thread
def add_message_to_thread(thread_id, message):
    # Debug: In ra thông tin tin nhắn được thêm vào
    logger.info(f"DEBUG - Thêm tin nhắn vào thread {thread_id}: {message['role']} - {message['content'][:30]}...")
    
    redis_client.rpush(f"messages:{thread_id}", json.dumps(message))
    
    # Debug: Kiểm tra số lượng tin nhắn sau khi thêm
    messages_count = redis_client.llen(f"messages:{thread_id}")
    logger.info(f"DEBUG - Số lượng tin nhắn sau khi thêm: {messages_count}")

# Hàm để tạo thread mới
def create_new_thread(user_id="default_user", thread_name=None):
    thread_id = f"thread_{uuid.uuid4().hex[:8]}"
    if not thread_name:
        thread_count = redis_client.scard(f"user_threads:{user_id}")
        thread_name = f"Cuộc hội thoại {thread_count + 1}"
    
    # Debug: In ra thông tin thread mới
    logger.info(f"DEBUG - Tạo thread mới: ID={thread_id}, Name={thread_name}")
    
    # Thay thế hset với mapping bằng nhiều lệnh hset riêng lẻ
    redis_client.hset(f"thread:{thread_id}", "name", thread_name)
    redis_client.hset(f"thread:{thread_id}", "created_at", str(time.time()))
    redis_client.sadd(f"user_threads:{user_id}", thread_id)
    
    # Debug: Kiểm tra thread đã được tạo chưa
    thread_exists = redis_client.exists(f"thread:{thread_id}")
    logger.info(f"DEBUG - Thread đã được tạo: {thread_exists}")
    
    # Thêm tin nhắn chào mừng
    welcome_message = {
        "role": "assistant",
        "content": "Xin chào! Tôi là trợ lý của PTC1. Tôi có thể giúp gì cho bạn?"
    }
    redis_client.rpush(f"messages:{thread_id}", json.dumps(welcome_message))
    
    return thread_id, thread_name

# Hàm để xóa thread
def delete_thread(user_id, thread_id):
    # Debug: In ra thông tin thread bị xóa
    logger.info(f"DEBUG - Xóa thread: ID={thread_id}")
    
    redis_client.srem(f"user_threads:{user_id}", thread_id)
    redis_client.delete(f"thread:{thread_id}")
    redis_client.delete(f"messages:{thread_id}")
    
    # Debug: Kiểm tra thread đã bị xóa chưa
    thread_exists = redis_client.exists(f"thread:{thread_id}")
    logger.info(f"DEBUG - Thread còn tồn tại sau khi xóa: {thread_exists}")

# API Configuration


# Page Configuration 
st.set_page_config(
    page_title="PTC1 Assistant",
    page_icon="🤖",
    layout="wide"
)

# Khởi tạo user_id (trong thực tế nên lấy từ hệ thống xác thực)
user_id = "default_user"

# Khởi tạo session state
if "current_thread" not in st.session_state:
    st.session_state.current_thread = "default"

if "max_references" not in st.session_state:
    st.session_state.max_references = 3

if "selected_model" not in st.session_state:
    st.session_state.selected_model = "gemini-2.0-flash-lite"

if "category" not in st.session_state:
    st.session_state.category = "Tất cả"

# Display chat title
st.title("💬 PTC1 Assistant")

# Add thread selection and management in sidebar
with st.sidebar:
    st.title("🧵 Quản lý cuộc hội thoại")
    
    # Lấy danh sách thread từ Redis
    thread_names = get_thread_names(user_id)
    
    # Thread selection with "+" button in the same row
    col1, col2 = st.columns([4, 1])
    with col1:
        # Đảm bảo thread hiện tại tồn tại trong danh sách
        if st.session_state.current_thread not in thread_names:
            st.session_state.current_thread = list(thread_names.keys())[0] if thread_names else "default"
            
        st.session_state.current_thread = st.selectbox(
            "Chọn cuộc hội thoại",
            options=list(thread_names.keys()),
            format_func=lambda x: thread_names[x],
            index=list(thread_names.keys()).index(st.session_state.current_thread) if st.session_state.current_thread in thread_names else 0,
            label_visibility="collapsed"
        )
    with col2:
        if st.button("➕", help="Tạo cuộc hội thoại mới"):
            new_thread_id, new_thread_name = create_new_thread(user_id)
            st.session_state.current_thread = new_thread_id
            st.success(f"Đã tạo cuộc hội thoại mới: {new_thread_name}")
            st.rerun()
    
    # Xóa thread hiện tại (chỉ khi có nhiều hơn 1 thread)
    if len(thread_names) > 1:
        if st.button("Xóa cuộc hội thoại hiện tại"):
            thread_name = thread_names[st.session_state.current_thread]
            delete_thread(user_id, st.session_state.current_thread)
            st.session_state.current_thread = list(thread_names.keys())[0] if st.session_state.current_thread == list(thread_names.keys())[0] else list(thread_names.keys())[1]
            st.warning(f"Đã xóa cuộc hội thoại: {thread_name}")
            st.rerun()
    
    st.title("⚙️ Cài đặt")
    # Slider điều chỉnh số lượng tài liệu tham khảo
    st.session_state.max_references = st.slider(
        "Số lượng tài liệu tham khảo tối đa",
        min_value=0,
        max_value=5,
        value=st.session_state.max_references,
        help="Điều chỉnh số lượng tài liệu tham khảo hiển thị trong mỗi câu trả lời"
    )
    
    # Thêm tùy chọn chọn danh mục tài liệu để lọc tìm kiếm
    st.session_state.category = st.selectbox(
        "Chọn danh mục tài liệu",
        options=["mba", "quytrinh_vanhanh", "All"],
        index=0,
        help="Chọn danh mục tài liệu để lọc kết quả tìm kiếm"
    )
    
    st.title("Hướng dẫn sử dụng")
    st.markdown("""
    1. Nhập câu hỏi về PTC1 vào ô chat
    2. Nhấn Enter để gửi câu hỏi
    3. Đợi phản hồi từ trợ lý
    4. Tạo cuộc hội thoại mới để bắt đầu chủ đề khác
    """)

# Add model selection in columns next to chat
col1, col2 = st.columns([3, 1])
with col2:
    st.session_state.selected_model = st.selectbox(
        "Chọn model",
        ["gemini-2.0-flash-lite",
        "gemini-2.0-flash",
        "gemini-2.0-flash-thinking-exp-01-21",
        "gemini-1.5-pro",
        "gemini-1.5-flash",
        ],
        index=0,
        help="Chọn model AI để xử lý câu hỏi của bạn"
    )

# Lấy tin nhắn của thread hiện tại từ Redis
current_messages = get_thread_messages(st.session_state.current_thread)

# Display chat history with proper styling for each message
for message in current_messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Modified message display code
if prompt := st.chat_input("Nhập câu hỏi của bạn..."):
    # Thêm tin nhắn người dùng vào Redis
    user_message = {"role": "user", "content": prompt}
    logger.info(f"DEBUG - Thêm tin nhắn người dùng vào Redis: {user_message}")
    add_message_to_thread(st.session_state.current_thread, user_message)
    
    with st.chat_message("user"):
        st.write(prompt)

    ai_chatbot_service = AI_Chatbot_Service(
        model=st.session_state.selected_model,
        thinking=True if "think" in st.session_state.selected_model else False
    )

    language = asyncio.run(ai_chatbot_service.detect_language.run(prompt))

    if language == "Unknown":
        language = "Tiếng Việt"
        
        
    rewrite_prompt = asyncio.run(ai_chatbot_service.single_query.run(
        messages=[
            {"role": "model", "content": message["content"]} if message["role"] == "assistant" else message
            for message in current_messages[-10:-1]
        ],
        question=prompt,
        summary_history="",
        thinking=True if "think" in st.session_state.selected_model else False
        )
    )
    
    relevant_documents = []
    seen_ids = set()
    documents = ai_chatbot_service.document_retriever.run(
        query_text=rewrite_prompt,
        threshold=0.2,
        category=st.session_state.category
    )
    for doc in documents['final_rerank']:
        if doc['id'] not in seen_ids:
            relevant_documents.append(doc)
            seen_ids.add(doc['id'])
    
    if not relevant_documents:
        documents = ai_chatbot_service.document_retriever.run(
            query_text=prompt,
            threshold=0.2,
            category=st.session_state.category
        )
        for doc in documents['final_rerank']:
            if doc['id'] not in seen_ids:
                relevant_documents.append(doc)
                seen_ids.add(doc['id'])
    
    relevant_documents = sorted(relevant_documents, key=lambda doc: doc['cross_score'], reverse=False) if relevant_documents else []
    
    for chunk in asyncio.run(ai_chatbot_service.answer_generator.run(
        messages=[
                {"role": "model", "content": message["content"]} if message["role"] == "assistant" else message
                for message in current_messages
            ],
        relevant_documents=relevant_documents,
        summary_history="",
        original_query=prompt,
        language=language,
        thinking=True if "think" in st.session_state.selected_model else False
    )):
        with st.chat_message("assistant"):
            st.write_stream(chunk)
    references = asyncio.run(ai_chatbot_service.extract_references.run(
        relevant_documents=relevant_documents,
        question=rewrite_prompt
    ))

            # # Xử lý references sau khi đã hiển thị nội dung
            # references = requests.post(
            #     f"{API_URL}/extract_references",
            #     json={
            #         "relevant_documents": response_json.get("references", []),
            #         "question": prompt
            #     },
            #     headers={"Content-Type": "application/json"}
            # )
            
            # if references:
            #     with st.chat_message("assistant"):
            #         st.write("**Xem thêm tại:**")
            #         for ref in references:
            #             if "references" in ref:
            #                 if ref['url'] is not None:
            #                     st.markdown(f"- {ref['references']}: {ref['url']}")
            #                 else:
            #                     st.markdown(f"- {ref['references']}")