import streamlit as st
import requests
import os
import time
import uuid

# API Configuration
API_URL = os.getenv("API_URL", "http://localhost:8000")

# Page Configuration 
st.set_page_config(
    page_title="PTC1 Assistant",
    page_icon="🤖",
    layout="wide"
)

# Khởi tạo session state
if "threads" not in st.session_state:
    st.session_state.threads = {
        "default": {
            "name": "Cuộc hội thoại mặc định",
            "messages": [{
                "role": "assistant",
                "content": "Xin chào! Tôi là trợ lý của PTC1. Tôi có thể giúp gì cho bạn?"
            }],
            "created_at": time.time()
        }
    }

if "current_thread" not in st.session_state:
    st.session_state.current_thread = "default"

if "max_references" not in st.session_state:
    st.session_state.max_references = 3

if "selected_model" not in st.session_state:
    st.session_state.selected_model = "gemini-2.0-flash-lite"

if "category" not in st.session_state:
    st.session_state.category = "mba"

def create_new_thread(thread_name=None):
    thread_id = f"thread_{uuid.uuid4().hex[:8]}"
    if not thread_name:
        thread_count = len(st.session_state.threads)
        thread_name = f"Cuộc hội thoại {thread_count + 1}"
    
    st.session_state.threads[thread_id] = {
        "name": thread_name,
        "messages": [{
            "role": "assistant",
            "content": "Xin chào! Tôi là trợ lý của PTC1. Tôi có thể giúp gì cho bạn?"
        }],
        "created_at": time.time()
    }
    
    return thread_id, thread_name

def delete_thread(thread_id):
    if thread_id in st.session_state.threads:
        del st.session_state.threads[thread_id]

# Display chat title
st.title("💬 PTC1 Assistant")

# Add thread selection and management in sidebar
with st.sidebar:
    st.title("🧵 Quản lý cuộc hội thoại")
    
    # Thread selection with "+" button in the same row
    col1, col2 = st.columns([4, 1])
    with col1:
        thread_names = {tid: thread["name"] for tid, thread in st.session_state.threads.items()}
        st.session_state.current_thread = st.selectbox(
            "Chọn cuộc hội thoại",
            options=list(thread_names.keys()),
            format_func=lambda x: thread_names[x],
            index=list(thread_names.keys()).index(st.session_state.current_thread),
            label_visibility="collapsed"
        )
    
    with col2:
        if st.button("➕", help="Tạo cuộc hội thoại mới"):
            new_thread_id, new_thread_name = create_new_thread()
            st.session_state.current_thread = new_thread_id
            st.success(f"Đã tạo cuộc hội thoại mới: {new_thread_name}")
            st.rerun()
    
    # Xóa thread hiện tại (chỉ khi có nhiều hơn 1 thread)
    if len(st.session_state.threads) > 1:
        if st.button("Xóa cuộc hội thoại hiện tại"):
            thread_name = st.session_state.threads[st.session_state.current_thread]["name"]
            delete_thread(st.session_state.current_thread)
            st.session_state.current_thread = list(st.session_state.threads.keys())[0]
            st.warning(f"Đã xóa cuộc hội thoại: {thread_name}")
            st.rerun()
    
    st.title("⚙️ Cài đặt")
    st.session_state.max_references = st.slider(
        "Số lượng tài liệu tham khảo tối đa",
        min_value=0,
        max_value=5,
        value=st.session_state.max_references,
        help="Điều chỉnh số lượng tài liệu tham khảo hiển thị trong mỗi câu trả lời"
    )
    
    st.session_state.category = st.selectbox(
        "Chọn danh mục tài liệu",
        options=["mba", "tai_lieu_ki_thuat"],
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
         "gemini-1.5-flash"],
        index=0,
        help="Chọn model AI để xử lý câu hỏi của bạn"
    )

# Display chat history
current_messages = st.session_state.threads[st.session_state.current_thread]["messages"]
for message in current_messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input and API interaction
if prompt := st.chat_input("Nhập câu hỏi của bạn..."):
    user_message = {"role": "user", "content": prompt}
    st.session_state.threads[st.session_state.current_thread]["messages"].append(user_message)
    
    with st.chat_message("user"):
        st.write(prompt)

    try:
        # Chuyển đổi role "assistant" thành "model" khi gửi API
        api_messages = [
            {"role": "model" if msg["role"] == "assistant" else msg["role"], "content": msg["content"]} 
            for msg in current_messages + [user_message]
        ]
        
        print("Request payload:", {
            "content": prompt,
            "histories": api_messages,
            "category": st.session_state.category,
            "model": st.session_state.selected_model,
            "thinking": True if "think" in st.session_state.selected_model else False
        })
        
        response = requests.post(
            f"{API_URL}/chat",
            json={
                "content": prompt,
                "histories": api_messages,
                "category": st.session_state.category,
                "model": st.session_state.selected_model,
                "thinking": True if "think" in st.session_state.selected_model else False
            },
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            response_json = response.json()
            answer = response_json.get("answer", "")
            
            if "think" in st.session_state.selected_model:
                for chunk in answer:
                    if chunk.text:
                        st.write(chunk.text, end="")
            else:
                references = response_json.get("references", [])
                
                with st.chat_message("assistant"):
                    st.write(answer)
                
                assistant_message = {"role": "assistant", "content": answer}
                st.session_state.threads[st.session_state.current_thread]["messages"].append(assistant_message)
                
                if references:
                    with st.chat_message("assistant"):
                        st.write(f"Xem thêm tại: {references}")

    except Exception as e:
        st.error(f"Lỗi kết nối đến API: {str(e)}")