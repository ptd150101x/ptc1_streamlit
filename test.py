import requests
import json

url = "http://localhost:8000/chat"

payload = json.dumps({
  "content": "máy biến áp là gì",
  "histories": [
    {
      "role": "model",
      "content": "Xin chào! Tôi là trợ lý của PTC1. Tôi có thể giúp gì cho bạn?"
    },
    {
      "role": "user",
      "content": "máy biến áp là gì"
    }
  ],
  "category": "mba",
  "model": "gemini-2.0-flash-lite",
  "thinking": False
})
headers = {
  'Content-Type': 'application/json'
}

try:
    # Quan trọng: sử dụng stream=True
    response = requests.request("POST", url, headers=headers, data=payload, stream=True)
    
    # Cách 1: Xử lý từng dòng riêng biệt
    for line in response.iter_lines():
        if line:
            print(line.decode('utf-8'))
            # Xử lý JSON tại đây, không tích lũy
            
    # Cách 2: Nếu muốn theo dõi tiến trình:
    # full_response = ""
    # for chunk in response.iter_content(chunk_size=None):
    #     if chunk:
    #         chunk_text = chunk.decode('utf-8')
    #         print(chunk_text, end="", flush=True)
    #         full_response += chunk_text

except requests.exceptions.ChunkedEncodingError:
    # Kết nối đã đóng, stream kết thúc
    print("\nStreaming đã hoàn tất.")
except Exception as e:
    print(f"Lỗi: {e}")