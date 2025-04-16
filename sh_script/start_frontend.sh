#!/bin/bash
# Đợi backend sẵn sàng
./sh_script/wait-for-it.sh backend:8000 --timeout=30 --strict -- echo "✅ Backend is ready"

# Đợi Redis sẵn sàng nếu frontend dùng
./sh_script/wait-for-it.sh redis:6379 --timeout=30 --strict -- echo "✅ Redis is ready"

# Chạy Streamlit app
exec streamlit run app/streamlit.py