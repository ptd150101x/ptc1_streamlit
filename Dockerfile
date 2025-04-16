FROM python:3.11-slim

# 1. Thiết lập thư mục làm việc
WORKDIR /app

# 2. Cài dependencies cần thiết cho pip & build
RUN apt-get update && apt-get install -y gcc libpq-dev && apt-get clean

# 3. Copy requirements và cài thư viện
COPY requirements.txt .

RUN pip install uv && \
    uv pip install --system -r requirements.txt

# 4. Copy toàn bộ source vào image
COPY . /app

# 5. Cấp quyền thực thi cho các script .sh
RUN chmod +x /app/sh_script/*.sh

# 6. (Tùy chọn) Tạo thư mục cache cho HuggingFace
RUN mkdir -p /root/.cache/huggingface

RUN python -c "from FlagEmbedding import BGEM3FlagModel, FlagReranker; \
    BGEM3FlagModel('BAAI/bge-m3', use_fp16=False); \
    FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True)"
