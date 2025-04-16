#!/bin/bash

# Đợi DB sẵn sàng
./sh_script/wait-for-it.sh db:5432 --timeout=60 --strict -- echo "✅ PostgreSQL is ready"

# Đợi Redis sẵn sàng 
./sh_script/wait-for-it.sh redis:6379 --timeout=60 --strict -- echo "✅ Redis is ready"

# Đợi embed_rerank sẵn sàng với timeout dài hơn
./sh_script/wait-for-it.sh embed_rerank:8001 --timeout=180 --strict -- echo "✅ Embed & Rerank is ready"

# Chạy app backend
exec python app/run.py