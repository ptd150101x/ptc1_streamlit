import psycopg2

# Kết nối đến database
conn = psycopg2.connect(
    dbname="chatbot",
    user="postgres",
    password="abc",  # thay bằng mật khẩu thật
    host="localhost",
    port=5432
)

cur = conn.cursor()

# Đường dẫn tới file CSV
csv_path = '/home/sv/source-code/chabot_gemini_api_key/embeddings_202504090037.csv'

# Tên bảng trong database
table_name = 'embeddings'  # thay bằng tên bảng thực tế

# Import file CSV vào bảng
with open(csv_path, 'r') as f:
    # Nếu CSV có dòng header, dùng HEADER
    cur.copy_expert(f"COPY {table_name} FROM STDIN WITH (FORMAT CSV, DELIMITER '|', HEADER)", f)

conn.commit()
cur.close()
conn.close()

print("Import thành công!")
