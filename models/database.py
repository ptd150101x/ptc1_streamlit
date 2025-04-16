import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv
from utils.monitor_log import logger
load_dotenv()

# Cấu hình database
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "abc")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5433")
POSTGRES_DB = os.getenv("POSTGRES_DB", "chatbot")

# URL kết nối đến postgres mặc định để tạo database
DEFAULT_DB_URL = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/postgres"
# URL kết nối đến database chatbot
DATABASE_URL = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"

def init_db():
    """Khởi tạo database, extensions, tables và indexes nếu chưa tồn tại"""
    try:
        # Đầu tiên kết nối đến database postgres mặc định
        temp_engine = create_engine(DEFAULT_DB_URL)
        with temp_engine.connect() as connection:
            # Tắt autocommit để có thể tạo database
            connection.execute(text("commit"))
            
            # Kiểm tra database tồn tại
            result = connection.execute(text(
                "SELECT 1 FROM pg_database WHERE datname = :database"
            ), {"database": POSTGRES_DB})
            exists = result.scalar()
            
            if not exists:
                # Tạo database mới nếu chưa tồn tại
                connection.execute(text(f"CREATE DATABASE {POSTGRES_DB}"))
                logger.info(f"Đã tạo database {POSTGRES_DB}")
            else:
                logger.info(f"Database {POSTGRES_DB} đã tồn tại")
        
        # Sau khi đã có database, tạo engine mới kết nối đến database đó
        engine = create_engine(DATABASE_URL)
        with engine.connect() as connection:
            # Cài đặt extensions
            connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
            connection.execute(text("CREATE EXTENSION IF NOT EXISTS pg_search;"))
            connection.commit()
            logger.info("Đã khởi tạo các extension thành công!")

            try:
                # Tạo bảng từ models
                from models.embedding import Base
                Base.metadata.create_all(bind=engine)
                logger.info("Đã tạo bảng thành công!")
            except Exception as e:
                logger.error(f"Lỗi khi tạo bảng: {str(e)}")

            # Kiểm tra và tạo indexes
            bm25_exists = connection.execute(text(
                "SELECT 1 FROM pg_indexes WHERE indexname = 'search_idx_bm25_index'"
            )).scalar()
            
            if not bm25_exists:
                connection.execute(text("""
                    CALL paradedb.create_bm25(
                        index_name => 'search_idx',
                        table_name => 'embeddings',
                        key_field => 'chunk_id',
                        text_fields => paradedb.field('page_content')
                    );
                """))
                logger.info("Đã tạo BM25 index")
            else:
                logger.info("BM25 index đã tồn tại")

            vector_exists = connection.execute(text(
                "SELECT 1 FROM pg_indexes WHERE indexname = 'embeddings_embedding_idx'"
            )).scalar()
            
            if not vector_exists:
                connection.execute(text("""
                    CREATE INDEX ON embeddings
                    USING hnsw (embedding vector_l2_ops);
                """))
                logger.info("Đã tạo Vector index")
            else:
                logger.info("Vector index đã tồn tại")
                
            connection.commit()
            logger.info("Đã hoàn tất kiểm tra và tạo indexes!")
            
        return engine

    except Exception as e:
        logger.error(f"Lỗi khi khởi tạo database: {str(e)}")
        raise e

# Khởi tạo engine sau khi tạo database
engine = init_db()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Hàm tạo session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()