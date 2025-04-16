from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, String, DateTime, Text, Integer
from sqlalchemy.sql import func
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Embedding(Base):
    __tablename__ = "embeddings"
    chunk_id = Column(Integer, primary_key=True, autoincrement=True)
    embedding = Column(Vector(1024))
    page_content = Column(Text)
    content_hash = Column(String(255), nullable=True)
    tables = Column(Text, nullable=True)
    images = Column(Text, nullable=True)
    videos = Column(Text, nullable=True)
    references = Column(Text, nullable=True)
    category = Column(Text, nullable=True)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    url = Column(Text, nullable = True)