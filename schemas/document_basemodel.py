from pydantic import BaseModel

class RelevantDocumentBaseModel(BaseModel):
    id: str
    page_content: str
    tables: str | None = None
    images: str | None = None
    videos: str | None = None
    references: str | None = None
    category: str | None = None
    url: str | None = None
    score: float | None = None
    cross_score: float | None = None

    def __str__(self):
        return f"Document(id={self.id}, page_content={self.page_content[:50]}...)"
    
    def __repr__(self):
        return self.__str__()

    def to_dict(self):
        return self.model_dump()
