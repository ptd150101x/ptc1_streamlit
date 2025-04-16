class RelevantDocument:
    def __init__(self,
                id: str,
                page_content: str,
                tables: str = None,
                images: str = None,
                videos: str = None,
                references: str = None,
                category: str = None,
                url: str = None,
                score=None,
                cross_score=None,
                ) -> None:
        self.id = id
        self.page_content = page_content
        self.tables = tables
        self.images = images
        self.videos = videos
        self.references = references
        self.category = category
        self.url = url
        self.score = score
        self.cross_score = cross_score


    def __str__(self):
        return f"Document(id={self.id}, page_content={self.page_content[:50]}...)"
    
    def __repr__(self):
        return self.__str__()

    def to_dict(self):
        return {
            "id": self.id,
            "page_content": self.page_content,
            "tables": self.tables,
            "images": self.images,
            "videos": self.videos,
            "references": self.references,
            "category": self.category,
            "url": self.url,
            "score": self.score,
            "cross_score": self.cross_score,
        }