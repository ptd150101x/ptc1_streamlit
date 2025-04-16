import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.embedding import Embedding
from models.database import SessionLocal
from langfuse.decorators import observe
from utils.monitor_log import logger
from typing import List, Set
from schemas.document import RelevantDocument
import numpy as np
import json
import requests
from .embedder import Embedder
from sqlalchemy import text, select, and_
import re
import traceback


API_URL = "http://localhost:8001"


LIMIT_SEARCH = 25

class DocumentRetriever:
    def __init__(self, session) -> None:
        # self.session = SessionLocal()
        self.session = session
        self.embedder = Embedder(
            url=f"{API_URL}/embed",
            batch_size=1,
            max_length=4096,
            max_retries=10,
            retry_delay=2.0
        )



    def _join_references(self, page_content: str, references: str) -> str:
        """
        Trích xuất phần nội dung trước dòng phân cách chứa dấu gạch (-) trong page_content 
        và ghép với nội dung references hiện có.
        """
        extracted_header = ""
        if page_content:
            # Tìm phần nội dung trước dòng chứa ít nhất 3 dấu gạch liên tiếp ở dòng mới
            match = re.search(r'^(.*?)(?:\n-+\n)', page_content, flags=re.DOTALL)
            if match:
                extracted_header = match.group(1).strip()
        if extracted_header and references and references.strip():
            return extracted_header + "\n" + references
        elif extracted_header:
            return extracted_header
        else:
            return references


    @observe(name="RetrieverAndReranker")
    def run(self,
            query_text: str,
            threshold: float,
            category: str = None
            ) -> List[RelevantDocument]:
        hybrid_search_documents = self.hybrid_search(
            query_text=query_text,
            category=category
            )
        rerank_hybrid_search = self.rerank_documents(
            query=query_text,
            documents=hybrid_search_documents,
            threshold=threshold
            )
        rerank_hybrid_search_documents = rerank_hybrid_search["top_reranked_documents"]
        backup_hybrid_search_documents = rerank_hybrid_search["reranked_documents"]


        all_documents = rerank_hybrid_search_documents
        unique_documents = {doc['id']: doc for doc in all_documents}.values()

        back_up_documents = backup_hybrid_search_documents
        unique_backup_documents = {doc['id']: doc for doc in back_up_documents}.values()
        return {
            "final_rerank": list(unique_documents),
            "backup_rerank": list(unique_backup_documents)
        }


    
    @observe(name="DocumentRetriever_hybrid_search")
    def hybrid_search(self,
                    query_text: str,
                    category: str = None
                    ) -> List[RelevantDocument]:
        query_embedding = self.embedder.run(query_text)
        documents: List[RelevantDocument] = []
        seen_ids: Set[str] = set()
        cleaned_query = re.sub(r'[^\w\s]', '', query_text)
        print("query_text: ", query_text)
        print("cleaned_query: ", cleaned_query)
        
        try:
            base_conditions = []
            if category:
                if category == "All":
                    base_conditions = []
                else:
                    base_conditions.append(Embedding.category == category)
            try:
                if cleaned_query:
                    full_text_conditions = base_conditions.copy()
                    full_text_conditions.append(Embedding.page_content.op('@@@')(cleaned_query))


                    full_text_query = self.session.execute(
                        select(Embedding.chunk_id,
                                Embedding.page_content,
                                Embedding.tables,
                                Embedding.images,
                                Embedding.references,
                                Embedding.category,
                                Embedding.url,
                                text("paradedb.score(embeddings.chunk_id)")
                                )
                        .where(and_(
                            *full_text_conditions
                        ))
                        .order_by(text("score DESC"))
                        .limit(LIMIT_SEARCH)
                    ).fetchall()
                    

                    logger.info(f'full_text_query: {len(full_text_query)}')
                    # Xử lý kết quả full text search
                    for row in full_text_query:
                        if row.chunk_id not in seen_ids:
                            new_references = self._join_references(row.page_content, row.references)
                            documents.append(
                                RelevantDocument(
                                    id=row.chunk_id,
                                    page_content=row.page_content,
                                    tables=row.tables,
                                    images=row.images,
                                    references=new_references,
                                    category=row.category,
                                    url=row.url
                                )
                            )
                            seen_ids.add(row.chunk_id)
                if not documents:
                    processed_query = ' AND '.join(cleaned_query.split()[:10]).strip()
                    if processed_query:
                        bm25_conditions = base_conditions.copy()
                        bm25_conditions.append(Embedding.page_content.op('@@@')(processed_query))

                        bm25_query = self.session.execute(
                            select(Embedding.chunk_id,
                                Embedding.page_content,
                                Embedding.tables,
                                Embedding.images,
                                Embedding.category,
                                Embedding.references,
                                Embedding.url,
                                text("paradedb.score(embeddings.chunk_id)"))
                            .where(and_(
                                *bm25_conditions
                            ))
                            .order_by(text("score DESC"))
                            .limit(LIMIT_SEARCH)
                        ).fetchall()

                        logger.info(f'bm25_query: {len(bm25_query)}')
                        # Xử lý kết quả BM25 search
                        for row in bm25_query:
                            if row.chunk_id not in seen_ids:
                                new_references = self._join_references(row.page_content, row.references)
                                documents.append(
                                    RelevantDocument(
                                        id=row.chunk_id,
                                        page_content=row.page_content,
                                        tables=row.tables,
                                        images=row.images,
                                        references=new_references,
                                        category=row.category,
                                        url=row.url
                                    )
                                )
                                seen_ids.add(row.chunk_id)


                # Query semantic search
                semantic_query = self.session.execute(
                    select(Embedding.chunk_id,
                        Embedding.page_content,
                        Embedding.tables,
                        Embedding.images,
                        Embedding.references,
                        Embedding.category,
                        Embedding.url
                        )
                    .where(and_(
                        *base_conditions
                        )
                    )
                    .order_by(Embedding.embedding.l2_distance(query_embedding))
                    .limit(LIMIT_SEARCH)
                ).fetchall()

                logger.info(f'semantic_query: {len(semantic_query)}')

                # Xử lý kết quả semantic search
                for row in semantic_query:
                    if row.chunk_id not in seen_ids:
                        new_references = self._join_references(row.page_content, row.references)
                        documents.append(
                            RelevantDocument(
                                id=row.chunk_id,
                                page_content=row.page_content,
                                tables=row.tables,
                                images=row.images,
                                references=new_references,
                                category=row.category,
                                url=row.url
                            )
                        )
                        seen_ids.add(row.chunk_id)
                    else:
                        logger.info(
                            f"Trùng kết quả semantic search và BM25 search với query = `{query_text}`, id = {row.chunk_id}"
                        )

            except Exception as e:
                traceback.print_exc()
                logger.warning(f"Full text search failed: {str(e)}")
                self.session.rollback()
            

            self.session.commit()

        except Exception as e:
            logger.error(f"Error in hybrid_search: {str(e)}")
            self.session.rollback()
            raise
        return [doc.to_dict() for doc in documents]


    @observe(name="Rerank_Document")
    def rerank_documents(
        self,
        query: str,
        documents: List[dict],
        threshold: float = 0.2
        ) -> List[RelevantDocument]:
        try:
            documents = [RelevantDocument(
                id=doc['id'],
                page_content=doc['page_content'],
                tables=doc['tables'],
                images=doc['images'],
                references=doc['references'],
                category=doc['category'],
                url=doc['url']
            ) for doc in documents]

            # Tạo các cặp câu truy vấn và nội dung tài liệu
            sentence_pairs = [[query, item.page_content] for item in documents]
            
            payload = json.dumps({
                "sentence_pairs": sentence_pairs,
                "normalized": False
            })

            url = f"{API_URL}/rerank"
            headers = {
                'Content-Type': 'application/json'
            }

            try:
                # Gửi yêu cầu POST với payload đến API rerank
                response = requests.request("POST", url, headers=headers, data=payload)

                # In phản hồi từ server (nếu cần debug)
                # print(response.text)

                # Parse phản hồi JSON từ server
                api_response = json.loads(response.text)
                
                if "scores" in api_response:
                    similarity_scores = api_response["scores"]
                else:
                    logger.error(f"Invalid API response format: {api_response}")
                    # Trả về documents gốc thay vì raise error
                    return {
                        "top_reranked_documents": [doc.to_dict() for doc in documents[:5]],
                        "reranked_documents": [doc.to_dict() for doc in documents[:3]]
                    }

            except requests.exceptions.RequestException as e:
                logger.error(f"API request failed: {str(e)}")
                # Trả về documents gốc nếu API fail
                return {
                    "top_reranked_documents": [doc.to_dict() for doc in documents[:5]],
                    "reranked_documents": [doc.to_dict() for doc in documents[:3]]
                }

            # Sigmoid và scoring
            def sigmoid(x):
                return 1 / (1 + np.exp(-x))

            # Gán điểm số từ API cho từng tài liệu
            for idx, document in enumerate(documents):
                document.cross_score = sigmoid(similarity_scores[idx])  # Lấy điểm từ API

            filter_documents = [doc for doc in documents if doc.cross_score >= threshold]

            no_filter_documents = sorted(
                documents,  # Sử dụng danh sách đã được lọc
                key=lambda item: item.cross_score,
                reverse=True  # Sắp xếp theo thứ tự giảm dần
            )[:4]


            # Sắp xếp tài liệu theo điểm số từ API (cross_score)
            reranked_documents = sorted(
                filter_documents,  # Sử dụng danh sách đã được lọc
                key=lambda item: item.cross_score,
                reverse=True  # Sắp xếp theo thứ tự giảm dần
            )

            # Chỉ lấy 5 tài liệu hàng đầu
            top_reranked_documents = reranked_documents[:5]

            print(f"Number of reranked documents: {len(reranked_documents)}")

            print("Top 5 hits with API rerank scores:")
            for item in top_reranked_documents:
                print("\t{:.3f}\t{}".format(item.cross_score, item.id))

            return {
                "top_reranked_documents": [doc.to_dict() for doc in top_reranked_documents],
                "reranked_documents": [doc.to_dict() for doc in no_filter_documents]
            }

        except Exception as e:
            logger.error(f"Error in rerank_documents: {str(e)}")
            # Trả về documents gốc trong trường hợp lỗi
            return {
                "top_reranked_documents": [doc.to_dict() for doc in documents[:5]],
                "reranked_documents": [doc.to_dict() for doc in documents[:3]]
            }