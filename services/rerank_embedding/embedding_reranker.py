from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from FlagEmbedding import BGEM3FlagModel, FlagReranker
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Any
import os
from collections import defaultdict
import numpy as np
import asyncio


app = FastAPI()

# Thêm middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

USE_GPU = "True"
DEFAULT_BATCH_SIZE = os.getenv("DEFAULT_BATCH_SIZE", 64)
LIMIT_MAX_BATCH_SIZE = os.getenv("LIMIT_MAX_BATCH_SIZE", 64)#epends on server hardware

DEFAULT_MAX_LENGTH = os.getenv("DEFAULT_MAX_LENGTH", 8192)
LIMIT_MAX_LENGTH = os.getenv("LIMIT_MAX_LENGTH", 8192)



if USE_GPU:
    embedder = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
    hardware_lock = asyncio.Lock()
else:
    embedder = BGEM3FlagModel('BAAI/bge-m3', use_fp16=False, device="cpu")
    hardware_lock = None


class EmbeddingRequest(BaseModel):
    sentences: List[str]
    params: Optional[dict] = None
    embedding_types: Optional[dict] = None

class EmbeddingResponse(BaseModel):
    embeddings: dict

async def process_embeddings(sentences: List[str], params: Optional[dict], embedding_types: Optional[dict]):
    if params is None:
        params = {
            "batch_size": DEFAULT_BATCH_SIZE,
            "max_length": DEFAULT_MAX_LENGTH
        }
    
   
    if embedding_types is None:
        embedding_types = {
            "dense": True,
            "sparse": False,
            "colbert": False
        }
        
    batch_size = params["batch_size"]
    max_length = params["max_length"]    
    
    
    # Assert valid batch size (> 0 and <= LIMIT_MAX_BATCH_SIZE)
    if batch_size < 1 or batch_size > LIMIT_MAX_BATCH_SIZE:
        raise HTTPException(status_code=400, detail=f"Invalid batch size, must be >=1 and <={LIMIT_MAX_BATCH_SIZE}.")
    
    # Assert valid max length (> 0 and <= LIMIT_MAX_LENGTH)
    if max_length < 1 or max_length > LIMIT_MAX_LENGTH:
        raise HTTPException(status_code=400, detail=f"Invalid max length, must be >=1 and <={LIMIT_MAX_LENGTH}.")
    
    return_dense_vecs: bool = embedding_types["dense"] if "dense" in embedding_types else False
    return_sparse_vecs: bool = embedding_types["sparse"] if "sparse" in embedding_types else False
    return_colbert_vecs: bool = embedding_types["colbert"] if "colbert" in embedding_types else False
    
    # if all False
    if not (return_dense_vecs or return_sparse_vecs or return_colbert_vecs):
        return []

    if USE_GPU:
        async with hardware_lock:
            total_embeddings = embedder.encode(sentences,
                                            batch_size=batch_size,
                                            max_length=max_length,
                                            return_dense=return_dense_vecs,
                                            return_sparse=return_sparse_vecs,
                                            return_colbert_vecs=return_colbert_vecs)
    else:
        total_embeddings = embedder.encode(sentences,
                                        batch_size=batch_size,
                                        max_length=max_length,
                                        return_dense=return_dense_vecs,
                                        return_sparse=return_sparse_vecs,
                                        return_colbert_vecs=return_colbert_vecs)

    embeddings: dict = {}
    dense_vecs: List[List[float]] = total_embeddings["dense_vecs"] if return_dense_vecs else None
    lexical_weights: List[defaultdict[int, Any]] = total_embeddings["lexical_weights"] if return_sparse_vecs else None
    colbert_vecs: List[np.ndarray] = total_embeddings["colbert_vecs"] if return_colbert_vecs else None
    
    if return_dense_vecs:
        embeddings["dense_vecs"] = np.array(dense_vecs, dtype=np.float32).tolist()
    if return_sparse_vecs:
        embeddings["sparse_vecs"] = [{key: float(value) for key, value in lexical_weights[i].items()} for i in range(len(sentences))]
    if return_colbert_vecs:
        embeddings["colbert_vecs"] = [colbert_vecs[i].astype(np.float32).tolist() for i in range(len(sentences))]
    
    return embeddings

@app.post("/embed", response_model=EmbeddingResponse)
async def embed_sentences(request: EmbeddingRequest):
    try:
        sentences = request.sentences
        params = request.params
        embedding_types = request.embedding_types
        
        embeddings = await process_embeddings(sentences, params, embedding_types)
        
        return EmbeddingResponse(embeddings=embeddings)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




# Khởi tạo model cho reranker
reranker = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True)

# Định nghĩa mô hình dữ liệu cho yêu cầu rerank
class RerankRequest(BaseModel):
    sentence_pairs: list
    normalize: bool = False
@app.post("/rerank")
async def rerank(request: RerankRequest):
    try:
        sentence_pairs = request.sentence_pairs
        normalize = request.normalize

        # Tính toán điểm số cho các cặp câu
        scores = reranker.compute_score(sentence_pairs, normalize=normalize)
        # Chuyển đổi mảng NumPy thành list Python
        return {"scores": scores}
    except Exception as e:
        print(f"Error in rerank: {str(e)}")
        return {"error": str(e)}

@app.get("/benchmark")
async def benchmark_rerank():
    import time

    # Tạo đoạn văn bản dài (giả lập tài liệu dài)
    long_doc = "Artificial intelligence is a field of computer science " * 10000
    dummy_pairs = [["what is AI?", long_doc]] * 64

    start = time.time()
    scores = reranker.compute_score(dummy_pairs, normalize=False)
    end = time.time()
    return {
        "time_taken": round(end - start, 4),
        "num_pairs": len(dummy_pairs),
        "qps": round(len(dummy_pairs) / (end - start), 2),
        "example_scores": scores[:3],
        "doc_char_len": len(long_doc)
    }




@app.get("/health")
async def health_check():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("embedding_reranker:app", host="0.0.0.0", port=8001, reload=True)