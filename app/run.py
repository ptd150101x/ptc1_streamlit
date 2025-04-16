import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from routers.chatbot_router import chatbot_router
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from models.database import init_db, engine
from models.embedding import Base

@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    Base.metadata.create_all(bind=engine)
    yield

# Add CORS middleware
origins = [
    "http://localhost",
]

app = FastAPI(debug=True, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(chatbot_router)


if __name__ == "__main__":
    uvicorn.run("run:app", host="0.0.0.0", port=8002, reload=True)
