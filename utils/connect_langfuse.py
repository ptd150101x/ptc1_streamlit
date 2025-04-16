from langfuse import Langfuse
import os
from dotenv import load_dotenv

load_dotenv()

def connect_langfuse():
    langfuse = Langfuse(
        secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        host=os.getenv("LANGFUSE_HOST")
    )
    return langfuse
