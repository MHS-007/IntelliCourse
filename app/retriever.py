import os
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX")
EMBED_MODEL = os.getenv("HUGGINGFACE_EMBEDDING_MODEL")

# Return a retriever connected to Pinecone with HuggingFace embeddings.
def get_retriever(k: int = 10):
    if not PINECONE_API_KEY or not INDEX_NAME or not EMBED_MODEL:
        raise ValueError("❌ Missing environment variables. Check your .env file.")

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    vectorstore = PineconeVectorStore.from_existing_index(
        index_name=INDEX_NAME,
        embedding=embeddings,
        namespace="default"
    )

    return vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": k})