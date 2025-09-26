import os
from dotenv import load_dotenv
from langchain_community.document_loaders import Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")
EMBED_MODEL = os.getenv("HUGGINGFACE_EMBEDDING_MODEL")

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

# Create index if not exists
if PINECONE_INDEX not in [i["name"] for i in pc.list_indexes()]:
    pc.create_index(
        name=PINECONE_INDEX,
        dimension=384,  # dimension for all-MiniLM-L6-v2 embeddings
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

# Load documents from folder
def load_documents(folder_path="data/course_catalogs_realistic"):
    docs = []
    for file in os.listdir(folder_path):
        if file.endswith(".docx"):
            loader = Docx2txtLoader(os.path.join(folder_path, file))
            docs.extend(loader.load())
    return docs

# Hybrid splitter: overview -> character chunks, courses -> single chunk
def split_documents(docs, chunk_size=500, chunk_overlap=50):
    chunks = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    for doc in docs:
        text = doc.page_content

        if "Course Code:" in text:
            parts = text.split("Course Code:")

            # First part = department overview
            if parts[0].strip():
                overview_parts = splitter.split_text(parts[0].strip())
                for part in overview_parts:
                    chunks.append(Document(page_content=part, metadata=doc.metadata))

            # Each course = single chunk
            for part in parts[1:]:
                course_text = "Course Code:" + part.strip()
                if course_text:
                    chunks.append(Document(page_content=course_text, metadata=doc.metadata))
        else:
            # If no "Course Code" → normal splitting
            normal_parts = splitter.split_text(text)
            for part in normal_parts:
                chunks.append(Document(page_content=part, metadata=doc.metadata))

    return chunks

# Build embeddings and store in Pinecone
def build_vector_store():
    print("🔍 Checking Pinecone index state...")
    index = pc.Index(PINECONE_INDEX)
    stats = index.describe_index_stats()

    if stats["total_vector_count"] > 0:
        print("⚠️ Index already contains data. Skipping re-indexing.")
        return PineconeVectorStore.from_existing_index(
            index_name=PINECONE_INDEX,
            embedding=HuggingFaceEmbeddings(model_name=EMBED_MODEL),
            namespace="default"
        )

    print("Loading documents...")
    documents = load_documents()
    print(f"Loaded {len(documents)} documents")

    print("Splitting into chunks (hybrid)...")
    chunks = split_documents(documents)
    print(f"Created {len(chunks)} chunks")

    print("Initializing embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    print("Indexing into Pinecone...")
    vectorstore = PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=PINECONE_INDEX,
        namespace="default"
    )

    print("✅ Data indexed successfully into Pinecone!")
    return vectorstore

# -> For quick manual test (run at least 1 time to build the index)
# if __name__ == "__main__":
#     build_vector_store()