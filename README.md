# IntelliCourse 🎓

An AI-Powered University Course Advisor built with **FastAPI**, **LangChain**, **LangGraph**, and **Pinecone/ChromaDB**.  
It helps students explore university courses by answering questions like:

- _"What are the prerequisites for Software Engineering?"_
- _"Which courses cover Python and data visualization?"_
- _"How many departments does the university have?"_
- _"What is the job market like for data scientists?"_

Instead of manually searching long PDFs or Word files, IntelliCourse provides intelligent, concise, and accurate answers using a **Retrieval-Augmented Generation (RAG)** pipeline.

---

## ✨ Features

- **REST API** built with FastAPI.
- **RAG Backend**: Retrieves relevant chunks from the course catalog using Hugging Face embeddings + Pinecone.
- **Multi-Tool Agent with LangGraph**:
  - Router Node: Decides if query is course-related or general.
  - Course Retriever Node: Uses vector DB for university course queries.
  - Web Search Node: Uses Tavily Search for general queries.
  - Generation Node: Synthesizes a clear final answer using Gemini LLM.
- **Supports free-tier LLMs**: Google Gemini (via `langchain-google-genai`).
- **Clean API design**: Structured request/response models with context + source tool info.

---

## 🛠️ Tech Stack

- **API Framework**: [FastAPI](https://fastapi.tiangolo.com/)
- **LLM Orchestration**: [LangChain](https://python.langchain.com/) + [LangGraph](https://langchain-ai.github.io/langgraph/)
- **LLM**: Google Gemini (`gemini-2.5-flash`) (free tier)
- **Embedding Model**: HuggingFace `all-MiniLM-L6-v2` (default)
- **Vector Database**: Pinecone (free tier)
- **Web Search Tool**: Tavily Search

---

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/MHS-007/IntelliCourse.git
cd IntelliCourse
```
### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate # On Linux/Mac
venv\Scripts\activate # On Windows
```
### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
### 4. Setup Environment Variables
To run this project, you will need to add the following environment variables to your .env file

# API Keys

- `PINECONE_API_KEY`=your_pinecone_api_key
- `GOOGLE_API_KEY`=your_google_api_key
- `TAVILY_API_KEY`=your_tavily_api_key

# Vector DB + Embeddings

- `PINECONE_INDEX`=intellicourse-index
- `HUGGINGFACE_EMBEDDING_MODEL`=sentence-transformers/all-MiniLM-L6-v2

### 5. Run FastAPI Server
```bash
uvicorn app.api:api_app --reload
```
Server will start at: http://127.0.0.1:8000

---

## 📡 API Documentation

- **Endpoint**: /chat
- **Method**: POST
- **Description**: Takes a student query and returns an intelligent answer.
- **Request Body**: {"query": "What are the prerequisites for Software Engineering?"}

- **Sample cURL**: 
```bash
curl -X POST http://127.0.0.1:8000/chat \
 -H "Content-Type: application/json" \
 -d '{"query": "What are the prerequisites for Software Engineering?"}'
```
- **Sample Python (requests)**:
```python
import requests

response = requests.post(
"http://127.0.0.1:8000/chat",
json={"query": "What are the prerequisites for Software Engineering?"}
)

print(response.json())
```
---

## 🌐 Frontend

A minimal web-based interface is included (`index.html`) that allows you to:

- Enter a query (course-related or general).
- View AI-generated answers with **source tool** and **retrieved context**.
- Maintain query history with the option to clear it.

### Running the Frontend

1. Make sure the FastAPI backend is running:
   ```bash
   uvicorn app.api:api_app --reload
   ```
2. Open `index.html` in your browser.

⚠️ Note: The frontend fetches responses from the FastAPI `/chat` endpoint. If the server is not running, the page won’t work.

---

## ⚠️ Notes & Limitations

- Google Gemini free tier allows only **10 requests per minute**. Exceeding this will return 429 quota exceeded errors.

- The RAG system is limited to the data you upload (e.g., course catalogs). It will not generate answers outside its knowledge base.

- Web search results may vary depending on Tavily API responses.

---

## 👨‍💻 Author
This was developed as part of the GenAI Bootcamp project at Iqra University.