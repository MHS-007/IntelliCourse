import warnings
from langchain_core._api import LangChainDeprecationWarning

# Suppress only unwanted warnings
warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)
warnings.filterwarnings("ignore", message="ALTS creds ignored")

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from app.agent import agent_app


# -------------------
# FastAPI setup
# -------------------
api_app = FastAPI(title="IntelliCourse API", version="1.0")

# -------------------
# Pydantic models
# -------------------
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    source_tool: str
    retrieved_context: list[str] | None = None

# -------------------
# Endpoint
# -------------------
@api_app.post("/chat", response_model=QueryResponse)
def chat_endpoint(request: QueryRequest):
    try:
        result = agent_app.invoke({"query": request.query, "context": [], "answer": ""})
        source_tool = "course" if result.get("route") == "course" else "web"
        return QueryResponse(
            answer=result["answer"],
            source_tool=source_tool,
            retrieved_context=result.get("context", [])
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )