import os
from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END, START
from typing import TypedDict, List, Literal
from app.retriever import get_retriever
from langchain.prompts import PromptTemplate

# -------------------
# Load environment vars
# -------------------
load_dotenv()

INDEX_NAME = os.getenv("PINECONE_INDEX")
EMBED_MODEL = os.getenv("HUGGINGFACE_EMBEDDING_MODEL")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# -------------------
# State definition
# -------------------
class AgentState(TypedDict):
    query: str
    context: List[str]
    answer: str
    route: Literal["course", "web"]

# -------------------
# Initialize Tools
# -------------------
retriever = get_retriever(k=20)
web_search = TavilySearchResults(max_results=3)

# LLM (Gemini model)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

# -------------------
# Prompt Templates (Runnable style)
# -------------------

# Router prompt → classify query
router_prompt = PromptTemplate(
    input_variables=["query"],
    template="""
Classify the following user query:

Query: {query}

Respond with only one word:
- 'course' if it is related to university courses, departments, prerequisites, or other university related information
- 'general' if it is a general knowledge question.
"""
)
router_chain = router_prompt | llm   # Runnable pipeline

# Generation prompt → final answer synthesis
generation_prompt = PromptTemplate(
    input_variables=["query", "context"],
    template="""
You are a helpful university assistant. Use the given context to answer student queries.

Student query: {query}

Context:
{context}

Guidelines:
- If the query is about **university courses**, use only the course catalog context:
  • Mention prerequisites clearly and only when it was asked about in the query.  
  • If one prerequisite → say "The prerequisite is ...".  
  • Always include course code with its respective course title when possible.  
  • If only the course code is found, look in the context for its title.  
  • If nothing relevant, reply exactly: "I don’t know".
- If the query is **general knowledge**, use the web search context:
  • Summarize key points in clear, student-friendly language.  
  • Do not copy text verbatim.  
  • If nothing relevant, reply exactly: "I don’t know".

General rules:
- Be concise and accurate.  
- Do not invent info beyond the given context.
"""
)
generation_chain = generation_prompt | llm   # Runnable pipeline

# -------------------
# Router Node
# -------------------
def router_node(state: AgentState):
    response = router_chain.invoke({"query": state["query"]})
    decision = response.content.strip().lower()
    if "course" in decision:
        return {"route": "course"}
    else:
        return {"route": "web"}

# -------------------
# Course Retriever Node
# -------------------
def course_node(state: AgentState): 
    query = state["query"] 
    docs = retriever.invoke(query) 
    state["context"] = [doc.page_content for doc in docs]
    return state

# -------------------
# Web Search Node
# -------------------
def web_node(state: AgentState):
    query = state["query"]
    results = web_search.invoke(query)
    state["context"] = [r["content"] for r in results]
    return state

# -------------------
# Generation Node
# -------------------
def generation_node(state: AgentState): 
    query = state["query"] 
    if not state["context"]: 
        context_text = "No relevant context found." 
    else: 
        context_text = "\n\n".join(
            doc if isinstance(doc, str) else str(doc)
            for doc in state["context"]
        )

    response = generation_chain.invoke({"query": query, "context": context_text})
    state["answer"] = response.content
    return state

# -------------------
# Build LangGraph
# -------------------
graph = StateGraph(AgentState)

graph.add_node("router", router_node)
graph.add_node("course_retrieval", course_node)
graph.add_node("web_search", web_node)
graph.add_node("generation", generation_node)

graph.add_edge(START, "router")

# Conditional edges
graph.add_conditional_edges(
    "router",
    lambda state: state["route"],
    {
        "course": "course_retrieval",
        "web": "web_search"
    }
)

graph.add_edge("course_retrieval", "generation")
graph.add_edge("web_search", "generation")
graph.add_edge("generation", END)

agent_app = graph.compile()
