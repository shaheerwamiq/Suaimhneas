# from langchain_google_genai import ChatGoogleGenerativeAI
# from dotenv import load_dotenv

# load_dotenv()

# model=ChatGoogleGenerativeAI(model="gemini-2.5-flash")
# prompt="What is your name?"
# result = model.invoke(prompt)

# print(result.content)

"""
Single-file RAG example combining LangChain + LangGraph, using the
tool-calling AGENT pattern (ToolNode + add_messages) rather than a fixed
retrieve -> generate pipeline.

The LLM is given a `retrieve_context` tool backed by a FAISS retriever.
It decides on its own whether/when to call it (possibly multiple times,
possibly not at all for a question it can answer directly), which is the
same pattern used for general LangGraph tool-calling agents.

Requirements (requirements.txt):
    langchain_community
    langchain_google_genai
    python-dotenv
    langchain_core
    pydantic
    langchain_huggingface
    sentence-transformers
    faiss-cpu
    pypdf
    bs4
    langgraph
    langchain-text-splitters

Setup:
    1. pip install -r requirements.txt
    2. Create a .env file with:
           GOOGLE_API_KEY=your_gemini_api_key_here
    3. Put a PDF at ./data/sample.pdf (or set SOURCE_URL below to use a webpage instead)
    4. python rag_langgraph_agent.py
"""

import os
from typing import Annotated, List, Sequence, TypedDict

from dotenv import load_dotenv

# --- LangChain imports ---
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langchain_core.tools import tool

# --- LangGraph imports ---
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SOURCE_PATH = "./data/sample.pdf"          # local PDF path
SOURCE_URL = None                          # e.g. "https://example.com/article" to use WebBaseLoader instead
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "gemini-2.5-flash"
TOP_K = 4


# ---------------------------------------------------------------------------
# 1. Load -> split -> embed -> vector store (plain LangChain, done once at startup)
# ---------------------------------------------------------------------------
def load_documents() -> List[Document]:
    if SOURCE_URL:
        print(f"Loading from URL: {SOURCE_URL}")
        loader = WebBaseLoader(SOURCE_URL)
    else:
        if not os.path.exists(SOURCE_PATH):
            raise FileNotFoundError(
                f"No file found at {SOURCE_PATH}. "
                f"Either place a PDF there, or set SOURCE_URL to load from the web."
            )
        print(f"Loading PDF from: {SOURCE_PATH}")
        loader = PyPDFLoader(SOURCE_PATH)

    docs = loader.load()
    print(f"Loaded {len(docs)} document(s).")
    return docs


def split_documents(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunk(s).")
    return chunks


def build_vectorstore(chunks: List[Document]) -> FAISS:
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    print("FAISS vector store built.")
    return vectorstore


docs = load_documents()
chunks = split_documents(docs)
vectorstore = build_vectorstore(chunks)
retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})


# ---------------------------------------------------------------------------
# 2. Define retrieval as a TOOL the LLM can call
# ---------------------------------------------------------------------------
@tool
def retrieve_context(query: str) -> str:
    """Search the knowledge base (loaded document) for information relevant
    to the query. Use this whenever the user's question might be answered
    by the document's contents. Returns the most relevant passages."""
    results = retriever.invoke(query)
    if not results:
        return "No relevant information found in the knowledge base."
    return "\n\n---\n\n".join(
        f"[Source: {d.metadata.get('source', 'unknown')}, page {d.metadata.get('page', '?')}]\n{d.page_content}"
        for d in results
    )


tools = [retrieve_context]

llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0)
llm_with_tools = llm.bind_tools(tools)


# ---------------------------------------------------------------------------
# 3. LangGraph state (message-list based, like a chat agent)
# ---------------------------------------------------------------------------
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


SYSTEM_PROMPT = SystemMessage(
    content=(
        "You are a helpful assistant. You have access to a `retrieve_context` "
        "tool that searches a knowledge base built from a specific document. "
        "Use it whenever a question could be answered from that document. "
        "If the tool doesn't return relevant information, say you don't know "
        "rather than making something up. Always cite the source/page when "
        "you use retrieved context."
    )
)


# ---------------------------------------------------------------------------
# 4. Nodes: agent (LLM decides) + tools (ToolNode executes)
# ---------------------------------------------------------------------------
def agent_node(state: AgentState) -> dict:
    messages = state["messages"]
    # Prepend the system prompt on every call so the model always has instructions
    response = llm_with_tools.invoke([SYSTEM_PROMPT, *messages])
    return {"messages": [response]}


tool_node = ToolNode(tools)


def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "tools"
    return END


# ---------------------------------------------------------------------------
# 5. Build the graph
# ---------------------------------------------------------------------------
graph_builder = StateGraph(AgentState)
graph_builder.add_node("agent", agent_node)
graph_builder.add_node("tools", tool_node)

graph_builder.add_edge(START, "agent")
graph_builder.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
graph_builder.add_edge("tools", "agent")  # after tool runs, let the LLM see results and respond

graph = graph_builder.compile()


# ---------------------------------------------------------------------------
# 6. Main chat loop
# ---------------------------------------------------------------------------
def main():
    if not os.getenv("GOOGLE_API_KEY"):
        raise EnvironmentError(
            "GOOGLE_API_KEY not found. Add it to a .env file: GOOGLE_API_KEY=your_key_here"
        )

    print("\nRAG agent (LangChain + LangGraph, tool-calling style) ready. Type 'exit' to quit.\n")

    history: List[BaseMessage] = []
    while True:
        question = input("You: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue

        history.append(HumanMessage(content=question))
        result = graph.invoke({"messages": history})
        history = list(result["messages"])

        final_answer = history[-1]
        print(f"\nAssistant: {final_answer.content}\n")


if __name__ == "__main__":
    main()