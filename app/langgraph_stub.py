"""
Optional LangGraph stub — not required for the FastAPI demo path.

Shows how a retrieve → generate graph would be wired once you add an LLM.
Install langgraph separately if you want to expand this.
"""

from __future__ import annotations

from typing import TypedDict


class RagState(TypedDict):
    question: str
    context: str
    answer: str


def retrieve_node(state: RagState) -> RagState:
    from app.rag import format_docs, get_vectorstore
    from app.config import get_settings

    settings = get_settings()
    docs = get_vectorstore(settings).similarity_search(state["question"], k=settings.top_k)
    return {**state, "context": format_docs(docs)}


def generate_node(state: RagState) -> RagState:
    # Placeholder: plug ChatGroq / ChatOpenAI / ChatOllama here.
    ctx = state.get("context") or ""
    if not ctx.strip():
        answer = "I do not know — no matching documents were retrieved."
    else:
        answer = f"(langgraph stub) Q={state['question']!r}\n\n{ctx}"
    return {**state, "answer": answer}


def run_stub_graph(question: str) -> RagState:
    """Linear retrieve → generate without depending on the langgraph package."""
    state: RagState = {"question": question, "context": "", "answer": ""}
    state = retrieve_node(state)
    state = generate_node(state)
    return state


# When langgraph is installed you can replace run_stub_graph with:
#
#   from langgraph.graph import StateGraph, END
#   g = StateGraph(RagState)
#   g.add_node("retrieve", retrieve_node)
#   g.add_node("generate", generate_node)
#   g.set_entry_point("retrieve")
#   g.add_edge("retrieve", "generate")
#   g.add_edge("generate", END)
#   app = g.compile()
