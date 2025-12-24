import streamlit as st
from typing import TypedDict, List
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from langgraph.graph import StateGraph, END
import tempfile
import os

llm=OllamaLLM(model="tinyllama")
embeddings=HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")
vector_db=None

class GraphState(TypedDict):
    query:str
    retrieved_docs: List[str]
    answer:str
    is_hallucinated:bool

def ingest_pdf(file_path:str):
    global vector_db
    loader=PyMuPDFLoader(file_path)
    docs=loader.load()
    splitter=RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50
    )
    chunks=splitter.split_documents(docs)
    vector_db=FAISS.from_documents(chunks,embeddings)

def retrieval_agent(state:GraphState):
    docs=vector_db.similarity_search(state["query"],k=5)
    return{
        "retrieved_docs":[d.page_content for d in docs]
    }

def answer_agent(state:GraphState):
    context="\n".join(state["retrieved_docs"])
    prompt=f"""
    Answer only from the context below.If answer is not present, say "i dont know".
    Context:{context}
    Question: {state["query"]}
    """
    response=llm.invoke(prompt)
    return {"answer":response}

def critic_agent(state:GraphState):
    critique_prompt=f"""
    Is the answer fully supported by the context?
    Answer YES or NO.

    Context:{state["retrieved_docs"]}
    Answer: {state["answer"]}
    """
    flag=llm.invoke(critique_prompt).strip()
    return {
        "is_hallucinated": flag=="NO"
    }

def supervisor(state:GraphState):
    if state.get("is_hallucinated"):
        return "retrieval"
    return "end"

graph=StateGraph(GraphState)
graph.add_node("retrieval",retrieval_agent)
graph.add_node("answer",answer_agent)
graph.add_node("critic",critic_agent)

graph.set_entry_point("retrieval")
graph.add_edge("retrieval","answer")
graph.add_edge("answer","critic")

graph.add_conditional_edges(
    "critic",
    supervisor,
    {
        "retrieval":"retrieval",
        "end":END
    }
)
app=graph.compile()
st.set_page_config(page_title="Enterprise GenAI Copilot", layout="wide")

st.title("🏢 Enterprise GenAI Knowledge Copilot")
st.caption("Welcome to Fastfix")

st.sidebar.header("📄 Upload Knowledge Base")
uploaded_file = st.sidebar.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    ingest_pdf(tmp_path)
    st.sidebar.success("Document indexed successfully")


if "chat" not in st.session_state:
    st.session_state.chat = []


st.subheader("💬 Ask Questions")

query = st.text_input("Enter your question")

if st.button("Ask") and query:
    if vector_db is None:
        st.warning("Please upload a document first.")
    else:
        result = app.invoke({
            "query": query,
            "retrieved_docs": [],
            "answer": "",
            "is_hallucinated": False
        })

        st.session_state.chat.append(
            {"question": query, "answer": result["answer"]}
        )

for chat in st.session_state.chat:
    st.markdown(f"**You:** {chat['question']}")
    st.markdown(f"**🤖 AI:** {chat['answer']}")
    st.markdown("---")