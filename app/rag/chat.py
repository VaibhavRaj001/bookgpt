# app/rag/chat.py
import os
from typing import List, Dict, Any, Tuple
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import LLMChain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.memory import ConversationBufferMemory
from db import SessionLocal, ChatMessage, ChatSession

# DB imports (adjust path if yours differ)
from db import SessionLocal, ChatMessage

# config
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_K = 6

# Prompt template — chain will use memory via ConversationBufferMemory (memory_key "chat_history")
PROMPT_TEMPLATE = """
You are BookGPT — an assistant that answers using ONLY the provided CONTEXT and the conversation history.
Do NOT hallucinate. If the answer is not present in the context, say exactly:
"The book does not mention this in the provided context."

CONTEXT:
{context}

USER QUERY:
{query}

If you used the conversation history to form your reply, you may reference it briefly but base facts only on CONTEXT.
Answer concisely, then provide 1-3 supporting evidence lines chosen verbatim from the context (if present).
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", PROMPT_TEMPLATE),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{query}")
])


def _get_vectorstore_for_book(book_id: str):
    persist_dir = os.path.join("storage", "chroma_db", book_id)
    if not os.path.isdir(persist_dir):
        raise FileNotFoundError(f"No Chroma DB at {persist_dir}")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vs = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    return vs


def _retrieve_context(book_id: str, query: str, k: int = DEFAULT_K) -> str:
    vs = _get_vectorstore_for_book(book_id)
    # similarity_search is stable across versions
    docs = vs.similarity_search(query, k=k)
    # join into one context (you can tune truncation)
    context = "\n\n".join([getattr(d, "page_content", "") for d in docs])
    return context, docs


def _load_history_into_memory(session_id: str, memory: ConversationBufferMemory):
    """
    Read messages from DB and add them to LangChain memory.
    Assumes ChatMessage.role is "user" or "assistant" (or "system").
    """
    db = SessionLocal()
    try:
        msgs = (
            db.query(ChatMessage)
            .filter(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.created_at.asc())
            .all()
        )
        for m in msgs:
            if m.role == "user":
                memory.chat_memory.add_user_message(m.content)
            elif m.role == "assistant":
                memory.chat_memory.add_ai_message(m.content)
            else:
                # system or others can be added as user messages or ignored
                memory.chat_memory.add_user_message(m.content)
    finally:
        db.close()


def _save_message_to_db(session_id: str, role: str, content: str):
    db = SessionLocal()
    try:
        msg = ChatMessage(session_id=session_id, role=role, content=content)
        db.add(msg)
        db.commit()
        db.refresh(msg)
        return msg
    finally:
        db.close()


def chat_with_book_chain(book_id: str, session_id: str, query: str, k: int = DEFAULT_K) -> Dict[str, Any]:
    """
    Runs retrieval -> LLMChain with memory flow.
    Returns a dict: {"answer": str, "supporting_evidence": [...], "retrieved_count": int}
    """

    # 1) Build context via RAG
    context, docs = _retrieve_context(book_id, query, k=k)
    if not context.strip():
        return {"error": "NO_MATCHES", "message": "No relevant passages found for the query."}

    # 2) Create LLM (read API key from env)
    api_key = "AIzaSyBEhYLJlghVrQSKz4HnMpxmTqLDP7IkiLg"
    if not api_key:
        raise RuntimeError("Set GOOGLE_API_KEY environment variable.")
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=api_key, temperature=0.4)

    # 3) Create ConversationBufferMemory and seed it with DB history
    memory = ConversationBufferMemory(memory_key="history",  input_key="query", return_messages=True)
    _load_history_into_memory(session_id, memory)

    # 4) Create LLMChain with memory
    chain = LLMChain(llm=llm, prompt=prompt, memory=memory, verbose=True)

    # 5) Before running, persist the user's message to DB (so_history includes it next time)
    _save_message_to_db(session_id=session_id, role="user", content=query)

    # 6) Run chain
    # Note: using .predict ensures prompt variables are passed in
    answer_text = chain.predict(context=context, query=query)

    # 7) Save assistant reply to DB (and memory automatically updated by chain)
    _save_message_to_db(session_id=session_id, role="assistant", content=answer_text)

    # 8) Optionally extract supporting evidence (we can pick first 1-3 doc snippets that best match)
    supporting = []
    for d in docs[:3]:
        supporting.append(getattr(d, "page_content", "")[:500])  # short excerpt
        
    # SAVE CHAT INTO DATABASE
    db = SessionLocal()
    try:
        # save user message
        db.add(ChatMessage(
            session_id=session_id,
            role="user",
            content=query
        ))

        # save assistant response
        db.add(ChatMessage(
            session_id=session_id,
            role="assistant",
            content=answer_text
        ))

        db.commit()
    finally:
        db.close()

    return {
        "answer": answer_text,
        "supporting_evidence": supporting,
        "retrieved_count": len(docs)
    }