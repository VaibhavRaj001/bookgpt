# app/main.py
import os
import uuid
import shutil
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from db import init_db, SessionLocal, ChatSession, ChatMessage

from ingest.ingest_book import ingest_book
from rag.chat import chat_with_book_chain

app = FastAPI()

# Initialize database tables
init_db()


# ------------------- 1) UPLOAD BOOK -------------------
@app.post("/books/upload")
async def upload_book(file: UploadFile = File(...)):
    try:
        os.makedirs("storage/uploads", exist_ok=True)

        save_path = f"storage/uploads/{file.filename}"
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        result = ingest_book(save_path)

        book_id = result["book_id"] if isinstance(result, dict) else result[0]

        return {"message": "Book uploaded!", "book_id": book_id}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# ------------------- 2) START CHAT SESSION -------------------
@app.post("/sessions/start")
def start_session(book_id: str = Form(...), user_id: str | None = None):
    # ensure book exists
    persist_dir = os.path.join("storage", "chroma_db", book_id)
    if not os.path.isdir(persist_dir):
        raise HTTPException(status_code=404, detail="Book not found. Did you upload it?")

    session_id = str(uuid.uuid4())

    db = SessionLocal()
    try:
        s = ChatSession(id=session_id, book_id=book_id, user_id=user_id)
        db.add(s)
        db.commit()
    finally:
        db.close()

    return {"session_id": session_id, "book_id": book_id}


# ------------------- 3) CHAT WITH MEMORY -------------------
@app.post("/sessions/{session_id}/chat")
def chat_with_session(session_id: str, query: str = Form(...)):
    # locate session in DB
    db = SessionLocal()
    try:
        sess = db.query(ChatSession).filter(ChatSession.id == session_id).first()
        if not sess:
            raise HTTPException(status_code=404, detail="Session not found")
        book_id = sess.book_id
    finally:
        db.close()

    # run chain with memory
    result = chat_with_book_chain(book_id=book_id, session_id=session_id, query=query)

    if "error" in result:
        raise HTTPException(status_code=500, detail=result["message"])

    return result

@app.get("/sessions/{session_id}/history")
def get_history(session_id: str):
    db = SessionLocal()
    try:
        msgs = db.query(ChatMessage).filter(
            ChatMessage.session_id == session_id
        ).order_by(ChatMessage.created_at).all()

        return [
            {"role": m.role, "content": m.content, "time": m.created_at}
            for m in msgs
        ]
    finally:
        db.close()
