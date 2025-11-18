import os
import pickle
from functools import lru_cache
from typing import Optional, Literal

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
import httpx

from reader3 import Book, BookMetadata, ChapterContent, TOCEntry

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Where are the book folders located?
BOOKS_DIR = "."

# LLM configuration (defaults target a local Ollama instance)
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "60"))
MAX_CONTEXT_CHARS = int(os.getenv("LLM_CONTEXT_LIMIT", "8000"))


class LLMQueryPayload(BaseModel):
    question: Optional[str] = Field(default=None, description="User prompt for QA mode")
    context: Optional[str] = Field(default="", description="Plain text context to send to the LLM")
    mode: Literal["qa", "summary"] = Field(default="qa")
    book_id: Optional[str] = None
    chapter_index: Optional[int] = None


def _truncate_context(text: str) -> str:
    if not text:
        return ""
    if len(text) <= MAX_CONTEXT_CHARS:
        return text
    return text[:MAX_CONTEXT_CHARS] + "\n\n[Context truncated]"


async def _call_ollama(prompt: str) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a concise, thoughtful reading companion. "
                    "Always ground answers in the provided context. "
                    "If the context lacks the answer, say you do not know."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "options": {
            "temperature": 0.2,
        },
    }

    async with httpx.AsyncClient(timeout=LLM_TIMEOUT) as client:
        try:
            response = await client.post(f"{OLLAMA_HOST}/api/chat", json=payload)
            response.raise_for_status()
        except httpx.RequestError as exc:
            raise HTTPException(status_code=502, detail=f"Could not reach LLM backend: {exc}") from exc
        except httpx.HTTPStatusError as exc:
            raise HTTPException(status_code=502, detail=f"LLM backend error: {exc.response.text}") from exc

    data = response.json()
    message = data.get("message", {})
    content = message.get("content") or data.get("response")
    if not content:
        raise HTTPException(status_code=502, detail="LLM backend returned an empty response")
    return content.strip()


def _build_prompt(question: str, context: str, mode: str, book_title: Optional[str]) -> str:
    header = "You are helping someone read a book."
    if book_title:
        header = f"You are helping someone read '{book_title}'."

    if mode == "summary":
        task = (
            question
            or "Summarize the provided passage into 5-7 concise bullet points that capture key ideas."
        )
    else:
        task = question or "Answer the reader's question based on the context."

    return (
        f"{header}\n\n"
        f"Context:\n{context}\n\n"
        f"Task:\n{task}\n\n"
        "When answering, be direct and cite details from the context. "
        "If information is missing, say you don't know."
    )

@lru_cache(maxsize=10)
def load_book_cached(folder_name: str) -> Optional[Book]:
    """
    Loads the book from the pickle file.
    Cached so we don't re-read the disk on every click.
    """
    file_path = os.path.join(BOOKS_DIR, folder_name, "book.pkl")
    if not os.path.exists(file_path):
        return None

    try:
        with open(file_path, "rb") as f:
            book = pickle.load(f)
        return book
    except Exception as e:
        print(f"Error loading book {folder_name}: {e}")
        return None

@app.get("/", response_class=HTMLResponse)
async def library_view(request: Request):
    """Lists all available processed books."""
    books = []

    # Scan directory for folders ending in '_data' that have a book.pkl
    if os.path.exists(BOOKS_DIR):
        for item in os.listdir(BOOKS_DIR):
            if item.endswith("_data") and os.path.isdir(item):
                # Try to load it to get the title
                book = load_book_cached(item)
                if book:
                    books.append({
                        "id": item,
                        "title": book.metadata.title,
                        "author": ", ".join(book.metadata.authors),
                        "chapters": len(book.spine)
                    })

    return templates.TemplateResponse("library.html", {"request": request, "books": books})

@app.get("/read/{book_id}", response_class=HTMLResponse)
async def redirect_to_first_chapter(book_id: str):
    """Helper to just go to chapter 0."""
    return await read_chapter(book_id=book_id, chapter_index=0)

@app.get("/read/{book_id}/{chapter_index}", response_class=HTMLResponse)
async def read_chapter(request: Request, book_id: str, chapter_index: int):
    """The main reader interface."""
    book = load_book_cached(book_id)
    if not book:
        raise HTTPException(status_code=404, detail="Book not found")

    if chapter_index < 0 or chapter_index >= len(book.spine):
        raise HTTPException(status_code=404, detail="Chapter not found")

    current_chapter = book.spine[chapter_index]

    # Calculate Prev/Next links
    prev_idx = chapter_index - 1 if chapter_index > 0 else None
    next_idx = chapter_index + 1 if chapter_index < len(book.spine) - 1 else None

    return templates.TemplateResponse("reader.html", {
        "request": request,
        "book": book,
        "current_chapter": current_chapter,
        "chapter_index": chapter_index,
        "book_id": book_id,
        "prev_idx": prev_idx,
        "next_idx": next_idx
    })

@app.get("/read/{book_id}/images/{image_name}")
async def serve_image(book_id: str, image_name: str):
    """
    Serves images specifically for a book.
    The HTML contains <img src="images/pic.jpg">.
    The browser resolves this to /read/{book_id}/images/pic.jpg.
    """
    # Security check: ensure book_id is clean
    safe_book_id = os.path.basename(book_id)
    safe_image_name = os.path.basename(image_name)

    img_path = os.path.join(BOOKS_DIR, safe_book_id, "images", safe_image_name)

    if not os.path.exists(img_path):
        raise HTTPException(status_code=404, detail="Image not found")

    return FileResponse(img_path)


@app.post("/api/llm/query")
async def query_llm(payload: LLMQueryPayload):
    if payload.mode == "qa":
        if not payload.question or not payload.question.strip():
            raise HTTPException(status_code=400, detail="Question is required for QA mode.")

    context = (payload.context or "").strip()
    book_title = None

    if payload.book_id:
        book = load_book_cached(payload.book_id)
        if book:
            book_title = book.metadata.title
            if (not context) and payload.chapter_index is not None:
                if 0 <= payload.chapter_index < len(book.spine):
                    context = book.spine[payload.chapter_index].text

    if not context:
        raise HTTPException(status_code=400, detail="No context provided for the LLM.")

    context = _truncate_context(context)
    question = payload.question.strip() if payload.question else ""
    prompt = _build_prompt(question, context, payload.mode, book_title)

    answer = await _call_ollama(prompt)
    return {
        "answer": answer,
        "model": OLLAMA_MODEL,
        "used_chars": len(context),
    }

if __name__ == "__main__":
    import uvicorn
    print("Starting server at http://127.0.0.1:8123")
    uvicorn.run(app, host="127.0.0.1", port=8123)
