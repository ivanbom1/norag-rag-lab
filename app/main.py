import os
import shutil
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.answering import ask
from app.indexing import ChunkSettings, delete_document, index_document, is_indexed

DOCUMENTS_DIR = "data/documents"
SEED_DIR = "corpus"
ALLOWED_EXTENSIONS = {".md", ".txt"}

settings = ChunkSettings()
top_k = 5

if not os.path.isdir(DOCUMENTS_DIR):
    shutil.copytree(SEED_DIR, DOCUMENTS_DIR)


def document_names():
    return sorted(os.listdir(DOCUMENTS_DIR))


def index_missing_documents():
    indexed = []
    for name in document_names():
        if not is_indexed(name, settings):
            index_document(os.path.join(DOCUMENTS_DIR, name), settings)
            indexed.append(name)
    return indexed


@asynccontextmanager
async def lifespan(app):
    index_missing_documents()
    yield


app = FastAPI(title="Monte Cristo RAG", lifespan=lifespan)


class SettingsUpdate(BaseModel):
    chunk_size: int = Field(ge=100, le=4000)
    overlap_percent: int = Field(ge=0, le=50)
    top_k: int = Field(ge=1, le=20)


def settings_response():
    return {"chunk_size": settings.chunk_size, "overlap_percent": settings.overlap_percent, "top_k": top_k}


@app.get("/api/settings")
def get_settings():
    return settings_response()


@app.put("/api/settings")
def update_settings(update: SettingsUpdate):
    global settings, top_k
    settings = ChunkSettings(chunk_size=update.chunk_size, overlap_percent=update.overlap_percent)
    top_k = update.top_k
    started = time.time()
    indexed = index_missing_documents()
    return {**settings_response(), "reindexed": indexed, "seconds": round(time.time() - started, 1)}


class Question(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    documents: list[str] | None = None


@app.post("/api/ask")
def ask_question(body: Question):
    available = document_names()
    sources = available if body.documents is None else body.documents
    unknown = [name for name in sources if name not in available]
    if unknown:
        raise HTTPException(400, f"Unknown documents: {', '.join(unknown)}")
    if not sources:
        raise HTTPException(400, "Select at least one document.")
    return ask(body.question, sources, settings, top_k)


def document_path(name):
    if name != os.path.basename(name) or os.path.splitext(name)[1].lower() not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, "Only .md and .txt file names are accepted.")
    return os.path.join(DOCUMENTS_DIR, name)


@app.get("/api/documents")
def list_documents():
    return [
        {"name": name, "size": os.path.getsize(os.path.join(DOCUMENTS_DIR, name))}
        for name in document_names()
    ]


@app.post("/api/documents", status_code=201)
def upload_document(file: UploadFile):
    path = document_path(file.filename)
    if os.path.exists(path):
        raise HTTPException(409, f"{file.filename} already exists. Delete it first.")
    try:
        text = file.file.read().decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(400, "The file is not valid UTF-8 text.")
    if not text.strip():
        raise HTTPException(400, "The file is empty.")

    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    return {"name": file.filename, "chunks": index_document(path, settings)}


@app.get("/api/documents/{name}/text", response_class=PlainTextResponse)
def document_text(name: str):
    path = document_path(name)
    if not os.path.exists(path):
        raise HTTPException(404, f"{name} not found.")
    with open(path, encoding="utf-8") as f:
        return f.read()


@app.delete("/api/documents/{name}", status_code=204)
def remove_document(name: str):
    path = document_path(name)
    if not os.path.exists(path):
        raise HTTPException(404, f"{name} not found.")
    os.remove(path)
    delete_document(name)


app.mount("/", StaticFiles(directory="app/static", html=True), name="static")
