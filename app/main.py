import os
import shutil

from fastapi import FastAPI, HTTPException, UploadFile

from app.indexing import ChunkSettings, delete_document, index_document

DOCUMENTS_DIR = "data/documents"
SEED_DIR = "corpus"
ALLOWED_EXTENSIONS = {".md", ".txt"}

settings = ChunkSettings()

if not os.path.isdir(DOCUMENTS_DIR):
    shutil.copytree(SEED_DIR, DOCUMENTS_DIR)

app = FastAPI(title="Monte Cristo RAG")


def document_path(name):
    if name != os.path.basename(name) or os.path.splitext(name)[1].lower() not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, "Only .md and .txt file names are accepted.")
    return os.path.join(DOCUMENTS_DIR, name)


@app.get("/api/documents")
def list_documents():
    return [
        {"name": name, "size": os.path.getsize(os.path.join(DOCUMENTS_DIR, name))}
        for name in sorted(os.listdir(DOCUMENTS_DIR))
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


@app.delete("/api/documents/{name}", status_code=204)
def remove_document(name: str):
    path = document_path(name)
    if not os.path.exists(path):
        raise HTTPException(404, f"{name} not found.")
    os.remove(path)
    delete_document(name)
