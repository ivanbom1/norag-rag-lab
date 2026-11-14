import os
import shutil
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, UploadFile
from fastapi.responses import PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app import graph_retriever, reranking
from app.answering import DEFAULT_RETRIEVERS, Retriever, ask, retrieve
from app.evaluation import check_quotes, load_questions, run_evaluation
from app.indexing import ChunkSettings, delete_document, ensure_indexed, index_document
from app.words import question_words, vocabulary, word_chunks

CORPORA_DIR = "data/corpora"
SEED_DIRS = {"monte-cristo": "corpus", "kestrel": "kestrel-kb/docs"}
ALLOWED_EXTENSIONS = {".md", ".txt"}

settings = ChunkSettings()
top_k = 5

for corpus, seed in SEED_DIRS.items():
    if not os.path.isdir(os.path.join(CORPORA_DIR, corpus)):
        shutil.copytree(seed, os.path.join(CORPORA_DIR, corpus), ignore=shutil.ignore_patterns(".*"))


def corpus_dir(corpus):
    if corpus not in SEED_DIRS:
        raise HTTPException(404, f"Unknown corpus: {corpus}")
    return os.path.join(CORPORA_DIR, corpus)


def document_names(corpus):
    return sorted(os.listdir(corpus_dir(corpus)))


def index_missing_documents():
    indexed = []
    for corpus in SEED_DIRS:
        for name in document_names(corpus):
            if ensure_indexed(os.path.join(corpus_dir(corpus), name), corpus, settings):
                indexed.append(f"{corpus}/{name}")
    return indexed


@asynccontextmanager
async def lifespan(app):
    index_missing_documents()
    try:
        reranking.load_model()
    except Exception as error:
        print(f"Reranking model not loaded, reranking will fall back to the search order: {error}")
    yield


app = FastAPI(title="Monte Cristo RAG", lifespan=lifespan)


class SettingsUpdate(BaseModel):
    chunk_size: int = Field(ge=100, le=4000)
    overlap_percent: int = Field(ge=0, le=50)
    top_k: int = Field(ge=1, le=20)


def settings_response():
    return {"chunk_size": settings.chunk_size, "overlap_percent": settings.overlap_percent, "top_k": top_k,
            "retrievers": DEFAULT_RETRIEVERS, "rerank": reranking.DEFAULT_RERANK,
            "graphs": {corpus: graph_retriever.available(corpus, settings) for corpus in SEED_DIRS}}


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


@app.get("/api/corpora")
def list_corpora():
    return list(SEED_DIRS)


class Question(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    documents: list[str] | None = None
    retrievers: list[Retriever] = Field(default=DEFAULT_RETRIEVERS, min_length=1)
    rerank: bool = reranking.DEFAULT_RERANK


def selected_sources(corpus, documents):
    available = document_names(corpus)
    sources = available if documents is None else documents
    unknown = [name for name in sources if name not in available]
    if unknown:
        raise HTTPException(400, f"Unknown documents: {', '.join(unknown)}")
    if not sources:
        raise HTTPException(400, "Select at least one document.")
    return sources


def check_retrievers(corpus, retrievers):
    if "graph" in retrievers and not graph_retriever.available(corpus, settings):
        raise HTTPException(400, f"No graph for {corpus} at chunk size {settings.chunk_size} / "
                                 f"{settings.overlap_percent} % overlap. Build it with python -m app.graph.")


@app.post("/api/corpora/{corpus}/ask")
def ask_question(corpus: str, body: Question):
    check_retrievers(corpus, body.retrievers)
    return ask(body.question, corpus, selected_sources(corpus, body.documents), settings, top_k, body.retrievers,
               body.rerank)


class SearchQuery(Question):
    top_k: int | None = Field(default=None, ge=1, le=20)


@app.post("/api/corpora/{corpus}/search")
def search(corpus: str, body: SearchQuery):
    sources = selected_sources(corpus, body.documents)
    check_retrievers(corpus, body.retrievers)
    return {"retrieved": retrieve(body.question, corpus, sources, settings, body.top_k or top_k, body.retrievers,
                                  body.rerank)}


class EvaluationRequest(BaseModel):
    answers: bool = False
    retrievers: list[Retriever] = Field(default=DEFAULT_RETRIEVERS, min_length=1)
    rerank: bool = reranking.DEFAULT_RERANK


@app.post("/api/corpora/{corpus}/evaluate")
def evaluate(corpus: str, body: EvaluationRequest):
    directory = corpus_dir(corpus)
    check_retrievers(corpus, body.retrievers)
    problems = check_quotes(load_questions(corpus), directory)
    if problems:
        raise HTTPException(400, "Question set does not match the corpus: " + "; ".join(problems))
    path, report = run_evaluation(corpus, document_names(corpus), settings, body.answers, top_k,
                                  delay=4 if body.answers else 0, retrievers=body.retrievers, rerank=body.rerank)
    return {"path": path, **report}


@app.get("/api/corpora/{corpus}/questions")
def list_questions(corpus: str):
    corpus_dir(corpus)
    return [{key: question[key] for key in ("id", "type", "question", "answerable")} for question in load_questions(corpus)]


@app.get("/api/corpora/{corpus}/words")
def get_vocabulary(corpus: str, contains: str = "", stopwords: bool = False, limit: int = Query(200, ge=1, le=5000)):
    corpus_dir(corpus)
    return vocabulary(corpus, settings, contains, stopwords, limit)


@app.get("/api/corpora/{corpus}/words/{word}")
def get_word_chunks(corpus: str, word: str):
    corpus_dir(corpus)
    return word_chunks(corpus, settings, word)


class WordsQuestion(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    question_id: str | None = None


@app.post("/api/corpora/{corpus}/words/question")
def get_question_words(corpus: str, body: WordsQuestion):
    corpus_dir(corpus)
    evidence = None
    if body.question_id:
        match = [q for q in load_questions(corpus) if q["id"] == body.question_id]
        if not match:
            raise HTTPException(404, f"Unknown question id: {body.question_id}")
        evidence = match[0]["evidence"]
    return question_words(corpus, settings, body.question, evidence)


def document_path(corpus, name):
    if name != os.path.basename(name) or os.path.splitext(name)[1].lower() not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, "Only .md and .txt file names are accepted.")
    return os.path.join(corpus_dir(corpus), name)


@app.get("/api/corpora/{corpus}/documents")
def list_documents(corpus: str):
    return [
        {"name": name, "size": os.path.getsize(os.path.join(corpus_dir(corpus), name))}
        for name in document_names(corpus)
    ]


@app.post("/api/corpora/{corpus}/documents", status_code=201)
def upload_document(corpus: str, file: UploadFile):
    path = document_path(corpus, file.filename)
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
    return {"name": file.filename, "chunks": index_document(path, corpus, settings)}


@app.get("/api/corpora/{corpus}/documents/{name}/text", response_class=PlainTextResponse)
def document_text(corpus: str, name: str):
    path = document_path(corpus, name)
    if not os.path.exists(path):
        raise HTTPException(404, f"{name} not found.")
    with open(path, encoding="utf-8") as f:
        return f.read()


@app.delete("/api/corpora/{corpus}/documents/{name}", status_code=204)
def remove_document(corpus: str, name: str):
    path = document_path(corpus, name)
    if not os.path.exists(path):
        raise HTTPException(404, f"{name} not found.")
    os.remove(path)
    delete_document(name, corpus)


app.mount("/", StaticFiles(directory="app/static", html=True), name="static")
