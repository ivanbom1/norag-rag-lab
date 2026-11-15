"""Traces: one per question asked in the app and one per indexing run, in a local SQLite file.

A trace is {trace_id, kind, created, status, duration_ms, ..., spans: [{name, duration_ms, ...}]}. It never holds
chunk text, prompts or keys: chunks are referred to by id, and questions and answers are stored redacted.
"""
import json
import os
import sqlite3
import uuid
from datetime import datetime, timedelta
from time import perf_counter

TRACE_DB = "./traces/traces.sqlite3"
RETENTION_DAYS = 30
# USD per million tokens (input, output), list price from https://ai.google.dev/gemini-api/docs/pricing (Nov 2026).
# Our free-tier key is not billed: this is what the call would cost on a paid key.
PRICES = {"gemini-3.5-flash-lite": (0.30, 2.50)}


def llm_cost(model, prompt_tokens, completion_tokens):
    price_in, price_out = PRICES.get(model, (0, 0))
    return round((prompt_tokens * price_in + completion_tokens * price_out) / 1_000_000, 6)


class Trace:
    def __init__(self, kind, **fields):
        self.start = perf_counter()
        self.data = {"trace_id": uuid.uuid4().hex, "kind": kind, "created": datetime.now().isoformat(timespec="seconds"),
                     "status": None, "duration_ms": None, **fields, "spans": []}

    @property
    def id(self):
        return self.data["trace_id"]

    def add_span(self, name, duration_ms, **fields):
        self.data["spans"].append({"name": name, "duration_ms": duration_ms, **fields})


def connect():
    os.makedirs(os.path.dirname(TRACE_DB), exist_ok=True)
    connection = sqlite3.connect(TRACE_DB)
    connection.execute("CREATE TABLE IF NOT EXISTS traces (trace_id TEXT PRIMARY KEY, kind TEXT, created TEXT, "
                       "status TEXT, duration_ms REAL, data TEXT)")
    return connection


def save(trace, status):
    trace.data["status"] = status
    trace.data["duration_ms"] = round((perf_counter() - trace.start) * 1000, 1)
    with connect() as connection:
        connection.execute("INSERT INTO traces VALUES (?, ?, ?, ?, ?, ?)",
                           (trace.id, trace.data["kind"], trace.data["created"], status, trace.data["duration_ms"],
                            json.dumps(trace.data, ensure_ascii=False)))
    return trace.id


def get(trace_id):
    with connect() as connection:
        row = connection.execute("SELECT data FROM traces WHERE trace_id = ?", (trace_id,)).fetchone()
    return json.loads(row[0]) if row else None


def recent(kind=None, limit=20):
    """Newest first, without the spans."""
    query = "SELECT trace_id, kind, created, status, duration_ms FROM traces"
    query += " WHERE kind = ?" if kind else ""
    with connect() as connection:
        rows = connection.execute(query + " ORDER BY created DESC LIMIT ?", (*([kind] if kind else []), limit)).fetchall()
    return [dict(zip(("trace_id", "kind", "created", "status", "duration_ms"), row)) for row in rows]


def delete_old(days=RETENTION_DAYS):
    cutoff = (datetime.now() - timedelta(days=days)).isoformat(timespec="seconds")
    with connect() as connection:
        return connection.execute("DELETE FROM traces WHERE created < ?", (cutoff,)).rowcount
