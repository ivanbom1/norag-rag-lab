import json
import os
import re

from dotenv import load_dotenv
from openai import BadRequestError, OpenAI

load_dotenv(".env.local")
load_dotenv(".env")

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

LLM_URL = os.getenv("LLM_URL") or GEMINI_URL
LLM_API_KEY = os.getenv("LLM_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
LLM_MODEL = os.getenv("LLM_MODEL") or "gemini-3.5-flash-lite"
LLM_GRAPH_MODEL = os.getenv("LLM_GRAPH_MODEL") or (
    "gemini-3-flash-preview" if LLM_URL == GEMINI_URL and LLM_MODEL == "gemini-3.5-flash-lite" else LLM_MODEL)

client = OpenAI(base_url=LLM_URL, api_key=LLM_API_KEY or "not-needed")


def chat(messages, model=None, schema=None, schema_name="response", **options):
    model = model or LLM_MODEL
    if schema is None:
        return client.chat.completions.create(model=model, messages=messages, **options)
    try:
        return client.chat.completions.create(
            model=model, messages=messages,
            response_format={"type": "json_schema", "json_schema": {"name": schema_name, "schema": schema}}, **options)
    except BadRequestError:
        instruction = {"role": "user", "content": "Reply with one JSON object only, with no other text, "
                                                  f"matching this JSON schema:\n{json.dumps(schema)}"}
        return client.chat.completions.create(model=model, messages=[*messages, instruction], **options)


def json_text(response):
    text = response.choices[0].message.content or ""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fenced:
        text = fenced.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    return text[start:end + 1] if start != -1 and end > start else text


def describe():
    return {"url": LLM_URL, "model": LLM_MODEL, "graph_model": LLM_GRAPH_MODEL, "api_key_set": bool(LLM_API_KEY)}
