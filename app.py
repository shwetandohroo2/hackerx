import os
import io
import re
import faiss
import pdfplumber
import logging
import asyncio
import aiohttp
import functools
import numpy as np
from datetime import datetime
from typing import List, Optional, Tuple
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Memory-Optimized Query Analyzer")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
GEMINI_API_URL = os.environ.get("base_url")

# Lazy init
embedder = None
index = None

def get_embedder():
    global embedder
    if embedder is None:
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer("distiluse-base-multilingual-cased-v1")
    return embedder

def get_index():
    global index
    if index is None:
        index = faiss.IndexFlatL2(512)
    return index

def extract_pdf_text(file_bytes: bytes) -> Tuple[Optional[str], Optional[str]]:
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            return "".join(page.extract_text() or "" for page in pdf.pages), None
    except Exception as e:
        return None, str(e)

def extract_text_file(file_bytes: bytes) -> Tuple[Optional[str], Optional[str]]:
    try:
        return file_bytes.decode("utf-8"), None
    except Exception as e:
        return None, str(e)

def chunk_text(text: str, max_words: int = 300) -> List[str]:
    words = re.findall(r'\S+', text)
    chunks, current = [], []
    for i, word in enumerate(words):
        current.append(word)
        if len(current) >= max_words:
            chunks.append(" ".join(current))
            current = []
    if current:
        chunks.append(" ".join(current))
    return chunks

async def query_llm_async(text: str, task: str) -> str:
    prompt = (
        f"{task.capitalize()} the following text in under 100 words:\n{text[:2000]}"
        if task.lower().startswith(("summarize", "overview", "key points"))
        else f"Answer the question based on the text below in under 100 words:\nText:\n{text[:2000]}\n\nQuestion: {task}"
    )
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"maxOutputTokens": 150}
    }
    headers = {"Content-Type": "application/json"}
    async with aiohttp.ClientSession() as session:
        async with session.post(f"{GEMINI_API_URL}?key={OPENAI_API_KEY}", json=payload, headers=headers) as resp:
            if resp.status != 200:
                logger.error(await resp.text())
                raise HTTPException(status_code=500, detail="LLM call failed")
            data = await resp.json()
            return data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")

async def summarize_document(text: str) -> str:
    chunks = chunk_text(text)
    res = await asyncio.gather(*[query_llm_async(c, "summarize") for c in chunks])
    return " ".join(r for r in res if r)

def embed_chunks(chunks: List[str]) -> np.ndarray:
    return get_embedder().encode(chunks)

class HackRxReq(BaseModel):
    documents: str
    questions: List[str]

class HackRxRes(BaseModel):
    answers: List[str]

@app.post("/hackrx/run", response_model=HackRxRes)
async def hackrx_run(req: HackRxReq):
    r = requests.get(req.documents)
    r.raise_for_status()
    text, err = extract_pdf_text(r.content)
    if err or not text:
        raise HTTPException(status_code=400, detail="Document parse failed")

    chunks = chunk_text(text)
    embeddings = embed_chunks(chunks)
    get_index().reset()
    get_index().add(np.array(embeddings))

    answers = []
    for q in req.questions:
        if q.lower().startswith(("summarize", "overview", "key points")):
            summary = await summarize_document(text)
            answer = await query_llm_async(summary, q)
        else:
            q_emb = get_embedder().encode([q])
            _, idxs = get_index().search(np.array(q_emb), k=3)
            ctx = " ".join([chunks[i] for i in idxs[0]])
            answer = await query_llm_async(ctx, q)
        answers.append(answer)

    return {"answers": answers}
