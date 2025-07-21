import fastapi
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
import pdfplumber
import io
import asyncio
import aiohttp
import functools
from datetime import datetime
from typing import Optional, Tuple, List
import logging
import re
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import requests

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Intelligent Query Analyzer API")

# Gemini API config – replace with your actual key/URL
GEMINI_API_KEY = "YOUR_GEMINI_KEY"
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

# Embedding model & FAISS
embedder = SentenceTransformer('all-MiniLM-L6-v2')
index = faiss.IndexFlatL2(384)

# Caches
@functools.lru_cache(maxsize=100)
def cache_document_summary(file_name: str, summary: str) -> str:
    return summary

@functools.lru_cache(maxsize=100)
def cache_document_embeddings(file_name: str, embeddings_tuple: tuple) -> tuple:
    return embeddings_tuple

async def query_llm_async(text: str, task: str) -> str:
    prompt = (
        f"{task.capitalize()} the following text concisely in 100 words or less:\n{text[:2000]}..."
        if task.lower().startswith(("summarize", "overview", "key points"))
        else f"Answer the query based on the text in 100 words or less:\nText: {text[:2000]}...\nQuery: {task}"
    )
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"maxOutputTokens": 150}
    }
    headers = {"Content-Type": "application/json"}
    async with aiohttp.ClientSession() as session:
        async with session.post(f"{GEMINI_API_URL}?key={GEMINI_API_KEY}", json=payload, headers=headers) as resp:
            if resp.status != 200:
                err = await resp.text()
                logger.error(f"Gemini error: {err}")
                raise HTTPException(status_code=500, detail="LLM error")
            data = await resp.json()
            return data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")

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

def chunk_text(text: str, max_words: int = 500) -> List[str]:
    words = re.findall(r'\S+', text)
    chunks, current, count = [], [], 0
    for w in words:
        current.append(w)
        count += 1
        if count >= max_words:
            chunks.append(" ".join(current))
            current, count = [], 0
    if current: chunks.append(" ".join(current))
    return chunks

def generate_embeddings(chunks: List[str]) -> np.ndarray:
    return embedder.encode(chunks)

async def summarize_document(text: str) -> str:
    chunks = chunk_text(text)
    res = await asyncio.gather(*[query_llm_async(c, "summarize") for c in chunks])
    return " ".join(r for r in res if r)

def generate_report(insights: str, query: str, output_format: str = "markdown") -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if output_format == "markdown":
        return f"# Report\n\n**Generated:** {ts}\n\n**Query:** {query}\n\n**Insights:**\n{insights}"
    if output_format == "csv":
        import pandas as pd
        return pd.DataFrame({"Query": [query], "Insights": [insights], "Timestamp": [ts]}).to_csv(index=False)
    return insights

@app.post("/analyze")
async def analyze_document(
    file: UploadFile = File(...),
    user_query: str = Form(...),
    output_format: str = Form("markdown")
):
    content = await file.read()
    text, err = extract_pdf_text(content) if file.content_type.startswith("application/pdf") else extract_text_file(content)
    if err or not text:
        raise HTTPException(status_code=400, detail=err or "Empty document")

    summary = await summarize_document(text)
    cache_document_summary(file.filename, summary)

    chunks = chunk_text(text)
    embeddings = generate_embeddings(chunks)
    index.reset()
    index.add(np.array(embeddings))
    cache_document_embeddings(file.filename, tuple(embeddings.flatten()))

    if user_query.lower().startswith(("summarize", "overview", "key points")):
        insights = await query_llm_async(summary, user_query)
    else:
        q_emb = embedder.encode([user_query])
        dists, idxs = index.search(np.array(q_emb), k=3)
        ctx = " ".join(chunks[i] for i in idxs[0])
        insights = await query_llm_async(ctx, user_query)

    report = generate_report(insights, user_query, output_format)
    return {"status": "success", "query": user_query, "insights": insights, "report": report, "format": output_format}

# New required endpoint
class HackRxReq(BaseModel):
    documents: str
    questions: List[str]

class HackRxRes(BaseModel):
    answers: List[str]

@app.post("/hackrx/run", response_model=HackRxRes)
async def hackrx_run(req: HackRxReq):
    try:
        r = requests.get(req.documents)
        r.raise_for_status()
        text, err = extract_pdf_text(r.content)
        if err or not text:
            raise HTTPException(status_code=400, detail="Error fetching or parsing document")

        chunks = chunk_text(text)
        embeddings = generate_embeddings(chunks)
        index.reset()
        index.add(np.array(embeddings))

        answers = []
        for q in req.questions:
            if q.lower().startswith(("summarize", "overview", "key points")):
                summary = await summarize_document(text)
                ans = await query_llm_async(summary, q)
            else:
                q_emb = embedder.encode([q])
                _, idxs = index.search(np.array(q_emb), k=3)
                ctx = " ".join(chunks[i] for i in idxs[0])
                ans = await query_llm_async(ctx, q)
            answers.append(ans)

        return {"answers": answers}
    except Exception as e:
        logger.error(e)
        raise HTTPException(status_code=500, detail="HackRx failure")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
