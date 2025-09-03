import io
import os
import re
import hashlib
from typing import List, Dict
from fastapi import Form
import json
import requests
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from pypdf import PdfReader
from dotenv import load_dotenv

from rag_engine import RAGEngine

load_dotenv() 
# =========================== Hugging Face Inference API (LLM) ===========================
# 1) Get a free token at: https://huggingface.co/settings/tokens
# 2) Set it in your terminal before starting the server:
#    macOS/Linux: export HF_API_KEY="hf_xxx"
#    Windows (PowerShell): setx HF_API_KEY "hf_xxx"  (then reopen terminal)
HF_API_KEY = os.getenv("HF_API_KEY", "")  # DO NOT hardcode secrets in production
HF_MODEL = os.getenv("HF_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")
HF_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL}"

def hf_generate(system_prompt: str, user_prompt: str,
                max_new_tokens: int = 512, temperature: float = 0.3) -> str:
    """
    Call Hugging Face Inference API with an instruction-style prompt.
    Raises when token is missing or API returns error; caller may fallback.
    """
    if not HF_API_KEY:
        raise RuntimeError("HF_API_KEY is not set. export HF_API_KEY=your_token first.")

    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    prompt = f"{system_prompt.strip()}\n\n{user_prompt.strip()}"
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature
        }
    }
    resp = requests.post(HF_URL, headers=headers, json=payload, timeout=120)
    data = resp.json()
    if resp.status_code != 200:
        # Common cases: 503 (model loading), 429 (rate limit)
        raise RuntimeError(f"HuggingFace API error {resp.status_code}: {data}")
    # Most text-generation endpoints return: [{"generated_text": "..."}]
    if isinstance(data, list) and data and "generated_text" in data[0]:
        out = data[0]["generated_text"]
        # Some providers echo the prompt; strip if present
        return out[len(prompt):].strip() if out.startswith(prompt) else out.strip()
    return str(data).strip()

# =========================== Prompt (RAG with strict citations) ===========================
PROMPT_SYSTEM = (
    "You are a precise assistant. Answer ONLY using the provided context.\n"
    "If the answer is not in the context, reply exactly: 'I could not find the answer in the textbook.'\n"
    "Always include 1–3 inline citations in the form (source|page[#chunk]) when possible."
)

PROMPT_TEMPLATE = """Context:
{context}

Task:
- Provide a concise answer (≤ 6 sentences) strictly based on the context.
- Use plain English.
- Include 1–3 citations like (source|p#) when applicable.
- If the context does not contain the answer, reply exactly:
  "I could not find the answer in the textbook."

Question: {question}
Answer:
"""

def force_citations(answer: str, sources: List[str]) -> str:
    """Append 1–2 sources if LLM forgot to cite (safety net)."""
    if "(" in answer and ")" in answer:
        return answer
    if sources:
        tail = " ".join(f"({s})" for s in sources[:2])
        return f"{answer} {tail}"
    return answer

# =========================== Chunking helpers ===========================
def split_text(text: str, size: int = 700, overlap: int = 120) -> List[str]:
    """
    Split long text into overlapping chunks to improve recall.
    """
    text = (text or "").strip()
    if not text:
        return []
    chunks, i, n = [], 0, len(text)
    step = max(1, size - overlap)
    while i < n:
        chunks.append(text[i : i + size])
        i += step
    return chunks

def hash_text(s: str) -> str:
    """Lightweight deduplication key."""
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

# =========================== FastAPI app ===========================
app = FastAPI(title="RAG + HuggingFace LLM Q&A")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag = RAGEngine()

@app.on_event("startup")
def _load_if_exists():
    try:
        if os.path.exists("faiss_index.bin") and os.path.exists("metadata.json"):
            rag.load("faiss_index.bin", "metadata.json")
            print("[startup] Loaded existing index & metadata.")
    except Exception as e:
        print(f"[startup] Load failed: {e}")

class QuestionRequest(BaseModel):
    question: str
    top_k: int = 3
    min_score: float = 0.30

class FeedbackRequest(BaseModel):
    question: str
    correct_answer: str
    wrong_answer: str

@app.get("/health")
def health():
    return {"status": "ok"}

# ---------- Ingest: PDF -> chunks -> embeddings ----------
try:
    # UploadFile form parsing needs python-multipart
    import multipart  # noqa: F401
except Exception:
    # Not fatal; FastAPI will error at runtime if missing.
    pass

@app.post("/ingest/pdf")
async def ingest_pdf(file: UploadFile = File(...)):
    try:
        content = await file.read()
        reader = PdfReader(io.BytesIO(content))
        docs: List[Dict] = []
        seen = set()

        for i, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            for j, chunk in enumerate(split_text(text, size=700, overlap=120)):
                if not chunk:
                    continue
                key = hash_text(chunk)
                if key in seen:
                    continue
                seen.add(key)
                docs.append({"text": chunk, "source": f"{file.filename}|p{i}#{j}"})

        if not docs:
            raise HTTPException(status_code=400, detail="No extractable text in PDF (OCR may be required).")
        rag.add_documents(docs)
        rag.save()
        return {"status": "success", "pages_loaded": len(reader.pages), "chunks_added": len(docs)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---------- Simple heuristic extractive fallback (no LLM) ----------
def extractive_answer(query: str, snippets: List[Dict], k_sent: int = 5) -> str:
    """
    Extract a few sentences most related to the query as a non-LLM fallback,
    using embedding similarity + light MMR to reduce redundancy.
    """
    enc = rag.embedder
    qv = enc.encode([f"query: {query}"], normalize_embeddings=True, convert_to_numpy=True)[0]

    sents, mapping = [], []
    for s in snippets:
        parts = re.split(r'(?<=[.!?。！？])\s+', s["text"])
        for t in parts:
            t = t.strip()
            if 20 <= len(t) <= 400:
                sents.append(t)
                mapping.append(s["source"])
    if not sents:
        return "No relevant content found in the textbook."

    sv = enc.encode([f"passage: {x}" for x in sents], normalize_embeddings=True, convert_to_numpy=True)
    sims = sv @ qv  # cosine because normalized

    chosen, used = [], set()
    for _ in range(min(k_sent, len(sents))):
        best_i, best_score = -1, -1e9
        for i, base in enumerate(sims):
            if i in used:
                continue
            # Simple MMR: penalize similarity to already chosen sentences
            penalty = 0.0
            for j in used:
                penalty = max(penalty, float(sv[i] @ sv[j]))
            score = float(base) - 0.4 * penalty
            if score > best_score:
                best_score, best_i = score, i
        if best_i >= 0:
            used.add(best_i)
            chosen.append((sents[best_i], mapping[best_i]))

    body = " ".join(x for x, _ in chosen)
    cites = list({src for _, src in chosen})[:2]
    if cites:
        body += " " + " ".join(f"({c})" for c in cites)
    return body



# ---------- Ingest: Tarot JSON (Kaggle / GitHub) ----------


@app.post("/ingest/json")
async def ingest_json(file: UploadFile = File(...)):
    """
    Accept a tarot JSON file and add it into the vector store.
    Supported shapes (auto-detected):
      A) {"cards": [{ "name":..., "meaning_up": "...", "meaning_rev": "...", "desc": "...", ... }]}
      B) {"tarot_interpretations": [{ "name":..., "fortune_telling": [...], "meanings": {"light":[...], "shadow":[...]}, ...}]}
      C) generic list/dict with 'name' and either 'meaning_up'/'meaning_rev' or 'meanings'
    We create short documents per card + orientation so retrieval is precise.
    """
    try:
        raw = await file.read()
        data = json.loads(raw)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")

    def norm(x):
        if x is None:
            return ""
        if isinstance(x, list):
            return "；".join([str(i) for i in x if i])
        return str(x)

    docs = []

    # ---- shape A: {"cards": [...]}
    if isinstance(data, dict) and "cards" in data and isinstance(data["cards"], list):
        for c in data["cards"]:
            name = norm(c.get("name") or c.get("name_short") or c.get("value") or "Unknown")
            desc = norm(c.get("desc"))
            suit = norm(c.get("suit") or c.get("type"))
            kw_up = norm(c.get("keywords") or c.get("keywords_up"))
            kw_rev = norm(c.get("keywords_rev"))
            up = norm(c.get("meaning_up") or c.get("meanings", {}).get("upright"))
            rev = norm(c.get("meaning_rev") or c.get("meanings", {}).get("reversed"))

            # upright doc
            text_up = f"# {name} (Upright)\nSuit/Type: {suit}\nKeywords: {kw_up}\nMeaning: {up}\nDesc: {desc}"
            docs.append({"text": text_up, "source": f"{file.filename}|{name}|upright"})

            # reversed doc
            if up or rev or kw_rev:
                text_rev = f"# {name} (Reversed)\nSuit/Type: {suit}\nKeywords: {kw_rev}\nMeaning: {rev}\nDesc: {desc}"
                docs.append({"text": text_rev, "source": f"{file.filename}|{name}|reversed"})

    # ---- shape B: corpora/tarot_interpretations.json
    elif isinstance(data, dict) and "tarot_interpretations" in data and isinstance(data["tarot_interpretations"], list):
        for c in data["tarot_interpretations"]:
            name = norm(c.get("name", "Unknown"))
            fortune = norm(c.get("fortune_telling", []))
            light = norm((c.get("meanings") or {}).get("light", []))
            shadow = norm((c.get("meanings") or {}).get("shadow", []))
            text_up = f"# {name} (Upright)\nLight: {light}\nFortune: {fortune}"
            text_rev = f"# {name} (Reversed)\nShadow: {shadow}\nFortune: {fortune}"
            docs.append({"text": text_up, "source": f"{file.filename}|{name}|upright"})
            docs.append({"text": text_rev, "source": f"{file.filename}|{name}|reversed"})

    # ---- shape C: fallback (list of cards)
    elif isinstance(data, list):
        for c in data:
            name = norm(c.get("name", "Unknown"))
            up = norm(c.get("meaning_up") or c.get("upright") or (c.get("meanings") or {}).get("upright"))
            rev = norm(c.get("meaning_rev") or c.get("reversed") or (c.get("meanings") or {}).get("reversed"))
            desc = norm(c.get("desc"))
            text_up = f"# {name} (Upright)\nMeaning: {up}\nDesc: {desc}"
            docs.append({"text": text_up, "source": f"{file.filename}|{name}|upright"})
            if rev:
                text_rev = f"# {name} (Reversed)\nMeaning: {rev}\nDesc: {desc}"
                docs.append({"text": text_rev, "source": f"{file.filename}|{name}|reversed"})
    else:
        raise HTTPException(status_code=400, detail="Unsupported JSON structure for tarot cards.")

    if not docs:
        raise HTTPException(status_code=400, detail="No tarot entries found in JSON.")

    rag.add_documents(docs)
    rag.save()
    return {"status": "success", "items_ingested": len(docs)}

# ---------- Ask: RAG -> (HF LLM or extractive fallback) ----------
@app.post("/ask")
async def ask_question(req: QuestionRequest):
    try:
        # 1) Dense retrieval (+ optional rerank inside RAGEngine)
        snippets = rag.retrieve(
            req.question,
            top_k=req.top_k,
            min_score=req.min_score,
            fetch_k=max(req.top_k * 8, 32),
        )
        if not snippets:
            return {"answer": "No relevant content found in the textbook.", "sources": []}

        # 2) Build context (limit length to keep prompt small)
        def build_context(snips: List[Dict], max_chars: int = 2500) -> str:
            blocks, total = [], 0
            for s in snips:
                piece = f"[{s['source']}]\n{s['text']}\n"
                if total + len(piece) > max_chars:
                    break
                blocks.append(piece)
                total += len(piece)
            return "\n---\n".join(blocks)

        context = build_context(snippets, max_chars=2500)
        sources = list({s["source"] for s in snippets})

        # 3) Generate answer with HF LLM if possible; otherwise extractive fallback
        try:
            user_prompt = PROMPT_TEMPLATE.format(context=context, question=req.question)
            answer = hf_generate(PROMPT_SYSTEM, user_prompt, max_new_tokens=512, temperature=0.2).strip()
            answer = force_citations(answer, sources)
        except Exception:
            answer = extractive_answer(req.question, snippets, k_sent=5)

        return {"answer": answer, "sources": sources}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---------- Feedback ----------
@app.post("/feedback")
async def submit_feedback(feedback: FeedbackRequest):
    try:
        with open("feedback_log.txt", "a", encoding="utf-8") as f:
            f.write(f"{feedback.question}|||{feedback.wrong_answer}|||{feedback.correct_answer}\n")
        return {"status": "feedback stored"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---------- Minimal built-in HTML page (upload + ask) ----------
HTML_PAGE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>RAG + HuggingFace LLM Q&A</title>
  <style>
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial; margin: 24px; max-width: 860px; }
    h1 { margin: 0 0 8px; }
    .card { border: 1px solid #e5e7eb; border-radius: 12px; padding: 16px; margin: 16px 0; }
    .row { display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }
    button { padding: 8px 14px; border-radius: 10px; border: 1px solid #11182720; background: #111827; color: white; cursor: pointer; }
    button:disabled { opacity: .6; cursor: not-allowed; }
    input[type="file"] { display: block; }
    textarea, input[type="text"] { width: 100%; padding: 10px; border-radius: 10px; border: 1px solid #e5e7eb; }
    .muted { color: #6b7280; font-size: 12px; }
    .badge { display: inline-block; padding: 2px 8px; border-radius: 999px; background: #eef2ff; color: #3730a3; font-size: 12px; margin-right: 6px; }
    .answer { white-space: pre-wrap; }
    .sources { margin-top: 8px; }
    .divider { height: 1px; background: #e5e7eb; margin: 16px 0; }
  </style>
</head>
<body>
  <h1>RAG + HuggingFace LLM Q&A</h1>
  <div class="muted">1) Upload a PDF. 2) Ask a question. Answers will cite sources. If HF key is missing, it falls back to extractive mode.</div>

  <div class="card">
    <h3>Upload PDF</h3>
    <input id="pdf" type="file" accept="application/pdf" />
    <div class="row">
      <button id="ingestBtn">Ingest</button>
      <span id="ingestMsg" class="muted"></span>
    </div>
  </div>

  <div class="card">
    <h3>Ask</h3>
    <textarea id="question" rows="3" placeholder="e.g., What is machine learning classification?"></textarea>
    <div class="row">
      <label class="muted">Top K:
        <input id="topk" type="text" value="3" style="width:60px; margin-left:6px;">
      </label>
      <label class="muted">Min Score:
        <input id="minscore" type="text" value="0.30" style="width:80px; margin-left:6px;">
      </label>
      <button id="askBtn">Ask</button>
      <span id="askMsg" class="muted"></span>
    </div>
    <div class="divider"></div>
    <div id="result">
      <div class="answer"></div>
      <div class="sources"></div>
    </div>
  </div>

  <script>
    const ingestBtn = document.getElementById('ingestBtn');
    const askBtn = document.getElementById('askBtn');

    ingestBtn.onclick = async () => {
      const fileInput = document.getElementById('pdf');
      const msg = document.getElementById('ingestMsg');
      if (!fileInput.files.length) { alert('Please select a PDF.'); return; }
      msg.textContent = 'Uploading...'; ingestBtn.disabled = true;

      try {
        const fd = new FormData();
        fd.append('file', fileInput.files[0]);
        const resp = await fetch('/ingest/pdf', { method: 'POST', body: fd });
        const data = await resp.json();
        if (!resp.ok) throw new Error(data.detail || 'Upload failed');
        msg.textContent = `Loaded ${data.chunks_added} chunks from ${data.pages_loaded} pages.`;
      } catch (e) {
        msg.textContent = 'Error: ' + e.message;
      } finally {
        ingestBtn.disabled = false;
      }
    };

    askBtn.onclick = async () => {
      const q = document.getElementById('question').value.trim();
      const topk = parseInt(document.getElementById('topk').value || '3', 10);
      const minscore = parseFloat(document.getElementById('minscore').value || '0.30');
      const msg = document.getElementById('askMsg');
      const answerEl = document.querySelector('#result .answer');
      const sourcesEl = document.querySelector('#result .sources');
      if (!q) { alert('Enter a question.'); return; }

      msg.textContent = 'Retrieving...'; askBtn.disabled = true;
      answerEl.textContent = ''; sourcesEl.innerHTML = '';

      try {
        const resp = await fetch('/ask', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ question: q, top_k: topk, min_score: minscore })
        });
        const data = await resp.json();
        if (!resp.ok) throw new Error(data.detail || 'Request failed');
        answerEl.textContent = data.answer || '(No answer)';
        if (data.sources && data.sources.length) {
          sourcesEl.innerHTML = data.sources.map(s => `<span class="badge">${s}</span>`).join('');
        }
        msg.textContent = 'Done';
      } catch (e) {
        msg.textContent = 'Error: ' + e.message;
      } finally {
        askBtn.disabled = false;
      }
    };
  </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def root_page():
    return HTML_PAGE