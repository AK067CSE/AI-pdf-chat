import io
import os
import re
import json
import hashlib
from typing import List, Dict

import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader

from rag_engine import RAGEngine
import requests

# =========================== Env & Config ===========================
load_dotenv()
HF_API_KEY = os.getenv("HF_API_KEY", "")
HF_MODEL = os.getenv("HF_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")
HF_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL}"

# =========================== LLM Call ===========================
def hf_generate(system_prompt: str, user_prompt: str,
                max_new_tokens: int = 512, temperature: float = 0.3) -> str:
    if not HF_API_KEY:
        raise RuntimeError("HF_API_KEY is not set. Set it in .env or environment.")

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
        raise RuntimeError(f"HuggingFace API error {resp.status_code}: {data}")
    if isinstance(data, list) and data and "generated_text" in data[0]:
        out = data[0]["generated_text"]
        return out[len(prompt):].strip() if out.startswith(prompt) else out.strip()
    return str(data).strip()

# =========================== Prompting ===========================
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
    if "(" in answer and ")" in answer:
        return answer
    if sources:
        tail = " ".join(f"({s})" for s in sources[:2])
        return f"{answer} {tail}"
    return answer

# =========================== Helpers ===========================
def split_text(text: str, size: int = 700, overlap: int = 120) -> List[str]:
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
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

# Simple extractive fallback without LLM
def extractive_answer(rag: RAGEngine, query: str, snippets: List[Dict], k_sent: int = 5) -> str:
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
    sims = sv @ qv

    chosen, used = [], set()
    for _ in range(min(k_sent, len(sents))):
        best_i, best_score = -1, -1e9
        for i, base in enumerate(sims):
            if i in used:
                continue
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

# =========================== RAG Singleton ===========================
@st.cache_resource(show_spinner=False)
def get_rag() -> RAGEngine:
    rag = RAGEngine()
    try:
        if os.path.exists("faiss_index.bin") and os.path.exists("metadata.json"):
            rag.load("faiss_index.bin", "metadata.json")
    except Exception as e:
        st.warning(f"Failed to load existing index: {e}")
    return rag

# =========================== UI ===========================
st.set_page_config(page_title="RAG Q&A (Streamlit)", layout="wide")
st.title("RAG Q&A with Hugging Face LLM")

rag = get_rag()

with st.sidebar:
    st.header("Settings")
    st.caption("Embedding and index load once per session.")
    top_k = st.slider("Top K", min_value=1, max_value=10, value=3, step=1)
    min_score = st.slider("Min Score", min_value=0.0, max_value=1.0, value=0.30, step=0.05)
    st.divider()
    if st.button("Save index now"):
        rag.save()
        st.success("Index saved to faiss_index.bin + metadata.json")
    if st.button("Clear index (memory only)"):
        # Recreate engine in session
        st.cache_resource.clear()
        rag = get_rag()
        st.success("Cleared in-memory engine. Reloaded fresh.")
    st.divider()
    st.markdown("**HF Model**: `" + HF_MODEL + "`")
    st.markdown("**HF Key set**: " + ("✅" if HF_API_KEY else "❌"))

# Ingestion Tabs
ingest_tab, ask_tab = st.tabs(["Ingest", "Ask"]) 

with ingest_tab:
    st.subheader("Ingest PDF(s)")
    pdf_files = st.file_uploader("Upload one or more PDFs", type=["pdf"], accept_multiple_files=True)
    if st.button("Process PDF(s)") and pdf_files:
        total_pages, total_chunks = 0, 0
        seen = set()
        docs: List[Dict] = []
        for f in pdf_files:
            try:
                content = f.read()
                reader = PdfReader(io.BytesIO(content))
                total_pages += len(reader.pages)
                for i, page in enumerate(reader.pages, start=1):
                    text = (page.extract_text() or "").strip()
                    for j, chunk in enumerate(split_text(text, size=700, overlap=120)):
                        if not chunk:
                            continue
                        key = hash_text(chunk)
                        if key in seen:
                            continue
                        seen.add(key)
                        docs.append({"text": chunk, "source": f"{f.name}|p{i}#{j}"})
            except Exception as e:
                st.error(f"Failed reading PDF {f.name}: {e}")
        if not docs:
            st.warning("No extractable text found. PDF might need OCR.")
        else:
            rag.add_documents(docs)
            rag.save()
            total_chunks = len(docs)
            st.success(f"Ingested {len(pdf_files)} file(s), pages: {total_pages}, chunks added: {total_chunks}")

    st.divider()
    st.subheader("Ingest JSON (any)")
    json_file = st.file_uploader("Upload a JSON file", type=["json"], accept_multiple_files=False)
    raw_json_text = st.text_area("Or paste raw JSON", placeholder="{}", height=150)
    if st.button("Process JSON"):
        data = None
        source_name = ""
        # Determine source: file or pasted text
        try:
            if json_file is not None:
                raw = json_file.read()
                data = json.loads(raw)
                source_name = json_file.name
            elif raw_json_text.strip():
                data = json.loads(raw_json_text)
                source_name = "pasted_json"
            else:
                st.warning("Please upload a JSON file or paste JSON.")
        except Exception as e:
            st.error(f"Invalid JSON: {e}")
            data = None

        # Generic JSON -> documents
        def scalar_to_str(x):
            if isinstance(x, str):
                return x
            try:
                return json.dumps(x, ensure_ascii=False)
            except Exception:
                return str(x)

        def walk(node, path: str, docs_out: List[Dict]):
            # Recursively traverse JSON and create docs per scalar leaf
            if isinstance(node, dict):
                for k, v in node.items():
                    new_path = f"{path}.{k}" if path else str(k)
                    walk(v, new_path, docs_out)
            elif isinstance(node, list):
                for i, v in enumerate(node):
                    new_path = f"{path}[{i}]" if path else f"[{i}]"
                    walk(v, new_path, docs_out)
            else:
                # Scalar leaf
                if isinstance(node, str) and len(node) > 700:
                    # Chunk long strings for better retrieval
                    for j, chunk in enumerate(split_text(node, size=700, overlap=120)):
                        if chunk.strip():
                            docs_out.append({
                                "text": f"[{path}]\n{chunk}",
                                "source": f"{source_name}|{path}#{j}"
                            })
                else:
                    text = f"[{path}] {scalar_to_str(node)}"
                    docs_out.append({
                        "text": text,
                        "source": f"{source_name}|{path}"
                    })

        if data is not None:
            try:
                docs: List[Dict] = []
                walk(data, path="", docs_out=docs)
                if not docs:
                    st.warning("No scalar values found to index in the provided JSON.")
                else:
                    rag.add_documents(docs)
                    rag.save()
                    st.success(f"Ingested {len(docs)} JSON entries from {source_name}.")
            except Exception as e:
                st.error(f"Failed processing JSON: {e}")

with ask_tab:
    st.subheader("Ask a question")
    question = st.text_area("Your question", placeholder="Ask about the ingested PDFs/JSON...", height=80)
    col1, col2 = st.columns([1, 3])
    with col1:
        go = st.button("Ask")
    with col2:
        st.caption("Answer will use the HF model if available; otherwise a non-LLM extractive fallback.")

    if go and question.strip():
        with st.spinner("Retrieving context and generating answer..."):
            try:
                snippets = rag.retrieve(
                    question, top_k=top_k, min_score=min_score, fetch_k=max(top_k * 8, 32)
                )
                if not snippets:
                    st.warning("No relevant content found in the textbook.")
                else:
                    # Build context
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

                    try:
                        user_prompt = PROMPT_TEMPLATE.format(context=context, question=question)
                        answer = hf_generate(PROMPT_SYSTEM, user_prompt, max_new_tokens=512, temperature=0.2).strip()
                        answer = force_citations(answer, sources)
                    except Exception:
                        answer = extractive_answer(rag, question, snippets, k_sent=5)

                    st.markdown("### Answer")
                    st.write(answer)
                    with st.expander("Sources"):
                        for s in sources:
                            st.write("- " + s)
            except Exception as e:
                st.error(f"Error: {e}")

st.divider()
st.caption("Index files: faiss_index.bin, metadata.json. They will be created in the working directory.")
