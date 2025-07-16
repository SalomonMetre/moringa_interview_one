import fitz  # PyMuPDF
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from chromadb import Client
from chromadb.config import Settings
import torch
import re

def extract_text_from_pdf(path: str | Path, exclude_last_n: int = 2) -> str:
    doc = fitz.open(path)
    pages = [doc[i].get_text() for i in range(max(0, len(doc) - exclude_last_n))]
    doc.close()
    
    # Join pages and clean up excessive newlines
    raw_text = "\n".join(pages)
    
    # Normalize: multiple newlines → one newline, excessive spaces → one space
    cleaned = re.sub(r"\n{2,}", "\n", raw_text)     # collapse multiple newlines
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)    # collapse multiple spaces/tabs
    cleaned = re.sub(r"\n+", " ", cleaned)          # remove all remaining newlines
    cleaned = re.sub(r"\s{2,}", " ", cleaned)       # extra spacing
    return cleaned.strip()


def chunk_by_tokens(text: str, tokenizer, max_tokens: int = 256, overlap: int = 30):
    input_ids = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    for i in range(0, len(input_ids), max_tokens - overlap):
        chunk_ids = input_ids[i:i + max_tokens]
        if chunk_ids:
            chunk = tokenizer.decode(chunk_ids)
            chunks.append(chunk)
    return chunks

# Embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# LLM (TinyLlama)
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

# ChromaDB
chroma_client = Client(Settings(anonymized_telemetry=False))
collection = chroma_client.create_collection("pdf_rag")

pdf_files = [
             "../documents/5th-grade-5-best-friend-blues.pdf", 
             "../documents/5th-grade-5-reading-after-flood.pdf", 
             "../documents/5th-grade-5-reading-astronomy-project.pdf",
             "../documents/5th-grade-5-reading-first-concert.pdf"
            ]

for pdf_path in pdf_files:
    text = extract_text_from_pdf(pdf_path)
    chunks = chunk_by_tokens(text, tokenizer, max_tokens = 256, overlap = 10)
    for i, chunk in enumerate(chunks):
        embedding = embedder.encode(chunk).tolist()
        chunk_id = f"{Path(pdf_path).stem}_{i}"
        collection.add(
            documents=[chunk],
            embeddings=[embedding],
            ids=[chunk_id]
        )

