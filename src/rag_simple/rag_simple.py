from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, util
import torch
import json
import os
import sys

# --- Configuration ---
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
embedding_model_name = "all-MiniLM-L6-v2"
documents_file = "documents.json" # Your JSON file containing documents

# --- Load Models and Tokenizers ---
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

embedding_model = SentenceTransformer(embedding_model_name)

# --- Document Loading and Embedding (One-time or pre-computed) ---
def load_and_embed_documents(file_path):
    documents = []
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for item in data:
            if "text" in item:
                documents.append(item["text"])
            elif "content" in item:
                documents.append(item["content"])
            # No warning print here, as per "avoid useless logs"
    document_embeddings = embedding_model.encode(documents, convert_to_tensor=True)
    return documents, document_embeddings

# Check if the document file exists, and exit if not
if not os.path.exists(documents_file):
    print(f"Error: Document file '{documents_file}' not found.")
    print("Please ensure 'documents.json' exists in the same directory as this script.")
    sys.exit(1)

documents, document_embeddings = load_and_embed_documents(documents_file)

# --- RAG Integration ---
def retrieve_documents(query, documents, document_embeddings, top_k=1):
    query_embedding = embedding_model.encode(query, convert_to_tensor=True)
    cosine_scores = util.cos_sim(query_embedding, document_embeddings)[0]
    top_results = torch.topk(cosine_scores, k=top_k)

    retrieved_docs = [documents[idx] for idx in top_results.indices]
    return retrieved_docs

# --- Main Query and Generation ---
QUERY = "Who is Harriet?"

# 1. Retrieve relevant documents
retrieved_documents = retrieve_documents(QUERY, documents, document_embeddings, top_k=2)

# 2. Construct the prompt with retrieved context
context = "\n".join(retrieved_documents)
rag_query = f"Based on the following information, answer the question:\n\nContext:\n{context}\n\nQuestion: {QUERY}"

# Prompt using chat template
prompt = tokenizer.apply_chat_template(
    [{"role": "user", "content": rag_query}],
    tokenize=False,
    add_generation_prompt=True
)

# Tokenize
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# Generate response
outputs = model.generate(**inputs, max_new_tokens=256, do_sample=True, temperature=0.7, top_p=0.9)

# Decode and print
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
# Clean the output to only show the assistant's response
cleaned_response = response.split("Assistant:")[-1].strip()
print(cleaned_response)
