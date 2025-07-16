import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, util
import torch
import json
import os
import time
from typing import List, Tuple
from pathlib import Path

base_dir = Path(__file__).parent.resolve()

# --- Streamlit Configuration ---
st.set_page_config(
    page_title="RAG Document Q&A System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sidebar-content {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .retrieved-doc {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
        color: #2c3e50;
        font-size: 0.95rem;
        line-height: 1.5;
    }
    .retrieved-doc-header {
        background-color: #e3f2fd;
        padding: 0.5rem 1rem;
        border-radius: 6px;
        margin-bottom: 0.5rem;
        color: #1565c0;
        font-weight: 600;
        font-size: 1rem;
    }
    .response-container {
        background-color: #e8f5e8;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #28a745;
        margin-top: 1rem;
        color: #2d5a2d;
        font-size: 1.1rem;
        line-height: 1.6;
    }
    .context-container {
        background-color: #fff3cd;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #ffc107;
        margin-top: 1rem;
        color: #856404;
        font-size: 0.95rem;
        line-height: 1.5;
    }
    .question-container {
        background-color: #d1ecf1;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #17a2b8;
        margin-top: 1rem;
        color: #0c5460;
        font-size: 1.05rem;
        font-weight: 500;
    }
    .component-title {
        font-size: 1.2rem;
        font-weight: bold;
        margin-bottom: 1rem;
        color: #343a40;
    }
    .metrics-container {
        background-color: #fff;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# --- Configuration ---
@st.cache_data
def get_config():
    return {
        "model_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "embedding_model_name": "all-MiniLM-L6-v2",
        # "documents_file": "documents.json",
        # "embeddings_file": "document_embeddings.pt" # File to store embeddings
        "documents_file": str(base_dir / "documents.json"),
        "embeddings_file": str(base_dir / "document_embeddings.pt")
    }

# --- Model Loading with Caching ---
@st.cache_resource
def load_models():
    config = get_config()
    with st.spinner("Loading AI models (Tokenizer, LLM, Embedding Model)... This may take a moment."):
        tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
        model = AutoModelForCausalLM.from_pretrained(
            config["model_name"],
            torch_dtype=torch.float16,
            device_map="auto"
        )
        embedding_model = SentenceTransformer(config["embedding_model_name"])
    return tokenizer, model, embedding_model

# --- Document Loading and Embedding ---
@st.cache_data(show_spinner=False) # Hide default spinner, we'll use custom ones
def load_and_embed_documents(file_path: str, embeddings_path: str) -> Tuple[List[str], torch.Tensor]:
    documents = []
    document_embeddings = torch.tensor([])
    
    # 1. Load documents from JSON file
    if not os.path.exists(file_path):
        st.error(f"Error: Document file '{file_path}' not found. Please ensure it exists and is correctly formatted.")
        return [], torch.tensor([])
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if not isinstance(data, list):
            st.error(f"Invalid JSON format in '{file_path}'. Expected a list of documents.")
            return [], torch.tensor([])

        for item in data:
            if "text" in item:
                documents.append(item["text"])
            elif "content" in item:
                documents.append(item["content"])
            elif item:
                st.warning(f"Document item missing 'text' or 'content' key, skipping: {item}")

        if not documents:
            st.warning("No valid documents found in the file. Please check your document structure and content keys (e.g., 'text' or 'content').")
            return [], torch.tensor([])

    except json.JSONDecodeError:
        st.error(f"Error: Invalid JSON in '{file_path}'. Please ensure the file is correctly formatted.")
        return [], torch.tensor([])
    except Exception as e:
        st.error(f"Error loading documents from JSON: {str(e)}")
        return [], torch.tensor([])

    # 2. Load or compute embeddings
    _, _, embedding_model = load_models() # Ensure models are loaded
    
    total_docs = len(documents)
    
    # Try to load embeddings from file first
    if os.path.exists(embeddings_path):
        try:
            loaded_embeddings = torch.load(embeddings_path)
            # Basic consistency check: Do the number of loaded embeddings match the number of documents?
            if loaded_embeddings.shape[0] == total_docs:
                st.success(f"Loaded {total_docs} document embeddings from '{embeddings_path}'.")
                return documents, loaded_embeddings
            else:
                st.warning(f"Number of loaded embeddings ({loaded_embeddings.shape[0]}) does not match current documents ({total_docs}). Re-embedding...")
        except Exception as e:
            st.warning(f"Could not load embeddings from '{embeddings_path}': {str(e)}. Re-embedding documents...")
    
    # If not loaded or mismatch, compute embeddings
    with st.spinner(f"Embedding {total_docs} documents. This may take some time for large datasets..."):
        embedding_progress_text = st.empty()
        embedding_bar = st.progress(0)
        
        batch_size = 32 # Adjust batch size based on your system's memory
        document_embeddings_list = []
        
        for i in range(0, total_docs, batch_size):
            batch_docs = documents[i:min(i + batch_size, total_docs)]
            # Ensure we're only embedding actual strings
            valid_batch_docs = [doc for doc in batch_docs if isinstance(doc, str)]
            if not valid_batch_docs:
                continue # Skip if batch is empty or only contains non-strings

            batch_embeddings = embedding_model.encode(valid_batch_docs, convert_to_tensor=True, show_progress_bar=False)
            document_embeddings_list.append(batch_embeddings)
            
            progress = (i + len(batch_docs)) / total_docs
            embedding_progress_text.text(f"Embedding documents: {i + len(batch_docs)}/{total_docs} completed...")
            embedding_bar.progress(progress)

        if document_embeddings_list:
            document_embeddings = torch.cat(document_embeddings_list)
            try:
                torch.save(document_embeddings, embeddings_path)
                st.success(f"Embeddings saved to '{embeddings_path}'.")
            except Exception as e:
                st.error(f"Could not save embeddings to file: {str(e)}")
        else:
            st.warning("No documents were embedded.")

        embedding_progress_text.empty() 
        embedding_bar.empty() 

    return documents, document_embeddings

# --- RAG Functions ---
def retrieve_documents(query: str, documents: List[str], document_embeddings: torch.Tensor, top_k: int = 1) -> Tuple[List[str], List[float]]:
    if len(documents) == 0 or document_embeddings.nelement() == 0:
        return [], []
    _, _, embedding_model = load_models()
    query_embedding = embedding_model.encode(query, convert_to_tensor=True)
    cosine_scores = util.cos_sim(query_embedding, document_embeddings)[0]
    
    actual_top_k = min(top_k, len(documents))
    if actual_top_k == 0:
        return [], []

    top_results = torch.topk(cosine_scores, k=actual_top_k)
    retrieved_docs = [documents[idx] for idx in top_results.indices]
    scores = [score.item() for score in top_results.values]
    return retrieved_docs, scores

# Removed temperature parameter and top_p from here
def generate_response(query: str, retrieved_documents: List[str], max_tokens: int = 256) -> Tuple[str, str, str]:
    tokenizer, model, _ = load_models()
    context = "\n".join(retrieved_documents)
    
    if not retrieved_documents:
        rag_query = f"Question: {query}\n\nNo relevant context found. Please answer based on general knowledge if possible, or state that information is not available."
    else:
        rag_query = f"Based on the following information, answer the question:\n\nContext:\n{context}\n\nQuestion: {query}"
    
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": rag_query}],
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,  # Set to False for greedy decoding
            # temperature=temperature, # Removed
            # top_p=0.9,             # Removed
            pad_token_id=tokenizer.eos_token_id
        )
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    cleaned_response = full_response
    if "Assistant:" in full_response:
        cleaned_response = full_response.split("Assistant:")[-1].strip()
    elif "<|assistant|>" in full_response:
        cleaned_response = full_response.split("<|assistant|>")[-1].strip()
    elif "[/INST]" in full_response:
        cleaned_response = full_response.split("[/INST]")[-1].strip()

    if cleaned_response.startswith(rag_query):
        cleaned_response = cleaned_response[len(rag_query):].strip()
    
    return cleaned_response, context, rag_query

# --- Main App ---
def main():
    st.markdown('<h1 class="main-header">🔍 RAG Document Q&A System</h1>', unsafe_allow_html=True)
    
    config = get_config() 

    with st.sidebar:
        st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
        st.markdown("### ⚙️ Configuration")
        st.info(f"**Model:** {config['model_name']}")
        st.info(f"**Embedding Model:** {config['embedding_model_name']}")
        st.info(f"**Documents File:** {config['documents_file']}")
        st.info(f"**Embeddings Cache File:** {config['embeddings_file']}")
        st.markdown("### 📊 Settings")
        top_k = st.slider("Number of documents to retrieve", 1, 5, 1)
        # temperature = st.slider("Response creativity (temperature)", 0.01, 1.0, 0.7, 0.01) # Removed
        max_tokens = st.slider("Maximum response length", 50, 512, 256, 50)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### 🛠️ Maintenance")
        st.info("Documents are loaded from `documents.json`.")
        if st.button("Force Re-embed All Documents", help="Click to re-process and re-save embeddings for all documents from scratch. Useful if `documents.json` was manually edited."):
            load_and_embed_documents.clear() # Clear the cache
            if os.path.exists(config["embeddings_file"]):
                os.remove(config["embeddings_file"])
                st.info(f"Removed '{config['embeddings_file']}' to force full re-embedding.")
            st.rerun() # Trigger a rerun to re-load and re-embed

    col1, col2 = st.columns([2, 1])

    # Load documents and embeddings once (will use cache or re-embed with progress)
    # The progress bar for embedding will now appear in the main area if re-embedding occurs
    documents, document_embeddings = load_and_embed_documents(config["documents_file"], config["embeddings_file"])

    with col1:
        st.markdown("### 💬 Ask a Question")
        query = st.text_input(
            "Enter your question:",
            placeholder="e.g., Who is Harriet?",
            help="Ask any question about the documents in your knowledge base"
        )
        if st.button("🔍 Search & Generate Answer", type="primary"):
            if not query.strip():
                st.warning("Please enter a question.")
                return
            
            if len(documents) == 0:
                st.error("No documents available in the knowledge base from 'documents.json'. Please ensure the file exists and contains valid documents.")
                return

            query_status_container = st.empty() 
            progress_bar = st.progress(0)
            
            query_status_container.text("Retrieving relevant documents...")
            progress_bar.progress(33)
            
            retrieved_docs, scores = retrieve_documents(query, documents, document_embeddings, top_k)
            
            if not retrieved_docs:
                query_status_container.warning("No highly relevant documents found for your query. Generating response based on general knowledge (if applicable).")
                # Call generate_response without temperature
                response, context, full_query = generate_response(query, [], max_tokens)
            else:
                query_status_container.text("Generating response...")
                progress_bar.progress(66)
                start_time = time.time()
                # Call generate_response without temperature
                response, context, full_query = generate_response(query, retrieved_docs, max_tokens)
                response_time = time.time() - start_time
            
            progress_bar.progress(100)
            query_status_container.success("Complete!")
            
            st.markdown("### 🎯 Your Question")
            st.markdown(f'<div class="question-container">{query}</div>', unsafe_allow_html=True)
            
            st.markdown("### 📚 Context Used")
            if context:
                st.markdown(f'<div class="context-container">{context}</div>', unsafe_allow_html=True)
            else:
                st.info("No specific context from documents was used for this query.")

            st.markdown("### 🤖 AI Response")
            st.markdown(f'<div class="response-container">{response}</div>', unsafe_allow_html=True)
            
            st.markdown("### 📈 Performance Metrics")
            col_metrics1, col_metrics2, col_metrics3 = st.columns(3)
            with col_metrics1:
                st.metric("Response Time", f"{response_time:.2f}s" if 'response_time' in locals() else "N/A")
            with col_metrics2:
                st.metric("Documents Retrieved", len(retrieved_docs))
            with col_metrics3:
                st.metric("Avg. Similarity Score", f"{sum(scores)/len(scores):.3f}" if scores else "N/A")
            
            st.markdown("### 📚 Retrieved Documents")
            if retrieved_docs:
                for i, (doc, score) in enumerate(zip(retrieved_docs, scores)):
                    st.markdown(f'<div class="retrieved-doc-header">Document {i+1} - Similarity Score: {score:.3f}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="retrieved-doc">{doc[:500]}{"..." if len(doc) > 500 else ""}</div>', unsafe_allow_html=True)
            else:
                st.info("No documents were retrieved for this query.")

            query_status_container.empty()
            progress_bar.empty()

    with col2:
        st.markdown("### 📋 Document Statistics")
        if documents:
            st.metric("Total Documents", len(documents))
            avg_length = sum(len(doc.split()) for doc in documents) / len(documents)
            st.metric("Avg. Document Length", f"{avg_length:.0f} words")
            st.markdown("### 📄 Sample Documents")
            for i, doc in enumerate(documents[:3]): 
                with st.expander(f"Document {i+1} (Length: {len(doc.split())} words)"):
                    st.text(doc[:500] + "..." if len(doc) > 500 else doc) 
            if len(documents) > 3:
                st.info(f"And {len(documents) - 3} more documents.")
        else:
            st.info("No documents loaded. Please ensure 'documents.json' exists and contains valid documents.")
    
    st.markdown("---")
    st.markdown("Built with ❤️ using Streamlit, Transformers, and Sentence Transformers")

if __name__ == "__main__":
    main()