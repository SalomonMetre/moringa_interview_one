## 🔍 RAG Document Q\&A System

An interactive Streamlit app for **retrieving and answering questions from a local document knowledge base** using Retrieval-Augmented Generation (RAG) with LLMs and embeddings.

Built with ❤️ using:

* 🤗 Transformers (`TinyLlama`)
* 🧠 Sentence Transformers (`all-MiniLM-L6-v2`)
* 🐍 PyTorch
* 🎈 Streamlit

---

### 📁 Project Structure

```
rag_example/
├── rag_simple/
│   ├── rag_simple_streamlit.py   # Main Streamlit app
│   ├── documents.json            # Local knowledge base (list of documents)
│   ├── document_embeddings.pt    # Cached embeddings (auto-generated)
├── README.md                     # Project documentation
├── requirements.txt              # Python dependencies
```

---

### ⚙️ Features

* 🔎 **Ask natural language questions** and get answers based on local documents.
* 📄 **Supports JSON file input** with documents (keys: `text` or `content`).
* 🚀 **Fast vector retrieval** using Sentence Transformers.
* 🤖 **Generates answers with context** using a quantized `TinyLlama` model.
* 💾 **Embeddings cached** in `document_embeddings.pt` to avoid recomputation.
* 🔁 **Force re-embedding** with one click in the sidebar.
* 🎨 **Beautiful, responsive UI** with Streamlit and custom CSS.

---

### 🧠 How It Works

1. Loads a list of documents from `documents.json`.
2. Encodes each document into embeddings using Sentence Transformers.
3. Stores and reuses embeddings to improve performance.
4. On question input:

   * Encodes the query
   * Retrieves top-K similar documents by cosine similarity
   * Constructs a prompt with context
   * Sends to `TinyLlama` for generation
5. Displays:

   * AI-generated response
   * Retrieved document context
   * Metrics and debug info

---

### 🧪 Example Document Format (`documents.json`)

```json
[
  {"text": "Harriet Tubman was an American abolitionist and political activist."},
  {"content": "Streamlit is a Python library used for building web apps for machine learning and data science."}
]
```

> Supported keys: `"text"` or `"content"`

---

### ▶️ How to Run the App

1. **Clone the repository**:

```bash
git clone https://github.com/yourusername/rag_example.git
cd rag_example/rag_simple
```

2. **Create a virtual environment**:

```bash
python -m venv venv
source venv/bin/activate
```

3. **Install dependencies**:

```bash
pip install -r ../requirements.txt
```

4. **Run the app**:

```bash
streamlit run rag_simple_streamlit.py
```

---

### 📦 Requirements

Your `requirements.txt` should include:

```txt
torch==2.7.1
sentence-transformers==2.7.0
transformers==4.42.1
streamlit==1.35.0
```

Optional CPU-only versions:

```txt
torch==2.7.1+cpu
torchaudio==2.7.1+cpu
torchvision==0.22.1+cpu
```

> ⚠️ `torch==2.7.1+cpu` must be installed from the official PyTorch channel or with `pip install torch==2.7.1+cpu -f https://download.pytorch.org/whl/torch_stable.html`

---

### 🛠 Maintenance Tips

* To reprocess documents or update `documents.json`, use the sidebar button:

  ```
  Force Re-embed All Documents
  ```

  It will:

  * Clear the cached embeddings
  * Rerun the app and re-embed the current documents

---

### 💡 Future Improvements

* Drag-and-drop document upload
* Support for PDF ingestion and chunking
* Multi-turn conversation memory
* LangChain or Haystack integration
* Evaluation of response quality with confidence score

---

### 📜 License

MIT License. © 2025 \SalomonMetre

