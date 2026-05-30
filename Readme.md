# 🕌 Kolhapur Tourism AI Chatbot

An **offline, closed-domain Retrieval-Augmented Generation (RAG)** chatbot for exploring Kolhapur's heritage and tourist landmarks. Built for the **Kolhapur AI Community Sprint** as a production architecture example.

[![Streamlit](https://img.shields.io/badge/Streamlit-1.50.0-FF4B4B.svg)](https://streamlit.io)
[![LangChain](https://img.shields.io/badge/LangChain-0.3.30-1C3C3C.svg)](https://langchain.com)
[![Ollama](https://img.shields.io/badge/Ollama-0.6.2-O разли.svg)](https://ollama.ai)
[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)

---

## 🎯 Features

| Feature | Description |
|---------|-------------|
| **Offline & Closed-Domain** | No internet required; AI answers only from uploaded PDFs, preventing hallucination |
| **Max 5 PDFs** | Limits processing time for fast demos; prevents slow indexing |
| **Real-Time Progress Bars** | Shows status: loading → chunking → indexing |
| **Reset / Reindex Button** | Clears database and rebuilds from scratch with one click |
| **Source Tracing (Architect Mode)** | Shows which PDF + page + chunk the answer came from |
| **Persistent Vector DB** | ChromaDB saves vectors to disk; index survives app restarts |
| **Strict Guardrails** | Model says "I'm sorry" for out-of-scope questions |

---

## 🏗️ Architecture Overview

User uploads PDFs (max 5)
↓
PyPDFLoader → reads PDF text
↓
RecursiveCharacterTextSplitter → chunks (700 chars, 70 overlap)
↓
OllamaEmbeddings (Mistral-7B) → converts chunks to vectors
↓
ChromaDB → stores vectors on disk (chroma_db/)
↓
User asks question
↓
Retriever fetches top-2 chunks → Mistral LLM (temperature=0.2) → answer
↓
Answer + source docs + latency → displayed in chat

---

## 🚀 Quick Start Guide

### Prerequisites

- **Python 3.9+**
- **Ollama** installed and running
- **Mistral model** pulled in Ollama

### Step 1: Clone the Repository

```bash
git clone <your-repo-url>
cd Kolhapur-Tourism-Chatbot
```

### Step 2: Create Virtual Environment

```bash
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate

# Windows
python -m venv .venv
.venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Start Ollama

```bash
# Terminal 1: Start Ollama server
ollama serve

# Terminal 2: Pull Mistral model
ollama pull mistral
```

> **Note:** Mistral-7B is ~4GB. Download once, then it's cached locally.

### Step 5: Run the Streamlit App

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## 📖 How to Use

### 1. Upload PDFs (Max 5)

- Click **"Upload Landmark PDFs to the Data Folder"** in the sidebar
- Select up to 5 PDF files about Kolhapur landmarks
- Click **"⚙️ Process & Index Documents"**
- Watch progress bars: loading → chunking → indexing

### 2. Ask Questions

- Type questions in the chat input:
  - "What is the history of Mahalaxmi Temple?"
  - "Where is Pankaj Pavilion located?"

### 3. View Source Traces (Architect Mode)

- Click the **"📌 Vector Database Trace Logs (Architect Mode)"** expander
- See:
  - **Inference latency** (response time)
  - **Source PDF name** (e.g., `mahalaxmi.pdf`)
  - **Page number** (e.g., Page 3)
  - **Retrieved chunk text** (exact text used for answer)

### 4. Reset / Reindex Database

- Click **"🗑️ Reset / Reindex Database"** in the sidebar
- Deletes `chroma_db/` and rebuilds index from all PDFs in `data/`
- Useful for testing or starting fresh

---

## ⚙️ Configuration

### Chunk Strategy

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=700,   # 700 characters per chunk
    chunk_overlap=70  # 70 chars overlap to preserve context
)
```

### Retrieval Settings

```python
retriever = vector_store.as_retriever(search_kwargs={"k": 2})
# Fetches top-2 most relevant chunks
```

### LLM Settings

```python
llm = OllamaLLM(model="mistral", temperature=0.2)
# Mistral-7B with low temperature for factual answers
```

---

## 🐛 Troubleshooting

### Error: "Database error: attempt to write a readonly database"

**Cause:** ChromaDB folder is locked by another process.

**Fix:**
```bash
# Stop Streamlit (Ctrl+C)
# Kill any Python processes using chroma_db
ps aux | grep python
kill -9 <PID>

# Delete chroma_db folder manually
rm -rf chroma_db

# Restart Streamlit
streamlit run app.py
```

### Error: "Ollama is not running"

**Cause:** Ollama server isn't started.

**Fix:**
```bash
# Terminal 1: Start Ollama
ollama serve

# Terminal 2: Run Streamlit
streamlit run app.py
```

### Error: "Model 'mistral' not found"

**Cause:** Mistral model isn't pulled.

**Fix:**
```bash
ollama pull mistral
```


---

## 🧪 Demo Script (For Presentations)

1. **Start Ollama:**
   ```bash
   ollama serve
   ollama pull mistral
   ```

2. **Start Streamlit:**
   ```bash
   streamlit run app.py
   ```

3. **Upload 2–3 PDFs** about Kolhapur landmarks

4. **Click "Process & Index Documents"** → Show progress bars

5. **Ask Questions:**
   - "What is the history of Mahalaxmi Temple?"
   - "Where is panhala fort?"

6. **Click "Architect Mode" expander** → Show source PDF + page + chunk

7. **Ask Out-of-Scope Question:**
   - "Who is the Prime Minister of India?"
   - Bot should reply: *"I'm sorry, but I can only answer questions about Kolhapur's tourist places..."*

8. **Click "Reset / Reindex Database"** → Show fresh reindexing

---

## 🚀 Future Enhancements

| Enhancement | Why It Matters |
|-------------|----------------|
| **Incremental Indexing** | Add new PDFs without rebuilding entire DB |
| **Better Chunking** | Semantic chunking or sentence-level splitting |
| **Multi-Modal RAG** | Include images + text from PDFs |
| **Docker Containerization** | Easy deployment to any machine |
| **Authentication** | Multi-user access with login |
| **Query Caching** | Cache common questions to reduce LLM calls |

---

## 📝 License

This project was developed as a **production architecture example** for the **Kolhapur AI Community Sprint**. Feel free to use and modify for educational purposes.

---

## 👥 Contributors

- **Kolhapur AI Community**

---
