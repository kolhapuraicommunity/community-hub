import streamlit as st
import os
import time
import shutil
from rag_engine import KolhapurRAGEngine

# --- 1. UI CONFIGURATION BASELINE ---
st.set_page_config(page_title="Kolhapur Tourism AI Guide", page_icon="🤖", layout="wide")

DATA_DIR = "data"
ASSETS_DIR = "assets"

for folder in [DATA_DIR, ASSETS_DIR]:
    if not os.path.exists(folder):
        os.makedirs(folder)

engine = KolhapurRAGEngine(data_dir=DATA_DIR, chroma_dir="chroma_db")

# --- 2. SIDEBAR RENDERING LAYOUTS ---
st.sidebar.title("⚙️ RAG System")

logo_path = os.path.join(ASSETS_DIR, "logo.png")
if os.path.exists(logo_path):
    st.sidebar.image(logo_path, width=100)  # fixed width

st.sidebar.markdown("### 📌 About This Application")
st.sidebar.info(
    """
    **Kolhapur Tourism AI** is an offline, closed-domain RAG system.
    
    * **LLM Core:** Mistral-7B (Served via Ollama)
    * **Orchestration:** LangChain v0.2
    * **Vector Database:** ChromaDB (Persistent)
    * **Chunk Strategy:** 700 chars / 70 overlap
    * **PDF Limit:** Max 5 files per ingestion
    
    *Developed for the Kolhapur AI Community.*
    """
)
st.sidebar.markdown("---")

st.sidebar.subheader("📤 Knowledge Base Ingestion")
uploaded_files = st.sidebar.file_uploader(
    "Upload Landmark PDFs (max 5)", 
    type=["pdf"], 
    accept_multiple_files=True,
    key="pdf_uploader"
)

# Progress state
if "ingestion_progress" not in st.session_state:
    st.session_state.ingestion_progress = None

def progress_callback(step, current, total, message):
    st.session_state.ingestion_progress = {
        "step": step,
        "current": current,
        "total": total,
        "message": message
    }

# Button trigger invoking our cumulative accumulation pipeline
if st.sidebar.button("⚙️ Process & Index Documents"):
    if not uploaded_files and len(os.listdir(DATA_DIR)) == 0:
        st.sidebar.warning("⚠️ Your data folder is empty. Please upload files first!")
    else:
        with st.sidebar.status("Invoking isolated RAG ingestion modules...", expanded=True) as status:
            st.session_state.ingestion_progress = None
            
            if uploaded_files:
                st.write("📁 Writing and saving new files into the local 'data/' folder...")
                total_files = len(uploaded_files)
                for i, uploaded_file in enumerate(uploaded_files):
                    file_path = os.path.join(DATA_DIR, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                
                # Show writing progress
                progress_bar = st.sidebar.progress(0)
                for i in range(total_files):
                    progress_bar.progress((i + 1) / total_files)
                
                st.sidebar.success(f"✅ Saved {total_files} PDF(s) to 'data/' folder")

            st.write("🧠 Compiling vector indexes via decoupled backend controller...")
            
            # Progress bar for indexing
            indexing_progress = st.sidebar.progress(0)
            progress_text = st.sidebar.empty()

            try:
                def streamlit_progress_callback(step, current, total, message):
                    progress_text.text(f"📌 {message}")
                    if total > 0:
                        indexing_progress.progress(min(current / total, 1.0))
                    st.session_state.ingestion_progress = {
                        "step": step,
                        "current": current,
                        "total": total,
                        "message": message
                    }

                # Count PDFs in data/ folder
                pdf_count = len([f for f in os.listdir(DATA_DIR) if f.endswith(".pdf")])
                if pdf_count > 5:
                    st.sidebar.error(f"⚠️ Found {pdf_count} PDFs in data/ folder. Please delete extras to keep only 5.")
                    st.stop()  # Stop execution

                engine.process_and_index_pdfs(progress_callback=streamlit_progress_callback)
                st.cache_resource.clear()
                
                status.update(label="✅ Database Synced with Data Folder!", state="complete", expanded=False)
                progress_text.text("✅ Indexing complete!")
                indexing_progress.progress(1.0)
                st.rerun()
                
            except Exception as e:
                status.update(label="🚨 Ingestion Pipeline Failure", state="error", expanded=True)
                st.sidebar.error(f"Execution Error details: {e}")

# === RESET / REINDEX BUTTON ===
st.sidebar.markdown("---")
st.sidebar.subheader("🔄 Reset System")

if st.sidebar.button("🗑️ Reset / Reindex Database"):
    chroma_path = "chroma_db"
    if not os.path.exists(chroma_path):
        st.sidebar.warning("⚠️ Database already empty; nothing to reset.")
    else:
        with st.sidebar.status("Resetting vector database...", expanded=True) as status:
            st.session_state.ingestion_progress = None
            
            reset_progress = st.sidebar.progress(0)
            reset_text = st.sidebar.empty()
            
            try:
                reset_text.text("🗑️ Deleting existing ChromaDB folder...")
                shutil.rmtree(chroma_path)
                reset_progress.progress(0.5)
                
                pdf_count = len([f for f in os.listdir(DATA_DIR) if f.endswith(".pdf")])
                if pdf_count > 0:
                    if pdf_count > 5:
                        reset_text.text(f"⚠️ Found {pdf_count} PDFs; limiting to 5 for reindexing")
                    reset_text.text(f"🧠 Re-indexing up to 5 PDFs from `data/` folder...")
                    
                    def streamlit_progress_callback(step, current, total, message):
                        reset_text.text(f"📌 {message}")
                        if total > 0:
                            reset_progress.progress(0.5 + 0.5 * (current / total))
                    
                    engine.process_and_index_pdfs(progress_callback=streamlit_progress_callback)
                    status.update(label="✅ Database Reset & Reindexed!", state="complete", expanded=False)
                else:
                    status.update(label="⚠️ Database Reset (no PDFs to index)", state="warning", expanded=False)
                    st.sidebar.info("💡 Upload PDFs and click 'Process & Index Documents' to build the index.")
                
                st.cache_resource.clear()
                reset_progress.progress(1.0)
                st.rerun()
            except Exception as e:
                status.update(label="🚨 Reset Failed", state="error", expanded=True)
                st.sidebar.error(f"Reset Error details: {e}")

st.sidebar.markdown("---")

# MONITOR VECTOR DATABASE METRICS LOGS LIVE
st.sidebar.subheader("📊 Vector DB Status Metrics")
if engine.is_db_indexed():
    st.sidebar.success("Database Status: ACTIVE & PERSISTED")
    
    indexed_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".pdf")] if os.path.exists(DATA_DIR) else []
    if indexed_files:
        if len(indexed_files) > 5:
            indexed_files = indexed_files[:5]
        st.sidebar.caption(f"Total Cached Files in Folder ({min(len(indexed_files), 5)} of {len([f for f in os.listdir(DATA_DIR) if f.endswith('.pdf')])}):")
        for idx_file in indexed_files:
            st.sidebar.code(f"• {idx_file}", language="text")
else:
    st.sidebar.warning("Database Status: EMPTY / UNINDEXED")

# Show current ingestion progress if any
if st.session_state.ingestion_progress:
    st.sidebar.markdown("---")
    st.sidebar.subheader("📈 Current Ingestion Progress")
    progress = st.session_state.ingestion_progress
    st.sidebar.caption(f"{progress['message']}")

# --- 3. UI CACHE LOADER LOOP ---
@st.cache_resource
def load_cached_qa_system():
    return engine.get_qa_chain()

qa_system = load_cached_qa_system()

# --- 4. STREAMLIT FRONTEND CHAT WINDOW ---
banner_path = os.path.join(ASSETS_DIR, "banner.jpeg")
if os.path.exists(banner_path):
    st.image(banner_path, width=800)

st.title("Kolhapur Tourism AI Chatbot")


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_query := st.chat_input("Ask a question about the indexed Kolhapur landmarks..."):
    if not qa_system:
        st.error("🚨 Retrieval mechanism missing. Please upload files and click index processing items.")
    else:
        with st.chat_message("user"):
            st.markdown(user_query)
        st.session_state.messages.append({"role": "user", "content": user_query})

        with st.chat_message("assistant"):
            with st.spinner("Invoking decoupled extraction and local inference chains..."):
                try:
                    start_time = time.time()
                    response = qa_system.invoke({"query": user_query})
                    latency = time.time() - start_time
                    
                    answer = response["result"]
                    source_docs = response["source_documents"]
                    
                    st.markdown(answer)
                    
                    if "I'm sorry, but I can only answer" not in answer and source_docs:
                        with st.expander("📌 Vector Database Trace Logs (Architect Mode)"):
                            st.caption(f"⚡ *Inference Latency:* {latency:.2f} seconds")
                            for doc in source_docs:
                                source_name = os.path.basename(doc.metadata.get('source', 'Uploaded Stream'))
                                page_num = doc.metadata.get('page', 0) + 1
                                st.caption(f"📁 Source Node: `{source_name}` (Page {page_num})")
                                st.write(f"*{doc.page_content.strip()}*")
                                
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"Chain execution exception thrown. Details: {e}")