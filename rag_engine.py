import os
import shutil
from tqdm import tqdm
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


class KolhapurRAGEngine:
    def __init__(self, data_dir="data", chroma_dir="chroma_db"):
        self.data_dir = data_dir
        self.chroma_dir = chroma_dir
        self.embeddings = OllamaEmbeddings(model="mistral")

    def is_db_indexed(self):
        return os.path.exists(self.chroma_dir) and len(os.listdir(self.chroma_dir)) > 0


    def process_and_index_pdfs(self, progress_callback=None):
        """
        progress_callback: optional callable(step, current, total, message)
        where step is a string like 'files', 'loading', 'chunking', 'indexing'
        """
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        pdf_files = [f for f in os.listdir(self.data_dir) if f.endswith(".pdf")]
        if not pdf_files:
            raise FileNotFoundError("Target data directory is completely empty.")

        # Limit to 5 PDFs
        if len(pdf_files) > 5:
            pdf_files = pdf_files[:5]

        if progress_callback:
            progress_callback("files", 0, len(pdf_files), f"Found {len(pdf_files)} PDFs (max 5, ignoring extras)")
       
        docs = []
        # Load PDFs with progress
        for i, file in enumerate(pdf_files):
            if progress_callback:
                progress_callback("loading", i, len(pdf_files), f"Loading PDF: {file}")
            file_path = os.path.join(self.data_dir, file)
            loader = PyPDFLoader(file_path)
            docs.extend(loader.load())

        if not docs:
            raise ValueError("No valid PDF data arrays parsed by loaders.")

        if progress_callback:
            progress_callback("chunking", 0, 1, "Chunking documents...")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=70)
        text_chunks = text_splitter.split_documents(docs)

        if progress_callback:
            progress_callback("chunking", 1, 1, f"Created {len(text_chunks)} chunks")

        if os.path.exists(self.chroma_dir):
            shutil.rmtree(self.chroma_dir)

        if progress_callback:
            progress_callback("indexing", 0, 1, "Building vector index...")

        vector_store = Chroma.from_documents(
            documents=text_chunks,
            embedding=self.embeddings,
            persist_directory=self.chroma_dir
        )

        if progress_callback:
            progress_callback("indexing", 1, 1, "Vector index built successfully")

        return True

    def get_qa_chain(self):
        if not self.is_db_indexed():
            return None

        vector_store = Chroma(persist_directory=self.chroma_dir, embedding_function=self.embeddings)
        retriever = vector_store.as_retriever(search_kwargs={"k": 2})
        llm = OllamaLLM(model="mistral", temperature=0.2)

        custom_prompt_template = """You are a helpful local tour guide AI assistant for Kolhapur. 
        Use the following pieces of context to answer the user's question accurately.
        If you do not know the answer based *only* on the provided context, or if the question is unrelated to the context, 
        respond exactly with: "I'm sorry, but I can only answer questions about Kolhapur's tourist places based on my current knowledge base." 
        Do not try to make up an answer.


        Context:
        {context}


        Question: {question}


        Helpful Answer:"""

        prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])

        return RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )