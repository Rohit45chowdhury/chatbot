import os
import shutil
import uuid
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

# -------------------------------
# CONFIG
# -------------------------------
PDF_DIR = "pdfs"
VECTORSTORE_DIR = "vectorstore"
FAISS_DB_PATH = os.path.join(VECTORSTORE_DIR, "db_faiss")
EMBEDDING_MODEL = "nomic-embed-text"

os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(VECTORSTORE_DIR, exist_ok=True)

# -------------------------------
# RESET FAISS DB
# -------------------------------
def reset_faiss_db():
    if os.path.exists(FAISS_DB_PATH):
        shutil.rmtree(FAISS_DB_PATH, ignore_errors=True)

    os.makedirs(VECTORSTORE_DIR, exist_ok=True)

# -------------------------------
# Upload PDF (🔥 SAFE & UNIQUE)
# -------------------------------
def upload_pdf(file):
    unique_name = f"{uuid.uuid4().hex}_{file.name}"
    file_path = os.path.join(PDF_DIR, unique_name)

    with open(file_path, "wb") as f:
        f.write(file.getbuffer())

    return file_path

# -------------------------------
# Load PDF (🔥 Metadata Fixed)
# -------------------------------
def load_pdf(file_path):
    loader = PDFPlumberLoader(file_path)
    documents = loader.load()

    source_name = os.path.basename(file_path)

    for doc in documents:
        doc.metadata.update({
            "source": source_name,
            "file_name": source_name
        })

    return documents

# -------------------------------
# Create Chunks
# -------------------------------
def create_chunks(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1600,
        chunk_overlap=300
    )

    chunks = splitter.split_documents(documents)

    if not chunks:
        raise ValueError("❌ No chunks created from PDF")

    return chunks

# -------------------------------
# Embeddings
# -------------------------------
def get_embedding_model():
    return OllamaEmbeddings(model=EMBEDDING_MODEL)

# -------------------------------
# Build FAISS (🔥 ALWAYS FRESH)
# -------------------------------
def build_faiss_db(text_chunks):
    if not text_chunks:
        raise ValueError("❌ Empty chunks. Cannot build FAISS.")

    embeddings = get_embedding_model()

    faiss_db = FAISS.from_documents(text_chunks, embeddings)
    faiss_db.save_local(FAISS_DB_PATH)

    return faiss_db

# -------------------------------
# Load FAISS
# -------------------------------
def load_faiss_db():
    if not os.path.isdir(FAISS_DB_PATH):
        raise FileNotFoundError("❌ FAISS DB not found. Upload PDF first.")

    embeddings = get_embedding_model()

    return FAISS.load_local(
        FAISS_DB_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )