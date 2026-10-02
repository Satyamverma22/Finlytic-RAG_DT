# load pdf
# split into chunks
# create embeddings
# store into chroma

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()

# load pdf
pdf_path = "./document loaders/Financial-Compass-Project.pdf"

loader = PyPDFLoader(pdf_path)
docs = loader.load()

print(f"Loaded {len(docs)} pages.")

# split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(docs)

print(f"Created {len(chunks)} chunks.")

# create embeddings
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# store into chroma
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="./chroma_db"
)

print("Chroma database created successfully.")
