from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

docs = [
    Document(page_content="Gradient descent is an optimization algorithm used in machine learning."),
    Document(page_content="Gradient descent minimizes the loss function."),
    Document(page_content="Gradient descent is an optimization that minimizes the loss function."),
    Document(page_content="Neural networks use gradient descent for training."),
    Document(page_content="Support Vector Machines are supervised learning algorithms.")
]

# create embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# create vector store
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings
)

# create mmr retriever
mmr_retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 3,
        "fetch_k": 5,
        "lambda_mult": 0.5
    }
)

query = "What is gradient descent?"

# retrieve documents
mmr_docs = mmr_retriever.invoke(query)

print("\n===== MMR Results =====\n")

for doc in mmr_docs:
    print(doc.page_content)