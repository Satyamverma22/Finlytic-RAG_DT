from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# create embeddings
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# load chroma
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embedding_model
)

# create retriever
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 4,
        "fetch_k": 10,
        "lambda_mult": 0.5
    }
)

# create gemini model
llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    temperature=0
)

# prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful AI assistant.
Use only the provided context to answer the question.

If the answer is not present in the context,
say: "I could not find the answer in the document."

Do not make up information."""
        ),
        (
            "human",
            """Context:
{context}

Question:
{question}"""
        )
    ]
)

print("RAG system created.")
print("Press 0 to exit.")

while True:
    query = input("You: ")

    if query == "0":
        break

    if not query.strip():
        continue

    docs = retriever.invoke(query)

    context = "\n\n".join(
        doc.page_content for doc in docs
    )

    final_prompt = prompt.invoke({
        "context": context,
        "question": query
    })

    response = llm.invoke(final_prompt)

    print(f"\nAI: {response.content}\n")
