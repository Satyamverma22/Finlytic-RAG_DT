import streamlit as st
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

st.set_page_config(
    page_title="Financial Compass AI",
    page_icon="💰",
    layout="centered"
)

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
    model="gemini-3.6-flash",
    temperature=0
)

# prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful AI assistant for Financial Compass.

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

# page title
st.title("💰 Financial Compass AI")
st.caption("Ask questions about the Financial Compass project")

# initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# chat input
query = st.chat_input("Ask something about Financial Compass...")

if query:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": query
        }
    )

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Searching the Financial Compass document..."):

            # retrieve relevant documents
            docs = retriever.invoke(query)

            # create context
            context = "\n\n".join(
                doc.page_content for doc in docs
            )

            # create prompt
            final_prompt = prompt.invoke(
                {
                    "context": context,
                    "question": query
                }
            )

            # generate answer
            response = llm.invoke(final_prompt)

            # extract text response
            if isinstance(response.content, list):
                answer = response.content[0].get("text", "")
            else:
                answer = response.content

        st.markdown(answer)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )