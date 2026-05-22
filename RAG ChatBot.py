import pdfplumber
import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

st.header("My First RAG Chatbot")

with st.sidebar:
    st.title("Your Documents")

    files = st.file_uploader(
        "Upload your documents here",
        type="pdf",
        accept_multiple_files=True
    )

# Extract text from uploaded PDFs
if files:

    text = ""

    for file in files:

        with pdfplumber.open(file) as pdf:

            for page in pdf.pages:

                extracted = page.extract_text()

                if extracted:
                    text += extracted + "\n"

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=1000,
        chunk_overlap=200,
    )

    chunks = text_splitter.split_text(text)

    # OPTIONAL DEBUG
    # st.write(chunks)

    # CHANGED EMBEDDINGS
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Store embeddings in FAISS
    vector_store = FAISS.from_texts(chunks, embeddings)

    # User question
    user_question = st.text_input("Ask a question")

    # Format retrieved docs
    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])

    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4}
    )

    # CHANGED LLM
    llm = Ollama(
        model="llama3"
    )

    # Prompt template
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a helpful assistant for answering questions based on the following context: {Context}\n"
            "Guidelines for answering the question:\n"
            "1. If you don't know the answer, say you don't know.\n"
            "2. Answer only from the provided context.\n"
            "3. Do not make up information.\n"
            "4. Provide clear and structured answers.\n"
            "5. Summarize long information concisely.\n"
            "6. Ask for clarification if the question is ambiguous.\n"
            "7. Include relevant related information from the context when useful."
        ),

        ("human", "{question}")
    ])

    # RAG Chain
    chain = (
        {
            "Context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    # Generate response
    if user_question:

        response = chain.invoke(user_question)

        st.write(response)