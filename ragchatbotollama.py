import warnings
warnings.filterwarnings("ignore")
import pdfplumber
import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

#page settings
st.set_page_config(
    page_title="RAG Chatbot",
    layout="wide"
)

st.header("My First RAG Chatbot")

#sidebar
with st.sidebar:

    st.title("Your Documents")

    files = st.file_uploader(
        "Upload your documents here",
        type="pdf",
        accept_multiple_files=True
    )

#cache vector store
@st.cache_resource
def create_vectorstore(chunks):

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = FAISS.from_texts(
        chunks,
        embeddings
    )

    return vector_store

#main app
if files:

    text = ""

    #extract text from pdfs
    with st.spinner("Reading PDFs..."):

        for file in files:

            with pdfplumber.open(file) as pdf:

                for page in pdf.pages:

                    extracted = page.extract_text()

                    if extracted:
                        text += extracted + "\n"

    #split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=500,
        chunk_overlap=50,
    )

    chunks = text_splitter.split_text(text)

    #create vector database
    with st.spinner("Processing documents..."):

        vector_store = create_vectorstore(chunks)

    #user input
    user_question = st.text_input("Ask a question")

    #format retrieved docs
    def format_docs(docs):

        return "\n\n".join(
            [doc.page_content for doc in docs]
        )

    #retriever
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    #ollama model
    llm = Ollama(
        model="llama3"
    )

    #prompt template
    prompt = ChatPromptTemplate.from_messages([

        (
            "system",

            "You are a helpful assistant for answering questions based ONLY on the provided context.\n\n"

            "Rules:\n"
            "1. If the answer is not in the context, say 'I don't know based on the provided context.'\n"
            "2. Do NOT make up information.\n"
            "3. Keep answers clear and structured.\n"
            "4. Summarize long information when needed.\n"
            "5. If the question is unclear, ask for clarification.\n\n"

            "Context:\n{Context}"
        ),

        ("human", "{question}")

    ])

    #rag chain
    chain = (

        {
            "Context": retriever | format_docs,
            "question": RunnablePassthrough()
        }

        | prompt
        | llm
        | StrOutputParser()
    )

    #generate response
    if user_question:

        with st.spinner("Generating answer..."):

            response = chain.invoke(user_question)

        st.subheader("Answer")

        st.write(response)