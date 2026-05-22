# RAG PDF Chatbot using Ollama + HuggingFace

A simple Retrieval-Augmented Generation (RAG) chatbot built using Python, Streamlit, LangChain, FAISS, Ollama, and HuggingFace embeddings.

Users can upload PDF documents and ask questions based on the uploaded document content.

---

# Features

- Upload multiple PDF documents
- Extract text from PDFs
- Split text into chunks
- Generate embeddings using HuggingFace
- Store embeddings in FAISS vector database
- Retrieve relevant context using semantic search
- Generate answers locally using Ollama + Llama 3
- Fully local and free setup (no OpenAI API required)

---

# Tech Stack

- Python
- Streamlit
- LangChain
- FAISS
- Ollama
- HuggingFace Embeddings
- pdfplumber

---

# Installation

## Clone Repository

```bash
git clone https://github.com/yourusername/rag-chatbot.git
```

---

## Go Into Project Folder

```bash
cd rag-chatbot
```

---

## Create Virtual Environment

```bash
python -m venv .venv
```

---

## Activate Virtual Environment

### Windows (PowerShell)

```powershell
.venv\Scripts\activate
```

### Git Bash

```bash
source .venv/Scripts/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Install Ollama

Download and install Ollama from:

https://ollama.com/

---

# Download Llama 3 Model

Run this once in terminal:

```bash
ollama run llama3
```

This downloads the local model to your system.

---

# Run the Project

```bash
streamlit run ragchatbot.py
```

Then open:

```text
http://localhost:8501
```

---

# Project Structure

```text
rag-chatbot/
│
├── ragchatbot.py
├── requirements.txt
├── README.md
├── .gitignore
└── .venv/
```

---

# Example Workflow

1. Upload PDF documents
2. Extract text from PDFs
3. Split text into chunks
4. Generate embeddings using HuggingFace
5. Store embeddings in FAISS
6. Retrieve relevant chunks using similarity search
7. Generate response locally using Ollama

---
