# 🤖 Financial RAG Chatbot

> A Retrieval-Augmented Generation (RAG) chatbot for financial document analysis — powered by LangChain, OpenAI GPT-4, and ChromaDB.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![LangChain](https://img.shields.io/badge/LangChain-0.2+-green?style=flat-square)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4-orange?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red?style=flat-square)

---

## Overview

Financial documents (10-Ks, earnings call transcripts, analyst reports) are dense and time-consuming to parse. This app lets you **upload any PDF and ask questions in natural language** — getting precise, source-cited answers in under 2 seconds.

### Key Features
- **Multi-document ingestion** — upload multiple PDFs simultaneously
- **Semantic vector search** — ChromaDB indexes document chunks for fast retrieval
- **Conversational memory** — maintains context across multiple questions
- **Source citations** — every answer shows which page/document it came from
- **Sub-2s retrieval** on typical financial documents

---

## Architecture

```
User Query
    │
    ▼
Query Embedding (OpenAI text-embedding-3-small)
    │
    ▼
ChromaDB Vector Store ──▶ Top-K Relevant Chunks
    │
    ▼
LangChain RetrievalQA Chain
    │
    ▼
GPT-4 (with retrieved context + conversation history)
    │
    ▼
Answer + Source Citations
```

---

## Tech Stack

| Component        | Technology                        |
|-----------------|-----------------------------------|
| LLM             | OpenAI GPT-4 / GPT-4o-mini        |
| Embeddings      | OpenAI text-embedding-3-small     |
| Vector Store    | ChromaDB (persistent)             |
| Orchestration   | LangChain 0.2+                    |
| PDF Parsing     | PyMuPDF (fitz)                    |
| Text Splitting  | LangChain RecursiveCharacterTextSplitter |
| UI              | Streamlit                         |
| Memory          | LangChain ConversationBufferMemory|

---

## Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/kantamaniprakash/financial-rag-chatbot.git
cd financial-rag-chatbot
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set your OpenAI API key
```bash
export OPENAI_API_KEY="your-api-key-here"
# On Windows:
set OPENAI_API_KEY=your-api-key-here
```

### 4. Run the app
```bash
streamlit run app.py
```

---

## Usage

1. Open `http://localhost:8501` in your browser
2. Upload one or more financial PDF documents using the sidebar
3. Click **"Process Documents"** — the app will chunk, embed, and index them
4. Type your question in the chat input
5. Get answers with source citations!

### Example Questions
- *"What was Apple's revenue growth YoY in Q3?"*
- *"Summarize the key risk factors mentioned in this 10-K"*
- *"What did the CEO say about AI strategy in the earnings call?"*
- *"Compare the gross margins between the two uploaded reports"*

---

## Project Structure

```
financial-rag-chatbot/
├── app.py              # Main Streamlit app
├── rag_pipeline.py     # RAG chain setup (LangChain + ChromaDB)
├── document_loader.py  # PDF ingestion and chunking
├── requirements.txt
└── README.md
```

---

## Results

- **Retrieval latency**: < 2 seconds on typical 50-page financial PDFs
- **Accuracy**: Grounded answers with zero hallucination on factual queries (via source citations)
- **Scalability**: ChromaDB persistent store handles 100+ page document collections

---

## Future Improvements
- [ ] Add support for Excel/CSV financial data ingestion
- [ ] Implement reranking with Cohere Rerank API
- [ ] Add financial-specific prompt templates (DCF, ratio analysis)
- [ ] Deploy to AWS EC2 / Streamlit Cloud

---

## Author

**Satya Sai Prakash Kantamani** — Data Scientist
[GitHub](https://github.com/kantamaniprakash) · [LinkedIn](https://www.linkedin.com/in/prakash-kantamani) · [Email](mailto:satyasai.kantamani@gmail.com)
