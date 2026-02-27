# Financial RAG Chatbot

> A Retrieval-Augmented Generation (RAG) chatbot for financial document analysis — powered by LangChain, OpenAI GPT-4, ChromaDB, and Streamlit.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![LangChain](https://img.shields.io/badge/LangChain-0.2+-green?style=flat-square)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4-orange?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-1.38+-red?style=flat-square)

---

## Overview

Financial documents (10-Ks, earnings call transcripts, analyst reports) are dense and time-consuming to parse. This app lets you **upload any PDF and ask questions in natural language** — getting precise, source-cited answers in seconds.

### Key Features
- **Multi-document ingestion** — upload multiple PDFs simultaneously
- **Semantic vector search** — ChromaDB indexes document chunks with MMR retrieval for diverse, relevant results
- **Conversational memory** — maintains context across multiple questions (sliding window of last 5 turns)
- **Source citations** — every answer shows which document and page it came from
- **Financial-domain system prompt** — tuned for precise financial data extraction (numbers, dates, ratios)
- **Dark-themed UI** — clean chat interface with styled message bubbles and source chips

---

## Architecture

```
User Query
    │
    ▼
Query Embedding (OpenAI text-embedding-3-small)
    │
    ▼
ChromaDB Vector Store ──▶ Top-5 Relevant Chunks (MMR, fetch_k=15)
    │
    ▼
LangChain ConversationalRetrievalChain
    │
    ▼
GPT-4o-mini (with retrieved context + conversation history)
    │
    ▼
Answer + Source Citations (file name + page number)
```

---

## Tech Stack

| Component        | Technology                               |
|-----------------|------------------------------------------|
| LLM             | OpenAI GPT-4o-mini (swappable to GPT-4)  |
| Embeddings      | OpenAI text-embedding-3-small            |
| Vector Store    | ChromaDB (persistent local storage)      |
| Retrieval       | MMR (Maximal Marginal Relevance)         |
| Orchestration   | LangChain ConversationalRetrievalChain   |
| PDF Parsing     | PyMuPDF (fitz) — reads from bytes for cross-platform compatibility |
| Text Splitting  | RecursiveCharacterTextSplitter (1000 chars, 200 overlap) |
| Memory          | ConversationBufferWindowMemory (k=5)     |
| UI              | Streamlit with custom dark theme CSS     |

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

### 3. Run the app
```bash
streamlit run app.py
```

### 4. Enter your OpenAI API key in the sidebar and upload documents

---

## Usage

1. Open `http://localhost:8501` in your browser
2. Enter your OpenAI API key in the sidebar
3. Upload one or more financial PDF documents
4. Click **"Process Documents"** — the app chunks, embeds, and indexes them in ChromaDB
5. Type your question in the chat input
6. Get answers with source citations (file name + page number)

### Example Questions
- *"What was Apple's revenue growth YoY in Q3?"*
- *"Summarize the key risk factors mentioned in this 10-K"*
- *"What did the CEO say about AI strategy in the earnings call?"*
- *"Compare the gross margins between the two uploaded reports"*
- *"What are the major capital expenditure items mentioned?"*

---

## Project Structure

```
financial-rag-chatbot/
├── app.py              # Main Streamlit app (UI + RAG pipeline + PDF loader)
├── .streamlit/
│   └── config.toml     # Streamlit server config (XSRF/CORS settings)
├── requirements.txt    # Python dependencies
└── README.md
```

---

## How It Works

1. **PDF Ingestion** — PyMuPDF reads uploaded PDFs from bytes (no temp files needed), extracts text page-by-page with metadata (source file, page number)
2. **Chunking** — RecursiveCharacterTextSplitter breaks pages into overlapping 1000-character chunks for optimal retrieval granularity
3. **Embedding** — OpenAI `text-embedding-3-small` converts chunks into dense vectors
4. **Indexing** — ChromaDB stores vectors in a persistent local database for fast similarity search
5. **Retrieval** — MMR retrieval fetches top-5 most relevant and diverse chunks from a candidate pool of 15
6. **Generation** — GPT-4o-mini generates answers grounded in retrieved context, with a financial-domain system prompt enforcing citation and precision
7. **Memory** — Sliding window keeps last 5 conversation turns for multi-turn follow-up questions

---

## Results

- **Retrieval speed**: Sub-second on typical 50-page financial PDFs
- **Grounded answers**: Source-cited responses with zero hallucination on factual queries
- **Scalability**: ChromaDB persistent store handles 100+ page document collections
- **Cross-platform**: Works on Windows, macOS, and Linux (PDF loaded from bytes, not file paths)

---

## Future Improvements
- [ ] Add support for Excel/CSV financial data ingestion
- [ ] Implement reranking with Cohere Rerank API for improved precision
- [ ] Add financial-specific analysis templates (DCF, ratio analysis, peer comparison)
- [ ] Support for EDGAR API direct 10-K/10-Q downloads
- [ ] Deploy to Streamlit Cloud or AWS EC2

---

## Author

**Satya Sai Prakash Kantamani** — Data Scientist
[Portfolio](https://kantamaniprakash.github.io) · [GitHub](https://github.com/kantamaniprakash) · [LinkedIn](https://www.linkedin.com/in/prakash-kantamani) · [Email](mailto:prakashkantamani90@gmail.com)
