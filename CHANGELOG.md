# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-07-05

### Added
- Retrieval-Augmented Generation chatbot for financial documents, built on LangChain, OpenAI GPT-4o-mini, ChromaDB, and a dark-themed Streamlit UI.
- Multi-PDF ingestion via PyMuPDF (read from bytes for cross-platform compatibility), with RecursiveCharacterTextSplitter chunking (1000 chars, 200 overlap).
- Semantic retrieval over a persistent ChromaDB store using MMR (top-5 from a candidate pool of 15) and `text-embedding-3-small` embeddings.
- Conversational memory with a sliding window of the last 5 turns for multi-turn follow-up questions.
- Source-cited answers showing the originating file name and page number for every response.
- Standalone evaluation harness (`eval_harness.py`) that scores retrieval (Hit Rate@k, MRR@k) and generation (faithfulness, answer relevancy, keyword coverage) against a built-in synthetic corpus and labeled eval set.
- GitHub Actions CI (lint + tests on Python 3.10–3.12), Dependabot updates, MIT license, and smoke tests.

[0.1.0]: https://github.com/Kantamaniprakash/financial-rag-chatbot/releases/tag/v0.1.0
