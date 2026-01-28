# 🎥 YouTube Video Question Answering using RAG

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline that allows users to ask questions about YouTube videos and receive answers grounded strictly in the video transcript.

The system extracts YouTube transcripts, semantically chunks the text, embeds it into a vector database (FAISS), retrieves relevant context, and uses an LLM to generate accurate answers.

---

## 🚀 Features

- 📄 Automatic YouTube transcript extraction (English & Hindi)
- 🧠 Semantic text chunking using embedding-based breakpoints
- 🔍 FAISS vector similarity search
- 🤖 LLM-powered question answering with strict context grounding
- ❌ Prevents hallucination by answering *only* from retrieved transcript data

---

## 🧠 Architecture Overview

1. **Transcript Extraction**
   - Fetches captions using `youtube-transcript-api`

2. **Semantic Chunking**
   - Splits transcript into meaningful chunks using embedding similarity
   - Improves retrieval accuracy compared to fixed-size chunking

3. **Embedding Generation**
   - Uses OpenAI embedding models via OpenRouter
   - Converts text chunks into dense vectors

4. **Vector Storage**
   - Stores embeddings in **FAISS** for fast similarity search

5. **Retrieval**
   - Retrieves top-k most relevant transcript chunks

6. **Augmentation (RAG)**
   - Feeds retrieved context into an LLM
   - Answers only if the context is sufficient

---

## 🛠 Tech Stack

- **Python**
- **LangChain**
- **FAISS**
- **YouTube Transcript API**
- **OpenAI Embeddings (via OpenRouter)**
- **Groq LLaMA 3.3 (70B)**

---

## 📂 Project Structure

```text
├── indexing.py        # Main pipeline (Transcript → Chunking → FAISS → Retrieval)
├── Embedding.py       # Embedding model configuration
├── Augmention.py      # RAG-based answer generation logic
├── .env               # API keys and environment variables
