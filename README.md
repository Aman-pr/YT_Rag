# 🎥 YouTube Video Question Answering using RAG

This project implements a **Retrieval-Augmented Generation (RAG)** system that allows users to ask questions about YouTube videos and receive answers grounded **strictly in the video transcript**.

The pipeline fetches YouTube captions, applies **semantic chunking**, stores embeddings in a **FAISS vector database**, retrieves the most relevant context, and generates answers using a large language model while preventing hallucinations.

---

##  Features

-  Automatic YouTube transcript extraction (English & Hindi)
-  Semantic chunking using embedding-based breakpoints
-  FAISS vector similarity search
-  LLM-powered question answering with strict context grounding
-  Hallucination prevention (answers only if context exists)

---

##  Architecture Overview

1. **Transcript Extraction**  
   Fetches captions using `youtube-transcript-api`

2. **Semantic Chunking**  
   Splits text into meaningful chunks using embedding similarity

3. **Embedding Generation**  
   Converts chunks into dense vectors using OpenAI embeddings (via OpenRouter)

4. **Vector Storage**  
   Stores vectors in **FAISS** for fast similarity search

5. **Context Retrieval**  
   Retrieves top-k relevant chunks

6. **Answer Generation (RAG)**  
   Uses LLaMA 3.3 (70B) via Groq to answer strictly from retrieved context

---

## 🛠 Tech Stack

- Python
- LangChain
- FAISS
- YouTube Transcript API
- OpenAI Embeddings (via OpenRouter)
- Groq (LLaMA 3.3 – 70B)

---

## Installation

1. Clone the repository
```bash
git clone https://github.com/yourusername/ai-chatbot-sentiment.git
cd ai-chatbot-sentiment
```

2. Create a Python virtual environment
```bash
python -m venv myenv
source myenv/bin/activate   # Linux/Mac
myenv\Scripts\activate      # Windows
```
   
2. Install dependencies
```bash
pip install -r requirements.txt
```

3. create .env file 
```bash
touch .env file
```

4. Add api key in .env file or Set environment variable 
```bash
export GROQ_API_KEY="your-groq-api-key"
export OPENROUTER_API_KEY="your-api-key"
```

5. Create virtual env  
```bash
   python3 -m venv ./path-to-new-venv
   
```

6. Run index file 
```bash
   python3 index.py
```


## Project Structure

```text
├── indexing.py        # Main pipeline (Transcript → Chunking → FAISS → Retrieval)
├── Embedding.py       # Embedding model configuration
├── Augmention.py      # RAG-based answer generation logic
├── .env               # API keys and environment variables
