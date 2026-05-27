# Gonzo ML Assistant 🤖

RAG-based Q&A bot for [gonzo_ML](https://t.me/gonzo_ML) Telegram channel with ML paper reviews.

## Results

| Metric | Score |
|--------|-------|
| Recall@3 | 85.7% |
| Faithfulness | 9.8 / 10 |
| Answer Relevance | 9.8 / 10 |
| Hallucination | 9.9 / 10 |

## Architecture

    User Question
          ↓
    Guardrail (LLM relevance check)
          ↓
    Hybrid Search (BM25 + Pinecone vector, alpha=0.6)
          ↓
    Reranker (BAAI/bge-reranker-v2-m3)
          ↓
    Generator (gpt-4.1-mini)
          ↓
    Answer + Source Links

## Stack

| Component | Tool |
|-----------|------|
| Embeddings | text-embedding-3-large (3072d) |
| Vector DB | Pinecone |
| BM25 | rank-bm25 |
| Reranker | BAAI/bge-reranker-v2-m3 |
| Generator | gpt-4.1-mini via ProxyAPI |
| Backend | FastAPI |
| Frontend | Streamlit |

## Data

- Source: Telegram channel gonzo_ML (JSON export)
- 5327 messages → 976 chunks (filtered >100 words)
- 3 post types: TL;DR reviews, plain reviews, author commentary
- Custom parser handles italic authors, section headers normalization, metadata extraction

## Project Structure

    gonzo_rag/
      app.py                    # FastAPI backend
      chat.py                   # Streamlit frontend
      rag/
        bot.py                  # RagBot class with conversation history
        retrieval.py            # Hybrid search: BM25 + Pinecone
        generation.py           # LLM wrapper + prompt template
        parser.py               # Telegram JSON parser
      scripts/
        parse_and_upload.py     # Parse + upload to Pinecone
      notebooks/
        gonzо-ml-rag.ipynb      # Raw research & evaluation notebook

## Setup

    uv venv
    uv pip install -r requirements.txt

**.env file:**

    PINECONE_API_KEY=your_key
    PROXY_API_KEY=your_key
    RERANKER_PATH=path/to/bge-reranker-v2-m3

## Usage

    # 1. Parse and upload
    python scripts/parse_and_upload.py

    # 2. Start backend
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload

    # 3. Start frontend
    streamlit run chat.py

## Evaluation

Evaluated on 245 synthetic questions generated from 5% of chunks (seed=42).

| Pipeline | Recall@3 | Recall@10 |
|----------|----------|-----------|
| Vector only | 0.714 | 0.799 |
| Hybrid | 0.741 | 0.873 |
| **Hybrid + Reranker** | **0.857** | **0.884** |

LLM-as-judge (gpt-4.1-mini): faithfulness, answer relevance, hallucination.