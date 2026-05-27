# Gonzo ML Assistant 🤖

RAG-based Q&A bot for [gonzo_ML](https://t.me/gonzo_ML) Telegram channel with ML paper reviews.

## Results

| Metric | Score |
|--------|-------|
| Recall@3 | 87.5% |
| Faithfulness | 9.77 / 10 |
| Answer Relevance | 9.89 / 10 |
| Hallucination | 9.89 / 10 |

## Architecture

User Question  
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
- 5327 messages → 977 chunks (filtered >100 words)
- 3 post types: TL;DR reviews, plain reviews, author commentary

## Project Structure

gonzo_rag/
app.py              # FastAPI backend
chat.py             # Streamlit frontend
rag/
bot.py            # RagBot class with conversation history
retrieval.py      # Hybrid search: BM25 + Pinecone
generation.py     # LLM wrapper + prompt template
data/
chunks_filtered.json
notebooks/
main.ipynb        # Research & evaluation notebook

## Setup

```bash
# Install dependencies
uv venv
uv pip install -r requirements.txt

# Add .env file
PINECONE_API_KEY=your_key
PROXY_API_KEY=your_key
RERANKER_PATH=path/to/bge-reranker-v2-m3

# Start backend
uvicorn app:app --host 0.0.0.0 --port 8000 --reload

# Start frontend
streamlit run chat.py
```

## Evaluation

Evaluated on 50 synthetic questions generated from 5% of chunks.
LLM-as-judge using gpt-4.1-mini with 3 criteria: faithfulness, answer relevance, hallucination.