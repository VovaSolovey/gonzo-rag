import os
import json
import hashlib
from openai import OpenAI
from pinecone import Pinecone
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv

load_dotenv()

# --- Constants ---
ENGINE = 'text-embedding-3-large'
INDEX_NAME = 'gonzo-ml-index'
NAMESPACE = 'gonzo-ml-namespace'


# --- Clients ---
pinecone_key = os.environ.get('PINECONE_API_KEY')
proxyai_key = os.environ.get('PROXY_API_KEY')

pc = Pinecone(api_key=pinecone_key)
index = pc.Index(name=INDEX_NAME)

client = OpenAI(
    api_key=proxyai_key,
    base_url="https://api.proxyapi.ru/openai/v1"
)

# --- BM25 ---
with open('data/chunks_filtered.json', encoding='utf-8') as f:
    chunks_filtered = json.load(f)

corpus = [chunk['text'].lower().split() for chunk in chunks_filtered]
bm25 = BM25Okapi(corpus)

print(f'Загружено {len(chunks_filtered)} чанков, BM25 готов')

# --- Helper functions ---
def my_hash(text: str) -> str:
    """Generate MD5 hash of a string. Used as unique chunk ID."""
    return hashlib.md5(text.encode()).hexdigest()

def get_embeddings(texts: list, engine: str = ENGINE) -> list:
    """Get embeddings from OpenAI API for a list of texts."""
    responses = client.embeddings.create(input=texts, model=engine)
    return [doc.embedding for doc in responses.data]

# --- Vector search ---
def query_from_pinecone(query: list, top_k: int = 15, namespace: str = NAMESPACE) -> list:
    """Search Pinecone vector index and return top_k matches with metadata."""
    query_embeddings = get_embeddings(query, engine=ENGINE)
    return index.query(
        vector=query_embeddings[0],
        top_k=top_k,
        namespace=namespace,
        include_metadata=True
    ).get('matches')

# --- BM25 search ---
def bm25_search(query: str, top_k: int = 15) -> list:
    """Search BM25 index and return top_k chunks with BM25 scores."""
    tokens = query.lower().split()
    scores = bm25.get_scores(tokens)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [{'chunk': chunks_filtered[i], 'bm25_score': scores[i]} for i in top_indices]

# --- Hybrid search ---
def hybrid_search(query: str, top_k: int = 15, alpha: float = 0.6) -> list:
    """
    Combine vector and BM25 search with weighted score fusion.
    alpha=0.6 means 60% vector, 40% BM25.
    Returns top_k candidates with metadata.
    """
    vector_results = query_from_pinecone([query], top_k=top_k)
    bm25_results = bm25_search(query, top_k=top_k)

    vector_scores = {r['id']: r['score'] for r in vector_results}
    bm25_scores = {my_hash(r['chunk']['text']): r['bm25_score'] for r in bm25_results}

    max_vector = max(vector_scores.values()) if vector_scores else 1
    max_bm25 = max(bm25_scores.values()) if bm25_scores else 1

    all_ids = set(vector_scores.keys()) | set(bm25_scores.keys())

    combined = []
    for doc_id in all_ids:
        v_score = vector_scores.get(doc_id, 0) / max_vector
        b_score = bm25_scores.get(doc_id, 0) / max_bm25
        combined.append({'id': doc_id, 'score': alpha * v_score + (1 - alpha) * b_score})

    combined = sorted(combined, key=lambda x: x['score'], reverse=True)[:top_k]

    vector_meta = {r['id']: r for r in vector_results}
    final = []
    for c in combined:
        if c['id'] in vector_meta:
            final.append(vector_meta[c['id']])
        else:
            chunk = next(r['chunk'] for r in bm25_results if my_hash(r['chunk']['text']) == c['id'])
            final.append({'id': c['id'], 'score': c['score'], 'metadata': chunk})

    return final

