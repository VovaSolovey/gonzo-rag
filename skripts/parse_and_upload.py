import os
import sys
import json
import hashlib
from datetime import datetime, timezone
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag.parser import parse_telegram_json

load_dotenv()

# --- Constants ---
ENGINE = 'text-embedding-3-large'
INDEX_NAME = 'gonzo-ml-index'
NAMESPACE = 'gonzo-ml-namespace'
CHANNEL_USERNAME = 'gonzo_ML'
MIN_WORDS = 100
BATCH_SIZE = 50

# --- Clients ---
pc = Pinecone(api_key=os.environ.get('PINECONE_API_KEY'))
client = OpenAI(
    api_key=os.environ.get('PROXY_API_KEY'),
    base_url="https://api.proxyapi.ru/openai/v1"
)


# --- Helpers ---
def my_hash(text: str) -> str:
    """Generate MD5 hash of a string. Used as unique chunk ID."""
    return hashlib.md5(text.encode()).hexdigest()


def get_embeddings(texts: list) -> list:
    """Get embeddings from OpenAI API."""
    responses = client.embeddings.create(input=texts, model=ENGINE)
    return [doc.embedding for doc in responses.data]


def prepare_batch(chunks: list) -> list:
    """Prepare a batch of chunks for Pinecone upsert."""
    now = datetime.now(timezone.utc).isoformat()
    texts = [c['text'] for c in chunks]
    embeddings = get_embeddings(texts)

    vectors = []
    for chunk, embedding in zip(chunks, embeddings):
        metadata = {
            'text':      chunk['text'],
            'title':     chunk.get('title') or '',
            'authors':   chunk.get('authors') or '',
            'arxiv_url': chunk.get('arxiv_url') or '',
            'arxiv_id':  chunk.get('arxiv_id') or '',
            'post_url':  chunk.get('post_url') or '',
            'date':      chunk.get('date') or 0,
            'post_id':   chunk.get('post_id') or 0,
            'has_tldr':  chunk.get('has_tldr', False),
            'uploaded':  now,
        }
        for key in ('code_url', 'model_url', 'review_url'):
            if chunk.get(key):
                metadata[key] = chunk[key]

        vectors.append((my_hash(chunk['text']), embedding, metadata))
    return vectors


def upload_chunks(chunks: list, index) -> int:
    """Upload chunks to Pinecone in batches."""
    total = 0
    for i in tqdm(range(0, len(chunks), BATCH_SIZE), desc='Uploading'):
        batch = chunks[i: i + BATCH_SIZE]
        vectors = prepare_batch(batch)
        try:
            result = index.upsert(vectors=vectors, namespace=NAMESPACE)
            total += result.upserted_count
        except Exception as e:
            print(f'Error uploading batch {i}-{i+BATCH_SIZE}: {e}')
    return total


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Parse Telegram JSON and upload to Pinecone')
    parser.add_argument('--input', required=True, help='Path to Telegram JSON export')
    parser.add_argument('--output', default='data/chunks_filtered.json', help='Path to save filtered chunks')
    args = parser.parse_args()

    # Parse
    print('Parsing Telegram JSON...')
    chunks = parse_telegram_json(args.input, CHANNEL_USERNAME)
    chunks_filtered = [c for c in chunks if len(c['text'].split()) > MIN_WORDS]
    print(f'Filtered: {len(chunks_filtered)} chunks')

    # Save chunks for BM25
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(chunks_filtered, f, ensure_ascii=False)
    print(f'Saved to {args.output}')

    # Create index if needed
    if INDEX_NAME not in pc.list_indexes().names():
        print(f'Creating index {INDEX_NAME}...')
        pc.create_index(
            name=INDEX_NAME,
            dimension=3072,
            metric='cosine',
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )

    index = pc.Index(name=INDEX_NAME)

    # Upload
    print('Uploading to Pinecone...')
    total = upload_chunks(chunks_filtered, index)
    print(f'Uploaded {total} vectors')