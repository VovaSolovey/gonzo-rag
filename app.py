import os
import uuid
from typing import Dict

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import CrossEncoder

from rag.bot import RagBot
from rag.generation import OpenAIChatLLM

load_dotenv()

app = FastAPI()

# --- Load reranker once at startup ---
RERANKER_PATH = os.environ.get('RERANKER_PATH', 'BAAI/bge-reranker-v2-m3')
reranker = CrossEncoder(RERANKER_PATH)

# --- Conversation storage ---
conversations: Dict[str, RagBot] = {}

# --- Request/Response schemas ---
class ConversationRequest(BaseModel):
    """Incoming request with user question and session id."""
    text: str
    conversation_id: str = None
    temperature: float = 0.0
    threshold: float = 0.4


class ConversationResponse(BaseModel):
    """Outgoing response with bot answer and session id."""
    response: str
    conversation_id: str


# --- Endpoints ---
@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/ask", response_model=ConversationResponse)
async def ask(request: ConversationRequest):
    """
    Main endpoint. Accepts a question, returns RAG answer.
    Creates a new RagBot per conversation_id.
    """
    # Generate new conversation_id if not provided
    if not request.conversation_id:
        request.conversation_id = str(uuid.uuid4())

    # Create new bot for new conversation
    if request.conversation_id not in conversations:
        conversations[request.conversation_id] = RagBot(
            llm=OpenAIChatLLM(
                model='gpt-4.1-mini',
                temperature=request.temperature
            ),
            reranker=reranker,
            top_k=15,
            top_n=3,
            threshold=request.threshold,
            use_hybrid=True,
            verbose=False
        )

    bot = conversations[request.conversation_id]
    response = bot.run(request.text)

    return ConversationResponse(
        response=response,
        conversation_id=request.conversation_id
    )


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)