from datetime import datetime
from typing import Any, List
from pydantic import BaseModel
from sentence_transformers import CrossEncoder

from rag.retrieval import hybrid_search, query_from_pinecone
from rag.generation import PROMPT_TEMPLATE, FINAL_ANSWER_TOKEN, STOP

# --- Reranker ---
def rerank(query: str, candidates: list, reranker: Any, top_n: int = 3) -> list:
    """Rerank candidates using cross-encoder. Returns top_n most relevant."""
    pairs = [(query, c['metadata']['text']) for c in candidates]
    scores = reranker.predict(pairs)
    
    results = []
    for candidate, score in zip(candidates, scores):
        results.append({
            'id': candidate['id'],
            'score': candidate['score'],
            'metadata': dict(candidate['metadata']),
            'reranker_score': float(score)
        })
    
    return sorted(results, key=lambda x: x['reranker_score'], reverse=True)[:top_n]

# --- RagBot ---
class RagBot(BaseModel):
    """
    RAG chatbot with hybrid search, reranking, and conversation history.
    Maintains context window of last MAX_TURNS turns.
    """
    llm: Any
    reranker: Any
    top_k: int = 15
    top_n: int = 3
    prompt_template: str = PROMPT_TEMPLATE
    stop_pattern: List[str] = [STOP]
    user_inputs: List[str] = []
    ai_responses: List[str] = []
    contexts: List[list] = []
    verbose: bool = False
    threshold: float = 0.4
    alpha: float = 0.6
    use_hybrid: bool = True
    MAX_TURNS: int = 5

    @property
    def running_convo(self) -> str:
        """Build conversation history string for the prompt. Limited to MAX_TURNS."""
        inputs = self.user_inputs[-self.MAX_TURNS:]
        responses = self.ai_responses[-self.MAX_TURNS:]
        contexts = self.contexts[-self.MAX_TURNS:]

        convo = ''
        for index in range(len(inputs)):
            convo += f'[START]\nUser Input: {inputs[index]}\n'
            ctx_list = contexts[index]

            convo += f'Context:\n'
            for i, ctx in enumerate(ctx_list):
                date_str = datetime.fromtimestamp(ctx['date']).strftime('%d.%m.%Y') if ctx['date'] else '—'
                convo += f'[Post {i+1}]\n'
                convo += f'Title: {ctx["title"] or "—"}\n'
                convo += f'Authors: {ctx["authors"] or "—"}\n'
                convo += f'Date: {date_str}\n'
                convo += f'Text: {ctx["text"]}\n\n'

            convo += f'Post URL: {ctx_list[0]["post_url"]}\n'
            if ctx_list[0].get('arxiv_url'):
                convo += f'Arxiv URL: {ctx_list[0]["arxiv_url"]}\n'
            if ctx_list[0].get('code_url'):
                convo += f'Code URL: {ctx_list[0]["code_url"]}\n'
            convo += f'Context Score: {ctx_list[0]["reranker_score"]:.2f}\n'

            if len(responses) > index:
                convo += responses[index]
                convo += '\n[END]\n'
        return convo.strip()

    def run(self, question: str) -> str:
        """Process a question through hybrid search, reranking, and generation."""
        self.user_inputs.append(question)

        # Search
        if self.use_hybrid:
            candidates = hybrid_search(question, top_k=self.top_k, alpha=self.alpha)
        else:
            candidates = query_from_pinecone([question], top_k=self.top_k)

        # Rerank
        reranked = rerank(question, candidates, reranker=self.reranker, top_n=self.top_n) if self.reranker else candidates[:self.top_n]
        top_score = reranked[0].get('reranker_score', reranked[0].get('score', 0))

        if self.verbose:
            print(f'Top score: {top_score:.3f}')

        # Build context
        if top_score >= self.threshold:
            context_list = [{
                'text':           chunk['metadata']['text'],
                'title':          chunk['metadata'].get('title', ''),
                'authors':        chunk['metadata'].get('authors', ''),
                'post_url':       chunk['metadata'].get('post_url', ''),
                'date':           chunk['metadata'].get('date', 0),
                'has_tldr':       chunk['metadata'].get('has_tldr', False),
                'arxiv_url':      chunk['metadata'].get('arxiv_url', ''),
                'code_url':       chunk['metadata'].get('code_url', ''),
                'reranker_score': chunk.get('reranker_score', chunk.get('score', 0)),
            } for chunk in reranked]
        else:
            context_list = [{
                'text': 'NO CONTEXT FOUND', 'title': '', 'authors': '',
                'post_url': 'NONE', 'date': 0, 'has_tldr': False,
                'arxiv_url': '', 'code_url': '', 'reranker_score': 0,
            }]

        self.contexts.append(context_list)

        # Generate
        prompt = self.prompt_template.format(
            today=datetime.now().strftime('%d.%m.%Y'),
            running_convo=self.running_convo
        )

        if self.verbose:
            print(f'--------\nPROMPT\n--------\n{prompt}\n--------\nEND PROMPT\n--------')

        generated = self.llm.generate(prompt, stop=self.stop_pattern)

        if self.verbose:
            print(f'--------\nGENERATED\n--------\n{generated}\n--------\nEND GENERATED\n--------')

        self.ai_responses.append(generated)
        if FINAL_ANSWER_TOKEN in generated:
            generated = generated.split(FINAL_ANSWER_TOKEN)[-1]
        return generated
