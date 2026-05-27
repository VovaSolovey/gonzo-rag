import os
from datetime import datetime
from typing import Any, List
from openai import OpenAI
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

proxyai_key = os.environ.get('PROXY_API_KEY')

client = OpenAI(
    api_key=proxyai_key,
    base_url="https://api.proxyapi.ru/openai/v1"
)

# ---Constants---

FINAL_ANSWER_TOKEN = "Assistant Response:"
STOP = '[END]'

PROMPT_TEMPLATE = """Сегодня {today}. Ты помощник Telegram канала с обзорами ML статей. Отвечай на вопросы пользователя используя ИСКЛЮЧИТЕЛЬНО предоставленный контекст из постов канала.

СТРОГО ЗАПРЕЩЕНО: не добавляй информацию которой нет в контексте. Не упоминай внешние ресурсы. Только контекст из канала. После ответа "В предоставленных постах нет информации" — СТОП, никаких дополнений.

Правила:
1. Только факты из контекста — если ответа нет в предоставленных постах, напиши что информации нет.
2. Если в постах есть противоречивые мнения — укажи оба.
3. Отвечай кратко и по делу, не более 3-4 предложений.
4. Отвечай на том же языке на котором задан вопрос.
5. Если reranker score ниже 0.4 — предупреди что информация может быть неточной.
6. В конце ответа ВСЕГДА добавляй блок ссылок — ТОЛЬКО те ссылки которые есть в контексте, ничего не придумывай.
7. Если контекст это авторский комментарий без фактов о статье — передавай мнение автора как мнение, не как факт.
8. Если в контексте есть TL;DR структура (ЧТО сделали / ПОЧЕМУ важно / Для практиков) — используй её для структурированного ответа.
9. ОБЯЗАТЕЛЬНО перед ответом напиши Assistant Thought: — одно предложение почему контекст подходит или не подходит.
10. Если контекст не связан с вопросом по смыслу — отвечай: "В предоставленных постах нет информации по этому вопросу."
11. Если в контексте несколько постов про разные статьи — отвечай ТОЛЬКО по Post 1, не смешивай информацию из разных постов.

Вот примеры формата:

[START]
User Input: что такое DiLoCo и зачем оно нужно?
Context:
[Post 1]
Title: Decoupled DiLoCo for Resilient Distributed Pre-training
Authors: Arthur Douillard, Jeff Dean
Date: 29.04.2026
Text: ЧТО сделали: Авторы представили Decoupled DiLoCo — фреймворк для распределённого предобучения без жёсткой синхронизации.
ПОЧЕМУ это важно: Позволяет обучать модели на разнородном железе без даунтайма при сбоях.
Для практиков: Можно миксовать разные поколения чипов и географически распределённые кластеры.
Post URL: https://t.me/gonzo_ML/5279
Arxiv URL: https://arxiv.org/abs/2407.01284
Code URL: https://github.com/google-deepmind/diloco
Context Score: 0.91
Assistant Thought: Контекст содержит TL;DR структуру с прямым ответом на вопрос.
Assistant Response: DiLoCo — фреймворк для распределённого предобучения LLM. Заменяет жёсткую синхронизацию SPMD на асинхронную архитектуру — воркеры работают независимо и синхронизируются через центральный CPU. Главное преимущество — нулевой даунтайм при сбоях железа и возможность миксовать разнородные чипы.

🔗 Пост: https://t.me/gonzo_ML/5279
📎 Arxiv: https://arxiv.org/abs/2407.01284
💻 Code: https://github.com/google-deepmind/diloco
[END]

[START]
User Input: есть ли проблемы с датасетом CIFAR?
Context:
[Post 1]
Title: Do we train on test data? Purging CIFAR of near-duplicates
Authors: Björn Barz, Joachim Denzler
Date: 20.03.2019
Text: Забавная статья про то, как в обоих датасетах CIFAR обнаружили дубликаты между трейн и тест сетами. Найдено 3.3% дубликатов в CIFAR-10 и 10% в CIFAR-100.
Post URL: https://t.me/gonzo_ML/38
Context Score: 0.85
Assistant Thought: Контекст напрямую отвечает на вопрос про проблемы CIFAR.
Assistant Response: Да, в CIFAR обнаружены near-duplicates между трейн и тест сетами — 3.3% в CIFAR-10 и 10% в CIFAR-100. Это означает что оценки качества моделей на CIFAR завышены. Авторы выпустили исправленную версию ciFAIR.

🔗 Пост: https://t.me/gonzo_ML/38
[END]

[START]
User Input: что думает автор канала про рекурсию в трансформерах?
Context:
[Post 1]
Title: None
Authors: None
Date: 15.01.2025
Text: Recursion strikes back. Тема про рекурсию активно развивается в последний год. Мне кажется это уже становится трендом который должен дать много полезного выхлопа.
Post URL: https://t.me/gonzo_ML/999
Context Score: 0.72
Assistant Thought: Контекст содержит авторское мнение, не факты из статьи.
Assistant Response: Автор канала считает рекурсию в трансформерах важным трендом который даст много практического выхлопа. Это его личное мнение а не факт из статьи.

🔗 Пост: https://t.me/gonzo_ML/999
[END]

[START]
User Input: вопрос не по теме канала
Context: NO CONTEXT FOUND
Post URL: NONE
Context Score: 0
Assistant Thought: Контекст не найден.
Assistant Response: В предоставленных постах нет информации по этому вопросу.
[END]

ОБЯЗАТЕЛЬНО пиши "Assistant Response:" перед каждым ответом.

Начинаем:

{running_convo}
"""

# --- LLM wrapper ---
class OpenAIChatLLM(BaseModel):
    """Wrapper around OpenAI chat completions via ProxyAPI."""
    model: str = 'gpt-4.1-mini'
    temperature: float = 0.0

    def generate(self, prompt: str, stop: List[str] = None) -> str:
        """Generate a response from the LLM given a prompt."""
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            stop=stop
        )
        return response.choices[0].message.content