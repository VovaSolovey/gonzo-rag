import json
import re
from datetime import datetime


# --- Constants ---
SECTION_HEADERS_NORMALIZED = {'ЧТО СДЕЛАЛИ', 'ПОЧЕМУ ЭТО ВАЖНО', 'ДЛЯ ПРАКТИКОВ'}


# --- Helpers ---
def normalize_header(text: str) -> str:
    """Normalize section header for comparison."""
    return text.strip().rstrip('?:').strip().upper()


# --- Parser ---
def extract_text_and_meta(message: dict, channel_username: str) -> dict | None:
    """
    Parse a single Telegram message and extract text + metadata.

    Three post types:
      Type 1 - TL;DR review: bold title + italic authors + arxiv + sections
      Type 2 - plain review: bold title + italic authors + arxiv + plain text
      Type 3 - author post: plain text, no title, no arxiv
    """
    if message.get('type') != 'message':
        return None

    raw_text = message.get('text', '')

    # Use text_entities — plain text is already marked up there
    text_entities = message.get('text_entities', [])
    if text_entities:
        entities = text_entities
    elif isinstance(raw_text, str):
        entities = [{'type': 'plain', 'text': raw_text}]
    else:
        return None

    # Metadata
    title = None
    authors = None
    arxiv_url = None
    arxiv_id = None
    has_tldr = False
    extra_links = {}
    tokens = []

    i = 0
    while i < len(entities):
        entity = entities[i]
        etype = entity.get('type', 'plain')
        etext = entity.get('text', '')

        # Title — first bold, not TL;DR and not a section header
        if etype == 'bold' and title is None and '# TL;DR' not in etext:
            stripped = etext.replace('\n', ' ').strip()
            if normalize_header(stripped) not in SECTION_HEADERS_NORMALIZED:
                title = stripped
                i += 1
                continue

        # Authors — first italic after title
        if etype == 'italic' and authors is None:
            if title is not None:
                stripped = etext.replace('\n', ' ').strip()
                words = stripped.split()
                is_authors = ',' in stripped or (len(words) >= 2 and sum(1 for w in words if w[0].isupper()) >= 2)
                if is_authors:
                    authors = stripped
                else:
                    tokens.append(('p', stripped))
            else:
                if etext.strip():
                    tokens.append(('p', etext.replace('\n', ' ').strip()))
            i += 1
            continue

        # TL;DR flag
        if '# TL;DR' in etext:
            has_tldr = True
            i += 1
            continue

        # Links — save to metadata, remove from text
        if etype == 'link':
            url = etext.strip()
            if 'arxiv.org' in url and arxiv_url is None:
                arxiv_url = url
                match = re.search(r'arxiv\.org/abs/([\d\.]+)', url)
                if match:
                    arxiv_id = match.group(1)
            elif 'github.com' in url:
                extra_links['code_url'] = url
            elif 'huggingface.co' in url:
                extra_links['model_url'] = url
            elif 'substack.com' in url or 'arxiviq' in url:
                extra_links['review_url'] = url
            i += 1
            continue

        # Filter label prefixes before links
        if re.match(r'^(Статья|Paper|Code|Код|Model|Review|Сайт|Ревью|Обзор|Подробнее|Обозревать здесь|Диффундировать здесь|Координироваться тут)\s*:?\s*$', etext.strip()):
            i += 1
            continue

        # TL;DR sections: merge bold header with next plain text
        if etype == 'bold' and normalize_header(etext) in SECTION_HEADERS_NORMALIZED:
            header = normalize_header(etext).capitalize() + ':'
            j = i + 1
            while j < len(entities) and not entities[j].get('text', '').strip():
                j += 1
            if j < len(entities):
                next_e = entities[j]
                next_text = next_e.get('text', '').strip()
                if next_e.get('type', 'plain') == 'plain' and next_text:
                    next_clean = next_text.replace('\n\n', ' ').replace('\n', ' ').strip()
                    tokens.append(('p', f"{header} {next_clean}"))
                    i = j + 1
                    continue
            tokens.append(('p', header))
            i += 1
            continue

        # Plain text — preserve \n\n as paragraph boundaries
        if etext.strip():
            paragraphs = etext.split('\n\n')
            for para in paragraphs:
                clean = para.replace('\n', ' ').strip()
                if clean:
                    tokens.append(('p', clean))

        i += 1

    clean_text = '\n\n'.join(text for _, text in tokens if text)
    if not clean_text.strip():
        return None

    # Add title to beginning of text (Kovalsky trick)
    if title:
        clean_text = f"{title}\n\n{clean_text}"

    # Timestamp
    date_str = message.get('date', '')
    try:
        dt = datetime.fromisoformat(date_str)
        timestamp = int(dt.timestamp())
    except Exception:
        timestamp = 0

    post_id = message.get('id')
    post_url = f"https://t.me/{channel_username}/{post_id}"

    return {
        'text': clean_text,
        'title': title,
        'authors': authors,
        'arxiv_url': arxiv_url,
        'arxiv_id': arxiv_id,
        'post_url': post_url,
        'date': timestamp,
        'post_id': post_id,
        'has_tldr': has_tldr,
        **extra_links
    }


def parse_telegram_json(path: str, channel_username: str) -> list[dict]:
    """
    Parse Telegram JSON export and return list of chunks.

    Args:
        path: path to Telegram JSON export
        channel_username: channel username e.g. 'gonzo_ML'
    """
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    messages = data.get('messages', [])
    chunks = []
    skipped = 0

    for msg in messages:
        result = extract_text_and_meta(msg, channel_username)
        if result is None:
            skipped += 1
            continue
        chunks.append(result)

    print(f"Total messages : {len(messages)}")
    print(f"Skipped        : {skipped}")
    print(f"Chunks ready   : {len(chunks)}")
    return chunks