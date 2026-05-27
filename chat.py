import streamlit as st
import requests
import uuid

# --- Page config ---
st.set_page_config(
    page_title="Gonzo ML Bot",
    page_icon="🤖",
    layout="centered"
)

# --- Session state ---
if 'conversation_id' not in st.session_state:
    st.session_state['conversation_id'] = str(uuid.uuid4())
if 'history' not in st.session_state:
    st.session_state['history'] = []

# --- API call ---
def get_bot_response(user_input: str) -> str:
    """Send question to FastAPI backend and return answer."""
    payload = {
        "text": user_input,
        "conversation_id": st.session_state['conversation_id'],
        "temperature": 0.0,
        "threshold": 0.4
    }
    try:
        response = requests.post("http://localhost:8000/ask", json=payload)
        if response.status_code == 200:
            data = response.json()
            st.session_state['conversation_id'] = data.get('conversation_id')
            return data['response']
        else:
            return "Ошибка сервера. Попробуй ещё раз."
    except Exception as e:
        return f"Не удалось подключиться к серверу: {e}"


# --- UI ---
st.title("🤖 Gonzo_ML Assistant")
st.caption("Задавай вопросы по постам канала gonzo_ML")

# Chat history
for chat in st.session_state['history']:
    with st.chat_message("user"):
        st.write(chat['user'])
    with st.chat_message("assistant"):
        st.write(chat['bot'])

# Input
if prompt := st.chat_input("Спроси что-нибудь про ML..."):
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Ищу в базе..."):
            bot_response = get_bot_response(prompt)
        st.write(bot_response)

    st.session_state['history'].append({
        'user': prompt,
        'bot': bot_response
    })