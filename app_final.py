import streamlit as st
import os
import pandas as pd
from llama_index.core import VectorStoreIndex, PromptTemplate, Document
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from datetime import datetime
import time

# ====================================================================
# A. KONFIGURASI APLIKASI
# ====================================================================

API_KEY_ANDA = os.environ.get("GEMINI_API_KEY")
LLM_MODEL = "gemini-2.0-flash" # Menggunakan model terbaru
DATA_FILE = "Alkitab.csv"
USER_LOG_FILE = "user_log.csv"

st.set_page_config(
    page_title="Alpha & Omega Chat",
    page_icon="🦉",
    layout="centered" # Mengikuti desain kotak di tengah
)

# CSS Custom untuk Glassmorphism Design
st.markdown(
    """
    <style>
    /* Background Utama */
    .stApp {
        background: linear-gradient(135deg, #1e1e2f 0%, #2d3436 100%);
    }

    /* Container Utama */
    .main-card {
        background: rgba(40, 44, 52, 0.7);
        backdrop-filter: blur(15px);
        border-radius: 20px;
        padding: 30px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 15px 35px rgba(0,0,0,0.5);
    }

    /* Header Styling */
    .brand-title {
        color: #f1c40f;
        font-family: 'Arial Black', Gadget, sans-serif;
        font-size: 32px;
        text-align: center;
        letter-spacing: 2px;
        margin-bottom: 0px;
    }
    .brand-subtitle {
        color: #ffffff;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        margin-top: -10px;
        margin-bottom: 20px;
    }

    /* Chat Bubbles */
    .bot-bubble {
        background: rgba(255, 255, 255, 0.1);
        color: white;
        padding: 12px 18px;
        border-radius: 15px 15px 15px 2px;
        margin-bottom: 10px;
        max-width: 80%;
        border: 1px solid rgba(255,255,255,0.05);
    }
    .user-bubble {
        background: #f39c12;
        color: white;
        padding: 12px 18px;
        border-radius: 15px 15px 2px 15px;
        margin-bottom: 10px;
        margin-left: auto;
        max-width: 80%;
        text-align: right;
    }

    /* Sidebar Buttons (Simulasi) */
    .feature-btn {
        background: rgba(255,255,255,0.1);
        color: white;
        padding: 8px 15px;
        border-radius: 20px;
        margin-bottom: 10px;
        text-align: center;
        font-size: 14px;
        border: 1px solid rgba(255,255,255,0.2);
    }
    .daily-verse { background: #5dade2; color: white; }

    /* Input Box */
    .stChatInputContainer {
        padding-bottom: 20px;
    }
    
    /* Image Owl */
    .owl-img {
        display: block;
        margin-left: auto;
        margin-right: auto;
        width: 100px;
        filter: drop-shadow(0 0 10px #5dade2);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ====================================================================
# B. LOGIKA RAG
# ====================================================================

@st.cache_resource
def load_and_index_data():
    if not API_KEY_ANDA:
        return None, None
    try:
        df = pd.read_csv(DATA_FILE)
        df['text_bersih'] = df['Isi'].astype(str).str.replace('<t/>', '', regex=False)
        df['referensi'] = df['Nama ayat'].astype(str) + ' ' + df['Bagian'].astype(str) + ':' + df['Ayat'].astype(str)
        
        documents = [Document(text=row['text_bersih'], metadata={"ref": row['referensi']}) for _, row in df.iterrows()]
        llm = GoogleGenAI(model=LLM_MODEL, api_key=API_KEY_ANDA)
        embed_model = GoogleGenAIEmbedding(model='models/embedding-001', api_key=API_KEY_ANDA)
        index = VectorStoreIndex.from_documents(documents, llm=llm, embed_model=embed_model)
        return index, llm
    except:
        return None, None

INDEX, LLM = load_and_index_data()

def generate_response(query):
    if not INDEX: return "Sistem tidak siap."
    query_engine = INDEX.as_query_engine(llm=LLM)
    response = query_engine.query(query)
    return str(response)

# ====================================================================
# C. TAMPILAN UI UTAMA
# ====================================================================

# Layout Header
st.markdown('<div class="brand-title">ALPHA & OMEGA</div>', unsafe_allow_html=True)
st.markdown('<div class="brand-subtitle">CHAT</div>', unsafe_allow_html=True)

# Grid Layout (Kiri: Bot Info, Tengah: Chat, Kanan: Fitur)
col_left, col_mid, col_right = st.columns([1, 2.5, 1.2])

with col_left:
    st.image("https://cdn-icons-png.flaticon.com/512/3503/3503786.png", width=80) # Placeholder icon burung hantu
    st.markdown("<p style='text-align:center; color:white; font-weight:bold;'>PROPHET-GPT</p>", unsafe_allow_html=True)

with col_mid:
    # Inisialisasi chat
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hey there! Ready to explore some ancient wisdom? 😊"},
            {"role": "assistant", "content": "Ask me anything about the Bible - stories, verses, meanings! 🔥✨"}
        ]

    # Container chat dengan scroll
    for msg in st.session_state.messages:
        if msg["role"] == "assistant":
            st.markdown(f'<div class="bot-bubble">{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="user-bubble">{msg["content"]}</div>', unsafe_allow_html=True)

with col_right:
    # Tombol Fitur sesuai gambar
    if st.button("Daily Verse 🙏", use_container_width=True):
        st.toast("Menampilkan ayat hari ini...")
    
    st.markdown('<div class="feature-btn">🙏 Prayer Wall ✨</div>', unsafe_allow_html=True)
    st.markdown('<div class="feature-btn">Study Plans</div>', unsafe_allow_html=True)
    st.markdown('<div class="feature-btn">Youth Groups 👨‍👩‍👧‍👦</div>', unsafe_allow_html=True)

# Chat Input
if prompt := st.chat_input("Type your question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.spinner("Thinking..."):
        ans = generate_response(prompt)
        st.session_state.messages.append({"role": "assistant", "content": ans})
    
    st.rerun()

# Footer
st.markdown("<br><p style='text-align:center; color:gray; font-size:10px;'>Powered by Divine AI. Connect. Learn. Grow.</p>", unsafe_allow_html=True)s
