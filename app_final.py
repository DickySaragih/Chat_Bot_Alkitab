import streamlit as st
import os
import pandas as pd
from datetime import datetime

# Menggunakan library LlamaIndex yang paling stabil untuk Gemini
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding

# ====================================================================
# 1. KONFIGURASI HALAMAN
# ====================================================================
# Pastikan API Key diatur di Streamlit Secrets atau Environment Variable
API_KEY = os.environ.get("GEMINI_API_KEY")

st.set_page_config(
    page_title="Bible is Journey",
    page_icon="📖",
    layout="wide"
)

# CSS Responsif & Detail Tampilan (Laptop & HP)
st.markdown("""
<style>
    .stApp {
        background: url("https://images.unsplash.com/photo-1507434965515-61970f2bd7c6?q=80&w=2070&auto=format&fit=crop") no-repeat center center fixed;
        background-size: cover;
    }
    header, footer, #MainMenu {visibility: hidden;}

    /* Kontainer Utama (Single Box) */
    .main-container {
        background: rgba(15, 15, 25, 0.85);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 25px;
        padding: 30px;
        max-width: 950px;
        margin: auto;
        box-shadow: 0 20px 50px rgba(0,0,0,0.6);
    }

    .app-title {
        color: #f1c40f;
        font-family: 'Georgia', serif;
        font-size: clamp(24px, 5vw, 42px);
        text-align: center;
        font-weight: bold;
        margin-bottom: 0px;
        text-shadow: 0 0 15px rgba(241, 196, 15, 0.4);
    }

    .bible-logo {
        text-align: center;
        font-size: 70px;
        margin: 10px 0;
    }

    /* Chat Bubble Styling */
    .bot-bubble {
        background: #f8f9fa;
        color: #2d3436;
        padding: 15px;
        border-radius: 20px 20px 20px 5px;
        margin: 10px 0;
        max-width: 85%;
        font-size: 14px;
        line-height: 1.6;
        box-shadow: 3px 3px 10px rgba(0,0,0,0.1);
    }
    .user-bubble {
        background: linear-gradient(135deg, #f1c40f, #e67e22);
        color: white;
        padding: 15px;
        border-radius: 20px 20px 5px 20px;
        margin: 10px 0 10px auto;
        max-width: 85%;
        text-align: right;
        font-size: 14px;
        box-shadow: -3px 3px 10px rgba(230, 126, 34, 0.3);
    }

    /* Responsivitas Mobile */
    @media (max-width: 768px) {
        .main-container { padding: 15px; border-radius: 0; }
        .stButton button { width: 100%; }
    }
</style>
""", unsafe_allow_html=True)

# ====================================================================
# 2. ENGINE ALKITAB (FIX ERROR HANDLING)
# ====================================================================

@st.cache_resource(show_spinner=False)
def initialize_bible_engine():
    if not API_KEY:
        st.error("API Key Gemini tidak ditemukan!")
        return None, None
    try:
        # Load Data CSV Anda
        df = pd.read_csv("Alkitab.csv")
        
        # Pembersihan data sesuai kolom Anda
        df['content'] = df['Isi'].astype(str).str.replace('<t/>', '', regex=False)
        df['reference'] = df['Nama ayat'].astype(str) + " " + \
                          df['Bagian'].astype(str) + ":" + \
                          df['Ayat'].astype(str)
        
        documents = [
            Document(text=row['content'], metadata={"ref": row['reference']}) 
            for _, row in df.iterrows()
        ]

        # Konfigurasi LLM (Menggunakan library Gemini yang lebih stabil)
        Settings.llm = Gemini(
            model="models/gemini-1.5-flash", 
            api_key=API_KEY,
            temperature=0.3
        )
        Settings.embed_model = GeminiEmbedding(
            model_name="models/embedding-001", 
            api_key=API_KEY
        )
        
        index = VectorStoreIndex.from_documents(documents)
        return index, df
    except Exception as e:
        st.error(f"Gagal memuat sistem: {e}")
        return None, None

INDEX, BIBLE_DF = initialize_bible_engine()

def get_bible_response(user_input):
    if not INDEX:
        return "Sistem sedang mengalami gangguan teknis."
    try:
        # PENTING: response_mode="compact" untuk menghindari ClientError API
        query_engine = INDEX.as_query_engine(
            response_mode="compact",
            similarity_top_k=3
        )
        prompt = f"Sebagai asisten Alkitab yang bijak, jawablah pertanyaan ini berdasarkan data Alkitab yang ada: {user_input}. Berikan referensi ayatnya."
        response = query_engine.query(prompt)
        return str(response)
    except Exception as e:
        return f"Maaf, terjadi kendala saat memproses permintaan. Mari coba tanyakan hal lain. 🙏"

# ====================================================================
# 3. ANTARMUKA CHAT (SINGLE BOX)
# ====================================================================

st.markdown('<div class="main-container">', unsafe_allow_html=True)
st.markdown('<div class="app-title">BIBLE IS JOURNEY</div>', unsafe_allow_html=True)
st.markdown('<div class="bible-logo">📖</div>', unsafe_allow_html=True)

# State untuk History Chat
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "ai", "content": "Halo! Saya teman perjalananmu. Apa yang ingin kamu pelajari dari Alkitab hari ini?"}
    ]

# Layout Chat (Menggunakan Container agar scrollable)
chat_box = st.container(height=400, border=False)
with chat_box:
    for chat in st.session_state.chat_history:
        if chat["role"] == "ai":
            st.markdown(f'<div class="bot-bubble">{chat["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="user-bubble">{chat["content"]}</div>', unsafe_allow_html=True)

# Input Chat
if user_prompt := st.chat_input("Ketik pesan atau pertanyaan Alkitab..."):
    st.session_state.chat_history.append({"role": "user", "content": user_prompt})
    
    with st.spinner("Mencari hikmat..."):
        ai_ans = get_bible_response(user_prompt)
        st.session_state.chat_history.append({"role": "ai", "content": ai_ans})
    
    st.rerun()

# Tombol Menu (Bottom)
st.markdown("<br>", unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
with c1:
    if st.button("Ayat Acak 🌟", use_container_width=True):
        if BIBLE_DF is not None:
            random_row = BIBLE_DF.sample(1).iloc[0]
            verse_text = f"🌟 **{random_row['reference']}**\n\n{random_row['content']}"
            st.session_state.chat_history.append({"role": "ai", "content": verse_text})
            st.rerun()
with c2:
    st.button("Rencana Baca 📚", use_container_width=True)
with c3:
    if st.button("Hapus Chat 🗑️", use_container_width=True):
        st.session_state.chat_history = st.session_state.chat_history[:1]
        st.rerun()

st.markdown('</div>', unsafe_allow_html=True)
