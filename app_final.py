import streamlit as st
import os
import pandas as pd
import random
from datetime import datetime
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding

# ====================================================================
# 1. KONFIGURASI HALAMAN & API
# ====================================================================

API_KEY_ANDA = os.environ.get("GEMINI_API_KEY")

st.set_page_config(
    page_title="Bible is Journey",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

DATA_FILE = "Alkitab.csv"

# ====================================================================
# 2. CSS RESPONSIF & UI BIBLE IS JOURNEY
# ====================================================================

st.markdown("""
<style>
    /* KONFIGURASI DASAR */
    .stApp {
        background: linear-gradient(135deg, #1e1e2f 0%, #2d3436 100%);
        color: white;
    }
    
    header, footer, #MainMenu {visibility: hidden;}

    /* CONTAINER UTAMA YANG RESPONSIF */
    .block-container {
        max-width: 1100px;
        padding: 1rem;
        margin: auto;
    }

    /* KARTU APLIKASI (GLASSMORPHISM) */
    .main-box {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(15px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
    }

    /* JUDUL BARU: BIBLE IS JOURNEY */
    .app-title {
        font-family: 'serif';
        font-size: clamp(24px, 5vw, 40px); /* Ukuran teks fleksibel */
        font-weight: 800;
        color: #f1c40f;
        text-align: center;
        margin-bottom: 5px;
        letter-spacing: 2px;
    }

    /* LOGO BIBLE */
    .bible-logo-container {
        text-align: center;
        margin: 20px 0;
    }
    .bible-icon {
        font-size: 80px;
        filter: drop-shadow(0 0 15px #f1c40f);
    }

    /* CHAT AREA RESPONSIF */
    .chat-container {
        height: 400px;
        overflow-y: auto;
        padding: 10px;
        background: rgba(0,0,0,0.2);
        border-radius: 15px;
        margin-bottom: 20px;
    }

    .bubble-bot {
        background: #ffffff;
        color: #333;
        padding: 12px;
        border-radius: 15px 15px 15px 2px;
        margin-bottom: 10px;
        width: fit-content;
        max-width: 85%;
        font-size: 14px;
    }

    .bubble-user {
        background: #f1c40f;
        color: #000;
        padding: 12px;
        border-radius: 15px 15px 2px 15px;
        margin-bottom: 10px;
        margin-left: auto;
        width: fit-content;
        max-width: 85%;
        text-align: right;
        font-size: 14px;
    }

    /* MEDIA QUERIES UNTUK HANDPHONE */
    @media (max-width: 768px) {
        .app-title { font-size: 24px; }
        .bible-icon { font-size: 60px; }
        .stButton button { font-size: 12px !important; padding: 10px !important; }
    }
</style>
""", unsafe_allow_html=True)

# ====================================================================
# 3. BACKEND & PERBAIKAN ERROR HANDLING
# ====================================================================

@st.cache_resource
def load_rag_system():
    if not API_KEY_ANDA:
        return None, None
    try:
        df = pd.read_csv(DATA_FILE)
        df['text_bersih'] = df['Isi'].astype(str).str.replace('<t/>', '', regex=False)
        df['referensi'] = df['Nama ayat'].astype(str) + ' ' + df['Bagian'].astype(str) + ':' + df['Ayat'].astype(str)
        
        documents = [Document(text=row['text_bersih'], metadata={"ref": row['referensi']}) for _, row in df.iterrows()]
        
        # Menggunakan model flash terbaru untuk kecepatan
        Settings.llm = GoogleGenAI(model="models/gemini-1.5-flash", api_key=API_KEY_ANDA)
        Settings.embed_model = GoogleGenAIEmbedding(model="models/embedding-001", api_key=API_KEY_ANDA)
        
        index = VectorStoreIndex.from_documents(documents)
        return index, df
    except Exception as e:
        st.error(f"Gagal memuat data: {e}")
        return None, None

INDEX, DF_ALKITAB = load_rag_system()

def get_answer(query):
    # PROTEKSI: Cek apakah INDEX tersedia sebelum query
    if INDEX is None:
        return "Sistem belum siap. Pastikan API Key benar dan file Alkitab.csv sudah diunggah."
    
    try:
        query_engine = INDEX.as_query_engine()
        response = query_engine.query(f"Jawab pertanyaan Alkitab ini dengan ramah dan sertakan ayat: {query}")
        return str(response)
    except Exception as e:
        # Menangkap error seperti yang Anda alami dan memberikan pesan ramah
        return f"Mohon maaf, terjadi gangguan teknis saat mencari jawaban. (Error: {str(e)[:50]}...)"

# ====================================================================
# 4. TAMPILAN UI (SINGLE BOX)
# ====================================================================

st.markdown('<div class="main-box">', unsafe_allow_html=True)

# Header & Logo
st.markdown('<p class="app-title">BIBLE IS JOURNEY</p>', unsafe_allow_html=True)
st.markdown('<div class="bible-logo-container"><span class="bible-icon">📖</span></div>', unsafe_allow_html=True)

# Layout Kolom (Otomatis menyusun vertikal di HP)
col_chat, col_menu = st.columns([3, 1])

with col_chat:
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "ai", "content": "Selamat datang di perjalanan imanmu. Ada yang bisa saya bantu hari ini? ✨"}]
    
    # Chat Area
    chat_placeholder = st.container(height=350)
    with chat_placeholder:
        for m in st.session_state.messages:
            div_class = "bubble-bot" if m["role"] == "ai" else "bubble-user"
            st.markdown(f'<div class="{div_class}">{m["content"]}</div>', unsafe_allow_html=True)

with col_menu:
    st.write("### Menu")
    if st.button("Daily Verse 🙏", use_container_width=True):
        if DF_ALKITAB is not None:
            row = DF_ALKITAB.sample(1).iloc[0]
            verse = f"📖 **{row['referensi']}**\n\n{row['text_bersih']}"
            st.session_state.messages.append({"role": "ai", "content": verse})
            st.rerun()
    
    st.button("Prayer Wall ✨", use_container_width=True)
    st.button("Study Plan 📚", use_container_width=True)

# Chat Input
if prompt := st.chat_input("Ketik perjalanan imanmu..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.spinner("Mencari hikmat..."):
        response = get_answer(prompt)
        st.session_state.messages.append({"role": "ai", "content": response})
    st.rerun()

st.markdown('<p style="text-align:center; font-size:10px; color:gray; margin-top:10px;">Bible is Journey • Powered by Divine AI</p>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
