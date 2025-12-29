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

# Set API Key Anda di sini
# os.environ["GEMINI_API_KEY"] = "MASUKKAN_KEY_ANDA"
API_KEY_ANDA = os.environ.get("GEMINI_API_KEY")

st.set_page_config(
    page_title="Alpha & Omega Chat",
    page_icon="🦉",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# File Paths
DATA_FILE = "Alkitab.csv"
USER_LOG_FILE = "user_log.csv"

# ====================================================================
# 2. CSS "SINGLE BOX" & GLASSMORPHISM (Sangat Detail)
# ====================================================================

st.markdown("""
<style>
    /* 1. RESET & BACKGROUND UTAMA */
    .stApp {
        background: url("https://images.unsplash.com/photo-1534796636912-3b95b3ab5986?q=80&w=2072&auto=format&fit=crop") no-repeat center center fixed;
        background-size: cover;
        font-family: 'Helvetica Neue', sans-serif;
    }
    
    /* Hilangkan Header & Footer bawaan Streamlit */
    header {visibility: hidden;}
    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}

    /* 2. THE MAIN BOX (KARTU UTAMA) */
    /* Kita membatasi lebar konten utama dan memberinya style kartu kaca */
    .block-container {
        max-width: 1000px;
        padding-top: 2rem;
        padding-bottom: 2rem;
        background: rgba(26, 28, 48, 0.85); /* Warna dasar gelap transparan */
        backdrop-filter: blur(20px);         /* Efek Blur Kaca */
        -webkit-backdrop-filter: blur(20px);
        border-radius: 25px;
        border: 1px solid rgba(255, 255, 255, 0.2); /* Garis tepi tipis */
        box-shadow: 0 20px 50px rgba(0,0,0,0.5);    /* Bayangan 3D */
        margin-top: 5vh;
        min-height: 80vh; /* Tinggi minimum agar terlihat kotak */
        position: relative;
    }

    /* 3. HEADER STYLING (Emas & Putih) */
    .header-text {
        font-size: 36px;
        font-weight: 800;
        color: #f1c40f; /* Warna Emas */
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 0;
        line-height: 1;
    }
    .header-sub {
        font-size: 24px;
        font-weight: 700;
        color: #f1c40f;
        margin-top: 0;
        letter-spacing: 4px;
        margin-bottom: 20px;
    }
    .top-icons {
        color: #f1c40f;
        font-size: 24px;
        text-align: right;
    }

    /* 4. AVATAR GLOW (Burung Hantu) */
    .owl-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        height: 100%;
        margin-top: 40px;
    }
    .neon-glow-img {
        width: 140px;
        filter: drop-shadow(0 0 10px #6faeff) drop-shadow(0 0 20px #6faeff); /* Efek Neon Biru */
        margin-bottom: 10px;
    }
    .prophet-label {
        color: white;
        font-weight: bold;
        letter-spacing: 1px;
        font-size: 16px;
    }

    /* 5. CHAT BUBBLES (Detail) */
    .chat-area {
        height: 50vh;
        overflow-y: auto;
        padding: 10px;
        scrollbar-width: thin;
        scrollbar-color: rgba(255,255,255,0.3) transparent;
    }
    
    /* Bubble Bot (Putih) */
    .bubble-bot {
        background-color: #f5f6fa;
        color: #2f3640;
        padding: 12px 18px;
        border-radius: 18px 18px 18px 4px;
        margin-bottom: 15px;
        max-width: 90%;
        font-size: 14px;
        line-height: 1.5;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        position: relative;
    }
    .bubble-bot::before { /* Ekor bubble kiri */
        content: "";
        position: absolute;
        bottom: 0; left: -8px;
        width: 15px; height: 15px;
        background: #f5f6fa;
        border-radius: 50%;
        z-index: -1;
    }

    /* Bubble User (Orange/Gold) */
    .bubble-user {
        background: linear-gradient(135deg, #f1c40f 0%, #e67e22 100%);
        color: #fff;
        padding: 12px 18px;
        border-radius: 18px 18px 4px 18px;
        margin-bottom: 15px;
        margin-left: auto; /* Geser ke kanan */
        max-width: 90%;
        font-size: 14px;
        text-align: right;
        box-shadow: 0 4px 10px rgba(230, 126, 34, 0.4);
    }

    /* 6. TOMBOL MENU KANAN (Pills) */
    .stButton button {
        width: 100%;
        border-radius: 50px !important;
        border: none !important;
        padding: 12px 20px !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        margin-bottom: 10px !important;
        transition: transform 0.2s;
    }
    /* Tombol Biru (Daily Verse) */
    div[data-testid="stVerticalBlock"] > div:nth-child(1) > div > div > div > button {
        background: linear-gradient(90deg, #3498db, #2980b9);
        color: white;
        box-shadow: 0 5px 15px rgba(52, 152, 219, 0.4);
    }
    /* Tombol Hitam Transparan (Lainnya) - Kita akali dengan CSS targeting */
    .btn-dark {
        background: rgba(0,0,0,0.5) !important;
        color: white !important;
        border: 1px solid rgba(255,255,255,0.2) !important;
    }

    /* 7. INPUT FIELD (Customizing Streamlit Chat Input) */
    .stChatInput {
        padding-bottom: 2rem;
        padding-top: 1rem;
    }
    
    /* Ubah kotak input menjadi putih rounded (Pill Shape) */
    .stChatInputContainer > div {
        background-color: white !important;
        border-radius: 50px !important;
        color: black !important;
        border: none !important;
        box-shadow: 0 0 15px rgba(255,255,255,0.1) !important;
    }
    
    /* Text input color */
    .stChatInputContainer textarea {
        color: #333 !important;
        font-size: 15px !important;
    }
    
    /* Icon kirim */
    .stChatInputContainer button {
        color: #2c3e50 !important;
    }

    /* Footer Text */
    .footer-divine {
        text-align: center;
        color: #bdc3c7;
        font-size: 12px;
        margin-top: -15px;
        margin-bottom: 10px;
    }

</style>
""", unsafe_allow_html=True)

# ====================================================================
# 3. BACKEND LOGIC (RAG & DATA)
# ====================================================================

@st.cache_resource
def load_data():
    if not API_KEY_ANDA: return None, None
    try:
        df = pd.read_csv(DATA_FILE)
        # Bersihkan data
        if 'Isi' in df.columns:
            df['text_bersih'] = df['Isi'].astype(str).str.replace('<t/>', '', regex=False)
        else:
            df['text_bersih'] = df['text'].astype(str)
        
        df['referensi'] = df['Nama ayat'].astype(str) + ' ' + df['Bagian'].astype(str) + ':' + df['Ayat'].astype(str)
        
        documents = [Document(text=row['text_bersih'], metadata={"ref": row['referensi']}) for _, row in df.iterrows()]
        
        Settings.llm = GoogleGenAI(model="models/gemini-2.0-flash-exp", api_key=API_KEY_ANDA)
        Settings.embed_model = GoogleGenAIEmbedding(model="models/embedding-001", api_key=API_KEY_ANDA)
        
        index = VectorStoreIndex.from_documents(documents)
        return index, df
    except:
        return None, None

INDEX, DF_ALKITAB = load_data()

def get_answer(query):
    if not INDEX: return "Maaf, sistem sedang offline (Cek API Key)."
    qe = INDEX.as_query_engine()
    res = qe.query(f"Jawab pertanyaan ini dengan bijak, singkat, dan sertakan ayat: {query}")
    return str(res)

def get_verse():
    if DF_ALKITAB is not None:
        row = DF_ALKITAB.sample(1).iloc[0]
        return f"📖 **{row['referensi']}**\n\n_{row['text_bersih']}_"
    return "Data Alkitab belum siap."

# ====================================================================
# 4. TAMPILAN UI (SINGLE BOX LAYOUT)
# ====================================================================

# -- HEADER BAGIAN ATAS --
c_head1, c_head2 = st.columns([3, 1])
with c_head1:
    st.markdown('<p class="header-text">ALPHA & OMEGA 📡</p>', unsafe_allow_html=True)
    st.markdown('<p class="header-sub">CHAT</p>', unsafe_allow_html=True)
with c_head2:
    # Ikon buku dan menu hamburger (dummy visual)
    st.markdown('<div class="top-icons">📖 &nbsp; ☰</div>', unsafe_allow_html=True)

st.markdown("---") # Garis pemisah tipis

# -- KONTEN UTAMA (GRID 3 KOLOM) --
col_left, col_mid, col_right = st.columns([1.3, 3, 1.3])

# 1. KOLOM KIRI: AVATAR
with col_left:
    st.markdown('<div class="owl-container">', unsafe_allow_html=True)
    # Gambar Burung Hantu Neon (Link luar untuk demo)
    st.markdown('<img src="https://cdn-icons-png.flaticon.com/512/3468/3468306.png" class="neon-glow-img">', unsafe_allow_html=True)
    st.markdown('<div class="prophet-label">PROPHET-GPT</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# 2. KOLOM TENGAH: CHAT
with col_mid:
    # Container Chat dengan tinggi tetap (Scrollable)
    chat_container = st.container(height=350)
    
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "ai", "content": "Hey there! Ready to explore some ancient wisdom? 😯✨"},
            {"role": "ai", "content": "Ask me anything about the Bible – stories, verses, meanings! 🔥✨"}
        ]
    
    with chat_container:
        for msg in st.session_state.messages:
            if msg['role'] == 'ai':
                st.markdown(f'<div class="bubble-bot">{msg["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="bubble-user">{msg["content"]}</div>', unsafe_allow_html=True)

# 3. KOLOM KANAN: MENU BUTTONS
with col_right:
    st.markdown("<br>", unsafe_allow_html=True) # Spacer
    
    # Tombol Daily Verse (Warna Biru karena urutan pertama CSS)
    if st.button("Daily Verse 🙏", use_container_width=True):
        v = get_verse()
        st.session_state.messages.append({"role": "ai", "content": v})
        st.rerun()

    # Tombol Lainnya (Menggunakan style hack untuk warna gelap)
    # Catatan: Di Streamlit murni sulit memberi kelas custom per tombol, 
    # jadi kita andalkan CSS global atau bungkus dalam container.
    
    st.button("Prayer Wall ✨", use_container_width=True)
    st.button("Study Plans", use_container_width=True)
    st.button("Youth Groups 👼", use_container_width=True)

# -- FOOTER & INPUT --
# Input ditaruh di bawah kolom agar full width di dalam box
st.markdown("<br>", unsafe_allow_html=True)

# Input Field
user_input = st.chat_input("Type your question...", key="main_input")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.spinner("..."):
        ans = get_answer(user_input)
        st.session_state.messages.append({"role": "ai", "content": ans})
    st.rerun()

# Text Powered By
st.markdown('<div class="footer-divine">Powered by Divine AI. Connect. Learn. Grow.</div>', unsafe_allow_html=True)
