import streamlit as st
import os
import pandas as pd
import random
from datetime import datetime
import time

# Import LlamaIndex & Gemini
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding

# ====================================================================
# 1. KONFIGURASI HALAMAN & API
# ====================================================================

# Pastikan API Key ada di environment variable atau set langsung di sini (tidak disarankan untuk production)
# os.environ["GEMINI_API_KEY"] = "MASUKKAN_KEY_ANDA_DISINI" 
API_KEY_ANDA = os.environ.get("GEMINI_API_KEY")

st.set_page_config(
    page_title="Alpha & Omega Chat",
    page_icon="🕊️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# File Paths
DATA_FILE = "Alkitab.csv"
USER_LOG_FILE = "user_log.csv"

# ====================================================================
# 2. CSS MODERN (GLASSMORPHISM & NEON) - MENIRU GAMBAR REFERENSI
# ====================================================================

st.markdown("""
<style>
    /* IMPORT FONT KEREN */
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;700&family=Poppins:wght@300;600&display=swap');

    /* BACKGROUND UTAMA (Sunset/Cosmic Gradient) */
    .stApp {
        background: linear-gradient(135deg, #1c1c3c 0%, #2e2e4d 50%, #4a3b52 100%);
        font-family: 'Poppins', sans-serif;
        color: white;
    }

    /* HEADER TEXT */
    .header-title {
        font-family: 'Montserrat', sans-serif;
        font-weight: 800;
        font-size: 42px;
        background: linear-gradient(to right, #f1c40f, #f39c12);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0px 0px 20px rgba(241, 196, 15, 0.3);
        margin-bottom: -10px;
    }
    
    .header-subtitle {
        font-family: 'Montserrat', sans-serif;
        font-weight: 700;
        font-size: 32px;
        color: #f1c40f;
        letter-spacing: 2px;
        margin-bottom: 20px;
    }

    /* GLASS CONTAINER (PANEL UTAMA) */
    .glass-panel {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-radius: 25px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        margin-top: 10px;
        height: 75vh; /* Tinggi fix agar scrollable */
        position: relative;
    }

    /* KOLOM KIRI: AVATAR */
    .avatar-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        height: 100%;
        padding-top: 50px;
    }
    .neon-owl {
        width: 150px;
        filter: drop-shadow(0 0 15px #00a8ff);
        animation: float 6s ease-in-out infinite;
    }
    @keyframes float {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
        100% { transform: translateY(0px); }
    }
    .prophet-text {
        margin-top: 15px;
        font-weight: 600;
        font-size: 18px;
        letter-spacing: 1px;
        color: #e0e0e0;
        text-shadow: 0 0 10px rgba(255,255,255,0.3);
    }

    /* KOLOM TENGAH: CHAT BUBBLES */
    .chat-scroll-area {
        height: 65vh;
        overflow-y: auto;
        padding-right: 10px;
        padding-left: 10px;
        scrollbar-width: thin;
    }
    
    /* Bot Message (Putih/Abu) */
    .bot-message {
        background-color: #f0f2f5;
        color: #2c3e50;
        padding: 15px 20px;
        border-radius: 20px 20px 20px 5px;
        margin-bottom: 15px;
        max-width: 85%;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        font-size: 15px;
        line-height: 1.5;
        position: relative;
    }
    .bot-message::before {
        content: '';
        position: absolute;
        bottom: 0; left: -10px;
        width: 20px; height: 20px;
        background: radial-gradient(circle at top right, transparent 50%, #f0f2f5 50%);
    }

    /* User Message (Orange/Emas) */
    .user-message {
        background: linear-gradient(135deg, #f39c12 0%, #d35400 100%);
        color: white;
        padding: 15px 20px;
        border-radius: 20px 20px 5px 20px;
        margin-bottom: 15px;
        max-width: 85%;
        margin-left: auto; /* Geser ke kanan */
        box-shadow: 0 4px 15px rgba(243, 156, 18, 0.4);
        font-size: 15px;
        line-height: 1.5;
        text-align: right;
    }

    /* KOLOM KANAN: TOMBOL MENU */
    .menu-container {
        display: flex;
        flex-direction: column;
        gap: 15px;
        padding-top: 50px;
    }
    
    /* Tombol Custom (Div simulated as button for visual, actual functionality via st.button logic) */
    div.stButton > button {
        width: 100%;
        border-radius: 30px;
        padding: 12px;
        font-weight: 600;
        border: none;
        transition: all 0.3s ease;
        text-transform: none; 
        font-size: 14px;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 10px;
    }

    /* Daily Verse Button (Blue) */
    div.stButton > button:first-child {
        background: linear-gradient(90deg, #3498db, #2980b9) !important;
        color: white !important;
        box-shadow: 0 4px 15px rgba(52, 152, 219, 0.4);
    }
    
    /* Other Buttons (Dark Glass) */
    div.stButton > button {
        background: rgba(0, 0, 0, 0.4);
        color: #e0e0e0;
        border: 1px solid rgba(255,255,255,0.1);
    }
    div.stButton > button:hover {
        transform: translateY(-3px);
        background: rgba(255, 255, 255, 0.1);
        color: white;
    }

    /* INPUT FIELD Customization */
    .stChatInput {
        position: fixed;
        bottom: 30px;
        width: 50%;
        left: 25%;
        z-index: 999;
    }
    .stChatInput > div {
        background-color: white !important;
        border-radius: 30px !important;
        color: black !important;
        box-shadow: 0 0 20px rgba(255, 255, 255, 0.2);
    }
    .stChatInput textarea {
        color: black !important;
    }
    
    /* Footer Styling */
    .footer-text {
        text-align: center;
        font-size: 12px;
        color: rgba(255,255,255,0.5);
        margin-top: 50px;
    }
    
    /* Hiding Standard Elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

</style>
""", unsafe_allow_html=True)

# ====================================================================
# 3. BACKEND LOGIC (RAG & DATA)
# ====================================================================

@st.cache_resource(show_spinner="Menghubungkan ke Hikmat Ilahi...")
def initialize_engine():
    """Memuat data dan inisialisasi Index LlamaIndex."""
    if not API_KEY_ANDA:
        return None

    try:
        # Load Data
        df = pd.read_csv(DATA_FILE)
        
        # Bersihkan data jika kolom ada
        if 'Isi' in df.columns:
            df['text_bersih'] = df['Isi'].astype(str).str.replace('<t/>', '', regex=False)
        else:
            df['text_bersih'] = df['text'].astype(str) # Fallback

        df['referensi'] = df['Nama ayat'].astype(str) + ' ' + df['Bagian'].astype(str) + ':' + df['Ayat'].astype(str)
        
        documents = [
            Document(
                text=row['text_bersih'],
                metadata={"referensi": row['referensi']}
            )
            for _, row in df.iterrows()
        ]

        # Setup Model
        Settings.llm = GoogleGenAI(model="models/gemini-2.0-flash-exp", api_key=API_KEY_ANDA, temperature=0.7)
        Settings.embed_model = GoogleGenAIEmbedding(model="models/embedding-001", api_key=API_KEY_ANDA)

        # Buat Index
        index = VectorStoreIndex.from_documents(documents)
        return index, df # Return df juga untuk fitur Daily Verse
    except Exception as e:
        st.error(f"Error System: {e}")
        return None, None

INDEX, DF_ALKITAB = initialize_engine()

def get_chat_response(query_text):
    if not INDEX:
        return "Maaf, sistem sedang offline. Periksa API Key Anda."
    
    # Custom Prompt agar personality cocok dengan "Prophet-GPT"
    prompt = f"""
    Anda adalah 'Prophet-GPT', asisten Alkitab yang bijaksana, hangat, dan modern.
    Jawablah pertanyaan ini: "{query_text}"
    
    Panduan:
    1. Gunakan bahasa yang santai namun hormat (seperti mentor pemuda).
    2. Sertakan referensi ayat Alkitab yang relevan jika ada.
    3. Jika pengguna sedih, berikan penghiburan.
    4. Jaga jawaban tetap ringkas (maksimal 3 paragraf pendek) agar enak dibaca di chat bubble.
    """
    
    query_engine = INDEX.as_query_engine()
    response = query_engine.query(prompt)
    return str(response)

def get_daily_verse():
    """Mengambil ayat acak dari DataFrame."""
    if DF_ALKITAB is not None and not DF_ALKITAB.empty:
        random_row = DF_ALKITAB.sample(1).iloc[0]
        ref = random_row['referensi']
        isi = random_row['text_bersih']
        return f"🌟 **Ayat Hari Ini ({ref})**\n\n_{isi}_"
    return "Data Alkitab belum dimuat."

def log_user_visit():
    """Mencatat sesi pengguna (sederhana tanpa login untuk tampilan cepat)."""
    if "user_logged" not in st.session_state:
        st.session_state.user_logged = True
        if not os.path.exists(USER_LOG_FILE):
            pd.DataFrame(columns=["Timestamp", "Activity"]).to_csv(USER_LOG_FILE, index=False)
        
        new_log = pd.DataFrame({"Timestamp": [datetime.now()], "Activity": ["New Session"]})
        new_log.to_csv(USER_LOG_FILE, mode='a', header=False, index=False)

# ====================================================================
# 4. LAYOUT UI & INTERAKSI
# ====================================================================

# Panggil Log
log_user_visit()

# HEADER
c1, c2, c3 = st.columns([1, 6, 1])
with c2:
    st.markdown('<div style="text-align: center;">', unsafe_allow_html=True)
    st.markdown('<div class="header-title">ALPHA & OMEGA</div>', unsafe_allow_html=True)
    st.markdown('<div class="header-subtitle">CHAT</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# MAIN CONTENT AREA (Grid 3 Kolom)
# Rasio: 1.5 (Kiri) - 4 (Chat) - 1.5 (Kanan)
col_left, col_center, col_right = st.columns([1.5, 4, 1.5])

# --- KOLOM KIRI: PROPHET AVATAR ---
with col_left:
    st.markdown('<div class="glass-panel avatar-container">', unsafe_allow_html=True)
    # Menggunakan placeholder gambar burung hantu neon
    st.markdown('<img src="https://cdn-icons-png.flaticon.com/512/4710/4710926.png" class="neon-owl">', unsafe_allow_html=True) 
    st.markdown('<div class="prophet-text">PROPHET-GPT</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# --- KOLOM TENGAH: CHAT AREA ---
with col_center:
    # Container Chat
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    
    # Inisialisasi History
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hai! Siap menjelajahi hikmat kuno hari ini? 😌✨"},
            {"role": "assistant", "content": "Tanya saya apa saja tentang Alkitab — cerita, ayat, atau maknanya! 🔥🙏"}
        ]
    
    # Render Chat History (Scroll Area)
    chat_container = st.container(height=500) # Streamlit native scroll container
    with chat_container:
        for msg in st.session_state.messages:
            if msg["role"] == "assistant":
                st.markdown(f'<div class="bot-message">{msg["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="user-message">{msg["content"]}</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# --- KOLOM KANAN: MENU BUTTONS ---
with col_right:
    # Menggunakan st.container untuk styling gap
    with st.container():
        st.markdown('<div style="margin-top: 20px;"></div>', unsafe_allow_html=True)
        
        # Tombol 1: Daily Verse (Fungsional)
        if st.button("Daily Verse 🙏", use_container_width=True):
            verse = get_daily_verse()
            st.session_state.messages.append({"role": "assistant", "content": verse})
            st.rerun()

        # Tombol Dekoratif (Bisa dikembangkan nanti)
        st.button("Prayer Wall ✨", use_container_width=True)
        st.button("Study Plans 📖", use_container_width=True)
        st.button("Youth Groups 👯‍♂️", use_container_width=True)

# --- INPUT AREA (FLOATING BOTTOM) ---
# Input ditaruh di luar kolom agar full width atau centered di bawah
input_placeholder = st.empty()
user_input = st.chat_input("Ketik pertanyaanmu di sini...", key="chat_input")

if user_input:
    # 1. Tampilkan pesan user
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # 2. Proses Jawaban (Loading state)
    with col_center:
        with st.spinner("🕊️ Mencari hikmat..."):
             response = get_chat_response(user_input)
             st.session_state.messages.append({"role": "assistant", "content": response})
    
    # 3. Refresh halaman untuk update UI
    st.rerun()

# FOOTER
st.markdown('<div class="footer-text">Powered by Divine AI. Connect. Learn. Grow.</div>', unsafe_allow_html=True)
