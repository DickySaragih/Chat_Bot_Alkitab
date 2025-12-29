import streamlit as st
import os
import pandas as pd
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding

# ====================================================================
# 1. KONFIGURASI HALAMAN
# ====================================================================
API_KEY = os.environ.get("GEMINI_API_KEY")

st.set_page_config(
    page_title="Bible is Journey",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ====================================================================
# 2. CSS "SINGLE BOX UI" (KONSISTEN LAPTOP & HP)
# ====================================================================
st.markdown("""
<style>
    /* Background Utama */
    .stApp {
        background: url("https://images.unsplash.com/photo-1464802686167-b939a6910659?q=80&w=2050&auto=format&fit=crop") no-repeat center center fixed;
        background-size: cover;
    }
    
    header, footer, #MainMenu {visibility: hidden;}

    /* KOTAK UTAMA (APP BOX) */
    .main-box {
        background: rgba(20, 20, 35, 0.85);
        backdrop-filter: blur(25px);
        -webkit-backdrop-filter: blur(25px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 30px;
        padding: 30px;
        max-width: 900px;
        margin: 20px auto;
        min-height: 85vh;
        box-shadow: 0 25px 50px rgba(0,0,0,0.5);
        position: relative;
        display: flex;
        flex-direction: column;
    }

    /* Judul & Logo */
    .app-header {
        text-align: center;
        margin-bottom: 20px;
    }
    .app-title {
        color: #f1c40f;
        font-family: 'Garamond', serif;
        font-size: clamp(24px, 5vw, 38px);
        font-weight: 800;
        letter-spacing: 2px;
        margin: 0;
    }
    .bible-icon { font-size: 50px; margin-bottom: 5px; }

    /* Area Chat */
    .chat-area {
        flex-grow: 1;
        overflow-y: auto;
        padding-right: 10px;
        margin-bottom: 20px;
    }

    .bubble-bot {
        background: #ffffff;
        color: #1a1a1a;
        padding: 12px 18px;
        border-radius: 20px 20px 20px 5px;
        margin-bottom: 12px;
        max-width: 80%;
        font-size: 14px;
        line-height: 1.5;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }

    .bubble-user {
        background: linear-gradient(135deg, #f1c40f 0%, #d35400 100%);
        color: white;
        padding: 12px 18px;
        border-radius: 20px 20px 5px 20px;
        margin-left: auto;
        margin-bottom: 12px;
        max-width: 80%;
        text-align: right;
        font-size: 14px;
        box-shadow: 0 4px 15px rgba(211, 84, 0, 0.3);
    }

    /* Input Field Custom */
    .stChatInputContainer {
        padding-bottom: 10px !important;
    }
    
    /* Tombol Menu */
    .stButton button {
        background: rgba(255,255,255,0.05) !important;
        border: 1px solid rgba(255,255,255,0.2) !important;
        color: #f1c40f !important;
        border-radius: 50px !important;
        transition: 0.3s;
    }
    .stButton button:hover {
        background: #f1c40f !important;
        color: black !important;
    }

    /* Penyesuaian Mobile */
    @media (max-width: 768px) {
        .main-box { padding: 15px; margin: 10px; min-height: 90vh; }
        .app-title { font-size: 20px; }
    }
</style>
""", unsafe_allow_html=True)

# ====================================================================
# 3. BACKEND (FIX 404 & MODEL NOT FOUND)
# ====================================================================

@st.cache_resource(show_spinner=False)
def load_bible_system():
    if not API_KEY:
        st.error("API Key Gemini tidak ditemukan!")
        return None, None
    try:
        # Load Data
        df = pd.read_csv("Alkitab.csv")
        df['clean_text'] = df['Isi'].astype(str).str.replace('<t/>', '', regex=False)
        df['ref'] = df['Nama ayat'].astype(str) + " " + df['Bagian'].astype(str) + ":" + df['Ayat'].astype(str)
        
        docs = [Document(text=row['clean_text'], metadata={"ref": row['ref']}) for _, row in df.iterrows()]
        
        # FIX: Gunakan penamaan model yang lebih spesifik untuk menghindari 404
        Settings.llm = Gemini(
            model_name="models/gemini-1.5-flash-latest", 
            api_key=API_KEY,
            temperature=0.5
        )
        Settings.embed_model = GeminiEmbedding(
            model_name="models/embedding-001", 
            api_key=API_KEY
        )
        
        index = VectorStoreIndex.from_documents(docs)
        return index, df
    except Exception as e:
        st.error(f"Sistem Gagal Dimuat: {e}")
        return None, None

INDEX, DATA_DF = load_bible_system()

def get_answer(query):
    if not INDEX: return "Sistem Offline."
    try:
        # Gunakan mode 'compact' agar lebih stabil di API v1beta
        query_engine = INDEX.as_query_engine(response_mode="compact", similarity_top_k=2)
        response = query_engine.query(f"Jawablah dengan penuh kasih dan singkat sebagai asisten Alkitab. Sertakan ayatnya: {query}")
        return str(response)
    except Exception as e:
        return f"Maaf, ada gangguan koneksi ke surga (API). Mohon coba lagi. 🙏"

# ====================================================================
# 4. TAMPILAN APLIKASI (DALAM SATU BOX)
# ====================================================================

st.markdown('<div class="main-box">', unsafe_allow_html=True)

# Header
st.markdown("""
<div class="app-header">
    <div class="bible-icon">📖</div>
    <p class="app-title">BIBLE IS JOURNEY</p>
</div>
""", unsafe_allow_html=True)

# Chat History Container
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "ai", "content": "Selamat datang di perjalanan imanmu. Apa yang ingin kamu tanyakan hari ini? ✨"}]

# Menampilkan Chat dalam Area Scrollable
chat_placeholder = st.container()
with chat_placeholder:
    for m in st.session_state.messages:
        cls = "bubble-bot" if m["role"] == "ai" else "bubble-user"
        st.markdown(f'<div class="{cls}">{m["content"]}</div>', unsafe_allow_html=True)

# Menu Buttons (Responsif)
st.markdown("<br>", unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    if st.button("Ayat Hari Ini 🙏", use_container_width=True):
        if DATA_DF is not None:
            r = DATA_DF.sample(1).iloc[0]
            st.session_state.messages.append({"role": "ai", "content": f"📖 **{r['ref']}**\n\n{r['clean_text']}"})
            st.rerun()
with col2:
    if st.button("Hapus Percakapan 🗑️", use_container_width=True):
        st.session_state.messages = st.session_state.messages[:1]
        st.rerun()

# Chat Input (Otomatis berada di bawah box)
if prompt := st.chat_input("Tulis pertanyaanmu di sini..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.spinner("Mencari jawaban..."):
        ans = get_answer(prompt)
        st.session_state.messages.append({"role": "ai", "content": ans})
    st.rerun()

st.markdown('</div>', unsafe_allow_html=True) # Tutup main-boxs
