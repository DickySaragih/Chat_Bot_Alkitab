import streamlit as st
import os
import pandas as pd
from datetime import datetime

# Menggunakan library yang lebih stabil untuk LlamaIndex
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
    layout="wide"
)

# CSS Responsif (Sesuai permintaan: Bible logo, Mobile & Laptop friendly)
st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #121212 0%, #1a1a2e 100%); color: white; }
    header, footer, #MainMenu {visibility: hidden;}
    .main-box {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(15px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 20px;
        margin: auto;
        max-width: 900px;
    }
    .app-title { font-family: 'serif'; font-size: clamp(24px, 5vw, 40px); color: #f1c40f; text-align: center; font-weight: 800; }
    .bible-logo { text-align: center; font-size: 60px; margin-bottom: 10px; }
    .bubble-bot { background: #fdfdfd; color: #222; padding: 12px; border-radius: 15px 15px 15px 2px; margin: 10px 0; max-width: 85%; font-size: 14px; box-shadow: 2px 2px 10px rgba(0,0,0,0.2); }
    .bubble-user { background: #f1c40f; color: #000; padding: 12px; border-radius: 15px 15px 2px 15px; margin: 10px 0 10px auto; max-width: 85%; text-align: right; font-size: 14px; }
</style>
""", unsafe_allow_html=True)

# ====================================================================
# 2. SISTEM RAG (DIPERBAIKI)
# ====================================================================

@st.cache_resource(show_spinner=False)
def init_system():
    if not API_KEY:
        st.error("API KEY TIDAK DITEMUKAN!")
        return None, None
    try:
        # Load Data
        df = pd.read_csv("Alkitab.csv")
        df['isi_clean'] = df['Isi'].astype(str).str.replace('<t/>', '', regex=False)
        df['ref'] = df['Nama ayat'].astype(str) + " " + df['Bagian'].astype(str) + ":" + df['Ayat'].astype(str)
        
        docs = [Document(text=row['isi_clean'], metadata={"ref": row['ref']}) for _, row in df.iterrows()]
        
        # PENTING: Gunakan klas Gemini & GeminiEmbedding (Bukan GoogleGenAI)
        # Ini lebih kompatibel dengan LlamaIndex versi terbaru
        Settings.llm = Gemini(model="models/gemini-1.5-flash", api_key=API_KEY)
        Settings.embed_model = GeminiEmbedding(model_name="models/embedding-001", api_key=API_KEY)
        Settings.chunk_size = 512
        
        index = VectorStoreIndex.from_documents(docs)
        return index, df
    except Exception as e:
        st.error(f"Error Inisialisasi: {e}")
        return None, None

INDEX, DF_ALKITAB = init_system()

def get_ai_response(user_query):
    if not INDEX:
        return "Sistem belum siap."
    try:
        # Gunakan mode respons 'compact' untuk menghindari error ClientError saat penggabungan teks
        query_engine = INDEX.as_query_engine(
            response_mode="compact", 
            similarity_top_k=2
        )
        response = query_engine.query(f"Jawablah dengan kasih, singkat, dan sertakan ayat: {user_query}")
        return str(response)
    except Exception as e:
        # Menangani error API tanpa mematikan aplikasi
        return f"Maaf, perjalanan ini menemui kendala teknis sementara. Mari coba tanyakan hal lain. 🙏"

# ====================================================================
# 3. TAMPILAN UI
# ====================================================================

st.markdown('<div class="main-box">', unsafe_allow_html=True)
st.markdown('<div class="bible-logo">📖</div>', unsafe_allow_html=True)
st.markdown('<div class="app-title">BIBLE IS JOURNEY</div>', unsafe_allow_html=True)

# State Management
if "history" not in st.session_state:
    st.session_state.history = [{"role": "ai", "content": "Halo! Saya adalah teman perjalananamu dalam memahami Alkitab. Apa yang ingin kamu ketahui?"}]

# Chat Display
chat_box = st.container()
with chat_box:
    for msg in st.session_state.history:
        style = "bubble-bot" if msg["role"] == "ai" else "bubble-user"
        st.markdown(f'<div class="{style}">{msg["content"]}</div>', unsafe_allow_html=True)

# Input
if prompt := st.chat_input("Tulis pertanyaanmu..."):
    st.session_state.history.append({"role": "user", "content": prompt})
    
    # Langsung jalankan AI
    with st.spinner("Mencari hikmat..."):
        res = get_ai_response(prompt)
        st.session_state.history.append({"role": "ai", "content": res})
    
    st.rerun()

# Tombol Menu Samping (Responsif)
st.markdown("---")
col1, col2 = st.columns(2)
with col1:
    if st.button("Ayat Hari Ini 🌟", use_container_width=True):
        if DF_ALKITAB is not None:
            r = DF_ALKITAB.sample(1).iloc[0]
            st.session_state.history.append({"role": "ai", "content": f"🌟 **{r['ref']}**\n\n{r['isi_clean']}"})
            st.rerun()
with col2:
    if st.button("Hapus Percakapan 🗑️", use_container_width=True):
        st.session_state.history = st.session_state.history[:1]
        st.rerun()

st.markdown('</div>', unsafe_allow_html=True)
