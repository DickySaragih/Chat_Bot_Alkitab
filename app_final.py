import streamlit as st
import os
import pandas as pd
from datetime import datetime
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding

# ====================================================================
# 1. KONFIGURASI HALAMAN & API
# ====================================================================

# Pastikan API Key diatur di Secrets Streamlit atau Environment Variable
API_KEY_ANDA = os.environ.get("GEMINI_API_KEY")

st.set_page_config(
    page_title="Bible is Journey",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

DATA_FILE = "Alkitab.csv"

# ====================================================================
# 2. CSS RESPONSIF (LAPTOP & HP)
# ====================================================================

st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #1e1e2f 0%, #2d3436 100%);
        color: white;
    }
    header, footer, #MainMenu {visibility: hidden;}
    .block-container { max-width: 1100px; padding: 1.5rem; margin: auto; }
    .main-box {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(15px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
    }
    .app-title {
        font-family: 'serif';
        font-size: clamp(22px, 5vw, 38px);
        font-weight: 800;
        color: #f1c40f;
        text-align: center;
        letter-spacing: 2px;
    }
    .bible-logo-container { text-align: center; margin: 15px 0; font-size: 70px; }
    
    /* Bubble Chat Styling */
    .bubble-bot { background: #ffffff; color: #333; padding: 12px; border-radius: 15px 15px 15px 2px; margin-bottom: 10px; max-width: 85%; font-size: 14px; }
    .bubble-user { background: #f1c40f; color: #000; padding: 12px; border-radius: 15px 15px 2px 15px; margin-bottom: 10px; margin-left: auto; max-width: 85%; text-align: right; font-size: 14px; }

    /* Penyesuaian Mobile */
    @media (max-width: 768px) {
        .block-container { padding: 0.8rem; }
        .main-box { padding: 15px; }
    }
</style>
""", unsafe_allow_html=True)

# ====================================================================
# 3. BACKEND DENGAN PERBAIKAN ERROR HANDLING (FIX CLIENT ERROR)
# ====================================================================

@st.cache_resource(show_spinner=False)
def load_rag_system():
    if not API_KEY_ANDA:
        st.error("API Key tidak ditemukan. Harap atur GEMINI_API_KEY.")
        return None, None
    try:
        if not os.path.exists(DATA_FILE):
            st.error(f"File {DATA_FILE} tidak ditemukan!")
            return None, None
            
        df = pd.read_csv(DATA_FILE)
        # Pastikan kolom sesuai dengan file CSV Anda
        df['text_bersih'] = df['Isi'].astype(str).str.replace('<t/>', '', regex=False)
        df['referensi'] = df['Nama ayat'].astype(str) + ' ' + df['Bagian'].astype(str) + ':' + df['Ayat'].astype(str)
        
        documents = [Document(text=row['text_bersih'], metadata={"ref": row['referensi']}) for _, row in df.iterrows()]
        
        # Konfigurasi LLM dan Embedding
        # Gunakan model gemini-1.5-flash untuk menghindari batas kuota yang ketat
        Settings.llm = GoogleGenAI(model="models/gemini-1.5-flash", api_key=API_KEY_ANDA)
        Settings.embed_model = GoogleGenAIEmbedding(model="models/embedding-001", api_key=API_KEY_ANDA)
        
        index = VectorStoreIndex.from_documents(documents)
        return index, df
    except Exception as e:
        st.error(f"Gagal Inisialisasi: {str(e)}")
        return None, None

INDEX, DF_ALKITAB = load_rag_system()

def get_answer(query):
    if INDEX is None:
        return "Sistem tidak siap. Periksa API Key atau data sumber Anda."
    
    try:
        # Menambahkan parameter tambahan untuk membatasi kompleksitas respons jika perlu
        query_engine = INDEX.as_query_engine(similarity_top_k=3)
        response = query_engine.query(f"Gunakan data Alkitab yang tersedia untuk menjawab: {query}")
        return str(response)
    except Exception as e:
        # Menangani ClientError API secara spesifik
        error_msg = str(e)
        if "429" in error_msg or "quota" in error_msg.lower():
            return "Maaf, batas penggunaan harian API telah tercapai. Mohon coba lagi nanti."
        elif "400" in error_msg:
            return "Maaf, terjadi kesalahan pada permintaan. Mohon gunakan kalimat lain."
        else:
            return "Terjadi kendala teknis saat menghubungi server. Mohon ulangi pertanyaan Anda."

# ====================================================================
# 4. TAMPILAN UI
# ====================================================================

st.markdown('<div class="main-box">', unsafe_allow_html=True)
st.markdown('<p class="app-title">BIBLE IS JOURNEY</p>', unsafe_allow_html=True)
st.markdown('<div class="bible-logo-container">📖</div>', unsafe_allow_html=True)

# Grid Layout
col_chat, col_menu = st.columns([3, 1])

with col_chat:
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "ai", "content": "Selamat datang! Bagaimana perjalanan imanmu hari ini? ✨"}]
    
    # Area chat dengan tinggi otomatis yang responsif
    for m in st.session_state.messages:
        div_class = "bubble-bot" if m["role"] == "ai" else "bubble-user"
        st.markdown(f'<div class="{div_class}">{m["content"]}</div>', unsafe_allow_html=True)

with col_menu:
    st.markdown("### Menu")
    if st.button("Daily Verse 🙏", use_container_width=True):
        if DF_ALKITAB is not None:
            row = DF_ALKITAB.sample(1).iloc[0]
            st.session_state.messages.append({"role": "ai", "content": f"📖 **{row['referensi']}**\n\n{row['text_bersih']}"})
            st.rerun()
    st.button("Study Plan 📚", use_container_width=True)

# Input area di bawah
if prompt := st.chat_input("Tanyakan perjalananmu..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Tampilkan pesan user segera
    st.rerun()

# Logika untuk menghasilkan jawaban setelah rerun
if len(st.session_state.messages) > 0 and st.session_state.messages[-1]["role"] == "user":
    with st.spinner("Mencari hikmat..."):
        ans = get_answer(st.session_state.messages[-1]["content"])
        st.session_state.messages.append({"role": "ai", "content": ans})
        st.rerun()

st.markdown('</div>', unsafe_allow_html=True)
