import streamlit as st
import os
import pandas as pd
from llama_index.core import VectorStoreIndex, PromptTemplate, Document
from llama_index.llms.google_genai import GoogleGenAI
# Kode di bawah ini MENGHINDARI konflik HuggingFace/CodeCarbon:
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from datetime import datetime

# ====================================================================
# A. KONFIGURASI APLIKASI DAN PENGAMANAN
# ====================================================================

# Kunci API Anda (AIzaSy...) harus disetel sebagai variabel lingkungan bernama GEMINI_API_KEY
API_KEY_ANDA = os.environ.get("GEMINI_API_KEY")
LLM_MODEL = "gemini-2.5-flash"
DATA_FILE = "Alkitab.csv" # Pastikan file ini ada di folder yang sama

# --- 1. SETUP TAMPILAN PROFESIONAL DAN TRENDY ---
st.set_page_config(
    page_title="God Creation",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Judul Utama Aplikasi (Tampilan Trendy dan Elegan)
st.markdown(
    """
    <style>
    /* Styling Judul Besar */
    .title-text {
        font-size: 3em;
        font-weight: 700;
        color: #FFFFFF;
        text-align: center;
        padding-top: 10px;
        padding-bottom: 5px;
        text-shadow: 2px 2px 5px #4682B4;
    }
    /* Mengubah warna background chat area */
    [data-testid="stAppViewContainer"] {
        background-color: #F8F8FF;
    }
    /* Warna sidebar */
    [data-testid="stSidebar"] {
        background-color: #E6E6FA;
    }
    /* Custom style untuk highlight */
    .highlight-verse {
        background-color: #e6f3ff; /* Biru muda lembut */
        border-left: 5px solid #4682B4;
        padding: 10px;
        margin: 5px 0;
        border-radius: 5px;
        font-style: italic;
    }
    </style>
    <div class='title-text'>God Creation 🕊️</div>
    <h5 style='text-align: center; color: #4682B4;'>— Pendamping Firman Anda Berdasarkan Ayat Alkitab  —</h5>
    """,
    unsafe_allow_html=True
)

# ====================================================================
# B. FUNGSI LOGIKA RAG (Indexing, Hanya Berjalan Sekali)
# ====================================================================

@st.cache_resource(show_spinner="Memuat dan mengindeks Firman Tuhan (Ini hanya terjadi sekali)...")
def load_and_index_data():
    """Memuat data CSV, membuat LLM, Embeddings, dan Index RAG."""

    # ⚠️ Pengecekan Kunci API
    if not API_KEY_ANDA:
        st.error("❌ ERROR: Kunci API Gemini tidak ditemukan di variabel lingkungan 'GEMINI_API_KEY'.")
        st.info("Mohon setel variabel lingkungan 'GEMINI_API_KEY' di CMD atau sistem Anda.")
        return None, None

    try:
        # Pemuatan dan Pemrosesan Data CSV
        # PERBAIKAN: Disesuaikan dengan nama kolom di gambar CSV Anda
        df = pd.read_csv(DATA_FILE)
        
        # Membersihkan teks (menghapus tag <t/> jika ada)
        # Asumsi nama kolom berdasarkan gambar: 'Nama ayat' (Kitab), 'Bagian' (Pasal), 'Ayat', 'Isi'
        
        # Kita buat kolom 'text_bersih' untuk isi ayat
        if 'Isi' in df.columns:
            df['text_bersih'] = df['Isi'].astype(str).str.replace('<t/>', '', regex=False)
        else:
            # Fallback jika nama kolom beda (misal text)
            df['text_bersih'] = df['text'].astype(str)

        # Membuat Referensi Lengkap (Contoh: Kejadian 1:1)
        # Menggunakan 'Nama ayat' (bukan 'Buku' yang isinya angka)
        df['referensi'] = df['Nama ayat'].astype(str) + ' ' + df['Bagian'].astype(str) + ':' + df['Ayat'].astype(str)
        
        documents = [
            Document(
                text=row['text_bersih'],
                metadata={
                    "referensi": row['referensi'], 
                    "kitab": row['Nama ayat'],
                    "pasal": row['Bagian'],
                    "ayat": row['Ayat']
                }
            )
            for index, row in df.iterrows()
        ]

        # Inisialisasi LLM
        llm = GoogleGenAI(model=LLM_MODEL, api_key=API_KEY_ANDA)

        # --- MENGGUNAKAN GoogleGenAIEmbedding ---
        embed_model = GoogleGenAIEmbedding(
             model='models/embedding-001',
             api_key=API_KEY_ANDA
        )
        # ---------------------------------------------------------

        # Pembuatan Index RAG
        index = VectorStoreIndex.from_documents(
            documents,
            llm=llm,
            embed_model=embed_model
        )
        return index, llm

    except KeyError as e:
        st.error(f"❌ ERROR: Nama kolom CSV tidak sesuai. Pastikan ada kolom 'Nama ayat', 'Bagian', 'Ayat', dan 'Isi'. Detail: {e}")
        return None, None
    except FileNotFoundError:
        st.error(f"❌ ERROR: File data '{DATA_FILE}' tidak ditemukan. Mohon pastikan file CSV ada di folder yang sama.")
        return None, None
    except Exception as e:
        st.error(f"❌ ERROR saat inisialisasi RAG: {e}")
        return None, None

INDEX, LLM = load_and_index_data()

# ====================================================================
# C. FUNGSI UTAMA CHAT DAN RIWAYAT
# ====================================================================

def get_query_engine():
    """Menginisialisasi Query Engine dengan Prompt Template yang profesional."""

    if INDEX is None or LLM is None:
        return None

    # Prompt Template Kustom (DIPERBARUI UNTUK FORMAT CANTIK)
    custom_qa_template = PromptTemplate(
        """Anda adalah 'Pendamping Firman', Chatbot Alkitab yang bijaksana.
        
        Tugas Anda: Menjawab pertanyaan berdasarkan konteks Alkitab di bawah ini.
        
        ATURAN FORMAT JAWABAN (PENTING):
        1. JANGAN PERNAH menyebutkan "Kitab 10" atau "Kitab 19". Gunakan nama kitab asli (misal: Kejadian, Mazmur).
        2. Jika menemukan ayat yang relevan, KUTIP AYAT TERSEBUT SECARA LENGKAP.
        3. Gunakan format Markdown ini agar tampilan indah:
           - Untuk Referensi Ayat, gunakan warna biru/bold. Contoh: **:blue[Kejadian 1:1]**
           - Untuk Isi Ayat, gunakan format kutipan (blockquote). Contoh: > "Pada mulanya..."
           - Untuk Penjelasan, gunakan teks biasa yang ramah.
        
        Contoh Tampilan yang Diharapkan:
        "Berikut adalah ayat yang Anda cari:
        
        **:blue[Roma 3:23]**
        > _"Karena semua orang telah berbuat dosa dan telah kehilangan kemuliaan Allah."_
        
        Ayat ini menjelaskan bahwa..."

        Berikut adalah konteks informasi Alkitab:
        {context_str}

        Pertanyaan pengguna:
        {query_str}

        Jawaban Anda:"""
    )

    return INDEX.as_query_engine(
        llm=LLM,
        response_mode='compact',
        text_qa_template=custom_qa_template
    )

def generate_response(query):
    """Memanggil Query Engine RAG dan menyimpan riwayat."""

    query_engine = get_query_engine()
    if query_engine is None:
        return "Sistem RAG belum siap. Mohon periksa error di bagian atas halaman (kunci API atau file data)."

    try:
        # Query dan Response
        response = query_engine.query(query)
        bot_response_text = str(response)

        # Simpan riwayat sesi
        st.session_state.chat_history.append({
            "user": query,
            "bot": bot_response_text,
            "time": datetime.now().strftime('%H:%M:%S')
        })

        return bot_response_text
    except Exception as e:
        if "API key not valid" in str(e) or "400 Bad Request" in str(e):
             st.error("Kesalahan Kunci API: Pastikan API Key Gemini Anda benar dan aktif.")
        else:
             st.error(f"Terjadi kesalahan saat memproses pertanyaan: {e}")
        return "Maaf, terjadi masalah teknis saat mencari jawaban."


# ====================================================================
# D. LOGIKA TAMPILAN UI (Login, Sidebar, Chat)
# ====================================================================

def check_login():
    """Simulasi form login di sidebar."""
    if "user_name" not in st.session_state or not st.session_state.user_name:
        with st.sidebar:
            st.header("👋 Autentikasi Sesi")
            st.info("Masukkan nama Anda untuk memulai Pendamping Firman.")

            user_input = st.text_input("Nama Pengguna (Wajib)", key="login_input")

            col1, col2 = st.columns(2)
            with col1:
                if st.button("Masuk dan Mulai Sesi", use_container_width=True):
                    if user_input:
                        st.session_state.user_name = user_input.strip()
                        st.session_state.chat_history = []
                        st.session_state.session_start = datetime.now()
                        st.session_state.messages = [{"role": "assistant", "content": f"Halo {st.session_state.user_name}! Saya Pendamping Firman. Apa yang bisa saya bantu hari ini terkait Firman Tuhan?"}]
                        st.rerun()
                    else:
                        st.warning("Nama pengguna tidak boleh kosong.")
            return False
    return True

def setup_sidebar():
    """Menampilkan Laporan Sesi."""
    with st.sidebar:
        st.markdown("---")
        st.header("🧾 Laporan Sesi")
        
        if "user_name" in st.session_state:
            st.success(f"Sesi Aktif: **{st.session_state.user_name}**")
            st.caption(f"Waktu Mulai: {st.session_state.session_start.strftime('%d-%m-%Y %H:%M:%S')}")
            st.metric("Jumlah Pertanyaan", len(st.session_state.chat_history))

            st.markdown("---")
            st.subheader("Detail Riwayat")

            with st.expander("Klik untuk melihat riwayat lengkap"):
                if st.session_state.chat_history:
                    for item in reversed(st.session_state.chat_history):
                        st.markdown(f"**[{item['time']}] Anda:** *{item['user']}*")
                        st.caption(f"**Jawab:** {item['bot'][:80]}...")
                        st.markdown("---") 
                else:
                    st.info("Mulai percakapan untuk melihat riwayat.")

            st.markdown("---")
            if st.button("Akhiri Sesi dan Logout", use_container_width=True):
                st.toast(f"Sesi {st.session_state.user_name} diakhiri.")
                st.session_state.clear()
                st.rerun()


# ====================================================================
# E. FUNGSI UTAMA STREAMLIT (Chat Interface)
# ====================================================================


if __name__ == "__main__":

    # 1. Cek Login dan Tampilkan Sidebar
    if not check_login():
        st.stop() # Hentikan eksekusi jika belum login

    setup_sidebar()

    # 2. Logika Utama Chat
    if "messages" not in st.session_state:
        pass

    # Menampilkan riwayat pesan
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Logika input pengguna
    if prompt := st.chat_input(f"Halo {st.session_state.user_name}, tanyakan sesuatu tentang Alkitab..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Mencari Firman Tuhan..."):
                full_response = generate_response(prompt)

            st.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
