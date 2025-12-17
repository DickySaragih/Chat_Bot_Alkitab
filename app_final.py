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
DATA_FILE = "Alkitab.csv"      # File Data Utama
USER_LOG_FILE = "user_log.csv" # File Baru untuk menyimpan nama pendaftar

# --- 1. SETUP TAMPILAN PROFESIONAL DAN TRENDY ---
st.set_page_config(
    page_title="GOD REMIND YOU🕊️❤️",
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
        background-color: #e6f3ff;
        border-left: 5px solid #4682B4;
        padding: 10px;
        margin: 5px 0;
        border-radius: 5px;
        font-style: italic;
    }
    </style>
    <div class='title-text'>GOD REMIND YOU🕊️❤️</div>
    <h5 style='text-align: center; color: #4682B4;'>— Pendamping Firman Anda Berdasarkan Alkitab AYT —</h5>
    """,
    unsafe_allow_html=True
)

# ====================================================================
# B. FUNGSI LOGIKA RAG (Indexing, Hanya Berjalan Sekali)
# ====================================================================

@st.cache_resource(show_spinner="Memuat dan mengindeks Firman Tuhan (Ini hanya terjadi sekali)...")
def load_and_index_data():
    """Memuat data CSV, membuat LLM, Embeddings, dan Index RAG."""

    if not API_KEY_ANDA:
        st.error("❌ ERROR: Kunci API Gemini tidak ditemukan di variabel lingkungan 'GEMINI_API_KEY'.")
        st.info("Mohon setel variabel lingkungan 'GEMINI_API_KEY' di CMD atau sistem Anda.")
        return None, None

    try:
        # Pemuatan dan Pemrosesan Data CSV
        df = pd.read_csv(DATA_FILE)
        
        # Membersihkan teks
        if 'Isi' in df.columns:
            df['text_bersih'] = df['Isi'].astype(str).str.replace('<t/>', '', regex=False)
        else:
            df['text_bersih'] = df['text'].astype(str)

        # Membuat Referensi Lengkap
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

        llm = GoogleGenAI(model=LLM_MODEL, api_key=API_KEY_ANDA)
        embed_model = GoogleGenAIEmbedding(model='models/embedding-001', api_key=API_KEY_ANDA)

        index = VectorStoreIndex.from_documents(documents, llm=llm, embed_model=embed_model)
        return index, llm

    except KeyError as e:
        st.error(f"❌ ERROR: Nama kolom CSV tidak sesuai. Detail: {e}")
        return None, None
    except FileNotFoundError:
        st.error(f"❌ ERROR: File data '{DATA_FILE}' tidak ditemukan.")
        return None, None
    except Exception as e:
        st.error(f"❌ ERROR saat inisialisasi RAG: {e}")
        return None, None

INDEX, LLM = load_and_index_data()

# ====================================================================
# C. FUNGSI UTAMA CHAT DAN RIWAYAT
# ====================================================================

def get_query_engine():
    if INDEX is None or LLM is None:
        return None

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
    query_engine = get_query_engine()
    if query_engine is None:
        return "Sistem RAG belum siap."

    try:
        response = query_engine.query(query)
        bot_response_text = str(response)

        st.session_state.chat_history.append({
            "user": query,
            "bot": bot_response_text,
            "time": datetime.now().strftime('%H:%M:%S')
        })

        return bot_response_text
    except Exception as e:
        if "API key not valid" in str(e):
             st.error("Kesalahan Kunci API.")
        else:
             st.error(f"Terjadi kesalahan: {e}")
        return "Maaf, terjadi masalah teknis saat mencari jawaban."


# ====================================================================
# D. LOGIKA PENYIMPANAN PENGGUNA (BUKU TAMU)
# ====================================================================

def log_user_to_csv(username):
    """Menyimpan nama pengguna baru ke file CSV."""
    try:
        # Cek apakah file sudah ada
        if not os.path.exists(USER_LOG_FILE):
            # Buat file baru dengan header
            df = pd.DataFrame(columns=['Nama Pengguna', 'Waktu Bergabung'])
            df.to_csv(USER_LOG_FILE, index=False)
        
        # Baca file yang ada
        df = pd.read_csv(USER_LOG_FILE)
        
        # Cek apakah user sudah ada (agar tidak duplikat)
        if username not in df['Nama Pengguna'].values:
            new_row = pd.DataFrame({
                'Nama Pengguna': [username],
                'Waktu Bergabung': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
            })
            # Gabungkan dan simpan
            df = pd.concat([df, new_row], ignore_index=True)
            df.to_csv(USER_LOG_FILE, index=False)
            
    except Exception as e:
        print(f"Gagal menyimpan log user: {e}")

# ====================================================================
# E. TAMPILAN UI (Login, Sidebar, Chat)
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
                        # 1. Simpan ke Session State
                        st.session_state.user_name = user_input.strip()
                        st.session_state.chat_history = []
                        st.session_state.session_start = datetime.now()
                        st.session_state.messages = [{"role": "assistant", "content": f"Halo {st.session_state.user_name}! Saya Pendamping Firman. Apa yang bisa saya bantu hari ini terkait Firman Tuhan?"}]
                        
                        # 2. CATAT KE BUKU TAMU (CSV)
                        log_user_to_csv(st.session_state.user_name)
                        
                        st.rerun()
                    else:
                        st.warning("Nama pengguna tidak boleh kosong.")
            return False
    return True

def setup_sidebar():
    """Menampilkan Laporan Sesi dan Buku Tamu."""
    with st.sidebar:
        st.markdown("---")
        st.header("🧾 Laporan Pribadi")
        
        if "user_name" in st.session_state:
            st.success(f"Sesi Aktif: **{st.session_state.user_name}**")
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

            # --- FITUR BARU: BUKU TAMU (INVOICE DAFTAR PENGGUNA) ---
            st.markdown("---")
            st.header("📒 Buku Daftar Pengguna")
            st.caption("Daftar orang yang telah menggunakan aplikasi ini:")
            
            try:
                if os.path.exists(USER_LOG_FILE):
                    df_users = pd.read_csv(USER_LOG_FILE)
                    # Tampilkan sebagai tabel interaktif
                    st.dataframe(
                        df_users, 
                        hide_index=True,
                        use_container_width=True
                    )
                else:
                    st.info("Belum ada data pengguna tersimpan.")
            except Exception:
                st.caption("Gagal memuat daftar pengguna.")
            # -------------------------------------------------------

            st.markdown("---")
            if st.button("Akhiri Sesi dan Logout", use_container_width=True):
                st.toast(f"Sesi {st.session_state.user_name} diakhiri.")
                st.session_state.clear()
                st.rerun()

# ====================================================================
# F. MAIN EXECUTION
# ====================================================================

if __name__ == "__main__":
    if not check_login():
        st.stop()

    setup_sidebar()

    if "messages" not in st.session_state:
        pass

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input(f"Halo {st.session_state.user_name}, tanyakan sesuatu tentang Alkitab..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Mencari Firman Tuhan..."):
                full_response = generate_response(prompt)
            st.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})

