import streamlit as st
import os
import pandas as pd
from llama_index.core import VectorStoreIndex, PromptTemplate, Document
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from datetime import datetime
import time

# ====================================================================
# A. KONFIGURASI APLIKASI DAN PENGAMANAN
# ====================================================================

# Kunci API Anda (AIzaSy...) harus disetel sebagai variabel lingkungan bernama GEMINI_API_KEY
API_KEY_ANDA = os.environ.get("GEMINI_API_KEY")
LLM_MODEL = "gemini-2.5-flash"
DATA_FILE = "Alkitab.csv"      # File Data Utama
USER_LOG_FILE = "user_log.csv" # File Baru untuk menyimpan nama pendaftar

# --- 1. SETUP TAMPILAN MODERN DAN TRENDY ---
st.set_page_config(
    page_title="GOD REMIND YOU🕊️❤️",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS Modern dan Trendy
st.markdown(
    """
    <style>
    /* Global Styles */
    body {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: #333;
    }
    
    /* Header Styling */
    .header-container {
        text-align: center;
        padding: 20px;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        margin-bottom: 20px;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }
    .title-text {
        font-size: 3.5em;
        font-weight: 800;
        color: #fff;
        text-shadow: 0 0 20px rgba(255, 255, 255, 0.5);
        animation: glow 2s ease-in-out infinite alternate;
    }
    @keyframes glow {
        from { text-shadow: 0 0 20px rgba(255, 255, 255, 0.5); }
        to { text-shadow: 0 0 30px rgba(255, 255, 255, 0.8); }
    }
    .subtitle {
        font-size: 1.2em;
        color: #e0e0e0;
        margin-top: 10px;
    }
    
    /* Chat Container */
    .chat-container {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
        max-height: 70vh;
        overflow-y: auto;
    }
    
    /* Message Styling */
    .user-message {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 15px;
        border-radius: 20px 20px 5px 20px;
        margin: 10px 0;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
        animation: slideIn 0.5s ease-out;
    }
    .bot-message {
        background: linear-gradient(135deg, #f093fb, #f5576c);
        color: white;
        padding: 15px;
        border-radius: 20px 20px 20px 5px;
        margin: 10px 0;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
        animation: slideIn 0.5s ease-out;
    }
    @keyframes slideIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    .sidebar-header {
        text-align: center;
        padding: 10px;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        margin-bottom: 20px;
    }
    
    /* Button Styling */
    .stButton>button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 10px 20px;
        font-weight: bold;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
    }
    
    /* Input Styling */
    .stTextInput>div>div>input {
        border-radius: 25px;
        border: 2px solid #667eea;
        padding: 10px 15px;
        transition: all 0.3s ease;
    }
    .stTextInput>div>div>input:focus {
        border-color: #764ba2;
        box-shadow: 0 0 10px rgba(102, 126, 234, 0.5);
    }
    
    /* Metric Styling */
    .metric-container {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 10px;
        margin: 10px 0;
        text-align: center;
    }
    
    /* Expander Styling */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        color: white;
    }
    
    /* Dataframe Styling */
    .dataframe {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 10px;
        overflow: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Header Modern
st.markdown(
    """
    <div class="header-container">
        <div class="title-text">GOD REMIND YOU🕊️❤️</div>
        <div class="subtitle">— Your Modern Biblical Companion Powered by AI —</div>
    </div>
    """,
    unsafe_allow_html=True
)

# ====================================================================
# B. FUNGSI LOGIKA RAG (Indexing, Hanya Berjalan Sekali)
# ====================================================================

@st.cache_resource(show_spinner="🔄 Loading Divine Wisdom... (This happens only once)")
def load_and_index_data():
    """Memuat data CSV, membuat LLM, Embeddings, dan Index RAG."""

    if not API_KEY_ANDA:
        st.error("❌ API Key Error: Please set GEMINI_API_KEY in your environment variables.")
        return None, None

    try:
        df = pd.read_csv(DATA_FILE)
        
        if 'Isi' in df.columns:
            df['text_bersih'] = df['Isi'].astype(str).str.replace('<t/>', '', regex=False)
        else:
            df['text_bersih'] = df['text'].astype(str)

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

    except Exception as e:
        st.error(f"❌ Initialization Error: {e}")
        return None, None

INDEX, LLM = load_and_index_data()

# ====================================================================
# C. FUNGSI UTAMA CHAT DAN RIWAYAT
# ====================================================================

def get_query_engine():
    if INDEX is None or LLM is None:
        return None

    custom_qa_template = PromptTemplate(
        """You are 'Divine Companion', a wise Biblical chatbot.
        
        Task: Answer questions based on the provided Biblical context.
        
        RESPONSE FORMAT RULES (IMPORTANT):
        1. NEVER mention "Kitab 10" or "Kitab 19". Use original book names (e.g., Genesis, Psalms).
        2. If relevant verses are found, QUOTE THEM FULLY.
        3. Use this Markdown format for beautiful display:
           - For Verse References, use blue bold. Example: **:blue[Genesis 1:1]**
           - For Verse Content, use blockquote. Example: > "In the beginning..."
           - For Explanations, use friendly plain text.
        
        Biblical Context:
        {context_str}

        User Question:
        {query_str}

        Your Answer:"""
    )

    return INDEX.as_query_engine(
        llm=LLM,
        response_mode='compact',
        text_qa_template=custom_qa_template
    )

def generate_response(query):
    query_engine = get_query_engine()
    if query_engine is None:
        return "System not ready. Please check API key."

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
        st.error(f"Error: {e}")
        return "Sorry, a technical issue occurred while seeking answers."

# ====================================================================
# D. LOGIKA PENYIMPANAN PENGGUNA (BUKU TAMU)
# ====================================================================

def log_user_to_csv(username):
    """Menyimpan nama pengguna baru ke file CSV."""
    try:
        if not os.path.exists(USER_LOG_FILE):
            df = pd.DataFrame(columns=['Username', 'Join Time'])
            df.to_csv(USER_LOG_FILE, index=False)
        
        df = pd.read_csv(USER_LOG_FILE)
        
        if username not in df['Username'].values:
            new_row = pd.DataFrame({
                'Username': [username],
                'Join Time': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
            })
            df = pd.concat([df, new_row], ignore_index=True)
            df.to_csv(USER_LOG_FILE, index=False)
            
    except Exception as e:
        st.error(f"Failed to log user: {e}")

# ====================================================================
# E. TAMPILAN UI (Login, Sidebar, Chat)
# ====================================================================

def check_login():
    """Modern login form in sidebar."""
    if "user_name" not in st.session_state or not st.session_state.user_name:
        with st.sidebar:
            st.markdown('<div class="sidebar-header"><h3>👋 Welcome Aboard</h3></div>', unsafe_allow_html=True)
            st.info("Enter your name to start your divine journey.")

            user_input = st.text_input("Username (Required)", key="login_input", placeholder="e.g., John Doe")

            if st.button("🚀 Start Session", use_container_width=True):
                if user_input.strip():
                    st.session_state.user_name = user_input.strip()
                    st.session_state.chat_history = []
                    st.session_state.session_start = datetime.now()
                    st.session_state.messages = [{"role": "assistant", "content": f"✨ Hello {st.session_state.user_name}! I'm your Divine Companion. What biblical wisdom can I share with you today?"}]
                    
                    log_user_to_csv(st.session_state.user_name)
                    
                    st.rerun()
                else:
                    st.warning("Username cannot be empty.")
            return False
    return True

def setup_sidebar():
    """Modern sidebar with session report and user registry."""
    with st.sidebar:
        st.markdown('<div class="sidebar-header"><h4>📊 Session Dashboard</h4></div>', unsafe_allow_html=True)
        
        if "user_name" in st.session_state:
            st.success(f"Active: **{st.session_state.user_name}**")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Questions Asked", len(st.session_state.chat_history))
            with col2:
                st.metric("Session Time", f"{(datetime.now() - st.session_state.session_start).seconds // 60}m")

            with st.expander("📜 Chat History"):
                if st.session_state.chat_history:
                    for item in reversed(st.session_state.chat_history):
                        st.markdown(f"**[{item['time']}] You:** {item['user']}")
                        st.caption(f"**Bot:** {item['bot'][:100]}...")
                else:
                    st.info("Start chatting to see history!")

            st.markdown("---")
            st.subheader("📖 User Registry")
            try:
                if os.path.exists(USER_LOG_FILE):
                    df_users = pd.read_csv(USER_LOG_FILE)
                    st.dataframe(df_users, hide_index=True, use_container_width=True)
                else:
                    st.info("No users registered yet.")
            except Exception:
                st.caption("Failed to load registry.")

            if st.button("🔚 End Session", use_container_width=True):
                st.toast(f"Session ended for {st.session_state.user_name}.")
                st.session_state.clear()
                st.rerun()

# ====================================================================
# F. MAIN EXECUTION
# ====================================================================

if __name__ == "__main__":
    if not check_login():
        st.stop()

    setup_sidebar()

    # Chat Container
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    if "messages" not in st.session_state:
        pass

    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f'<div class="user-message">👤 {message["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="bot-message">🤖 {message["content"]}</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    if prompt := st.chat_input(f"💬 Ask about the Bible, {st.session_state.user_name}..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Add user message to chat
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        st.markdown(f'<div class="user-message">👤 {prompt}</div>', unsafe_allow_html=True)
        
        with st.spinner("🔍 Seeking divine wisdom..."):
            full_response = generate_response(prompt)
        
        st.markdown(f'<div class="bot-message">🤖 {full_response}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        
        # Auto-scroll to bottom (using rerun for simplicity)
        time.sleep(0.1)
        st.rerun()
