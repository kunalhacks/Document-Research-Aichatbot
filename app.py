import streamlit as st
import os
import sys
import uuid
from typing import List, Dict, Any
from datetime import datetime

# Ensure internal app directory is on PYTHONPATH
ROOT_DIR = os.path.dirname(__file__)
SERVICES_DIR = os.path.join(ROOT_DIR, "app")
if SERVICES_DIR not in sys.path:
    sys.path.append(SERVICES_DIR)

# Imports from internal services package
from services.document_processor import DocumentProcessor
from services.vector_store import create_vector_store, SearchResult

# (Removed duplicate st.set_page_config; defined in streamlit_app.py)

# Custom CSS for better styling
st.markdown("""
<style>
    .main {
        max-width: 1200px;
        margin: 0 auto;
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .result-card {
        border-left: 4px solid #4CAF50;
        padding: 1rem;
        margin: 1rem 0;
        background-color: #f9f9f9;
        border-radius: 0 8px 8px 0;
    }
    .theme-card {
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        background-color: #f0f8ff;
    }
    .citation {
        font-size: 0.8rem;
        color: #666;
        font-style: italic;
    }
    .card {background: rgba(255,255,255,0.05); border:1px solid rgba(255,255,255,0.1); border-radius:12px; padding:1.5rem; margin:1rem 0; backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px); box-shadow:0 4px 12px rgba(0,0,0,.4);}
    .card:hover {border-color:#8b5cf6; box-shadow:0 6px 18px rgba(139,92,246,.6);}
    h3.card-title {color:#a78bfa;margin:0 0 .75rem;}
    body {
        background: radial-gradient(circle at top left, #4f46e5 0%, #111827 40%, #0f0f17 100%);
    }
    .bg-gradient {display:none;}
    .hero {
        text-align:center;
        padding:3rem 0 2rem;
        animation: fadeSlide 1s ease-out;
        position:relative; z-index:1;
    }
    .hero-title {
        font-size:3rem;
        font-weight:700;
        background: linear-gradient(90deg,#8b5cf6,#4f46e5,#06b6d4);
        -webkit-background-clip:text;
        -webkit-text-fill-color:transparent;
        background-size:200% auto;
        animation: gradientMove 3s linear infinite;
    }
    @keyframes gradientMove {to {background-position: -200% center;}}
    @keyframes fadeSlide {from{opacity:0;transform:translateY(-20px);}to{opacity:1;transform:translateY(0);}}
    #stars, #stars2, #stars3 {display:none !important;}
    :root {
        --shadow-stars: 1000px 2000px #fff, 1500px 500px #fff, 2000px 1500px #fff, 2500px 800px #fff, 300px 1200px #fff, 600px 300px #fff, 900px 1800px #fff;
    }
    @keyframes animStar {from{transform:translateY(0);}to{transform:translateY(-2000px);} }
    #stars {animation: animStar 200s linear infinite;}
    #stars2 {animation: animStar 400s linear infinite; opacity:0.6;}
    #stars3 {animation: animStar 600s linear infinite; opacity:0.3;}
    [data-testid="stAppViewContainer"] > div > div:nth-child(2) [data-testid="stDecoration"] {
        display:none !important;
    }
    [data-testid="stAppViewContainer"] > div > div:nth-child(2) [data-testid="block-container"] {
        animation:none !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []
if 'search_history' not in st.session_state:
    st.session_state.search_history = []

# Initialize processor and vector store (cached in session state to avoid re-init)
if 'doc_processor' not in st.session_state:
    st.session_state.doc_processor = DocumentProcessor()
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = create_vector_store(persist_directory="./data/vector_db")

# Sidebar
with st.sidebar:
    st.title("Document Research Chatbot")
    st.write("---")
    
    # Document upload section
    st.subheader("📂 Upload Documents")
    uploaded_files = st.file_uploader(
        "Choose files",
        type=["pdf", "docx", "txt", "pptx"],
        accept_multiple_files=True,
        key="file_uploader"
    )
    
    # Display uploaded files
    if uploaded_files:
        st.subheader("📝 Uploaded Files")
        for file in uploaded_files:
            if file.name not in [f['name'] for f in st.session_state.uploaded_files]:
                st.session_state.uploaded_files.append({
                    'id': str(uuid.uuid4()),
                    'name': file.name,
                    'uploaded_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                # Save file to disk and process
                save_path = os.path.join("data/raw", file.name)
                with open(save_path, "wb") as f:
                    f.write(file.getbuffer())
                doc = st.session_state.doc_processor.process_document(save_path)
                if doc:
                    st.session_state.vector_store.add_document(doc)
        
        for file in st.session_state.uploaded_files:
            col1, col2 = st.columns([4, 1])
            with col1:
                st.write(f"📄 {file['name']}")
            with col2:
                if st.button(f"❌", key=f"del_{file['id']}"):
                    st.session_state.uploaded_files = [f for f in st.session_state.uploaded_files if f['id'] != file['id']]
                    st.rerun()
    
    st.write("---")
    
    # Search history
    st.subheader("🔍 Search History")
    if st.session_state.search_history:
        for i, search in enumerate(reversed(st.session_state.search_history[-5:])):
            if st.button(search['query'][:30] + ("..." if len(search['query']) > 30 else ""), 
                        key=f"history_{i}", 
                        use_container_width=True):
                st.session_state.current_query = search['query']
                st.rerun()
    else:
        st.caption("Your search history will appear here")

# Animated hero header
st.markdown("""
<div class='hero'>
  <h1 class='hero-title'>📚 Document Research Assistant</h1>
  <lottie-player src="https://assets1.lottiefiles.com/packages/lf20_jtbfg2nb.json"  background="transparent"  speed="1"  style="width: 150px; height: 150px; margin:auto;"  loop  autoplay></lottie-player>
</div>
""", unsafe_allow_html=True)

# Search bar
st.markdown("<div style='padding:1rem;'>", unsafe_allow_html=True)
with st.form(key='search_form'):
    query = st.text_area(
        "🔍 Ask a question about your documents:",
        height=100,
        key='query_input',
        value=st.session_state.get('current_query', '')
    )
    
    col1, col2 = st.columns([1, 4])
    with col1:
        search_btn = st.form_submit_button("Search", type="primary", use_container_width=True)
    with col2:
        clear_btn = st.form_submit_button("Clear Results", type="secondary", use_container_width=True)

# Process search
if search_btn and query.strip():
    if not st.session_state.uploaded_files:
        st.warning("⚠️ Please upload some documents first!")
    else:
        # Add to search history
        if query not in [s['query'] for s in st.session_state.search_history]:
            st.session_state.search_history.append({
                'query': query,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
        
        with st.spinner("🔍 Analyzing documents..."):

            search_results: List[SearchResult] = st.session_state.vector_store.search(query, n_results=10)
            
            
            docs_map: Dict[str, Dict[str, Any]] = {}
            for res in search_results:
                doc_id = res.document.doc_id
                if doc_id not in docs_map:
                    docs_map[doc_id] = {
                        'id': doc_id,
                        'filename': res.document.title or "Unnamed Document",
                        'answer': res.chunk.content.strip().replace('\n', ' ')[:500] + ("..." if len(res.chunk.content) > 500 else ""),
                        'citation': f"Page {res.chunk.page_number+1}, Chunk {res.chunk.chunk_index+1}",
                        'relevance': res.score
                    }
            
            documents_list = sorted(docs_map.values(), key=lambda x: x['relevance'], reverse=True)
            
            st.session_state.search_results = {
                'query': query,
                'documents': documents_list,
                'themes': []
            }

if 'search_results' in st.session_state and not clear_btn:
    results = st.session_state.search_results
    
    # Document Results Section
    st.header("📄 Document Answers")
    
    if results['documents']:
        # Create a card of results
        for doc in results['documents']:
            st.markdown(f"""<div class='card'>
            <h3 class='card-title'>📄 {doc['filename']}</h3>
            <p><b>Relevance:</b> {doc['relevance']*100:.1f}%</p>
            <p>{doc['answer']}</p>
            <p class='citation'>📌 {doc['citation']}</p>
            </div>""", unsafe_allow_html=True)
    else:
        st.info("No relevant document matches found.")
    
    # Themes Section
    if results.get('themes'):
        st.header("✨ Key Themes")
        
        for theme in results['themes']:
            with st.expander(f"🔍 {theme['name']} (Confidence: {theme['confidence']*100:.0f}%)", expanded=True):
                st.markdown(f"**Summary:** {theme['summary']}")
                
                st.markdown("**Supporting Evidence:**")
                for cite in theme['citations']:
                    st.markdown(f"- 📌 {cite['doc_id']}: {cite['citation']}")
                
                with st.chat_message("assistant"):
                    st.write(f"Based on the analysis of your documents, I've identified '{theme['name']}' as a key theme. {theme['summary']} This insight is supported by evidence from multiple sources as shown above.")
    
    # Clear results if needed
    if clear_btn:
        if 'search_results' in st.session_state:
            del st.session_state.search_results
        st.rerun()

st.markdown("</div>", unsafe_allow_html=True)  # close padding wrapper

# Add footer
st.markdown("---")
st.caption("Document Research Assistant v1.0 | Powered by Streamlit")

# Add some JavaScript for better UX
st.components.v1.html("""
<script>
    // Auto-scroll to results
    document.addEventListener('DOMContentLoaded', function() {
        if (window.location.hash === '#results') {
            const element = document.getElementById('results');
            if (element) {
                element.scrollIntoView({behavior: 'smooth'});
            }
        }
    });
</script>
""")
