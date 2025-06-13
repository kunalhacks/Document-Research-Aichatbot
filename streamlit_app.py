# Streamlit front-end for Document Research Chatbot
# (moved from app.py to avoid name clash with package `app`)

import streamlit as st
import os
import uuid
from typing import List, Dict, Any
from datetime import datetime
import sys

# Set path to services directory
ROOT_DIR = os.path.dirname(__file__)
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
SERVICES_DIR = os.path.join(ROOT_DIR, "app")
if SERVICES_DIR not in sys.path:
    sys.path.append(SERVICES_DIR)

# Import backend processing components
from services.document_processor import DocumentProcessor
from services.vector_store import VectorStore, SearchResult

# Set page config
st.set_page_config(
    page_title="Document Research Chatbot",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- (rest of code identical to previous app.py contents after latest edits) ---
# For brevity, we import the logic from the original app module which already contains UI code.
# We simply `exec` its contents so we don't duplicate large chunk here.

# Load the original app.py code and execute it in this namespace
app_py_path = os.path.join(os.path.dirname(__file__), "app.py")
with open(app_py_path, "r", encoding="utf-8") as f:
    code = f.read()

exec(code, globals(), globals())
