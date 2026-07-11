import os
import asyncio
from pathlib import Path
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings

# --- Core Backend Imports ---
from database.metrics import get_metrics
from database.postgres_sql import create_database
from database.create_table import create_tables

# --- Safe Route & Ingestion Imports ---
try:
    from routes.chat import chat as chat_endpoint
    from routes.chat import ChatRequest
    
    def get_chat_response(query):
        """ Wraps the FastAPI async endpoint to work seamlessly inside Streamlit """
        request_data = ChatRequest(question=query, company=None, year=None)
        try:
            # Sync wrapper for async FastAPI call
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            response = loop.run_until_complete(chat_endpoint(request_data))
            loop.close()
            return response.get("answer", "No answer found.")
        except Exception as e:
            return f"Error executing RAG pipeline: {str(e)}"
except ImportError:
    def get_chat_response(query): 
        return "RAG Connection Warning: Could not import chat endpoint from routes.chat."

try:
    from ingestion.ingest_documents import ingest_document 
except ImportError:
    def ingest_document(pdf_path, embeddings, client, collection_name):
        return "Ingestion module import missing."

load_dotenv()

# --- 1. Global Initialization Layer (Cached Once) ---
@st.cache_resource
def init_application_backend():
    """ Initializes PostgreSQL and pre-loads the Qdrant Vector client and Embeddings model """
    create_database()
    create_tables()
    
    embeddings_model = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
    
    qdrant_url = os.getenv("QDRANT_URL") or "http://localhost:6333"
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key, check_compatibility=False)
    collection_name = os.getenv("QDRANT_COLLECTION_NAME") or "investor_intelligence"
    
    return qdrant_client, embeddings_model, collection_name

client, embeddings, collection_name = init_application_backend()


# --- 2. Page Configuration ---
st.set_page_config(
    page_title="Investor Intelligence Platform",
    page_icon="📈",
    layout="wide"
)


# --- 3. Sidebar: Control Panel & Smart Ingestion ---
with st.sidebar:
    st.title("⚙️ Control Panel")
    st.write("Manage your RAG pipeline and upload new financial PDFs.")
    
    st.markdown("### 📁 Upload Financial Reports")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
    
    if uploaded_file is not None:
        if st.button("🔄 Process & Ingest Document", type="primary", use_container_width=True):
            with st.spinner("Processing PDF (Markdown + Semantic Chunking)..."):
                save_dir = Path("data/raw_pdfs")
                save_dir.mkdir(parents=True, exist_ok=True)
                file_path = save_dir / uploaded_file.name
                
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                try:
                    ingest_document(
                        pdf_path=str(file_path),
                        embeddings=embeddings,
                        client=client,
                        collection_name=collection_name
                    )
                    st.success(f"Processing complete for {uploaded_file.name}!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Ingestion Failed: {e}")
                    
    st.divider()
    st.info("💡 **Tip:** Uploaded PDFs are instantly checked against Qdrant to skip redundant vector computation.")


# --- 4. Main UI Header ---
st.title("📈 AI-Powered Investor Intelligence Platform")
st.caption("Advanced financial analytics and RAG-driven insights for smart investing.")


# --- 5. Financial Dashboard Section (PostgreSQL Analytics) ---
st.subheader("📊 Financial Dashboard")
metrics_data = get_metrics()

if metrics_data:
    col1, col2 = st.columns(2)
    with col1:
        total_companies = len(set(d.get('company') for d in metrics_data if 'company' in d))
        st.metric(label="Total Companies Tracked", value=total_companies)
    with col2:
        st.metric(label="Total Reports Processed", value=len(metrics_data))
        
    st.write("### Extracted Key Performance Indicators (KPIs)")
    df = pd.DataFrame(metrics_data)
    st.dataframe(df, use_container_width=True)
else:
    st.warning("No metrics found in PostgreSQL database. Please upload a report using the sidebar control panel.")

st.divider()


# --- 6. RAG Chatbot Section ---
st.subheader("💬 AI Investor Assistant")
st.write("Ask context-aware questions about the uploaded financial reports:")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_query := st.chat_input("E.g., What was the revenue growth or EBITDA margin for Apple?"):
    st.chat_message("user").markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    with st.spinner("Analyzing financial logs and generating response..."):
        ai_response = get_chat_response(user_query)
            
    with st.chat_message("assistant"):
        st.markdown(ai_response)
    st.session_state.messages.append({"role": "assistant", "content": ai_response})