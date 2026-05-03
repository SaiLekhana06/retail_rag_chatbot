"""
app.py — Streamlit Chat Interface for the Retail RAG Chatbot.
Run with: streamlit run app.py
Requires: python ingest.py must be run first to build the vector store.
"""
 
import os
from dotenv import load_dotenv

load_dotenv()  # 👈 THIS is missing

import streamlit as st
from retriever import Retriever
from generator import generate_response
 
# ── Page Configuration ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="Retail AI Assistant",
    page_icon="🛒",
    layout="centered",
    initial_sidebar_state="collapsed"
)
 
# ── Custom CSS Styling ─────────────────────────────────────────────────────
st.markdown("""
<style>
    .stChatMessage { border-radius: 12px; padding: 8px; }
    .main-header { text-align: center; color: #1F4E8C; font-size: 2rem; font-weight: bold; }
    .sub-header  { text-align: center; color: #666; font-size: 0.95rem; margin-bottom: 1rem; }
    .source-tag  { font-size: 0.75rem; color: #888; margin-top: 4px; }
</style>
""", unsafe_allow_html=True)
 
# ── Header ─────────────────────────────────────────────────────────────────
st.markdown('<p class="main-header">🛒 Retail Product Assistant</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Ask me anything about our products — features, price, warranty, availability.</p>', unsafe_allow_html=True)
st.divider()
 
# ── Load Retriever (cached — loaded only once across sessions) ─────────────
@st.cache_resource
def load_retriever():
    """Load and cache the Retriever to avoid re-loading FAISS index on every query."""
    try:
        return Retriever()
    except FileNotFoundError as e:
        st.error(f"Vector store not found: {e}")
        st.info("Please run `python ingest.py` in your terminal first.")
        st.stop()
 
retriever = load_retriever()
 
# ── Session State: Conversation History ────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": ("Hello! I am your Retail Product Assistant. "
                        "I can answer questions about product features, prices, "
                        "warranties, return policies, and availability. How can I help you today?"),
        }
    ]
 
# ── Sidebar: Settings & Example Queries ───────────────────────────────────
with st.sidebar:
    st.header("💡 Example Queries")
    example_queries = [
        "Does SmartPhone X1 support 5G?",
        "What is the warranty for LaptopPro Z5?",
        "Which products are under Rs.30,000?",
        "Compare SmartPhone X1 and X2",
        "What is the return policy for electronics?",
        "Tell me about earbuds",
        "Is the BudgetBook A3 in stock?",
    ]
    for eq in example_queries:
        if st.button(eq, use_container_width=True):
            st.session_state.pending_query = eq
 
    st.divider()
    st.header("⚙️ Settings")
    top_k = st.slider("Chunks to retrieve (Top-K)", min_value=1, max_value=5, value=3)
    show_sources = st.toggle("Show retrieved sources", value=False)
 
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = st.session_state.messages[:1]
        st.rerun()
 
# ── Display Conversation History ───────────────────────────────────────────
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="🛒" if message["role"] == "assistant" else "👤"):
        st.markdown(message["content"])
        if show_sources and "sources" in message:
            with st.expander("📂 Retrieved Sources", expanded=False):
                for src in message["sources"]:
                    st.markdown(f"**{src['name']}** — Distance: {src['distance']:.4f} | Source: `{src['source']}`")
                    st.text(src["text"][:300] + "...")
 
# ── Handle Example Query Click from Sidebar ────────────────────────────────
pending = st.session_state.pop("pending_query", None)
 
# ── Chat Input ─────────────────────────────────────────────────────────────
user_input = st.chat_input("Ask about our products... (e.g., Does X1 support 5G?)")
 
query = pending or user_input
 
if query:
    # Display user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user", avatar="👤"):
        st.markdown(query)
 
    # Generate response
    with st.chat_message("assistant", avatar="🛒"):
        with st.spinner("Searching product catalogue..."):
            # Step 1: Retrieve relevant product chunks
            retrieved = retriever.retrieve(query, top_k=top_k)
 
            # Step 2: Generate grounded LLM response
            answer = generate_response(retrieved, query)
 
        st.markdown(answer)
 
        # Optionally show sources inline
        if show_sources and retrieved:
            with st.expander("📂 Retrieved Sources", expanded=False):
                for src in retrieved:
                    st.markdown(f"**{src['name']}** — Distance: {src['distance']:.4f} | Source: `{src['source']}`")
                    st.text(src["text"][:300] + "...")
 
    # Save assistant message to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": retrieved
    })
