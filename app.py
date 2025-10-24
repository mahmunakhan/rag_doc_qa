import streamlit as st
import os
import sys
from utils.document_processor import DocumentProcessor
from utils.vector_store import VectorStore
from utils.rag_engine import RAGEngine
import time

# Page configuration
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS for both dark and light themes
st.markdown("""
<style>
    /* Main text colors that work in both themes */
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Chat message styling with high contrast */
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border: 1px solid #ddd;
    }
    
    .user-message {
        background-color: #e6f3ff;
        border-left: 4px solid #1f77b4;
        color: #000000 !important;
    }
    
    .assistant-message {
        background-color: #f8f9fa;
        border-left: 4px solid #ff6b6b;
        color: #000000 !important;
    }
    
    /* Ensure text is visible in both themes */
    .stChatMessage {
        color: #000000 !important;
    }
    
    .chat-message strong {
        color: #000000 !important;
    }
    
    .chat-message div {
        color: #000000 !important;
    }
    
    /* Source box styling */
    .source-box {
        background-color: #ffffff;
        border: 2px solid #dee2e6;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-top: 0.5rem;
        font-size: 0.875rem;
        color: #000000 !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .source-box strong {
        color: #1f77b4 !important;
    }
    
    /* Make sure all text in expanders is visible */
    .streamlit-expanderHeader {
        color: #000000 !important;
        font-weight: bold;
    }
    
    .streamlit-expanderContent {
        color: #000000 !important;
    }
    
    /* Metric cards styling */
    [data-testid="metric-container"] {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #ddd;
    }
    
    [data-testid="metric-container"] label {
        color: #000000 !important;
    }
    
    [data-testid="metric-container"] div {
        color: #000000 !important;
    }
    
    /* Force dark text in all containers */
    .stApp {
        color: #000000 !important;
    }
    
    /* Sidebar styling */
    .css-1d391kg, .css-1lcbmhc {
        background-color: #f0f2f6;
    }
    
    /* Make uploaded file names visible */
    .uploadedFile {
        color: #000000 !important;
    }
    
    /* Button styling */
    .stButton button {
        background-color: #1f77b4;
        color: white;
    }
    
    .stButton button:hover {
        background-color: #1668a1;
        color: white;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# CRITICAL: Cache expensive resources to prevent reloading
# ============================================================

@st.cache_resource
def get_vector_store():
    """
    Initialize and cache vector store
    This prevents reloading embeddings and reranker on every interaction
    """
    print("🔄 Initializing Vector Store (will be cached)...")
    return VectorStore()


@st.cache_resource
def get_rag_engine():
    """
    Initialize and cache RAG engine
    This prevents reloading the LLM on every interaction
    """
    print("🔄 Initializing RAG Engine (will be cached)...")
    return RAGEngine()


# ============================================================
# Main RAG Chatbot Class
# ============================================================

class RAGChatbot:
    def __init__(self):
        # Document processor is lightweight, no need to cache
        self.document_processor = DocumentProcessor()
        
        # Use cached versions of heavy components
        self.vector_store = get_vector_store()
        self.rag_engine = get_rag_engine()
        
        # Initialize session state
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "documents_processed" not in st.session_state:
            st.session_state.documents_processed = False
        if "collection_info" not in st.session_state:
            st.session_state.collection_info = self.vector_store.get_collection_info()
    
    def process_uploaded_files(self, uploaded_files):
        """Process all uploaded files"""
        if not uploaded_files:
            return
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        all_documents = []
        total_files = len(uploaded_files)
        
        for i, uploaded_file in enumerate(uploaded_files):
            status_text.text(f"Processing {uploaded_file.name}...")
            
            try:
                documents = self.document_processor.process_uploaded_file(uploaded_file)
                all_documents.extend(documents)
                st.success(f"✅ Processed {uploaded_file.name} - {len(documents)} chunks")
                
            except Exception as e:
                st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
            
            progress_bar.progress((i + 1) / total_files)
        
        # Add documents to vector store
        if all_documents:
            status_text.text("Adding documents to vector database...")
            self.vector_store.add_documents(all_documents)
            
            # Update collection info
            st.session_state.collection_info = self.vector_store.get_collection_info()
            st.session_state.documents_processed = True
            
            status_text.text("✅ All documents processed successfully!")
            st.balloons()
        
        progress_bar.empty()
        status_text.empty()
    
    def display_chat_interface(self):
        """Display the main chat interface"""
        st.markdown('<div class="main-header">🤖 RAG Chatbot</div>', unsafe_allow_html=True)
        
        # Display collection info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Documents", st.session_state.collection_info["total_documents"])
        with col2:
            st.metric("Collection", st.session_state.collection_info["collection_name"])
        with col3:
            status = "Ready" if st.session_state.documents_processed else "No Documents"
            st.metric("Status", status)
        
        # Debug info in sidebar
        if st.sidebar.checkbox("Show Debug Info"):
            st.sidebar.write("Session state:", st.session_state)
            if st.session_state.messages:
                st.sidebar.write("Last message:", st.session_state.messages[-1])
        
        # Chat messages display
        chat_container = st.container()
        
        with chat_container:
            for message in st.session_state.messages:
                if message["role"] == "user":
                    st.markdown(f"""
                    <div class="chat-message user-message">
                        <strong>You:</strong> {message["content"]}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    # Assistant message with response and sources
                    st.markdown(f"""
                    <div class="chat-message assistant-message">
                        <strong>Assistant:</strong> {message["content"]}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Display sources if available
                    if "sources" in message and message["sources"]:
                        with st.expander("📚 Sources Used (Click to Expand)"):
                            for i, source in enumerate(message["sources"]):
                                # Get similarity score safely
                                similarity_score = source.get('similarity_score', 0)
                                score_display = f"{similarity_score:.3f}" if isinstance(similarity_score, (int, float)) else 'N/A'
                                
                                # Find the actual content from search results
                                content = ""
                                if "search_results" in message and i < len(message["search_results"]):
                                    content = message["search_results"][i]["content"][:200] + "..." if len(message["search_results"][i]["content"]) > 200 else message["search_results"][i]["content"]
                                
                                st.markdown(f"""
                                <div class="source-box">
                                    <strong>Source {i+1}:</strong> {source['source']} (Chunk {source['chunk_id'] + 1})<br>
                                    <strong>Similarity Score:</strong> {score_display}<br>
                                    <strong>Content:</strong> {content}
                                </div>
                                """, unsafe_allow_html=True)
        
        # Chat input
        st.markdown("---")
        user_input = st.chat_input("Ask a question about your documents...")
        
        if user_input:
            # Add user message to chat
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            # Generate response
            with st.spinner("🔍 Searching documents and generating response..."):
                try:
                    # Search for relevant documents
                    search_results = self.vector_store.search(user_input, n_results=3)
                    st.sidebar.write(f"Debug: Found {len(search_results)} search results")
                    
                    if search_results:
                        st.sidebar.write(f"Debug: Top result similarity: {search_results[0].get('similarity_score', 'N/A')}")
                    
                    # Generate RAG response
                    rag_result = self.rag_engine.rag_query(user_input, search_results)
                    
                    # Add assistant response to chat
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": rag_result["response"],
                        "sources": rag_result["sources"],
                        "search_results": rag_result["search_results"]
                    })
                    
                    # Rerun to update display
                    st.rerun()
                    
                except Exception as e:
                    error_msg = f"❌ Error generating response: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": error_msg
                    })
                    st.rerun()
    
    def display_sidebar(self):
        """Display the sidebar with file upload and controls"""
        st.sidebar.title("📁 Document Management")
        
        # File upload
        uploaded_files = st.sidebar.file_uploader(
            "Upload Documents",
            type=["pdf", "txt", "md"],
            accept_multiple_files=True,
            help="Upload PDF, TXT, or Markdown files"
        )
        
        if uploaded_files:
            if st.sidebar.button("Process Documents", type="primary"):
                self.process_uploaded_files(uploaded_files)
        
        st.sidebar.markdown("---")
        
        # Collection management
        st.sidebar.subheader("Database Management")
        
        if st.sidebar.button("Clear All Documents"):
            self.vector_store.clear_collection()
            st.session_state.collection_info = self.vector_store.get_collection_info()
            st.session_state.documents_processed = False
            st.session_state.messages = []
            st.sidebar.success("🗑️ All documents cleared!")
            st.rerun()
        
        # Display current documents info
        st.sidebar.markdown("---")
        st.sidebar.subheader("Current Documents")
        st.sidebar.write(f"📊 Total chunks: **{st.session_state.collection_info['total_documents']}**")
        
        # Debug information
        st.sidebar.markdown("---")
        st.sidebar.subheader("Debug Info")
        if st.sidebar.button("Show Session State"):
            st.sidebar.write(st.session_state)
        
        # Instructions
        st.sidebar.markdown("---")
        st.sidebar.subheader("Instructions")
        st.sidebar.markdown("""
        1. 📄 Upload PDF, TXT, or MD files
        2. ⚡ Click 'Process Documents'  
        3. 💬 Ask questions in the chat
        4. 📚 View sources for each response
        """)


# ============================================================
# Main Application Entry Point
# ============================================================

def main():
    """Main application function"""
    # Initialize chatbot (uses cached models)
    chatbot = RAGChatbot()
    
    # Display sidebar
    chatbot.display_sidebar()
    
    # Display main chat interface
    chatbot.display_chat_interface()


if __name__ == "__main__":
    main()