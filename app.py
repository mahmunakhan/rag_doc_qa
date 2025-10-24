import streamlit as st
import os
import sys
from utils.document_processor import DocumentProcessor
from utils.vector_store import VectorStore
from utils.rag_engine import RAGEngine
from utils.highlighter import DocumentHighlighter, highlight_in_chunk
import time
import tempfile

# Page configuration
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    
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
    
    .highlight-container {
        background: #fffef0;
        border: 2px solid #ffd700;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-top: 0.5rem;
    }
    
    mark {
        background-color: yellow;
        padding: 2px 4px;
        font-weight: bold;
        border-radius: 2px;
    }
    
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


@st.cache_resource
def get_vector_store():
    """Initialize and cache vector store"""
    print("📄 Initializing Vector Store (will be cached)...")
    return VectorStore()


@st.cache_resource
def get_rag_engine():
    """Initialize and cache RAG engine"""
    print("📄 Initializing RAG Engine (will be cached)...")
    return RAGEngine()


@st.cache_resource
def get_highlighter():
    """Initialize and cache document highlighter"""
    print("📄 Initializing Document Highlighter (will be cached)...")
    return DocumentHighlighter()


class RAGChatbot:
    def __init__(self):
        self.document_processor = DocumentProcessor()
        self.vector_store = get_vector_store()
        self.rag_engine = get_rag_engine()
        self.highlighter = get_highlighter()
        
        # Initialize session state
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "documents_processed" not in st.session_state:
            st.session_state.documents_processed = False
        if "collection_info" not in st.session_state:
            st.session_state.collection_info = self.vector_store.get_collection_info()
        if "uploaded_files_paths" not in st.session_state:
            st.session_state.uploaded_files_paths = {}  # filename -> temp_path mapping
    
    def process_uploaded_files(self, uploaded_files):
        """Process all uploaded files and store their paths"""
        if not uploaded_files:
            return
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        all_documents = []
        total_files = len(uploaded_files)
        
        for i, uploaded_file in enumerate(uploaded_files):
            status_text.text(f"Processing {uploaded_file.name}...")
            
            try:
                # Save file to temp location
                temp_dir = tempfile.gettempdir()
                temp_path = os.path.join(temp_dir, f"rag_{uploaded_file.name}")
                
                with open(temp_path, 'wb') as f:
                    f.write(uploaded_file.getvalue())
                
                # Store the path for later highlighting
                st.session_state.uploaded_files_paths[uploaded_file.name] = temp_path
                
                # Process the document
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
            
            st.session_state.collection_info = self.vector_store.get_collection_info()
            st.session_state.documents_processed = True
            
            status_text.text("✅ All documents processed successfully!")
            st.balloons()
        
        progress_bar.empty()
        status_text.empty()
    
    def display_chat_interface(self):
        """Display the main chat interface"""
        st.markdown('<div class="main-header">🤖 RAG Chatbot with Smart Highlighting</div>', unsafe_allow_html=True)
        
        # Display collection info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Documents", st.session_state.collection_info["total_documents"])
        with col2:
            st.metric("Collection", st.session_state.collection_info["collection_name"])
        with col3:
            status = "Ready" if st.session_state.documents_processed else "No Documents"
            st.metric("Status", status)
        
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
                    # Assistant message
                    st.markdown(f"""
                    <div class="chat-message assistant-message">
                        <strong>Assistant:</strong> {message["content"]}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Display sources with highlighted content
                    if "sources" in message and message["sources"]:
                        with st.expander("📚 Sources Used (Click to Expand)"):
                            for i, source in enumerate(message["sources"]):
                                
                                similarity_score = 0
                                if "search_results" in message and i < len(message["search_results"]):
                                    similarity_score = message["search_results"][i].get('similarity_score', 0)
                                score_display = f"{similarity_score:.3f}" if isinstance(similarity_score, (int, float)) else 'N/A'
                                # Get content and apply smart highlighting
                                content = ""
                                if "search_results" in message and i < len(message["search_results"]):
                                    raw_content = message["search_results"][i]["content"]
                                    
                                    # Always attempt highlighting - function will only highlight matching terms
                                    if "answer_text" in message:
                                        content = highlight_in_chunk(raw_content, message["answer_text"])
                                    else:
                                        content = raw_content
                                    
                                    # Truncate if too long (but preserve HTML tags)
                                    if len(content) > 800:
                                        # Find a good breaking point
                                        content = content[:800]
                                        # Try to end at a sentence
                                        last_period = content.rfind('.')
                                        if last_period > 600:
                                            content = content[:last_period + 1]
                                        content += "..."
                                
                                st.markdown(f"""
                                <div class="source-box">
                                    <strong>Source {i+1}:</strong> {source['source']} (Chunk {source['chunk_id'] + 1})<br>
                                    <strong>Similarity Score:</strong> {score_display}<br>
                                    <div style="background: #f8f9fa; padding: 10px; border-radius: 5px; margin-top: 5px;">
                                        <strong>Content:</strong><br>
                                        {content}
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                    
                    # Show document-level highlighting if available
                    if "show_highlighting" in message and message["show_highlighting"]:
                        with st.expander("🔍 View Highlighted Evidence in Original Document"):
                            if "highlight_results" in message:
                                highlight_results = message["highlight_results"]
                                
                                if isinstance(highlight_results, list):  # PDF case
                                    if highlight_results:
                                        for img, page_number in highlight_results:
                                            st.image(
                                                img,
                                                caption=f"📄 Highlighted Page {page_number}",
                                                use_column_width=True
                                            )
                                    else:
                                        st.info("No exact matches found in the original document for highlighting.")
                                
                                elif isinstance(highlight_results, str):  # Text/DOCX/MD case
                                    st.markdown(highlight_results, unsafe_allow_html=True)
                                else:
                                    st.warning("Could not generate highlights for this document type.")
                            else:
                                st.info("Highlighting not available for this response.")
        
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
                    search_results = self.vector_store.search(user_input, n_results=5)
                    
                    if search_results:
                        # Generate RAG response
                        rag_result = self.rag_engine.rag_query(user_input, search_results)
                        
                        # Extract key phrases from answer for highlighting (first 150 chars)
                        answer_text = rag_result["response"]
                        highlight_text = answer_text[:150]  # Only use first part for highlighting
                        
                        # Try to highlight in the original document
                        highlight_results = None
                        if search_results and search_results[0]["metadata"].get("source"):
                            source_filename = search_results[0]["metadata"]["source"]
                            
                            # Check if we have the file path
                            if source_filename in st.session_state.uploaded_files_paths:
                                file_path = st.session_state.uploaded_files_paths[source_filename]
                                
                                try:
                                    # Attempt to highlight
                                    highlight_results = self.highlighter.highlight_document(
                                        file_path,
                                        user_input,  # search sentence
                                        highlight_text  # Use shortened text
                                    )
                                except Exception as e:
                                    print(f"Highlighting error: {e}")
                                    highlight_results = None
                        
                        # Add assistant response to chat
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": rag_result["response"],
                            "sources": rag_result["sources"],
                            "search_results": rag_result["search_results"],
                            "answer_text": highlight_text,  # Store shortened version
                            "show_highlighting": highlight_results is not None,
                            "highlight_results": highlight_results
                        })
                    else:
                        # No results found
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": "I couldn't find relevant information in the uploaded documents to answer your question. Please try rephrasing or upload more relevant documents."
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
            # Clean up temp files
            for filename, temp_path in st.session_state.uploaded_files_paths.items():
                try:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                except:
                    pass
            
            self.vector_store.clear_collection()
            st.session_state.collection_info = self.vector_store.get_collection_info()
            st.session_state.documents_processed = False
            st.session_state.messages = []
            st.session_state.uploaded_files_paths = {}
            st.sidebar.success("🗑️ All documents cleared!")
            st.rerun()
        
        # Display current documents info
        st.sidebar.markdown("---")
        st.sidebar.subheader("Current Documents")
        st.sidebar.write(f"📊 Total chunks: **{st.session_state.collection_info['total_documents']}**")
        st.sidebar.write(f"📁 Files uploaded: **{len(st.session_state.uploaded_files_paths)}**")
        
        # Instructions
        st.sidebar.markdown("---")
        st.sidebar.subheader("✨ Features")
        st.sidebar.markdown("""
        - 🔍 **Smart Search**: Advanced semantic search
        - 💡 **Reranking**: Best results first
        - 📚 **Source Citations**: See where answers come from
        - 🎯 **Highlighting**: View evidence in context
        - ⚡ **Fast**: Cached models for speed
        """)
        
        st.sidebar.markdown("---")
        st.sidebar.subheader("📖 How to Use")
        st.sidebar.markdown("""
        1. 📄 Upload documents (PDF/TXT/MD)
        2. ⚡ Click 'Process Documents'  
        3. 💬 Ask questions in the chat
        4. 📚 Click 'Sources Used' to see evidence
        5. 🔍 Click 'View Highlighted Evidence' to see exact locations
        """)


def main():
    """Main application function"""
    chatbot = RAGChatbot()
    chatbot.display_sidebar()
    chatbot.display_chat_interface()


if __name__ == "__main__":
    main()