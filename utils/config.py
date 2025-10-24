"""
RAG System Configuration
Centralized configuration for all RAG components
"""

# Document Processing Configuration
DOCUMENT_PROCESSING = {
    # Chunking strategy
    "chunk_size": 512,              # Tokens (not characters)
    "chunk_overlap": 128,           # 25% overlap recommended
    "use_semantic_chunking": True,  # Respect paragraph/sentence boundaries
    
    # Text cleaning
    "remove_special_chars": True,
    "normalize_whitespace": True,
    "preserve_formatting": True,
}

# Embedding Configuration
EMBEDDING_CONFIG = {
    # Model selection (choose based on speed vs. quality tradeoff)
    # Options:
    # - "sentence-transformers/all-MiniLM-L6-v2" (384d, fast, baseline)
    # - "sentence-transformers/all-mpnet-base-v2" (768d, balanced, recommended)
    # - "BAAI/bge-large-en-v1.5" (1024d, high quality, slower)
    # - "intfloat/e5-large-v2" (1024d, high quality, slower)
    
    "model_name": "sentence-transformers/all-mpnet-base-v2",
    "normalize_embeddings": True,   # Critical for cosine similarity
    "batch_size": 32,                # Batch size for embedding generation
    "show_progress": True,
}

# Retrieval Configuration
RETRIEVAL_CONFIG = {
    # Basic retrieval
    "top_k_candidates": 20,          # Initial retrieval (before reranking)
    "final_top_k": 5,                # Final results to use for generation
    
    # Reranking (highly recommended for accuracy)
    "use_reranker": True,
    "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    # Alternative rerankers:
    # - "cross-encoder/ms-marco-electra-base" (better quality, slower)
    # - "cross-encoder/ms-marco-MiniLM-L-12-v2" (balanced)
    
    # Maximum Marginal Relevance (for diversity)
    "use_mmr": True,
    "mmr_lambda": 0.5,               # 0=diverse, 1=relevant
    
    # Hybrid search (semantic + keyword)
    "use_hybrid": False,             # Enable for technical/exact match queries
    "semantic_weight": 0.7,
    "keyword_weight": 0.3,
    
    # Metadata filtering
    "enable_metadata_filters": True,
    "filter_by_source": None,        # None or list of sources
    "filter_by_date": None,          # None or date range
}

# Vector Store Configuration
VECTOR_STORE_CONFIG = {
    "persist_directory": "./chroma_db",
    "collection_name": "document_embeddings",
    "distance_metric": "cosine",     # Options: cosine, l2, ip
    "batch_size": 100,               # Batch size for adding documents
}

# LLM Configuration
LLM_CONFIG = {
    # Model selection
    # API-based (best quality):
    # - "gpt-4" / "gpt-3.5-turbo" (OpenAI)
    # - "claude-3-opus" / "claude-3-sonnet" (Anthropic)
    # - "gemini-pro" (Google)
    
    # Local models (good quality):
    # - "mistralai/Mistral-7B-Instruct-v0.2" (recommended)
    # - "meta-llama/Llama-2-7b-chat-hf"
    # - "meta-llama/Llama-2-13b-chat-hf" (better but needs more GPU)
    # - "HuggingFaceH4/zephyr-7b-beta"
    
    # Lightweight (fast):
    # - "microsoft/phi-2"
    # - "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    "model_name": "mistralai/Mistral-7B-Instruct-v0.2",
    "use_api": False,                # True for API-based models
    "api_key": None,                 # Set if using API
    
    # Generation parameters
    "max_new_tokens": 256,
    "temperature": 0.7,              # 0=deterministic, 1=creative
    "top_p": 0.9,                    # Nucleus sampling
    "top_k": 50,
    "repetition_penalty": 1.1,
    "no_repeat_ngram_size": 3,
    
    # Context management
    "max_context_length": 2048,      # Maximum context window
    "max_context_tokens": 1500,      # Tokens reserved for context
}

# Prompt Templates
PROMPT_TEMPLATES = {
    "system_prompt": """You are a helpful AI assistant that answers questions based on provided context.

Your role is to:
1. Carefully read the provided sources
2. Answer the question accurately using information from the sources
3. Reference specific sources when making claims (e.g., "According to Source 1...")
4. If the sources don't contain enough information, say so clearly
5. Be concise but comprehensive
6. Do not make up information not present in the sources""",
    
    "user_prompt_template": """Context from documents:
{context}

Question: {query}

Instructions: Based on the context above, provide a clear and accurate answer. Reference the sources you use.

Answer:""",
    
    "no_context_prompt": """I don't have relevant documents in my knowledge base to answer this question accurately. Please provide relevant documents or rephrase your query."""
}

# Evaluation Configuration
EVALUATION_CONFIG = {
    # Metrics to compute
    "compute_recall": True,
    "compute_mrr": True,
    "compute_ndcg": True,
    "compute_bert_score": False,     # Requires additional package
    
    # Confidence scoring
    "enable_confidence": True,
    "confidence_threshold": 0.5,     # Below this, flag for review
    "confidence_weights": {
        "retrieval_score": 0.5,
        "completeness": 0.3,
        "keyword_overlap": 0.2
    },
}

# Logging Configuration
LOGGING_CONFIG = {
    "log_queries": True,
    "log_retrieval": True,
    "log_generation": True,
    "log_metrics": True,
    "log_file": "./rag_system.log",
    "log_level": "INFO",             # DEBUG, INFO, WARNING, ERROR
}

# Performance Configuration
PERFORMANCE_CONFIG = {
    # GPU settings
    "use_gpu": True,                 # Auto-detect if None
    "gpu_memory_fraction": 0.8,      # Fraction of GPU memory to use
    
    # Caching
    "cache_embeddings": True,
    "cache_dir": "./cache",
    
    # Batch processing
    "enable_batching": True,
    "batch_size": 8,
    
    # Timeouts
    "retrieval_timeout": 5.0,        # Seconds
    "generation_timeout": 30.0,      # Seconds
}

# Experimental Features
EXPERIMENTAL_CONFIG = {
    "use_query_expansion": False,    # Expand query with synonyms
    "use_self_query": False,         # Let model extract metadata filters
    "use_parent_document": False,    # Retrieve + expand to parent doc
    "use_ensemble": False,           # Combine multiple retrievers
}

# Application Configuration
APP_CONFIG = {
    "app_name": "RAG Chatbot",
    "version": "2.0",
    "debug_mode": False,
    "show_sources": True,
    "show_scores": True,
    "max_chat_history": 10,
}


# Helper function to validate configuration
def validate_config():
    """Validate configuration settings"""
    issues = []
    
    # Check chunk size
    if DOCUMENT_PROCESSING["chunk_size"] < 100:
        issues.append("chunk_size too small (< 100)")
    if DOCUMENT_PROCESSING["chunk_size"] > 2000:
        issues.append("chunk_size too large (> 2000)")
    
    # Check retrieval
    if RETRIEVAL_CONFIG["final_top_k"] > RETRIEVAL_CONFIG["top_k_candidates"]:
        issues.append("final_top_k cannot exceed top_k_candidates")
    
    # Check weights
    weight_sum = (RETRIEVAL_CONFIG["semantic_weight"] + 
                  RETRIEVAL_CONFIG["keyword_weight"])
    if abs(weight_sum - 1.0) > 0.01:
        issues.append(f"Hybrid weights don't sum to 1.0 (sum={weight_sum})")
    
    if issues:
        print("Configuration Issues:")
        for issue in issues:
            print(f"  ⚠️  {issue}")
        return False
    
    print("✅ Configuration validated successfully")
    return True


# Print current configuration
def print_config():
    """Print current configuration"""
    print("\n" + "="*60)
    print("RAG SYSTEM CONFIGURATION")
    print("="*60)
    
    print("\n📄 Document Processing:")
    print(f"  Chunk Size: {DOCUMENT_PROCESSING['chunk_size']} tokens")
    print(f"  Chunk Overlap: {DOCUMENT_PROCESSING['chunk_overlap']} tokens")
    print(f"  Semantic Chunking: {DOCUMENT_PROCESSING['use_semantic_chunking']}")
    
    print("\n🔍 Embeddings:")
    print(f"  Model: {EMBEDDING_CONFIG['model_name']}")
    print(f"  Normalize: {EMBEDDING_CONFIG['normalize_embeddings']}")
    
    print("\n📊 Retrieval:")
    print(f"  Initial Candidates: {RETRIEVAL_CONFIG['top_k_candidates']}")
    print(f"  Final Results: {RETRIEVAL_CONFIG['final_top_k']}")
    print(f"  Use Reranker: {RETRIEVAL_CONFIG['use_reranker']}")
    if RETRIEVAL_CONFIG['use_reranker']:
        print(f"  Reranker Model: {RETRIEVAL_CONFIG['reranker_model']}")
    print(f"  Use MMR: {RETRIEVAL_CONFIG['use_mmr']}")
    print(f"  Use Hybrid: {RETRIEVAL_CONFIG['use_hybrid']}")
    
    print("\n🤖 LLM:")
    print(f"  Model: {LLM_CONFIG['model_name']}")
    print(f"  Max Tokens: {LLM_CONFIG['max_new_tokens']}")
    print(f"  Temperature: {LLM_CONFIG['temperature']}")
    
    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    print_config()
    validate_config()