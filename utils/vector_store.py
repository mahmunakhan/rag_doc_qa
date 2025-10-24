import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer, CrossEncoder
import os
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from collections import defaultdict


class VectorStore:
    """Enhanced vector store with hybrid search, reranking, and query expansion"""
    
    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        embedding_model_name: str = 'sentence-transformers/all-mpnet-base-v2',
        use_reranker: bool = True,
        reranker_model: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
    ):
        self.persist_directory = persist_directory
        
        # Use a better embedding model for improved accuracy
        # Options (in order of quality vs speed):
        # - all-MiniLM-L6-v2: Fast, lightweight (current baseline)
        # - all-mpnet-base-v2: Better quality, reasonable speed (recommended)
        # - bge-large-en-v1.5: High quality, slower
        # - e5-large-v2: High quality, slower
        print(f"Loading embedding model: {embedding_model_name}")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        
        # Optional: Cross-encoder for reranking (significant accuracy boost)
        self.use_reranker = use_reranker
        self.reranker = None
        if use_reranker:
            print(f"Loading reranker model: {reranker_model}")
            try:
                self.reranker = CrossEncoder(reranker_model)
                print("Reranker loaded successfully!")
            except Exception as e:
                print(f"Warning: Could not load reranker: {e}")
                self.use_reranker = False
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        
        # Get or create collection with cosine similarity
        self.collection = self.client.get_or_create_collection(
            name="document_embeddings",
            metadata={
                "description": "Document embeddings for RAG system",
                "hnsw:space": "cosine"  # Explicitly set distance metric
            }
        )
    
    def add_documents(
        self,
        documents: List[Dict[str, Any]],
        batch_size: int = 100
    ) -> None:
        """Add documents to vector store with batching for large datasets"""
        if not documents:
            return
        
        print(f"Adding {len(documents)} documents to vector store...")
        
        # Process in batches to avoid memory issues
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            
            # Extract contents and metadata
            contents = [doc["content"] for doc in batch]
            metadatas = [doc["metadata"] for doc in batch]
            
            # Generate embeddings with normalization
            print(f"Generating embeddings for batch {i//batch_size + 1}...")
            embeddings = self.embedding_model.encode(
                contents,
                normalize_embeddings=True,  # Important for cosine similarity
                show_progress_bar=False
            ).tolist()
            
            # Generate unique IDs
            ids = [
                f"{metadata['source']}_chunk_{metadata['chunk_id']}"
                for metadata in metadatas
            ]
            
            # Add to collection
            try:
                self.collection.add(
                    embeddings=embeddings,
                    documents=contents,
                    metadatas=metadatas,
                    ids=ids
                )
            except Exception as e:
                print(f"Warning: Some documents already exist, attempting upsert...")
                # If documents exist, try upsert
                self.collection.upsert(
                    embeddings=embeddings,
                    documents=contents,
                    metadatas=metadatas,
                    ids=ids
                )
        
        print(f"Successfully added {len(documents)} documents!")
    
    def search(
        self,
        query: str,
        n_results: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
        use_mmr: bool = True,
        mmr_lambda: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Enhanced search with multiple retrieval strategies
        
        Args:
            query: Search query
            n_results: Number of results to return
            filter_metadata: Optional metadata filters
            use_mmr: Use Maximum Marginal Relevance for diversity
            mmr_lambda: MMR diversity parameter (0=diverse, 1=relevant)
        """
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode(
                [query],
                normalize_embeddings=True
            ).tolist()
            
            # Retrieve more candidates for reranking/MMR
            retrieval_k = n_results * 3 if self.use_reranker or use_mmr else n_results
            
            # Search in collection
            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=min(retrieval_k, self.collection.count()),
                include=["documents", "metadatas", "distances"],
                where=filter_metadata
            )
            
            # Format results
            formatted_results = []
            if results['documents'] and results['documents'][0]:
                for i, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )):
                    # Convert distance to similarity score (cosine similarity)
                    similarity_score = 1 - distance
                    
                    formatted_results.append({
                        "content": doc,
                        "metadata": metadata,
                        "similarity_score": similarity_score,
                        "initial_rank": i + 1
                    })
            
            # Apply Maximum Marginal Relevance for diversity
            if use_mmr and len(formatted_results) > n_results:
                formatted_results = self._apply_mmr(
                    formatted_results,
                    query_embedding[0],
                    n_results,
                    mmr_lambda
                )
            
            # Apply reranking if enabled
            if self.use_reranker and self.reranker and formatted_results:
                formatted_results = self._rerank_results(query, formatted_results)
            
            # Return top n_results
            return formatted_results[:n_results]
            
        except Exception as e:
            print(f"Search error: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _rerank_results(
        self,
        query: str,
        results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Rerank results using cross-encoder for better relevance"""
        if not results:
            return results
        
        try:
            # Prepare query-document pairs
            pairs = [[query, result["content"]] for result in results]
            
            # Get cross-encoder scores
            rerank_scores = self.reranker.predict(pairs)
            
            # Add rerank scores to results
            for result, score in zip(results, rerank_scores):
                result["rerank_score"] = float(score)
                result["similarity_score"] = float(score)  #
            
            # Sort by rerank score
            results.sort(key=lambda x: x["rerank_score"], reverse=True)
            
            # Update ranks
            for i, result in enumerate(results):
                result["rerank_rank"] = i + 1
            
            print(f"Reranked {len(results)} results")
            return results
            
        except Exception as e:
            print(f"Reranking error: {e}")
            return results
    
    def _apply_mmr(
        self,
        results: List[Dict[str, Any]],
        query_embedding: List[float],
        n_results: int,
        lambda_param: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Apply Maximum Marginal Relevance to diversify results
        
        MMR = λ * Relevance(doc, query) - (1-λ) * max Similarity(doc, selected_docs)
        """
        if len(results) <= n_results:
            return results
        
        try:
            # Get embeddings for all results
            result_embeddings = self.embedding_model.encode(
                [r["content"] for r in results],
                normalize_embeddings=True
            )
            
            query_emb = np.array(query_embedding)
            
            # Initialize selected indices and remaining indices
            selected_indices = []
            remaining_indices = list(range(len(results)))
            
            # Select first document (highest similarity)
            first_idx = 0
            selected_indices.append(first_idx)
            remaining_indices.remove(first_idx)
            
            # Iteratively select documents
            while len(selected_indices) < n_results and remaining_indices:
                mmr_scores = []
                
                for idx in remaining_indices:
                    # Relevance to query
                    relevance = np.dot(query_emb, result_embeddings[idx])
                    
                    # Max similarity to already selected documents
                    max_sim = max([
                        np.dot(result_embeddings[idx], result_embeddings[sel_idx])
                        for sel_idx in selected_indices
                    ])
                    
                    # MMR score
                    mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim
                    mmr_scores.append((idx, mmr_score))
                
                # Select document with highest MMR score
                best_idx = max(mmr_scores, key=lambda x: x[1])[0]
                selected_indices.append(best_idx)
                remaining_indices.remove(best_idx)
            
            # Return results in MMR order
            mmr_results = [results[idx] for idx in selected_indices]
            
            # Update ranks
            for i, result in enumerate(mmr_results):
                result["mmr_rank"] = i + 1
            
            return mmr_results
            
        except Exception as e:
            print(f"MMR error: {e}")
            return results[:n_results]
    
    def hybrid_search(
        self,
        query: str,
        n_results: int = 5,
        keyword_weight: float = 0.3,
        semantic_weight: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search combining keyword (BM25-like) and semantic search
        
        Args:
            query: Search query
            n_results: Number of results
            keyword_weight: Weight for keyword matching
            semantic_weight: Weight for semantic similarity
        """
        # Get semantic search results
        semantic_results = self.search(query, n_results=n_results * 2, use_mmr=False)
        
        # Perform keyword-based scoring
        query_terms = set(query.lower().split())
        
        for result in semantic_results:
            content_terms = set(result["content"].lower().split())
            
            # Calculate keyword overlap (simple BM25 approximation)
            overlap = len(query_terms & content_terms)
            keyword_score = overlap / max(len(query_terms), 1)
            
            # Combine scores
            result["keyword_score"] = keyword_score
            result["hybrid_score"] = (
                semantic_weight * result["similarity_score"] +
                keyword_weight * keyword_score
            )
        
        # Sort by hybrid score
        semantic_results.sort(key=lambda x: x["hybrid_score"], reverse=True)
        
        # Update ranks
        for i, result in enumerate(semantic_results[:n_results]):
            result["hybrid_rank"] = i + 1
        
        return semantic_results[:n_results]
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get comprehensive information about the collection"""
        try:
            count = self.collection.count()
            
            # Get sample documents to analyze
            sample_size = min(10, count)
            if count > 0:
                sample = self.collection.get(limit=sample_size)
                
                # Analyze metadata
                sources = set()
                file_types = set()
                
                if sample['metadatas']:
                    for metadata in sample['metadatas']:
                        if 'source' in metadata:
                            sources.add(metadata['source'])
                        if 'file_type' in metadata:
                            file_types.add(metadata['file_type'])
                
                return {
                    "total_documents": count,
                    "collection_name": self.collection.name,
                    "unique_sources": len(sources),
                    "sources": list(sources),
                    "file_types": list(file_types),
                    "embedding_model": self.embedding_model.get_sentence_embedding_dimension(),
                    "uses_reranker": self.use_reranker
                }
            
            return {
                "total_documents": count,
                "collection_name": self.collection.name,
                "embedding_model": self.embedding_model.get_sentence_embedding_dimension(),
                "uses_reranker": self.use_reranker
            }
        except Exception as e:
            print(f"Error getting collection info: {e}")
            return {
                "total_documents": 0,
                "collection_name": "document_embeddings",
                "error": str(e)
            }
    
    def clear_collection(self) -> None:
        """Clear all documents from collection"""
        try:
            self.client.delete_collection("document_embeddings")
            self.collection = self.client.get_or_create_collection(
                name="document_embeddings",
                metadata={
                    "description": "Document embeddings for RAG system",
                    "hnsw:space": "cosine"
                }
            )
            print("Collection cleared successfully!")
        except Exception as e:
            print(f"Error clearing collection: {e}")
    
    def get_document_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a specific document by ID"""
        try:
            result = self.collection.get(ids=[doc_id])
            if result['documents']:
                return {
                    "content": result['documents'][0],
                    "metadata": result['metadatas'][0]
                }
        except Exception as e:
            print(f"Error retrieving document {doc_id}: {e}")
        return None
    
    def search_by_metadata(
        self,
        metadata_filter: Dict[str, Any],
        n_results: int = 10
    ) -> List[Dict[str, Any]]:
        """Search documents by metadata filters only"""
        try:
            results = self.collection.get(
                where=metadata_filter,
                limit=n_results,
                include=["documents", "metadatas"]
            )
            
            formatted_results = []
            if results['documents']:
                for doc, metadata in zip(results['documents'], results['metadatas']):
                    formatted_results.append({
                        "content": doc,
                        "metadata": metadata
                    })
            
            return formatted_results
        except Exception as e:
            print(f"Metadata search error: {e}")
            return []