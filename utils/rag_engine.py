from typing import List, Dict, Any, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import re


class RAGEngine:
    """Enhanced RAG engine with better models, prompting, and answer generation"""
    
    def __init__(
        self,
        model_name: str = "google/gemma-2b-it",  # Better base model
        use_local_model: bool = False,
        api_based: bool = False,
        max_context_length: int = 2048
    ):
        """
        Initialize RAG Engine
        
        Recommended models (in order of preference):
        1. API-based (best quality):
           - OpenAI GPT-4/GPT-3.5-turbo
           - Anthropic Claude
           - Google PaLM
        
        2. Local models (good quality):
           - meta-llama/Llama-2-7b-chat-hf (7B, good balance)
           - meta-llama/Llama-2-13b-chat-hf (13B, better quality)
           - mistralai/Mistral-7B-Instruct-v0.2 (7B, excellent)
           - HuggingFaceH4/zephyr-7b-beta (7B, good for chat)
        
        3. Smaller models (faster, lower quality):
           - microsoft/phi-2 (2.7B)
           - TinyLlama/TinyLlama-1.1B-Chat-v1.0 (1.1B)
        """
        self.model_name = model_name
        self.max_context_length = max_context_length
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Initializing RAG engine with {model_name} on {self.device}")
        
        # For this example, we'll use a fallback that works without authentication
        # In production, use authenticated models or API-based solutions
        fallback_model = "microsoft/phi-2"  # Smaller model that doesn't require auth
        
        try:
            print(f"Attempting to load {model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True
            )
            print(f"Successfully loaded {model_name}")
            
        except Exception as e:
            print(f"Could not load {model_name}: {e}")
            print(f"Falling back to {fallback_model}...")
            
            try:
                self.model_name = fallback_model
                self.tokenizer = AutoTokenizer.from_pretrained(fallback_model, trust_remote_code=True)
                self.model = AutoModelForCausalLM.from_pretrained(
                    fallback_model,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map="auto" if self.device == "cuda" else None,
                    trust_remote_code=True
                )
                print(f"Successfully loaded fallback model: {fallback_model}")
            except Exception as e2:
                print(f"Fallback also failed: {e2}")
                raise Exception("Could not load any suitable model")
        
        # Set padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Move model to device
        if self.device == "cuda":
            self.model = self.model.to(self.device)
        
        print("Model loaded successfully!")
    
    def format_context(
        self,
        search_results: List[Dict[str, Any]],
        max_context_tokens: int = 1500
    ) -> str:
        """
        Format search results into well-structured context
        
        Key improvements:
        - Numbered sources for easy reference
        - Include relevance scores
        - Truncate if too long
        - Add source metadata
        """
        if not search_results:
            return "No relevant documents found in the knowledge base."
        
        context_parts = []
        total_tokens = 0
        
        for i, result in enumerate(search_results):
            content = result["content"]
            metadata = result["metadata"]
            
            # Calculate relevance indicator
            score = result.get("similarity_score", 0)
            rerank_score = result.get("rerank_score")
            
            if rerank_score is not None:
                score_display = f"Relevance: {rerank_score:.3f} (reranked)"
            else:
                score_display = f"Similarity: {score:.3f}"
            
            # Format source info
            source = metadata.get("source", "Unknown")
            page = metadata.get("page", "")
            page_info = f", Page {page}" if page else ""
            
            # Build context entry
            entry = f"""[Source {i+1}] {source}{page_info} ({score_display})
{content}
"""
            
            # Estimate tokens (rough approximation: 1 token ≈ 4 chars)
            entry_tokens = len(entry) // 4
            
            if total_tokens + entry_tokens > max_context_tokens:
                break
            
            context_parts.append(entry)
            total_tokens += entry_tokens
        
        context = "\n".join(context_parts)
        
        return context
    
    def generate_response(
        self,
        query: str,
        context: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> str:
        """
        Generate response with improved prompting and quality
        
        Key improvements:
        - Better prompt structure
        - Chain-of-thought reasoning
        - Citation enforcement
        - Factual grounding
        """
        
        # Construct improved prompt
        if context and "No relevant documents" not in context:
            system_prompt = """You are a helpful AI assistant that answers questions based on provided context. 
Your role is to:
1. Carefully read the provided sources
2. Answer the question accurately using information from the sources
3. Reference specific sources when making claims (e.g., "According to Source 1...")
4. If the sources don't contain enough information, say so clearly
5. Be concise but comprehensive"""
            
            prompt = f"""{system_prompt}

Context from documents:
{context}

Question: {query}

Instructions: Based on the context above, provide a clear and accurate answer. Reference the sources you use.

Answer:"""
        else:
            prompt = f"""Question: {query}

Answer: I don't have relevant documents to answer this question. Please provide documents or rephrase your query."""
        
        # Tokenize
        try:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_context_length - max_new_tokens
            ).to(self.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                    no_repeat_ngram_size=3
                )
            
            # Decode
            full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract answer (everything after "Answer:")
            if "Answer:" in full_output:
                response = full_output.split("Answer:")[-1].strip()
            else:
                response = full_output.replace(prompt, "").strip()
            
            # Clean up response
            response = self._clean_response(response, query, context)
            
            # Validate response quality
            if not response or len(response.split()) < 5:
                response = self._generate_fallback_response(query, context)
            
            return response
            
        except Exception as e:
            print(f"Generation error: {e}")
            import traceback
            traceback.print_exc()
            return self._generate_fallback_response(query, context)
    
    def _clean_response(self, response: str, query: str, context: str) -> str:
        """Clean and improve the generated response"""
        
        # Remove common artifacts
        response = re.sub(r'^(Answer|Response|Output):\s*', '', response, flags=re.IGNORECASE)
        
        # Remove repetitive patterns
        lines = response.split('\n')
        unique_lines = []
        seen = set()
        
        for line in lines:
            line_clean = line.strip().lower()
            if line_clean and line_clean not in seen:
                unique_lines.append(line.strip())
                seen.add(line_clean)
        
        response = '\n'.join(unique_lines)
        
        # Remove trailing incomplete sentences
        response = re.sub(r'\s+[^\s.!?]*$', '', response)
        
        # Ensure proper capitalization
        if response and response[0].islower():
            response = response[0].upper() + response[1:]
        
        return response.strip()
    
    def _generate_fallback_response(self, query: str, context: str) -> str:
        """Generate a rule-based fallback response when generation fails"""
        
        if not context or "No relevant documents" in context:
            return "I don't have enough information in the provided documents to answer this question accurately."
        
        # Extract key information from context
        lines = context.split('\n')
        relevant_lines = [line for line in lines if line.strip() and not line.startswith('[Source')]
        
        if not relevant_lines:
            return "I found some relevant documents, but couldn't extract a clear answer. Please try rephrasing your question."
        
        # Create a simple extractive answer
        response = "Based on the documents:\n\n"
        response += " ".join(relevant_lines[:3])  # Take first 3 relevant lines
        
        return response
    
    def rag_query(
        self,
        query: str,
        search_results: List[Dict[str, Any]],
        include_metadata: bool = True
    ) -> Dict[str, Any]:
        """
        Perform complete RAG query with enhanced output
        
        Returns:
            Dictionary with:
            - response: Generated answer
            - context_used: Formatted context
            - sources: Source metadata
            - search_results: Full search results
            - confidence: Confidence score
        """
        print(f"\n{'='*60}")
        print(f"RAG Query: {query}")
        print(f"{'='*60}")
        print(f"Search results retrieved: {len(search_results)}")
        
        if search_results:
            print("\nTop 3 results:")
            for i, result in enumerate(search_results[:3]):
                score = result.get('rerank_score', result.get('similarity_score', 0))
                print(f"  {i+1}. Score: {score:.3f} | Source: {result['metadata'].get('source', 'Unknown')}")
                print(f"     Preview: {result['content'][:100]}...")
        
        # Format context
        context = self.format_context(search_results)
        
        # Generate response
        print("\nGenerating response...")
        response = self.generate_response(query, context)
        
        # Calculate confidence score
        confidence = self._calculate_confidence(query, response, search_results)
        
        print(f"\nGenerated response (confidence: {confidence:.2f}):")
        print(f"{response[:200]}...")
        print(f"{'='*60}\n")
        
        result = {
            "response": response,
            "context_used": context,
            "sources": [result["metadata"] for result in search_results],
            "search_results": search_results,
            "confidence": confidence,
            "num_sources": len(search_results)
        }
        
        return result
    
    def _calculate_confidence(
        self,
        query: str,
        response: str,
        search_results: List[Dict[str, Any]]
    ) -> float:
        """
        Calculate confidence score for the generated response
        
        Based on:
        - Relevance of retrieved documents
        - Response length and completeness
        - Keyword overlap
        """
        if not search_results:
            return 0.0
        
        # Factor 1: Average similarity score of top results
        top_scores = [
            r.get('rerank_score', r.get('similarity_score', 0))
            for r in search_results[:3]
        ]
        avg_score = sum(top_scores) / len(top_scores) if top_scores else 0
        
        # Factor 2: Response completeness (length, proper sentences)
        response_sentences = len([s for s in response.split('.') if s.strip()])
        completeness_score = min(response_sentences / 3, 1.0)  # Expect 3+ sentences
        
        # Factor 3: Keyword overlap between query and response
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())
        overlap = len(query_words & response_words) / max(len(query_words), 1)
        
        # Combined confidence score
        confidence = (
            0.5 * avg_score +
            0.3 * completeness_score +
            0.2 * overlap
        )
        
        return min(confidence, 1.0)
    
    def batch_query(
        self,
        queries: List[str],
        search_results_list: List[List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """Process multiple queries efficiently"""
        
        results = []
        for query, search_results in zip(queries, search_results_list):
            result = self.rag_query(query, search_results)
            results.append(result)
        
        return results