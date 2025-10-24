"""
RAG System Evaluation and Benchmarking
Measure retrieval accuracy, answer quality, and overall system performance
"""

import json
import time
from typing import List, Dict, Any, Tuple
from collections import defaultdict
import numpy as np


class RAGEvaluator:
    """Comprehensive evaluation suite for RAG systems"""
    
    def __init__(self, vector_store, rag_engine):
        self.vector_store = vector_store
        self.rag_engine = rag_engine
        self.results = []
    
    # ==================== Retrieval Metrics ====================
    
    def recall_at_k(
        self,
        retrieved_docs: List[Dict[str, Any]],
        relevant_doc_ids: List[str],
        k: int = 5
    ) -> float:
        """
        Recall@K: Proportion of relevant documents retrieved in top K
        
        Range: 0.0 to 1.0 (higher is better)
        """
        if not relevant_doc_ids:
            return 0.0
        
        retrieved_ids = set([
            f"{doc['metadata']['source']}_chunk_{doc['metadata']['chunk_id']}"
            for doc in retrieved_docs[:k]
        ])
        
        relevant_found = sum(1 for doc_id in relevant_doc_ids if doc_id in retrieved_ids)
        
        return relevant_found / len(relevant_doc_ids)
    
    def precision_at_k(
        self,
        retrieved_docs: List[Dict[str, Any]],
        relevant_doc_ids: List[str],
        k: int = 5
    ) -> float:
        """
        Precision@K: Proportion of retrieved documents that are relevant
        
        Range: 0.0 to 1.0 (higher is better)
        """
        if not retrieved_docs:
            return 0.0
        
        retrieved_ids = [
            f"{doc['metadata']['source']}_chunk_{doc['metadata']['chunk_id']}"
            for doc in retrieved_docs[:k]
        ]
        
        relevant_found = sum(1 for doc_id in retrieved_ids if doc_id in relevant_doc_ids)
        
        return relevant_found / min(k, len(retrieved_ids))
    
    def mean_reciprocal_rank(
        self,
        retrieved_docs: List[Dict[str, Any]],
        relevant_doc_ids: List[str]
    ) -> float:
        """
        MRR: 1 / rank of first relevant document
        
        Range: 0.0 to 1.0 (higher is better)
        1.0 = relevant doc at rank 1
        0.5 = relevant doc at rank 2
        0.0 = no relevant docs found
        """
        if not relevant_doc_ids:
            return 0.0
        
        for rank, doc in enumerate(retrieved_docs, 1):
            doc_id = f"{doc['metadata']['source']}_chunk_{doc['metadata']['chunk_id']}"
            if doc_id in relevant_doc_ids:
                return 1.0 / rank
        
        return 0.0
    
    def average_precision(
        self,
        retrieved_docs: List[Dict[str, Any]],
        relevant_doc_ids: List[str]
    ) -> float:
        """
        Average Precision: Average of precision values at each relevant doc
        
        Range: 0.0 to 1.0 (higher is better)
        """
        if not relevant_doc_ids:
            return 0.0
        
        precisions = []
        relevant_found = 0
        
        for rank, doc in enumerate(retrieved_docs, 1):
            doc_id = f"{doc['metadata']['source']}_chunk_{doc['metadata']['chunk_id']}"
            
            if doc_id in relevant_doc_ids:
                relevant_found += 1
                precision_at_rank = relevant_found / rank
                precisions.append(precision_at_rank)
        
        if not precisions:
            return 0.0
        
        return sum(precisions) / len(relevant_doc_ids)
    
    def ndcg_at_k(
        self,
        retrieved_docs: List[Dict[str, Any]],
        relevant_doc_ids: List[str],
        k: int = 5
    ) -> float:
        """
        Normalized Discounted Cumulative Gain
        Rewards relevant documents appearing early in results
        
        Range: 0.0 to 1.0 (higher is better)
        """
        if not relevant_doc_ids:
            return 0.0
        
        # Calculate DCG
        dcg = 0.0
        for rank, doc in enumerate(retrieved_docs[:k], 1):
            doc_id = f"{doc['metadata']['source']}_chunk_{doc['metadata']['chunk_id']}"
            relevance = 1 if doc_id in relevant_doc_ids else 0
            dcg += relevance / np.log2(rank + 1)
        
        # Calculate IDCG (ideal DCG)
        idcg = sum(1 / np.log2(rank + 1) for rank in range(1, min(k, len(relevant_doc_ids)) + 1))
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    # ==================== Answer Quality Metrics ====================
    
    def exact_match(self, predicted: str, reference: str) -> float:
        """Binary exact match (after normalization)"""
        pred_normalized = predicted.strip().lower()
        ref_normalized = reference.strip().lower()
        return 1.0 if pred_normalized == ref_normalized else 0.0
    
    def f1_score(self, predicted: str, reference: str) -> float:
        """
        Token-level F1 score
        Harmonic mean of precision and recall at token level
        """
        pred_tokens = set(predicted.lower().split())
        ref_tokens = set(reference.lower().split())
        
        if not pred_tokens or not ref_tokens:
            return 0.0
        
        common = pred_tokens & ref_tokens
        
        if not common:
            return 0.0
        
        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(ref_tokens)
        
        return 2 * (precision * recall) / (precision + recall)
    
    def rouge_l(self, predicted: str, reference: str) -> float:
        """
        ROUGE-L: Longest Common Subsequence based metric
        """
        pred_tokens = predicted.lower().split()
        ref_tokens = reference.lower().split()
        
        # Compute LCS
        lcs_length = self._lcs_length(pred_tokens, ref_tokens)
        
        if lcs_length == 0:
            return 0.0
        
        precision = lcs_length / len(pred_tokens) if pred_tokens else 0
        recall = lcs_length / len(ref_tokens) if ref_tokens else 0
        
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1
    
    def _lcs_length(self, seq1: List[str], seq2: List[str]) -> int:
        """Compute length of longest common subsequence"""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    def citation_accuracy(
        self,
        response: str,
        sources_used: List[Dict[str, Any]],
        claimed_sources: List[str]
    ) -> float:
        """
        Check if citations in response match actual sources used
        """
        if not claimed_sources:
            return 1.0  # No claims, no errors
        
        actual_sources = set([s['source'] for s in sources_used])
        claimed_set = set(claimed_sources)
        
        correct = len(claimed_set & actual_sources)
        total = len(claimed_set)
        
        return correct / total if total > 0 else 0.0
    
    # ==================== End-to-End Evaluation ====================
    
    def evaluate_single_query(
        self,
        query: str,
        relevant_doc_ids: List[str],
        reference_answer: str = None
    ) -> Dict[str, Any]:
        """
        Evaluate a single query end-to-end
        """
        start_time = time.time()
        
        # Retrieval
        retrieval_start = time.time()
        retrieved_docs = self.vector_store.search(query, n_results=10)
        retrieval_time = time.time() - retrieval_start
        
        # Generation
        generation_start = time.time()
        rag_result = self.rag_engine.rag_query(query, retrieved_docs[:5])
        generation_time = time.time() - generation_start
        
        total_time = time.time() - start_time
        
        # Compute retrieval metrics
        retrieval_metrics = {
            "recall@1": self.recall_at_k(retrieved_docs, relevant_doc_ids, k=1),
            "recall@3": self.recall_at_k(retrieved_docs, relevant_doc_ids, k=3),
            "recall@5": self.recall_at_k(retrieved_docs, relevant_doc_ids, k=5),
            "precision@3": self.precision_at_k(retrieved_docs, relevant_doc_ids, k=3),
            "mrr": self.mean_reciprocal_rank(retrieved_docs, relevant_doc_ids),
            "map": self.average_precision(retrieved_docs, relevant_doc_ids),
            "ndcg@5": self.ndcg_at_k(retrieved_docs, relevant_doc_ids, k=5),
        }
        
        # Compute answer quality metrics (if reference provided)
        answer_metrics = {}
        if reference_answer:
            answer_metrics = {
                "exact_match": self.exact_match(rag_result['response'], reference_answer),
                "f1_score": self.f1_score(rag_result['response'], reference_answer),
                "rouge_l": self.rouge_l(rag_result['response'], reference_answer),
            }
        
        result = {
            "query": query,
            "response": rag_result['response'],
            "retrieval_metrics": retrieval_metrics,
            "answer_metrics": answer_metrics,
            "confidence": rag_result.get('confidence', 0.0),
            "num_sources": rag_result.get('num_sources', 0),
            "timing": {
                "retrieval_time": retrieval_time,
                "generation_time": generation_time,
                "total_time": total_time
            },
            "retrieved_docs": retrieved_docs[:5]
        }
        
        return result
    
    def evaluate_dataset(
        self,
        test_cases: List[Dict[str, Any]],
        save_results: bool = True,
        output_file: str = "evaluation_results.json"
    ) -> Dict[str, Any]:
        """
        Evaluate on a full test dataset
        
        test_cases format:
        [
            {
                "query": "What is...",
                "relevant_docs": ["doc1_chunk_0", "doc1_chunk_1"],
                "reference_answer": "The answer is..."  # Optional
            },
            ...
        ]
        """
        print(f"\n{'='*60}")
        print(f"Evaluating RAG System on {len(test_cases)} queries")
        print(f"{'='*60}\n")
        
        all_results = []
        
        # Aggregated metrics
        retrieval_metrics = defaultdict(list)
        answer_metrics = defaultdict(list)
        timings = defaultdict(list)
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"[{i}/{len(test_cases)}] Evaluating: {test_case['query'][:50]}...")
            
            result = self.evaluate_single_query(
                query=test_case['query'],
                relevant_doc_ids=test_case.get('relevant_docs', []),
                reference_answer=test_case.get('reference_answer')
            )
            
            all_results.append(result)
            
            # Aggregate metrics
            for metric, value in result['retrieval_metrics'].items():
                retrieval_metrics[metric].append(value)
            
            for metric, value in result['answer_metrics'].items():
                answer_metrics[metric].append(value)
            
            for metric, value in result['timing'].items():
                timings[metric].append(value)
        
        # Calculate averages
        avg_retrieval = {k: np.mean(v) for k, v in retrieval_metrics.items()}
        avg_answer = {k: np.mean(v) for k, v in answer_metrics.items()}
        avg_timing = {k: np.mean(v) for k, v in timings.items()}
        
        # Summary
        summary = {
            "total_queries": len(test_cases),
            "avg_retrieval_metrics": avg_retrieval,
            "avg_answer_metrics": avg_answer,
            "avg_timing": avg_timing,
            "individual_results": all_results
        }
        
        # Print summary
        self._print_summary(summary)
        
        # Save results
        if save_results:
            with open(output_file, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"\n✅ Results saved to {output_file}")
        
        return summary
    
    def _print_summary(self, summary: Dict[str, Any]):
        """Print evaluation summary"""
        print(f"\n{'='*60}")
        print("EVALUATION SUMMARY")
        print(f"{'='*60}")
        
        print(f"\n📊 Retrieval Metrics (Average across {summary['total_queries']} queries):")
        for metric, value in summary['avg_retrieval_metrics'].items():
            print(f"  {metric:15s}: {value:.3f}")
        
        if summary['avg_answer_metrics']:
            print(f"\n📝 Answer Quality Metrics:")
            for metric, value in summary['avg_answer_metrics'].items():
                print(f"  {metric:15s}: {value:.3f}")
        
        print(f"\n⏱️  Performance:")
        for metric, value in summary['avg_timing'].items():
            print(f"  {metric:20s}: {value:.3f}s")
        
        print(f"\n{'='*60}\n")
    
    def compare_configurations(
        self,
        test_cases: List[Dict[str, Any]],
        config_names: List[str],
        config_results: List[Dict[str, Any]]
    ):
        """
        Compare multiple configurations side by side
        """
        print(f"\n{'='*60}")
        print("CONFIGURATION COMPARISON")
        print(f"{'='*60}\n")
        
        metrics_to_compare = [
            "recall@3",
            "precision@3",
            "mrr",
            "ndcg@5",
            "f1_score",
            "rouge_l"
        ]
        
        print(f"{'Metric':<20}", end="")
        for name in config_names:
            print(f"{name:<15}", end="")
        print()
        print("-" * (20 + 15 * len(config_names)))
        
        for metric in metrics_to_compare:
            print(f"{metric:<20}", end="")
            
            for result in config_results:
                # Check in retrieval or answer metrics
                value = result['avg_retrieval_metrics'].get(
                    metric,
                    result['avg_answer_metrics'].get(metric, 0.0)
                )
                
                # Color code: green if > 0.7, yellow if > 0.5, red otherwise
                if value > 0.7:
                    color = "🟢"
                elif value > 0.5:
                    color = "🟡"
                else:
                    color = "🔴"
                
                print(f"{color} {value:.3f}       ", end="")
            
            print()
        
        print(f"\n{'='*60}\n")


# ==================== Sample Test Cases ====================

SAMPLE_TEST_CASES = [
    {
        "query": "What are the main categories of LLMs?",
        "relevant_docs": ["llm_doc_chunk_0", "llm_doc_chunk_1"],
        "reference_answer": "LLMs can be categorized into general-purpose models, code-specialized models, instruction-tuned models, and domain-specific models."
    },
    {
        "query": "What is the maternity leave policy?",
        "relevant_docs": ["hr_policy_chunk_5"],
        "reference_answer": "The maternity leave policy provides 16 weeks of fully paid leave."
    },
    {
        "query": "How long is paternity leave?",
        "relevant_docs": ["hr_policy_chunk_5"],
        "reference_answer": "Paternity leave is 4 weeks, fully paid."
    }
]


# ==================== Usage Example ====================

def run_evaluation_example():
    """
    Example of how to run evaluation
    """
    # This is a placeholder - in real usage, you'd initialize with your actual components
    print("""
    Example Usage:
    
    # 1. Initialize evaluator
    evaluator = RAGEvaluator(vector_store, rag_engine)
    
    # 2. Evaluate single query
    result = evaluator.evaluate_single_query(
        query="What is RAG?",
        relevant_doc_ids=["rag_paper_chunk_0", "rag_paper_chunk_1"],
        reference_answer="RAG is Retrieval-Augmented Generation..."
    )
    
    # 3. Evaluate full dataset
    results = evaluator.evaluate_dataset(test_cases, save_results=True)
    
    # 4. Compare configurations
    config_a_results = evaluator.evaluate_dataset(test_cases_a)
    config_b_results = evaluator.evaluate_dataset(test_cases_b)
    
    evaluator.compare_configurations(
        test_cases,
        ["Config A", "Config B"],
        [config_a_results, config_b_results]
    )
    """)


if __name__ == "__main__":
    run_evaluation_example()