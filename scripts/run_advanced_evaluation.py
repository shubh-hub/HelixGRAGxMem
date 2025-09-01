#!/usr/bin/env python3
"""
Advanced Evaluation Framework - Hit@10, MRR@10, NDCG, Groundedness
================================================================

Comprehensive evaluation using the large, diverse dataset with proper metrics.
"""

import asyncio
import json
import logging
import sys
import time
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple
from collections import defaultdict
import math

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.core.retrieval_engine import HybridRetriever

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdvancedEvaluator:
    """Advanced evaluation with comprehensive metrics"""
    
    def __init__(self, eval_file: str):
        self.eval_file = Path(eval_file)
        self.hybrid_retriever = None
        self.results = {"dense": [], "kg": [], "hybrid": []}
        
    async def setup(self):
        """Initialize system"""
        logger.info("Initializing advanced evaluation system...")
        self.hybrid_retriever = HybridRetriever()
        await self.hybrid_retriever.initialize()
        logger.info("✅ System ready for evaluation")
        
    def load_dataset(self) -> List[Dict[str, Any]]:
        """Load evaluation dataset"""
        logger.info(f"Loading dataset from {self.eval_file}")
        dataset = []
        with open(self.eval_file, 'r') as f:
            for line in f:
                if line.strip():
                    dataset.append(json.loads(line))
        logger.info(f"Loaded {len(dataset)} evaluation queries")
        return dataset
    
    async def evaluate_query(self, query_data: Dict[str, Any], mode: str) -> Dict[str, Any]:
        """Evaluate single query with comprehensive metrics"""
        query = query_data["query"]
        expected_answers = query_data.get("expected_answers", [])
        intent = query_data.get("intent", "factoid")
        
        start_time = time.time()
        
        try:
            # Map our modes to retrieval engine modes
            retrieval_mode = {"dense": "dense_only", "kg": "kg_only", "hybrid": "hybrid"}[mode]
            
            results = await self.hybrid_retriever.retrieve(
                query=query, intent=intent, mode=retrieval_mode, top_k=10
            )
            
            duration = time.time() - start_time
            
            # Extract candidates
            candidates = self._extract_candidates(results, mode)
            
            # Calculate all metrics
            hit_at_10 = self._calculate_hit_at_k(candidates, expected_answers, k=10)
            mrr_at_10 = self._calculate_mrr_at_k(candidates, expected_answers, k=10)
            ndcg_at_10 = self._calculate_ndcg_at_k(candidates, expected_answers, k=10)
            precision_at_5 = self._calculate_precision_at_k(candidates, expected_answers, k=5)
            groundedness = self._calculate_groundedness(results, candidates, query_data)
            
            return {
                "query_id": query_data.get("id", "unknown"),
                "query": query,
                "mode": mode,
                "success": True,
                "duration": duration,
                "results_count": len(candidates),
                "hit_at_10": hit_at_10,
                "mrr_at_10": mrr_at_10,
                "ndcg_at_10": ndcg_at_10,
                "precision_at_5": precision_at_5,
                "groundedness": groundedness,
                "query_metadata": {
                    "complexity": query_data.get("complexity", "unknown"),
                    "difficulty": query_data.get("difficulty", "unknown"),
                    "query_type": query_data.get("query_type", "unknown")
                }
            }
            
        except Exception as e:
            logger.error(f"Query evaluation failed: {e}")
            return {
                "query_id": query_data.get("id", "unknown"),
                "query": query,
                "mode": mode,
                "success": False,
                "error": str(e),
                "duration": time.time() - start_time,
                "hit_at_10": 0.0, "mrr_at_10": 0.0, "ndcg_at_10": 0.0,
                "precision_at_5": 0.0, "groundedness": 0.0
            }
    
    def _extract_candidates(self, results: List[Dict], mode: str) -> List[Dict[str, Any]]:
        """Extract and normalize candidates"""
        candidates = []
        for i, result in enumerate(results):
            text = result.get("text", result.get("content", ""))
            if not text and "metadata" in result:
                metadata = result["metadata"]
                if "path" in metadata:
                    text = str(metadata["path"])
                elif "subject" in metadata:
                    text = metadata["subject"]
            
            candidates.append({
                "text": text[:200] if text else f"result_{i}",
                "score": result.get("score", 1.0 / (i + 1)),
                "rank": i + 1,
                "source": mode
            })
        return candidates[:10]
    
    def _calculate_hit_at_k(self, candidates: List[Dict], expected: List[str], k: int = 10) -> float:
        """Calculate Hit@K metric"""
        if not expected:
            return 0.0
        
        candidate_texts = [c["text"].lower() for c in candidates[:k]]
        for exp in expected:
            exp_lower = exp.lower()
            for cand_text in candidate_texts:
                if exp_lower in cand_text or cand_text in exp_lower:
                    return 1.0
        return 0.0
    
    def _calculate_mrr_at_k(self, candidates: List[Dict], expected: List[str], k: int = 10) -> float:
        """Calculate Mean Reciprocal Rank@K"""
        if not expected:
            return 0.0
        
        for i, candidate in enumerate(candidates[:k]):
            cand_text = candidate["text"].lower()
            for exp in expected:
                exp_lower = exp.lower()
                if exp_lower in cand_text or cand_text in exp_lower:
                    return 1.0 / (i + 1)
        return 0.0
    
    def _calculate_ndcg_at_k(self, candidates: List[Dict], expected: List[str], k: int = 10) -> float:
        """Calculate Normalized Discounted Cumulative Gain@K"""
        if not expected:
            return 0.0
        
        # Calculate relevance scores (1 if relevant, 0 if not)
        relevance_scores = []
        for candidate in candidates[:k]:
            cand_text = candidate["text"].lower()
            is_relevant = any(exp.lower() in cand_text or cand_text in exp.lower() for exp in expected)
            relevance_scores.append(1.0 if is_relevant else 0.0)
        
        # Calculate DCG
        dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(relevance_scores))
        
        # Calculate IDCG (ideal DCG)
        ideal_relevance = sorted(relevance_scores, reverse=True)
        idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal_relevance))
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def _calculate_precision_at_k(self, candidates: List[Dict], expected: List[str], k: int = 5) -> float:
        """Calculate Precision@K"""
        if not expected:
            return 0.0
        
        relevant_count = 0
        for candidate in candidates[:k]:
            cand_text = candidate["text"].lower()
            if any(exp.lower() in cand_text or cand_text in exp.lower() for exp in expected):
                relevant_count += 1
        
        return relevant_count / k
    
    def _calculate_groundedness(self, results: List[Dict], candidates: List[Dict], query_data: Dict) -> float:
        """Calculate groundedness score"""
        if not results:
            return 0.0
        
        # Evidence diversity (0-0.3)
        sources = set(r.get("source", "unknown") for r in results)
        diversity_score = min(len(sources) / 3.0, 1.0) * 0.3
        
        # Result quality (0-0.4)
        avg_score = np.mean([r.get("score", 0.0) for r in results])
        quality_score = avg_score * 0.4
        
        # Query complexity handling (0-0.3)
        complexity = query_data.get("complexity", "simple")
        complexity_bonus = {"simple": 0.3, "moderate": 0.25, "complex": 0.2, "expert": 0.15}[complexity]
        
        return min(diversity_score + quality_score + complexity_bonus, 1.0)
    
    async def run_evaluation(self, sample_size: int = 100) -> Dict[str, Any]:
        """Run comprehensive evaluation"""
        logger.info("Starting advanced evaluation with comprehensive metrics...")
        
        await self.setup()
        dataset = self.load_dataset()
        
        # Sample dataset for manageable evaluation
        if len(dataset) > sample_size:
            import random
            random.seed(42)
            dataset = random.sample(dataset, sample_size)
            logger.info(f"Sampled {sample_size} queries for evaluation")
        
        modes = ["dense", "kg", "hybrid"]
        
        for mode in modes:
            logger.info(f"\n=== Evaluating {mode.upper()} mode ===")
            mode_results = []
            
            for i, query_data in enumerate(dataset, 1):
                if i % 20 == 0:
                    logger.info(f"[{mode}] Progress: {i}/{len(dataset)}")
                
                result = await self.evaluate_query(query_data, mode)
                mode_results.append(result)
                
                # Brief pause
                await asyncio.sleep(0.05)
            
            self.results[mode] = mode_results
            
            # Calculate summary
            summary = self._calculate_mode_summary(mode_results)
            logger.info(f"[{mode}] Hit@10: {summary['hit_at_10']:.3f}, "
                       f"MRR@10: {summary['mrr_at_10']:.3f}, "
                       f"NDCG@10: {summary['ndcg_at_10']:.3f}")
        
        return self._generate_comprehensive_summary()
    
    def _calculate_mode_summary(self, results: List[Dict]) -> Dict[str, float]:
        """Calculate summary metrics for a mode"""
        successful = [r for r in results if r["success"]]
        
        if not successful:
            return {k: 0.0 for k in ["hit_at_10", "mrr_at_10", "ndcg_at_10", "precision_at_5", "groundedness", "success_rate", "avg_duration"]}
        
        return {
            "hit_at_10": np.mean([r["hit_at_10"] for r in successful]),
            "mrr_at_10": np.mean([r["mrr_at_10"] for r in successful]),
            "ndcg_at_10": np.mean([r["ndcg_at_10"] for r in successful]),
            "precision_at_5": np.mean([r["precision_at_5"] for r in successful]),
            "groundedness": np.mean([r["groundedness"] for r in successful]),
            "success_rate": len(successful) / len(results),
            "avg_duration": np.mean([r["duration"] for r in successful])
        }
    
    def _generate_comprehensive_summary(self) -> Dict[str, Any]:
        """Generate comprehensive evaluation summary"""
        mode_summaries = {}
        for mode in ["dense", "kg", "hybrid"]:
            mode_summaries[mode] = self._calculate_mode_summary(self.results[mode])
        
        # Best performers
        best_hit = max(mode_summaries.keys(), key=lambda m: mode_summaries[m]["hit_at_10"])
        best_mrr = max(mode_summaries.keys(), key=lambda m: mode_summaries[m]["mrr_at_10"])
        best_ndcg = max(mode_summaries.keys(), key=lambda m: mode_summaries[m]["ndcg_at_10"])
        best_groundedness = max(mode_summaries.keys(), key=lambda m: mode_summaries[m]["groundedness"])
        
        return {
            "evaluation_metadata": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_queries": len(self.results["dense"]),
                "dataset_file": str(self.eval_file)
            },
            "mode_summaries": mode_summaries,
            "best_performers": {
                "hit_at_10": {"mode": best_hit, "score": mode_summaries[best_hit]["hit_at_10"]},
                "mrr_at_10": {"mode": best_mrr, "score": mode_summaries[best_mrr]["mrr_at_10"]},
                "ndcg_at_10": {"mode": best_ndcg, "score": mode_summaries[best_ndcg]["ndcg_at_10"]},
                "groundedness": {"mode": best_groundedness, "score": mode_summaries[best_groundedness]["groundedness"]}
            },
            "acceptance_criteria": self._evaluate_acceptance_criteria(mode_summaries),
            "detailed_results": self.results
        }
    
    def _evaluate_acceptance_criteria(self, summaries: Dict) -> Dict[str, Any]:
        """Evaluate acceptance criteria"""
        hybrid = summaries.get("hybrid", {})
        dense = summaries.get("dense", {})
        
        if not hybrid or not dense:
            return {"status": "incomplete", "reason": "Missing mode results"}
        
        criterion_1 = hybrid.get("mrr_at_10", 0) >= dense.get("mrr_at_10", 0)
        criterion_2 = (hybrid.get("groundedness", 0) - dense.get("groundedness", 0)) > 0.1
        
        return {
            "criterion_1_hybrid_mrr_ge_dense": {
                "passed": criterion_1,
                "hybrid_mrr": hybrid.get("mrr_at_10", 0),
                "dense_mrr": dense.get("mrr_at_10", 0),
                "improvement": hybrid.get("mrr_at_10", 0) - dense.get("mrr_at_10", 0)
            },
            "criterion_2_hybrid_groundedness_improvement": {
                "passed": criterion_2,
                "hybrid_groundedness": hybrid.get("groundedness", 0),
                "dense_groundedness": dense.get("groundedness", 0),
                "improvement": hybrid.get("groundedness", 0) - dense.get("groundedness", 0)
            },
            "overall_status": "PASS" if criterion_1 and criterion_2 else "FAIL"
        }
    
    def print_evaluation_summary(self, summary: Dict[str, Any]):
        """Print comprehensive evaluation summary"""
        print("\n" + "="*100)
        print("ADVANCED EVALUATION RESULTS - Hit@10, MRR@10, NDCG@10, Groundedness")
        print("="*100)
        
        metadata = summary["evaluation_metadata"]
        print(f"Evaluation Date: {metadata['timestamp']}")
        print(f"Total Queries: {metadata['total_queries']}")
        print(f"Dataset: {metadata['dataset_file']}")
        
        print(f"\nCOMPREHENSIVE METRICS:")
        print("-" * 100)
        print(f"{'Mode':<8} {'Hit@10':<8} {'MRR@10':<8} {'NDCG@10':<9} {'P@5':<6} {'Groundedness':<12} {'Success%':<9} {'Avg Time':<9}")
        print("-" * 100)
        
        for mode, metrics in summary["mode_summaries"].items():
            print(f"{mode:<8} {metrics['hit_at_10']:<8.3f} {metrics['mrr_at_10']:<8.3f} "
                  f"{metrics['ndcg_at_10']:<9.3f} {metrics['precision_at_5']:<6.3f} "
                  f"{metrics['groundedness']:<12.3f} {metrics['success_rate']:<9.1%} "
                  f"{metrics['avg_duration']:<9.2f}s")
        
        print(f"\nBEST PERFORMERS:")
        best = summary["best_performers"]
        print(f"  Hit@10: {best['hit_at_10']['mode']} ({best['hit_at_10']['score']:.3f})")
        print(f"  MRR@10: {best['mrr_at_10']['mode']} ({best['mrr_at_10']['score']:.3f})")
        print(f"  NDCG@10: {best['ndcg_at_10']['mode']} ({best['ndcg_at_10']['score']:.3f})")
        print(f"  Groundedness: {best['groundedness']['mode']} ({best['groundedness']['score']:.3f})")
        
        # Acceptance criteria
        criteria = summary["acceptance_criteria"]
        print(f"\nACCEPTANCE CRITERIA EVALUATION:")
        print("-" * 50)
        
        c1 = criteria["criterion_1_hybrid_mrr_ge_dense"]
        c2 = criteria["criterion_2_hybrid_groundedness_improvement"]
        
        print(f"  Criterion 1 (Hybrid ≥ Dense MRR): {'✅ PASS' if c1['passed'] else '❌ FAIL'}")
        print(f"    Hybrid MRR: {c1['hybrid_mrr']:.3f}, Dense MRR: {c1['dense_mrr']:.3f}")
        print(f"    Improvement: {c1['improvement']:+.3f}")
        
        print(f"  Criterion 2 (Hybrid ≫ Dense Groundedness): {'✅ PASS' if c2['passed'] else '❌ FAIL'}")
        print(f"    Hybrid Groundedness: {c2['hybrid_groundedness']:.3f}, Dense: {c2['dense_groundedness']:.3f}")
        print(f"    Improvement: {c2['improvement']:+.3f}")
        
        overall_status = criteria["overall_status"]
        print(f"\n  🎯 OVERALL STATUS: {'🎉 ALL CRITERIA PASSED!' if overall_status == 'PASS' else '⚠️  SOME CRITERIA FAILED'}")
        
        print("="*100)

async def main():
    """Main evaluation function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run advanced evaluation")
    parser.add_argument("--eval-file", default="data/processed/comprehensive_eval_dataset.jsonl")
    parser.add_argument("--sample-size", type=int, default=100, help="Sample size for evaluation")
    
    args = parser.parse_args()
    
    evaluator = AdvancedEvaluator(args.eval_file)
    
    try:
        summary = await evaluator.run_evaluation(args.sample_size)
        evaluator.print_evaluation_summary(summary)
        
        # Return exit code based on acceptance criteria
        criteria_status = summary["acceptance_criteria"]["overall_status"]
        return 0 if criteria_status == "PASS" else 1
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return 2

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
