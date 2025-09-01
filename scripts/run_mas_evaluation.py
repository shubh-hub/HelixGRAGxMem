#!/usr/bin/env python3
"""
MAS Evaluation Script - Hit@10, MRR@10, and Groundedness Metrics
===============================================================

Comprehensive evaluation of the Multi-Agent System with MCP integration.
Runs evaluation JSONL through MAS (all modes) and reports:
- Hit@10: Top-10 hit rate
- MRR@10: Mean Reciprocal Rank at 10
- Groundedness: Evidence-based answer quality
- Mode-specific performance analysis
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple
import sys
import numpy as np
from collections import defaultdict

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.mas.state import MASState
from src.mas.graph import create_mas_graph
from src.mcp.client import MCPClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MASEvaluator:
    """Comprehensive MAS evaluation with multiple metrics"""
    
    def __init__(self, eval_file: str, output_dir: str = "data/evaluation_results"):
        self.eval_file = Path(eval_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        self.mas_graph = None
        self.evaluation_data = []
        self.results = {
            "dense": [],
            "kg": [],
            "hybrid": []
        }
        
    async def setup(self):
        """Setup evaluation environment"""
        logger.info("Setting up MAS evaluation environment...")
        
        # Load evaluation data
        self._load_evaluation_data()
        
        # Create MAS graph
        self.mas_graph = create_mas_graph()
        logger.info("MAS graph created successfully")
        
        # Verify MCP servers
        await self._verify_mcp_servers()
        
    def _load_evaluation_data(self):
        """Load evaluation dataset"""
        logger.info(f"Loading evaluation data from {self.eval_file}")
        
        with open(self.eval_file, 'r') as f:
            for line in f:
                if line.strip():
                    self.evaluation_data.append(json.loads(line))
        
        logger.info(f"Loaded {len(self.evaluation_data)} evaluation queries")
        
    async def _verify_mcp_servers(self):
        """Verify all MCP servers are healthy"""
        trace_id = str(uuid.uuid4())
        
        async with MCPClient(trace_id) as client:
            servers = ['trace', 'kg', 'dense', 'memory', 'validator', 'explain']
            
            for server in servers:
                try:
                    health = await client.health_check(server)
                    if health.get('status') != 'healthy':
                        raise RuntimeError(f"Server {server} is not healthy: {health}")
                    logger.info(f"✓ {server} server is healthy")
                except Exception as e:
                    logger.error(f"✗ {server} server health check failed: {e}")
                    raise
    
    async def evaluate_query(self, query_data: Dict[str, Any], mode: str) -> Dict[str, Any]:
        """Evaluate a single query in specified mode"""
        
        trace_id = str(uuid.uuid4())
        query = query_data["query"]
        expected_answers = query_data.get("answers", [])
        
        logger.debug(f"[{trace_id[:8]}] Evaluating: {query} (mode: {mode})")
        
        async with MCPClient(trace_id) as mcp_client:
            # Create initial state with forced mode
            initial_state = MASState(
                trace_id=trace_id,
                query=query,
                persona="researcher",  # Use researcher for evaluation
                mcp_client=mcp_client,
                timestamp=datetime.utcnow().isoformat(),
                status="initialized",
                mode=mode,  # Force specific mode
                mode_params={"k": 10, "max_hops": 3, "max_paths": 5}
            )
            
            start_time = datetime.utcnow()
            
            try:
                # Execute MAS workflow
                final_state = await self.mas_graph.ainvoke(initial_state)
                
                end_time = datetime.utcnow()
                duration = (end_time - start_time).total_seconds()
                
                # Extract answer candidates
                answer_candidates = self._extract_answer_candidates(final_state)
                
                # Calculate metrics
                hit_at_10 = self._calculate_hit_at_k(answer_candidates, expected_answers, k=10)
                mrr_at_10 = self._calculate_mrr_at_k(answer_candidates, expected_answers, k=10)
                groundedness_score = self._calculate_groundedness(final_state, answer_candidates)
                
                result = {
                    "trace_id": trace_id,
                    "query": query,
                    "mode": mode,
                    "success": True,
                    "duration_seconds": duration,
                    "expected_answers": expected_answers,
                    "answer_candidates": answer_candidates,
                    "hit_at_10": hit_at_10,
                    "mrr_at_10": mrr_at_10,
                    "groundedness_score": groundedness_score,
                    "kg_paths": len(final_state.get("kg_paths", [])),
                    "dense_hits": len(final_state.get("dense_hits", [])),
                    "confidence_score": final_state.get("confidence_score", 0.0),
                    "safety_verified": final_state.get("safety_verified", False),
                    "final_answer": final_state.get("final_answer", ""),
                    "error": None
                }
                
                return result
                
            except Exception as e:
                logger.error(f"[{trace_id[:8]}] Evaluation failed: {e}")
                
                return {
                    "trace_id": trace_id,
                    "query": query,
                    "mode": mode,
                    "success": False,
                    "duration_seconds": (datetime.utcnow() - start_time).total_seconds(),
                    "error": str(e),
                    "hit_at_10": 0.0,
                    "mrr_at_10": 0.0,
                    "groundedness_score": 0.0
                }
    
    def _extract_answer_candidates(self, final_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract ranked answer candidates from final state"""
        candidates = []
        
        # Extract from KG paths
        kg_paths = final_state.get("kg_paths", [])
        for i, path in enumerate(kg_paths):
            # Extract entities from path as potential answers
            entities = path.get("entities", [])
            for entity in entities:
                candidates.append({
                    "text": entity,
                    "score": path.get("score", 0.0) * (1.0 - i * 0.1),  # Decay by rank
                    "source": "kg",
                    "reasoning": path.get("reasoning", "")
                })
        
        # Extract from dense hits
        dense_hits = final_state.get("dense_hits", [])
        for i, hit in enumerate(dense_hits):
            # Extract key terms from content as potential answers
            content = hit.get("content", "")
            candidates.append({
                "text": content[:100],  # First 100 chars as candidate
                "score": hit.get("score", 0.0) * (1.0 - i * 0.05),  # Decay by rank
                "source": "dense",
                "reasoning": content
            })
        
        # Extract from final answer
        final_answer = final_state.get("final_answer", "")
        if final_answer:
            candidates.append({
                "text": final_answer,
                "score": final_state.get("confidence_score", 0.0),
                "source": "final",
                "reasoning": "Generated final answer"
            })
        
        # Sort by score and return top 10
        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates[:10]
    
    def _calculate_hit_at_k(self, candidates: List[Dict], expected_answers: List[str], k: int = 10) -> float:
        """Calculate Hit@K metric"""
        if not expected_answers:
            return 0.0
        
        # Check if any expected answer appears in top-k candidates
        candidate_texts = [c["text"].lower() for c in candidates[:k]]
        
        for expected in expected_answers:
            expected_lower = expected.lower()
            for candidate_text in candidate_texts:
                if expected_lower in candidate_text or candidate_text in expected_lower:
                    return 1.0
        
        return 0.0
    
    def _calculate_mrr_at_k(self, candidates: List[Dict], expected_answers: List[str], k: int = 10) -> float:
        """Calculate Mean Reciprocal Rank@K metric"""
        if not expected_answers:
            return 0.0
        
        # Find the rank of the first correct answer
        for i, candidate in enumerate(candidates[:k]):
            candidate_text = candidate["text"].lower()
            
            for expected in expected_answers:
                expected_lower = expected.lower()
                if expected_lower in candidate_text or candidate_text in expected_lower:
                    return 1.0 / (i + 1)  # Reciprocal rank (1-indexed)
        
        return 0.0
    
    def _calculate_groundedness(self, final_state: Dict[str, Any], candidates: List[Dict]) -> float:
        """Calculate groundedness score based on evidence quality"""
        
        # Factors for groundedness:
        # 1. Evidence diversity (KG + Dense)
        # 2. Confidence scores
        # 3. Safety verification
        # 4. Citation quality
        
        kg_paths = final_state.get("kg_paths", [])
        dense_hits = final_state.get("dense_hits", [])
        confidence = final_state.get("confidence_score", 0.0)
        safety_verified = final_state.get("safety_verified", False)
        citations = final_state.get("citations", [])
        
        # Evidence diversity score (0-0.3)
        diversity_score = 0.0
        if kg_paths and dense_hits:
            diversity_score = 0.3  # Both sources
        elif kg_paths or dense_hits:
            diversity_score = 0.15  # Single source
        
        # Confidence score (0-0.4)
        confidence_score = confidence * 0.4
        
        # Safety score (0-0.1)
        safety_score = 0.1 if safety_verified else 0.0
        
        # Citation quality score (0-0.2)
        citation_score = min(len(citations) * 0.05, 0.2)
        
        total_groundedness = diversity_score + confidence_score + safety_score + citation_score
        
        return min(total_groundedness, 1.0)
    
    async def run_evaluation(self) -> Dict[str, Any]:
        """Run comprehensive evaluation across all modes"""
        logger.info("Starting comprehensive MAS evaluation...")
        
        await self.setup()
        
        modes = ["dense", "kg", "hybrid"]
        total_queries = len(self.evaluation_data)
        
        for mode in modes:
            logger.info(f"\n=== Evaluating {mode.upper()} mode ===")
            mode_results = []
            
            for i, query_data in enumerate(self.evaluation_data, 1):
                logger.info(f"[{mode}] Query {i}/{total_queries}: {query_data['query'][:50]}...")
                
                result = await self.evaluate_query(query_data, mode)
                mode_results.append(result)
                
                # Brief pause between queries
                await asyncio.sleep(0.5)
            
            self.results[mode] = mode_results
            
            # Calculate mode summary
            mode_summary = self._calculate_mode_summary(mode_results)
            logger.info(f"[{mode}] Summary: Hit@10={mode_summary['hit_at_10']:.3f}, "
                       f"MRR@10={mode_summary['mrr_at_10']:.3f}, "
                       f"Groundedness={mode_summary['groundedness']:.3f}")
        
        # Generate comprehensive summary
        summary = self._generate_evaluation_summary()
        
        # Save results
        await self._save_evaluation_results(summary)
        
        return summary
    
    def _calculate_mode_summary(self, mode_results: List[Dict]) -> Dict[str, float]:
        """Calculate summary metrics for a mode"""
        successful_results = [r for r in mode_results if r["success"]]
        
        if not successful_results:
            return {
                "hit_at_10": 0.0,
                "mrr_at_10": 0.0,
                "groundedness": 0.0,
                "success_rate": 0.0,
                "avg_duration": 0.0
            }
        
        hit_at_10 = np.mean([r["hit_at_10"] for r in successful_results])
        mrr_at_10 = np.mean([r["mrr_at_10"] for r in successful_results])
        groundedness = np.mean([r["groundedness_score"] for r in successful_results])
        success_rate = len(successful_results) / len(mode_results)
        avg_duration = np.mean([r["duration_seconds"] for r in successful_results])
        
        return {
            "hit_at_10": hit_at_10,
            "mrr_at_10": mrr_at_10,
            "groundedness": groundedness,
            "success_rate": success_rate,
            "avg_duration": avg_duration
        }
    
    def _generate_evaluation_summary(self) -> Dict[str, Any]:
        """Generate comprehensive evaluation summary"""
        
        mode_summaries = {}
        for mode in ["dense", "kg", "hybrid"]:
            mode_summaries[mode] = self._calculate_mode_summary(self.results[mode])
        
        # Overall statistics
        total_queries = len(self.evaluation_data)
        total_evaluations = sum(len(results) for results in self.results.values())
        
        # Best performing mode analysis
        best_hit_mode = max(mode_summaries.keys(), key=lambda m: mode_summaries[m]["hit_at_10"])
        best_mrr_mode = max(mode_summaries.keys(), key=lambda m: mode_summaries[m]["mrr_at_10"])
        best_groundedness_mode = max(mode_summaries.keys(), key=lambda m: mode_summaries[m]["groundedness"])
        
        summary = {
            "evaluation_metadata": {
                "timestamp": datetime.utcnow().isoformat(),
                "total_queries": total_queries,
                "total_evaluations": total_evaluations,
                "modes_evaluated": list(self.results.keys()),
                "eval_file": str(self.eval_file)
            },
            "mode_summaries": mode_summaries,
            "best_performers": {
                "hit_at_10": {
                    "mode": best_hit_mode,
                    "score": mode_summaries[best_hit_mode]["hit_at_10"]
                },
                "mrr_at_10": {
                    "mode": best_mrr_mode,
                    "score": mode_summaries[best_mrr_mode]["mrr_at_10"]
                },
                "groundedness": {
                    "mode": best_groundedness_mode,
                    "score": mode_summaries[best_groundedness_mode]["groundedness"]
                }
            },
            "detailed_results": self.results
        }
        
        return summary
    
    async def _save_evaluation_results(self, summary: Dict[str, Any]):
        """Save evaluation results to files"""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        
        # Save comprehensive results
        results_file = self.output_dir / f"mas_evaluation_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Save CSV summary for analysis
        csv_file = self.output_dir / f"mas_evaluation_summary_{timestamp}.csv"
        self._save_csv_summary(summary, csv_file)
        
        logger.info(f"Evaluation results saved to: {results_file}")
        logger.info(f"CSV summary saved to: {csv_file}")
    
    def _save_csv_summary(self, summary: Dict[str, Any], csv_file: Path):
        """Save CSV summary for easy analysis"""
        import csv
        
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow([
                "Mode", "Hit@10", "MRR@10", "Groundedness", 
                "Success Rate", "Avg Duration (s)", "Total Queries"
            ])
            
            # Data rows
            for mode, metrics in summary["mode_summaries"].items():
                writer.writerow([
                    mode,
                    f"{metrics['hit_at_10']:.4f}",
                    f"{metrics['mrr_at_10']:.4f}",
                    f"{metrics['groundedness']:.4f}",
                    f"{metrics['success_rate']:.4f}",
                    f"{metrics['avg_duration']:.2f}",
                    len(summary["detailed_results"][mode])
                ])
    
    def print_evaluation_summary(self, summary: Dict[str, Any]):
        """Print evaluation summary to console"""
        print("\n" + "="*80)
        print("MAS EVALUATION RESULTS - Hit@10, MRR@10, Groundedness")
        print("="*80)
        
        metadata = summary["evaluation_metadata"]
        print(f"Evaluation Date: {metadata['timestamp']}")
        print(f"Total Queries: {metadata['total_queries']}")
        print(f"Modes Evaluated: {', '.join(metadata['modes_evaluated'])}")
        
        print(f"\nPER-MODE PERFORMANCE:")
        print("-" * 80)
        print(f"{'Mode':<10} {'Hit@10':<8} {'MRR@10':<8} {'Groundedness':<12} {'Success%':<9} {'Avg Time':<9}")
        print("-" * 80)
        
        for mode, metrics in summary["mode_summaries"].items():
            print(f"{mode:<10} {metrics['hit_at_10']:<8.3f} {metrics['mrr_at_10']:<8.3f} "
                  f"{metrics['groundedness']:<12.3f} {metrics['success_rate']:<9.1%} "
                  f"{metrics['avg_duration']:<9.2f}s")
        
        print(f"\nBEST PERFORMERS:")
        best = summary["best_performers"]
        print(f"  Hit@10: {best['hit_at_10']['mode']} ({best['hit_at_10']['score']:.3f})")
        print(f"  MRR@10: {best['mrr_at_10']['mode']} ({best['mrr_at_10']['score']:.3f})")
        print(f"  Groundedness: {best['groundedness']['mode']} ({best['groundedness']['score']:.3f})")
        
        # Performance analysis
        hybrid_metrics = summary["mode_summaries"].get("hybrid", {})
        dense_metrics = summary["mode_summaries"].get("dense", {})
        
        print(f"\nPERFORMANCE ANALYSIS:")
        if hybrid_metrics and dense_metrics:
            mrr_improvement = hybrid_metrics.get("mrr_at_10", 0) - dense_metrics.get("mrr_at_10", 0)
            groundedness_improvement = hybrid_metrics.get("groundedness", 0) - dense_metrics.get("groundedness", 0)
            
            print(f"  Hybrid vs Dense MRR improvement: {mrr_improvement:+.3f}")
            print(f"  Hybrid vs Dense Groundedness improvement: {groundedness_improvement:+.3f}")
            
            if hybrid_metrics.get("mrr_at_10", 0) >= dense_metrics.get("mrr_at_10", 0):
                print("  ✓ Acceptance criterion: hybrid ≥ dense_only MRR")
            else:
                print("  ✗ Acceptance criterion: hybrid ≥ dense_only MRR")
                
            if groundedness_improvement > 0.1:  # 10% improvement threshold
                print("  ✓ Acceptance criterion: hybrid ≫ dense_only groundedness")
            else:
                print("  ✗ Acceptance criterion: hybrid ≫ dense_only groundedness")
        
        print("="*80)

async def main():
    """Main evaluation function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run MAS evaluation with Hit@10, MRR@10, and groundedness metrics")
    parser.add_argument("--eval-file", default="data/vat_kg_eval_sample.jsonl", 
                       help="Evaluation dataset file")
    parser.add_argument("--output-dir", default="data/evaluation_results",
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    evaluator = MASEvaluator(args.eval_file, args.output_dir)
    
    try:
        summary = await evaluator.run_evaluation()
        evaluator.print_evaluation_summary(summary)
        
        # Check acceptance criteria
        hybrid_metrics = summary["mode_summaries"].get("hybrid", {})
        dense_metrics = summary["mode_summaries"].get("dense", {})
        
        if hybrid_metrics and dense_metrics:
            mrr_criterion = hybrid_metrics.get("mrr_at_10", 0) >= dense_metrics.get("mrr_at_10", 0)
            groundedness_criterion = (hybrid_metrics.get("groundedness", 0) - 
                                    dense_metrics.get("groundedness", 0)) > 0.1
            
            if mrr_criterion and groundedness_criterion:
                logger.info("✓ All acceptance criteria met!")
                return 0
            else:
                logger.warning("✗ Some acceptance criteria not met")
                return 1
        else:
            logger.warning("Could not evaluate acceptance criteria")
            return 2
            
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return 3

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
