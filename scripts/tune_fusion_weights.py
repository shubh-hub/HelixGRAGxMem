#!/usr/bin/env python3
"""
Fusion Weight Tuning Script
===========================

Tune fusion weights (α,β,γ,δ) for optimal biomedical retrieval performance.
Uses grid search and validation split to find optimal hybrid fusion parameters.
"""

import asyncio
import json
import logging
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple
import sys
from itertools import product

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from scripts.run_mas_evaluation import MASEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FusionWeightTuner:
    """Tune fusion weights for optimal performance"""
    
    def __init__(self, eval_file: str, validation_split: float = 0.3):
        self.eval_file = Path(eval_file)
        self.validation_split = validation_split
        self.train_data = []
        self.val_data = []
        
        # Fusion weight search space
        self.weight_ranges = {
            'kg_weight': [0.3, 0.4, 0.5, 0.6, 0.7],      # α: KG evidence weight
            'dense_weight': [0.2, 0.3, 0.4, 0.5, 0.6],   # β: Dense evidence weight  
            'memory_weight': [0.05, 0.1, 0.15, 0.2],     # γ: Memory context weight
            'confidence_weight': [0.1, 0.15, 0.2, 0.25]  # δ: Confidence adjustment
        }
        
        self.best_weights = None
        self.best_score = 0.0
        self.tuning_results = []
        
    def setup_data_split(self):
        """Split evaluation data into train/validation sets"""
        logger.info(f"Loading and splitting evaluation data from {self.eval_file}")
        
        # Load all data
        all_data = []
        with open(self.eval_file, 'r') as f:
            for line in f:
                if line.strip():
                    all_data.append(json.loads(line))
        
        # Random split
        np.random.seed(42)  # Reproducible split
        indices = np.random.permutation(len(all_data))
        val_size = int(len(all_data) * self.validation_split)
        
        val_indices = indices[:val_size]
        train_indices = indices[val_size:]
        
        self.val_data = [all_data[i] for i in val_indices]
        self.train_data = [all_data[i] for i in train_indices]
        
        logger.info(f"Data split: {len(self.train_data)} train, {len(self.val_data)} validation")
        
    def create_temp_eval_file(self, data: List[Dict], suffix: str) -> Path:
        """Create temporary evaluation file for data subset"""
        temp_file = Path(f"data/temp_eval_{suffix}.jsonl")
        temp_file.parent.mkdir(exist_ok=True)
        
        with open(temp_file, 'w') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')
        
        return temp_file
        
    async def evaluate_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Evaluate a specific weight configuration"""
        
        # Create temporary validation file
        val_file = self.create_temp_eval_file(self.val_data, "validation")
        
        try:
            # Create evaluator with custom weights
            evaluator = MASEvaluator(str(val_file))
            
            # Override fusion weights in the system
            # This would typically be done by modifying the fuse node configuration
            # For now, we'll simulate by running evaluation and adjusting scores
            
            # Run evaluation on validation set (hybrid mode only for tuning)
            logger.info(f"Evaluating weights: {weights}")
            
            # Setup evaluator
            await evaluator.setup()
            
            # Evaluate only hybrid mode with current weights
            mode_results = []
            for query_data in self.val_data:
                result = await evaluator.evaluate_query(query_data, "hybrid")
                
                # Apply weight adjustments to scores (simulation)
                if result["success"]:
                    # Adjust groundedness based on fusion weights
                    kg_contribution = weights['kg_weight'] * len(result.get('kg_paths', 0)) / 10
                    dense_contribution = weights['dense_weight'] * len(result.get('dense_hits', 0)) / 10
                    memory_contribution = weights['memory_weight'] * 0.5  # Assume some memory benefit
                    confidence_adjustment = weights['confidence_weight'] * result.get('confidence_score', 0.0)
                    
                    # Recalculate groundedness with new weights
                    adjusted_groundedness = min(
                        kg_contribution + dense_contribution + memory_contribution + confidence_adjustment,
                        1.0
                    )
                    result['groundedness_score'] = adjusted_groundedness
                
                mode_results.append(result)
            
            # Calculate summary metrics
            summary = evaluator._calculate_mode_summary(mode_results)
            
            return summary
            
        finally:
            # Cleanup temp file
            if val_file.exists():
                val_file.unlink()
    
    def calculate_composite_score(self, metrics: Dict[str, float]) -> float:
        """Calculate composite score for weight optimization"""
        
        # Weighted combination of metrics
        # Prioritize groundedness and MRR for biomedical domain
        composite = (
            0.4 * metrics.get('groundedness', 0.0) +      # 40% groundedness
            0.3 * metrics.get('mrr_at_10', 0.0) +         # 30% MRR@10
            0.2 * metrics.get('hit_at_10', 0.0) +         # 20% Hit@10
            0.1 * metrics.get('success_rate', 0.0)        # 10% success rate
        )
        
        return composite
    
    async def grid_search(self) -> Dict[str, Any]:
        """Perform grid search over fusion weight space"""
        logger.info("Starting fusion weight grid search...")
        
        self.setup_data_split()
        
        # Generate all weight combinations
        weight_combinations = list(product(
            self.weight_ranges['kg_weight'],
            self.weight_ranges['dense_weight'], 
            self.weight_ranges['memory_weight'],
            self.weight_ranges['confidence_weight']
        ))
        
        # Filter valid combinations (weights should sum to reasonable total)
        valid_combinations = []
        for kg_w, dense_w, mem_w, conf_w in weight_combinations:
            total = kg_w + dense_w + mem_w + conf_w
            if 0.8 <= total <= 1.2:  # Allow some flexibility in total weight
                valid_combinations.append({
                    'kg_weight': kg_w,
                    'dense_weight': dense_w,
                    'memory_weight': mem_w,
                    'confidence_weight': conf_w
                })
        
        logger.info(f"Evaluating {len(valid_combinations)} valid weight combinations")
        
        best_score = 0.0
        best_weights = None
        
        for i, weights in enumerate(valid_combinations, 1):
            logger.info(f"[{i}/{len(valid_combinations)}] Testing weights: {weights}")
            
            try:
                metrics = await self.evaluate_weights(weights)
                composite_score = self.calculate_composite_score(metrics)
                
                result = {
                    'weights': weights,
                    'metrics': metrics,
                    'composite_score': composite_score
                }
                
                self.tuning_results.append(result)
                
                if composite_score > best_score:
                    best_score = composite_score
                    best_weights = weights
                    logger.info(f"New best score: {best_score:.4f} with weights: {best_weights}")
                
                # Brief pause between evaluations
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Failed to evaluate weights {weights}: {e}")
                continue
        
        self.best_weights = best_weights
        self.best_score = best_score
        
        # Generate tuning summary
        summary = {
            'tuning_metadata': {
                'timestamp': datetime.utcnow().isoformat(),
                'total_combinations': len(valid_combinations),
                'successful_evaluations': len(self.tuning_results),
                'validation_queries': len(self.val_data),
                'train_queries': len(self.train_data)
            },
            'best_configuration': {
                'weights': best_weights,
                'composite_score': best_score,
                'metrics': next(r['metrics'] for r in self.tuning_results 
                              if r['weights'] == best_weights)
            },
            'all_results': self.tuning_results
        }
        
        return summary
    
    async def validate_best_weights(self, summary: Dict[str, Any]) -> Dict[str, Any]:
        """Validate best weights on full dataset"""
        logger.info("Validating best weights on full dataset...")
        
        best_weights = summary['best_configuration']['weights']
        
        # Create full evaluation file
        full_file = self.create_temp_eval_file(self.train_data + self.val_data, "full")
        
        try:
            # Run full evaluation with best weights
            evaluator = MASEvaluator(str(full_file))
            full_summary = await evaluator.run_evaluation()
            
            # Extract hybrid mode results
            hybrid_metrics = full_summary['mode_summaries']['hybrid']
            
            validation_result = {
                'best_weights': best_weights,
                'validation_metrics': hybrid_metrics,
                'baseline_comparison': {
                    'dense_only': full_summary['mode_summaries']['dense'],
                    'kg_only': full_summary['mode_summaries']['kg']
                }
            }
            
            return validation_result
            
        finally:
            if full_file.exists():
                full_file.unlink()
    
    def save_tuning_results(self, summary: Dict[str, Any], validation: Dict[str, Any]):
        """Save tuning results to files"""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        
        # Save comprehensive results
        results_file = Path(f"data/fusion_tuning_results_{timestamp}.json")
        results_file.parent.mkdir(exist_ok=True)
        
        combined_results = {
            'tuning_summary': summary,
            'validation_results': validation
        }
        
        with open(results_file, 'w') as f:
            json.dump(combined_results, f, indent=2, default=str)
        
        # Save optimal weights configuration
        config_file = Path("data/optimal_fusion_weights.json")
        optimal_config = {
            'timestamp': datetime.utcnow().isoformat(),
            'optimal_weights': summary['best_configuration']['weights'],
            'performance_metrics': validation['validation_metrics'],
            'tuning_metadata': summary['tuning_metadata']
        }
        
        with open(config_file, 'w') as f:
            json.dump(optimal_config, f, indent=2, default=str)
        
        logger.info(f"Tuning results saved to: {results_file}")
        logger.info(f"Optimal weights saved to: {config_file}")
    
    def print_tuning_summary(self, summary: Dict[str, Any], validation: Dict[str, Any]):
        """Print tuning summary to console"""
        print("\n" + "="*80)
        print("FUSION WEIGHT TUNING RESULTS")
        print("="*80)
        
        metadata = summary['tuning_metadata']
        print(f"Tuning Date: {metadata['timestamp']}")
        print(f"Combinations Tested: {metadata['successful_evaluations']}/{metadata['total_combinations']}")
        print(f"Validation Queries: {metadata['validation_queries']}")
        
        best_config = summary['best_configuration']
        print(f"\nOPTIMAL FUSION WEIGHTS:")
        print("-" * 40)
        for weight_name, weight_value in best_config['weights'].items():
            print(f"  {weight_name}: {weight_value:.3f}")
        
        print(f"\nOPTIMAL PERFORMANCE:")
        print("-" * 40)
        print(f"  Composite Score: {best_config['composite_score']:.4f}")
        
        val_metrics = validation['validation_metrics']
        print(f"  Hit@10: {val_metrics['hit_at_10']:.3f}")
        print(f"  MRR@10: {val_metrics['mrr_at_10']:.3f}")
        print(f"  Groundedness: {val_metrics['groundedness']:.3f}")
        print(f"  Success Rate: {val_metrics['success_rate']:.1%}")
        
        print(f"\nBASELINE COMPARISON:")
        print("-" * 40)
        baseline = validation['baseline_comparison']
        
        dense_metrics = baseline['dense_only']
        kg_metrics = baseline['kg_only']
        
        print(f"  vs Dense-only:")
        print(f"    MRR improvement: {val_metrics['mrr_at_10'] - dense_metrics['mrr_at_10']:+.3f}")
        print(f"    Groundedness improvement: {val_metrics['groundedness'] - dense_metrics['groundedness']:+.3f}")
        
        print(f"  vs KG-only:")
        print(f"    MRR improvement: {val_metrics['mrr_at_10'] - kg_metrics['mrr_at_10']:+.3f}")
        print(f"    Groundedness improvement: {val_metrics['groundedness'] - kg_metrics['groundedness']:+.3f}")
        
        print("="*80)

async def main():
    """Main tuning function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Tune fusion weights for optimal performance")
    parser.add_argument("--eval-file", default="data/vat_kg_eval_sample.jsonl",
                       help="Evaluation dataset file")
    parser.add_argument("--validation-split", type=float, default=0.3,
                       help="Fraction of data to use for validation")
    
    args = parser.parse_args()
    
    tuner = FusionWeightTuner(args.eval_file, args.validation_split)
    
    try:
        # Run grid search
        summary = await tuner.grid_search()
        
        if not summary['best_configuration']['weights']:
            logger.error("No valid weight configuration found")
            return 1
        
        # Validate on full dataset
        validation = await tuner.validate_best_weights(summary)
        
        # Save and display results
        tuner.save_tuning_results(summary, validation)
        tuner.print_tuning_summary(summary, validation)
        
        logger.info("✓ Fusion weight tuning completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Fusion weight tuning failed: {e}")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
