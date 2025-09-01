import asyncio
import time
import json
import logging
import sys
import os
from pathlib import Path

from typing import List, Dict, Any

from src.core.retrieval_engine import HybridRetriever

logger = logging.getLogger(__name__)

def calculate_metrics(results: List[Dict[str, Any]], ground_truth: List[str], k: int = 10) -> (float, float):
    """Calculates Hit Rate and MRR for a set of retrieval results."""
    hits = 0
    mrr_score = 0.0
    
    # Extract relevant entities from retrieved results
    retrieved_entities = []
    for res in results[:k]:
        # For dense retrieval, check the object in metadata
        if 'metadata' in res and 'object' in res['metadata']:
            retrieved_entities.append(res['metadata']['object'].lower())
        # For KG retrieval, extract from the result structure
        elif 'text' in res and 'metadata' in res:
            retrieved_entities.append(str(res['id']).lower())
    
    # Check if any ground truth entity is found in retrieved entities
    ground_truth_lower = [gt.lower() for gt in ground_truth]
    
    for i, entity in enumerate(retrieved_entities):
        if entity in ground_truth_lower:
            hits = 1
            mrr_score = 1.0 / (i + 1)
            break # Found first relevant document
            
    return hits, mrr_score

async def main():
    """Main function to run the retrieval evaluation."""
    logger.info("Starting retrieval evaluation...")
    
    # Initialize the retriever
    retriever = HybridRetriever()
    await retriever.initialize()

    # Load evaluation dataset
    eval_file = Path("data/processed/vat_kg_eval_sample.jsonl")
    if not eval_file.exists():
        logger.error(f"Evaluation file not found at {eval_file}. Please run scripts/create_eval_dataset.py first.")
        return

    with open(eval_file, 'r') as f:
        evaluation_data = [json.loads(line) for line in f]

    modes = ["kg_only", "dense_only", "hybrid"]
    all_metrics = {}

    for mode in modes:
        logger.info(f"--- Running evaluation for mode: {mode} ---")
        total_hits = 0
        total_mrr = 0.0
        
        for i, item in enumerate(evaluation_data):
            query = item['question']
            start_time = time.time()
            logger.debug(f"Running query {i+1}/{len(evaluation_data)}: '{query}' with intent 'factoid'")
            
            try:
                results = await retriever.retrieve(
                    query=query,
                    intent="factoid",
                    mode=mode,
                    top_k=10
                )
                
                hits, mrr = calculate_metrics(results, [item['answer_node']])
                total_hits += hits
                total_mrr += mrr

            except Exception as e:
                logger.error(f"Error processing query '{query}': {e}", exc_info=True)

        num_queries = len(evaluation_data)
        hit_rate = total_hits / num_queries if num_queries > 0 else 0
        mean_mrr = total_mrr / num_queries if num_queries > 0 else 0
        
        all_metrics[mode] = {"Hit_Rate@10": hit_rate, "MRR@10": mean_mrr}
        logger.info(f"Results for {mode}: Hit_Rate@10 = {hit_rate:.4f}, MRR@10 = {mean_mrr:.4f}")

    logger.info("\n--- Final Evaluation Summary ---")
    print(json.dumps(all_metrics, indent=2))
    logger.info("----------------------------------")

if __name__ == "__main__":
    asyncio.run(main())
