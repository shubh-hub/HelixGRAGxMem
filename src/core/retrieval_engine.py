"""
Hybrid Retrieval Engine - Orchestration Layer
============================================

Orchestrates KG and Dense retrieval with intelligent fusion scoring.
Implements confidence-guided search and adaptive weighting strategies.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import asyncio
from concurrent.futures import ThreadPoolExecutor
import numpy as np

from .dense_retriever import DenseRetriever
from .walker import KGWalker
from .edge_predictor import EdgePredictor
from .relation_pruner import RelationPruner
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from config import settings

logger = logging.getLogger(__name__)

class HybridRetriever:
    """Hybrid retrieval engine combining KG and Dense search"""
    
    def __init__(self):
        self.dense_retriever = DenseRetriever()
        self.kg_walker = KGWalker()
        self.edge_predictor = EdgePredictor()
        self.relation_pruner = RelationPruner()
        self.initialized = False
        
    def initialize(self):
        """Initialize all retrieval components"""
        try:
            logger.info("Initializing hybrid retrieval engine...")
            
            # Initialize dense retriever
            self.dense_retriever.initialize()
            
            # Initialize KG components
            self.kg_walker.connect_kg()
            
            self.initialized = True
            logger.info("✅ Hybrid retrieval engine initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize hybrid retriever: {e}")
            raise
    
    def search(self, 
               query: str, 
               entities: List[str] = None,
               k: int = 10,
               kg_weight: float = 0.6,
               dense_weight: float = 0.4,
               intent: str = "factoid",
               trace_id: str = None) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining KG and Dense retrieval
        
        Args:
            query: Search query
            entities: Optional seed entities for KG search
            k: Number of results to return
            kg_weight: Weight for KG results (0-1)
            dense_weight: Weight for Dense results (0-1)
            trace_id: Optional trace ID for logging
            
        Returns:
            Fused and ranked search results
        """
        if not self.initialized:
            self.initialize()
        
        try:
            # Normalize weights
            total_weight = kg_weight + dense_weight
            if total_weight > 0:
                kg_weight = kg_weight / total_weight
                dense_weight = dense_weight / total_weight
            else:
                kg_weight, dense_weight = 0.5, 0.5
            
            # Perform parallel retrieval
            kg_results = []
            dense_results = []
            
            # Always perform dense search with intent awareness
            dense_results = self.dense_retriever.search_with_intent(query, intent=intent, k=k*2, trace_id=trace_id)
            
            # Perform KG search if entities provided or can be extracted
            if entities or self._has_openai_key():
                if not entities:
                    # Try to extract entities from query (simplified)
                    entities = self._extract_entities_from_query(query)
                
                if entities:
                    kg_results = self._kg_search(query, entities, k=k*2, intent=intent, trace_id=trace_id)
            
            # Fuse results
            fused_results = self._fuse_results(
                kg_results, dense_results, kg_weight, dense_weight, k
            )
            
            logger.info(f"Hybrid search: {len(kg_results)} KG + {len(dense_results)} Dense → {len(fused_results)} fused")
            return fused_results
            
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            # Fallback to dense-only search
            return self.dense_retriever.search(query, k=k, trace_id=trace_id)
    
    def _has_openai_key(self) -> bool:
        """Check if OpenAI API key is available for KG search"""
        return bool(settings.OPENAI_API_KEY or settings.GROQ_API_KEY)
    
    def _extract_entities_from_query(self, query: str) -> List[str]:
        """Extract entities from query (simplified implementation)"""
        # This is a simplified fallback - in practice, this would use NER
        # For now, return empty list to avoid KG search without proper entity extraction
        return []
    
    def _kg_search(self, 
                   query: str, 
                   entities: List[str], 
                   k: int = 10,
                   intent: str = "factoid",
                   trace_id: str = None) -> List[Dict[str, Any]]:
        """Perform KG-based search with intent awareness"""
        try:
            # Use KG walker to find relevant paths with intent
            paths = self.kg_walker.walk(
                start_entities=entities,
                query_context=query,
                max_hops=3,
                max_paths=k,
                confidence_threshold=0.3,
                intent=intent
            )
            
            # Convert paths to retrieval results
            kg_results = []
            for i, path_info in enumerate(paths):
                # Create result from path
                result_text = self._path_to_text(path_info)
                
                kg_results.append({
                    "id": f"kg_path_{i}",
                    "text": result_text,
                    "score": path_info["confidence"],
                    "rank": i + 1,
                    "source": "kg",
                    "metadata": {
                        "path": path_info["path"],
                        "relations": path_info["relations"],
                        "hops": path_info["hops"],
                        "entities": path_info.get("entities_found", [])
                    }
                })
            
            return kg_results
            
        except Exception as e:
            logger.error(f"KG search failed: {e}")
            return []
    
    def _path_to_text(self, path_info: Dict[str, Any]) -> str:
        """Convert KG path to natural language text"""
        path = path_info["path"]
        relations = path_info["relations"]
        
        if len(path) < 2:
            return f"Entity: {path[0]}" if path else "Empty path"
        
        # Create natural language description of path
        text_parts = [path[0]]
        
        for i, relation in enumerate(relations):
            if i + 1 < len(path):
                # Convert relation to natural language
                relation_text = relation.replace("_", " ")
                text_parts.append(f"{relation_text} {path[i + 1]}")
        
        return " → ".join(text_parts)
    
    def _fuse_results(self, 
                     kg_results: List[Dict[str, Any]], 
                     dense_results: List[Dict[str, Any]],
                     kg_weight: float,
                     dense_weight: float,
                     k: int) -> List[Dict[str, Any]]:
        """Fuse KG and Dense results using weighted scoring"""
        
        # Normalize scores within each result set
        kg_results = self._normalize_scores(kg_results, "kg")
        dense_results = self._normalize_scores(dense_results, "dense")
        
        # Combine results with weighted scores
        all_results = []
        
        # Add KG results with weighted scores
        for result in kg_results:
            result["fused_score"] = result["normalized_score"] * kg_weight
            result["kg_component"] = result["normalized_score"] * kg_weight
            result["dense_component"] = 0.0
            all_results.append(result)
        
        # Add Dense results with weighted scores
        for result in dense_results:
            result["fused_score"] = result["normalized_score"] * dense_weight
            result["kg_component"] = 0.0
            result["dense_component"] = result["normalized_score"] * dense_weight
            all_results.append(result)
        
        # Check for overlapping results and boost them
        all_results = self._handle_overlaps(all_results)
        
        # Sort by fused score and return top k
        all_results.sort(key=lambda x: x["fused_score"], reverse=True)
        
        # Re-rank and add final rank
        for i, result in enumerate(all_results[:k]):
            result["final_rank"] = i + 1
        
        return all_results[:k]
    
    def _normalize_scores(self, results: List[Dict[str, Any]], source: str) -> List[Dict[str, Any]]:
        """Normalize scores within result set"""
        if not results:
            return results
        
        scores = [r["score"] for r in results]
        
        if len(scores) == 1:
            results[0]["normalized_score"] = 1.0
            return results
        
        # Min-max normalization
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            for result in results:
                result["normalized_score"] = 1.0
        else:
            for result in results:
                normalized = (result["score"] - min_score) / (max_score - min_score)
                result["normalized_score"] = normalized
        
        return results
    
    def _handle_overlaps(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Handle overlapping results between KG and Dense"""
        # Simple overlap detection based on text similarity
        # In practice, this could be more sophisticated
        
        seen_texts = {}
        final_results = []
        
        for result in results:
            text_key = result["text"][:100].lower()  # Use first 100 chars as key
            
            if text_key in seen_texts:
                # Boost score for overlapping results
                existing_result = seen_texts[text_key]
                existing_result["fused_score"] += result["fused_score"] * 0.5  # Boost factor
                existing_result["kg_component"] += result.get("kg_component", 0)
                existing_result["dense_component"] += result.get("dense_component", 0)
                existing_result["overlap_boost"] = True
            else:
                seen_texts[text_key] = result
                final_results.append(result)
        
        return final_results
    
    def search_kg_only(self, query: str, entities: List[str], k: int = 10, intent: str = "factoid", trace_id: str = None) -> List[Dict[str, Any]]:
        """Perform KG-only search"""
        if not self.initialized:
            self.initialize()
        
        return self._kg_search(query, entities, k, intent, trace_id)
    
    def search_dense_only(self, query: str, k: int = 10, intent: str = "factoid", trace_id: str = None) -> List[Dict[str, Any]]:
        """Perform Dense-only search"""
        if not self.initialized:
            self.initialize()
        
        return self.dense_retriever.search_with_intent(query, intent, k, trace_id)
    
    def get_retrieval_stats(self) -> Dict[str, Any]:
        """Get retrieval statistics and health info"""
        stats = {
            "initialized": self.initialized,
            "components": {
                "dense_retriever": bool(self.dense_retriever.model),
                "kg_walker": bool(self.kg_walker.kg_db),
                "edge_predictor": bool(self.edge_predictor.llm),
                "relation_pruner": bool(self.relation_pruner.relation_stats)
            }
        }
        
        if self.initialized:
            try:
                # Get component-specific stats
                if self.dense_retriever.passages:
                    stats["dense_passages"] = len(self.dense_retriever.passages)
                
                if self.kg_walker.graph:
                    stats["kg_nodes"] = self.kg_walker.graph.number_of_nodes()
                    stats["kg_edges"] = self.kg_walker.graph.number_of_edges()
                
                if self.relation_pruner.relation_stats:
                    stats["kg_relations"] = len(self.relation_pruner.relation_stats)
                    
            except Exception as e:
                stats["stats_error"] = str(e)
        
        return stats
    
    async def retrieve_batch(self, queries: List[str], mode: str = "hybrid", max_results_per_query: int = 10, kg_weight: float = 0.6, dense_weight: float = 0.4) -> List[List[Dict[str, Any]]]:
        """Batch retrieval for multiple queries"""
        if not self.initialized:
            self.initialize()
        
        batch_results = []
        
        if mode == "dense":
            # Dense-only batch search
            batch_results = self.dense_retriever.batch_search(queries, k=max_results_per_query)
        
        elif mode == "kg":
            # KG-only batch search
            for query in queries:
                entities = self.kg_walker._resolve_entities_from_query(query)
                if entities:
                    kg_results = self._kg_search(query, entities, k=max_results_per_query)
                    batch_results.append(kg_results)
                else:
                    batch_results.append([])
        
        else:  # hybrid mode
            # Hybrid batch search
            for query in queries:
                result = self.search(
                    query=query,
                    k=max_results_per_query,
                    kg_weight=kg_weight,
                    dense_weight=dense_weight
                )
                batch_results.append(result)
        
        return batch_results
    
    async def retrieve_adaptive(self, query: str, intent: str = "factoid", max_results: int = 10) -> List[Dict[str, Any]]:
        """Adaptive retrieval based on query characteristics"""
        if not self.initialized:
            self.initialize()
        
        # Analyze query to determine optimal strategy
        query_words = len(query.split())
        
        # Intent-based weight adjustment
        if intent == "causal":
            kg_weight, dense_weight = 0.7, 0.3  # Favor KG for causal reasoning
        elif intent == "factoid":
            kg_weight, dense_weight = 0.5, 0.5  # Balanced
        elif intent == "therapeutic":
            kg_weight, dense_weight = 0.8, 0.2  # Strong KG preference for treatments
        else:
            kg_weight, dense_weight = 0.6, 0.4  # Default
        
        # Query complexity adjustment
        if query_words > 15:
            # Complex queries benefit from dense search
            dense_weight += 0.1
            kg_weight -= 0.1
        elif query_words < 5:
            # Simple queries benefit from KG
            kg_weight += 0.1
            dense_weight -= 0.1
        
        return self.search(
            query=query,
            k=max_results,
            kg_weight=kg_weight,
            dense_weight=dense_weight,
            intent=intent
        )
