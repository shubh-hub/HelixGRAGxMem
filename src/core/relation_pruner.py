"""
Relation Pruner - Multi-signal Relation Filtering
===============================================

Implements intelligent relation filtering using multiple signals:
- Semantic relevance to query
- Biomedical domain knowledge
- Graph topology metrics
- Confidence-guided pruning
"""

import logging
from typing import List, Dict, Any, Set, Tuple
import duckdb
import numpy as np
from collections import defaultdict, Counter

from ..config import settings

logger = logging.getLogger(__name__)

class RelationPruner:
    """Multi-signal relation filtering for KG traversal optimization"""
    
    def __init__(self):
        self.kg_db = None
        self.relation_stats = {}
        self.biomedical_relations = self._get_biomedical_relations()
        
    def connect_kg(self, db_path: str = None):
        """Connect to knowledge graph database"""
        try:
            db_path = db_path or settings.DB_PATH
            self.kg_db = duckdb.connect(db_path)
            self._compute_relation_statistics()
            logger.info("✅ Connected to KG database for relation pruning")
        except Exception as e:
            logger.error(f"Failed to connect to KG database: {e}")
    
    def _get_biomedical_relations(self) -> Dict[str, float]:
        """Define biomedical relation priorities"""
        return {
            # High priority - direct therapeutic/causal relations
            "treats": 1.0,
            "causes": 1.0,
            "prevents": 0.95,
            "cures": 0.95,
            
            # Medium-high priority - functional relations
            "interacts_with": 0.8,
            "affects": 0.8,
            "regulates": 0.8,
            "inhibits": 0.8,
            "activates": 0.8,
            
            # Medium priority - structural/locational
            "part_of": 0.7,
            "located_in": 0.7,
            "contains": 0.7,
            "connected_to": 0.7,
            
            # Lower priority - associative
            "associated_with": 0.6,
            "related_to": 0.5,
            "similar_to": 0.5,
            
            # Contextual relations
            "occurs_in": 0.6,
            "found_in": 0.6,
            "expressed_in": 0.7,
        }
    
    def _compute_relation_statistics(self):
        """Compute statistics for all relations in KG"""
        if not self.kg_db:
            self.connect_kg()
        
        try:
            # Get relation frequencies
            result = self.kg_db.execute("""
                SELECT predicate, COUNT(*) as frequency
                FROM triples 
                GROUP BY predicate
            """).fetchall()
            
            total_triples = 0
            relation_counts = {}
            
            for predicate, count in result:
                relation_counts[predicate] = count
                total_triples += count
            
            # Calculate normalized frequencies and other stats
            for predicate, count in relation_counts.items():
                frequency = count / total_triples if total_triples > 0 else 0
                
                # Get unique entities connected by this relation
                unique_result = self.kg_db.execute("""
                    SELECT COUNT(DISTINCT subject), COUNT(DISTINCT object)
                    FROM triples 
                    WHERE predicate = ?
                """, (predicate,)).fetchone()
                
                unique_subjects, unique_objects = unique_result
                
                self.relation_stats[predicate] = {
                    "frequency": frequency,
                    "count": count,
                    "unique_subjects": unique_subjects,
                    "unique_objects": unique_objects,
                    "connectivity": unique_subjects + unique_objects,
                    "biomedical_priority": self.biomedical_relations.get(predicate, 0.3)
                }
            
            logger.info(f"✅ Computed statistics for {len(self.relation_stats)} relations")
            
        except Exception as e:
            logger.error(f"Failed to compute relation statistics: {e}")
    
    def prune_relations(self, 
                       relations: List[str], 
                       query_context: str = "",
                       current_entity: str = "",
                       max_relations: int = 5,
                       min_confidence: float = 0.3) -> List[Dict[str, Any]]:
        """
        Prune relations using multi-signal filtering
        
        Args:
            relations: List of candidate relations
            query_context: Query context for semantic filtering
            current_entity: Current entity for context
            max_relations: Maximum relations to keep
            min_confidence: Minimum confidence threshold
            
        Returns:
            List of pruned relations with scores
        """
        if not relations:
            return []
        
        if not self.relation_stats:
            self._compute_relation_statistics()
        
        scored_relations = []
        
        for relation in relations:
            score = self._calculate_relation_score(
                relation, query_context, current_entity
            )
            
            if score >= min_confidence:
                scored_relations.append({
                    "relation": relation,
                    "score": score,
                    "components": self._get_score_components(relation, query_context)
                })
        
        # Sort by score and return top relations
        scored_relations.sort(key=lambda x: x["score"], reverse=True)
        return scored_relations[:max_relations]
    
    def _calculate_relation_score(self, 
                                relation: str, 
                                query_context: str = "",
                                current_entity: str = "") -> float:
        """Calculate composite score for relation"""
        
        # Get relation statistics
        stats = self.relation_stats.get(relation, {})
        
        # Component scores (0-1 each)
        biomedical_score = stats.get("biomedical_priority", 0.3)
        frequency_score = min(1.0, stats.get("frequency", 0) * 100)  # Normalize frequency
        connectivity_score = min(1.0, stats.get("connectivity", 0) / 1000)  # Normalize connectivity
        
        # Semantic relevance (simple keyword matching for now)
        semantic_score = self._calculate_semantic_relevance(relation, query_context)
        
        # Weighted combination
        weights = {
            "biomedical": 0.4,
            "semantic": 0.3,
            "frequency": 0.2,
            "connectivity": 0.1
        }
        
        composite_score = (
            weights["biomedical"] * biomedical_score +
            weights["semantic"] * semantic_score +
            weights["frequency"] * frequency_score +
            weights["connectivity"] * connectivity_score
        )
        
        return min(1.0, composite_score)
    
    def _calculate_semantic_relevance(self, relation: str, query_context: str) -> float:
        """Calculate semantic relevance between relation and query"""
        if not query_context:
            return 0.5  # Neutral score if no context
        
        query_lower = query_context.lower()
        relation_lower = relation.lower()
        
        # Direct keyword matching
        if relation_lower in query_lower:
            return 1.0
        
        # Semantic keyword groups
        semantic_groups = {
            "treatment": ["treats", "cures", "prevents", "therapy"],
            "causation": ["causes", "leads_to", "results_in"],
            "interaction": ["interacts_with", "affects", "regulates"],
            "location": ["located_in", "found_in", "occurs_in"],
            "structure": ["part_of", "contains", "composed_of"]
        }
        
        # Check if query contains keywords related to relation
        for group, keywords in semantic_groups.items():
            if relation_lower in keywords:
                for keyword in keywords:
                    if keyword in query_lower:
                        return 0.8
        
        # Partial matching
        relation_words = relation_lower.replace("_", " ").split()
        query_words = query_lower.split()
        
        matches = sum(1 for word in relation_words if word in query_words)
        if matches > 0:
            return 0.6 * (matches / len(relation_words))
        
        return 0.3  # Default low relevance
    
    def _get_score_components(self, relation: str, query_context: str) -> Dict[str, float]:
        """Get detailed score breakdown for debugging"""
        stats = self.relation_stats.get(relation, {})
        
        return {
            "biomedical_priority": stats.get("biomedical_priority", 0.3),
            "semantic_relevance": self._calculate_semantic_relevance(relation, query_context),
            "frequency_score": min(1.0, stats.get("frequency", 0) * 100),
            "connectivity_score": min(1.0, stats.get("connectivity", 0) / 1000),
            "raw_frequency": stats.get("frequency", 0),
            "raw_connectivity": stats.get("connectivity", 0)
        }
    
    def get_high_value_relations(self, top_k: int = 10) -> List[Dict[str, Any]]:
        """Get top-k highest value relations across the entire KG"""
        if not self.relation_stats:
            self._compute_relation_statistics()
        
        relations_with_scores = []
        for relation, stats in self.relation_stats.items():
            score = self._calculate_relation_score(relation)
            relations_with_scores.append({
                "relation": relation,
                "score": score,
                "stats": stats
            })
        
        relations_with_scores.sort(key=lambda x: x["score"], reverse=True)
        return relations_with_scores[:top_k]
    
    def filter_by_entity_types(self, 
                              relations: List[str], 
                              source_type: str = "", 
                              target_type: str = "") -> List[str]:
        """Filter relations based on entity types they typically connect"""
        if not source_type and not target_type:
            return relations
        
        # Entity type compatibility rules for biomedical domain
        type_compatible_relations = {
            ("Disease", "Drug"): ["treats", "prevents", "cures"],
            ("Drug", "Disease"): ["treats", "prevents", "cures", "causes"],
            ("Gene", "Disease"): ["causes", "associated_with", "affects"],
            ("Disease", "Gene"): ["affects", "regulates", "involves"],
            ("Drug", "Gene"): ["affects", "regulates", "targets"],
            ("Gene", "Drug"): ["targeted_by", "affected_by"],
            ("Anatomy", "Disease"): ["affected_by", "location_of"],
            ("Disease", "Anatomy"): ["affects", "located_in", "occurs_in"]
        }
        
        compatible = type_compatible_relations.get((source_type, target_type), relations)
        return [r for r in relations if r in compatible]
