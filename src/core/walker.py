"""
KG Walker - Entropy-based Graph Traversal
========================================

Implements intelligent knowledge graph traversal using:
- Entropy-based node selection
- LLM-guided relation prediction
- Confidence-guided search algorithms
- Multi-hop reasoning paths
"""

import logging
from typing import List, Dict, Any, Optional, Set, Tuple
import duckdb
import networkx as nx
import json
from dataclasses import dataclass, field
from collections import defaultdict, deque
import math
import time
import heapq
from concurrent.futures import ThreadPoolExecutor, as_completed
from fuzzywuzzy import fuzz

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from config import settings
from .edge_predictor import EdgePredictor
from .relation_pruner import RelationPruner

logger = logging.getLogger(__name__)

@dataclass
class PathState:
    """State representation for priority queue traversal."""
    entities: Tuple[str, ...]  # Path of entities
    confidence: float
    depth: int
    last_relation: str = ""
    
    def __lt__(self, other):
        # Priority: (1 - confidence) / (depth + 1) for min-heap
        self_priority = (1 - self.confidence) / (self.depth + 1)
        other_priority = (1 - other.confidence) / (other.depth + 1)
        return self_priority < other_priority
    
    def get_cycle_key(self) -> Tuple[str, ...]:
        """Get key for cycle detection using entity path."""
        return self.entities

# Intent-based traversal policies
INTENT_POLICY = {
    "factoid": dict(top_k_rel=6, max_nei=3, max_hops=2, time_budget=10.0),
    "enumeration": dict(top_k_rel=14, max_nei=8, max_hops=2, time_budget=15.0),
    "causal": dict(top_k_rel=10, max_nei=4, max_hops=4, time_budget=20.0),
    "therapeutic": dict(top_k_rel=8, max_nei=5, max_hops=3, time_budget=15.0),
}

class KGWalker:
    """Intelligent knowledge graph walker with entropy-based navigation"""
    
    def __init__(self):
        self.kg_db = None
        self.graph = None
        self.edge_predictor = EdgePredictor()
        self.relation_pruner = RelationPruner()
        self.entity_cache = {}
        self.path_cache = {}
        self.expansion_count = 0
        self.start_time = None
        
    def connect_kg(self, db_path: str = None):
        """Connect to knowledge graph database"""
        try:
            db_path = db_path or settings.DB_PATH
            self.kg_db = duckdb.connect(db_path)
            self.edge_predictor.connect_kg(db_path)
            self.relation_pruner.connect_kg(db_path)
            self._build_graph_structure()
            logger.info("✅ KG Walker connected to database")
        except Exception as e:
            logger.error(f"Failed to connect KG Walker to database: {e}")
    
    def _build_graph_structure(self):
        """Build NetworkX graph for topology analysis"""
        try:
            self.graph = nx.Graph()
            
            # Add all triples as edges
            result = self.kg_db.execute("SELECT subject, predicate, object FROM triples").fetchall()
            
            for subject, predicate, object_name in result:
                self.graph.add_edge(subject, object_name, relation=predicate)
            
            # Compute entity degrees for entropy calculation
            self.entity_degrees = dict(self.graph.degree())
            
            logger.info(f"✅ Built graph with {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
            
        except Exception as e:
            logger.error(f"Failed to build graph structure: {e}")
    
    def walk(self, 
             start_entities: List[str], 
             query_context: str,
             max_hops: int = 3,
             max_paths: int = 10,
             confidence_threshold: float = 0.3,
             intent: str = "factoid") -> List[Dict[str, Any]]:
        """
        Perform confidence-guided graph walk with adaptive thresholds
        
        Args:
            start_entities: Starting entities for traversal
            query_context: Query context for guidance
            max_hops: Maximum number of hops
            max_paths: Maximum paths to explore
            confidence_threshold: Minimum confidence for path continuation
            intent: Query intent for adaptive threshold adjustment
            
        Returns:
            List of discovered paths with confidence scores
        """
        if not self.kg_db:
            self.connect_kg()
        
        # Adaptive threshold based on intent
        adaptive_threshold = self._adjust_threshold_by_intent(confidence_threshold, intent)
        
        # Entity resolution if start_entities is empty or contains query terms
        if not start_entities or any(len(entity.split()) > 3 for entity in start_entities):
            start_entities = self._resolve_entities_from_query(query_context)
        
        all_paths = []
        
        for start_entity in start_entities:
            if start_entity not in self.graph:
                logger.warning(f"Start entity '{start_entity}' not found in KG")
                continue
            
            # Perform entropy-guided BFS from each start entity
            entity_paths = self._entropy_guided_search(
                start_entity, query_context, max_hops, max_paths, adaptive_threshold, intent
            )
            
            all_paths.extend(entity_paths)
        
        # Sort paths by confidence and return top results
        all_paths.sort(key=lambda x: x["confidence"], reverse=True)
        return all_paths[:max_paths]
    
    def _entropy_guided_search(self, 
                              start_entity: str, 
                              query_context: str,
                              max_hops: int,
                              max_paths: int,
                              confidence_threshold: float,
                              intent: str = "factoid") -> List[Dict[str, Any]]:
        """Perform entropy-guided search from start entity"""
        
        # Priority queue: (negative_confidence, path_info)
        from heapq import heappush, heappop
        
        search_queue = []
        completed_paths = []
        visited_states = set()
        
        # Initialize with start entity
        initial_path = {
            "entities": [start_entity],
            "relations": [],
            "confidence": 1.0,
            "hops": 0
        }
        
        heappush(search_queue, (-1.0, id(initial_path), initial_path))
        
        while search_queue and len(completed_paths) < max_paths:
            neg_confidence, _, current_path = heappop(search_queue)
            current_confidence = -neg_confidence
            
            if current_confidence < confidence_threshold:
                continue
            
            current_entity = current_path["entities"][-1]
            current_hops = current_path["hops"]
            
            # Create state key to avoid cycles
            state_key = (current_entity, tuple(current_path["entities"]))
            if state_key in visited_states:
                continue
            visited_states.add(state_key)
            
            # If we've reached max hops, add to completed paths
            if current_hops >= max_hops:
                completed_paths.append({
                    "path": current_path["entities"],
                    "relations": current_path["relations"],
                    "confidence": current_confidence,
                    "hops": current_hops,
                    "entities_found": self._extract_path_entities(current_path),
                    "reasoning_type": self._determine_reasoning_type(current_path, intent),
                    "entropy_score": current_path.get("entropy_score", 0.0)
                })
                continue
            
            # Get next possible steps using edge predictor
            next_steps = self._get_next_steps(current_entity, query_context)
            
            # Expand search with entropy-based prioritization
            for next_entity, relation, step_confidence in next_steps:
                if next_entity in current_path["entities"]:  # Avoid cycles
                    continue
                
                # Calculate path confidence with enhanced entropy
                entropy_bonus = self._calculate_enhanced_entropy_bonus(next_entity, query_context, intent)
                new_confidence = current_confidence * step_confidence * entropy_bonus
                
                if new_confidence >= confidence_threshold:
                    new_path = {
                        "entities": current_path["entities"] + [next_entity],
                        "relations": current_path["relations"] + [relation],
                        "confidence": new_confidence,
                        "hops": current_hops + 1
                    }
                    
                    heappush(search_queue, (-new_confidence, id(new_path), new_path))
        
        return completed_paths
    
    def _entity_exists(self, entity: str) -> bool:
        """Check if entity exists in the knowledge graph."""
        if not self.kg_db:
            return False
        
        try:
            result = self.kg_db.execute(
                "SELECT 1 FROM nodes WHERE name = ? LIMIT 1", 
                (entity,)
            ).fetchone()
            return result is not None
        except Exception as e:
            logger.error(f"Failed to check entity existence for {entity}: {e}")
            return False
    
    def _get_neighbors(self, entity: str, relation: str) -> List[str]:
        """Get neighbors connected via specific relation."""
        try:
            result = self.kg_db.execute("""
                SELECT object FROM triples 
                WHERE subject = ? AND predicate = ?
                ORDER BY object
            """, (entity, relation)).fetchall()
            
            return [row[0] for row in result]
            
        except Exception as e:
            logger.error(f"Failed to get neighbors for {entity} via {relation}: {e}")
            return []
            
            # Create relation confidence map
            relation_confidence = {}
            for pred in predicted_relations:
                relation_confidence[pred["relation"]] = pred["confidence"]
            
            # Build next steps with confidence scores
            next_steps = []
            for neighbor, relation in neighbors:
                confidence = relation_confidence.get(relation, 0.3)  # Default confidence
                next_steps.append((neighbor, relation, confidence))
            
            # Sort by confidence and return top steps
            next_steps.sort(key=lambda x: x[2], reverse=True)
            return next_steps[:5]  # Limit branching factor
            
        except Exception as e:
            logger.error(f"Failed to get next steps: {e}")
            return []
    
    def _calculate_entropy_bonus(self, entity: str) -> float:
        """Calculate entropy-based bonus for entity selection"""
        if not self.entity_degrees:
            return 1.0
        
        degree = self.entity_degrees.get(entity, 1)
        
        # Entities with moderate degree get higher bonus (not too common, not too rare)
        if degree == 1:
            return 0.7  # Too specific
        elif degree < 5:
            return 1.0  # Good specificity
        elif degree < 20:
            return 0.9  # Moderate connectivity
        elif degree < 100:
            return 0.8  # High connectivity
        else:
            return 0.6  # Too general
    
    def _calculate_enhanced_entropy_bonus(self, entity: str, query_context: str, intent: str) -> float:
        """Enhanced entropy calculation with query context and intent awareness"""
        base_entropy = self._calculate_entropy_bonus(entity)
        
        # Query relevance bonus
        query_bonus = self._calculate_query_relevance(entity, query_context)
        
        # Intent-specific adjustment
        intent_multiplier = {
            "factoid": 1.0,
            "causal": 1.2,  # Boost for causal reasoning
            "enumeration": 0.9,  # Slight penalty for enumeration
            "comparison": 1.1
        }.get(intent, 1.0)
        
        # Information-theoretic entropy calculation
        degree = self.entity_degrees.get(entity, 1)
        total_entities = len(self.entity_degrees) if self.entity_degrees else 1
        
        # Shannon entropy: -p * log2(p)
        probability = degree / sum(self.entity_degrees.values()) if self.entity_degrees else 0.5
        shannon_entropy = -probability * math.log2(probability + 1e-10)  # Add small epsilon
        
        # Normalize shannon entropy (typical range 0-10, normalize to 0-1)
        normalized_entropy = min(1.0, shannon_entropy / 10.0)
        
        # Combine all factors
        enhanced_bonus = (base_entropy * 0.4 + 
                         query_bonus * 0.3 + 
                         normalized_entropy * 0.3) * intent_multiplier
        
        return min(1.0, enhanced_bonus)
    
    def _extract_path_entities(self, path_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract entities with metadata from path"""
        entities = []
        
        try:
            for entity in path_info["entities"]:
                # Get entity type and additional info
                result = self.kg_db.execute("""
                    SELECT DISTINCT subject_type FROM triples WHERE subject = ?
                    UNION
                    SELECT DISTINCT object_type FROM triples WHERE object = ?
                """, (entity, entity)).fetchall()
                
                types = [row[0] for row in result]
                
                entities.append({
                    "name": entity,
                    "types": types,
                    "degree": self.entity_degrees.get(entity, 0)
                })
        
        except Exception as e:
            logger.error(f"Failed to extract path entities: {e}")
        
        return entities
    
    def _adjust_threshold_by_intent(self, base_threshold: float, intent: str) -> float:
        """Adjust confidence threshold based on query intent"""
        adjustments = {
            "factoid": 1.0,      # Standard threshold
            "causal": 0.8,       # Lower threshold for causal chains
            "enumeration": 1.2,  # Higher threshold for enumeration
            "comparison": 0.9    # Slightly lower for comparisons
        }
        return base_threshold * adjustments.get(intent, 1.0)
    
    def _resolve_entities_from_query(self, query: str) -> List[str]:
        """Simple entity resolution from query text"""
        if not self.kg_db:
            return []
        
        try:
            # Extract potential entities using simple keyword matching
            query_words = query.lower().split()
            entities = []
            
            # Search for entities that match query terms
            for word in query_words:
                if len(word) > 3:  # Skip short words
                    result = self.kg_db.execute("""
                        SELECT DISTINCT subject FROM triples 
                        WHERE LOWER(subject) LIKE ? 
                        LIMIT 3
                    """, (f"%{word}%",)).fetchall()
                    
                    entities.extend([row[0] for row in result])
            
            # Remove duplicates and return top matches
            return list(set(entities))[:5]
            
        except Exception as e:
            logger.error(f"Entity resolution failed: {e}")
            return []
    
    def _calculate_query_relevance(self, entity: str, query_context: str) -> float:
        """Calculate relevance between entity and query context"""
        if not query_context:
            return 0.5
        
        entity_lower = entity.lower()
        query_lower = query_context.lower()
        
        # Direct match
        if entity_lower in query_lower:
            return 1.0
        
        # Word overlap
        entity_words = set(entity_lower.replace('_', ' ').split())
        query_words = set(query_lower.split())
        
        overlap = len(entity_words.intersection(query_words))
        if overlap > 0:
            return 0.7 + (0.3 * overlap / len(entity_words))
        
        return 0.3  # Default low relevance
    
    def _determine_reasoning_type(self, path_info: Dict[str, Any], intent: str) -> str:
        """Determine the type of reasoning represented by the path"""
        relations = path_info.get("relations", [])
        
        # Causal reasoning indicators
        causal_relations = {"causes", "leads_to", "results_in", "triggers"}
        if any(rel in causal_relations for rel in relations):
            return "causal"
        
        # Treatment reasoning
        treatment_relations = {"treats", "cures", "prevents", "alleviates"}
        if any(rel in treatment_relations for rel in relations):
            return "therapeutic"
        
        # Structural reasoning
        structural_relations = {"part_of", "located_in", "contains"}
        if any(rel in structural_relations for rel in relations):
            return "structural"
        
        # Default based on intent
        return {
            "causal": "causal",
            "factoid": "factual",
            "enumeration": "enumerative",
            "comparison": "comparative"
        }.get(intent, "associative")
    
    def find_shortest_paths(self, start_entity: str, target_entities: List[str], max_length: int = 4) -> List[Dict[str, Any]]:
        """Find shortest paths between start and target entities"""
        if not self.graph:
            return []
        
        paths = []
        
        for target in target_entities:
            if target not in self.graph:
                continue
            
            try:
                # Use NetworkX to find shortest path
                if nx.has_path(self.graph, start_entity, target):
                    path = nx.shortest_path(self.graph, start_entity, target)
                    
                    if len(path) <= max_length + 1:  # +1 because path includes both endpoints
                        # Get relations along the path
                        relations = []
                        for i in range(len(path) - 1):
                            edge_data = self.graph.get_edge_data(path[i], path[i+1])
                            relations.append(edge_data.get("relation", "unknown"))
                        
                        paths.append({
                            "path": path,
                            "relations": relations,
                            "length": len(path) - 1,
                            "target": target,
                            "confidence": 1.0 / len(path)  # Shorter paths get higher confidence
                        })
                        
            except nx.NetworkXNoPath:
                continue
            except Exception as e:
                logger.error(f"Error finding path to {target}: {e}")
        
        return sorted(paths, key=lambda x: x["confidence"], reverse=True)
    
    def get_entity_neighborhood(self, entity: str, radius: int = 1) -> Dict[str, Any]:
        """Get neighborhood information for entity"""
        if not self.kg_db:
            self.connect_kg()
        
        try:
            # Get direct neighbors
            result = self.kg_db.execute("""
                SELECT object, predicate, 'outgoing' as direction FROM triples WHERE subject = ?
                UNION
                SELECT subject, predicate, 'incoming' as direction FROM triples WHERE object = ?
            """, (entity, entity)).fetchall()
            
            neighbors = []
            relations = set()
            
            for neighbor, relation, direction in result:
                neighbors.append({
                    "entity": neighbor,
                    "relation": relation,
                    "direction": direction
                })
                relations.add(relation)
            
            return {
                "entity": entity,
                "neighbors": neighbors,
                "neighbor_count": len(neighbors),
                "unique_relations": list(relations),
                "degree": len(neighbors)
            }
            
        except Exception as e:
            logger.error(f"Failed to get entity neighborhood: {e}")
            return {"entity": entity, "neighbors": [], "neighbor_count": 0}
    
    async def walk_batch(self, queries: List[str], max_hops: int = 3, max_paths_per_query: int = 10, confidence_threshold: float = 0.3) -> List[List[Dict[str, Any]]]:
        """Batch processing for multiple queries with shared entity resolution"""
        batch_results = []
        
        for query in queries:
            # Resolve entities for each query
            entities = self._resolve_entities_from_query(query)
            
            # Perform walk for this query
            paths = self.walk(
                start_entities=entities,
                query_context=query,
                max_hops=max_hops,
                max_paths=max_paths_per_query,
                confidence_threshold=confidence_threshold
            )
            
            batch_results.append(paths)
        
        return batch_results
    
    def walk_priority_queue(self,
                          query: str,
                          seed_entities: List[str],
                          intent: str = "factoid",
                          max_expansions: int = 1000) -> List[Dict[str, Any]]:
        """Priority queue-based traversal matching reference methodology."""
        if not self.kg_db:
            self.connect_kg()
        
        # Get intent policy
        policy = INTENT_POLICY.get(intent, INTENT_POLICY["factoid"])
        
        # Initialize tracking
        self.expansion_count = 0
        self.start_time = time.time()
        
        # Priority queue: min-heap with PathState objects
        priority_queue = []
        visited_paths = set()  # For cycle detection
        results = []
        
        # Initialize with seed entities
        for entity in seed_entities:
            if self._entity_exists(entity):
                path_state = PathState(
                    entities=(entity,),
                    confidence=1.0,
                    depth=0
                )
                heapq.heappush(priority_queue, path_state)
                
                # Add seed entity as result
                results.append({
                    "entity": entity,
                    "path": [entity],
                    "confidence": 1.0,
                    "depth": 0,
                    "reasoning": f"Seed entity for {intent} query"
                })
        
        # Main traversal loop
        while (priority_queue and 
               self.expansion_count < max_expansions and 
               time.time() - self.start_time < policy["time_budget"] and
               len(results) < policy["max_nei"] * 10):  # Reasonable result limit
            
            current_state = heapq.heappop(priority_queue)
            
            # Check for cycles
            cycle_key = current_state.get_cycle_key()
            if cycle_key in visited_paths:
                continue
            visited_paths.add(cycle_key)
            
            # Early stopping if depth exceeded
            if current_state.depth >= policy["max_hops"]:
                continue
            
            # Get current entity
            current_entity = current_state.entities[-1]
            
            # Get relations using relation pruner
            relations_with_scores = self.relation_pruner.get_relations_for_node(
                current_entity, query, intent
            )
            
            # Limit relations based on policy
            top_relations = relations_with_scores[:policy["top_k_rel"]]
            
            # Expand each relation
            expansion_count = 0
            for relation, rel_score in top_relations:
                if expansion_count >= policy["max_nei"]:
                    break
                
                # Get neighbors
                neighbors = self._get_neighbors(current_entity, relation)
                
                for neighbor in neighbors[:policy["max_nei"]]:
                    # Calculate path confidence
                    path_confidence = current_state.confidence * rel_score * 0.9  # Decay factor
                    
                    # Create new path state
                    new_entities = current_state.entities + (neighbor,)
                    new_state = PathState(
                        entities=new_entities,
                        confidence=path_confidence,
                        depth=current_state.depth + 1,
                        last_relation=relation
                    )
                    
                    # Add to queue if confidence is reasonable
                    if path_confidence > 0.1:
                        heapq.heappush(priority_queue, new_state)
                    
                    # Add to results
                    results.append({
                        "entity": neighbor,
                        "path": list(new_entities),
                        "confidence": path_confidence,
                        "depth": new_state.depth,
                        "relation": relation,
                        "reasoning": f"Reached via {relation} from {current_entity}"
                    })
                    
                    expansion_count += 1
            
            self.expansion_count += 1
        
        # Sort results by confidence and return
        results.sort(key=lambda x: x["confidence"], reverse=True)
        
        logger.info(f"Priority queue traversal completed: {len(results)} results, "
                   f"{self.expansion_count} expansions, "
                   f"{time.time() - self.start_time:.2f}s")
        
        return results
    
    def walk_with_adaptive_threshold(self, 
                                   query: str,
                                   seed_entities: List[str],
                                   intent: str = "factoid",
                                   initial_threshold: float = 0.7,
                                   min_threshold: float = 0.3,
                                   threshold_decay: float = 0.1,
                                   max_results: int = 50) -> List[Dict[str, Any]]:
        """Perform entropy-guided search from start entity"""
        
        # Priority queue: (negative_confidence, path_info)
        from heapq import heappush, heappop
        
        search_queue = []
        completed_paths = []
        visited_states = set()
        
        # Initialize with seed entities
        for entity in seed_entities:
            if self._entity_exists(entity):
                initial_path = {
                    "entities": [entity],
                    "relations": [],
                    "confidence": 1.0,
                    "hops": 0
                }
                
                heappush(search_queue, (-1.0, id(initial_path), initial_path))
        
        # Main traversal loop
        while search_queue and len(completed_paths) < max_results:
            neg_confidence, _, current_path = heappop(search_queue)
            current_confidence = -neg_confidence
            
            if current_confidence < min_threshold:
                continue
            
            current_entity = current_path["entities"][-1]
            current_hops = current_path["hops"]
            
            # Create state key to avoid cycles
            state_key = (current_entity, tuple(current_path["entities"]))
            if state_key in visited_states:
                continue
            visited_states.add(state_key)
            
            # If we've reached max hops, add to completed paths
            if current_hops >= 5:  # Default max hops
                completed_paths.append({
                    "path": current_path["entities"],
                    "relations": current_path["relations"],
                    "confidence": current_confidence,
                    "hops": current_hops,
                    "entities_found": self._extract_path_entities(current_path),
                    "reasoning_type": self._determine_reasoning_type(current_path, intent),
                    "entropy_score": current_path.get("entropy_score", 0.0)
                })
                continue
            
            # Get next possible steps using edge predictor
            next_steps = self._get_next_steps(current_entity, query)
            
            # Expand search with entropy-based prioritization
            for next_entity, relation, step_confidence in next_steps:
                if next_entity in current_path["entities"]:  # Avoid cycles
                    continue
                
                # Calculate path confidence with enhanced entropy
                entropy_bonus = self._calculate_enhanced_entropy_bonus(next_entity, query, intent)
                new_confidence = current_confidence * step_confidence * entropy_bonus
                
                if new_confidence >= min_threshold:
                    new_path = {
                        "entities": current_path["entities"] + [next_entity],
                        "relations": current_path["relations"] + [relation],
                        "confidence": new_confidence,
                        "hops": current_hops + 1
                    }
                    
                    heappush(search_queue, (-new_confidence, id(new_path), new_path))
        
        return completed_paths
    
    def walk_with_adaptive_threshold(self, start_entities: List[str], query_context: str, intent: str, max_hops: int = 3, max_paths: int = 10) -> List[Dict[str, Any]]:
        """Walk with adaptive threshold based on query characteristics"""
        # Analyze query complexity to set threshold
        base_threshold = 0.3
        
        # Adjust based on query length and complexity
        query_words = len(query_context.split())
        if query_words > 10:
            base_threshold *= 0.9  # Lower threshold for complex queries
        elif query_words < 5:
            base_threshold *= 1.1  # Higher threshold for simple queries
        
        return self.walk(
            start_entities=start_entities,
            query_context=query_context,
            max_hops=max_hops,
            max_paths=max_paths,
            confidence_threshold=base_threshold,
            intent=intent
        )
