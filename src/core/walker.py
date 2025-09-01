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
from typing import List, Dict, Any, Set, Tuple, Optional
import duckdb
import math
from collections import defaultdict, deque
import networkx as nx

from .edge_predictor import EdgePredictor
from .relation_pruner import RelationPruner
from ..config import settings

logger = logging.getLogger(__name__)

class KGWalker:
    """Intelligent knowledge graph walker with entropy-based navigation"""
    
    def __init__(self):
        self.kg_db = None
        self.edge_predictor = EdgePredictor()
        self.relation_pruner = RelationPruner()
        self.graph = None
        self.entity_degrees = {}
        
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
             confidence_threshold: float = 0.3) -> List[Dict[str, Any]]:
        """
        Perform confidence-guided graph walk
        
        Args:
            start_entities: Starting entities for traversal
            query_context: Query context for guidance
            max_hops: Maximum number of hops
            max_paths: Maximum paths to explore
            confidence_threshold: Minimum confidence for path continuation
            
        Returns:
            List of discovered paths with confidence scores
        """
        if not self.kg_db:
            self.connect_kg()
        
        all_paths = []
        
        for start_entity in start_entities:
            if start_entity not in self.graph:
                logger.warning(f"Start entity '{start_entity}' not found in KG")
                continue
            
            # Perform entropy-guided BFS from each start entity
            entity_paths = self._entropy_guided_search(
                start_entity, query_context, max_hops, max_paths, confidence_threshold
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
                              confidence_threshold: float) -> List[Dict[str, Any]]:
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
                    "entities_found": self._extract_path_entities(current_path)
                })
                continue
            
            # Get next possible steps using edge predictor
            next_steps = self._get_next_steps(current_entity, query_context)
            
            # Expand search with entropy-based prioritization
            for next_entity, relation, step_confidence in next_steps:
                if next_entity in current_path["entities"]:  # Avoid cycles
                    continue
                
                # Calculate path confidence
                new_confidence = current_confidence * step_confidence * self._calculate_entropy_bonus(next_entity)
                
                if new_confidence >= confidence_threshold:
                    new_path = {
                        "entities": current_path["entities"] + [next_entity],
                        "relations": current_path["relations"] + [relation],
                        "confidence": new_confidence,
                        "hops": current_hops + 1
                    }
                    
                    heappush(search_queue, (-new_confidence, id(new_path), new_path))
        
        return completed_paths
    
    def _get_next_steps(self, current_entity: str, query_context: str) -> List[Tuple[str, str, float]]:
        """Get next possible steps from current entity"""
        try:
            # Get all neighbors and relations
            neighbors = self.kg_db.execute("""
                SELECT object, predicate FROM triples WHERE subject = ?
                UNION
                SELECT subject, predicate FROM triples WHERE object = ?
            """, (current_entity, current_entity)).fetchall()
            
            if not neighbors:
                return []
            
            # Get relation predictions from edge predictor
            available_relations = list(set(relation for _, relation in neighbors))
            predicted_relations = self.edge_predictor.predict_next_relations(
                current_entity, "", query_context, max_relations=10
            )
            
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
