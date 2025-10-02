"""
Edge Predictor - LLM-based Relation Prediction
==============================================

Uses LLM (GPT-3.5/Groq) for intelligent 0-hop relation prediction
with temperature-based calibration and structured JSON prompts.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import json
import duckdb
import numpy as np
from abc import ABC, abstractmethod
from openai import OpenAI
from langchain_groq import ChatGroq

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from config import settings

logger = logging.getLogger(__name__)

class BaseEdgePredictor(ABC):
    """Abstract base class for edge predictors."""
    
    @abstractmethod
    async def predict(self, entity: str, subquestion: str, intent: str, hop: int = 0) -> List[Tuple[str, float]]:
        """Predict relations for entity given subquestion and intent."""
        pass

class EdgePredictor(BaseEdgePredictor):
    """LLM-based edge prediction for intelligent KG traversal"""
    
    def __init__(self, T_seed: float = 0.85):
        self.llm = None
        self.kg_db = None
        self.T_seed = T_seed  # Temperature for 0-hop calibration
        self._initialize_llm()
        
    def _initialize_llm(self):
        """Initialize LLM client based on configuration"""
        try:
            if settings.LLM_PROVIDER == "groq":
                self.llm = ChatGroq(
                    groq_api_key=settings.GROQ_API_KEY,
                    model_name=settings.LLM_MODEL,
                    temperature=settings.LLM_TEMPERATURE,
                    max_tokens=settings.LLM_MAX_TOKENS
                )
                logger.info("✅ Groq LLM initialized for edge prediction")
            else:
                # Fallback to OpenAI
                self.llm = OpenAI(api_key=settings.OPENAI_API_KEY)
                logger.info("✅ OpenAI LLM initialized for edge prediction")
                
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            self.llm = None
    
    def connect_kg(self, db_path: str = None):
        """Connect to knowledge graph database"""
        try:
            db_path = db_path or settings.DB_PATH
            self.kg_db = duckdb.connect(db_path)
            logger.info("✅ Connected to KG database for edge prediction")
        except Exception as e:
            logger.error(f"Failed to connect to KG database: {e}")
    
    async def predict(self, entity: str, subquestion: str, intent: str, hop: int = 0) -> List[Tuple[str, float]]:
        """Predict potential relations for entity using OpenAI API with calibration."""
        if not self.llm:
            logger.warning("LLM not available, using fallback relation prediction")
            fallback = self._fallback_relation_prediction(entity, 5)
            return [(pred["relation"], pred["confidence"]) for pred in fallback]
        
        try:
            # Get available relations with prior scores
            available_relations = self._get_available_relations_with_priors(entity)
            if not available_relations:
                return []
            
            # Create structured JSON prompt
            prompt = self._build_structured_prompt(entity, subquestion, intent, available_relations)
            
            # Get LLM prediction
            if settings.LLM_PROVIDER == "groq":
                response = self.llm.invoke(prompt)
                prediction_text = response.content
            else:
                response = self.llm.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "Answer in strict JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.2,
                    response_format={"type": "json_object"}
                )
                prediction_text = response.choices[0].message.content
            
            # Parse and apply calibration
            predictions = self._parse_and_calibrate(prediction_text, available_relations, hop)
            return predictions
            
        except Exception as e:
            logger.error(f"Edge prediction failed: {e}")
            fallback = self._fallback_relation_prediction(entity, 5)
            return [(pred["relation"], pred["confidence"]) for pred in fallback]
    
    def predict_next_relations(self, 
                             current_entity: str, 
                             target_entity: str, 
                             query_context: str,
                             max_relations: int = 5,
                             intent: str = "factoid") -> List[Dict[str, Any]]:
        """
        Predict most likely relations to traverse from current entity
        
        Args:
            current_entity: Current entity in traversal
            target_entity: Target entity we're trying to reach
            query_context: Original query for context
            max_relations: Maximum relations to return
            
        Returns:
            List of predicted relations with confidence scores
        """
        if not self.llm:
            logger.warning("LLM not available, using fallback relation prediction")
            return self._fallback_relation_prediction(current_entity, max_relations)
        
        try:
            # Get available relations from current entity
            available_relations = self._get_available_relations(current_entity)
            if not available_relations:
                return []
            
            # Create LLM prompt for relation prediction with intent (legacy method)
            prompt = self._create_relation_prediction_prompt(
                current_entity, target_entity, query_context, available_relations, intent
            )
            
            # Get LLM prediction
            if settings.LLM_PROVIDER == "groq":
                response = self.llm.invoke(prompt)
                prediction_text = response.content
            else:
                response = self.llm.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=settings.LLM_TEMPERATURE,
                    max_tokens=settings.LLM_MAX_TOKENS
                )
                prediction_text = response.choices[0].message.content
            
            # Parse LLM response
            predicted_relations = self._parse_relation_predictions(prediction_text, available_relations)
            
            return predicted_relations[:max_relations]
            
        except Exception as e:
            logger.error(f"LLM relation prediction failed: {e}")
            return self._fallback_relation_prediction(current_entity, max_relations)
    
    def _create_relation_prediction_prompt(self, current_entity: str, target_entity: str, 
                                         query_context: str, available_relations: List[str], intent: str = "factoid") -> str:
        """Create prompt for LLM relation prediction"""
        return f"""
You are an expert biomedical knowledge graph navigator. Given a current entity, target entity, and query context, predict the most likely relations to traverse.

Current Entity: {current_entity}
Target Entity: {target_entity}
Query Context: {query_context}

Available Relations from {current_entity}:
{', '.join(available_relations)}

Task: Rank the available relations by likelihood of leading to relevant information for the query. Consider:
1. Semantic relevance to the query
2. Likelihood of connecting to target entity
3. Biomedical domain knowledge
4. Query intent: {intent} (factoid=direct facts, causal=cause-effect chains, therapeutic=treatments)

Respond with JSON format:
{{
  "predictions": [
    {{
      "relation": "relation_name",
      "confidence": 0.95,
      "reasoning": "why this relation is likely relevant"
    }}
  ]
}}

Rank all available relations, most likely first.
"""
    
    def _parse_relation_predictions(self, prediction_text: str, available_relations: List[str]) -> List[Dict[str, Any]]:
        """Parse LLM prediction response"""
        try:
            # Try to extract JSON from response
            if "```json" in prediction_text:
                json_start = prediction_text.find("```json") + 7
                json_end = prediction_text.find("```", json_start)
                json_text = prediction_text[json_start:json_end].strip()
            elif "{" in prediction_text:
                json_start = prediction_text.find("{")
                json_end = prediction_text.rfind("}") + 1
                json_text = prediction_text[json_start:json_end]
            else:
                raise ValueError("No JSON found in response")
            
            parsed = json.loads(json_text)
            predictions = parsed.get("predictions", [])
            
            # Validate predictions against available relations
            valid_predictions = []
            for pred in predictions:
                if pred.get("relation") in available_relations:
                    valid_predictions.append({
                        "relation": pred["relation"],
                        "confidence": float(pred.get("confidence", 0.5)),
                        "reasoning": pred.get("reasoning", ""),
                        "source": "llm_prediction"
                    })
            
            return valid_predictions
            
        except Exception as e:
            logger.error(f"Failed to parse LLM predictions: {e}")
            return self._fallback_relation_prediction(available_relations[0] if available_relations else "", len(available_relations))
    
    def _get_available_relations(self, entity: str) -> List[str]:
        """Get available relations from entity in KG"""
        if not self.kg_db:
            self.connect_kg()
        
        try:
            result = self.kg_db.execute("""
                SELECT DISTINCT predicate 
                FROM triples 
                WHERE subject = ? OR object = ?
            """, (entity, entity)).fetchall()
            
            relations = [row[0] for row in result]
            return relations
            
        except Exception as e:
            logger.error(f"Failed to get available relations: {e}")
            return []
    
    def _fallback_relation_prediction(self, current_entity: str, max_relations: int) -> List[Dict[str, Any]]:
        """Fallback relation prediction using heuristics"""
        available_relations = self._get_available_relations(current_entity)
        
        # Simple heuristic: prioritize common biomedical relations
        priority_relations = [
            "treats", "causes", "associated_with", "interacts_with", 
            "part_of", "located_in", "affects", "regulates"
        ]
        
        predictions = []
        for relation in available_relations:
            confidence = 0.8 if relation in priority_relations else 0.5
            predictions.append({
                "relation": relation,
                "confidence": confidence,
                "reasoning": "heuristic_fallback",
                "source": "fallback"
            })
        
        # Sort by confidence
        predictions.sort(key=lambda x: x["confidence"], reverse=True)
        # Apply confidence calibration
        for pred in predictions:
            if self.kg_db:
                # Get entity type for calibration
                try:
                    entity_type_result = self.kg_db.execute(
                        "SELECT DISTINCT subject_type FROM triples WHERE subject = ? LIMIT 1",
                        (current_entity,)
                    ).fetchone()
                    entity_type = entity_type_result[0] if entity_type_result else ""
                    
                    calibrated_conf = self.get_relation_confidence_calibration(pred["relation"], entity_type)
                    pred["confidence"] = (pred["confidence"] + calibrated_conf) / 2  # Average with calibrated
                    
                except Exception:
                    pass
        
        return predictions[:max_relations]
    
    def calculate_path_confidence(self, path: List[Tuple[str, str, str]]) -> float:
        """Calculate confidence score for a complete path"""
        if not path:
            return 0.0
        
        # Simple confidence calculation based on path length and relation types
        base_confidence = 1.0
        length_penalty = 0.1 * len(path)  # Longer paths are less confident
        
        return max(0.1, base_confidence - length_penalty)
    
    def predict_relations_batch(self, entities_contexts: List[Tuple[str, str, str]], max_relations: int = 5) -> List[List[Dict[str, Any]]]:
        """Batch relation prediction for multiple entity-context pairs"""
        batch_results = []
        
        for current_entity, target_entity, query_context in entities_contexts:
            predictions = self.predict_next_relations(
                current_entity, target_entity, query_context, max_relations
            )
            batch_results.append(predictions)
        
        return batch_results
    
    def get_relation_confidence_calibration(self, relation: str, entity_type: str = "") -> float:
        """Get calibrated confidence for relation based on entity type"""
        # Base confidence from biomedical knowledge
        base_confidences = {
            "treats": 0.9,
            "causes": 0.85,
            "prevents": 0.8,
            "interacts_with": 0.75,
            "regulates": 0.7,
            "associated_with": 0.6,
            "part_of": 0.65,
            "located_in": 0.6
        }
        
        base_conf = base_confidences.get(relation, 0.5)
        
        # Entity type specific adjustments
        if entity_type.lower() == "drug" and relation in ["treats", "causes", "prevents"]:
            base_conf *= 1.1
        elif entity_type.lower() == "gene" and relation in ["regulates", "associated_with"]:
            base_conf *= 1.1
        elif entity_type.lower() == "disease" and relation in ["causes", "presents"]:
            base_conf *= 1.1
        
        return min(1.0, base_conf)
    
    def _get_available_relations_with_priors(self, entity: str) -> List[Tuple[str, float]]:
        """Get available relations with prior scores for structured prompts."""
        if not self.kg_db:
            self.connect_kg()
        
        try:
            # Get relations with statistical priors from relation pruner tables
            result = self.kg_db.execute("""
                SELECT DISTINCT t.predicate, 
                       COALESCE(tp.prior, 0.1) as prior_score
                FROM triples t
                LEFT JOIN type_rel_prior tp ON t.subject_type = tp.source_type 
                                            AND t.predicate = tp.predicate
                WHERE t.subject = ?
                ORDER BY prior_score DESC
            """, (entity,)).fetchall()
            
            return [(row[0], float(row[1])) for row in result]
            
        except Exception as e:
            logger.error(f"Failed to get relations with priors: {e}")
            # Fallback to basic relations
            basic_relations = self._get_available_relations(entity)
            return [(rel, 0.5) for rel in basic_relations]
    
    def _build_structured_prompt(self, entity: str, subquestion: str, intent: str, 
                               cand_relations: List[Tuple[str, float]]) -> str:
        """Build structured prompt matching reference methodology."""
        rel_list = [{"relation": r, "prior_score": round(s, 3)} for r, s in cand_relations]
        
        return f"""
You are a biomedical KG traversal oracle.

Entity: {entity}
Subquestion: {subquestion}
Intent: {intent}

You may ONLY choose from these relations (with prior scores):
{json.dumps(rel_list, indent=2)}

Return a JSON object where keys are relations and values are probabilities in [0,1] 
that expanding this relation will progress toward answering the subquestion.
The sum of probabilities should be <= 1 (sparse is okay).
"""
    
    def _parse_and_calibrate(self, prediction_text: str, available_relations: List[Tuple[str, float]], 
                           hop: int) -> List[Tuple[str, float]]:
        """Parse LLM response and apply 0-hop calibration."""
        try:
            # Parse JSON response
            if "```json" in prediction_text:
                json_start = prediction_text.find("```json") + 7
                json_end = prediction_text.find("```", json_start)
                json_text = prediction_text[json_start:json_end].strip()
            elif "{" in prediction_text:
                json_start = prediction_text.find("{")
                json_end = prediction_text.rfind("}") + 1
                json_text = prediction_text[json_start:json_end]
            else:
                raise ValueError("No JSON found in response")
            
            parsed = json.loads(json_text)
            
            # Extract predictions
            items = []
            for rel, _ in available_relations:
                prob = float(parsed.get(rel, 0.0))
                items.append((rel, prob))
            
            # Apply 0-hop calibration if hop == 0
            if hop == 0 and self.T_seed is not None:
                calibrated_items = []
                for rel, p in items:
                    if p > 0:
                        # Temperature scaling calibration
                        eps = 1e-6
                        p = min(max(p, eps), 1 - eps)
                        logit = np.log(p / (1 - p))
                        calibrated_logit = logit / self.T_seed
                        calibrated_p = float(1 / (1 + np.exp(-calibrated_logit)))
                        calibrated_items.append((rel, calibrated_p))
                    else:
                        calibrated_items.append((rel, p))
                items = calibrated_items
            
            # Sort by confidence
            items.sort(key=lambda x: x[1], reverse=True)
            return items
            
        except Exception as e:
            logger.error(f"Failed to parse and calibrate predictions: {e}")
            # Fallback to uniform distribution
            return [(rel, 0.5) for rel, _ in available_relations[:5]]
