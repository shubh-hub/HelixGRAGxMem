"""
Edge Predictor - LLM-based Relation Prediction
==============================================

Uses LLM (GPT-3.5/Groq) for intelligent 0-hop relation prediction
and next-node selection in knowledge graph traversal.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import json
import duckdb
from openai import OpenAI
from langchain_groq import ChatGroq

from ..config import settings

logger = logging.getLogger(__name__)

class EdgePredictor:
    """LLM-based edge prediction for intelligent KG traversal"""
    
    def __init__(self):
        self.llm = None
        self.kg_db = None
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
    
    def predict_next_relations(self, 
                             current_entity: str, 
                             target_entity: str, 
                             query_context: str,
                             max_relations: int = 5) -> List[Dict[str, Any]]:
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
            
            # Create LLM prompt for relation prediction
            prompt = self._create_relation_prediction_prompt(
                current_entity, target_entity, query_context, available_relations
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
                                         query_context: str, available_relations: List[str]) -> str:
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
        return predictions[:max_relations]
    
    def calculate_path_confidence(self, path: List[Tuple[str, str, str]]) -> float:
        """Calculate confidence score for a complete path"""
        if not path:
            return 0.0
        
        # Simple confidence calculation based on path length and relation types
        base_confidence = 1.0
        length_penalty = 0.1 * len(path)  # Longer paths are less confident
        
        return max(0.1, base_confidence - length_penalty)
