from pydantic import BaseModel
from typing import List, Optional, Any, Dict

# --- Request Models ---

class KGQuery(BaseModel):
    subject: Optional[str] = None
    predicate: Optional[str] = None
    object: Optional[str] = None
    limit: int = 10

class DenseQuery(BaseModel):
    query: str
    k: int = 5

class HybridQuery(BaseModel):
    query: str
    k: int = 5
    kg_weight: float = 0.5
    dense_weight: float = 0.5

# --- Response Models ---

class KGResult(BaseModel):
    subject: str
    predicate: str
    object: str

class DenseResult(BaseModel):
    score: float
    document: Dict[str, Any]

class HybridResult(BaseModel):
    final_score: float
    kg_results: List[KGResult]
    dense_results: List[DenseResult]

class SearchResponse(BaseModel):
    kg_results: Optional[List[KGResult]] = None
    dense_results: Optional[List[DenseResult]] = None
    hybrid_results: Optional[List[HybridResult]] = None

class AgentQuery(BaseModel):
    query: str

class AgentSearchResponse(BaseModel):
    original_query: str
    plan: List[str]
    retrieved_docs: Dict[str, Any]
    context: str
    answer: str
    review: Dict[str, Any]
