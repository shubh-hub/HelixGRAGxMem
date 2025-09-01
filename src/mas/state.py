"""
MAS State Management - Single Source of Truth
============================================

This module defines the complete typed state for the Multi-Agent System.
All agents read/write this state, and every mutation is logged with trace_id.
"""

from typing import List, Dict, Any, Optional, TypedDict, Literal
from datetime import datetime
import uuid

# Core type definitions
Intent = Literal["factoid", "enumeration", "causal", "comparative"]
Mode = Literal["kg_only", "dense_only", "hybrid"]
Action = Literal["expand", "replan", "clarify", "accept"]

class KGPath(TypedDict):
    """Knowledge Graph reasoning path with confidence and metadata"""
    triples: List[tuple]          # [(s, p, o), ...]
    path_conf: float              # product/sum of hop confidences
    meta: Dict[str, Any]          # hop stats, timings, entity info

class DenseHit(TypedDict):
    """Dense retrieval result with FAISS similarity"""
    id: int
    text: str
    score: float                  # FAISS similarity score
    metadata: Dict[str, Any]      # includes (subject, predicate, object)

class Candidate(TypedDict):
    """Fused candidate answer with evidence references"""
    answer: str                   # canonical object or formatted text
    score: float                  # fusion score (α*dense + β*kg + γ*corr + δ*rarity)
    evidence_refs: Dict[str, List[int] | List[tuple]]  # ids for dense, triples for KG

class SpanData(TypedDict):
    """Observability span for tracing node execution"""
    node: str
    started_at: datetime
    ended_at: Optional[datetime]
    duration_ms: Optional[float]
    inputs_hash: str
    outputs_digest: str
    tokens_in: Optional[int]
    tokens_out: Optional[int]
    errors: List[str]

class MASState(TypedDict):
    """
    Complete MAS state - single source of truth for all agents.
    Every mutation is logged with trace_id for full auditability.
    """
    # Core identifiers
    trace_id: str
    persona: str                  # "doctor", "patient", "researcher", "general"
    query: str
    timestamp: datetime

    # Planner outputs
    intent: Intent
    subqs: List[str]              # ≤3 sub-questions
    entities_surface: List[str]   # raw extracted entities
    entities_canonical: List[str] # KG node names/ids after canonicalization
    ambiguity_flags: List[str]    # canonicalization issues/warnings
    policy: Dict[str, Any]        # max_hops, topK_rel, max_neighbors, time_budget
    mode: Mode                    # router decision

    # Retrieval evidence
    evidence: Dict[str, Any]      # {"kg_paths": [KGPath], "dense_hits": [DenseHit]}
    candidates: List[Candidate]   # fused and scored candidates

    # Verification/Quality
    verdict: Dict[str, Any]       # {"ok": bool, "issues": [...], "action": Action}
    final_answer: Optional[str]
    explanation: Optional[str]
    confidence: float             # overall confidence score (0.0-1.0)
    
    # Observability & debugging
    spans: Dict[str, SpanData]    # node_name -> span_data
    errors: List[Dict[str, Any]]  # accumulated errors with context
    retry_count: int              # for expand/replan loops
    
    # Memory integration
    memory_context: Dict[str, Any]  # retrieved relevant memories
    
    # General metadata for workflow steps
    metadata: Dict[str, Any]  # flexible metadata for NLI workflow steps
    learned_patterns: List[str]     # patterns to store after completion

def create_initial_state(query: str, persona: str = "doctor") -> MASState:
    """Create initial MAS state with defaults"""
    return MASState(
        trace_id=str(uuid.uuid4()),
        persona=persona,
        query=query,
        timestamp=datetime.utcnow(),
        
        # Defaults - will be populated by nodes
        intent="factoid",
        subqs=[],
        entities_surface=[],
        entities_canonical=[],
        ambiguity_flags=[],
        policy={},
        mode="hybrid",
        evidence={"kg_paths": [], "dense_hits": []},
        candidates=[],
        verdict={},
        final_answer=None,
        explanation=None,
        confidence=0.0,
        spans={},
        errors=[],
        retry_count=0,
        memory_context={},
        metadata={},
        learned_patterns=[]
    )

def copy_state(state: MASState) -> MASState:
    """Deep copy state for immutable updates"""
    import copy
    return copy.deepcopy(state)

def log_state_mutation(state: MASState, node: str, mutation: str, details: Dict[str, Any]):
    """Log state mutations for debugging and audit"""
    mutation_log = {
        "trace_id": state["trace_id"],
        "node": node,
        "mutation": mutation,
        "details": details,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    # Add to state errors for now (will integrate with trace server)
    if "mutations" not in state:
        state["mutations"] = []
    state["mutations"].append(mutation_log)

# Policy templates by intent
INTENT_POLICIES = {
    "factoid": {
        "max_hops": 2,
        "topK_rel": 6,
        "max_neighbors": 3,
        "time_budget": 2.0,
        "coverage_threshold": 1  # Single answer expected
    },
    "enumeration": {
        "max_hops": 2,
        "topK_rel": 14,
        "max_neighbors": 8,
        "time_budget": 3.0,
        "coverage_threshold": 3  # Multiple answers expected
    },
    "causal": {
        "max_hops": 4,
        "topK_rel": 10,
        "max_neighbors": 4,
        "time_budget": 4.0,
        "coverage_threshold": 2  # Cause-effect pairs
    },
    "comparative": {
        "max_hops": 4,
        "topK_rel": 10,
        "max_neighbors": 4,
        "time_budget": 4.0,
        "coverage_threshold": 2  # Compare entities
    }
}

def get_policy_for_intent(intent: Intent) -> Dict[str, Any]:
    """Get policy configuration for given intent"""
    return INTENT_POLICIES.get(intent, INTENT_POLICIES["factoid"]).copy()

# Fusion scoring weights (configurable)
FUSION_WEIGHTS = {
    "alpha": 0.45,  # dense similarity weight
    "beta": 0.35,   # KG path confidence weight  
    "gamma": 0.15,  # corroboration bonus
    "delta": 0.05   # rarity bonus
}
