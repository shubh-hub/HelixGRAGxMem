"""
Multi-Agent System (MAS) Package
===============================

This package implements the SOTA biomedical reasoning MAS with:
- Typed state management
- MCP-based service architecture  
- LangGraph orchestration
- Full observability and tracing
"""

from .state import (
    MASState, 
    KGPath, 
    DenseHit, 
    Candidate,
    Intent,
    Mode,
    Action,
    create_initial_state,
    copy_state,
    get_policy_for_intent,
    FUSION_WEIGHTS
)

__version__ = "1.0.0"
__all__ = [
    "MASState",
    "KGPath", 
    "DenseHit",
    "Candidate",
    "Intent",
    "Mode", 
    "Action",
    "create_initial_state",
    "copy_state",
    "get_policy_for_intent",
    "FUSION_WEIGHTS"
]
