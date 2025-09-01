"""
Dense Retriever - BGE Embeddings + FAISS Search
==============================================

Implements dense vector retrieval using BGE-large-en-v1.5 embeddings
and FAISS similarity search for biomedical passages.
"""

import json
import logging
from typing import List, Dict, Any, Optional
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from ..config import settings

logger = logging.getLogger(__name__)

class DenseRetriever:
    """Dense retrieval using BGE embeddings and FAISS index"""
    
    def __init__(self):
        self.model = None
        self.index = None
        self.id_map = None
        self.passages = None
        
    def initialize(self):
        """Initialize BGE model and FAISS index"""
        try:
            # Load BGE embedding model
            logger.info("Loading BGE-large-en-v1.5 model...")
            self.model = SentenceTransformer(settings.EMBEDDING_MODEL)
            
            # Load FAISS index
            logger.info("Loading FAISS index...")
            self.index = faiss.read_index(settings.FAISS_INDEX_PATH)
            
            # Load ID mapping (list format)
            with open(settings.FAISS_ID_MAP_PATH, 'r') as f:
                id_list = json.load(f)
                # Convert list to dict for O(1) lookup
                self.id_map = {str(i): str(id_val) for i, id_val in enumerate(id_list)}
                
            # Load verbalized passages
            self.passages = {}
            with open(settings.VERBALIZED_KG_PATH, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    self.passages[str(data['id'])] = data
                    
            logger.info(f"✅ Dense retriever initialized with {len(self.passages)} passages")
            
        except Exception as e:
            logger.error(f"Failed to initialize dense retriever: {e}")
            raise
    
    def search(self, query: str, k: int = 10, trace_id: str = None) -> List[Dict[str, Any]]:
        """
        Search for relevant passages using dense similarity
        
        Args:
            query: Search query
            k: Number of results to return
            trace_id: Optional trace ID for logging
            
        Returns:
            List of search results with scores and metadata
        """
        if not self.model or not self.index:
            self.initialize()
            
        try:
            # Generate query embedding with instruction
            query_with_instruction = settings.QUERY_INSTRUCTION + query
            query_embedding = self.model.encode([query_with_instruction])
            
            # Search FAISS index
            scores, indices = self.index.search(query_embedding.astype('float32'), k)
            
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx == -1:  # Invalid index
                    continue
                    
                # Get passage ID from mapping
                passage_id = self.id_map.get(str(idx))
                if not passage_id or passage_id not in self.passages:
                    continue
                    
                passage_data = self.passages[passage_id]
                
                results.append({
                    "id": passage_id,
                    "text": passage_data.get("text", ""),
                    "score": float(score),
                    "rank": i + 1,
                    "metadata": {
                        "subject": passage_data.get("metadata", {}).get("subject", ""),
                        "predicate": passage_data.get("predicate", ""),
                        "object": passage_data.get("metadata", {}).get("object", ""),
                        "subject_type": passage_data.get("metadata", {}).get("subject_type", ""),
                        "object_type": passage_data.get("metadata", {}).get("object_type", "")
                    }
                })
                
            logger.info(f"Dense search returned {len(results)} results for query: {query[:50]}...")
            return results
            
        except Exception as e:
            logger.error(f"Dense search failed: {e}")
            return []
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text"""
        if not self.model:
            self.initialize()
        return self.model.encode([text])[0]
    
    def batch_search(self, queries: List[str], k: int = 10) -> List[List[Dict[str, Any]]]:
        """Batch search for multiple queries"""
        return [self.search(query, k) for query in queries]
    
    def load_index(self, index_path: str = None, id_map_path: str = None, passages_path: str = None):
        """Load FAISS index and related data - compatibility method for MCP server"""
        # Use provided paths or defaults from settings
        if index_path:
            self.index = faiss.read_index(index_path)
        if id_map_path:
            with open(id_map_path, 'r') as f:
                id_list = json.load(f)
                self.id_map = {str(i): str(id_val) for i, id_val in enumerate(id_list)}
        if passages_path:
            self.passages = {}
            with open(passages_path, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    self.passages[str(data['id'])] = data
        
        # If no paths provided, use initialize method
        if not any([index_path, id_map_path, passages_path]):
            self.initialize()
