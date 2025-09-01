# HelixRAGxMem: Technology Stack & Architecture Choices

**Document Version**: 1.0  
**Last Updated**: January 5, 2025  

---

## 1. Technology Stack Overview

### Core Components
| Component | Technology | Version | Reasoning |
|-----------|------------|---------|-----------|
| **Language** | Python | 3.11+ | Ecosystem maturity, ML libraries |
| **Knowledge Graph** | DuckDB + NetworkX | 0.9.2 + 3.2 | Performance + graph algorithms |
| **Dense Retrieval** | FAISS + BGE-large-en | 1.7.4 + HF | Scalability + SOTA embeddings |
| **LLM Integration** | OpenAI API | GPT-3.5-turbo | Cost-effective, reliable |
| **Agent Framework** | LangGraph | 0.2.0 | Multi-agent orchestration |
| **Memory Storage** | SQLite + DuckDB | 3.44 + 0.9.2 | Lightweight + analytical |
| **API Framework** | FastAPI | 0.104.0 | Async support, MCP integration |
| **Observability** | OpenTelemetry + Jaeger | 1.21.0 + 1.50 | Distributed tracing |

## 2. Core Technology Decisions

### 2.1 Programming Language: Python 3.11+

**Decision**: Python 3.11+ as primary language

**Reasoning**:
- **ML Ecosystem**: Comprehensive libraries (transformers, faiss, networkx)
- **Async Support**: Native async/await for multi-agent coordination
- **Performance**: 3.11+ offers significant speed improvements
- **Community**: Large research community and extensive documentation

**Alternatives Considered**:
- **Rust**: Higher performance but limited ML ecosystem
- **JavaScript/TypeScript**: Good for web integration but weaker ML support
- **Julia**: Excellent for scientific computing but smaller ecosystem

### 2.2 Knowledge Graph Storage: DuckDB

**Decision**: DuckDB for knowledge graph storage with NetworkX for algorithms

**Reasoning**:
- **Performance**: Columnar storage optimized for analytical queries
- **Simplicity**: Embedded database, no separate server required
- **SQL Interface**: Standard SQL for complex graph queries
- **Memory Efficiency**: Excellent compression and memory usage
- **Integration**: Native Python integration

**Implementation**:
```sql
-- Optimized schema for biomedical triples
CREATE TABLE med_triples (
    subj TEXT NOT NULL,
    rel TEXT NOT NULL,
    obj TEXT NOT NULL,
    src TEXT DEFAULT 'UMLS',
    conf REAL DEFAULT 1.0,
    PRIMARY KEY(subj, rel, obj)
);

-- Indexes for fast traversal
CREATE INDEX idx_subj_rel ON med_triples(subj, rel);
CREATE INDEX idx_obj_rel ON med_triples(obj, rel);
```

**Alternatives Considered**:
- **Neo4j**: Native graph DB but requires separate server
- **PostgreSQL**: Mature but less optimized for analytical workloads
- **ArangoDB**: Multi-model but complex setup

### 2.3 Dense Retrieval: FAISS + BGE-large-en

**Decision**: FAISS for vector indexing with BGE-large-en embeddings

**Reasoning**:
- **FAISS**: Facebook's optimized similarity search library
  - Excellent performance for large-scale retrieval
  - Multiple index types (IVFPQ for memory efficiency)
  - GPU support available
- **BGE-large-en**: BAAI's SOTA embedding model
  - Superior performance on retrieval benchmarks
  - 1024-dimensional embeddings
  - Optimized for English biomedical text

**Configuration**:
```python
# FAISS index configuration
index_config = {
    "index_type": "IVFPQ",
    "nlist": 4096,  # Number of clusters
    "m": 16,        # Number of subquantizers
    "nbits": 8      # Bits per subquantizer
}

# BGE embedding configuration
embedding_config = {
    "model_name": "BAAI/bge-large-en",
    "max_length": 512,
    "batch_size": 32,
    "normalize_embeddings": True
}
```

**Alternatives Considered**:
- **Pinecone**: Managed service but cost and latency concerns
- **Weaviate**: Good features but complex deployment
- **Sentence-BERT**: Good but BGE shows better performance

### 2.4 LLM Integration: OpenAI GPT-3.5-turbo

**Decision**: OpenAI GPT-3.5-turbo for edge prediction and reasoning

**Reasoning**:
- **Cost-Effectiveness**: Significantly cheaper than GPT-4
- **Performance**: Sufficient for edge prediction tasks
- **Reliability**: Stable API with good uptime
- **Speed**: Faster inference than larger models
- **Research Constraint**: Inference-only requirement met

**Usage Patterns**:
```python
# Edge prediction prompt template
edge_prediction_prompt = """
Given the biomedical entity "{entity}" and question context "{context}",
predict the most relevant relationship types from this vocabulary:
{relationship_vocab}

Return top 5 relationships with confidence scores as JSON.
"""

# API configuration
openai_config = {
    "model": "gpt-3.5-turbo",
    "temperature": 0.1,  # Low temperature for consistency
    "max_tokens": 200,
    "timeout": 30
}
```

**Alternatives Considered**:
- **GPT-4**: Better quality but 10x cost increase
- **Claude**: Good alternative but API limitations
- **Open-source LLMs**: Deployment complexity and quality concerns

### 2.5 Agent Framework: LangGraph

**Decision**: LangGraph for multi-agent orchestration

**Reasoning**:
- **Multi-Agent Support**: Native support for agent workflows
- **State Management**: Built-in state persistence and management
- **Observability**: Integration with tracing and monitoring
- **Flexibility**: Supports complex agent interaction patterns
- **Community**: Active development and good documentation

**Agent Architecture**:
```python
from langgraph import StateGraph, END

# Define agent workflow
workflow = StateGraph(AgentState)
workflow.add_node("planner", planner_agent)
workflow.add_node("retriever", retriever_agent)
workflow.add_node("reviewer", reviewer_agent)
workflow.add_node("explainer", explainer_agent)

# Define transitions
workflow.add_edge("planner", "retriever")
workflow.add_conditional_edges(
    "reviewer",
    should_continue,
    {"continue": "retriever", "finish": "explainer"}
)
```

**Alternatives Considered**:
- **AutoGen**: Microsoft's framework but less mature
- **CrewAI**: Good for simple workflows but limited flexibility
- **Custom Implementation**: Full control but significant development overhead

### 2.6 Memory System: SQLite + DuckDB

**Decision**: Hybrid approach with SQLite for episodic memory and DuckDB for vault storage

**Reasoning**:
- **SQLite for Episodic**: 
  - Lightweight for short-term memory
  - ACID transactions for consistency
  - Built-in Python support
- **DuckDB for Vault**:
  - Analytical queries for memory analysis
  - Better compression for long-term storage
  - SQL interface for complex queries

**Schema Design**:
```sql
-- Episodic memory (SQLite)
CREATE TABLE episodic_memory (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    query TEXT NOT NULL,
    reasoning_chain JSON NOT NULL,
    evidence JSON NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    ttl DATETIME  -- Time to live
);

-- Vault memory (DuckDB)
CREATE TABLE vault_memory (
    subj TEXT NOT NULL,
    rel TEXT NOT NULL,
    obj TEXT NOT NULL,
    quality_score REAL NOT NULL,
    first_seen DATETIME NOT NULL,
    last_used DATETIME NOT NULL,
    usage_count INTEGER DEFAULT 1,
    PRIMARY KEY(subj, rel, obj)
);
```

### 2.7 API Framework: FastAPI

**Decision**: FastAPI for MCP server implementation

**Reasoning**:
- **Async Support**: Native async/await for concurrent requests
- **Performance**: High performance with automatic optimization
- **Documentation**: Auto-generated OpenAPI documentation
- **Type Safety**: Pydantic integration for request/response validation
- **MCP Compatibility**: Easy integration with MCP protocol

**MCP Server Implementation**:
```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="HelixRAGxMem MCP Server")

class KGQueryRequest(BaseModel):
    subj: str
    rel: Optional[str] = None
    trace_id: str

@app.post("/kg/get_triples")
async def get_triples(request: KGQueryRequest):
    # Implementation with tracing
    with tracer.start_span("kg_query", context=request.trace_id):
        results = await kg_service.query(request.subj, request.rel)
        return {"triples": results}
```

## 3. Infrastructure Choices

### 3.1 Observability: OpenTelemetry + Jaeger

**Decision**: OpenTelemetry for instrumentation with Jaeger for visualization

**Reasoning**:
- **Standard**: Industry standard for distributed tracing
- **Vendor Neutral**: Not locked to specific monitoring vendor
- **Rich Context**: Detailed trace information for debugging
- **Multi-Agent Support**: Track interactions across agents

**Implementation**:
```python
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# Configure tracing
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)

jaeger_exporter = JaegerExporter(
    agent_host_name="localhost",
    agent_port=6831,
)

span_processor = BatchSpanProcessor(jaeger_exporter)
trace.get_tracer_provider().add_span_processor(span_processor)
```

### 3.2 Development Environment

**Decision**: Conda for environment management with specific versions

**Environment Setup**:
```yaml
# environment.yml
name: helixragxmem
channels:
  - conda-forge
  - pytorch
dependencies:
  - python=3.11
  - duckdb=0.9.2
  - networkx=3.2
  - faiss-cpu=1.7.4
  - sentence-transformers=2.2.2
  - fastapi=0.104.0
  - uvicorn=0.24.0
  - opentelemetry-api=1.21.0
  - opentelemetry-sdk=1.21.0
  - langgraph=0.2.0
  - pip
  - pip:
    - openai==1.3.0
    - jaeger-client==4.8.0
```

### 3.3 Deployment Architecture

**Decision**: Containerized deployment with Docker

**Reasoning**:
- **Reproducibility**: Consistent environment across development and production
- **Scalability**: Easy horizontal scaling of components
- **Isolation**: Component isolation for debugging and maintenance

**Docker Configuration**:
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose ports
EXPOSE 8000 9000

# Start command
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## 4. Performance Considerations

### 4.1 Scalability Design

**Memory Management**:
- **FAISS**: Memory-mapped indexes for large datasets
- **DuckDB**: Streaming query processing for large results
- **Caching**: Redis for frequently accessed data

**Concurrency**:
- **Async/Await**: Non-blocking I/O operations
- **Connection Pooling**: Database connection management
- **Rate Limiting**: API request throttling

### 4.2 Optimization Strategies

**Query Optimization**:
```python
# Optimized graph traversal
async def optimized_graph_traversal(seeds, max_hops=3):
    # Batch database queries
    batch_size = 100
    for i in range(0, len(seeds), batch_size):
        batch = seeds[i:i+batch_size]
        # Process batch in parallel
        tasks = [traverse_from_seed(seed) for seed in batch]
        results = await asyncio.gather(*tasks)
        yield from results
```

**Memory Optimization**:
- **Lazy Loading**: Load data only when needed
- **Memory Pools**: Reuse memory allocations
- **Garbage Collection**: Explicit cleanup of large objects

## 5. Security Considerations

### 5.1 API Security

**Authentication**: API key-based authentication for MCP endpoints
**Rate Limiting**: Request throttling to prevent abuse
**Input Validation**: Strict validation of all inputs
**HTTPS**: TLS encryption for all communications

### 5.2 Data Security

**Encryption**: Encrypt sensitive data at rest
**Access Control**: Role-based access to different components
**Audit Logging**: Comprehensive logging of all operations
**Privacy**: No storage of personally identifiable information

## 6. Testing Strategy

### 6.1 Unit Testing

**Framework**: pytest with async support
**Coverage**: >90% code coverage target
**Mocking**: Mock external services (OpenAI, databases)

### 6.2 Integration Testing

**End-to-End**: Full pipeline testing with real data
**Performance**: Load testing with realistic workloads
**Regression**: Automated testing of benchmark performance

### 6.3 Evaluation Testing

**Benchmark Suite**: Automated evaluation on all benchmarks
**Ablation Studies**: Systematic component testing
**Baseline Comparison**: Regular comparison with baseline systems

---

**Next Steps**:
1. Set up development environment
2. Implement core components
3. Integrate observability framework
4. Deploy and test infrastructure
