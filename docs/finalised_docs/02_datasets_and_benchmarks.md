# HelixRAGxMem: Datasets and Benchmarks

**Document Version**: 1.1  
**Last Updated**: August 6, 2025

---

## 1. Phased Dataset Strategy

To rigorously evaluate the complementarity of Knowledge Graph (KG) and Dense retrieval, we will adopt a phased approach. Phase 1 establishes a controlled experiment using a single data source, while subsequent phases will introduce multiple sources to build a robust, cross-modal system.

### Phase 1: Single-Source Controlled Experiment

**Objective**: Isolate and measure the performance gains from combining KG and Dense retrieval mechanisms, eliminating data source variability.

| Dataset | Role | Source | Format | Description |
|---------|------|--------|--------|-------------|
| **VAT-KG** | Knowledge Graph | Hetionet | Triples `(s, p, o)` | A curated biomedical KG forming the foundation for both retrieval arms. |
| **Verbalized VAT-KG** | Dense Corpus | VAT-KG | Natural Language | Triples from VAT-KG are converted into sentences for dense retrieval. |
| **KG-LLM-Bench** | Evaluation | Benchmark | Q&A Pairs | Used for multi-hop reasoning and path-finding evaluation. |

### Phase 2: Multi-Source Hybrid System (Future Work)

**Objective**: Enhance the system with diverse, cross-modal data sources to improve coverage and real-world performance.

| Dataset | Purpose | Format |
|---------|---------|--------|
| **PubMedQA** | Dense Retrieval Corpus | Text Passages |
| **UMLS** | KG Enrichment | RDF/TTL Triples |

---

## 2. Phase 1 Dataset Details

### 2.1 VAT-KG (Validation and Testing Knowledge Graph)
**Purpose**: Primary data source for the controlled complementarity experiment.

**Characteristics**:
- **Source**: Derived from the public **Hetionet** dataset.
- **Size**: ~10,000 biomedical triples.
- **Domain**: Diseases, symptoms, treatments, drug interactions.
- **Format**: `(subject, predicate, object)` triples stored in DuckDB.

**Sample Triple**:
```json
{
  "subject": "Pneumonia",
  "predicate": "treated_by",
  "object": "Azithromycin"
}
```

### 2.2 Verbalized VAT-KG (For Dense Retrieval)
**Purpose**: Create a dense retrieval corpus from the exact same information as the knowledge graph.

**Generation Process**:
- Each triple from VAT-KG is converted into a natural language sentence using predefined templates.
- This ensures semantic equivalence between the KG and dense retrieval sources.

**Sample Verbalization**:
- **Triple**: `(Pneumonia, treated_by, Azithromycin)`
- **Template**: `"{subject} is a condition that is treated by {object}."`
- **Output**: `"Pneumonia is a condition that is treated by Azithromycin."`

**Processing Pipeline**:
1. Verbalize all triples from VAT-KG.
2. Generate embeddings for each sentence using **BGE-large-en**.
3. Build a **FAISS** index for efficient semantic search.

---

## 3. Evaluation Benchmarks

### 3.1 KG-LLM-Bench (Biomedical Subset)
**Purpose**: Evaluate the system's ability to perform multi-hop reasoning and answer complex questions.

**Characteristics**:
- **Size**: ~1,200 biomedical question/answer pairs relevant to the VAT-KG domain.
- **Task**: Given a question, the system must retrieve the correct reasoning path from the knowledge graph.

**Evaluation Metrics**:
- **Recall@5**: Measures if the correct entities are retrieved.
- **Path-F1**: Measures the accuracy of the retrieved reasoning path.

## 4. Knowledge Graph Data

### 4.1 VAT-KG (Validation and Testing Knowledge Graph)
**Purpose**: Primary evaluation dataset for biomedical knowledge graph retrieval

**Characteristics**:
- **Size**: ~10,000 biomedical triples
- **Domain**: Medical entities, relationships, and facts
- **Format**: `(subject, predicate, object)` triples in JSONL
- **Coverage**: Diseases, symptoms, treatments, drug interactions

**Sample Data**:
```json
{
  "subject": "Pneumonia",
  "predicate": "treated_by",
  "object": "Azithromycin",
  "source": "VAT-KG",
  "confidence": 0.95
}
```

**Preprocessing Steps**:
1. Entity normalization and standardization
2. Relationship type mapping to UMLS standards
3. Confidence score validation
4. Duplicate removal and conflict resolution

### 4.2 UMLS Knowledge Graph Subset
**Purpose**: Core biomedical knowledge graph for reasoning

**Characteristics**:
- **Size**: ~500,000 curated triples
- **Source**: UMLS Metathesaurus 2024AA
- **Coverage**: Comprehensive biomedical ontology
- **Quality**: Manually curated and validated

**Entity Types**:
- Diseases and Disorders
- Chemicals and Drugs
- Anatomy
- Physiological Processes
- Medical Procedures

**Relationship Types**:
- `treats` / `treated_by`
- `causes` / `caused_by`
- `symptom_of` / `has_symptom`
- `interacts_with`
- `part_of` / `has_part`

## 5. Dense Retrieval Data

### 5.1 PubMedQA Passages
**Purpose**: Dense retrieval corpus for semantic search

**Characteristics**:
- **Size**: 1.4GB of biomedical text passages
- **Source**: PubMed abstracts and full-text articles
- **Processing**: Chunked into 512-token passages
- **Embeddings**: BGE-large-en (1024-dimensional)

**Preprocessing Pipeline**:
```python
def preprocess_pubmed_passages():
    steps = [
        "extract_abstracts_and_fulltext",
        "chunk_into_passages(max_tokens=512)",
        "remove_duplicates",
        "quality_filter(min_length=50)",
        "generate_embeddings(model='BGE-large-en')",
        "build_faiss_index(index_type='IVFPQ')"
    ]
    return steps
```

### 5.2 Medical Entity Aliases
**Purpose**: Entity linking and normalization

**Characteristics**:
- **Size**: ~100k entity-alias pairs
- **Source**: UMLS synonym tables
- **Coverage**: Medical terminology variations
- **Format**: `{canonical_entity: [alias1, alias2, ...]}`

## 6. Evaluation Benchmarks

### 6.1 VAT-KG Retrieval Benchmark
**Task**: Knowledge graph-based information retrieval

**Metrics**:
- **Recall@5**: Percentage of relevant facts in top-5 results
- **MRR (Mean Reciprocal Rank)**: Average reciprocal rank of first relevant result
- **Precision@1**: Accuracy of top-ranked result

**Query Types**:
- **Factual**: "What treats pneumonia?"
- **Multi-hop**: "What are the side effects of drugs that treat diabetes?"
- **Exploratory**: "Find connections between obesity and heart disease"

**Evaluation Protocol**:
```python
def evaluate_vatkg_retrieval(system, test_queries):
    results = {}
    for query in test_queries:
        retrieved = system.retrieve(query, k=5)
        ground_truth = get_ground_truth(query)
        
        results[query] = {
            "recall_at_5": calculate_recall(retrieved, ground_truth, k=5),
            "mrr": calculate_mrr(retrieved, ground_truth),
            "precision_at_1": calculate_precision(retrieved, ground_truth, k=1)
        }
    
    return aggregate_results(results)
```

### 6.2 KG-LLM-Bench Multi-Hop Reasoning
**Task**: Complex multi-hop reasoning over knowledge graphs

**Characteristics**:
- **Size**: 126,000 question-answer pairs
- **Complexity**: 1-4 hop reasoning paths
- **Gold Standard**: Human-annotated reasoning paths

**Metrics**:
- **Path-F1**: F1 score of predicted vs. gold reasoning paths
- **Answer Accuracy**: Exact match accuracy of final answers
- **Path Coverage**: Percentage of gold path steps covered

**Sample Question**:
```json
{
  "question": "What are the contraindications for medications used to treat hypertension in diabetic patients?",
  "gold_path": [
    "Diabetes -> treated_by -> Metformin",
    "Hypertension -> treated_by -> ACE_inhibitors", 
    "ACE_inhibitors -> contraindicated_with -> Kidney_disease",
    "Diabetes -> causes -> Kidney_disease"
  ],
  "answer": "ACE inhibitors are contraindicated in diabetic patients with kidney disease"
}
```

### 6.3 Memory Recall Benchmark
**Task**: Evaluate memory system effectiveness

**Methodology**:
1. **Day 0**: Insert 1000 memory triples
2. **Day 1-7**: Continuous interaction simulation
3. **Day 7**: Test recall of original facts

**Metrics**:
- **Memory Recall@1**: Accuracy of retrieving stored memories
- **Memory Precision**: Relevance of retrieved memories
- **Temporal Decay**: Memory performance over time

## 7. Synthetic Data Generation

### 7.1 Complementarity Analysis Dataset
**Purpose**: Analyze when KG vs Dense retrieval performs better

**Generation Method**:
```python
def generate_complementarity_queries():
    query_types = {
        "factual": generate_factual_queries(n=250),
        "conceptual": generate_conceptual_queries(n=250), 
        "reasoning": generate_reasoning_queries(n=250),
        "synthesis": generate_synthesis_queries(n=250)
    }
    
    # Label with ground truth about which method should work better
    for query_type, queries in query_types.items():
        for query in queries:
            query["expected_better_method"] = get_expected_method(query_type)
    
    return query_types
```

### 7.2 Memory Stream Simulation
**Purpose**: Test memory system with realistic interaction patterns

**Characteristics**:
- **Duration**: 7-day simulation
- **Volume**: 1000 interactions per day
- **Patterns**: Realistic query distributions and temporal patterns

**Generation Process**:
1. Model realistic user interaction patterns
2. Generate queries with temporal dependencies
3. Simulate memory formation and retrieval needs
4. Create ground truth for memory evaluation

## 8. Data Quality and Validation

### 8.1 Quality Assurance Metrics
| Metric | Threshold | Purpose |
|--------|-----------|---------|
| **Entity Coverage** | >95% | Ensure comprehensive entity representation |
| **Relationship Consistency** | >90% | Validate relationship coherence |
| **Duplicate Rate** | <5% | Minimize redundant information |
| **Missing Value Rate** | <2% | Ensure data completeness |

### 8.2 Validation Protocols
- **Manual Review**: 10% random sample manual validation
- **Cross-Reference**: Validation against multiple biomedical sources
- **Expert Review**: Domain expert validation of complex relationships
- **Automated Checks**: Consistency and format validation scripts

## 9. Baseline Systems

### 9.1 Dense-Only Baseline
**Implementation**: BGE-large-en + FAISS retrieval
**Configuration**:
- Embedding model: `BAAI/bge-large-en`
- Index type: `IVFPQ` with 4096 clusters
- Retrieval: Top-k similarity search

### 9.2 KG-Only Baseline
**Implementation**: Traditional graph traversal
**Configuration**:
- Traversal: Breadth-first search
- Max hops: 3
- Scoring: Path length + edge confidence

### 9.3 Naive Hybrid Baseline
**Implementation**: Simple score combination
**Configuration**:
- Fusion: Equal weighting (0.5 KG + 0.5 Dense)
- No learning or adaptation
- Basic result merging

### 9.4 State-of-the-Art Baselines
- **GraphRAG**: Microsoft's graph-augmented retrieval
- **RAG-KG**: Knowledge graph enhanced RAG
- **DenseRAG**: Pure dense retrieval system

## 10. Evaluation Infrastructure

### 10.1 Automated Evaluation Pipeline
```python
class EvaluationPipeline:
    def __init__(self):
        self.benchmarks = {
            "vatkg": VATKGBenchmark(),
            "kg_llm_bench": KGLLMBenchmark(), 
            "memory_recall": MemoryRecallBenchmark()
        }
    
    def run_full_evaluation(self, system):
        results = {}
        for name, benchmark in self.benchmarks.items():
            results[name] = benchmark.evaluate(system)
        
        return self.generate_report(results)
```

### 10.2 Continuous Integration
- **Daily Runs**: Automated evaluation on development system
- **Regression Testing**: Performance monitoring across changes
- **Benchmark Tracking**: Historical performance tracking

## 11. Data Access and Licensing

### 11.1 Public Datasets
- **VAT-KG**: Creative Commons Attribution 4.0
- **KG-LLM-Bench**: MIT License
- **PubMedQA**: Academic use permitted

### 11.2 Restricted Datasets
- **UMLS**: Requires UMLS license agreement
- **Full PubMed**: Requires publisher agreements

### 11.3 Data Sharing Plan
- **Preprocessed Data**: Share cleaned and processed versions
- **Evaluation Scripts**: Open-source evaluation framework
- **Synthetic Data**: Freely available for research use

---

**Next Steps**:
1. Acquire dataset licenses and access
2. Implement preprocessing pipelines
3. Set up evaluation infrastructure
4. Validate data quality and benchmarks

## Phase 1: Controlled Complementarity Study

### Hypothesis
KG and Dense retrieval are complementary even when using the same underlying data

### Test Conditions
1. **KG-Only**: Structured graph traversal on VAT-KG triples
2. **Dense-Only**: Semantic search on verbalized VAT-KG passages  
3. **Hybrid**: Combined KG + Dense with fusion
4. **Multi-Agent**: Our full system with agent orchestration

### Evaluation Protocol
```python
def phase1_evaluation():
    """
    Controlled experiment proving complementarity
    """
    systems = {
        'kg_only': KGOnlySystem(vatkg_graph),
        'dense_only': DenseOnlySystem(verbalized_vatkg),
        'hybrid_fusion': HybridSystem(kg_weight=0.6, dense_weight=0.4),
        'multi_agent': HelixRAGxMemSystem()
    }
    
    results = {}
    for system_name, system in systems.items():
        results[system_name] = evaluate_system(system, test_queries)
    
    # Prove complementarity: hybrid > max(kg_only, dense_only)
    complementarity_gain = (
        results['hybrid_fusion']['recall_at_5'] - 
        max(results['kg_only']['recall_at_5'], 
            results['dense_only']['recall_at_5'])
    )
    
    assert complementarity_gain > 0.05  # 5% minimum improvement
    
    return results
```

## Future Expansion (Phase 2+)

### Multi-Source Complementarity
**Planned Data Sources**:
- **KG**: UMLS, Hetionet, custom biomedical KGs
- **Dense**: PubMedQA, PMC articles, clinical guidelines
- **Evaluation**: Expanded benchmarks, clinical scenarios

### Advanced Experiments
1. **Information-Theoretic Complementarity**: Measure unique information in each source
2. **Failure Mode Analysis**: Identify orthogonal failure patterns
3. **Clinical Decision Support**: Real-world medical scenario evaluation
4. **Expert Validation**: Medical professional assessment

### Expanded Benchmarks
- **LOCOMO**: Long-form memory evaluation
- **ScreenshotVQA**: Multimodal memory assessment  
- **Clinical Scenarios**: Custom medical decision support tasks
- **Multi-turn Conversations**: Context retention and personalization

## Success Criteria

### Phase 1 Targets
- **VAT-KG Recall@5**: 75%+ (proving complementarity with same data)
- **KG-LLM-Bench Path-F1**: 70%+ (multi-hop reasoning capability)
- **Complementarity Gain**: 5%+ improvement of hybrid over individual methods
- **Latency**: <1s average response time
- **Memory Efficiency**: <8GB RAM usage (MacBook Air compatible)

### Future Phase Targets  
- **Multi-source VAT-KG**: 80%+ Recall@5
- **Clinical Scenarios**: 85%+ expert approval rating
- **LOCOMO**: 85%+ memory recall accuracy
- **Real-world Deployment**: Production-ready system

## Data Access and Licensing

### VAT-KG
- **Access**: Custom dataset creation from Hetionet
- **License**: CC0 (public domain)
- **Download**: Automated script provided

### KG-LLM-Bench
- **Access**: GitHub repository (AKSW/LLM-KG-Bench)
- **License**: MIT License
- **Setup**: Clone repository, extract biomedical subset

### Preprocessing Scripts
All data preparation scripts are provided in `/scripts/data_preparation/`:
- `create_vatkg_dataset.py`: Generate VAT-KG from Hetionet
- `verbalize_kg_triples.py`: Convert triples to natural language
- `extract_kg_llm_bench.py`: Extract biomedical reasoning queries
- `validate_datasets.py`: Data quality checks and statistics

## Quality Assurance

### Data Validation
- **Completeness**: All required entities and relations present
- **Consistency**: No contradictory triples or passages
- **Coverage**: Balanced representation across biomedical domains
- **Quality**: Manual review of verbalization accuracy

### Reproducibility
- **Deterministic Processing**: Fixed random seeds for all operations
- **Version Control**: All datasets tagged and versioned
- **Documentation**: Complete preprocessing pipeline documentation
- **Validation Scripts**: Automated data quality checks
