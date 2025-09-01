# HelixRAGxMem: Research Methodology

**Document Version**: 1.0  
**Last Updated**: January 5, 2025  

---

## 1. Research Methodology Overview

### 1.1 Research Paradigm
**Approach**: Mixed-methods research combining quantitative evaluation with qualitative analysis
- **Quantitative**: Benchmark performance metrics and statistical analysis
- **Qualitative**: System behavior analysis and error case studies
- **Experimental**: Controlled experiments with ablation studies

### 1.2 Research Design
**Type**: Comparative experimental design with multiple baselines
- **Control Groups**: Dense-only, KG-only, naive hybrid baselines
- **Treatment Group**: HelixRAGxMem with multi-agent memory system
- **Variables**: Retrieval method, memory configuration, agent orchestration

## 2. Core Methodological Approaches

### 2.1 Entropy-Based Graph Traversal

**Theoretical Foundation**:
The entropy-based traversal algorithm prioritizes exploration based on uncertainty, following information-theoretic principles where higher entropy indicates greater information potential.

### 5.1.1 Dense-to-Graph seeding (“anchor entities”)
Run dense search first, take the top-k sentences, extract entities, then start the graph walk from those anchors. In papers this often adds +3–4 pp recall over 0-hop alone.

### 5.1.2 Edge Predictor
*Prompt* includes question, current node label, 300‑token edge vocabulary → returns JSON list with `prob`.  
Temperature **T** calibrated on 200 validation Q/A using binary 
cross‑entropy.
Temperature-calibrated 0-Hop Edge Predictor

Prompt GPT-3.5 with question + current node + 300-token edge vocab.

Collect 200 validation pairs → fit T by minimising binary-cross-entropy.

Use calibrated edge_conf = σ(logit(p)/T) in entropy queue.

How temperature-calibration of the LLM’s edge probabilities works

**Goal** Make the LLM’s self-reported “prob” numbers mean “real likelihood” so the entropy queue is stable across queries.

### 1-A Collect a small validation sample

*Pick 200 questions* from KG-LLM-Bench.
For each:

1. We already know the **gold reasoning path** (e.g. `Pneumonia ─treated_by→ Azithromycin`).
2. At each node on that path we ask the 0-hop predictor to output its top-k relation list:

```json
[
 {"rel":"treated_by",      "prob":0.34},
 {"rel":"risk_factor_for", "prob":0.28},
 ...
]
```

3. Record **`y = 1` if the gold edge is ranked #1**, else `y = 0`, and store the predicted probability `p = 0.34`.

Now we have \~200 `(p, y)` pairs.

### 1-B Fit a one-parameter temperature **T**

We solve:

```
min_T  Σ  BCE( y , sigmoid( logit(p)/T ) )
```

(where `BCE` = binary-cross-entropy).
`T < 1` sharpens probs, `T > 1` flattens them.

Implementation (10 lines):

```python
import torch, torch.nn.functional as F
T = torch.nn.Parameter(torch.ones(1))
opt = torch.optim.LBFGS([T], lr=0.1)

def step():
    opt.zero_grad()
    preds = torch.sigmoid(torch.logit(p_tensor) / T)
    loss  = F.binary_cross_entropy(preds, y_tensor)
    loss.backward(); return loss
opt.step(step)
```

The calibrated confidence becomes

```python
edge_conf = torch.sigmoid(torch.logit(raw_prob)/T_opt).item()
```

Result: if the LLM said 0.34 but, empirically, that was correct 50 % of the time, the scaling fixes it; the entropy queue then truly prioritises “uncertain but promising” edges.

---

### 5.2 Entropy Queue
`priority = (1‑conf) / (depth+1)`; budget caps: `depth≤3`, `nodes≤300`, `wall_time≤800 ms`.

### 5.2.1 Dense Back-off & Seed Injection
If frontier starves, FAISS retrieves top-n sentences, maps mentions to entities (similarity ≥ 0.7), injects ≤ 5 new seeds.

| Potential issue                                                             | Why it hurts                   | Safeguard                                                                 |
| --------------------------------------------------------------------------- | ------------------------------ | ------------------------------------------------------------------------- |
| Dense hit maps to **wrong entity** (bad NER)                                | Graph walk chases wrong branch | Require sentence ↔ entity string similarity ≥ 0.7                         |
| Large latency if FAISS hit list huge                                        | slows < 2 s budget             | Cap to top-10 dense sentences                                             |
| Domain mismatch (dense text says “apple fruit”, graph node is “Apple Inc.”) | Answer nonsense                | Keep a domain tag check: if sentence domain tag ≠ KG domain tag → discard |

---

**Algorithm Design**:
```python
def entropy_based_traversal(seeds, max_hops=3, max_nodes=300):
    """
    Entropy-priority graph traversal with LLM edge prediction
    
    Args:
        seeds: Starting entities
        max_hops: Maximum traversal depth
        max_nodes: Budget constraint for exploration
    
    Returns:
        List of reasoning paths with evidence
    """
    frontier = []
    for seed in seeds:
        initial_entropy = calculate_initial_entropy(seed)
        heapq.heappush(frontier, (initial_entropy, [seed]))
    
    visited = set()
    paths = []
    
    while frontier and len(visited) < max_nodes:
        entropy_score, path = heapq.heappop(frontier)
        current_node = path[-1]
        
        if current_node in visited or len(path) > max_hops:
            continue
            
        visited.add(current_node)
        
        # LLM-based edge prediction
        predicted_edges = await edge_predictor.predict(
            current_node, context=path
        )
        
        for edge_rel, confidence in predicted_edges:
            next_nodes = kg.get_neighbors(current_node, edge_rel)
            for next_node in next_nodes:
                new_path = path + [(edge_rel, next_node)]
                # Entropy calculation: (1 - edge_confidence) / (path_length)
                entropy = (1 - confidence) / len(new_path)
                heapq.heappush(frontier, (entropy, new_path))
        
        paths.append(path)
    
    return paths
```

**Entropy Calculation**:
- **Formula**: `entropy_score = (1 - edge_confidence) / (path_length + 1)`
- **Rationale**: Lower confidence edges have higher entropy (more exploration value)
- **Path Length Penalty**: Longer paths receive lower priority to prevent infinite exploration

**Edge Prediction Methodology**:
```python
async def llm_edge_prediction(entity, context, relationship_vocab):
    """
    GPT-3.5 based edge prediction for biomedical entities
    """
    prompt = f"""
    Given biomedical entity: {entity}
    Query context: {context}
    Available relationships: {relationship_vocab}
    
    Predict the top 5 most relevant relationships for this entity
    in the context of the given query. Return as JSON with confidence scores.
    
    Format: {{"relationships": [{{"rel": "treats", "confidence": 0.9}}]}}
    """
    
    response = await openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=200
    )
    
    return parse_edge_predictions(response.choices[0].message.content)
```

### 2.2 Multi-Agent Orchestration

**Agent Architecture Design**:
Based on cognitive science principles of distributed problem-solving and collaborative reasoning.

**Agent Roles and Responsibilities**:

#### 2.2.1 Planner Agent
**Purpose**: Query analysis and retrieval strategy planning
**Methodology**: 
- Reason over the query, understand intent of the user, break down of complex queries.
- Entity extraction using biomedical NER
- Query complexity classification
- Budget allocation for retrieval components

```python
class PlannerAgent:
    async def plan_retrieval(self, query):
        # Extract biomedical entities
        entities = await self.ner_extractor.extract(query)
        
        # Classify query complexity
        complexity = await self.complexity_classifier.classify(query)
        
        # Allocate computational budget
        budget = self.allocate_budget(complexity)
        
        return RetrievalPlan(
            seeds=entities,
            complexity=complexity,
            budget=budget,
            strategy=self.select_strategy(complexity)
        )
```

#### 2.2.2 Retriever Agent
**Purpose**: Execute hybrid retrieval strategy
**Methodology**:
- Parallel execution of KG and dense retrieval
- Result fusion with fixed weights (Phase 1)
- Evidence collection and ranking

```python
class RetrieverAgent:
    async def execute_retrieval(self, plan):
        # Parallel retrieval
        kg_task = self.kg_retriever.retrieve(plan.seeds, plan.budget.kg)
        dense_task = self.dense_retriever.retrieve(plan.query, plan.budget.dense)
        
        kg_results, dense_results = await asyncio.gather(kg_task, dense_task)
        
        # Fusion with fixed weights
        fused_results = self.fusion_engine.fuse(
            kg_results, dense_results, weights={"kg": 0.6, "dense": 0.4}
        )
        
        return fused_results
```

#### 2.2.3 Reviewer Agent
**Purpose**: Quality assessment and memory gating
**Methodology**:
- Evidence quality scoring
- Reasoning chain validation
- Memory insertion decisions

```python
class ReviewerAgent:
    async def review_evidence(self, evidence, reasoning_chain):
        # Quality scoring based on multiple factors
        quality_score = await self.calculate_quality_score(
            evidence, reasoning_chain
        )
        
        # Decision thresholds
        if quality_score > self.memory_threshold:
            await self.memory_system.store(evidence, reasoning_chain)
        
        if quality_score > self.confidence_threshold:
            return ReviewDecision(accept=True, confidence=quality_score)
        else:
            return ReviewDecision(accept=False, confidence=quality_score)
```

#### 2.2.4 Explainer Agent
**Purpose**: Generate human-readable explanations
**Methodology**:
- Reasoning path visualization
- Evidence summarization
- Natural language generation

### 2.3 Memory System Design

**Cognitive Memory Model**:
Based on MIRIX architecture adapted for biomedical reasoning.

#### 2.3.1 Episodic Memory
**Purpose**: Short-term interaction context
**Methodology**:
- Sliding window approach (50 interactions)
- Temporal decay function
- Session-based organization

```python
class EpisodicMemory:
    def __init__(self, window_size=50):
        self.window_size = window_size
        self.interactions = deque(maxlen=window_size)
    
    async def store_interaction(self, query, reasoning_chain, evidence):
        interaction = {
            "timestamp": datetime.now(),
            "query": query,
            "reasoning_chain": reasoning_chain,
            "evidence": evidence,
            "session_id": self.current_session_id
        }
        self.interactions.append(interaction)
    
    async def retrieve_relevant(self, query, k=5):
        # Semantic similarity + temporal relevance
        candidates = []
        for interaction in self.interactions:
            similarity = self.calculate_similarity(query, interaction["query"])
            temporal_weight = self.calculate_temporal_weight(interaction["timestamp"])
            score = similarity * temporal_weight
            candidates.append((score, interaction))
        
        return sorted(candidates, reverse=True)[:k]
```

#### 2.3.2 Vault Memory
**Purpose**: Long-term knowledge storage
**Methodology**:
- Quality-gated insertion
- Deduplication and conflict resolution
- Usage-based relevance scoring

```python
class VaultMemory:
    async def store_knowledge(self, triples, quality_score):
        if quality_score < self.quality_threshold:
            return False
        
        for triple in triples:
            existing = await self.find_existing(triple)
            if existing:
                # Update with higher quality version
                if quality_score > existing.quality_score:
                    await self.update_triple(triple, quality_score)
            else:
                await self.insert_triple(triple, quality_score)
        
        return True
```

### 2.4 Hybrid Retrieval Fusion

**Fusion Methodology**:
Fixed-weight linear combination with result normalization.

**Mathematical Formulation**:
```
final_score(item) = w_kg * normalize(kg_score(item)) + w_dense * normalize(dense_score(item))

where:
- w_kg = 0.6 (knowledge graph weight)
- w_dense = 0.4 (dense retrieval weight)
- normalize(x) = (x - min) / (max - min)
```

**Implementation**:
```python
class HybridFusion:
    def __init__(self, kg_weight=0.6, dense_weight=0.4):
        self.kg_weight = kg_weight
        self.dense_weight = dense_weight
    
    def fuse_results(self, kg_results, dense_results):
        # Normalize scores to [0, 1] range
        kg_normalized = self.normalize_scores(kg_results)
        dense_normalized = self.normalize_scores(dense_results)
        
        # Combine all unique items
        all_items = set(kg_results.keys()) | set(dense_results.keys())
        
        fused_results = {}
        for item in all_items:
            kg_score = kg_normalized.get(item, 0)
            dense_score = dense_normalized.get(item, 0)
            
            final_score = (
                self.kg_weight * kg_score + 
                self.dense_weight * dense_score
            )
            fused_results[item] = final_score
        
        return sorted(fused_results.items(), key=lambda x: x[1], reverse=True)
```

## 3. Experimental Design

### 3.1 Controlled Experiments

**Experimental Variables**:
- **Independent Variables**: Retrieval method, memory configuration, agent setup
- **Dependent Variables**: Recall@5, Path-F1, Memory Recall@1, Latency
- **Control Variables**: Dataset, evaluation metrics, hardware configuration

**Experimental Conditions**:
1. **Baseline Conditions**:
   - Dense-only retrieval
   - KG-only retrieval
   - Naive hybrid (equal weights)
   - No memory system

2. **Treatment Conditions**:
   - HelixRAGxMem full system
   - Ablated versions (no memory, no multi-agent, etc.)

### 3.2 Ablation Study Design

**Component Ablation**:
```python
ablation_configs = {
    "full_system": {
        "hybrid_retrieval": True,
        "multi_agent": True,
        "memory_system": True,
        "entropy_traversal": True
    },
    "no_memory": {
        "hybrid_retrieval": True,
        "multi_agent": True,
        "memory_system": False,
        "entropy_traversal": True
    },
    "no_multi_agent": {
        "hybrid_retrieval": True,
        "multi_agent": False,
        "memory_system": True,
        "entropy_traversal": True
    },
    "no_entropy": {
        "hybrid_retrieval": True,
        "multi_agent": True,
        "memory_system": True,
        "entropy_traversal": False
    }
}
```

### 3.3 Statistical Analysis Plan

**Statistical Methods**:
- **Significance Testing**: Paired t-tests for performance comparisons
- **Effect Size**: Cohen's d for practical significance
- **Confidence Intervals**: 95% CI for all reported metrics
- **Multiple Comparisons**: Bonferroni correction for multiple tests

**Sample Size Calculation**:
```python
def calculate_sample_size(effect_size=0.5, alpha=0.05, power=0.8):
    """
    Calculate required sample size for detecting meaningful differences
    """
    from scipy import stats
    
    # Two-tailed test
    z_alpha = stats.norm.ppf(1 - alpha/2)
    z_beta = stats.norm.ppf(power)
    
    n = 2 * ((z_alpha + z_beta) / effect_size) ** 2
    return int(np.ceil(n))

# For detecting 5% improvement in Recall@5
required_queries = calculate_sample_size(effect_size=0.5)
```

## 4. Evaluation Methodology

### 4.1 Benchmark Evaluation Protocol

**VAT-KG Evaluation**:
```python
async def evaluate_vatkg(system, test_queries):
    results = {"recall_at_5": [], "mrr": [], "precision_at_1": []}
    
    for query in test_queries:
        # System retrieval
        retrieved = await system.retrieve(query, k=5)
        
        # Ground truth
        ground_truth = get_vatkg_ground_truth(query)
        
        # Calculate metrics
        results["recall_at_5"].append(
            calculate_recall(retrieved, ground_truth, k=5)
        )
        results["mrr"].append(
            calculate_mrr(retrieved, ground_truth)
        )
        results["precision_at_1"].append(
            calculate_precision(retrieved, ground_truth, k=1)
        )
    
    # Aggregate results
    return {
        metric: {
            "mean": np.mean(values),
            "std": np.std(values),
            "ci_95": stats.t.interval(0.95, len(values)-1, 
                                    loc=np.mean(values), 
                                    scale=stats.sem(values))
        }
        for metric, values in results.items()
    }
```

### 4.2 Memory System Evaluation

**Memory Effectiveness Protocol**:
1. **Baseline Phase**: System operation without memory
2. **Learning Phase**: 1000 interactions with memory enabled
3. **Testing Phase**: Evaluate memory-enhanced performance

```python
async def evaluate_memory_system(system, interaction_stream):
    # Phase 1: Baseline performance
    baseline_performance = await evaluate_without_memory(system)
    
    # Phase 2: Learning phase
    for interaction in interaction_stream:
        await system.process_with_memory(interaction)
    
    # Phase 3: Memory-enhanced performance
    memory_performance = await evaluate_with_memory(system)
    
    # Calculate memory benefit
    memory_benefit = {
        metric: memory_performance[metric] - baseline_performance[metric]
        for metric in baseline_performance.keys()
    }
    
    return memory_benefit
```

### 4.3 Qualitative Analysis

**Error Analysis Methodology**:
- **Failure Case Collection**: Systematic collection of system failures
- **Error Categorization**: Classification of error types and causes
- **Root Cause Analysis**: Deep dive into system behavior for critical errors

**Human Evaluation Protocol**:
- **Expert Review**: Domain expert evaluation of reasoning quality
- **Explanation Quality**: Human assessment of system explanations
- **Usability Study**: User experience evaluation

## 5. Reproducibility Framework

### 5.1 Experimental Reproducibility

**Version Control**:
- All code versioned with Git
- Exact dependency versions specified
- Docker containers for environment reproducibility

**Data Versioning**:
- Dataset checksums and versions
- Preprocessing pipeline documentation
- Intermediate result storage

**Random Seed Management**:
```python
def set_reproducible_seeds(seed=42):
    """Set seeds for all random number generators"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
```

### 5.2 Result Validation

**Cross-Validation Strategy**:
- 5-fold cross-validation for all experiments
- Stratified sampling to ensure balanced evaluation
- Independent test set for final evaluation

**Replication Protocol**:
- Detailed experimental protocols
- Automated experiment execution scripts
- Result verification procedures

## 6. Ethical Considerations

### 6.1 Bias Mitigation

**Dataset Bias Analysis**:
- Systematic analysis of dataset representativeness
- Evaluation across different medical conditions
- Demographic bias assessment where applicable

**Algorithm Fairness**:
- Performance evaluation across different query types
- Bias detection in retrieval results
- Fairness metrics integration

### 6.2 Privacy and Security

**Data Privacy**:
- No personally identifiable information storage
- Anonymization of any user interaction data
- Compliance with biomedical data usage guidelines

**Security Measures**:
- Secure API key management
- Input validation and sanitization
- Audit logging for all operations

---

**Next Steps**:
1. Implement experimental framework
2. Conduct pilot studies
3. Execute full experimental protocol
4. Perform statistical analysis and validation

## Hybrid Retrieval Fusion

### Phase 1: Controlled Complementarity Design

**Core Innovation**: Using the same underlying data for both KG and Dense retrieval to isolate complementarity effects from data source differences.

#### Data Preparation Strategy
```python
# Single source, dual format approach
def prepare_dual_format_data():
    """
    Convert VAT-KG triples to both structured and dense formats
    """
    # Load base knowledge graph
    triples = load_vatkg_triples()  # (subject, relation, object)
    
    # Format 1: Structured KG (DuckDB + NetworkX)
    kg_store = DuckDBKnowledgeGraph()
    kg_store.bulk_insert(triples)
    graph = networkx.from_triples(triples)
    
    # Format 2: Verbalized Dense Corpus
    verbalized_passages = []
    for subject, relation, object in triples:
        # Template-based verbalization
        passage = verbalize_triple(subject, relation, object)
        verbalized_passages.append({
            'text': passage,
            'source_triple': (subject, relation, object),
            'entities': [subject, object],
            'relation': relation
        })
    
    # Create dense embeddings
    embeddings = bge_model.encode([p['text'] for p in verbalized_passages])
    faiss_index = build_faiss_index(embeddings)
    
    return kg_store, graph, verbalized_passages, faiss_index
```

#### Verbalization Templates
```python
verbalization_templates = {
    'treats': "{subject} is used to treat {object} and provides therapeutic benefit for patients with this condition.",
    'causes': "{subject} is a known causative factor for {object} and significantly increases disease risk in affected individuals.",
    'associated_with': "{subject} shows strong clinical association with {object} based on extensive medical research and patient studies.",
    'prevents': "{subject} serves as a preventive measure against {object} through established protective mechanisms.",
    'interacts_with': "{subject} has documented pharmacological interactions with {object} that require clinical monitoring.",
    'contraindicated_with': "{subject} is medically contraindicated in patients with {object} due to safety concerns.",
    'metabolized_by': "{subject} undergoes metabolic processing by {object} enzyme systems in the human body."
}
```

### Fusion Algorithm Design

#### Static Fusion (Phase 1)
```python
def static_fusion(kg_results, dense_results, query_context):
    """
    Fixed-weight fusion optimized for controlled experiment
    """
    # Empirically determined weights for biomedical queries
    kg_weight = 0.6    # Higher precision from structured relationships
    dense_weight = 0.4  # Supporting semantic context
    
    fused_scores = []
    for kg_item in kg_results:
        # Find corresponding dense evidence
        matching_dense = find_supporting_passages(kg_item, dense_results)
        
        # Cross-validation scoring
        cross_validation_bonus = calculate_cross_validation_score(
            kg_item, matching_dense
        )
        
        fused_score = (
            kg_weight * kg_item.confidence + 
            dense_weight * matching_dense.similarity +
            0.1 * cross_validation_bonus  # Bonus for cross-validation
        )
        
        fused_scores.append({
            'item': kg_item,
            'supporting_evidence': matching_dense,
            'fused_score': fused_score,
            'kg_contribution': kg_weight * kg_item.confidence,
            'dense_contribution': dense_weight * matching_dense.similarity,
            'cross_validation': cross_validation_bonus
        })
    
    return sorted(fused_scores, key=lambda x: x['fused_score'], reverse=True)
```

#### Cross-Validation Mechanism
```python
def calculate_cross_validation_score(kg_item, dense_passages):
    """
    Measure agreement between KG and Dense evidence
    """
    kg_entities = extract_entities_from_path(kg_item.reasoning_path)
    
    cross_validation_score = 0.0
    for passage in dense_passages:
        # Entity overlap score
        passage_entities = extract_entities_from_text(passage.text)
        entity_overlap = len(kg_entities.intersection(passage_entities)) / len(kg_entities)
        
        # Semantic similarity of relations
        kg_relation = kg_item.reasoning_path[-1].relation
        inferred_relation = infer_relation_from_text(passage.text)
        relation_similarity = calculate_relation_similarity(kg_relation, inferred_relation)
        
        # Combined cross-validation score
        passage_score = 0.7 * entity_overlap + 0.3 * relation_similarity
        cross_validation_score = max(cross_validation_score, passage_score)
    
    return cross_validation_score
```

### Experimental Design for Complementarity Proof

#### Test Conditions
```python
experimental_conditions = {
    'kg_only': {
        'system': KGOnlySystem(),
        'data': vatkg_triples,
        'retrieval_method': 'entropy_based_graph_traversal',
        'hypothesis': 'Structured reasoning, limited coverage'
    },
    
    'dense_only': {
        'system': DenseOnlySystem(),
        'data': verbalized_vatkg_passages,
        'retrieval_method': 'semantic_similarity_search',
        'hypothesis': 'Broad coverage, limited reasoning'
    },
    
    'hybrid_fusion': {
        'system': HybridSystem(kg_weight=0.6, dense_weight=0.4),
        'data': 'both_formats',
        'retrieval_method': 'fused_kg_plus_dense',
        'hypothesis': 'Combined strengths, minimal weaknesses'
    },
    
    'multi_agent_orchestrated': {
        'system': HelixRAGxMemSystem(),
        'data': 'both_formats_plus_memory',
        'retrieval_method': 'agent_orchestrated_hybrid',
        'hypothesis': 'Emergent reasoning capabilities'
    }
}
```

#### Complementarity Metrics
```python
def measure_complementarity(kg_results, dense_results, ground_truth):
    """
    Quantify how KG and Dense retrieval complement each other
    """
    metrics = {}
    
    # Coverage Analysis
    kg_coverage = calculate_coverage(kg_results, ground_truth)
    dense_coverage = calculate_coverage(dense_results, ground_truth)
    union_coverage = calculate_coverage(kg_results + dense_results, ground_truth)
    
    metrics['coverage_complementarity'] = union_coverage - max(kg_coverage, dense_coverage)
    
    # Failure Mode Analysis
    kg_failures = identify_failures(kg_results, ground_truth)
    dense_failures = identify_failures(dense_results, ground_truth)
    
    # Measure orthogonality of failures
    failure_overlap = len(kg_failures.intersection(dense_failures))
    total_failures = len(kg_failures.union(dense_failures))
    metrics['failure_orthogonality'] = 1 - (failure_overlap / total_failures)
    
    # Information Uniqueness
    kg_unique_info = extract_unique_information(kg_results, dense_results)
    dense_unique_info = extract_unique_information(dense_results, kg_results)
    
    metrics['kg_unique_contribution'] = len(kg_unique_info) / len(kg_results)
    metrics['dense_unique_contribution'] = len(dense_unique_info) / len(dense_results)
    
    # Cross-Validation Success Rate
    cross_validated_items = count_cross_validated_results(kg_results, dense_results)
    metrics['cross_validation_rate'] = cross_validated_items / len(kg_results)
    
    return metrics
```

## Future Methodology Extensions (Phase 2+)

### Multi-Source Complementarity
**Planned Expansion**:
- **KG Sources**: UMLS, Hetionet, custom domain KGs
- **Dense Sources**: PubMedQA, PMC articles, clinical guidelines
- **Evaluation**: Cross-source validation and information-theoretic analysis

### Advanced Fusion Strategies
```python
# Adaptive fusion based on query complexity
def adaptive_fusion(query, kg_results, dense_results):
    """
    Dynamic weight adjustment based on query characteristics
    """
    query_complexity = classify_query_complexity(query)
    entity_coverage = calculate_entity_coverage(query, kg_results)
    
    if query_complexity == 'factual' and entity_coverage > 0.8:
        return fusion_weights(kg=0.8, dense=0.2)  # Favor KG for simple facts
    elif query_complexity == 'reasoning':
        return fusion_weights(kg=0.6, dense=0.4)  # Balanced for reasoning
    elif entity_coverage < 0.3:
        return fusion_weights(kg=0.3, dense=0.7)  # Favor dense for missing entities
    else:
        return fusion_weights(kg=0.6, dense=0.4)  # Default balanced
```

### Clinical Decision Support Extensions
```python
# Medical domain-specific fusion
def clinical_fusion(kg_results, dense_results, patient_context):
    """
    Medical domain fusion with safety considerations
    """
    # Safety-first weighting
    safety_critical = identify_safety_critical_queries(patient_context)
    
    if safety_critical:
        # Higher weight on structured KG for drug interactions, contraindications
        return fusion_weights(kg=0.8, dense=0.2, safety_bonus=0.1)
    else:
        # Standard fusion for general medical queries
        return fusion_weights(kg=0.6, dense=0.4)
```

This controlled methodology for Phase 1 will provide strong evidence for complementarity while maintaining experimental rigor and reproducibility. The single-source, dual-format approach eliminates confounding variables and isolates the complementarity effect to the retrieval mechanisms themselves.
