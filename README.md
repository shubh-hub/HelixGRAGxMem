
# 📄 Project Specification & System Design  
**Project Title:** Hybrid Graph + Dense Retrieval with MIRIX‑Slim Memory (Phase‑1)  
**Date:** 2025-08-02  

---

## 1 High‑Level Goal
Build, in 4 weeks, an inference‑only retrieval‑augmented generation (RAG) system that:
1. Beats dense‑only baselines on two public biomedical KG benchmarks.
2. Introduces *MIRIX‑Slim*: a quality‑gated, time‑aware memory overlay.
3. Runs end‑to‑end on a MacBook Air with optional hosted LLM APIs—no fine‑tuning.

---

## 2 Scope Summary
| Included | Excluded |
|----------|----------|
| Hybrid graph + dense retrieval, 0‑hop edge predictor, entropy queue, PET reranker | Any model fine‑tuning or large GNN training |
| MIRIX‑Slim memory (episodic + vault) & Intelligent Memory Manager | Full MIRIX semantic compression or provenance graph |
| Benchmarks on VAT‑KG & KG‑LLM‑Bench | Full guideline therapy planning |
| Optional Neo4j/APOC swap‑in (1‑day extra) | Production integration, EHR actions |

---

## 3 Datasets
| Name | Size | Purpose |
|------|------|---------|
| **VAT‑KG slice** | ~10 k triples | Retrieval + Recall@5 |
| **KG‑LLM‑Bench QA** | 126 k Q/A with gold paths | Path‑F1 multi‑hop eval |
| **Synthetic memory stream** | 7 k triples/7 days | Recall@1(7‑day) metric |
| *(opt)* PubMedQA passages | 1.4 GB | Dense back‑off text |

---

## 4 System Architecture (bird’s‑eye)

```
User ─▶ Planner ─▶ Retriever ──┬─▶ GraphWalker (DuckDB+NetworkX)
                               ├─▶ DenseSearch (FAISS)
                               ├─▶ IMM Router  ⇅  MIRIX‑Slim
                               └─▶ Reviewer (PET / GPT)
                                     │  (accept?)
                                     ▼
                               Explainer ─▶ UI / Demo
```

---

## 5 Module‑Level Design

| ID | Module | Key Classes / Scripts | Core Data Structures |
|----|--------|----------------------|----------------------|
| M1 | **Env & Data** | `dl_vatkg.py`, `etl_duck.py` | `public.duckdb` |
| M2 | **Embeddings & FAISS** | `build_faiss.py` | `faiss.index` |
| M3 | **0‑hop Edge Predictor** | `edge_pred.py` (GPT‑3.5) | cache `edge_cache.jsonl` |
| M4 | **Entropy GraphWalker** | `walker.py` | priority‑queue of `(score,path)` |
| M5 | **PET Re‑ranker** | `rerank.py` (HF API) | logits csv |
| M6 | **MIRIX‑Slim** | `memory.py` | `episodic.sqlite`, `vault.duckdb`, `mem_index` |
| M7 | **IMM Router** | `imm_router.py` (rule YAML) | index table rows |
| M8 | **Agents Orchestrator** | `agent_loop.py` (LangGraph‑style) | async tasks |
| M9 | **Eval Harness** | `benchmark.py` | MLflow runs |
| M10| **Docs & Paper** | `paper.tex` | Figures, tables |

---

### 5.1 Edge Predictor
*Prompt* includes question, current node label, 300‑token edge vocabulary → returns JSON list with `prob`.  
Temperature **T** calibrated on 200 validation Q/A using binary 
cross‑entropy.
Temperature-calibrated 0-Hop Edge Predictor

Prompt GPT-3.5 with question + current node + 300-token edge vocab.

Collect 200 validation pairs → fit T by minimising binary-cross-entropy.

Use calibrated edge_conf = σ(logit(p)/T) in entropy queue.

### 5.2 Entropy Queue
`priority = (1‑conf) / (depth+1)`; budget caps: `depth≤3`, `nodes≤300`, `wall_time≤800 ms`.

### 5.2.1 Dense Back-off & Seed Injection
If frontier starves, FAISS retrieves top-n sentences, maps mentions to entities (similarity ≥ 0.7), injects ≤ 5 new seeds.

### 5.3 MIRIX‑Slim Tables
episodic.sqlite (TTL 30 min) – scratch-pad.

vault.duckdb – long-term, quality=1 triples only, (t_start,t_end) for temporal queries.

Score boost: if vault edge matched → score *= 1.15.

Insert happens only when Reviewer says “yes”.
```sql
-- episodic
id PK, subj, rel, obj, evidence TEXT, ts_created
-- vault (quality‑gated)
subj, rel, obj, first_seen, last_used, quality, t_start, t_end
PRIMARY KEY(subj,rel,obj)
```
Stored in `vault.duckdb`; indexed on `(subj,rel)`.

### 5.4 IMM Router Rules (excerpt YAML)

mem_index table (hash→row_ptr) for subjects, aliases, embeddings.

Router YAML chooses store set (episodic, vault, public, dense) from query intent.
```yaml
- match: ["past", "previous", "last visit"]
  stores: ["episodic","vault","public"]
- match: ["recent trial", "new study"]
  stores: ["dense","public"]
- default:
  stores: ["vault","public"]
```

---

## 6 Indexes & Performance Targets
| Store | Index | Expected QPS |
|-------|-------|--------------|
| Episodic | b‑tree on `subj` | 50 k lookups/s |
| Vault | compound `(subj,rel)` | 2 k lookups/s |
| Public KG | same | 2k |
| FAISS | IVFPQ 4096×16 | 1 ms per 768‑d query |

Latency target: **< 2 s** end‑to‑end (average).

---
## End-to-End Scenario Table

| ID  | Scenario        | Key Path                                                                 |
|-----|------------------|-------------------------------------------------------------------------|
| S1  | Success depth-1  | Graph edge found → Reviewer yes → vault write                          |
| S2  | Ambiguous        | Planner asks follow-up → trace status NEEDS_CLARIFICATION              |
| S3  | Deep success     | Depth-3 path accepted within budget                                     |
| S4  | Dense rescue     | Graph fails, FAISS seeds entity, path found                             |
| S5  | Not found        | Budget exhausted, answer “no verified evidence”                         |
| S6  | Memory recall    | Same user 3 days later, vault boosts score                              |
| S7  | Dense outage     | dense.search 500 → Graph-only fallback, error span logged 

---

## 7 Evaluation Plan

| Metric | Dataset | Baseline | Target |
|--------|---------|----------|--------|
| Recall@5 | VAT‑KG | 0.70 | **≥ 0.78** |
| Path‑F1 | KG‑LLM‑Bench | 0.46 | **≥ 0.60** |
| Memory Recall@1 (7‑day) | Synthetic | 0.62 | **≥ 0.90** |
| Median latency | Mixed | — | **< 2 s** |

Ablations: Dense‑only, Graph‑only, No‑memory, IMM‑off.

---

## 8 Timeline & Hours (20 work‑days ~ 160 h)

| Week | Focus | Hours |
|------|-------|-------|
| 1 | Env, data ingest, FAISS build | 32 |
| 2 | Edge predictor + GraphWalker | 38 |
| 3 | MIRIX‑Slim, IMM, memory metric | 40 |
| 4 | Eval runs, paper writing, polish | 36 |
| Buffer | Neo4j/APOC swap or BioLink‑BERT link‑score | 14 |

---

## 9 Risks & Mitigations
| Risk | Plan |
|------|------|
| Edge predictor noisy | fallback BFS depth‑1 ; calibrate T |
| Vault bloat | quality flag + LRU pruning |
| Dense back‑off noise | NER similarity ≥0.7, k≤5 seeds |
| Hosted API quota | cache results; gpt‑3.5‑turbo only |

---

## 10 Novel Contributions
1. **Entropy‑pruned 0‑hop traversal** Inference-time temperature-calibrated 0-hop + entropy BFS on biomedical KG.
2. **MIRIX‑Slim**: MIRIX-Slim – first lightweight, quality-gated, time-aware memory overlay measurable via Recall@1 (7-day).
3. **IMM Router** Intelligent Memory Manager that routes queries to the correct store and indexes both symbolic & semantic keys.
4. Publish **Recall@1(7‑day)** metric + open scripts.
5. **Integrated OTEL** trace across MCP and LangGraph agents with end-session log summarisation.
---

## 11 Example End‑to‑End Trace

1. *User*: “What drug treats bacterial pneumonia in the elderly?”  
2. Planner extracts seed “Pneumonia”.  
3. Edge predictor suggests `[treated_by 0.34,…]`.  
4. GraphWalker depth‑1 hits `Pneumonia─treated_by→Azithromycin`.  
5. Reviewer yes → evidence stored in episodic & vault.  
6. Explainer returns answer + path.

Query time ≈ 0.9 s.

---

ℹ️  **File saved as:** `spec_phase1_v2.md`
