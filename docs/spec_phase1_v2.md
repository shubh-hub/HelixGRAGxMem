
# 📄 Project Specification & System Design  
**Project Title:** Hybrid Graph + Dense Retrieval with MIRIX‑Slim Memory (Phase‑1)  
**Date:** 2025-08-02  

---

## 1 High‑Level Goal
Build, in 4 weeks, an inference‑only retrieval‑augmented generation (RAG) system that:
1. Beats dense‑only baselines on two public biomedical KG benchmarks.
2. Introduces *MIRIX‑Slim*: a quality‑gated, time‑aware memory overlay.
3. Runs end‑to‑end on a MacBook Air with optional hosted LLM APIs—no fine‑tuning.


### Feature ↔ Methodology Map
| Feature block                | Concrete method(s)                                                                                                                                                             | Why it matters                                                                                  |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------- |
| **Graph retrieval**          | • Triples stored in **DuckDB**<br>• Walked in **NetworkX**<br>• **Hydra 0-hop** edge-type predictor (GPT-3.5) to prune fan-out<br>• **Entropy-priority queue** for deeper hops | Gives symbolic, explainable paths; edge prediction and entropy cut search cost without training |
| **Dense retrieval**          | • **BGE-large-en** embeddings (HF hosted)<br>• **FAISS** IVFPQ index on laptop                                                                                                 | Captures synonyms / evidence outside KG                                                         |
| **Fusion layer**             | Score = 0.6 · graph-score + 0.4 · dense-score; top-k merged                                                                                                                    | Balances precision (graph) and recall (dense)                                                   |
| **Agentic orchestration**    | `Planner` → `Retriever` → `Reviewer` → `Explainer` (LangGraph style)                                                                                                           | Modular, traceable; Reviewer can reject low-confidence partial answers                          |
| **MIRIX-Slim memory**        | • `episodic.sqlite` (50-turn window)<br>• `vault.duckdb` (dedup triples with time-stamps)                                                                                      | Boosts future queries; lets us test long-term recall                                            |
| **Inference-only operation** | All logic uses hosted LLM calls—**no fine-tuning**                                                                                                                             | Meets “MacBook Air + hosted components” constraint                                              |


---

### Evaluation Plan

| Step             | How we measure                                                                  |
| ---------------- | ------------------------------------------------------------------------------- |
| 1. **Retrieval** | Run 1 000 VAT-KG queries → collect top-5 evidences → compute Recall\@5 & MRR    |
| 2. **Reasoning** | Feed 500 KG-LLM-Bench questions → check answer & full KG path → compute Path-F1 |
| 3. **Memory**    | Day 0 insert memory triples → Day 7 ask 500 random facts → compute Recall\@1    |
| 4. **Latency**   | Median wall-time for 100 mixed queries (graph+dense+LLM)                        |
| 5. **Ablations** | Dense-only vs Graph-only vs Hybrid; Memory-off vs Memory-on                       |

---

### Feature ↔ Implementation Map

| #      | Feature we promise                     | How we do it (code-level)                                                                                                                                                               | Dataset(s) touched                        | Metric we report    |
| ------ | -------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------- | ------------------- |
| **F1** | **Graph retrieval** with smart hop-cut | • DuckDB holds triples<br>• NetworkX BFS<br>• *Hydra-style 0-hop* edge predictor → keep top k = 4 relations (GPT-3.5 prompt)<br>• **Entropy queue**: score = `(1−edge_conf) / (hops+1)` | VAT-KG, KG-LLM-Bench                      | Recall\@5, Path-F1  |
| **F2** | **Dense retrieval**                    | BGE-large-en embeddings → FAISS IVFPQ (768-d, PQ-M16)                                                                                                                                   | VAT-KG text; PubMedQA passages (optional) | Recall\@5           |
| **F3** | **Fusion & re-rank**                   | Merge lists; PET-Large (DeBERTa-v3 via HF API) reranks by path evidence                                                                                                                 | same                                      | Recall\@5 ↑         |
| **F4** | **Agentic loop**                       | `Planner` → `Retriever` → `Reviewer` (LLM yes/no) → `Explainer` (template)                                                                                                              | —                                         | Latency, trace log  |
| **F5** | **MIRIX-Slim memory**                  | `episodic.sqlite` (50 turns) + `vault.duckdb` (dedup; `t_start, t_end`)                                                                                                                 | synthetic 7-day triple stream             | Memory Recall\@1    |
| **F6** | **Zero-train rule**                    | All LLM calls are inference only; no fine-tune                                                                                                                                          | —                                         | Cost report (< \$5) |


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

### 5.3 MIRIX‑Slim Tables
episodic.sqlite (TTL 30 min) – scratch-pad.

vault.duckdb – long-term, quality=1 triples only, (t_start,t_end) for temporal queries.

We mark each stored triple with `quality_flag`.
```sql
CREATE TABLE vault (
   subj TEXT, rel TEXT, obj TEXT,
   t_start TIMESTAMP, t_end TIMESTAMP,
   quality INT -- 1 = accepted, 0 = rejected
);
```

*Boost rule* happens only when `quality = 1`.
If Reviewer flagged previous evidence as low-confidence, no multiplier is applied.
Prevents “bad facts snowball”.

Reviewer sets it:
```python
quality_flag = 1 if reviewer == "yes" else 0
```

Memory boost only if past interaction was a success
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

## 5.4 Multi Agent System
### Planner

### Retriever

### Reviewer

### Explainer

## What if the **user query is ambiguous or needs multi-hop reasoning?**

### Ambiguity handling

*Planner* runs a fast **intent pass** before spawning retrieval:

```python
problem, missing = llm_clarify(question)
if missing: return ask_user(missing)
```

*Example*

> **User**: “Is high WBC bad?”
> **Planner** detects:
> – missing **context** (age / clinical setting)
> – missing **metric** (how high?)
> It replies:
> “Could you specify the patient’s age and the exact WBC count?”

*Rationale* – committees like to see that the system refuses to hallucinate when context is absent.
*(Implementation load: one GPT-3.5 call; if answer arrives later, we resume pipeline.)*

### Multi-hop reasoning

The retrieval stack already supports k-hop paths:

* Graph side: BFS / entropy queue explores 1-, 2-, 3-hop trails.
* Dense side: if the answer requires bridging two distant entities, dense-to-graph seeding (optional Idea b) pulls in the missing intermediary.

Evaluation: Path-F1 on **KG-LLM-Bench** explicitly rewards multi-hop correctness, so we will know it works.

---

## 1 Keeping a k-hop search from “going rogue”

| Potential failure                                                    | Concrete guard-rail                                                                                                                                                                                               | Why it works                                                                                         |
| -------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| **1 A. Endless traversal**<br> (graph too big; answer really absent) | • **Budget stops**  <br>  `max_hops = 3`, `max_nodes_expanded = 300`, `wall_time < 800 ms`  <br>• When budget hits, we **return “not found”** with trace                                                          | Fixed cost ⇒ worst-case latency bounded; no hidden loops.                                            |
| **1 B. Prunes too early; answer one hop further**                    | • **Progressive depth**: depth-1 → review → if confidence < τ, *then* depth-2, depth-3. <br>• Reviewer computes coverage score; only escalates depth when needed.                                                 | We seldom skip the “answer-at-d+1” case because depth escalation uses live feedback, not a hard cut. |
| **1 C. Entropy path picked is wrong**                                | • After each hop, **partial answer** is scored by PET-Large; if score < σ, planner backtracks and explores next-best frontier.                                                                                    | Ensures a single high-entropy but spurious branch doesn’t dominate the search.                       |
| **1 D. Dense fall-back adds noisy seeds**                            | • Dense→Entity mapping requires:   <br>`sim(entity, mention) > 0.7` **AND** domain prefix match.  <br>• At most **k = 5** seeds injected.  <br>• Reviewer again can veto if dense-seed path stays low-confidence. | Keeps the recall bump of dense back-off but caps noise and still passes Reviewer checkpoint.         |

### Flow diagram (simplified)

```text
depth=1 →
   recall good?  → yes → stop
                 ↙ no
depth=2 →
   reviewer score ≥ τ? → yes → stop
                        ↙ no
depth=3 →
   budget hit? → yes → "answer not found"
```

*Typical latency*: 400 – 900 ms end-to-end on VAT-KG slice with these caps.

---

## 2 Handling synonyms or descriptive phrases not present verbatim in the KG

### 2 A. Entity surface-form table

During ETL we build a **`surface_aliases`** table:

| node\_id | canonical\_label | alias                   |
| -------- | ---------------- | ----------------------- |
| C0038454 | “Neutrophilia”   | “high neutrophil count” |
| …        | …                | “elevated WBC”          |

Source of aliases:
*UMLS* synonyms (or simple wiki redirects) → fits easily inside DuckDB (few 100 k rows).

**Query mapping**:

```python
SELECT node_id
FROM surface_aliases
WHERE alias = ? OR SOUNDEX(alias)=SOUNDEX(?)
LIMIT 3;
```

If still no hit → fallback to **embedding similarity**:

```python
faiss.search(entity_embeds, embed(question_span))
```

> *Example*
> User says: “elevated white-cell count”.
> Alias lookup maps to node “Neutrophilia”; traversal continues exactly as if the user had typed the canonical term.

### 2 B. Relation paraphrases

Edge predictor prompt includes exemplars:

> “If the question describes *“treated with”* or *“remedy for”*, map that to relation `treated_by`.”

LLM thus normalises relation synonyms; we never rely on verbatim match.

---

## 3 How is **`edge_confidence`** computed and kept consistent?

### 3 A. Derivation

In the 0-hop edge-prediction prompt we ask the LLM to output a **ranked JSON list**:

```json
[
 {"rel":"treated_by", "prob":0.34},
 {"rel":"risk_factor_for", "prob":0.28},
 {"rel":"associated_with", "prob":0.18},
 {"rel":"has_symptom", "prob":0.13}
]
```

The “prob” is the model’s self-estimated likelihood (we request it). We then:

* Softmax-renormalise to ensure Σ = 1
* Store `edge_conf = prob` for each expanded edge.

### 3 B. Calibration across queries

LLM self-probs can drift; we apply **temperature scaling** once per dataset:

1. Collect 200 random validation queries.
2. Compute raw LLM probs; check how often the top edge actually appears in gold path.
3. Fit a 1-parameter temperature `T` so that calibrated confidence ≈ empirical accuracy.

This standard post-hoc calibration keeps `edge_confidence` meaningful across domains.

### 3 C. Stability on re-runs

We cache 0-hop predictions keyed by `(question_hash, node_id)` → identical queries reuse the same edge list, eliminating randomness in repeated calls.

---

## 4 Answering all tough cases in practice

| Scenario                                     | What happens                                                                                                             | Returned message                                                                          |
| -------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------- |
| **No answer within budget**                  | Planner emits `"NOT_ANSWERED"`                                                                                           | “I couldn’t locate a verified answer in my medical graph. Could you specify more detail?” |
| **Dense fallback dominates** (lots of noise) | Reviewer sees low PET score, returns low confidence → Planner triggers clarification or returns “insufficient evidence”. | Explanation + request for clarification                                                   |
| **Synonym not in alias table**               | Embedding search links to nearest entity; if similarity < 0.6 it rejects and asks user to rephrase.                      | “I’m not sure which condition you mean by ‘sticky lung’. Could you clarify?”              |

---



### 5.4.1 IMM Router Rules (excerpt YAML)

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
## Observability / Interoperability

OTEL spans inside every agent & MCP RPC.

Session-end task summarises trace to ≤ 2 KB via GPT-3.5; archives to log_archive.
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

