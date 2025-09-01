Perfect—let’s walk **from absolute ground-zero** through every module, exactly in the order you’ll execute the research project.
I’ll give, for **each module**:

* Purpose & success check
* Inputs / outputs & on-disk artefacts
* Key implementation steps (with code/SQL snippets)
* Built-in mitigation hooks
* Unit-test checklist

At the end you’ll get a **plain-English, end-to-end flow** covering all runtime scenarios.

---

# 🔑 MODULE-BY-MODULE DESIGN

| #     | Module                           | Clock week           | Owner hrs |
| ----- | -------------------------------- | -------------------- | --------- |
| **0** | Environment + Data Ingestion     | Week-1 Day-1 → Day-2 | 16 h      |
| **1** | Embedding Build + Dense Index    | Week-1 Day-2 → Day-3 | 8 h       |
| **2** | 0-Hop Edge Predictor (+ T-calib) | Week-1 Day-3 → Day-4 | 10 h      |
| **3** | GraphWalker + Entropy Queue      | Week-1 Day-4 → Day-5 | 10 h      |
| **4** | PET Reviewer & Rerank            | Week-1 Day-5 → Day-6 | 8 h       |
| **5** | MIRIX-Slim Memory Layer          | Week-2 Day-1 → Day-2 | 16 h      |
| **6** | Intelligent Memory Manager       | Week-2 Day-2 → Day-3 | 8 h       |
| **7** | LangGraph Agent Orchestrator     | Week-2 Day-3 → Day-4 | 10 h      |
| **8** | MCP Server & FAISS/KG RPC        | Week-2 Day-4 → Day-5 | 8 h       |
| **9** | Observability (OTEL + Jaeger)    | Week-3 Day-1 → Day-2 | 10 h      |
| 10    | Log Summariser & Archive         | Week-3 Day-2         | 4 h       |
| 11    | Evaluation Harness               | Week-3 Day-3         | 8 h       |
| 12    | UI + Demo Glue                   | Week-3 Day-4         | 6 h       |

*(Week-4 = buffer, benchmarks, paper.)*

---

## MODULE 0 Environment + Data Ingestion

**Purpose**  Install toolchain, fetch datasets, create “public” DuckDB.

### Inputs

* `vatkg_med.jsonl` (≈10 k triples)
* `kglbench_qapaths.jsonl`
* (opt.) `pubmedqa.tsv`

### Steps & code

```bash
conda create -n rag python=3.11 duckdb faiss-cpu openai \
  networkx sentence-transformers opentelemetry-sdk uvicorn fastapi
```

```python
import duckdb, json
con = duckdb.connect("public.duckdb")
con.sql("CREATE TABLE med_triples(subj TEXT, rel TEXT, obj TEXT, src TEXT, conf REAL)")
with open('vatkg_med.jsonl') as f, con as c:
    c.executemany(
      "INSERT INTO med_triples VALUES (?,?,?,?,?)",
      [(t['s'],t['p'],t['o'],'VAT',1.0) for t in map(json.loads,f)])
```

### Mitigations

* **Dataset hash check** (`sha256sum`) to detect corrupt download.
* DuckDB import inside transaction → auto-rollback on failure.

### Unit tests

* `pytest tests/test_etl.py` verifies row counts match file counts.
* `duckdb` query sanity: `SELECT COUNT(*) WHERE subj IS NULL` -> 0.

---

## MODULE 1 Embeddings & Dense Index

**Purpose**  Prepare FAISS IVFPQ index.

### Inputs

* All *unique* entity aliases + PubMedQA passages.

### Steps

```python
from sentence_transformers import SentenceTransformer
bge = SentenceTransformer("BAAI/bge-large-en")
vecs = bge.encode(texts, batch_size=64, show_progress_bar=True)
index = faiss.index_factory(768, "IVF4096,PQ16")
index.train(vecs)
index.add(vecs)
faiss.write_index(index, "dense.index")
```

Store mapping table 💾 `dense_meta.duckdb`:

```sql
CREATE TABLE dense_meta (emb_id BIGINT PRIMARY KEY, text TEXT, src TEXT);
```

### Mitigations

* If FAISS train quantiser fails (`faiss` error) → fallback to `FlatIP`.
* `--no-pubmed` flag to skip large text if time/space low.

### Tests

* `faiss_selftest.py` — recall\@10 on 1 k held-out sentences > 0.9.

---

## MODULE 2 Temperature-Calibrated 0-Hop Predictor

**Inputs**

* Gold paths JSONL (200 validation samples)

### Steps

1. **Prompt design**

   ```
   You are EdgeBot.
   Node: "{node}"
   Question: "{q}"
   Choose up to 6 relations from list: {edge_vocab}
   Output JSON list [{{"rel":"X","prob":0.42}},...]
   ```
2. **Calibration**

   ```python
   # collect (p,y)
   p_raw, y = [], []
   for sample in val:
       out = edge_predict_api(...)
       top = out[0]
       p_raw.append(top["prob"])
       y.append(int(top["rel"]==sample["gold_rel"]))
   T = learn_temperature(p_raw, y)
   json.dump({"T":T}, open("cal_T.json","w"))
   ```
3. **Runtime**

   ```python
   T = json.load(open("cal_T.json"))["T"]
   edge_conf = sigmoid(logit(p)/T)
   ```

### Mitigations

* If GPT quota hit → pull from `edge_cache.jsonl`.
* If JSON parse error → retry once; else empty list.

### Tests

* ensure calibrated Brier score < 0.20 on validation.

---

## MODULE 3 GraphWalker + Entropy Queue

**Key constants**

```python
MAX_DEPTH = 3
MAX_NODES = 300
MAX_TIME  = 0.8      # seconds
```

**Budget stops** raise `TraversalBudgetExceeded` → Planner returns NOT\_FOUND.

**Neo4j path (optional)**

```cypher
CALL apoc.path.expandConfig($startId,{
  relationshipFilter:$relList,
  minLevel:1,maxLevel:3,limit:300
})
```

Mitigations: if NetworkX median latency > 1.2 s on VAT-KG, dev toggles `GRAPH_BACKEND=neo4j`.

---

## MODULE 4 PET Reviewer & Rerank

* Hosted HF model: `Intel/PET-large-uncased-QA-retri`.
* Calls once per candidate answer; 50 ms typical.

Mitigation: fallback to `gpt-3.5-turbo {"role":"assistant_check","content":answer}` yes/no.

---

## MODULE 5 MIRIX-Slim Memory

### Inserts

```python
def write_memory(triple, evidence, quality):
    con.execute(
      "INSERT OR REPLACE INTO vault VALUES (?,?,?,?,?,?,?,?)",
      (*triple, today, today, quality, triple_date_start, triple_date_end))
    if quality:
        # also add to mem_index
        for alias in [triple[0], triple[2]]:
            con.execute("INSERT INTO mem_index VALUES (?,?,?,?)",
                        ('vault','subj',md5(alias.encode()).hexdigest(),rowid))
```

### Prune job ⚙︎

```sql
DELETE FROM vault
WHERE quality=0 AND last_used < DATE('now', '-90 day');
```

---

## MODULE 6 Intelligent Memory Manager

*Router rules* in `rules.yaml`, loaded at runtime.
Keys hashed with MD5; lookup via index table.

Mitigation: if `len(keys)==0` → dense back-off automatically chosen.

---

## MODULE 7 LangGraph Agent Orchestrator

**State machine**

```text
START → Planner
  ├─ needs_clarify → UI.ask → Planner
  └─ seeds → Retriever
        ├─ partial_ok → Reviewer
        │     ├─ yes → Explainer → END
        │     └─ no → Planner.depth+1
        └─ no_path → Planner.NO_RESULT
```

Retry: any agent span exception triggers one retry; else error span logged.

---

## MODULE 8 MCP Server

**Routes & payloads**

```yaml
POST /kg/get_triples
  in : {subj:str, rel:str|null, trace_id}
  out: [{subj,rel,obj,src,conf}]

POST /dense/search
  in : {text:str, top_k:int}
  out: [{emb_id:int, text, score}]

POST /mem/vault/insert
  in : {subj,rel,obj,quality:int,t_start,t_end}
  out: {status:"ok"}
```

Headers: `x-trace-id` echoed to OTEL span.

Mitigation: 503 if DuckDB locked; caller retries once.

---

## MODULE 9 Observability & Log Archive

### OTEL wiring

```python
tracer = trace.get_tracer("hybrid-rag")
span = tracer.start_span("GraphWalker", context=set_span_in_context(parent_span))
span.set_attribute("depth", depth)
```

Exporter: `BatchSpanProcessor(JaegerExporter(agent_host_name="localhost"))`

### Summariser ⚙︎ (on `SessionEnd`)

* Query span exporter for session spans → GPT bullet list.
* Insert into `log_archive` table in vault.

Mitigation: if GPT fails, fall back to `'\n'.join(top 10 span names)`.

---

## MODULE 10 Evaluation Harness

* CLI flags: `--memory on/off`, `--backend neo4j/networkx`, `--runs 1000`.
* Stores MLflow metrics; assertion tests verify targets met.

---

# 📜 End-to-End System Flow (Narrative)

1. **User query arrives** (`trace-id` assigned `abc-123`).
2. **Planner** extracts entities; no ambiguity → seeds `[Neutrophilia]`.
3. **IMM Router** picks stores `[vault,public]`; returns 2 vault hits (quality=1).
4. **GraphWalker** starts BFS:

   * depth-1 edge `treated_by` with conf 0.34 passes entropy queue.
   * Node expansion ≤ 300, within 400 ms.
5. **Reviewer** (PET) scores evidence 0.93 (above τ 0.7) → YES.

   * Inserts triple into episodic & vault, writes mem\_index rows.
6. **Explainer** formats answer, cites path provenance.
7. OTEL spans (Planner, Retriever, MCP kg, GraphWalker, Reviewer, Explainer) collected under `trace-id`.
8. **UI** displays answer + link “View trace” (opens Jaeger filtered by `abc-123`).
9. On user “End session”:

   * `log_summariser` condenses spans → bullet list.
   * Archive row written; episodic purged after 30 min cron.

**Fail paths**:

* **Dense outage** – MCP `/dense/search` returns 500 → Retriever logs error span → falls back graph only, returns partial or NOT\_FOUND.
* **Ambiguity** – Planner detects missing age, returns Clarify → UI prompts user → new query continues same trace-id.
* **Vault bloat** – weekly cron removes quality=0 & unused > 90 days.

---
