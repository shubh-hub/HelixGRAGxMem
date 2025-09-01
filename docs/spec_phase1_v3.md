Below is a **full-length, engineer-ready specification**—*not just an outline*—covering every component down to schemas, API payloads, background jobs, and on-disk layout.  You can hand this directly to a teammate (or an AI coding tool) and start building.

---

## 0 Legend & Conventions

| Symbol          | Meaning                     |
| --------------- | --------------------------- |
| **💾** file     | persistent on-disk artefact |
| **📥** / **📤** | request / response payload  |
| **⚙︎** job      | background task or cron     |
| **🔑**          | primary or composite key    |

All code snippets are Python 3.11 unless noted.

---

## 1 Data Stores & Schemas

### 1.1 Public KG 💾 `public.duckdb`

```sql
CREATE TABLE med_triples (
  subj TEXT,
  rel  TEXT,
  obj  TEXT,
  src  TEXT,          -- provenance, e.g. "PubMed", "VAT"
  conf REAL,          -- 0-1 score
  PRIMARY KEY(subj,rel,obj)
);

CREATE INDEX idx_med_sr ON med_triples(subj, rel);
```

### 1.2 Vault Memory 💾 `vault.duckdb`

```sql
CREATE TABLE vault (
  subj       TEXT,
  rel        TEXT,
  obj        TEXT,
  first_seen DATE,
  last_used  DATE,
  quality    BOOLEAN DEFAULT 0,  -- set by Reviewer
  t_start    DATE,
  t_end      DATE,
  src        TEXT DEFAULT 'MEM', -- for provenance
  PRIMARY KEY(subj,rel,obj)
);

CREATE INDEX idx_vault_sr ON vault(subj, rel);
```

### 1.3 Episodic Cache 💾 `episodic.sqlite`

```sql
CREATE TABLE episodic (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  subj        TEXT,
  rel         TEXT,
  obj         TEXT,
  evidence    TEXT,
  source_qid  TEXT,
  ts_created  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX idx_epi_subj ON episodic(subj);
```

`⚙︎ cron`:

```bash
DELETE FROM episodic
WHERE ts_created < DATETIME('now', '-30 minutes');
```

### 1.4 Intelligent Memory Manager Index 💾 inside `vault.duckdb`

```sql
CREATE TABLE mem_index (
  store    TEXT,                   -- 'episodic' / 'vault' / 'public' / 'dense'
  key_type TEXT,                   -- 'subj' | 'alias' | 'emb'
  key_hash TEXT,
  row_ptr  BIGINT
);
CREATE INDEX idx_mem_hash ON mem_index(key_hash);
```

*Insertion* happens via trigger on vault OR during FAISS build for `key_type='emb'`.

---

## 2 Micro-Servers

### 2.1 MCP Server (`mcp.py`) – FastAPI

| Route                    | 📥 request                                             | 📤 response               | Notes                      |
| ------------------------ | ------------------------------------------------------ | ------------------------- | -------------------------- |
| `POST /kg/get_triples`   | `{ "subj": "Pneumonia", "rel": null, "trace_id": "…"}` | `[{subj,rel,obj,src}]`    | `rel==null` ⇒ any outgoing |
| `POST /dense/search`     | `{ "text": "lung infection", "top_k":10}`              | `[{text, score, emb_id}]` | uses FAISS IVFPQ           |
| `POST /mem/vault/insert` | `{subj,rel,obj,quality}`                               | `200 OK`                  | called by Reviewer         |

*OpenTelemetry middleware* wraps every route; `x-trace-id` header becomes root span if present.

### 2.2 Jaeger All-in-One

`docker run -d --name jaeger -p16686:16686 jaegertracing/all-in-one:1.56`

### 2.3 (Opt) Neo4j+APOC

*Docker*

```bash
docker run -d --name neo4j \
  -p7474:7474 -p7687:7687 \
  -e NEO4J_AUTH=none \
  neo4j:5-community
```

*CSV Import* executed once; traversal switched with env flag `GRAPH_BACKEND=neo4j`.

---

## 3 Critical Algorithms

### 3.1 Temperature-Calibrated 0-Hop Predictor

```python
def edge_predict(node: str, question: str) -> list[dict]:
    prompt = PROMPT_TMPL.format(node=node, q=question, edges=EDGE_VOCAB)
    raw = openai.ChatCompletion.create(...).choices[0].message.content
    preds = json.loads(raw)                       # list of {"rel":..., "prob":...}
    for p in preds:
        p["conf"] = sigmoid(logit(p["prob"])/T)   # T learned offline
    return preds[:K]                              # K=6
```

*Offline calibration* (`calibrate_T.py`) loads 200 validation queries, minimises BCE using LBFGS → writes `cal_T.json`.

### 3.2 Entropy Queue GraphWalker

```python
import heapq, time
def walker(seeds):
    start = time.time()
    frontier = []
    for s in seeds:
        heapq.heappush(frontier, (0.5, [s]))      # (priority, path)
    explored = set()
    while frontier and time.time()-start < 0.8:
        prio, path = heapq.heappop(frontier)
        node = path[-1]
        if (node,len(path)) in explored: continue
        explored.add((node,len(path)))
        if reviewer_precheck(path):
            yield path
        if len(path) >= 3: continue
        for rel_conf in edge_predict(node, GLOBAL_Q)[:4]:
            rel, conf = rel_conf["rel"], rel_conf["conf"]
            for obj in kg_lookup(node, rel):
                heapq.heappush(frontier,
                    ((1-conf)/(len(path)+1), path+[rel,obj]))
        if len(explored) > 300: break
```

### 3.3 IMM Router

```python
def choose_stores(question: str) -> list[str]:
    for rule in RULES_YAML:
        if any(k in question.lower() for k in rule["match"]):
            return rule["stores"]
    return RULES_YAML[-1]["stores"]

def mem_lookup(keys):
    stores = choose_stores(GLOBAL_Q)
    cands = []
    for k in keys:
        h = md5(k.encode()).hexdigest()
        for store,row in duck_con.execute(
           "SELECT store,row_ptr FROM mem_index WHERE key_hash=?", [h]):
            if store not in stores: continue
            cands.append(load_row(store,row))
    return cands[:10]
```

---

## 4 Agents & Message Flow

```
┌──── Planner (span: plan) ─────────────────────────────────────────┐
│ extract_entities() → seeds                                       │
│ if missing context → return CLARIFY                              │
└──┬────────────────────────────────────────────────────────────────┘
   │ seeds + trace-id
   ▼
Retriever (span: retrieve)
   ├─ GraphRetriever.span
   │     ↳ MCP /kg/get_triples  (trace-link)
   ├─ DenseRetriever.span
   │     ↳ MCP /dense/search    (trace-link)
   └─ IMM lookup (local)
   ▼
Reviewer (span: review)
   PET-lr predicts YES/NO + score
   │ YES → write evidence to episodic & vault (/mem/vault/insert)
   │      → Explainer
   │ NO  → if depth<budget → Planner (ask follow-up or deeper)
   ▼
Explainer (span: explain)
   Build persona-aware NL answer + cite path provenance
   Return to UI; OTEL exporter flushes
```

Trace tree is viewable in Jaeger with `trace-id` matching UI request id.

---

## 5 Logging, Summaries, Archiving

*Every OTEL span* emits key attributes: `component`, `store`, `depth`, `hit_type`, `latency_ms`.
`⚙︎ log_summariser` runs on `SessionEnd`:

```python
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
spans = span_exporter.get_finished_spans(trace_id=session.trace_id)
summary = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[{"role":"system","content":SUMMARISER_PROMPT},
            {"role":"user","content":json.dumps([s.to_json() for s in spans])}]
).choices[0].message.content[:2000]
duck.execute("INSERT INTO log_archive VALUES (?,?,datetime('now'))",
             [session.id, summary, session.trace_id])
```

Summary ≤ 2 KB, e.g.:

> • Graph traversal (depth 2) accepted, 220 ms.
> • Dense back-off not used.
> • Reviewer confidence 0.93; evidence stored to vault row 42.

---

## 6 Evaluation Harness (`benchmark.py`)

```bash
python benchmark.py --dataset vatkg --runs 1000 --memory off
python benchmark.py --dataset vatkg --runs 1000 --memory on
python benchmark.py --dataset kglbench --runs 500
```

Outputs MLflow metrics:

* `recall@5`, `mrr`, `path_f1`, `latency_p50`, `memory_recall@1`.

*Scenario tests* scripted in `tests/scenarios_test.py` (PyTest).

---

## 7 Timeline (detail)

| Day | Deliverable                                      | Hours | Owner |
| --- | ------------------------------------------------ | ----- | ----- |
| 1   | repo scaffold, Docker compose, data download     | 8     | you   |
| 2   | ETL DuckDB, FAISS build                          | 8     | you   |
| 3   | Edge predictor prompt, cache, calibration script | 10    | you   |
| 4   | GraphWalker prototype, unit tests                | 10    | you   |
| 5   | PET Re-ranker integration, latency test          | 8     | you   |
| 6   | MIRIX-Slim tables + TTL cron                     | 8     | you   |
| 7   | IMM Router + alias loader                        | 8     | you   |
| 8   | MCP routes + OTEL middleware                     | 8     | you   |
| 9   | End-to-end Happy path (S1) passes                | 8     | you   |
| 10  | Reviewer negative flow (S2, S5)                  | 8     | you   |
| 11  | Dense back-off flow (S4)                         | 8     | you   |
| 12  | Memory recall scenario (S6)                      | 6     | you   |
| 13  | Log summariser + archive                         | 6     | you   |
| 14  | Jaeger docker & dashboards                       | 4     | you   |
| 15  | Benchmark scripts + MLflow                       | 8     | you   |
| 16  | Write Results section                            | 6     | you   |
| 17  | Write Methods section                            | 6     | you   |
| 18  | Paper polish, figs                               | 6     | you   |
| 19  | Demo UI (Streamlit)                              | 6     | you   |
| 20  | Buffer / Neo4j or link-score                     | 6     | you   |

Total ≈ 160 h.

---

## 8 Open Risks & Contingencies

| Risk                    | Mitigation                                    |
| ----------------------- | --------------------------------------------- |
| GPT-3.5 API quota       | cache edge predictions; local TransE fallback |
| PET endpoint latency    | fallback to GPT-3.5 classification            |
| Vault bloat             | weekly LRU prune or quality<1 purge           |
| Jaeger overhead >100 ms | switch to OTLP exporter with sampling         |

---

## 9 Future-Work Hooks (Phase-2 stretch)

* Add Guideline KG & TherapyPlanner agent.
* Swap NetworkX BFS for APOC `shortestPathBi` (after Neo4j import).
* Memory distillation (LLM summary → replace 1 k old triples).

---

