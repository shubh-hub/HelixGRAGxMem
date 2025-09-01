# HelixRAGxMem: Dataset Acquisition Guide (Phase 1)

**Last Updated**: August 6, 2025  
**Status**: Finalized for Phase 1 Implementation

---

## 1. Phase 1 Objective: Controlled Complementarity Experiment

This guide details the data acquisition and preparation steps for **Phase 1**. The primary goal is to create a controlled environment to test the complementarity of Knowledge Graph (KG) and Dense retrieval. To achieve this, both retrieval systems will use the **exact same underlying data source**: a biomedical KG derived from **Hetionet**.

### Required Data Assets for Phase 1:

| Asset | Source | Format | Purpose |
|-------|--------|--------|---------|
| **VAT-KG** | Hetionet | Triples | The Knowledge Graph for the KG-only and Hybrid systems. |
| **Verbalized VAT-KG** | VAT-KG | Sentences | The dense retrieval corpus for the Dense-only and Hybrid systems. |
| **KG-LLM-Bench** | Public | Q&A Pairs | The benchmark for evaluating multi-hop reasoning performance. |

---

## 2. Step 1: Acquire and Process Hetionet to Create VAT-KG

The first step is to download the Hetionet knowledge graph and process it into our standardized `VAT-KG` format.

### Access and Download

```bash
# Create directories for raw and processed data
mkdir -p data/raw data/processed

# Download Hetionet v1.0
wget -O data/raw/hetionet-v1.0.json.bz2 https://github.com/hetio/hetionet/raw/master/hetnet/json/hetionet-v1.0.json.bz2

# Extract the file
bunzip2 data/raw/hetionet-v1.0.json.bz2
```

### Process into VAT-KG Format

We will extract relevant biomedical triples to create our `vatkg.jsonl` file.

```python
# scripts/prepare_vatkg.py
import json

def process_hetionet_to_vatkg():
    """
    Loads the raw Hetionet graph and extracts a curated set of biomedical triples
    to create the VAT-KG dataset.
    """
    with open('data/raw/hetionet-v1.0.json', 'r') as f:
        hetionet = json.load(f)

    # Example: Extract a subset of 'treats' and 'causes' relationships
    # In the full implementation, we'll select a diverse ~10k triples.
    vatkg_triples = []
    for edge in hetionet.get('edges', []):
        if edge['kind'] in ['treats', 'causes']:
            source_node = hetionet['nodes'][edge['source_id']]
            target_node = hetionet['nodes'][edge['target_id']]
            vatkg_triples.append({
                "subject": source_node['name'],
                "predicate": edge['kind'],
                "object": target_node['name']
            })

    with open('data/processed/vatkg.jsonl', 'w') as f:
        for triple in vatkg_triples:
            f.write(json.dumps(triple) + '\n')

    print(f"Successfully created data/processed/vatkg.jsonl with {len(vatkg_triples)} triples.")

if __name__ == "__main__":
    process_hetionet_to_vatkg()
```

---

## 3. Step 2: Create Verbalized VAT-KG for Dense Retrieval

Next, we convert the structured triples from `VAT-KG` into natural language sentences to create the corpus for dense retrieval.

```python
# scripts/verbalize_kg.py
import json

def verbalize_vatkg():
    """
    Reads the VAT-KG triples and converts them into natural language sentences
    using predefined templates.
    """
    templates = {
        "treats": "{subject} is a condition that is treated by {object}.",
        "causes": "{subject} is known to be a cause of {object}.",
        # Add other templates as needed
        "default": "There is a relationship of type '{predicate}' between {subject} and {object}."
    }

    verbalized_data = []
    with open('data/processed/vatkg.jsonl', 'r') as f:
        for line in f:
            triple = json.loads(line)
            template = templates.get(triple['predicate'], templates['default'])
            sentence = template.format(**triple)
            verbalized_data.append({
                "id": f"{triple['subject']}-{triple['predicate']}-{triple['object']}",
                "text": sentence,
                "metadata": triple
            })

    with open('data/processed/vatkg_verbalized.jsonl', 'w') as f:
        for item in verbalized_data:
            f.write(json.dumps(item) + '\n')

    print(f"Successfully created data/processed/vatkg_verbalized.jsonl with {len(verbalized_data)} sentences.")

if __name__ == "__main__":
    verbalize_vatkg()
```

---

## 4. Step 3: Acquire KG-LLM-Bench for Evaluation

Finally, we clone the standard benchmark suite we will use for evaluation.

```bash
# Clone the repository
git clone https://github.com/AKSW/LLM-KG-Bench.git vendor/LLM-KG-Bench

# We will use its biomedical subset for our evaluation tasks.
# No further processing is needed at this stage.
```

---

## 5. Final Directory Structure for Phase 1

After running these steps, your `data` directory should look like this:

```
data/
├── processed/
│   ├── vatkg.jsonl                 # Structured KG triples
│   └── vatkg_verbalized.jsonl    # Dense retrieval corpus
└── raw/
    └── hetionet-v1.0.json        # Original downloaded KG

vendor/
└── LLM-KG-Bench/                 # Evaluation benchmark suite
