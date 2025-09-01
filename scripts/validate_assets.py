import duckdb
import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer

def validate_all_assets():
    """
    Performs a validation check on all data assets created during Week 1.
    It checks the KG database and the dense retrieval index.
    """
    print("--- Starting Asset Validation ---")

    # --- 1. Validate Knowledge Graph (DuckDB) --- #
    print("\n[1/2] Validating Knowledge Graph (DuckDB)...")
    db_path = 'data/processed/knowledge_graph.db'
    try:
        con = duckdb.connect(database=db_path, read_only=True)
        total_triples = con.execute("SELECT COUNT(*) FROM triples").fetchone()[0]
        print(f"✅  Successfully connected to DB. Total triples: {total_triples:,}")
        sample_triple = con.execute("SELECT * FROM triples LIMIT 1").fetchone()
        print(f"✅  Sample triple: {sample_triple}")
        con.close()
    except Exception as e:
        print(f"❌  Error validating DuckDB: {e}")
        return

    # --- 2. Validate Dense Index (FAISS) --- #
    print("\n[2/2] Validating Dense Retrieval Index (FAISS)...")
    index_path = 'data/processed/dense_index.faiss'
    map_path = 'data/processed/faiss_id_map.json'
    sentences_path = 'data/processed/verbalized_kg.jsonl'
    try:
        # Load all assets
        index = faiss.read_index(index_path)
        with open(map_path, 'r') as f:
            id_map = json.load(f)
        with open(sentences_path, 'r') as f:
            sentences_data = [json.loads(line) for line in f]
        print(f"✅  Successfully loaded FAISS index ({index.ntotal} vectors), ID map, and sentences.")

        # Perform a sample search
        model = SentenceTransformer('BAAI/bge-large-en-v1.5')
        # For BGE models, it's recommended to add an instruction to the query for retrieval tasks.
        query_text = "Represent this sentence for searching relevant passages: What drug treats hypertension?"
        query_vector = model.encode([query_text])
        faiss.normalize_L2(query_vector)

        k = 5  # Number of results to retrieve
        distances, indices = index.search(query_vector, k)

        print(f"\n✅  Sample query: '{query_text}'")
        print(f"✅  Top {k} results:")
        for i in range(k):
            doc_id = id_map[indices[0][i]]
            retrieved_sentence = sentences_data[doc_id]['text']
            score = distances[0][i]
            print(f"  - (Score: {score:.4f}) {retrieved_sentence}")

    except Exception as e:
        print(f"❌  Error validating FAISS index: {e}")

    print("\n--- Asset Validation Complete ---")

if __name__ == "__main__":
    validate_all_assets()
