import duckdb
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import os
import sys

# Add src to path to import config
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from config import settings

# --- Configuration ---
DB_PATH = settings.DB_PATH
OUTPUT_DIR = 'data/processed'
EMBEDDING_MODEL = 'BAAI/bge-large-en-v1.5'

# --- Main Script ---
def generate_relation_glosses():
    """
    Connects to the KG, extracts unique typed relations, generates natural
    language glosses, embeds them, and saves the results.
    """
    if not os.path.exists(DB_PATH):
        print(f"Error: Database not found at {DB_PATH}. Please run build_kg.py first.")
        return

    print(f"Connecting to database at {DB_PATH}...")
    con = duckdb.connect(DB_PATH, read_only=True)

    print("Extracting unique typed relations from the knowledge graph...")
    try:
        query = """
        SELECT DISTINCT
            s.type AS source_type,
            t.predicate AS relation,
            o.type AS object_type
        FROM triples t
        JOIN nodes s ON t.subject = s.id
        JOIN nodes o ON t.object = o.id
        """
        typed_relations = con.execute(query).fetchall()
        print(f"Found {len(typed_relations)} unique typed relations.")
    except duckdb.Error as e:
        print(f"DuckDB Error: {e}")
        print("Please ensure your 'triples' and 'nodes' tables are correctly populated.")
        con.close()
        return
    finally:
        con.close()

    # Generate glosses
    glosses = [f"A {src} {rel.replace('_', ' ')} a {obj}." for src, rel, obj in typed_relations]
    print(f"Generated {len(glosses)} relation glosses.")

    # Embed glosses
    print(f"Loading embedding model: {EMBEDDING_MODEL}...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    print("Embedding relation glosses... (This may take a moment)")
    embeddings = model.encode(glosses, show_progress_bar=True, normalize_embeddings=True)
    print(f"Generated {embeddings.shape[0]} embeddings of dimension {embeddings.shape[1]}.")

    # Save results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    gloss_labels_path = os.path.join(OUTPUT_DIR, 'relation_gloss_labels.json')
    gloss_embs_path = os.path.join(OUTPUT_DIR, 'relation_gloss_embs.npy')

    with open(gloss_labels_path, 'w') as f:
        json.dump(glosses, f, indent=2)
    print(f"Saved gloss labels to {gloss_labels_path}")

    np.save(gloss_embs_path, embeddings)
    print(f"Saved gloss embeddings to {gloss_embs_path}")

    print("\nSuccessfully generated and saved relation glosses and embeddings.")

if __name__ == "__main__":
    generate_relation_glosses()
