import json
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# --- Configuration ---
INPUT_FILE = 'data/processed/verbalized_kg.jsonl'
INDEX_PATH = 'data/processed/dense_index.faiss'
ID_MAP_PATH = 'data/processed/faiss_id_map.json'
MODEL_NAME = 'BAAI/bge-large-en-v1.5'

# Relation glosses - should be consistent with verbalize_kg.py
RELATION_GLOSS = {
    'treats':       "treatment relation (drug treats disease)",
    'palliates':    "relieves symptoms relation (drug palliates disease)",
    'causes':       "causation relation (X causes Y)",
    'presents':     "symptom relation (disease presents symptom)",
    'interacts_with':"biological interaction relation (gene/compound interacts)",
    'associates':   "association relation (gene associates with disease)",
    'binds':        "binding relation (compound binds gene/protein)",
    'regulates':    "regulation relation (gene regulates gene/protein)",
}
GLOSS_EMB_PATH = 'data/processed/relation_gloss_embs.npy'
GLOSS_LABELS_PATH = 'data/processed/relation_gloss_labels.json'


def build_dense_index():
    """
    Builds a FAISS index for dense retrieval from the verbalized KG sentences.
    It uses a sentence-transformer model to create embeddings and saves the
    resulting index and an ID mapping to disk.
    """
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Input file not found at {INPUT_FILE}")
        print("Please run the verbalize_kg.py script first.")
        return

    print("Loading verbalized sentences...")
    sentences = []
    doc_ids = []
    with open(INPUT_FILE, 'r') as f:
        for line in f:
            data = json.loads(line)
            sentences.append(data['text'])
            doc_ids.append(data['id'])

    # --- Initialize Sentence Transformer Model --- #
    # Using a powerful model suitable for biomedical text retrieval
    print(f"Loading sentence transformer model: {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)
    embedding_dim = model.get_sentence_embedding_dimension()
    print(f"Model loaded. Embedding dimension: {embedding_dim}")

    # --- Encode Sentences into Embeddings --- #
    print(f"Encoding {len(sentences):,} sentences into embeddings...")
    embeddings = model.encode(sentences, show_progress_bar=True, convert_to_numpy=True)

    # --- Build and Save FAISS Index --- #
    print(f"Building FAISS index...")
    # Using a flat index with Inner Product (IP) for similarity, as recommended for BGE models
    index = faiss.IndexFlatIP(embedding_dim)
    # BGE models require normalization for effective IP search
    faiss.normalize_L2(embeddings)
    index.add(embeddings)

    print(f"Saving FAISS index to {INDEX_PATH}...")
    faiss.write_index(index, INDEX_PATH)

    # --- Save ID Mapping --- #
    print(f"Saving ID map to {ID_MAP_PATH}...")
    with open(ID_MAP_PATH, 'w') as f:
        json.dump(doc_ids, f)

    print(f"\nSuccessfully built and saved FAISS index with {index.ntotal} vectors.")

    # --- 2. Build and save relation gloss embeddings ---
    print("\nEncoding relation glosses...")
    gloss_labels = sorted(RELATION_GLOSS.keys())
    gloss_texts = [RELATION_GLOSS[k] for k in gloss_labels]

    gloss_embs = model.encode(gloss_texts, convert_to_numpy=True, show_progress_bar=False)
    faiss.normalize_L2(gloss_embs)

    np.save(GLOSS_EMB_PATH, gloss_embs)
    with open(GLOSS_LABELS_PATH, 'w') as f:
        json.dump(gloss_labels, f)

    print(f"Saved relation gloss embeddings to {GLOSS_EMB_PATH}")
    print(f"Saved relation gloss labels to {GLOSS_LABELS_PATH}")

if __name__ == "__main__":
    build_dense_index()
