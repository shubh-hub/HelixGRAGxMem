import ijson
import json
import os
from tqdm import tqdm

def process_hetionet_to_vatkg_streaming():
    """
    Processes the raw Hetionet graph using a streaming parser (ijson) to be memory-efficient.
    It makes two passes: one to build a node map, and a second to process edges.
    This prevents system crashes on machines with limited RAM.
    """
    raw_path = 'data/raw/hetionet-v1.0.json'
    triples_path = 'data/processed/vatkg.jsonl'
    nodes_path = 'data/processed/nodes.jsonl' # New output file

    if not os.path.exists(raw_path):
        print(f"Error: Raw data file not found at {raw_path}")
        print("Please run the download script first: ./scripts/download_data.sh")
        return

    # --- Pass 1: Build Node Map from Stream --- #
    print("Starting Pass 1: Building node map from stream...")
    node_map = {}
    with open(raw_path, 'rb') as f:
        nodes = ijson.items(f, 'nodes.item')
        for node in tqdm(nodes, desc="Pass 1/2: Mapping Nodes"):
            # Store both name and kind (type) for each node
            node_map[node['identifier']] = {
                "name": node['name'],
                "type": node['kind']
            }
    print(f"Node map created with {len(node_map):,} entries.")

    # --- New Step: Write Node Type Map to File --- #
    print(f"Writing node type map to {nodes_path}...")
    with open(nodes_path, 'w') as f_out:
        for node_data in tqdm(node_map.values(), desc="Writing Node Types"):
            node_record = {
                "node_name": node_data['name'],
                "node_type": node_data['type']
            }
            f_out.write(json.dumps(node_record) + '\n')
    print(f"Successfully wrote node types for {len(node_map):,} nodes.")

    # --- Predicate Selection --- #
    included_predicates = {
        'treats', 'palliates', 'causes', 'presents',
        'interacts_with', 'associates', 'binds', 'regulates'
    }

    # --- Pass 2: Stream Edges and Write Triples --- #
    print("\nStarting Pass 2: Streaming edges and writing triples...")
    triples_written = 0
    with open(raw_path, 'rb') as f_in, open(triples_path, 'w') as f_out:
        edges = ijson.items(f_in, 'edges.item')
        for edge in tqdm(edges, desc="Pass 2/2: Extracting Triples"):
            predicate = edge['kind']
            if predicate in included_predicates:
                try:
                    # The source/target IDs are lists: [NodeType, NodeIdentifier]
                    # We need the second element, the actual identifier.
                    # We also cast to string to handle numeric IDs (e.g., for Genes).
                    source_identifier = str(edge['source_id'][1])
                    target_identifier = str(edge['target_id'][1])

                    # Look up names from the node map
                    subject_name = node_map.get(source_identifier, {}).get('name')
                    object_name = node_map.get(target_identifier, {}).get('name')
                    subject_type = node_map.get(source_identifier, {}).get('type')
                    object_type = node_map.get(target_identifier, {}).get('type')

                    if subject_name and object_name:
                        triple = {
                            "s_id": source_identifier,
                            "s_name": subject_name,
                            "s_type": subject_type,
                            "predicate": predicate,
                            "o_id": target_identifier,
                            "o_name": object_name,
                            "o_type": object_type
                        }
                        f_out.write(json.dumps(triple) + '\n')
                        triples_written += 1
                except (KeyError, IndexError):
                    continue

    print(f"\nFinished processing edges. Wrote {triples_written} triples to {triples_path}")

if __name__ == "__main__":
    process_hetionet_to_vatkg_streaming()
