import duckdb
import os

def build_kg_database():
    """
    Loads the processed KG triples from the JSONL file into a DuckDB database.
    It creates a 'triples' table and builds indexes on all columns for fast lookups.
    """
    # Define paths
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
    db_path = os.path.join(data_dir, 'processed', 'knowledge_graph.db')
    triples_path = os.path.join(data_dir, 'processed', 'vatkg.jsonl')
    nodes_path = os.path.join(data_dir, 'processed', 'nodes.jsonl')

    # Ensure processed directory exists
    os.makedirs(os.path.join(data_dir, 'processed'), exist_ok=True)

    # Remove old DB if it exists
    if os.path.exists(db_path):
        os.remove(db_path)
        print(f"Removed existing database at {db_path}")

    print(f"Connecting to DuckDB database at {db_path}...")
    con = duckdb.connect(database=db_path)

    print(f"Creating 'nodes' table and loading data from {nodes_path}...")
    # Load nodes first since we need them to enrich triples with type information
    con.execute(f"""CREATE TEMP TABLE temp_nodes AS SELECT * FROM read_json_auto('{nodes_path}');""")

    # Create the final nodes table with the primary key constraint
    con.execute("""
    CREATE TABLE nodes (
        id VARCHAR PRIMARY KEY,
        name VARCHAR,
        type VARCHAR
    );
    INSERT INTO nodes (id, name, type)
    SELECT 
        node_name AS id,
        node_name AS name, 
        MIN(node_type) AS type
    FROM temp_nodes
    WHERE node_name IS NOT NULL
    GROUP BY node_name;
    """)
    print(f"Successfully loaded {con.execute('SELECT COUNT(*) FROM nodes').fetchone()[0]} rows into 'nodes' table.")

    print(f"Creating 'triples' table and loading data from {triples_path}...")
    # First load raw triples into a temp table
    con.execute(f"""CREATE TEMP TABLE temp_triples AS SELECT * FROM read_json_auto('{triples_path}');""")
    
    # Create final triples table with type information joined from nodes
    con.execute("""
    CREATE TABLE triples (
        subject_id VARCHAR,
        subject VARCHAR,
        subject_type VARCHAR,
        predicate VARCHAR,
        object_id VARCHAR,
        object VARCHAR,
        object_type VARCHAR
    );
    INSERT INTO triples (subject_id, subject, subject_type, predicate, object_id, object, object_type)
    SELECT 
        t.subject AS subject_id,
        t.subject,
        n1.type AS subject_type,
        t.predicate,
        t.object AS object_id,
        t.object,
        n2.type AS object_type
    FROM temp_triples t
    LEFT JOIN nodes n1 ON t.subject = n1.name
    LEFT JOIN nodes n2 ON t.object = n2.name;
    """)
    print(f"Successfully loaded {con.execute('SELECT COUNT(*) FROM triples').fetchone()[0]} rows into 'triples' table with type information.")
    print(f"Successfully loaded {con.execute('SELECT COUNT(*) FROM nodes').fetchone()[0]} rows into 'nodes' table.")

    print("Creating statistical tables for relation pruning...")

    # 1. Type-level predicate prior (what relations are common for a given node type?)
    con.execute("""
    CREATE OR REPLACE TABLE type_rel_prior AS
    WITH type_rel_stats AS (
        SELECT subject_type AS source_type, predicate, COUNT(*) AS freq
        FROM triples
        GROUP BY 1,2
    )
    SELECT source_type, predicate,
           CAST(freq AS DOUBLE) / SUM(freq) OVER (PARTITION BY source_type) AS prior
    FROM type_rel_stats;
    """)
    print("Created 'type_rel_prior' table.")

    # 2. Node-local predicate degree (does a node have a relation, and how many?)
    con.execute("""
    CREATE OR REPLACE TABLE node_rel_degree AS
    SELECT subject_id AS node_id, subject AS node_name, predicate, COUNT(*) AS deg
    FROM triples
    GROUP BY 1,2,3;
    """)
    print("Created 'node_rel_degree' table.")

    # 3. Target-type compatibility (what object types are expected for a given source/relation?)
    con.execute("""
    CREATE OR REPLACE TABLE rel_target_type AS
    SELECT subject_type AS source_type, predicate, object_type AS target_type, COUNT(*) AS freq
    FROM triples
    GROUP BY 1,2,3;
    """)
    print("Created 'rel_target_type' table.")

    print("Creating composite indexes for faster queries...")
    con.execute("CREATE INDEX IF NOT EXISTS idx_subj_pred ON triples(subject, predicate);")
    con.execute("CREATE INDEX IF NOT EXISTS idx_pred_obj ON triples(predicate, object);")
    con.execute("CREATE INDEX IF NOT EXISTS idx_subject_type ON triples(subject_type);")
    con.execute("CREATE INDEX IF NOT EXISTS idx_object_type ON triples(object_type);")
    con.execute("CREATE INDEX IF NOT EXISTS idx_node_rel_deg ON node_rel_degree(node_id, predicate);")
    con.execute("CREATE INDEX IF NOT EXISTS idx_type_rel_prior ON type_rel_prior(source_type, predicate);")
    con.execute("CREATE INDEX IF NOT EXISTS idx_rel_target_type ON rel_target_type(source_type, predicate, target_type);")
    print("Indexes created successfully.")

    con.close()
    print("\nDatabase build complete.")
    print(f"DB created at: {db_path}")

if __name__ == "__main__":
    build_kg_database()
