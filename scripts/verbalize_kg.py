import json
import os
from tqdm import tqdm

# --- Configuration ---
INPUT_FILE = 'data/processed/vatkg.jsonl'
OUTPUT_FILE = 'data/processed/verbalized_kg.jsonl'

RELATION_GLOSS = {
    'treats':       "treatment relation (drug treats disease)",
    'palliates':    "relieves symptoms relation (drug palliates disease)",
    'causes':       "causation relation (X causes Y)",
    'presents':     "symptom relation (disease presents symptom)",
    'interacts_with':"biological interaction relation (gene/compound interacts)",
    'associates':   "association relation (gene associates with disease)",
    'binds':        "binding relation (compound binds gene/protein)",
    'regulates':    "regulation relation (gene regulates gene/protein)",
    # Add other relations from your KG as needed
}

def get_verbalization_template(predicate):
    """Returns a natural language template for a given predicate."""
    templates = {
        'treats': "{subject} is a medication that is used for the treatment of {object}.",
        'palliates': "{subject} is a medication used to palliate or relieve the symptoms of {object}.",
        'causes': "{subject} is known to cause the side effect or condition {object}.",
        'presents': "The disease {subject} commonly presents with the symptom {object}.",
        'interacts_with': "The compound or gene {subject} interacts with the compound or gene {object}.",
        'associates': "The gene {subject} is associated with the disease {object}.",
        'binds': "The compound {subject} binds to the gene {object}.",
        'regulates': "The gene {subject} regulates the activity of the gene {object}.",
        'default': "There is a relationship of type '{predicate}' between {subject} and {object}."
    }
    return templates.get(predicate, templates['default'])

def verbalize_kg():
    """
    Reads the structured triples from vatkg.jsonl and converts them into
    natural language sentences using predefined templates.
    """
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Input file not found at {INPUT_FILE}")
        print("Please run the prepare_vatkg.py script first.")
        return

    print(f"Starting verbalization of {INPUT_FILE}...")
    verbalized_sentences = []
    with open(INPUT_FILE, 'r') as f_in:
        for line in tqdm(f_in, desc="Verbalizing Triples"):
            triple = json.loads(line)
            subject = triple['s_name']
            predicate = triple['predicate']
            obj = triple['o_name']

            template = get_verbalization_template(predicate)
            sentence = template.format(subject=subject, object=obj, predicate=predicate)
            
            verbalized_sentences.append({
                "id": len(verbalized_sentences),
                "text": sentence,
                "predicate": predicate,
                "relation_gloss": RELATION_GLOSS.get(predicate, predicate), # Fallback to predicate if no gloss
                "metadata": {
                    "subject": subject,
                    "subject_type": triple.get("s_type"),
                    "predicate": predicate,
                    "object": obj,
                    "object_type": triple.get("o_type")
                }
            })

    print(f"Writing {len(verbalized_sentences):,} verbalized sentences to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w') as f_out:
        for item in verbalized_sentences:
            f_out.write(json.dumps(item) + '\n')

    print(f"\nSuccessfully created {OUTPUT_FILE}")

if __name__ == "__main__":
    verbalize_kg()
