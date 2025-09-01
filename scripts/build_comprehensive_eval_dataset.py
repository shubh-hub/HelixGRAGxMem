#!/usr/bin/env python3
"""
Comprehensive Evaluation Dataset Builder
=======================================

Creates a large, diverse, critically distributed evaluation dataset for biomedical QA.
Covers all edge cases, query types, and difficulty levels for robust system assessment.
"""

import json
import logging
import random
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple
import itertools

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ComprehensiveEvalDatasetBuilder:
    """Builds comprehensive evaluation dataset with critical distribution coverage"""
    
    def __init__(self):
        self.dataset = []
        
        # Biomedical entity categories for systematic coverage
        self.drugs = [
            "aspirin", "metformin", "ibuprofen", "acetaminophen", "insulin",
            "warfarin", "digoxin", "furosemide", "lisinopril", "atorvastatin",
            "omeprazole", "levothyroxine", "amlodipine", "simvastatin", "losartan",
            "hydrochlorothiazide", "prednisone", "albuterol", "gabapentin", "tramadol"
        ]
        
        self.diseases = [
            "diabetes", "hypertension", "heart disease", "cancer", "asthma",
            "arthritis", "depression", "anxiety", "obesity", "stroke",
            "pneumonia", "bronchitis", "migraine", "epilepsy", "osteoporosis",
            "alzheimer disease", "parkinson disease", "multiple sclerosis", "lupus", "psoriasis"
        ]
        
        self.symptoms = [
            "chest pain", "shortness of breath", "fatigue", "nausea", "headache",
            "dizziness", "fever", "cough", "abdominal pain", "joint pain",
            "muscle weakness", "memory loss", "confusion", "tremor", "seizure",
            "rash", "swelling", "palpitations", "insomnia", "weight loss"
        ]
        
        self.anatomy = [
            "heart", "lungs", "liver", "kidney", "brain", "stomach", "intestine",
            "pancreas", "thyroid", "adrenal gland", "spleen", "gallbladder",
            "bladder", "prostate", "uterus", "ovary", "bone", "muscle", "skin", "eye"
        ]
        
        # Query complexity levels
        self.complexity_levels = ["simple", "moderate", "complex", "expert"]
        
        # Intent distributions
        self.intents = ["factoid", "enumeration", "causal", "comparative", "procedural"]
        
    def generate_factoid_queries(self, count: int = 100) -> List[Dict[str, Any]]:
        """Generate factoid queries (What is X? How does Y work?)"""
        queries = []
        
        templates = [
            ("What is {entity}?", ["drug", "disease", "symptom", "anatomy"]),
            ("How does {entity} work?", ["drug"]),
            ("What causes {entity}?", ["disease", "symptom"]),
            ("What is the function of {entity}?", ["anatomy"]),
            ("How is {entity} diagnosed?", ["disease"]),
            ("What are the mechanisms of {entity}?", ["drug"]),
            ("What is the pathophysiology of {entity}?", ["disease"]),
            ("How does {entity} affect the body?", ["drug", "disease"]),
            ("What is the role of {entity} in the body?", ["anatomy"]),
            ("How is {entity} metabolized?", ["drug"])
        ]
        
        for i in range(count):
            template, entity_types = random.choice(templates)
            entity_type = random.choice(entity_types)
            
            if entity_type == "drug":
                entity = random.choice(self.drugs)
                expected_answers = [entity, f"{entity} mechanism", f"{entity} action"]
            elif entity_type == "disease":
                entity = random.choice(self.diseases)
                expected_answers = [entity, f"{entity} pathology", f"{entity} etiology"]
            elif entity_type == "symptom":
                entity = random.choice(self.symptoms)
                expected_answers = [entity, f"{entity} causes", f"{entity} mechanism"]
            else:  # anatomy
                entity = random.choice(self.anatomy)
                expected_answers = [entity, f"{entity} function", f"{entity} anatomy"]
            
            query = template.format(entity=entity)
            
            queries.append({
                "query": query,
                "intent": "factoid",
                "complexity": random.choice(["simple", "moderate"]),
                "entity_type": entity_type,
                "primary_entity": entity,
                "expected_answers": expected_answers,
                "query_type": "factoid",
                "difficulty": "basic" if entity in self.drugs[:10] else "intermediate"
            })
        
        return queries
    
    def generate_enumeration_queries(self, count: int = 100) -> List[Dict[str, Any]]:
        """Generate enumeration queries (What are the X? List all Y that...)"""
        queries = []
        
        templates = [
            ("What drugs treat {entity}?", ["disease"]),
            ("What are the side effects of {entity}?", ["drug"]),
            ("What are the symptoms of {entity}?", ["disease"]),
            ("What are the causes of {entity}?", ["disease", "symptom"]),
            ("What medications are used for {entity}?", ["disease"]),
            ("What are the complications of {entity}?", ["disease"]),
            ("What are the risk factors for {entity}?", ["disease"]),
            ("What are the contraindications of {entity}?", ["drug"]),
            ("What are the interactions of {entity}?", ["drug"]),
            ("What conditions affect the {entity}?", ["anatomy"])
        ]
        
        for i in range(count):
            template, entity_types = random.choice(templates)
            entity_type = random.choice(entity_types)
            
            if entity_type == "drug":
                entity = random.choice(self.drugs)
                expected_answers = [f"{entity} side effects", "nausea", "headache", "dizziness"]
            elif entity_type == "disease":
                entity = random.choice(self.diseases)
                if "treat" in template:
                    expected_answers = [random.choice(self.drugs), "medication", "treatment"]
                else:
                    expected_answers = [random.choice(self.symptoms), "symptom", "sign"]
            elif entity_type == "symptom":
                entity = random.choice(self.symptoms)
                expected_answers = [random.choice(self.diseases), "infection", "inflammation"]
            else:  # anatomy
                entity = random.choice(self.anatomy)
                expected_answers = [random.choice(self.diseases), "disorder", "condition"]
            
            query = template.format(entity=entity)
            
            queries.append({
                "query": query,
                "intent": "enumeration",
                "complexity": random.choice(["moderate", "complex"]),
                "entity_type": entity_type,
                "primary_entity": entity,
                "expected_answers": expected_answers,
                "query_type": "enumeration",
                "difficulty": "intermediate" if len(expected_answers) <= 3 else "advanced"
            })
        
        return queries
    
    def generate_causal_queries(self, count: int = 80) -> List[Dict[str, Any]]:
        """Generate causal queries (Why does X cause Y? What leads to Z?)"""
        queries = []
        
        templates = [
            ("Why does {entity1} cause {entity2}?", ["drug", "symptom"], ["disease", "symptom"]),
            ("How does {entity1} lead to {entity2}?", ["disease", "drug"], ["symptom", "disease"]),
            ("What causes {entity1} to affect {entity2}?", ["drug", "disease"], ["anatomy", "function"]),
            ("Why is {entity1} associated with {entity2}?", ["disease", "drug"], ["symptom", "condition"]),
            ("How does {entity1} result in {entity2}?", ["condition", "medication"], ["outcome", "effect"]),
            ("What mechanism explains {entity1} causing {entity2}?", ["drug", "pathology"], ["effect", "symptom"])
        ]
        
        for i in range(count):
            template, entity1_types, entity2_types = random.choice(templates)
            entity1_type = random.choice(entity1_types)
            entity2_type = random.choice(entity2_types)
            
            if entity1_type == "drug":
                entity1 = random.choice(self.drugs)
            elif entity1_type == "disease":
                entity1 = random.choice(self.diseases)
            else:
                entity1 = random.choice(self.symptoms)
            
            if entity2_type == "symptom":
                entity2 = random.choice(self.symptoms)
            elif entity2_type == "disease":
                entity2 = random.choice(self.diseases)
            else:
                entity2 = random.choice(self.anatomy)
            
            query = template.format(entity1=entity1, entity2=entity2)
            expected_answers = [f"{entity1} mechanism", f"{entity2} pathway", "causal relationship"]
            
            queries.append({
                "query": query,
                "intent": "causal",
                "complexity": random.choice(["complex", "expert"]),
                "entity_type": "multi",
                "primary_entity": entity1,
                "secondary_entity": entity2,
                "expected_answers": expected_answers,
                "query_type": "causal",
                "difficulty": "advanced"
            })
        
        return queries
    
    def generate_comparative_queries(self, count: int = 60) -> List[Dict[str, Any]]:
        """Generate comparative queries (X vs Y, Which is better?)"""
        queries = []
        
        templates = [
            ("What is the difference between {entity1} and {entity2}?", ["drug", "drug"], ["disease", "disease"]),
            ("Which is more effective: {entity1} or {entity2}?", ["drug", "drug"], ["treatment", "treatment"]),
            ("How do {entity1} and {entity2} compare?", ["medication", "medication"], ["condition", "condition"]),
            ("What are the advantages of {entity1} over {entity2}?", ["drug", "drug"], ["therapy", "therapy"]),
            ("When should {entity1} be used instead of {entity2}?", ["medication", "medication"], ["treatment", "treatment"])
        ]
        
        for i in range(count):
            template, entity1_types, entity2_types = random.choice(templates)
            
            # Ensure different entities for comparison
            if entity1_types[0] == "drug":
                entities = random.sample(self.drugs, 2)
                entity1, entity2 = entities[0], entities[1]
            else:
                entities = random.sample(self.diseases, 2)
                entity1, entity2 = entities[0], entities[1]
            
            query = template.format(entity1=entity1, entity2=entity2)
            expected_answers = [entity1, entity2, "comparison", "difference"]
            
            queries.append({
                "query": query,
                "intent": "comparative",
                "complexity": "expert",
                "entity_type": "multi",
                "primary_entity": entity1,
                "secondary_entity": entity2,
                "expected_answers": expected_answers,
                "query_type": "comparative",
                "difficulty": "expert"
            })
        
        return queries
    
    def generate_edge_case_queries(self, count: int = 60) -> List[Dict[str, Any]]:
        """Generate edge case queries (rare conditions, complex interactions)"""
        queries = []
        
        # Complex multi-hop queries
        complex_templates = [
            "What are the contraindications of {drug} in patients with {disease} and {symptom}?",
            "How does {drug} interact with {drug2} in the treatment of {disease}?",
            "What are the long-term effects of {drug} on {anatomy} function in {disease} patients?",
            "Which biomarkers predict response to {drug} in {disease} with comorbid {disease2}?",
            "How does {disease} progression affect {drug} metabolism through {anatomy}?"
        ]
        
        for i in range(count):
            template = random.choice(complex_templates)
            
            # Fill template with random entities
            drug = random.choice(self.drugs)
            drug2 = random.choice([d for d in self.drugs if d != drug])
            disease = random.choice(self.diseases)
            disease2 = random.choice([d for d in self.diseases if d != disease])
            symptom = random.choice(self.symptoms)
            anatomy = random.choice(self.anatomy)
            
            query = template.format(
                drug=drug, drug2=drug2, disease=disease, disease2=disease2,
                symptom=symptom, anatomy=anatomy
            )
            
            expected_answers = [drug, disease, "interaction", "contraindication", "effect"]
            
            queries.append({
                "query": query,
                "intent": "procedural",
                "complexity": "expert",
                "entity_type": "multi",
                "primary_entity": drug,
                "expected_answers": expected_answers,
                "query_type": "edge_case",
                "difficulty": "expert"
            })
        
        return queries
    
    def generate_negation_queries(self, count: int = 40) -> List[Dict[str, Any]]:
        """Generate negation queries (What doesn't X do? Which drugs don't...)"""
        queries = []
        
        templates = [
            "What conditions does {drug} NOT treat?",
            "Which drugs should NOT be used for {disease}?",
            "What are the situations where {drug} is NOT recommended?",
            "Which patients should NOT take {drug}?",
            "What symptoms are NOT associated with {disease}?"
        ]
        
        for i in range(count):
            template = random.choice(templates)
            
            if "drug" in template:
                entity = random.choice(self.drugs)
                expected_answers = ["contraindication", "not indicated", "avoid"]
            else:
                entity = random.choice(self.diseases)
                expected_answers = ["not associated", "unrelated", "different condition"]
            
            query = template.format(drug=entity, disease=entity)
            
            queries.append({
                "query": query,
                "intent": "enumeration",
                "complexity": "complex",
                "entity_type": "negation",
                "primary_entity": entity,
                "expected_answers": expected_answers,
                "query_type": "negation",
                "difficulty": "advanced"
            })
        
        return queries
    
    def add_query_metadata(self, queries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Add comprehensive metadata to queries"""
        
        for i, query in enumerate(queries):
            query.update({
                "id": f"eval_{i+1:04d}",
                "source": "comprehensive_eval_builder",
                "version": "1.0",
                "created_date": "2025-08-27",
                
                # Evaluation metadata
                "evaluation_focus": self._determine_evaluation_focus(query),
                "expected_retrieval_mode": self._determine_expected_mode(query),
                "evaluation_criteria": self._determine_criteria(query),
                
                # Difficulty scoring
                "difficulty_score": self._calculate_difficulty_score(query),
                "complexity_factors": self._identify_complexity_factors(query),
                
                # Answer validation
                "answer_validation_type": self._determine_validation_type(query),
                "minimum_expected_results": self._determine_min_results(query)
            })
        
        return queries
    
    def _determine_evaluation_focus(self, query: Dict[str, Any]) -> List[str]:
        """Determine what aspects this query evaluates"""
        focus = []
        
        if query["query_type"] == "factoid":
            focus.extend(["precision", "accuracy", "entity_recognition"])
        elif query["query_type"] == "enumeration":
            focus.extend(["recall", "completeness", "ranking"])
        elif query["query_type"] == "causal":
            focus.extend(["reasoning", "multi_hop", "causality"])
        elif query["query_type"] == "comparative":
            focus.extend(["comparison", "ranking", "differentiation"])
        elif query["query_type"] == "edge_case":
            focus.extend(["robustness", "complex_reasoning", "multi_entity"])
        elif query["query_type"] == "negation":
            focus.extend(["negation_handling", "precision", "false_positive_control"])
        
        return focus
    
    def _determine_expected_mode(self, query: Dict[str, Any]) -> str:
        """Determine which retrieval mode should perform best"""
        if query["complexity"] in ["simple", "moderate"]:
            return "dense"
        elif query["query_type"] in ["causal", "edge_case"]:
            return "kg"
        else:
            return "hybrid"
    
    def _determine_criteria(self, query: Dict[str, Any]) -> Dict[str, float]:
        """Determine evaluation criteria weights"""
        if query["query_type"] == "factoid":
            return {"precision": 0.4, "accuracy": 0.4, "speed": 0.2}
        elif query["query_type"] == "enumeration":
            return {"recall": 0.5, "precision": 0.3, "ranking": 0.2}
        elif query["query_type"] == "causal":
            return {"reasoning": 0.5, "accuracy": 0.3, "completeness": 0.2}
        else:
            return {"precision": 0.3, "recall": 0.3, "reasoning": 0.4}
    
    def _calculate_difficulty_score(self, query: Dict[str, Any]) -> float:
        """Calculate difficulty score (0-1)"""
        base_score = {
            "simple": 0.2,
            "moderate": 0.4,
            "complex": 0.7,
            "expert": 0.9
        }.get(query["complexity"], 0.5)
        
        # Adjust based on query type
        type_adjustment = {
            "factoid": 0.0,
            "enumeration": 0.1,
            "causal": 0.2,
            "comparative": 0.15,
            "edge_case": 0.3,
            "negation": 0.25
        }.get(query["query_type"], 0.0)
        
        return min(1.0, base_score + type_adjustment)
    
    def _identify_complexity_factors(self, query: Dict[str, Any]) -> List[str]:
        """Identify what makes this query complex"""
        factors = []
        
        if query.get("secondary_entity"):
            factors.append("multi_entity")
        if query["query_type"] == "causal":
            factors.append("causal_reasoning")
        if query["complexity"] == "expert":
            factors.append("domain_expertise_required")
        if "NOT" in query["query"].upper():
            factors.append("negation")
        if len(query["query"].split()) > 10:
            factors.append("long_query")
        
        return factors
    
    def _determine_validation_type(self, query: Dict[str, Any]) -> str:
        """Determine how to validate answers"""
        if query["query_type"] in ["factoid", "causal"]:
            return "exact_match_or_semantic"
        elif query["query_type"] == "enumeration":
            return "list_overlap"
        elif query["query_type"] == "comparative":
            return "comparative_analysis"
        else:
            return "expert_judgment"
    
    def _determine_min_results(self, query: Dict[str, Any]) -> int:
        """Determine minimum expected results"""
        if query["query_type"] == "factoid":
            return 1
        elif query["query_type"] == "enumeration":
            return 3
        elif query["query_type"] in ["causal", "comparative"]:
            return 2
        else:
            return 1
    
    def build_dataset(self, total_size: int = 500) -> List[Dict[str, Any]]:
        """Build comprehensive evaluation dataset"""
        logger.info(f"Building comprehensive evaluation dataset with {total_size} queries...")
        
        # Distribute queries across types for critical coverage
        distribution = {
            "factoid": int(total_size * 0.25),      # 25% - Basic facts
            "enumeration": int(total_size * 0.25),  # 25% - Lists/collections
            "causal": int(total_size * 0.20),       # 20% - Reasoning
            "comparative": int(total_size * 0.15),  # 15% - Comparisons
            "edge_case": int(total_size * 0.10),    # 10% - Complex cases
            "negation": int(total_size * 0.05)      # 5% - Negation handling
        }
        
        all_queries = []
        
        # Generate each type
        all_queries.extend(self.generate_factoid_queries(distribution["factoid"]))
        all_queries.extend(self.generate_enumeration_queries(distribution["enumeration"]))
        all_queries.extend(self.generate_causal_queries(distribution["causal"]))
        all_queries.extend(self.generate_comparative_queries(distribution["comparative"]))
        all_queries.extend(self.generate_edge_case_queries(distribution["edge_case"]))
        all_queries.extend(self.generate_negation_queries(distribution["negation"]))
        
        # Add metadata
        all_queries = self.add_query_metadata(all_queries)
        
        # Shuffle for random distribution
        random.shuffle(all_queries)
        
        # Trim to exact size
        all_queries = all_queries[:total_size]
        
        logger.info(f"Generated {len(all_queries)} queries with distribution:")
        query_types = {}
        complexities = {}
        difficulties = {}
        
        for query in all_queries:
            query_types[query["query_type"]] = query_types.get(query["query_type"], 0) + 1
            complexities[query["complexity"]] = complexities.get(query["complexity"], 0) + 1
            difficulties[query["difficulty"]] = difficulties.get(query["difficulty"], 0) + 1
        
        for qtype, count in query_types.items():
            logger.info(f"  {qtype}: {count} ({count/len(all_queries)*100:.1f}%)")
        
        logger.info(f"Complexity distribution:")
        for complexity, count in complexities.items():
            logger.info(f"  {complexity}: {count} ({count/len(all_queries)*100:.1f}%)")
        
        logger.info(f"Difficulty distribution:")
        for difficulty, count in difficulties.items():
            logger.info(f"  {difficulty}: {count} ({count/len(all_queries)*100:.1f}%)")
        
        return all_queries
    
    def save_dataset(self, dataset: List[Dict[str, Any]], output_path: str):
        """Save dataset to JSONL file"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            for query in dataset:
                f.write(json.dumps(query) + '\n')
        
        logger.info(f"Saved {len(dataset)} queries to {output_file}")
        
        # Also save summary statistics
        summary_file = output_file.with_suffix('.summary.json')
        summary = self._generate_dataset_summary(dataset)
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Saved dataset summary to {summary_file}")
    
    def _generate_dataset_summary(self, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate dataset summary statistics"""
        summary = {
            "total_queries": len(dataset),
            "creation_date": "2025-08-27",
            "version": "1.0",
            "description": "Comprehensive biomedical QA evaluation dataset with critical distribution coverage",
            
            "distributions": {
                "query_types": {},
                "complexities": {},
                "difficulties": {},
                "intents": {},
                "entity_types": {}
            },
            
            "statistics": {
                "avg_query_length": sum(len(q["query"].split()) for q in dataset) / len(dataset),
                "avg_expected_answers": sum(len(q["expected_answers"]) for q in dataset) / len(dataset),
                "avg_difficulty_score": sum(q["difficulty_score"] for q in dataset) / len(dataset)
            },
            
            "coverage": {
                "unique_drugs": len(set(q.get("primary_entity", "") for q in dataset if q.get("entity_type") == "drug")),
                "unique_diseases": len(set(q.get("primary_entity", "") for q in dataset if q.get("entity_type") == "disease")),
                "multi_entity_queries": sum(1 for q in dataset if q.get("secondary_entity")),
                "negation_queries": sum(1 for q in dataset if q.get("query_type") == "negation")
            }
        }
        
        # Calculate distributions
        for field in ["query_type", "complexity", "difficulty", "intent", "entity_type"]:
            dist = {}
            for query in dataset:
                value = query.get(field, "unknown")
                dist[value] = dist.get(value, 0) + 1
            summary["distributions"][f"{field}s"] = dist
        
        return summary

def main():
    """Main function to build comprehensive evaluation dataset"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Build comprehensive evaluation dataset")
    parser.add_argument("--size", type=int, default=500, help="Total dataset size")
    parser.add_argument("--output", default="data/processed/comprehensive_eval_dataset.jsonl", 
                       help="Output file path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    
    # Build dataset
    builder = ComprehensiveEvalDatasetBuilder()
    dataset = builder.build_dataset(args.size)
    
    # Save dataset
    builder.save_dataset(dataset, args.output)
    
    logger.info("Comprehensive evaluation dataset build complete!")

if __name__ == "__main__":
    main()
