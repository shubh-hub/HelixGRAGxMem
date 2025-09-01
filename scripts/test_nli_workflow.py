#!/usr/bin/env python3
"""
NLI Workflow Validation Script
==============================

Tests the 12-step NLI-aligned MAS workflow to ensure it meets user expectations:
1. Query Understanding
2. Intent Detection  
3. Query Transformation
4. Evidence Resource Identification
5. Entity/Phrase Extraction
6. Canonical Entity Resolution
7. Seed Node & Constraint Identification
8. Retrieval Engine (confidence-guided)
9. Evidence Aggregation & Pruning
10. LLM-driven Reasoning
11. Answer Generation, Review & Explanation
12. Summarized Logging

Validates LLM-driven intelligence and stdio MCP integration.
"""

import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.mas.nli_orchestrator import NLISession
from src.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NLIWorkflowValidator:
    """Comprehensive NLI workflow validator"""
    
    def __init__(self):
        self.test_queries = [
            {
                "query": "What causes Alzheimer's disease?",
                "expected_intent": "causal",
                "expected_mode": "hybrid",
                "description": "Causal reasoning query requiring both KG and literature evidence"
            },
            {
                "query": "List all FDA-approved treatments for diabetes",
                "expected_intent": "enumeration", 
                "expected_mode": "hybrid",
                "description": "Enumeration query requiring comprehensive evidence gathering"
            },
            {
                "query": "How does metformin work?",
                "expected_intent": "factoid",
                "expected_mode": "kg_only",
                "description": "Mechanism query best served by structured knowledge"
            },
            {
                "query": "Compare efficacy of statins versus fibrates for cholesterol management",
                "expected_intent": "comparative",
                "expected_mode": "dense_only", 
                "description": "Comparative analysis requiring literature evidence"
            }
        ]
        
        self.validation_results = {}
    
    async def run_validation(self) -> Dict[str, Any]:
        """Run comprehensive NLI workflow validation"""
        logger.info("🚀 Starting NLI Workflow Validation")
        
        # Test 1: Workflow execution for each query type
        logger.info("📋 Test 1: Workflow Execution Validation")
        execution_results = await self._test_workflow_execution()
        self.validation_results["workflow_execution"] = execution_results
        
        # Test 2: 12-step compliance validation
        logger.info("📋 Test 2: 12-Step Compliance Validation")
        compliance_results = await self._test_step_compliance()
        self.validation_results["step_compliance"] = compliance_results
        
        # Test 3: LLM intelligence validation
        logger.info("📋 Test 3: LLM Intelligence Validation")
        intelligence_results = await self._test_llm_intelligence()
        self.validation_results["llm_intelligence"] = intelligence_results
        
        # Test 4: MCP integration validation
        logger.info("📋 Test 4: MCP Integration Validation")
        mcp_results = await self._test_mcp_integration()
        self.validation_results["mcp_integration"] = mcp_results
        
        # Generate summary
        summary = self._generate_validation_summary()
        self.validation_results["summary"] = summary
        
        return self.validation_results
    
    async def _test_workflow_execution(self) -> Dict[str, Any]:
        """Test workflow execution for different query types"""
        results = {}
        
        async with NLISession() as orchestrator:
            for i, test_case in enumerate(self.test_queries):
                query = test_case["query"]
                logger.info(f"  Testing query {i+1}: {query}")
                
                try:
                    start_time = time.time()
                    result = await orchestrator.execute_nli_workflow(
                        query=query,
                        persona="doctor"
                    )
                    execution_time = time.time() - start_time
                    
                    # Validate execution results
                    test_result = {
                        "status": "success" if "error" not in result else "error",
                        "execution_time": execution_time,
                        "has_answer": bool(result.get("final_answer") or result.get("answer")),
                        "has_explanation": bool(result.get("explanation")),
                        "has_citations": bool(result.get("citations")),
                        "confidence": result.get("confidence", 0.0),
                        "trace_id": result.get("trace_id"),
                        "error": result.get("error")
                    }
                    
                    if test_result["status"] == "success":
                        logger.info(f"    ✅ Query {i+1}: Success (confidence: {test_result['confidence']:.2f})")
                    else:
                        logger.error(f"    ❌ Query {i+1}: Failed - {test_result['error']}")
                    
                    results[f"query_{i+1}"] = test_result
                    
                except Exception as e:
                    logger.error(f"    ❌ Query {i+1}: Exception - {e}")
                    results[f"query_{i+1}"] = {
                        "status": "exception",
                        "error": str(e)
                    }
        
        return results
    
    async def _test_step_compliance(self) -> Dict[str, Any]:
        """Test compliance with the 12-step NLI workflow"""
        results = {}
        
        # Test with a single representative query
        test_query = "What are the side effects of aspirin?"
        
        async with NLISession() as orchestrator:
            try:
                result = await orchestrator.execute_nli_workflow(
                    query=test_query,
                    persona="patient"
                )
                
                if "full_state" in result:
                    state = result["full_state"]
                    metadata = state.get("metadata", {})
                    
                    # Check each step's completion
                    step_checks = {
                        "step_1_query_understanding": "query_understanding" in metadata,
                        "step_2_intent_detection": "intent_analysis" in metadata,
                        "step_3_query_transformation": "query_transformation" in metadata,
                        "step_4_resource_identification": "resource_strategy" in metadata,
                        "step_5_entity_extraction": "entity_extraction" in metadata,
                        "step_6_canonicalization": len(state.get("entities", [])) > 0,
                        "step_7_seed_identification": "seed_strategy" in metadata,
                        "step_8_retrieval_engine": len(state.get("kg_results", [])) > 0 or len(state.get("dense_results", [])) > 0,
                        "step_9_evidence_aggregation": "evidence_aggregation" in metadata,
                        "step_10_reasoning": "reasoning" in metadata,
                        "step_11_answer_generation": bool(state.get("final_answer")),
                        "step_12_logging": "execution_summary" in metadata
                    }
                    
                    completed_steps = sum(step_checks.values())
                    total_steps = len(step_checks)
                    
                    results = {
                        "total_steps": total_steps,
                        "completed_steps": completed_steps,
                        "completion_rate": completed_steps / total_steps,
                        "step_details": step_checks,
                        "status": "pass" if completed_steps >= 10 else "fail"
                    }
                    
                    logger.info(f"  Step compliance: {completed_steps}/{total_steps} steps completed")
                    
                else:
                    results = {
                        "status": "error",
                        "error": "No full_state in result"
                    }
                    
            except Exception as e:
                results = {
                    "status": "exception",
                    "error": str(e)
                }
        
        return results
    
    async def _test_llm_intelligence(self) -> Dict[str, Any]:
        """Test LLM-driven intelligence in each step"""
        results = {}
        
        # Test query requiring complex reasoning
        complex_query = "How do ACE inhibitors reduce cardiovascular risk in diabetic patients?"
        
        async with NLISession() as orchestrator:
            try:
                result = await orchestrator.execute_nli_workflow(
                    query=complex_query,
                    persona="doctor"
                )
                
                if "full_state" in result:
                    state = result["full_state"]
                    metadata = state.get("metadata", {})
                    
                    # Check for LLM-driven intelligence indicators
                    intelligence_checks = {
                        "intent_classification": metadata.get("intent_analysis", {}).get("primary_intent") in ["causal", "factoid", "comparative", "enumeration"],
                        "query_decomposition": bool(metadata.get("query_transformation", {}).get("subquestions")),
                        "resource_strategy": bool(metadata.get("resource_strategy", {}).get("mode")),
                        "entity_recognition": len(state.get("entities", [])) > 0,
                        "reasoning_chain": bool(metadata.get("reasoning", {}).get("reasoning_chain")),
                        "answer_quality": len(state.get("final_answer", "")) > 50,
                        "explanation_generation": len(state.get("explanation", "")) > 50,
                        "confidence_assessment": isinstance(state.get("confidence"), (int, float))
                    }
                    
                    intelligence_score = sum(intelligence_checks.values()) / len(intelligence_checks)
                    
                    results = {
                        "intelligence_score": intelligence_score,
                        "intelligence_checks": intelligence_checks,
                        "status": "pass" if intelligence_score >= 0.75 else "fail",
                        "reasoning_depth": len(str(metadata.get("reasoning", {}))),
                        "answer_length": len(state.get("final_answer", "")),
                        "explanation_length": len(state.get("explanation", ""))
                    }
                    
                    logger.info(f"  LLM intelligence score: {intelligence_score:.2f}")
                    
                else:
                    results = {
                        "status": "error",
                        "error": "No full_state in result"
                    }
                    
            except Exception as e:
                results = {
                    "status": "exception",
                    "error": str(e)
                }
        
        return results
    
    async def _test_mcp_integration(self) -> Dict[str, Any]:
        """Test MCP integration and tool usage"""
        results = {}
        
        # Test query that should use multiple MCP servers
        integration_query = "What is the mechanism of action of metformin?"
        
        async with NLISession() as orchestrator:
            try:
                result = await orchestrator.execute_nli_workflow(
                    query=integration_query,
                    persona="researcher"
                )
                
                if "full_state" in result:
                    state = result["full_state"]
                    
                    # Check MCP integration indicators
                    mcp_checks = {
                        "trace_logging": bool(state.get("trace_id")),
                        "memory_storage": "execution_summary" in state.get("metadata", {}),
                        "validation_used": "answer_review" in state.get("metadata", {}),
                        "explanation_generated": bool(state.get("explanation")),
                        "evidence_retrieved": len(state.get("kg_results", [])) > 0 or len(state.get("dense_results", [])) > 0
                    }
                    
                    mcp_score = sum(mcp_checks.values()) / len(mcp_checks)
                    
                    results = {
                        "mcp_score": mcp_score,
                        "mcp_checks": mcp_checks,
                        "status": "pass" if mcp_score >= 0.8 else "fail",
                        "trace_id": state.get("trace_id"),
                        "servers_used": self._count_servers_used(state)
                    }
                    
                    logger.info(f"  MCP integration score: {mcp_score:.2f}")
                    
                else:
                    results = {
                        "status": "error", 
                        "error": "No full_state in result"
                    }
                    
            except Exception as e:
                results = {
                    "status": "exception",
                    "error": str(e)
                }
        
        return results
    
    def _count_servers_used(self, state: Dict[str, Any]) -> int:
        """Count how many MCP servers were used based on state"""
        servers_used = 0
        
        # Check for evidence of each server's usage
        if state.get("kg_results"):
            servers_used += 1  # KG server
        if state.get("dense_results"):
            servers_used += 1  # Dense server
        if state.get("metadata", {}).get("execution_summary"):
            servers_used += 1  # Memory server
        if state.get("trace_id"):
            servers_used += 1  # Trace server
        if state.get("explanation"):
            servers_used += 1  # Explain server
        if state.get("metadata", {}).get("answer_review"):
            servers_used += 1  # Validator server
            
        return servers_used
    
    def _generate_validation_summary(self) -> Dict[str, Any]:
        """Generate comprehensive validation summary"""
        
        # Count successful executions
        execution_results = self.validation_results.get("workflow_execution", {})
        successful_executions = sum(1 for result in execution_results.values() 
                                  if result.get("status") == "success")
        total_executions = len(execution_results)
        
        # Get compliance rate
        compliance = self.validation_results.get("step_compliance", {})
        compliance_rate = compliance.get("completion_rate", 0.0)
        
        # Get intelligence score
        intelligence = self.validation_results.get("llm_intelligence", {})
        intelligence_score = intelligence.get("intelligence_score", 0.0)
        
        # Get MCP integration score
        mcp_integration = self.validation_results.get("mcp_integration", {})
        mcp_score = mcp_integration.get("mcp_score", 0.0)
        
        # Calculate overall score
        overall_score = (
            (successful_executions / max(total_executions, 1)) * 0.3 +
            compliance_rate * 0.3 +
            intelligence_score * 0.25 +
            mcp_score * 0.15
        )
        
        return {
            "overall_score": overall_score,
            "overall_status": "PASS" if overall_score >= 0.8 else "FAIL",
            "execution_success_rate": successful_executions / max(total_executions, 1),
            "step_compliance_rate": compliance_rate,
            "llm_intelligence_score": intelligence_score,
            "mcp_integration_score": mcp_score,
            "total_queries_tested": total_executions,
            "successful_queries": successful_executions,
            "recommendations": self._get_recommendations(overall_score),
            "validation_type": "NLI Workflow Compliance",
            "timestamp": time.time()
        }
    
    def _get_recommendations(self, overall_score: float) -> List[str]:
        """Get recommendations based on validation results"""
        recommendations = []
        
        if overall_score < 0.6:
            recommendations.append("Critical: Major NLI workflow issues detected")
            recommendations.append("Review LLM integration and MCP server functionality")
            recommendations.append("Check 12-step workflow implementation")
        elif overall_score < 0.8:
            recommendations.append("Warning: Some NLI workflow issues detected")
            recommendations.append("Optimize LLM prompts and reasoning chains")
            recommendations.append("Improve evidence aggregation and validation")
        else:
            recommendations.append("Success: NLI workflow meets user expectations")
            recommendations.append("System ready for production biomedical queries")
            recommendations.append("Consider performance optimization and monitoring")
        
        return recommendations

async def main():
    """Main validation function"""
    validator = NLIWorkflowValidator()
    
    print("=" * 70)
    print("🔍 NLI WORKFLOW VALIDATION")
    print("Testing 12-Step LLM-Driven Biomedical MAS Workflow")
    print("=" * 70)
    
    try:
        results = await validator.run_validation()
        
        # Print detailed results
        print("\n📊 VALIDATION RESULTS:")
        print("=" * 50)
        
        # Workflow execution results
        if "workflow_execution" in results:
            print("\n🔸 WORKFLOW EXECUTION:")
            execution = results["workflow_execution"]
            for query_id, result in execution.items():
                status_icon = "✅" if result.get("status") == "success" else "❌"
                print(f"  {status_icon} {query_id}: {result.get('status', 'unknown')}")
                if result.get("confidence"):
                    print(f"    Confidence: {result['confidence']:.2f}")
                if result.get("error"):
                    print(f"    Error: {result['error']}")
        
        # Step compliance results
        if "step_compliance" in results:
            print("\n🔸 12-STEP COMPLIANCE:")
            compliance = results["step_compliance"]
            if "completed_steps" in compliance:
                print(f"  Steps completed: {compliance['completed_steps']}/{compliance['total_steps']}")
                print(f"  Compliance rate: {compliance['completion_rate']:.1%}")
                
                # Show step details
                if "step_details" in compliance:
                    for step, completed in compliance["step_details"].items():
                        icon = "✅" if completed else "❌"
                        print(f"    {icon} {step.replace('_', ' ').title()}")
        
        # LLM intelligence results
        if "llm_intelligence" in results:
            print("\n🔸 LLM INTELLIGENCE:")
            intelligence = results["llm_intelligence"]
            if "intelligence_score" in intelligence:
                print(f"  Intelligence score: {intelligence['intelligence_score']:.2f}")
                print(f"  Answer quality: {intelligence.get('answer_length', 0)} chars")
                print(f"  Reasoning depth: {intelligence.get('reasoning_depth', 0)} chars")
        
        # MCP integration results
        if "mcp_integration" in results:
            print("\n🔸 MCP INTEGRATION:")
            mcp = results["mcp_integration"]
            if "mcp_score" in mcp:
                print(f"  Integration score: {mcp['mcp_score']:.2f}")
                print(f"  Servers used: {mcp.get('servers_used', 0)}")
                print(f"  Trace ID: {mcp.get('trace_id', 'N/A')}")
        
        # Summary
        if "summary" in results:
            summary = results["summary"]
            print(f"\n🎯 SUMMARY:")
            print("=" * 30)
            print(f"Overall Score: {summary['overall_score']:.2f}")
            print(f"Overall Status: {summary['overall_status']}")
            print(f"Execution Success: {summary['execution_success_rate']:.1%}")
            print(f"Step Compliance: {summary['step_compliance_rate']:.1%}")
            print(f"LLM Intelligence: {summary['llm_intelligence_score']:.2f}")
            print(f"MCP Integration: {summary['mcp_integration_score']:.2f}")
            
            print(f"\n💡 RECOMMENDATIONS:")
            for rec in summary['recommendations']:
                print(f"  • {rec}")
        
        print("\n" + "=" * 70)
        print("✨ NLI Workflow Validation Complete!")
        print("=" * 70)
        
        # Save results
        results_file = project_root / "data" / "processed" / "nli_workflow_validation.json"
        results_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"📄 Results saved to: {results_file}")
        
        return results
        
    except Exception as e:
        logger.error(f"Validation failed with error: {e}")
        print(f"\n💥 VALIDATION FAILED: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    asyncio.run(main())
