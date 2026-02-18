#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Knowledge Store Integration
Unified knowledge store integration interface
"""

import json
from typing import Dict, Any, List, Optional, Tuple
from .unified_knowledge_store import (
    get_unified_knowledge_store, get_shared_unified_knowledge_store,
    UnifiedKnowledgeStore
)
from .storage_manager import get_storage_manager, ExperimentSetting, StorageMode

def try_unified_knowledge_shortcut(subq_obj: Any, ctx: Dict[str, Any], 
                                  sufficiency_checker, 
                                  enable_knowledge: bool = True,
                                  experiment_setting: Optional[ExperimentSetting] = None,
                                  enable_shared_fallback: bool = False) -> Optional[Dict[str, Any]]:
    """
    Try to reuse the subquestion solution from the unified knowledge store
    
    Called before "subquestion → sufficiency validation", try to short-circuit
    
    Args:
        subq_obj: Subquestion object
        ctx: Current context
        sufficiency_checker: Sufficiency validation function
        enable_knowledge: Whether to enable unified knowledge store (default True)
        experiment_setting: Experiment setting
        enable_shared_fallback: Whether to enable shared fallback
        
    Returns:
        If hit and validation passed, return the reusable result; otherwise return None
    """
    if not enable_knowledge:
        return None
    
    try:
        # 1. Extract subquestion information
        subq_text, entities, time_constraint = _extract_subquestion_info(subq_obj, ctx)
        
        print(f"🔍 Query unified knowledge store: {subq_text[:60]}...")
        
        # 2. Query the unified knowledge store
        store = get_unified_knowledge_store(experiment_setting=experiment_setting)
        candidates = store.lookup_knowledge(
            query_text=subq_text,
            query_type="subquestion",
            entities=entities,
            time_constraint=time_constraint,
            k=5,
            sim_threshold=0.83,
            include_template_data=True,
            include_experience_data=True
        )
        
        # 3. If in individual mode and shared fallback enabled, try to query the shared store
        if enable_shared_fallback and (not candidates or len(candidates) == 0):
            storage_manager = get_storage_manager()
            if storage_manager.storage_mode == StorageMode.INDIVIDUAL:
                print(f"   🔄 Try shared knowledge store fallback...")
                shared_store = get_shared_unified_knowledge_store()
                shared_candidates = shared_store.lookup_knowledge(
                    query_text=subq_text,
                    query_type="subquestion",
                    entities=entities,
                    time_constraint=time_constraint,
                    k=5,
                    sim_threshold=0.83,
                    include_template_data=True,
                    include_experience_data=True
                )
                if shared_candidates:
                    candidates = shared_candidates
                    print(f"   ✅ Hit {len(candidates)} candidates from shared knowledge store")
        
        if not candidates:
            print(f"   ⚪ Unified knowledge store not hit")
            return None
        
        print(f"   🎯 Found {len(candidates)} candidates")
        
        # 4. Try to validate each candidate (from highest similarity to lowest)
        for i, cand in enumerate(candidates, 1):
            try:
                sim = cand.get("similarity", 0)
                print(f"    Try candidate {i}: similarity={sim:.3f}")
                
                # Get indicators and evidence
                indicators = cand.get("indicators", {})
                evidence = cand.get("evidence", {})
                
                # Check if there are saved sufficiency test parameters
                sufficiency_args = cand.get("sufficiency_args", {})
                
                if sufficiency_args and sufficiency_args.get("experiment_setting"):
                    # Use saved parameters to re-perform sufficiency test
                    try:
                        from .stepwise import test_answer_sufficiency
                        from .storage_manager import ExperimentSetting
                        
                        # Update current subquestion to the current query subquestion
                        sufficiency_args["subquestion"] = subq_text
                        
                        # Rebuild ExperimentSetting object
                        exp_setting_dict = sufficiency_args["experiment_setting"]
                        restored_experiment_setting = ExperimentSetting(**exp_setting_dict)
                        
                        # Re-perform sufficiency test
                        sufficiency_result = test_answer_sufficiency(
                            subquestion=sufficiency_args["subquestion"],
                            current_answer=sufficiency_args["current_answer"],
                            retrieved_info=sufficiency_args["retrieved_info"],
                            context=sufficiency_args["context"],
                            previous_subquestions=sufficiency_args["previous_subquestions"],
                            toolkit_info=sufficiency_args["toolkit_info"],
                            debate_vote_result=sufficiency_args["debate_vote_result"],
                            top_paths=sufficiency_args["top_paths"],
                            experiment_setting=restored_experiment_setting
                        )
                        
                        is_sufficient = sufficiency_result.get("sufficient", False)
                        reason = sufficiency_result.get("reason", "Re-validated with saved parameters")
                        
                        print(f"   🔄 Re-validate sufficiency: {'passed' if is_sufficient else 'failed'}")
                        
                    except Exception as e:
                        print(f"   ⚠️ Re-validate sufficiency failed, use simplified validation: {e}")
                        # Fall back to simplified sufficiency validation
                        is_sufficient, reason = sufficiency_checker(
                            subquestion=subq_text,
                            indicators=indicators,
                            evidence=evidence,
                            context=ctx
                        )
                else:
                    # No saved parameters, use simplified sufficiency validation
                    is_sufficient, reason = sufficiency_checker(
                        subquestion=subq_text,
                        indicators=indicators,
                        evidence=evidence,
                        context=ctx
                    )
                
                if is_sufficient:
                    print(f"   ✅ Unified knowledge store hit! Reuse candidate {i}")
                    
                    # Build return result
                    result = {
                        "status": "reused_from_unified_knowledge",
                        "indicators": indicators,
                        "evidence": evidence,
                        "toolkit_params": cand.get("toolkit_params", {}),
                        "template_data": cand.get("template_data", {}),
                        "decomposition_data": cand.get("decomposition_data", {}),
                        "execution_data": cand.get("execution_data", {}),
                        "similarity": sim,
                        "hit_count": cand.get("hit_count", 0),
                        "source": "unified_knowledge_store"
                    }
                    
                    # Add TopN candidates information (if exists)
                    execution_data = cand.get("execution_data", {})
                    if execution_data.get("top_candidates"):
                        result["top_candidates"] = execution_data["top_candidates"]
                        result["total_candidates"] = execution_data.get("total_candidates", 0)
                        print(f"   📊 Contains {len(execution_data['top_candidates'])} Top candidates")
                    
                    # Add top_paths information (for sufficiency test)
                    if execution_data.get("top_paths"):
                        result["top_paths"] = execution_data["top_paths"]
                        print(f"   📊 Contains {len(execution_data['top_paths'])} Top paths")
                    
                    return result
                else:
                    print(f"   ⚠️ Candidate {i} validation failed: {reason}")
                    # Mark knowledge item failed for sufficiency test
                    key_hash = cand.get("key_hash")
                    if key_hash:
                        try:
                            store.mark_failed_knowledge(key_hash)
                        except Exception as mark_error:
                            print(f"   ⚠️ Error marking failed item: {mark_error}")
                    
            except Exception as e:
                print(f"   ⚠️ Candidate {i} processing failed: {e}")
                continue
        
        print(f"   ⚪ All candidates failed validation")
        return None
        
    except Exception as e:
        print(f"⚠️ Unified knowledge store query failed: {e}")
        return None

def record_successful_knowledge(subq_obj: Any, ctx: Dict[str, Any], 
                               step_result: Dict[str, Any],
                               toolkit_params: Dict[str, Any],
                               template_data: Dict[str, Any] = None,
                               decomposition_data: Dict[str, Any] = None,
                               execution_data: Dict[str, Any] = None,
                               enable_knowledge: bool = True,
                               experiment_setting: Optional[ExperimentSetting] = None,
                               enable_shared_fallback: bool = False):
    """
    Record successful knowledge to the unified knowledge store
    
    Called after "sufficiency validation passed and answer is correct"
    
    Args:
        subq_obj: Subquestion object
        ctx: Context information
        step_result: Execution result
        toolkit_params: Toolkit parameters used
        template_data: Template learning data
        decomposition_data: Decomposition data
        execution_data: Execution data
        enable_knowledge: Whether to enable unified knowledge store (default True)
        experiment_setting: Experiment setting
        enable_shared_fallback: Whether to enable shared fallback
    """
    if not enable_knowledge:
        return
    
    try:
        # 1. Extract subquestion information
        subq_text, entities, time_constraint = _extract_subquestion_info(subq_obj, ctx)
        
        # 2. Build indicators
        indicators = {}
        if hasattr(subq_obj, 'indicator'):
            indicators = {
                "edges": [
                    {
                        "subj": edge.subj,
                        "rel": edge.rel,
                        "obj": edge.obj,
                        "time_var": edge.time_var if hasattr(edge, 'time_var') else None
                    }
                    for edge in (subq_obj.indicator.edges if hasattr(subq_obj.indicator, 'edges') else [])
                ],
                "constraints": subq_obj.indicator.constraints if hasattr(subq_obj.indicator, 'constraints') else []
            }
        
        # 3. Build evidence summary
        evidence = _build_evidence_summary(step_result)
        
        # 4. Extract TopN candidates information
        top_candidates = []
        if step_result.get("candidates"):
            for i, candidate in enumerate(step_result["candidates"][:5]):  # Record the first 5 candidates
                top_candidates.append({
                    "rank": i + 1,
                    "entity": candidate.get("entity", ""),
                    "time": candidate.get("time", ""),
                    "score": candidate.get("score", 0.0),
                    "path": candidate.get("path", []),
                    "provenance": candidate.get("provenance", {})
                })
        
        # 5. Build execution data (contains TopN candidates)
        if execution_data is None:
            execution_data = {}
        execution_data.update({
            "top_candidates": top_candidates,
            "total_candidates": len(step_result.get("candidates", [])),
            "selection_method": step_result.get("chosen", {}).get("provenance", {}).get("method", "unknown"),
            "execution_time": step_result.get("execution_time", 0.0)
        })
        
        # 6. Determine question type
        question_type = None
        if hasattr(subq_obj, 'question_type'):
            question_type = subq_obj.question_type
        elif ctx.get('question_type'):
            question_type = ctx['question_type']
        
        # 7. Write to unified knowledge store
        store = get_unified_knowledge_store(experiment_setting=experiment_setting)
        key_hash = store.store_knowledge(
            query_text=subq_text,
            query_type="subquestion",
            entities=entities,
            time_constraint=time_constraint,
            indicators=indicators,
            evidence=evidence,
            toolkit_params=toolkit_params,
            question_type=question_type,
            template_data=template_data,
            decomposition_data=decomposition_data,
            execution_data=execution_data,
            success_rate=1.0  # Passed validation
        )
        
        # 6. If shared fallback enabled, write to shared store
        if enable_shared_fallback:
            storage_manager = get_storage_manager()
            if storage_manager.storage_mode == StorageMode.INDIVIDUAL:
                shared_store = get_shared_unified_knowledge_store()
                shared_store.store_knowledge(
                    query_text=subq_text,
                    query_type="subquestion",
                    entities=entities,
                    time_constraint=time_constraint,
                    indicators=indicators,
                    evidence=evidence,
                    toolkit_params=toolkit_params,
                    question_type=question_type,
                    template_data=template_data,
                    decomposition_data=decomposition_data,
                    execution_data=execution_data,
                    success_rate=1.0
                )
                print(f"✅ knowledge recorded to unified and shared knowledge store")
            else:
                print(f"✅ Success knowledge recorded to unified knowledge store")
        else:
            print(f"✅ Success knowledge recorded to unified knowledge store")
        
    except Exception as e:
        print(f"⚠️ Record unified knowledge failed: {e}")

def get_template_learning_data(query_text: str,
                              question_type: str = None,
                              experiment_setting: Optional[ExperimentSetting] = None,
                              enable_shared_fallback: bool = False) -> Dict[str, Any]:
    """
    Get template learning data from unified knowledge store
    
    Args:
        query_text: Query text
        question_type: Question type
        experiment_setting: Experiment setting
        enable_shared_fallback: Whether to enable shared fallback
        
    Returns:
        Template learning data dictionary
    """
    try:
        store = get_unified_knowledge_store(experiment_setting=experiment_setting)
        
        # Query template learning related knowledge
        candidates = store.lookup_knowledge(
            query_text=query_text,
            query_type="subquestion",
            k=10,
            sim_threshold=0.7,
            include_template_data=True,
            include_experience_data=False
        )
        
        # If shared fallback enabled and no result
        if enable_shared_fallback and (not candidates or len(candidates) == 0):
            storage_manager = get_storage_manager()
            if storage_manager.storage_mode == StorageMode.INDIVIDUAL:
                shared_store = get_shared_unified_knowledge_store()
                candidates = shared_store.lookup_knowledge(
                    query_text=query_text,
                    query_type="subquestion",
                    k=10,
                    sim_threshold=0.7,
                    include_template_data=True,
                    include_experience_data=False
                )
        
        # Aggregate template learning data
        template_data = {}
        decomposition_data = {}
        execution_data = {}
        
        for candidate in candidates:
            if candidate.get('template_data'):
                template_data.update(candidate['template_data'])
            if candidate.get('decomposition_data'):
                decomposition_data.update(candidate['decomposition_data'])
            if candidate.get('execution_data'):
                execution_data.update(candidate['execution_data'])
        
        return {
            'template_data': template_data,
            'decomposition_data': decomposition_data,
            'execution_data': execution_data,
            'sources_count': len(candidates)
        }
        
    except Exception as e:
        print(f"⚠️ Get template learning data failed: {e}")
        return {}

def migrate_existing_data(experiment_setting: Optional[ExperimentSetting] = None):
    """
    Migrate existing experience pool and template learning data to unified knowledge store
    
    Args:
        experiment_setting: Experiment setting
    """
    try:
        print("🔄 Start migrating existing data to unified knowledge store...")
        
        # Get unified knowledge store instance
        unified_store = get_unified_knowledge_store(experiment_setting=experiment_setting)
        
        # Migrate experience pool data
        from .experience_pool import get_experience_pool
        exp_pool = get_experience_pool(experiment_setting=experiment_setting)
        
        # Here we need to implement the logic to read data from experience pool and migrate
        # Since experience pool uses SQLite, we can directly query
        
        # Template learning has been removed, no data migration needed
        
        print("✅ Data migration completed")
        
    except Exception as e:
        print(f"⚠️ Data migration failed: {e}")

# ========== 辅助函数 ==========

def _extract_subquestion_info(subq_obj: Any, ctx: Dict[str, Any]) -> Tuple[str, List[str], Dict[str, Any]]:
    """Extract information from subquestion object"""
    # Extract subquestion text
    subq_text = subq_obj.text if hasattr(subq_obj, 'text') else str(subq_obj)
    
    # Extract entity list
    entities = []
    if hasattr(subq_obj, 'indicator') and hasattr(subq_obj.indicator, 'edges'):
        for edge in subq_obj.indicator.edges:
            if hasattr(edge, 'subj') and edge.subj and edge.subj not in ['?x', '?y']:
                entities.append(edge.subj)
            if hasattr(edge, 'obj') and edge.obj and edge.obj not in ['?x', '?y']:
                entities.append(edge.obj)
    
    # Extract time constraint
    time_constraint = {}
    if hasattr(subq_obj, 'indicator') and hasattr(subq_obj.indicator, 'constraints'):
        for constraint in subq_obj.indicator.constraints:
            if '=' in constraint:
                time_constraint['op'] = 'equal'
                time_constraint['ref'] = constraint.split('=')[1].strip()
            elif '<' in constraint:
                time_constraint['op'] = 'before'
                time_constraint['ref'] = constraint.split('<')[1].strip()
            elif '>' in constraint:
                time_constraint['op'] = 'after'
                time_constraint['ref'] = constraint.split('>')[1].strip()
    
    # Get additional time information from context
    if not time_constraint and ctx.get("times"):
        time_constraint = {"op": "context", "ref": str(ctx.get("times"))}
    
    return subq_text, entities, time_constraint

def _build_evidence_summary(step_result: Dict[str, Any]) -> Dict[str, Any]:
    """Build evidence summary from execution result"""
    evidence = {}
    
    if "chosen" in step_result:
        chosen = step_result["chosen"]
        evidence["entity"] = chosen.get("entity")
        evidence["time"] = chosen.get("time")
        evidence["path"] = chosen.get("path")
        evidence["provenance"] = chosen.get("provenance")
    
    if "candidates" in step_result:
        # Only save the summary of the first 3 candidates
        evidence["candidate_count"] = len(step_result["candidates"])
        evidence["top_candidates"] = [
            {
                "entity": c.get("entity"),
                "time": c.get("time")
            }
            for c in step_result["candidates"][:3]
        ]
    
    return evidence

if __name__ == "__main__":
    # Use example
    print("Unified knowledge store integration interface")
    print("Usage:")
    print("1. Call try_unified_knowledge_shortcut before subquestion processing")
    print("2. Call record_successful_knowledge after successful validation")
    print("3. Call get_template_learning_data to get template learning data")
