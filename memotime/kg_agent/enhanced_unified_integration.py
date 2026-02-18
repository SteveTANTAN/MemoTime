#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Unified Knowledge Store Integration
enhanced unified knowledge store integration interface
support enhanced functionality for question decomposition, seed selection, and toolkit selection
"""

import json
from typing import Dict, Any, List, Optional, Tuple
from .unified_knowledge_store import (
    get_unified_knowledge_store, get_shared_unified_knowledge_store,
    UnifiedKnowledgeStore
)
from .storage_manager import get_storage_manager, ExperimentSetting, StorageMode

def _create_default_experiment_setting() -> ExperimentSetting:
    """create default ExperimentSetting (from TPKGConfig)"""
    try:
        # try multiple import ways
        TPKGConfig = None
        try:
            from ..config import TPKGConfig
        except (ImportError, ValueError) as e1:
            try:
                from memotime.config import TPKGConfig
            except ImportError as e2:
                # try to get TPKGConfig through sys.modules
                import sys
                if 'TPKG.config' in sys.modules:
                    TPKGConfig = sys.modules['TPKG.config'].TPKGConfig
                else:
                    raise ImportError(f"Cannot import TPKGConfig: {e1}, {e2}")
        
        # use config_name from command line first
        config_name = getattr(TPKGConfig, 'CONFIG_NAME', None)
        
        # if not provided, generate automatically
        if not config_name:
            storage_mode = getattr(TPKGConfig, 'STORAGE_MODE', 'shared')
            config_name_parts = [
                f"retry{TPKGConfig.MAX_RETRIES}",
                f"depth{TPKGConfig.MAX_DEPTH}",
                f"branch{TPKGConfig.MAX_TOTAL_BRANCHES}"
            ]
            if TPKGConfig.USE_HYBRID_RETRIEVAL:
                config_name_parts.append("hybrid")
            config_name_parts.append("unified")
            config_name_parts.append(storage_mode)
            
            if not getattr(TPKGConfig, 'ENABLE_SHARED_FALLBACK', False):
                config_name_parts.append("nofallback")
            
            # 简化LLM模型名称
            if TPKGConfig.DEFAULT_LLM_MODEL:
                model_short = TPKGConfig.DEFAULT_LLM_MODEL.replace('gpt-', 'gpt').replace('-', '').replace('o', '')
                config_name_parts.append(model_short)
            
            config_name = "_".join(config_name_parts)
        
        return ExperimentSetting(
            max_retries=TPKGConfig.MAX_RETRIES,
            max_depth=TPKGConfig.MAX_DEPTH,
            max_total_branches=TPKGConfig.MAX_TOTAL_BRANCHES,
            use_hybrid_retrieval=TPKGConfig.USE_HYBRID_RETRIEVAL,
            use_experience_pool=True,
            use_template_learning=False, 
            config_name=config_name
        )
    except Exception as e:
        import traceback
        print(f"Warning: Failed to create default ExperimentSetting: {e}")
        traceback.print_exc()
        return None

def record_question_decomposition_enhancement(
    question: str,
    question_type: str,
    decomposition_result: Dict[str, Any],
    llm_output: str,
    experiment_setting: Optional[ExperimentSetting] = None,
    enable_knowledge: bool = True
):
    """
    record question decomposition enhancement data
    
    called after final sufficiency test success, record the decomposition information of the whole question
    
    Args:
        question: original question
        question_type: question type (after_first, before_last, etc.)
        decomposition_result: decomposition result
        llm_output: LLM decomposition output
        experiment_setting: experiment setting
        enable_knowledge: whether to enable knowledge storage
    """
    if not enable_knowledge:
        return
    
    try:
        # build enhancement data
        enhancement_data = {
            "type": "question_decomposition",
            "question": question,
            "question_type": question_type,
            "subquestions": decomposition_result.get("subquestions", []),
            "indicators": decomposition_result.get("indicators", []),
            "constraints": decomposition_result.get("constraints", []),
            "time_vars": decomposition_result.get("time_vars", []),
            "llm_output": llm_output,
            "timestamp": {"created_at": "now"}  # directly use dictionary format
        }
        
        # save to unified knowledge store
        store = get_unified_knowledge_store(experiment_setting=experiment_setting)
        
        # create query text (for similarity matching)
        query_text = f"Question: {question}\nType: {question_type}"
        
        store.store_knowledge(
            query_text=query_text,
            query_type="question_decomposition",
            execution_data=enhancement_data,
            sufficiency_args=None,
            success_rate=1.0,
            question_type=question_type
        )
        
        print(f"✅ question decomposition enhancement data recorded")
        
    except Exception as e:
        print(f"❌ record question decomposition enhancement data failed: {e}")

def record_seed_selection_enhancement(
    subquestion: str,
    available_entities: List[str],
    selected_seeds: List[str],
    llm_output: str,
    question_type: str,
    experiment_setting: Optional[ExperimentSetting] = None,
    enable_knowledge: bool = True
):
    """
    record seed selection enhancement data
    
    called after subquestion sufficiency test success
    
    Args:
        subquestion: subquestion text
        available_entities: available entities list
        selected_seeds: selected seeds
        llm_output: LLM seed selection output
        question_type: question type
        experiment_setting: experiment setting
        enable_knowledge: whether to enable knowledge storage
    """
    if not enable_knowledge:
        return
    
    try:
        # build enhancement data
        enhancement_data = {
            "type": "seed_selection",
            "subquestion": subquestion,
            "entities": available_entities,
            "output": [available_entities.index(seed) + 1 for seed in selected_seeds if seed in available_entities],
            "selected_seeds": selected_seeds,
            "llm_output": llm_output,
            "question_type": question_type,
            "timestamp": {"created_at": "now"}
        }
        
        # save to unified knowledge store
        store = get_unified_knowledge_store(experiment_setting=experiment_setting)
        
        # create query text
        query_text = f"Subquestion: {subquestion}\nEntities: {', '.join(available_entities)}"
        
        store.store_knowledge(
            query_text=query_text,
            query_type="seed_selection",
            execution_data=enhancement_data,
            sufficiency_args=None,
            success_rate=1.0,
            question_type=question_type
        )
        
        print(f"✅ seed selection enhancement data recorded")
        
    except Exception as e:
        print(f"❌ record seed selection enhancement data failed: {e}")

def record_toolkit_selection_enhancement(
    subquestion: str,
    indicator: Dict[str, Any],
    seed_info: List[str],
    toolkit_name: str,
    actual_function: str,
    parameters: Dict[str, Any],
    context: Dict[str, Any],
    time_hints: Dict[str, Any],
    reasoning: str,
    llm_output: str,
    question_type: str,
    experiment_setting: Optional[ExperimentSetting] = None,
    enable_knowledge: bool = True
):
    """
    record toolkit selection enhancement data
    
    called after subquestion sufficiency test success
    
    Args:
        subquestion: subquestion text
        indicator: indicator information
        seed_info: seed information
        toolkit_name: toolkit name (LLM selected)
        actual_function: actual executed function name
        parameters: parameters
        context: context
        time_hints: time hints
        reasoning: reasoning
        llm_output: LLM toolkit selection output
        question_type: question type
        experiment_setting: experiment setting
        enable_knowledge: whether to enable knowledge storage
    """
    if not enable_knowledge:
        return
    
    try:
        # build enhancement data
        enhancement_data = {
            "type": "toolkit_selection",
            "subquestion": subquestion,
            "indicator": indicator,
            "seed_info": seed_info,
            "toolkit": toolkit_name,
            "actual_function": actual_function,
            "parameters": parameters,
            "context": context,
            "time_hints": time_hints,
            "reasoning": reasoning,
            "llm_output": llm_output,
            "question_type": question_type,
            "timestamp": {"created_at": "now"}
        }
        
        # save to unified knowledge store
        store = get_unified_knowledge_store(experiment_setting=experiment_setting)
        
        # create query text
        query_text = f"Subquestion: {subquestion}\nToolkit: {toolkit_name}"
        
        store.store_knowledge(
            query_text=query_text,
            query_type="toolkit_selection",
            execution_data=enhancement_data,
            sufficiency_args=None,
            success_rate=1.0,
            question_type=question_type
        )
        
        print(f"✅ toolkit selection enhancement data recorded")
        
    except Exception as e:
        print(f"❌ record toolkit selection enhancement data failed: {e}")

def get_question_decomposition_enhanced(
    given_question: str, 
    topk: int = 10, 
    question_type: str = None,
    similarity_threshold: float = 0.5,
    experiment_setting: Optional[ExperimentSetting] = None,
    enable_knowledge: bool = True
) -> List[Dict[str, Any]]:
    """
    get question decomposition enhancement examples
    
    Args:
        given_question: current question
        topk: return top k examples
        question_type: question type filter
        experiment_setting: experiment setting
        enable_knowledge: whether to enable knowledge storage
        
    Returns:
        enhanced examples list in specified format
        [{
            "Q": "question text",
            "Subquestions": ["subquestion1", "subquestion2"],
            "Indicators": ["indicator1", "indicator2"],
            "Constraints": ["constraint1", "constraint2"],
            "Time_vars": ["t1", "t2"]
        }]
    """
    if not enable_knowledge:
        return []
    
    # if experiment_setting not provided, use global setting first
    if experiment_setting is None:
        try:
            from .stepwise import CURRENT_EXPERIMENT_SETTING
            experiment_setting = CURRENT_EXPERIMENT_SETTING
        except:
            pass
        
        # if global setting also not provided, try to create default setting
        if experiment_setting is None:
            experiment_setting = _create_default_experiment_setting()
            if experiment_setting is None:
                # if cannot create default setting, return empty list
                print("⚠️  cannot get ExperimentSetting, skip knowledge query")
                return []
    
    # try:
    store = get_unified_knowledge_store(experiment_setting=experiment_setting)

    # query similar question decomposition examples
    # in shared mode, search all individual databases
    try:
        from memotime.kg_agent.storage_manager import StorageMode
    except ImportError:
        from .storage_manager import StorageMode
    
    search_all = (store.storage_manager.storage_mode == StorageMode.SHARED)
    
    candidates = store.lookup_knowledge(
        query_text=given_question,
        query_type="question_decomposition",
        k=topk,
        sim_threshold=similarity_threshold,
        search_all_individual=search_all
    )
    examples = []
    for candidate in candidates:
    # try:
        # get enhancement data from execution_data
        execution_data = candidate.get("execution_data", {})
        
        if execution_data.get("type") == "question_decomposition":
            # check question type filter (allow unknown type through)
            ex_question_type = execution_data.get("question_type")
            if question_type and ex_question_type and ex_question_type != "unknown" and ex_question_type != question_type:
                continue
            
            example = {
                "Q": execution_data.get("question", ""),
                "Subquestions": execution_data.get("subquestions", []),
                "Indicators": execution_data.get("indicators", []),
                "Constraints": execution_data.get("constraints", []),
                "Time_vars": execution_data.get("time_vars", [])
            }
            examples.append(example)
            # except Exception as e:
            #     print(f"❌ parse candidate data failed: {e}")
            #     continue
                
    return examples
        
    # except Exception as e:
    #     print(f"❌ get decomposition enhancement failed: {e}")
    #     return []

def get_seed_selection_enhanced(
    given_subquestion: str, 
    topk: int = 10, 
    question_type: str = None,
    experiment_setting: Optional[ExperimentSetting] = None,
    enable_knowledge: bool = True,
    similarity_threshold: float = 0.5
) -> List[Dict[str, Any]]:
    """
    get seed selection enhancement examples
    
    Args:
        given_subquestion: current subquestion
        topk: return top k examples
        question_type: question type filter
        experiment_setting: experiment setting
        enable_knowledge: whether to enable knowledge storage
        
    Returns:
        enhanced examples list in specified format
        [{
            "subquestion": "subquestion text",
            "entities": ["1. entity1", "2. entity2", "3. entity3"],
            "output": [1, 2]
        }]
    """
    if not enable_knowledge:
        return []
    
    # if experiment_setting not provided, use global setting first
    if experiment_setting is None:
        try:
            from .stepwise import CURRENT_EXPERIMENT_SETTING
            experiment_setting = CURRENT_EXPERIMENT_SETTING
        except:
            pass
        
        # if global setting also not provided, try to create default setting
        if experiment_setting is None:
            experiment_setting = _create_default_experiment_setting()
            if experiment_setting is None:
                # if cannot create default setting, return empty list
                print("⚠️  cannot get ExperimentSetting, skip knowledge query")
                return []
    
    try:
        store = get_unified_knowledge_store(experiment_setting=experiment_setting)
        
        # query similar seed selection examples
        # in shared mode, search all individual databases
        try:
            from memotime.kg_agent.storage_manager import StorageMode
        except ImportError:
            from .storage_manager import StorageMode
        
        search_all = (store.storage_manager.storage_mode == StorageMode.SHARED)
        
        candidates = store.lookup_knowledge(
            query_text=given_subquestion,
            query_type="seed_selection",
            k=topk,
            sim_threshold=similarity_threshold,
            search_all_individual=search_all
        )
        
        examples = []
        for candidate in candidates:
            try:
                # get enhancement data from execution_data
                execution_data = candidate.get("execution_data", {})
                if execution_data.get("type") == "seed_selection":
                    # check question type filter (allow unknown type through)
                    ex_question_type = execution_data.get("question_type")
                    if question_type and ex_question_type and ex_question_type != "unknown" and ex_question_type != question_type:
                        continue
                    
                    # format entity list
                    entities = execution_data["entities"]
                    formatted_entities = [f"{i+1}. {entity}" for i, entity in enumerate(entities)]
                    
                    example = {
                        "subquestion": execution_data["subquestion"],
                        "entities": formatted_entities,
                        "output": execution_data["output"]
                    }
                    examples.append(example)
            except Exception as e:
                print(f"❌ parse candidate data failed: {e}")
                continue
                
        return examples
        
    except Exception as e:
        print(f"❌ get seed selection enhancement failed: {e}")
        return []

def get_toolkit_selection_enhanced(
    given_subquestion: str, 
    topk: int = 10, 
    question_type: str = None,
    experiment_setting: Optional[ExperimentSetting] = None,
    enable_knowledge: bool = True,
    similarity_threshold: float = 0.5
) -> List[Dict[str, Any]]:
    """
    get toolkit selection enhancement examples
    
    Args:
        given_subquestion: current subquestion
        topk: return top k examples
        question_type: question type filter
        experiment_setting: experiment setting
        enable_knowledge: whether to enable knowledge storage
        
    Returns:
        enhanced examples list in specified format
        [{
            "subquestion": "subquestion text",
            "indicator": {
                "edges": [{"subj": "...", "rel": "...", "obj": "...", "time_var": "..."}],
                "constraints": []
            },
            "seed_info": ["ID: xxx, Name: xxx"],
            "toolkit": "toolkit name",
            "parameters": {"param1": "value1"},
            "context": {},
            "time_hints": {},
            "reasoning": "selection reasoning"
        }]
    """
    if not enable_knowledge:
        return []
    
    # if experiment_setting not provided, use global setting first
    if experiment_setting is None:
        try:
            from .stepwise import CURRENT_EXPERIMENT_SETTING
            experiment_setting = CURRENT_EXPERIMENT_SETTING
        except:
            pass
        
        # if global setting also not provided, try to create default setting
        if experiment_setting is None:
            experiment_setting = _create_default_experiment_setting()
            if experiment_setting is None:
                # if cannot create default setting, return empty list
                print("⚠️  无法获取ExperimentSetting，跳过知识查询")
                return []
    
    try:
        store = get_unified_knowledge_store(experiment_setting=experiment_setting)
        
        # query similar toolkit selection examples
        # in shared mode, search all individual databases
        try:
            from memotime.kg_agent.storage_manager import StorageMode
        except ImportError:
            from .storage_manager import StorageMode
        
        search_all = (store.storage_manager.storage_mode == StorageMode.SHARED)
        
        candidates = store.lookup_knowledge(
            query_text=given_subquestion,
            query_type="toolkit_selection",
            k=topk,
            sim_threshold=similarity_threshold,
            search_all_individual=search_all
        )
        
        examples = []
        for candidate in candidates:
            try:
                # get enhancement data from execution_data
                execution_data = candidate.get("execution_data", {})
                if execution_data.get("type") == "toolkit_selection":
                    # check question type filter (allow unknown type through)
                    ex_question_type = execution_data.get("question_type")
                    if question_type and ex_question_type and ex_question_type != "unknown" and ex_question_type != question_type:
                        continue
                    
                    
                    example = {
                        "subquestion": execution_data["subquestion"],
                        "indicator": execution_data["indicator"],
                        "seed_info": execution_data["seed_info"],
                        "toolkit": execution_data["toolkit"],
                        "parameters": execution_data["parameters"],
                        "context": execution_data["context"],
                        "time_hints": execution_data["time_hints"],
                        "reasoning": execution_data["reasoning"]
                    }
                    examples.append(example)
            except Exception as e:
                print(f"❌ parse candidate data failed: {e}")
                continue
                
        return examples
        
    except Exception as e:
        print(f"❌ get toolkit selection enhancement failed: {e}")
        return []

def map_function_to_toolkit(function_name: str) -> str:
    """
    map actual executed function name to toolkit name
    
    Args:
        function_name: actual executed function name
        
    Returns:
        toolkit name
    """
    # function name to toolkit name mapping
    function_to_toolkit = {
        # one hop retrieval related
        "retrieve_one_hop": "OneHop",
        "intelligent_retrieve_one_hop": "OneHop",
        
        # temporal related
        "find_after_first": "AfterFirst",
        "find_before_last": "BeforeLast",
        "intelligent_find_after_first": "AfterFirst",
        "intelligent_find_before_last": "BeforeLast",
        
        # direct connection
        "find_direct_connection": "DirectConnection",
        "intelligent_find_direct_connection": "DirectConnection",
        "events_on_day": "DayEvents",
        "events_in_month": "MonthEvents",
        "events_in_year": "YearEvents",
        "intelligent_events_on_day": "DayEvents",
        "intelligent_events_in_month": "MonthEvents",
        "intelligent_events_in_year": "YearEvents",
        
        # timeline
        "get_timeline": "Timeline",
        "intelligent_get_timeline": "Timeline",
        
        # range query
        "find_between_range": "BetweenRange",
        "intelligent_find_between_range": "BetweenRange",
        
        # event query
        "get_day_events": "DayEvents",
        "get_month_events": "MonthEvents",
        "get_year_events": "YearEvents",
    }
    
    return function_to_toolkit.get(function_name, function_name)

def record_experience_pool_enhancement(
    subquestion: str,
    evidence: Dict[str, Any],
    toolkit_info: Dict[str, Any],
    context: Dict[str, Any],
    experiment_setting: Optional[ExperimentSetting] = None,
    enable_knowledge: bool = True
):
    """
    directly record experience pool data to unified knowledge store
    
    Args:
        subquestion: subquestion text
        evidence: evidence data
        toolkit_info: toolkit info
        context: context information
        experiment_setting: experiment setting
        enable_knowledge: whether to enable knowledge storage
    """
    if not enable_knowledge:
        return
    
    try:
        # extract entities and time constraint
        entities = []
        time_constraint = None
        
        # extract all entities from subquestion text (keep consistent with query)
        if hasattr(context, 'get') and context.get('subq_obj'):
            subq_obj = context['subq_obj']
            # subq_obj is dictionary format, need to handle correctly
            if isinstance(subq_obj, dict) and 'indicator' in subq_obj:
                indicator = subq_obj['indicator']
                if 'edges' in indicator:
                    for edge in indicator['edges']:
                        if edge.get('subj') and edge['subj'] not in ['?x', '?y']:
                            entities.append(edge['subj'])
                        if edge.get('obj') and edge['obj'] not in ['?x', '?y']:
                            entities.append(edge['obj'])
        
        # if no entities extracted from subquestion, use evidence entities as fallback
        if not entities and evidence.get('entity'):
            entities.append(evidence['entity'])
        
        if evidence.get('time'):
            time_constraint = evidence['time']
        
        # build indicators
        indicators = {
            "edges": [],
            "constraints": []
        }
        
        # extract path information from evidence
        if evidence.get('path'):
            path = evidence['path']
            if len(path) >= 3:
                heads = path[0] if isinstance(path[0], list) else [path[0]]
                relation = path[1]
                tails = path[2] if isinstance(path[2], list) else [path[2]]
                
                for head in heads:
                    for tail in tails:
                        indicators["edges"].append({
                            "subj": head,
                            "rel": relation,
                            "obj": tail,
                            "time_var": "t1"
                        })
        
        # build toolkit parameters
        toolkit_params = toolkit_info.copy() if toolkit_info else {}
        
        # build execution data
        execution_data = {
            "top_candidates": [evidence] if evidence else [],
            "total_candidates": 1,
            "selection_method": toolkit_params.get('method', 'unknown'),
            "execution_time": 0.0,
            "top_paths": context.get('top_paths', [])  # add top_paths information
        }
        
        # determine question type
        question_type = context.get('question_type', 'unknown')
        
        # get sufficiency_args from context
        sufficiency_args = context.get('sufficiency_args', {})
        
        # write to unified knowledge store
        store = get_unified_knowledge_store(experiment_setting=experiment_setting)
        store.store_knowledge(
            query_text=subquestion,
            query_type="subquestion",
            entities=entities,
            time_constraint=time_constraint,
            indicators=indicators,
            evidence=evidence,
            toolkit_params=toolkit_params,
            question_type=question_type,
            template_data={},
            decomposition_data={},
            execution_data=execution_data,
            sufficiency_args=sufficiency_args,
            success_rate=1.0
        )
        
        print(f"✅ experience pool data recorded to unified knowledge store")
        
    except Exception as e:
        print(f"⚠️ record experience pool data failed: {e}")

def get_experience_pool_enhanced(
    given_subquestion: str, 
    topk: int = 10, 
    question_type: str = None,
    experiment_setting: Optional[ExperimentSetting] = None,
    enable_knowledge: bool = True,
    similarity_threshold: float = 0.5
) -> List[Dict[str, Any]]:
    """
    get experience pool enhancement examples
    
    Args:
        given_subquestion: current subquestion
        topk: return top k examples
        question_type: question type filter
        experiment_setting: experiment setting
        enable_knowledge: whether to enable knowledge storage
        
    Returns:
        enhanced examples list in specified format
        [{
            "subquestion": "subquestion text",
            "evidence": {"entity": "entity", "time": "time"},
            "toolkit_info": {"method": "method", "parameters": {}},
            "context": {"question_type": "question type"}
        }]
    """
    if not enable_knowledge:
        return []
    
    try:
        store = get_unified_knowledge_store(experiment_setting=experiment_setting)
        
        # query similar experience pool examples
        candidates = store.lookup_knowledge(
            query_text=given_subquestion,
            query_type="subquestion",
            k=topk,
            sim_threshold=similarity_threshold
        )
        
        examples = []
        for candidate in candidates:
            example = {
                "subquestion": candidate.get('query_text', ''),
                "evidence": candidate.get('evidence', {}),
                "toolkit_info": candidate.get('toolkit_params', {}),
                "context": {"question_type": candidate.get('question_type', '')}
            }
            examples.append(example)
        
        return examples
        
    except Exception as e:
        print(f"⚠️ get experience pool enhancement failed: {e}")
        return []
        
