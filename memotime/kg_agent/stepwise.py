
# =============================
# file: kg_agent/stepwise.py
# =============================
import json
import numpy as np
import re
from dataclasses import asdict
from typing import Any, Dict, List
from .decompose import decompose_question, select_seeds_for_subq, DecompositionResult
# Template learner removed, using unified knowledge store

# Import embedding model related libraries
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Warning: sentence-transformers not available. Using fallback embedding method for entity filtering.")
from .registry import TemplateCard
from .kg_ops import KG
from .prompts import (LLM_SYSTEM_PROMPT, 
                     LLM_SUFFICIENT_TEST_PROMPT, LLM_FINAL_SUFFICIENT_TEST_PROMPT,
                     LLM_REGENERATE_SUBQUESTION_PROMPT, LLM_REGENERATE_FINAL_QUESTION_PROMPT,
                     LLM_SEED_SELECT_PROMPT, LLM_TOOLKIT_SELECT_PROMPT,
                     LLM_INTELLIGENT_TOOLKIT_SELECT_PROMPT, LLM_FALLBACK_ANSWER_PROMPT)
from .llm import LLM
from .toolkit_selector import ToolkitSelector
# TemplateLearner removed, using unified knowledge store
from .intelligent_toolkit_selector import IntelligentToolkitSelector
# Experience integration removed, using unified knowledge store
from .unified_integration import (
    try_unified_knowledge_shortcut,
    record_successful_knowledge,
    get_template_learning_data
)
# Experience config removed, using unified knowledge store
from .performance_monitor import get_performance_monitor, record_llm_call, set_decomposition_info
from .intelligent_retrieval import IntelligentRetrieval
from .debate_vote import DebateVoteSystem

# Template learner removed, using unified knowledge store

# Global intelligent toolkit selector instance
INTELLIGENT_TOOLKIT_SELECTOR = None

# Global intelligent retrieval instance
INTELLIGENT_RETRIEVAL = None

# Global experiment setting (for unified knowledge store)
CURRENT_EXPERIMENT_SETTING = None

# get_template_learner function removed, using unified knowledge store

def get_intelligent_toolkit_selector():
    """Get intelligent toolkit selector instance"""
    global INTELLIGENT_TOOLKIT_SELECTOR
    if INTELLIGENT_TOOLKIT_SELECTOR is None:
        INTELLIGENT_TOOLKIT_SELECTOR = IntelligentToolkitSelector()
    return INTELLIGENT_TOOLKIT_SELECTOR

def get_intelligent_retrieval():
    """Get intelligent retrieval instance"""
    global INTELLIGENT_RETRIEVAL
    if INTELLIGENT_RETRIEVAL is None:
        db_path = KG.get_db_path()  # Use dynamic method to get database path
        INTELLIGENT_RETRIEVAL = IntelligentRetrieval(db_path)
    return INTELLIGENT_RETRIEVAL


# Global embedding model instance
EMBEDDING_MODEL = None

def get_embedding_model():
    """Get embedding model instance"""
    global EMBEDDING_MODEL
    if EMBEDDING_MODEL is None and SENTENCE_TRANSFORMERS_AVAILABLE:
        try:
            # Use multilingual BERT model, supports Chinese and English
            EMBEDDING_MODEL = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            print("✅ Loading embedding model: paraphrase-multilingual-MiniLM-L12-v2")
        except Exception as e:
            print(f"❌ Failed to load embedding model: {e}")
            EMBEDDING_MODEL = None
    return EMBEDDING_MODEL


def get_text_embedding(text: str) -> np.ndarray:
    """Get text embedding vector"""
    model = get_embedding_model()
    if model:
        return model.encode(text, convert_to_tensor=False)
    else:
        # Fallback: simple word frequency vector
        words = text.lower().split()
        unique_words = list(set(words))
        if not unique_words:
            return np.zeros(1)
        vector = np.zeros(len(unique_words))
        for word in words:
            if word in unique_words:
                vector[unique_words.index(word)] += 1
        return vector / (np.linalg.norm(vector) + 1e-8)


def calculate_semantic_similarity(text1: str, text2: str) -> float:
    """Calculate semantic similarity between two texts"""
    try:
        embedding1 = get_text_embedding(text1)
        embedding2 = get_text_embedding(text2)
        
        # Calculate cosine similarity
        similarity = np.dot(embedding1, embedding2) / (
            np.linalg.norm(embedding1) * np.linalg.norm(embedding2) + 1e-8
        )
        return float(similarity)
    except Exception as e:
        print(f"Failed to calculate semantic similarity: {e}")
        return 0.0


def filter_relevant_entities_with_embedding(subquestion: str, topic_entities: List[Dict[str, Any]], 
                                          similarity_threshold: float = 0.4, 
                                          max_entities: int = 50) -> List[Dict[str, Any]]:
    """
    Use embedding model to filter entities related to the subquestion
    
    Args:
        subquestion: Subquestion text
        topic_entities: Candidate entity list
        similarity_threshold: Similarity threshold
        max_entities: Maximum number of entities to keep
        
    Returns:
        Filtered list of relevant entities
    """
    if not topic_entities:
        return []
    
    print(f"🔍 Starting entity semantic filtering: threshold={similarity_threshold}, max_count={max_entities}")
    
    # Calculate similarity between each entity and subquestion
    entity_similarities = []
    
    for entity in topic_entities:
        entity_name = entity.get("name", "")
        if not entity_name:
            continue
        
        # Calculate semantic similarity between entity name and subquestion
        similarity = calculate_semantic_similarity(subquestion, entity_name)
        
        # If entity name appears directly in subquestion, give bonus
        name_lower = entity_name.lower()
        subq_lower = subquestion.lower()
        
        # Direct match bonus - Normalize separators for accurate matching
        normalized_name_lower = name_lower.replace("_", " ").replace("-", " ")
        if normalized_name_lower in subq_lower or any(word in subq_lower for word in normalized_name_lower.split()):
            similarity += 0.3
        
        # Partial match bonus - Also perform normalization preprocessing  
        name_words = set(normalized_name_lower.split())
        subq_words = set(subq_lower.split())
        common_words = name_words.intersection(subq_words)
        if common_words:
            similarity += 0.2 * len(common_words) / len(name_words)
        
        entity_similarities.append({
            'entity': entity,
            'similarity': similarity,
            'name': entity_name
        })
    
    # Sort by similarity
    entity_similarities.sort(key=lambda x: x['similarity'], reverse=True)
    
    # filterlowsimilarentity
    filtered_entities = []
    for item in entity_similarities:
        if item['similarity'] >= similarity_threshold:
            filtered_entities.append(item['entity'])
        
        # limitmax_count
        if len(filtered_entities) >= max_entities:
            break
    
    # iffilterafterentitytoofew，relaxthreshold
    if len(filtered_entities) < 1 and entity_similarities:
        print(f"⚠️ filterafterentityfew({len(filtered_entities)})，relaxthresholdgetTop5highentity")
        filtered_entities = [item['entity'] for item in entity_similarities[:5]]
    
    # displayfilterresultstatistics
    if entity_similarities:
        max_sim = entity_similarities[0]['similarity']
        min_sim = entity_similarities[-1]['similarity']
        avg_sim = sum(item['similarity'] for item in entity_similarities) / len(entity_similarities)
        
        print(f"📊 Similarity statistics: max={max_sim:.3f}, min={min_sim:.3f}, avg={avg_sim:.3f}")
        print(f"✅ Kept top {len(filtered_entities)} relevant entities")
        
        # displayTop5highentity
        # print("🏆 Top5mostrelevantentity:")
        for i, item in enumerate(entity_similarities[:5]):
            print(f"  {i+1}. {item['name']} (similar: {item['similarity']:.3f})")
    
    return filtered_entities


def apply_sorting_and_limit(candidates: List[Dict[str, Any]], parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Apply sorting and limiting based on toolkit parameters
    """
    sort_by = parameters.get("sort_by", "time_asc")
    limit = parameters.get("limit", 10)
    
    # Apply sorting
    if sort_by == "time_asc":
        candidates.sort(key=lambda x: x.get("time", ""))
    elif sort_by == "time_desc":
        candidates.sort(key=lambda x: x.get("time", ""), reverse=True)
    elif sort_by == "time_equal":
        # For equal type, maintain original order
        pass
    
    # Apply limit
    return candidates[:limit]

def id_to_name_list(id: int) -> str:
    return KG.get_entity_name(id)

def execute_indicator_with_toolkit(subq, seeds: List[int], ctx: Dict[str, Any], question_type: str = None, subq_index: int = 0) -> Dict[str, Any]:
    """
    Execute indicator using intelligent toolkit selection
    """
    # Get toolkit instance
    toolkit = KG.get_toolkit()
    
    # Use intelligent toolkit selection
    toolkit_config = intelligent_toolkit_selection(subq, seeds, ctx, question_type)
    # from .intelligent_toolkit_selector import id_to_name

    seeds_info_name_list = [f"ID: {seed}, Name: {id_to_name_list(seed)}" for seed in seeds]
    # seeds_info_name_list = [id to for seed in seeds]
    toolkit_config['seeds_info_name_list'] = seeds_info_name_list
    print(f"Intelligently selected toolkit: {toolkit_config['method_name']}")
    print(f"Toolkit description: {toolkit_config['description']}")
    print(f"Toolkit parameters: {toolkit_config['parameters']}")
    
    # Check if there are multiple toolkit selections
    all_selections = toolkit_config.get('all_selections', [])
    
    # debug：outputallselectdetailedinformation
    print(f"🔍 All selections count: {len(all_selections)}")
    for i, selection in enumerate(all_selections):
        method_name = selection.get('toolkit_name', selection.get('method_name', ''))
        print(f"   Selection {i+1}: {method_name}")
    
    # onlycheckisnohavedifferenttypeToolkit（different method_name）
    unique_methods = set()
    for selection in all_selections:
        method_name = selection.get('toolkit_name', selection.get('method_name', ''))
        if method_name:
            unique_methods.add(method_name)
    
    print(f"🔍 Unique methods: {unique_methods}")
    if len(unique_methods) > 1:
        print(f"Detected {len(unique_methods)} distinct toolkit methods, will try multiple retrieval requests...")
        
        # aseachuniquemethodonlykeeponerepresentativeselect
        filtered_by_method = {}
        for selection in all_selections:
            method_name = selection.get('toolkit_name', selection.get('method_name', ''))
            if method_name and method_name in unique_methods:
                if method_name not in filtered_by_method:
                    filtered_by_method[method_name] = selection
        
        # byuniquemethodorderbuilddeduplicateafterselectlist
        unique_selections = list(filtered_by_method.values())
        print(f"🔍 Filtered to {len(unique_selections)} unique method selections")
        
        return execute_multiple_toolkit_requests(subq, seeds, ctx, unique_selections, toolkit)
    
    # Execute single toolkit method
    try:
        method_name = toolkit_config['method_name']
        params = toolkit_config['parameters'].copy()
        
        # Force use seeds as entity parameters (seeds are already intelligently selected)
        # Seeds should take priority over LLM's entity selection in parameters
        if seeds:
            if len(seeds) == 1:
                # Single seed: always override entity parameter
                params['entity'] = seeds[0]
                print(f"✅ Using seed entity {seeds[0]} as query entity (override)")
                # Remove any extra entity parameters that LLM might have added
                for extra_key in ['entity1', 'entity2', 'entities']:
                    if extra_key in params:
                        removed_val = params.pop(extra_key)
                        print(f"   🧹 Removed extra parameter '{extra_key}': {removed_val}")
            else:
                # Multiple seeds: use as entities list
                params['entities'] = seeds[:3]
                params['entity'] = seeds[0]  # Primary entity
                print(f"✅ Using seed entities {seeds[:3]} as query entities (override)")
        
        # For some methods, ensure necessary parameters
        if method_name == "find_after_first" and 'reference_time' not in params:
            # Extract reference time from context or subquestion
            if ctx.get('times'):
                # Use first time variable as reference time
                time_vars = list(ctx['times'].keys())
                if time_vars:
                    params['reference_time'] = ctx['times'][time_vars[0]]
                    print(f"Using context time {params['reference_time']} as reference time")
        
        if method_name == "find_temporal_sequence" and 'relation' not in params:
            # Extract relation from subquestion indicator
            if subq.indicator.edges:
                relation = subq.indicator.edges[0].rel
                params['relation'] = relation
                print(f"Using subquestion relation {relation} as query relation")
        
        # After First optimization: checkfirstSubquestionrelation，inaftercontinueSubquestioninasasprecisequeryrelation
        first_answer = ctx.get("answers", {}).get("s1", {})
        if first_answer and subq_index > 0:  # forfirstSubquestionafterquestions
            # extractaccuraterelationinformation
            extracted_relation = _extract_relation_from_answer(first_answer, ctx)
            if extracted_relation:
                # Updaterelevantlookupparameters
                for param_key in ['relation', 'target_relation', 'after_relation', 'query_relation']:
                    if param_key in params:
                        print(f"🔄 Update{param_key}parametersto reflectfirststepconfirmedrelation: {extracted_relation}")
                        params[param_key] = extracted_relation
                    elif not params.get('relation'):  # ensurehaveonebaserelation 
                        params['relation'] = extracted_relation
                        print(f"🔄 applyconfirmedrelation{extracted_relation}asasbasequeryrelation")
                        
                # alsoaddtosubq.indicatoredgesin
                if subq.indicator.edges:
                    for edge in subq.indicator.edges:
                        if edge.rel and not edge.rel.startswith('?'):
                            # Updateallnonplaceholderrelation，notagainlimitasaction_relation
                            edge.rel = extracted_relation
                            print(f"🔄 inindicatoredgesinapplyconfirmedrelation: {extracted_relation}")
                            break
        
        # completelyuseLLMselectmode，notsubject tooldtemplatelimit  
        print(f"🚀 useintelligentLLMsearchmode: {method_name}")
        
        # Check if method exists
        if not hasattr(toolkit, method_name):
            print(f"Method {method_name} does not exist, fallback to basic retrieval")
            return execute_indicator(subq, seeds, ctx)
        
        # Use intelligent retrieval for all toolkit methods
        print(f"🚀 useunifiedIntelligent retrievalsystem: {method_name}")
        return execute_intelligent_retrieval(subq, seeds, ctx, method_name, params, toolkit, False)
        
    except Exception as e:
        print(f"Toolkitexecutefailed: {e}")
        # Fallback to basic retrieval
        return execute_indicator(subq, seeds, ctx)


def execute_intelligent_retrieval(subq, seeds: List[int], ctx: Dict[str, Any], 
                                method_name: str, params: Dict[str, Any], toolkit, 
                                has_other_toolkits: bool = False) -> Dict[str, Any]:
    """
    useIntelligent retrievalsystemexecutequery，integrateto现haveToolkitcall processin
    """
    try:
        # getIntelligent retrievalinstance
        intelligent_retrieval = get_intelligent_retrieval()
        
        # determineentity（supportmultipleseedentity）
        if seeds and len(seeds) > 1:
            # multipleseedentitycase
            entity_names = []
            for seed in seeds:
                if isinstance(seed, int) or (isinstance(seed, str) and seed.isdigit()):
                    entity_name = KG.get_entity_name(int(seed))
                    if entity_name:
                        entity_names.append(entity_name)
                    else:
                        print(f"nogetentityname: {seed}")
                else:
                    entity_names.append(seed)
            
            if not entity_names:
                print("No valid entities found for intelligent retrieval")
                return {"ok": False, "candidates": [], "answers": {}, "times": {}}
            
            entity = entity_names
            print(f"🚀 multipleseedIntelligent retrieval：{entity}")
        else:
            # singleseed entitiescase
            entity = params.get('entity', seeds[0] if seeds else None)
            if not entity:
                print("No entity specified for intelligent retrieval")
                return {"ok": False, "candidates": [], "answers": {}, "times": {}}
            
            entity_name = entity
            if isinstance(entity, int) or (isinstance(entity, str) and entity.isdigit()):
                entity_name = KG.get_entity_name(int(entity))
                if not entity_name:
                    print(f"nogetentityname: {entity}")
                    return {"ok": False, "candidates": [], "answers": {}, "times": {}}
            
            entity = entity_name
        
        # useintegratequestions Intelligent retrieval
        clean_params = params.copy()
        if 'entity' in clean_params:
            del clean_params['entity']
        
        result = intelligent_retrieval.intelligent_retrieve_with_analysis(
            entity=entity,
            subquestion=subq.text,
            subquestion_obj=subq,
            method_name=method_name,
            **clean_params
        )
        
        if "error" in result:
            print(f"Intelligent retrievalfailed: {result['error']}")
            return {"ok": False, "candidates": [], "answers": {}, "times": {}}
        
        # processresult - 保持 with 原haveformat兼容
        candidates = []
        selected_path = None
        
        # 首先checkselected_path
        if "selected_path" in result:
            selected_path = result["selected_path"]
            
            if not isinstance(selected_path, dict):
                print(f"⚠️ selected_pathnotis字典: {type(selected_path)}, 内容: {selected_path}")
                selected_path = None
            
            elif (not selected_path or 
                  selected_path == {} or
                  (not selected_path.get('relation') and 
                   not selected_path.get('head') and 
                   not selected_path.get('heads'))):
                print(f"⚠️ selected_pathas空 or no效，attempt备选方案: {selected_path}")
                selected_path = None
        
        if selected_path is None and "top_3_paths" in result:
            top_paths = result.get("top_3_paths", [])
            if top_paths and len(top_paths) > 0:
                print(f"🔍 usebackuppath，from {len(top_paths)} pathsinfirstpath")
                selected_path = top_paths[0]
                if selected_path and isinstance(selected_path, dict):
                    print(f"🔍 backupselected_path: {selected_path}")
                else:
                    selected_path = None
        
        # Finalcheck：if still isnovalidpath，returnfailed
        if not selected_path or not isinstance(selected_path, dict):
            print(f"❌ noobtainedvalidpathresult")
            return {"ok": False, "candidates": [], "answers": {}, "times": {}}
        
        print(f"🔍 useselected_path: {selected_path}")
        
            # buildcompleteproofinformation
        proof_info = {
            "heads": selected_path.get('heads', []),
            "heads_str": selected_path.get('heads_str', ''),
            "heads_count": selected_path.get('heads_count', 1),
            "relation": selected_path.get('relation', 'Unknown'),
            "tail": selected_path.get('tail', 'Unknown'),
            "time_start": selected_path.get('time_start', 'Unknown'),
            "time_start_epoch": selected_path.get('time_start_epoch', 0),
            "similarity": selected_path.get('similarity', 0),
            "count": selected_path.get('count', 1),
            "original_paths": selected_path.get('original_paths', []),
            "max_similarity": selected_path.get('max_similarity', 0),
            "min_similarity": selected_path.get('min_similarity', 0),
            "selection_reason": selected_path.get('selection_reason', '')
        }
        
        if selected_path.get('tail_count', 1) >= 1:
            tails_list = selected_path.get('tails', [selected_path.get('tail', 'Unknown')])
            print(f"🎯 aggregatedpath，Found{len(tails_list)}tailentity: {selected_path.get('tails_str', '')}")
            
            for tail_entity in tails_list:
                subquestion_text = subq.text if hasattr(subq, 'text') else str(subq)
                if any(word in subquestion_text.lower() for word in ['who', 'which entity', 'what entity']):
                    entity_answer = tail_entity
                else:
                    entity_answer = tail_entity
                
                candidate = {
                    "entity": entity_answer,
                    "time": selected_path.get('time_start', 'Unknown'),
                    "path": [
                        selected_path.get('heads', [selected_path.get('head', 'Unknown')]),  # heads as list
                        selected_path.get('relation', 'Unknown'),
                        selected_path.get('tails', [tail_entity])  # tails as list
                    ],
                    "provenance": {
                        "method": f"intelligent_{method_name}",
                        "parameters": params,
                        "similarity": selected_path.get('similarity', 0),
                        "selection_reason": selected_path.get('selection_reason', ''),
                        "proof": proof_info  # addcompleteproofinformation
                    }
                }
                candidates.append(candidate)
                print(f"✓ addaggregatedtail: {tail_entity} at {candidate['time']}")
        elif selected_path.get('heads_count', 1) > 1:
            heads_list = selected_path.get('heads', [selected_path.get('head', 'Unknown')])
            print(f"🎯 aggregatedpath，Found{len(heads_list)}headentity: {selected_path.get('heads_str', '')}")
            
            for head_entity in heads_list:
                subquestion_text = subq.text if hasattr(subq, 'text') else str(subq)
                if any(word in subquestion_text.lower() for word in ['who', 'which entity', 'what entity']):
                    entity_answer = head_entity
                else:
                    entity_answer = head_entity
                
                candidate = {
                    "entity": entity_answer,
                    "time": selected_path.get('time_start', 'Unknown'),
                    "path": [
                        selected_path.get('heads', [head_entity]),  # heads as list
                        selected_path.get('relation', 'Unknown'),
                        selected_path.get('tails', [selected_path.get('tail', 'Unknown')])  # tails as list
                    ],
                    "provenance": {
                        "method": f"intelligent_{method_name}",
                        "parameters": params,
                        "similarity": selected_path.get('similarity', 0),
                        "selection_reason": selected_path.get('selection_reason', ''),
                        "proof": proof_info  # addcompleteproofinformation
                    }
                }
                candidates.append(candidate)
                print(f"✓ addaggregatedhead: {head_entity} at {candidate['time']}")
        else:
            # singlepath or singletail/headcase
            subquestion_text = subq.text if hasattr(subq, 'text') else str(subq)
            if any(word in subquestion_text.lower() for word in ['who', 'which entity', 'what entity']):
                entity_answer = selected_path.get('heads_str', selected_path.get('head', 'Unknown'))
            else:
                entity_answer = selected_path.get('tail', 'Unknown')

            candidate = {
                "entity": entity_answer,
                "time": selected_path.get('time_start', 'Unknown'),
                "path": [
                    selected_path.get('heads', [selected_path.get('head', 'Unknown')]),  # heads as list
                    selected_path.get('relation', 'Unknown'),
                    selected_path.get('tails', [selected_path.get('tail', 'Unknown')])  # tails as list
                ],
                "provenance": {
                    "method": f"intelligent_{method_name}",
                    "parameters": params,
                    "similarity": selected_path.get('similarity', 0),
                    "selection_reason": selected_path.get('selection_reason', ''),
                    "proof": proof_info  # addcompleteproofinformation
                }
            }
            candidates.append(candidate)
        
        chosen = candidates[0] if candidates else None
        times = {}
        answers = {}
        
        if chosen and isinstance(chosen, dict):
            # Extract time information
            if subq.indicator.edges and chosen.get("time"):
                time_var = subq.indicator.edges[0].time_var
                times[time_var] = chosen.get("time")
            
            if len(candidates) > 1:
                all_entities = [candidate.get("entity") for candidate in candidates]
                entity_answer = ", ".join(all_entities)
                
                chosen["entity"] = entity_answer
                synthetic_relation_tail = f"{chosen.get('provenance', {}).get('proof', {}).get('relation', 'Unknown')} -> {', '.join(all_entities)}"
                                
            else:
                entity_answer = chosen.get("entity")
                synthetic_relation_tail = f"{chosen.get('provenance', {}).get('proof', {}).get('relation', 'Unknown')} -> {chosen.get('entity')}"
            
            answers[subq.sid] = {
                "entity": entity_answer, 
                "time": chosen.get("time"),
                "score": chosen.get("provenance", {}).get("similarity", 0.9),
                "reason": f"Selected by intelligent {method_name}",
                "proof": chosen.get("provenance", {}).get("proof", {}),  # addproofinformation
                "selection_reason": chosen.get("provenance", {}).get("selection_reason", ''),
                "candidates": [candidate.get("entity") for candidate in candidates] if len(candidates) > 1 else None,
                "candidate_count": len(candidates),
                "aggregated_entities": [candidate.get("entity") for candidate in candidates] if len(candidates) > 1 else None,
                "aggregated_path_summary": synthetic_relation_tail if len(candidates) > 1 else None   
            }
        
        print(f"✅ Intelligent retrievalcomplete: Found {len(candidates)} 候选")
        if len(candidates) > 1:
            all_entities = [candidate.get("entity") for candidate in candidates]
            entities_str = ", ".join(all_entities)
            print(f"select: {entities_str} at {chosen.get('time') if chosen else 'Unknown'}")
            print(f"🎯 aggregatedcandidates: {len(candidates)} entity")
        elif chosen and isinstance(chosen, dict):
            print(f"select: {chosen.get('entity')} at {chosen.get('time')}")
            print(f"🎯 similar: {chosen.get('provenance', {}).get('similarity', 0):.3f}")
        
        top_3_paths = result.get("top_3_paths", [])
        if not isinstance(top_3_paths, list):
            print(f"⚠️ top_3_pathsnotislist: {type(top_3_paths)}")
            top_3_paths = []
        
        enhanced_params = params.copy()
        
        if "retrieval_analysis" in result:
            analysis = result["retrieval_analysis"]
            time_dim = analysis.get("time_dimension", {})
            
            if time_dim.get("type") == "after" and time_dim.get("value"):
                enhanced_params["after"] = time_dim["value"]
            elif time_dim.get("type") == "before" and time_dim.get("value"):
                enhanced_params["before"] = time_dim["value"]
            elif time_dim.get("type") == "between" and time_dim.get("value"):
                enhanced_params["between"] = time_dim["value"]
            
            if time_dim.get("constraint") == "after" and time_dim.get("value"):
                enhanced_params["reference_time"] = time_dim["value"]
            elif time_dim.get("constraint") == "before" and time_dim.get("value"):
                enhanced_params["reference_time"] = time_dim["value"]
        
        return {
            "ok": bool(candidates),
            "chosen": chosen,
            "candidates": candidates,
            "answers": answers,
            "times": times,
            "explanations": [f"Used intelligent {method_name} with semantic pruning"],
            "toolkit_config": {
                "method_name": f"intelligent_{method_name}",
                "description": "Intelligent retrieval with semantic pruning and LLM selection",
                "parameters": enhanced_params,  # useenhanceparameters，containstimeconstraints
                "retrieval_analysis": result.get("retrieval_analysis", {}),  # add析result
                "retrieval_strategy": result.get("retrieval_strategy", {})  # add策略information
            },
            "top_paths": top_3_paths  # addtop3pathinformation
        }
        
    except Exception as e:
        print(f"Intelligent retrievalexecutefailed: {e}")
        import traceback
        traceback.print_exc()
        # Fallback to basic retrieval
        return execute_indicator(subq, seeds, ctx)

def execute_indicator(subq, seeds: List[int], ctx: Dict[str, Any]) -> Dict[str, Any]:
    """
    """
    pseudo_card = TemplateCard(
        workflow_id="indicator_exec",
        spec={
            "intent": {"short": "Execute indicator triples with temporal constraints (KG-only)"},
            "core_steps": [
                {"id": "E1", "name": "Query by triples", "actions": ["match edges with seeds"]},
                {"id": "E2", "name": "Apply constraints", "actions": subq.indicator.constraints},
                {"id": "E3", "name": "Select earliest/valid", "actions": ["respect non-decreasing times"]}
            ],
            "output_schema": {"items": [{"entity": "str", "time": "str", "path": [], "provenance": {}}]},
        }
    )
    
    # Build context information including previous subquestion answers
    linked_stub = {
        "seeds": seeds, 
        "ctx": ctx, 
        "indicator": {
        "edges": [asdict(e) for e in subq.indicator.edges],
        "constraints": subq.indicator.constraints
        }
    }
    
    # executeKG retrieval and LLM pruning
    res = KG.run_workflow(pseudo_card, subq.text, linked_stub)
    
    # Process results
    candidates = res.get("items", [])
    chosen = None
    times = {}
    answers = {}
    
    if candidates:
        # selectmaxscorecandidate
        chosen = candidates[0]  # 经byScoreSort
        
        # Extract time information
        if subq.indicator.edges and chosen.get("time"):
            time_var = subq.indicator.edges[0].time_var
            times[time_var] = chosen.get("time")
        
        # buildanswer
        answers[subq.sid] = {
            "entity": chosen.get("entity"),
            "time": chosen.get("time"),
            "score": chosen.get("provenance", {}).get("score", 0.0),
            "reason": chosen.get("provenance", {}).get("reason", "")
        }
    
    return {
        "ok": bool(candidates),
        "chosen": chosen,
        "candidates": candidates,
        "answers": answers,
        "times": times,
        "explanations": res.get("explanations", []),
    }




def solve_with_decomposition(agent, question: str, topic_entities: List[Dict[str, Any]],
                             max_retries: int = None, 
                             max_depth: int = None,
                             max_branch: int = None,
                             quid: int = None,
                             use_hybrid: bool = None,
                             use_template_learning: bool = None,
                             storage_mode: str = None,
                             llm_model: str = None) -> Dict[str, Any]:
    try:
        try:
            from ..config import TPKGConfig
        except (ImportError, ValueError):
            from memotime.config import TPKGConfig
        
        if max_retries is None:
            max_retries = TPKGConfig.MAX_RETRIES
        if max_depth is None:
            max_depth = TPKGConfig.MAX_DEPTH
        if max_branch is None:
            max_branch = TPKGConfig.MAX_TOTAL_BRANCHES
        if use_hybrid is None:
            use_hybrid = TPKGConfig.USE_HYBRID_RETRIEVAL
    except ImportError:
        if max_retries is None:
            max_retries = 2
        if max_depth is None:
            max_depth = 3
        if max_branch is None:
            max_branch = 20  
        if use_hybrid is None:
            use_hybrid = True
    
    use_template_learning = False
    
    from .storage_manager import ExperimentSetting, StorageMode
    if storage_mode is None:
        try:
            from ..config import TPKGConfig
            storage_mode = getattr(TPKGConfig, 'STORAGE_MODE', 'shared')
        except ImportError:
            storage_mode = "shared"
    if llm_model is None:
        try:
            from ..config import TPKGConfig
            llm_model = getattr(TPKGConfig, 'DEFAULT_LLM_MODEL', 'gpt-4o')
        except ImportError:
            llm_model = "gpt-4o"
    import os
    import json
    from pathlib import Path
    
    config_file_path = str(Path(__file__).parent.parent.parent / "Data" / ".config_name.json")
    
    config_params = {
        'max_retries': max_retries,
        'max_depth': max_depth,
        'max_branch': max_branch,
        'use_hybrid': use_hybrid,
        'storage_mode': storage_mode,
        'llm_model': llm_model
    }
    try:
        try:
            from ..config import TPKGConfig
        except (ImportError, ValueError):
            from memotime.config import TPKGConfig
        
        config_params['enable_shared_fallback'] = getattr(TPKGConfig, 'ENABLE_SHARED_FALLBACK', False)
        if llm_model is None:
            llm_model = TPKGConfig.DEFAULT_LLM_MODEL
            config_params['llm_model'] = llm_model
    except:
        pass
    
    config_name = None
    if os.path.exists(config_file_path):
        try:
            with open(config_file_path, 'r') as f:
                saved_configs = json.load(f)
                key_params = {
                    'max_retries': config_params.get('max_retries'),
                    'max_depth': config_params.get('max_depth'),
                    'max_branch': config_params.get('max_branch'),
                    'use_hybrid': config_params.get('use_hybrid'),
                    'storage_mode': config_params.get('storage_mode'),
                    'llm_model': config_params.get('llm_model')  
                }
                
                for saved_config in saved_configs:
                    saved_params = saved_config.get('params', {})
                    saved_key_params = {
                        'max_retries': saved_params.get('max_retries'),
                        'max_depth': saved_params.get('max_depth'),
                        'max_branch': saved_params.get('max_branch'),
                        'use_hybrid': saved_params.get('use_hybrid'),
                        'storage_mode': saved_params.get('storage_mode'),
                        'llm_model': saved_params.get('llm_model')  
                    }
                    
                    if saved_key_params == key_params:
                        config_name = saved_config.get('name')
                        print(f"📋 Use saved configuration name: {config_name}")
                        break
        except Exception as e:
            print(f"Warning: readconfigurationfilefailed: {e}")
    
    # ifnoFoundmatchconfiguration，generate new configuration name
    if not config_name:
        config_name_parts = []
        config_name_parts.append(f"retry{max_retries}")
        config_name_parts.append(f"depth{max_depth}")
        config_name_parts.append(f"branch{max_branch}")
        if use_hybrid:
            config_name_parts.append("hybrid")
        config_name_parts.append("unified")
        config_name_parts.append(storage_mode)
        # config_name_parts.append(llm_model)
        
        try:
            try:
                from ..config import TPKGConfig
            except (ImportError, ValueError):
                from memotime.config import TPKGConfig
            
            if not getattr(TPKGConfig, 'ENABLE_SHARED_FALLBACK', False):
                config_name_parts.append("nofallback")
            
            llm_model = TPKGConfig.DEFAULT_LLM_MODEL
            if llm_model:
                model_short = llm_model.replace('gpt-', 'gpt').replace('-', '').replace('o', '')
                config_name_parts.append(model_short)
        except:
            pass
        
        config_name = "_".join(config_name_parts)
        
        # save new configuration to file
        try:
            saved_configs = []
            if os.path.exists(config_file_path):
                with open(config_file_path, 'r') as f:
                    saved_configs = json.load(f)
            
            saved_configs.append({
                'name': config_name,
                'params': config_params
            })
            
            os.makedirs(os.path.dirname(config_file_path), exist_ok=True)
            with open(config_file_path, 'w') as f:
                json.dump(saved_configs, f, indent=2)
            
            print(f"💾 save newconfigurationname: {config_name}")
        except Exception as e:
            print(f"Warning: Saveconfigurationfilefailed: {e}")
    
    experiment_setting = ExperimentSetting(
        max_retries=max_retries,
        max_depth=max_depth,
        max_total_branches=max_branch,
        use_hybrid_retrieval=use_hybrid,
        use_experience_pool=True,  # unified knowledge storage always enabled
        use_template_learning=use_template_learning,
        config_name=config_name
    )
    
    # set global experiment_setting for other function use
    global CURRENT_EXPERIMENT_SETTING
    CURRENT_EXPERIMENT_SETTING = experiment_setting
    
    # set storage mode
    from .storage_manager import set_storage_mode
    if storage_mode == "individual":
        set_storage_mode(StorageMode.INDIVIDUAL)
    else:
        set_storage_mode(StorageMode.SHARED)
    
    original_question = question
    global_retry_count = 0
    max_original_decompose_retries = max_retries  # useconfigurationretrytimes
    original_decompose_retry_count = 0  # original retry count

    monitor = get_performance_monitor()
    
    # store original result，eachtimes global retry都fromoriginal start
    original_decomposition = None

    # --- fallback：avoid sufficiency_test not defined ---
    def _safe_suff(default_answer=None):
        return {"sufficient": False, "answer": default_answer, "reason": "no_result_or_error", "action": "retreval again"}
    
    # --- checkbranchcountlimit ---
    def _check_branch_limit(dec, current_depth):
        total_branches = len(dec.subquestions)
        if total_branches > max_branch:
            print(f"[branch_limit] current branch count {total_branches} exceeds limit {max_branch}")
            print(f"[branch_limit] current depth: {current_depth}")
            return True
        return False
    
    # --- original retry ---
    def _original_decompose_retry(question):
        nonlocal original_decompose_retry_count
        
        if original_decompose_retry_count >= max_original_decompose_retries:
            print(f"[original_decompose_limit] reached most original retry times {max_original_decompose_retries}")
            return None
        
        print(f"[original_decompose_retry] the {original_decompose_retry_count + 1}/{max_original_decompose_retries} times original retry")
        
        try:
            with monitor.monitor_stage("classification"):
                new_dec = decompose_question(question)
            print(f"[original_decompose_retry] new decomposition complete, contains {len(new_dec.subquestions)} subquestions")
            return new_dec
        except Exception as e:
            print(f"[warn] original retry failed: {e}")
            return None

    # --- subquestion action execute, only support three actions ---
    def _apply_action_for_subq(action: str,
                               updated_subq,
                               ctx: Dict[str, Any],
                               step: Dict[str, Any],
                               dec,
                               subq_index: int,
                               current_depth: int):
        """
        return control flags:
          {"continue_same_subq": True, "depth_increased": bool}    继continuein当Top子问位置retry
        """
        # read top text with optional retrieval configuration
        tk_cfg = {}
        if isinstance(step, dict):
            tk_cfg = step.get("toolkit_config", {}) or {}

        def _set_cfg(k, v):
            try:
                tk_cfg[k] = v
                step["toolkit_config"] = tk_cfg
            except Exception:
                pass

        subq_text = getattr(updated_subq, "text", None) or (updated_subq.get("text") if isinstance(updated_subq, dict) else "")
        a = (action or "").lower().replace(" ", "_")

        # 1) decomposition → do two times decomposition and insert to top position
        if "decompo" in a:
            # checkdepthlimit：if top depth is greater than max_depth, then not continue decomposition
            if current_depth >= max_depth:
                print(f"[warn] reached most decomposition depth {max_depth}, skip decomposition")
                return {"continue_same_subq": True, "depth_increased": False, "depth_limit_reached": True}
            
            try:
                new_dec = decompose_question(subq_text)
                if hasattr(new_dec, "subquestions") and len(new_dec.subquestions) > 1:
                    # check decomposition after is no greater than branchcountlimit
                    new_total_branches = len(dec.subquestions) - 1 + len(new_dec.subquestions)
                    if new_total_branches > max_branch:
                        print(f"[branch_limit] decomposition after will produce {new_total_branches} branch, exceeds limit {max_branch}")
                        return {"continue_same_subq": True, "depth_increased": False, "depth_limit_reached": False, "branch_limit_reached": True}
                    
                    dec.subquestions = (
                        dec.subquestions[:subq_index] +
                        new_dec.subquestions +
                        dec.subquestions[subq_index + 1:]
                    )
                    print(f"[decomposition] decomposed {len(new_dec.subquestions)} subquestions, depth from {current_depth} to {current_depth + 1}")
                    return {"continue_same_subq": True, "depth_increased": True, "depth_limit_reached": False, "branch_limit_reached": False}
                else:
                    print(f"[decomposition] decomposition failed, not produce multiple subquestions")
                    return {"continue_same_subq": True, "depth_increased": False, "depth_limit_reached": False, "branch_limit_reached": False}
            except Exception as e:
                print(f"[warn] decomposition action failed: {e}")
                return {"continue_same_subq": True, "depth_increased": False, "depth_limit_reached": False, "branch_limit_reached": False}

        # 2) refine → refine subquestion text (if support regenerate_subquestion), and lightly adjust retrieval parameters
        if "ref" in a:
            # lightly adjust retrieval parameters: relax candidate number, slightly lower threshold, allow semantic neighbors
            _set_cfg("candidate_topk", min(200, int(tk_cfg.get("candidate_topk", 50)) + 20))
            _set_cfg("score_threshold", max(0.0, float(tk_cfg.get("score_threshold", 0.5)) - 0.05))
            _set_cfg("allow_semantic_neighbors", True)
            _set_cfg("extra_seed_k", min(10, int(tk_cfg.get("extra_seed_k", 0)) + 2))
            # optional: call regenerate_subquestion function; if not in then ignore
            try:
                refined = regenerate_subquestion(subq_text, ctx)  # noqa: F821
                if refined and isinstance(refined, dict) and refined.get("new_subquestion"):
                    new_text = refined["new_subquestion"]
                    if hasattr(updated_subq, "text"):
                        updated_subq.text = new_text
                    else:
                        updated_subq["text"] = new_text
                    print(f"[refine] subquestion rewritten to: {new_text}")
            except NameError:
                # no this function just keep retrieval parameters changed
                pass
            except Exception as e:
                print(f"[warn] refine action regenerate_subquestion failed: {e}")
            return {"continue_same_subq": True, "depth_increased": False}

        # 3) retreval again / retrieval_again / retry → use original subquestion again times retrieval (only lightly adjust parameters)
        if a in ("retreval_again", "retrieval_again", "retry"):
            _set_cfg("candidate_topk", min(200, int(tk_cfg.get("candidate_topk", 50)) + 20))
            _set_cfg("score_threshold", max(0.0, float(tk_cfg.get("score_threshold", 0.5)) - 0.05))
            return {"continue_same_subq": True, "depth_increased": False}

        # not recognized then by "again times retrieval" process
        _set_cfg("candidate_topk", min(200, int(tk_cfg.get("candidate_topk", 50)) + 20))
        _set_cfg("score_threshold", max(0.0, float(tk_cfg.get("score_threshold", 0.5)) - 0.05))
        return {"continue_same_subq": True, "depth_increased": False}

    while original_decompose_retry_count < max_original_decompose_retries:
        print("="*100)
        print("Start solving problem")
        print("Question Classification and Decomposition: ", question)
        print("Topic Entities: ", topic_entities)
        print("Try times: ", original_decompose_retry_count)
        print("Max retries: ", max_original_decompose_retries)
        print("="*100)

        with monitor.monitor_stage("Question Classification and Decomposition"):
            dec = decompose_question(question)
        if original_decompose_retry_count == 0:
            original_decomposition = dec  # save original decomposition
            print("[initial] Original decomposition completed, depth=1, contains {len(dec.subquestions)} subquestions")
            # print(f"original decomposition: {dec.subquestions}")
        else:
            print("[original_decompose_retry] The {original_decompose_retry_count + 1} time from the original problem to re-decompose, contains {len(dec.subquestions)} subquestions")

        subquestion_texts = [subq.text for subq in dec.subquestions]
        set_decomposition_info(question, subquestion_texts, "llm_decomposition")

        ctx: Dict[str, Any] = {"times": {}, "answers": {}, "entities": {}, "answer_edges": [], "time_constraints": {}, "question_type": dec.question_type}
        trajectory: List[Dict[str, Any]] = []
        reasoning_path: List[Dict[str, Any]] = []
        all_retrieved_info: List[Dict[str, Any]] = []

        subq_index = 0
        max_subq_retries = 3
        subquestion_retry_states: Dict[int, int] = {}
        current_depth = 1  # when top decomposition depth, original decomposition depth=1

        while subq_index < len(dec.subquestions):
            subq = dec.subquestions[subq_index]
            current_subq_retry = subquestion_retry_states.get(subq_index, 0)
            print("="*100)
            print("Processing Subquestion: ", subq_index + 1)
            print(f"(Try {current_subq_retry + 1}/{max_subq_retries})")
            print(f"Subquestion: {subq.text}")
            print(f"Current depth: {current_depth}/{max_depth}")
            print("="*100)

            updated_subq = update_subquestion_with_context(subq, ctx, subq_index)
            if updated_subq != subq:
                print(f"Updated subquestion: {updated_subq.text}")
            print("Question indicator:", updated_subq.indicator)

            with monitor.monitor_stage("Retrieval and Pruning"):
                print("Retrieval and Pruning")
                
                # === Experience pool short circuit attempt ===
                def _check_sufficiency(subquestion, indicators, evidence, context):
                    """sufficient性checkwrapper - return(bool, str)元组"""
                    if not evidence or not isinstance(evidence, dict):
                        return (False, "No evidence")
                    
                    try:
                        result = test_answer_sufficiency(
                            subquestion, evidence, [evidence], context, [], {}, {}, [], experiment_setting
                        )
                        return (result.get('sufficient', False), result.get('reason', 'Unknown'))
                    except Exception as e:
                        return (False, f"Sufficiency check failed: {e}")
                
                pool_result = None

                if pool_result:
                        # experience pool hit，direct use result
                    if pool_result.get("source") == "unified_knowledge_store":
                        print("🚀 unified knowledge storage hit！skip seed select and retrieval")
                        source_name = "unified knowledge storage"
                    else:
                        print("🚀 Experience poolhit！skip seed select and retrieval")
                        source_name = "experience pool"
                    
                    # build candidate list (contains top n candidates)
                    candidates = [pool_result["evidence"]]
                    if pool_result.get("top_candidates"):
                        # convert top n candidates to candidates format
                        for top_cand in pool_result["top_candidates"]:
                            candidate = {
                                "entity": top_cand.get("entity", ""),
                                "time": top_cand.get("time", ""),
                                "score": top_cand.get("score", 0.0),
                                "path": top_cand.get("path", []),
                                "provenance": top_cand.get("provenance", {})
                            }
                            if candidate not in candidates:  # 避免重复
                                candidates.append(candidate)
                        print(f"📊 contains {len(pool_result['top_candidates'])} top candidates")
                    
                    step = {
                        "ok": True,
                        "candidates": candidates,
                        "answers": {updated_subq.sid: pool_result["evidence"]},
                        "chosen": pool_result["evidence"],
                        "times": {updated_subq.sid: pool_result["evidence"].get("time")},
                        "source": "step_answer_souce",
                        "similarity": pool_result.get("similarity", 1.0),
                        "toolkit_config": pool_result.get("toolkit_params", {}),
                        "top_candidates": pool_result.get("top_candidates", []),
                        "total_candidates": pool_result.get("total_candidates", 1)
                    }
                    seeds = []  # experience pool hit, not need seeds
                    selected_seed_names = []
                else:
                    # experience pool not hit，execute original process
                    print("⚪ Experience poolnot hit，Execute original retrieval process")
                    enhanced_prompt = get_seeds_enhanced_prompt(updated_subq.text, dec.question_type)
                    seeds = intelligent_seed_selection(updated_subq, topic_entities, ctx, enhanced_prompt)
                    print(f"Selected {len(seeds)} seed entities: {seeds[:5]}...")
                    print("="*100)

                    selected_seed_names = []
                    for j in range(len(seeds)):
                        for k in topic_entities:
                            if k["id"] == seeds[j]:
                                selected_seed_names.append(k["name"])
                                break
                    print(f"Selected seed names: {selected_seed_names}")

                    step = execute_indicator_with_toolkit(updated_subq, seeds, ctx, dec.question_type, subq_index)
                    step_answer_souce = "new_retrieval"
            success = False
            sufficiency_test = _safe_suff()
            with monitor.monitor_stage("qa"):
                if step is None:
                    print(f"❌ Step {subq_index + 1} execution failed (return None)")
                elif step.get("ok"):
                    print(f"✓ Found {len(step.get('candidates', []))} candidates")
                    all_retrieved_info.extend(step.get("candidates", []))
                    chosen_answer = (step.get("answers", {}) or {}).get(updated_subq.sid, {})
                    print(f"Chosen answer: {chosen_answer.get('entity', 'None')} at {chosen_answer.get('time', '')}")

                    if chosen_answer:
                        print("="*30)
                        print("Sufficiency_test for subquestion: ", updated_subq.text)
                        print("="*30)
                        try:
                            sufficiency_test = test_answer_sufficiency(
                                updated_subq.text, chosen_answer, step.get("candidates", []), ctx,
                                [], step.get("toolkit_config", {}), step.get("debate_vote_result", {}),
                                step.get("top_paths", []), experiment_setting
                            ) or _safe_suff(chosen_answer)
                        except Exception as e:
                            print(f"[warn] test_answer_sufficiency failed: {e}")
                            sufficiency_test = _safe_suff(chosen_answer)

                        print(f"sufficiency_test['answer']: {sufficiency_test.get('answer')}")

                        if sufficiency_test.get('sufficient'):
                            success = True
                            
                            # extractdetailedToolkitparametersinformation
                            detailed_toolkit_info = {}
                            if step.get("toolkit_config"):
                                toolkit_config = step["toolkit_config"]
                                detailed_toolkit_info = {
                                    "method_name": toolkit_config.get('method_name', ''),
                                    "method": toolkit_config.get('method', ''),
                                    "description": toolkit_config.get('description', ''),
                                    "parameters": toolkit_config.get('parameters', {}),
                                    "seeds": toolkit_config.get('seeds', []),
                                    "seeds_info_name_list": [f"ID: {seed}, Name: {id_to_name_list(seed)}" for seed in toolkit_config.get('seeds', [])],
                                    "question_type": toolkit_config.get('question_type', ''),
                                    "subq_index": toolkit_config.get('subq_index', 0),
                                    "all_selections": toolkit_config.get('all_selections', [])
                                }
                            
                            sufficiency_args = {
                                "subquestion": updated_subq.text,
                                "current_answer": chosen_answer,
                                "retrieved_info": step.get("candidates", []),
                                "context": ctx,

                                "previous_subquestions": [],
                                "toolkit_info": step.get("toolkit_config", {}),
                                "debate_vote_result": step.get("debate_vote_result", {}),
                                "top_paths": step.get("top_paths", []),
                                "experiment_setting": experiment_setting.to_dict() if experiment_setting else {}
                            }
                            
                            trajectory.append({
                                "subq": asdict(updated_subq),
                                "selected_seed_names": selected_seed_names,
                                "available_seeds": [i["name"] for i in topic_entities],
                                "result": step,
                                "step_answer_souce": step_answer_souce,
                                "sufficiency_test": sufficiency_test,
                                "detailed_toolkit_info": detailed_toolkit_info,
                                "sufficiency_args": sufficiency_args  # Savesufficient性测试parameters
                            })
                            chosen_answer["LLM_selected_answer"] = sufficiency_test.get('answer')
                            print(f"chosen_answer: {chosen_answer}")
                            
                            # === record success samples to unified knowledge storage ===
                            # experience pool data now in unified knowledge storage in final sufficient test record
                            pass
                                    
                                    
                                    
                                    # # also record to original experience pool for compatibility
                                    # record_successful_subquestion(
                                    #     subq_obj=updated_subq,
                                    #     ctx=ctx,
                                    #     step_result=step,
                                    #     toolkit_params={
                                    #         "seeds": seeds if seeds else [],
                                    #         "method": step.get("chosen", {}).get("provenance", {}).get("method", "unknown"),
                                    #         "question_type": dec.question_type,
                                    #         "subq_index": subq_index
                                    #     },
                                    #     enable_pool=True  # unified knowledge storage always enabled
                                #     # )
                                # except Exception as exp_err:
                                #     print(f"⚠️ recordExperience poolfailed: {exp_err}")

                            # with monitor.monitor_stage("qa"):
                            if step.get("times"):
                                ctx["times"].update(step["times"])
                                print(f"Updated times: {step['times']}")
                                
                                # Updatetimeconstraintsinformation，ensureaftercontinueSubquestion能够use
                                for time_key, time_value in step["times"].items():
                                    if time_key.startswith("s") or time_key.startswith("t"):  # Subquestiontime变量
                                        # add time value to above below text, for after continue subquestion use
                                        ctx["time_constraints"][time_key] = time_value
                                        
                                        # special process：as after_firsttypequestions build t1 mapping
                                        if time_key.startswith("s") and subq_index == 0:  # firstSubquestion
                                            ctx["time_constraints"]["t1"] = time_value
                                            print(f"🕐 build t1 mapping: t1 -> {time_value}")
                                        
                                        print(f"🕐 Updatetimeconstraints {time_key}: {time_value}")
                            
                            if step.get("answers"):
                                ctx["answers"].update(step["answers"])
                                if chosen_answer.get("entity"):
                                    ctx["entities"][f"a{subq_index + 1}"] = chosen_answer["entity"]
                                    print(f"✓ Answer for {updated_subq.sid}: {chosen_answer['entity']} at {chosen_answer.get('time', '')}")

                                    answer_entity = chosen_answer["entity"]
                                    aggregated_path_list = (step.get("chosen", {}) or {}).get("path", [])
                                    # extract detailed toolkit parameters information
                                    detailed_toolkit_info = {}
                                    if step.get("toolkit_config"):
                                        toolkit_config = step["toolkit_config"]
                                        detailed_toolkit_info = {
                                            "method_name": toolkit_config.get('method_name', ''),
                                            "method": toolkit_config.get('method', ''),
                                            "description": toolkit_config.get('description', ''),
                                            "parameters": toolkit_config.get('parameters', {}),
                                            "seeds": toolkit_config.get('seeds', []),
                                            "question_type": toolkit_config.get('question_type', ''),
                                            "subq_index": toolkit_config.get('subq_index', 0),
                                            "all_selections": toolkit_config.get('all_selections', [])
                                        }
                                    
                                    answer_edge = {
                                        "step": subq_index + 1,
                                        "subquestion": updated_subq.text,
                                        "entity": answer_entity,
                                        "time": chosen_answer.get("time", ""),
                                        "path": aggregated_path_list,
                                        "provenance": (step.get("chosen", {}) or {}).get("provenance", {}),
                                        "score": chosen_answer.get("score", 0.0),
                                        "LLM_selected_answer": sufficiency_test.get('answer'),
                                        "sufficiency_test": sufficiency_test,
                                        "detailed_toolkit_info": detailed_toolkit_info
                                    }
                                    ctx["answer_edges"].append(answer_edge)
                                    reasoning_path.append(answer_edge)
                        else:
                            print(f"Answer insufficient, reason: {sufficiency_test.get('reason')}")
                else:
                    print(f"✗ No valid candidates found for subquestion {subq_index + 1}")

            if success:
                subq_index += 1
                subquestion_retry_states[subq_index - 1] = 0
            else:
                if current_subq_retry < max_subq_retries:
                    subquestion_retry_states[subq_index] = current_subq_retry + 1
                    print(f"🔄 Subquestion {subq_index + 1} retry {current_subq_retry + 1}/{max_subq_retries}")

                    next_action = (sufficiency_test or {}).get('action', 'retreval again')
                    
                    # if reached most depth and action is decomposition, then force change to retrieval
                    if current_depth >= max_depth and "decompo" in next_action.lower():
                        print(f"[depth_limit] reached most depth {max_depth}, change action to retrieval")
                        next_action = "retreval again"
                    
                    act_out = _apply_action_for_subq(next_action, updated_subq, ctx, step, dec, subq_index, current_depth)
                    
                    if act_out.get("depth_limit_reached", False):
                        print(f"[depth_limit] reached depthlimit，fromoriginalquestions重new解")
                        new_dec = _original_decompose_retry(question)
                        if new_dec:
                            dec = new_dec
                            current_depth = 1
                            subquestion_retry_states = {}  # reset subquestion retry state
                            subq_index = 0  # reset new start process subquestion
                            continue
                        else:
                            print(f"[depth_limit] original decomposition retry failed，continue when top process")
                    
                    # checkisno触及branchcountlimit
                    if act_out.get("branch_limit_reached", False):
                        print(f"[branch_limit] reached branchcountlimit，fromoriginalquestions重new解")
                        new_dec = _original_decompose_retry(question)
                        if new_dec:
                            dec = new_dec
                            current_depth = 1
                            subquestion_retry_states = {}  # reset subquestion retry state
                            subq_index = 0  # reset new start process subquestion
                            continue
                        else:
                            print(f"[branch_limit] original decomposition retry failed，continue when top process")
                    
                    # Updatedepth
                    if act_out.get("depth_increased", False):
                        current_depth += 1
                        print(f"[depth] when top decomposition depth: {current_depth}/{max_depth}")
                        
                        # check if after is no greater than branchcountlimit
                        if _check_branch_limit(dec, current_depth):
                            print(f"[branch_limit] after branchcount exceeds limit，fromoriginalquestions重new解")
                            new_dec = _original_decompose_retry(question)
                            if new_dec:
                                dec = new_dec
                                current_depth = 1
                                subquestion_retry_states = {}
                                subq_index = 0
                                continue
                            else:
                                print(f"[branch_limit] original decomposition retry failed，continue when top process")

                    # fallback：first question second times failed after extra attempt again decomposition；third times failed after consider again generate question ( with your original logic一致）
                    if subq_index == 0 and current_subq_retry >= 1:
                        if current_subq_retry == 1 and current_depth < max_depth:
                            print("🔄 first question fallback：again do one times decomposition insertion")
                            try:
                                new_subqs = decompose_question(updated_subq.text)
                                if len(new_subqs.subquestions) > 1:
                                    # check if after is no greater than branchcountlimit
                                    new_total_branches = len(dec.subquestions) - 1 + len(new_subqs.subquestions)
                                    if new_total_branches > max_branch:
                                        print(f"[branch_limit] after will produce {new_total_branches} branch，exceeds limit {max_branch}")
                                        new_dec = _original_decompose_retry(question)
                                        if new_dec:
                                            dec = new_dec
                                            current_depth = 1
                                            subquestion_retry_states = {}
                                            subq_index = 0
                                            continue
                                        else:
                                            print(f"[branch_limit] original decomposition retry failed，skip fallback decomposition")
                                            continue
                                    
                                    dec.subquestions = (
                                        dec.subquestions[:subq_index] +
                                        new_subqs.subquestions +
                                        dec.subquestions[subq_index + 1:]
                                    )
                                    subquestion_retry_states[subq_index] = 0
                                    current_depth += 1
                                    print(f"decomposed {len(new_subqs.subquestions)} Subquestion（fallback），depth: {current_depth}/{max_depth}")
                                    
                                    # check if after is no greater than branchcountlimit
                                    if _check_branch_limit(dec, current_depth):
                                        print(f"[branch_limit] after branchcount exceeds limit，fromoriginalquestions重new解")
                                        new_dec = _original_decompose_retry(question)
                                        if new_dec:
                                            dec = new_dec
                                            current_depth = 1
                                            subquestion_retry_states = {}
                                            subq_index = 0
                                            continue
                                        else:
                                            print(f"[branch_limit] original decomposition retry failed，continue when top process")
                                    continue
                            except Exception as e:
                                print(f"[warn] fallback decomposition failed: {e}")
                        elif current_depth >= max_depth:
                            print(f"[warn] reached most depth {max_depth}，fromoriginalquestions重new解")
                            new_dec = _original_decompose_retry(question)
                            if new_dec:
                                dec = new_dec
                                current_depth = 1
                                subquestion_retry_states = {}  # reset subquestion retry state
                                subq_index = 0  # reset new start process subquestion
                                continue
                            else:
                                print(f"[depth_limit] original decomposition retry failed，continue when top process")
                        elif current_subq_retry == 2:
                            print("🔄 first question fallback：fromoriginalquestions重new解")
                            original_decompose_retry_count += 1
                            question = original_question  # reset as original questions
                            break

                    # continue in same one subquestion position retry
                    continue
                else:
                    print(f"❌ Subquestion {subq_index + 1} reached most retry times")
                    if subq_index == 0:
                        break
                    else:
                        return {
                            "ok": False,
                            "error": f"Subquestion {subq_index + 1} failed and reached most retry times",
                            "original_question": question,
                            "attempts": global_retry_count + 1
                        }

        if subq_index >= len(dec.subquestions):
            sorted_reasoning_path = build_temporal_reasoning_path(reasoning_path)
            # final_answer = generate_final_answer(question, sorted_reasoning_path)
            final_answer = ctx["answers"].get("final_answer", "")
            print("="*30)
            print("Final sufficient test for final answer: ", final_answer)
            print("="*30)
            final_sufficiency_test = test_final_answer_sufficiency(
                original_question,  sorted_reasoning_path,
                all_retrieved_info, trajectory, experiment_setting, ctx
            )

            

            # print(f"\n=== FinalAnswersufficient test ===")
            # print(f"sufficient: {final_sufficiency_test.get('sufficient')}")
            # print(f"reason: {final_sufficiency_test.get('reason')}")

            if final_sufficiency_test.get('final_answer') and final_sufficiency_test['final_answer'] != final_answer:
                print(f"LLMgenerate improved FinalAnswer: {final_sufficiency_test['final_answer']}")
                final_answer = final_sufficiency_test['final_answer']

            if final_sufficiency_test.get('sufficient') or global_retry_count >= max_retries:
                final_result = {
                    "ok": True,
                    "question": question,
                    "original_question": original_question,
                    "decomposition": {
                        "subquestions": [asdict(s) for s in dec.subquestions],
                        "time_vars": dec.time_vars
                    },
                    "trajectory": trajectory,
                    "reasoning_path": sorted_reasoning_path,
                    "final_answer": final_answer,
                    "final_context": ctx,
                    "final_sufficiency_test": final_sufficiency_test,
                    "retry_count": global_retry_count,
                    "all_retrieved_info": all_retrieved_info,
                    "summary": {
                        "total_subquestions": len(dec.subquestions),
                        "successful_subquestions": len([t for t in trajectory if t["result"]["ok"]]),
                        "total_candidates": sum(len(t["result"].get("candidates", [])) for t in trajectory),
                        "final_answers": ctx["answers"]
                    }
                }
                print(f"final sufficient: {final_sufficiency_test} ") 
                # exit()  

                if quid is not None:
                    try:
                
                        from .answer_verifier import get_answer_verifier
                        verifier = get_answer_verifier()
                        verification_result = verifier.verify_answer(quid, final_answer)
                        is_correct = verification_result['is_correct']
                        
                        if is_correct:
                            print(f"✅ Answer verificationcorrect: {final_answer}")
                        else:
                            print(f"❌ Answer verificationerror: {final_answer}")
                            print(f"   Golden: {verification_result['golden_answers']}")
                            print(f"   Details: {verification_result['match_details']}")

                        if is_correct:
                            print(f"✅ Answer verified as correct for QID {quid}")
                            answer_type = verification_result.get('answer_type', 'unknown')
                            if answer_type != 'entity':
                                question_type = verification_result.get('question_type', 'unknown')
                                topic_entity_names = [entity.get('name', '') for entity in topic_entities if isinstance(entity, dict)]
                                # Template learnerremoved，skiprecord
                                print(f"ℹ️ Template learnerremoved，skipanswerrecord (QID {quid})")
                            else:
                                print(f"ℹ️ Entity answer verified as correct, but skipping template learning for QID {quid}")

                            final_result["learning"] = {"verified": True, "verification": verification_result}
                        else:
                            print(f"❌ Answer verification failed for QID {quid}")
                            final_result["learning"] = {"verified": False, "verification": verification_result}
                    except Exception as e:
                        print(f"⚠️ Learning system error: {e}")
                        final_result["learning"] = {"error": str(e)}

                return final_result
            else:
                print(f"Final answernotsufficient，reason: {final_sufficiency_test.get('reason')}")
                print(f"from original questions retry (No.{original_decompose_retry_count + 1}/{max_original_decompose_retries} times)")
                original_decompose_retry_count += 1
                question = original_question  # reset as original questions
                continue
        else:
            print(f"Subquestionprocessfailed，from original questions (No.{original_decompose_retry_count + 1}/{max_original_decompose_retries} times)")
            original_decompose_retry_count += 1
            question = original_question  # reset as original questions

    # when reached most retry times, attempt generate fallback answer
    print(f"reached most retry times, attempt generate fallback answer...")
    try:
        print("="*30)
        print("Fallback answer generation")
        print("="*30)
        fallback_result = generate_fallback_answer(original_question, reasoning_path, all_retrieved_info, trajectory)
        print(f"Fallbackanswergeneration result: {fallback_result}")
        
        if fallback_result.get("fallback", False):
            # buildfallbackresult
            fallback_final_result = {
                "ok": True,  # mark as success, because as generate answer
                "question": original_question,
                "original_question": original_question,
                "answer": fallback_result.get("final_answer", ""),
                "reasoning": fallback_result.get("reason", ""),
                "sufficient": False,  # mark as not sufficient
                "fallback": True,  # mark as fallback answer
                "trajectory": trajectory,
                "reasoning_path": reasoning_path,
                "all_retrieved_info": all_retrieved_info,
                "retry_count": original_decompose_retry_count,
                "raw_response": fallback_result.get("raw_response", "")
            }
            
            # ifhavequid，performanswer verification但notperformtemplate learning
            if quid is not None:
                try:
                    # use new answer verification
                    from .answer_verifier import get_answer_verifier
                    fallback_answer = fallback_result.get("final_answer", "")
                    verifier = get_answer_verifier()
                    verification_result = verifier.verify_answer(quid, fallback_answer)
                    is_correct = verification_result['is_correct']
                    
                    if is_correct:
                        print(f"✅ FallbackAnswer verificationcorrect: {fallback_answer}")
                    else:
                        print(f"❌ FallbackAnswer verificationerror: {fallback_answer}")
                        print(f"   Golden: {verification_result['golden_answers']}")
                        print(f"   Details: {verification_result['match_details']}")

                    if is_correct:
                        print(f"✅ Fallback answer verified as correct for QID {quid}")
                        answer_type = verification_result.get('answer_type', 'unknown')
                        if answer_type != 'entity':
                            question_type = verification_result.get('question_type', 'unknown')
                            topic_entity_names = [entity.get('name', '') for entity in topic_entities if isinstance(entity, dict)]
                            # for fallback answer，not perform template learning，but record verification result
                            print(f"ℹ️ Fallback answer verified as correct, but skipping template learning for QID {quid}")
                        else:
                            print(f"ℹ️ Fallback entity answer verified as correct, but skipping template learning for QID {quid}")

                        fallback_final_result["learning"] = {
                            "verified": True, 
                            "verification": verification_result,
                            "fallback": True,
                            "note": "Fallback answer generated due to max retry limit"
                        }
                    else:
                        print(f"❌ Fallback answer verification failed for QID {quid}")
                        fallback_final_result["learning"] = {
                            "verified": False, 
                            "verification": verification_result,
                            "fallback": True,
                            "note": "Fallback answer generated due to max retry limit"
                        }
                    
                    fallback_final_result["quid"] = quid
                    print(f"📚 Fallbackanswerrecorded to database，QID: {quid}")
                except Exception as e:
                    print(f"⚠️ Fallbackanswerverificationfailed: {e}")
                    fallback_final_result["learning"] = {
                        "error": str(e),
                        "fallback": True,
                        "note": "Fallback answer generated due to max retry limit"
                    }
            
            return fallback_final_result
        else:
            print(f"Fallbackanswergeneratefailed")
    except Exception as e:
        print(f"⚠️ Fallbackanswergenerateexception: {e}")
    
    # if fallback also failed，return original error
    return {
        "ok": False,
        "question": question,
        "original_question": original_question,
        "error": "reached most retry times and fallback answer generate failed",
        "retry_count": original_decompose_retry_count,
        "fallback_failed": True
    }

# =============================
# Solution 2: auxiliary function - optional enhance
# =============================

def validate_subquestion_result(step, subq_index: int) -> tuple[bool, str]:
    """
    verify subquestion result auxiliary function
    """
    if step is None:
        return False, f"Step {subq_index + 1} executefailed（returnNone）"
    
    if not step.get("ok"):
        return False, f"Step {subq_index + 1} not found valid candidates"
    
    if not step.get("candidates"):
        return False, f"Step {subq_index + 1} candidates list as empty"
    
    chosen_answer = step.get("answers", {})
    if not chosen_answer:
        return False, f"Step {subq_index + 1} not generate answer"
    
    return True, "verify successfully"


def should_attempt_decomposition(subq_index: int, retry_count: int, error_type: str) -> bool:
    """
    check if should attempt decomposition strategy
    """
    # only in first subquestion and retry 1-2 times when attempt decomposition
    if subq_index != 0:
        return False
    
    if retry_count < 1 or retry_count > 2:
        return False
    
    decomposition_friendly_errors = [
        "No valid candidates",
        "Answer insufficient", 
        "Step returned None"
    ]
    
    return any(error in error_type for error in decomposition_friendly_errors)


def log_retry_attempt(subq_index: int, retry_count: int, max_retries: int, strategy: str = ""):
    """
    unified retry log record
    """
    print(f"🔄 Subquestion {subq_index + 1} retry {retry_count}/{max_retries}")
    if strategy:
        print(f"   Strategy: {strategy}")



def update_subquestion_with_context(subq, ctx: Dict[str, Any], step_num: int):

    from .decompose import SubQuestion, Indicator, IndicatorEdge
    
    # Update subquestion text, replace Time variables
    updated_text = subq.text
    
    # prioritize use time_constraints in values
    time_vars = ctx.get("time_constraints", {})
    if not time_vars:
        time_vars = ctx.get("times", {})
    
    # replace time variables (including t1, t2 etc. generic time variables)
    for time_var, time_value in time_vars.items():
        if time_var in updated_text:
            updated_text = updated_text.replace(time_var, time_value)
            print(f"🕐 replace time variable {time_var} -> {time_value}")
    
    # special process：if subquestion contains t1, t2 etc., but no corresponding value, attempt from abovebelow text in inference
    import re
    time_pattern = r'\bt\d+\b'  # match t1, t2, t3 等
    time_matches = re.findall(time_pattern, updated_text)
    
    for time_match in time_matches:
        # check if contains after time value (to avoid duplicate replacement)
        if time_match in updated_text and not re.search(r'\d{4}-\d{2}-\d{2}', updated_text):
            # attempt from time_constraints in Found corresponding value
            if time_match in ctx.get("time_constraints", {}):
                time_value = ctx["time_constraints"][time_match]
                updated_text = updated_text.replace(time_match, time_value)
                print(f"🕐 infer time variable {time_match} -> {time_value}")
            elif time_match in ctx.get("times", {}):
                time_value = ctx["times"][time_match]
                updated_text = updated_text.replace(time_match, time_value)
                print(f"🕐 infer time variable {time_match} -> {time_value}")
            elif time_match == "t1" and ctx.get("time_constraints"):
                first_time = list(ctx["time_constraints"].values())[0]
                updated_text = updated_text.replace(time_match, first_time)
                print(f"🕐 infer t1 -> {first_time}")
            elif time_match == "t1" and ctx.get("times"):
                first_time = list(ctx["times"].values())[0]
                updated_text = updated_text.replace(time_match, first_time)
                print(f"🕐 infer t1 -> {first_time}")
    
    confirmed_relation = None
    if step_num > 1:  # not is first subquestion
        # check if no have from first subquestion confirmed relation information
        first_answer = ctx.get("answers", {}).get("s1", {})
        if first_answer:
            extracted_relation = _extract_relation_from_answer(first_answer, ctx)
            if extracted_relation:
                confirmed_relation = extracted_relation
                print(f"🔗 Detected first subquestion relation: {confirmed_relation}，用于Update当TopSubquestion")
                
                # Update subquestion text to reflect relation replacement
                if "after" in updated_text.lower() and confirmed_relation:
                    print(f"🔄 consider Update subquestion text to reflect confirmed relation: {confirmed_relation}")
                    # recognize and replace text in use generic relation words as accurate confirmed relation
                    for generic_rel_word in ["action_relation", "perform", "do", "act", "task", "operation"]:
                        if generic_rel_word in updated_text.lower():
                            updated_text = updated_text.replace(generic_rel_word.lower(), confirmed_relation.lower())
                            break
    
    updated_edges = []
    for edge in subq.indicator.edges:
        # only keep contains top subquestion variables edges (like ?x, ?y etc.)
        #  or contains top time variables edges (like t2, t3 etc., but not including solved time variables)
        should_include = False
        
        # check if no contains top subquestion variables
        if edge.subj.startswith("?") or edge.obj.startswith("?"):
            should_include = True
        
        # check if no contains top time variables (not is solved time variables)
        current_time_vars = [var for var in subq.indicator.edges if hasattr(var, 'time_var')]
        if edge.time_var and edge.time_var not in ctx.get("times", {}):
            should_include = True
        
        if (edge.subj not in ctx.get("times", {}) and 
            edge.obj not in ctx.get("times", {}) and
            not edge.subj.startswith("?") and 
            not edge.obj.startswith("?")):
            if edge.time_var and edge.time_var not in ctx.get("times", {}):
                should_include = True
        
        if should_include:
            updated_edge = IndicatorEdge(
                subj=edge.subj,
                rel=edge.rel,
                obj=edge.obj,
                time_var=edge.time_var
            )
            
            if confirmed_relation:
                print(f"🔄 replace relation: {updated_edge.rel} -> {confirmed_relation}")
                updated_edge.rel = confirmed_relation
            
            # replace time variables
            if updated_edge.subj in ctx.get("times", {}):
                updated_edge.subj = ctx["times"][updated_edge.subj]
            if updated_edge.obj in ctx.get("times", {}):
                updated_edge.obj = ctx["times"][updated_edge.obj]
            
            updated_edges.append(updated_edge)
    
    # if no edges, keep original edges
    if not updated_edges:
        updated_edges = []
        for edge in subq.indicator.edges:
            updated_edge = IndicatorEdge(
                subj=edge.subj,
                rel=edge.rel,
                obj=edge.obj,
                time_var=edge.time_var
            )
            
            if confirmed_relation:
                print(f"🔄 replace relation: {updated_edge.rel} -> {confirmed_relation}")
                updated_edge.rel = confirmed_relation
            
            if updated_edge.subj in ctx.get("times", {}):
                updated_edge.subj = ctx["times"][updated_edge.subj]
            if updated_edge.obj in ctx.get("times", {}):
                updated_edge.obj = ctx["times"][updated_edge.obj]
            
            updated_edges.append(updated_edge)
    
    updated_indicator = Indicator(edges=updated_edges, constraints=subq.indicator.constraints)
    
    return SubQuestion(
        sid=subq.sid,
        text=updated_text,
        indicator=updated_indicator,
        depends_on=subq.depends_on
    )


def _extract_relation_from_answer(first_answer: Dict[str, Any], ctx: Dict[str, Any]) -> str:
    """
    """
    try:
        print(f"🔍 start from first subquestion answer in extract relation information, mainly from path field relation part")
        print(f"   answer structure: {list(first_answer.keys()) if first_answer else 'None'}")
        
        path = first_answer.get('path')
        if path and isinstance(path, list) and len(path) >= 2:
            relation = path[1]  # path[1]isrelation
            if isinstance(relation, str) and relation.strip() and relation != 'Unknown':
                print(f"📊 from first subquestion answer path field extract relation: {relation}")
                return relation.strip()
        
        proof = first_answer.get('proof', {})
        if isinstance(proof, dict) and 'relation' in proof:
            relation = proof.get('relation')
            if isinstance(relation, str) and relation.strip() and relation != 'Unknown':
                print(f"📊 fromfirstSubquestionanswerproof.relationextractrelation: {relation}")
                return relation.strip()
        
        # 优先方案2：fromcandidatesinfirst选候选path
        candidates = first_answer.get('candidates', [])
        if candidates:
            # checkfirst候选path
            first_candidate = candidates[0]
            path = first_candidate.get('path')
            if path and isinstance(path, list) and len(path) >= 2:
                relation = path[1]
                if isinstance(relation, str) and relation.strip() and relation != 'Unknown':
                    print(f"📊 fromfirstSubquestionanswerfirstcandidate.pathextractrelation: {relation}")
                    return relation.strip()
        
        trajectory = ctx.get('trajectory', [])
        if trajectory and len(trajectory) > 0:
            first_step = trajectory[0]
            first_result = first_step.get('result', {})
            if first_result:
                path = first_result.get('path')
                if path and isinstance(path, list) and len(path) >= 2:
                    relation = path[1]
                    if isinstance(relation, str) and relation.strip() and relation != 'Unknown':
                        print(f"📊 from trajectory first step result.path extract relation: {relation}")
                        return relation.strip()
                        
                # then check candidates
                candidates = first_result.get('candidates', [])
                if candidates:
                    first_candidate = candidates[0]  # select first candidate
                    path = first_candidate.get('path', [])
                    if path and isinstance(path, list) and len(path) >= 2:
                        relation = path[1]
                        if isinstance(relation, str) and relation.strip() and relation != 'Unknown':
                            print(f"📊 from trajectory first step candidates.path extract relation: {relation}")
                            return relation.strip()
        
        # 方案4：from provenance in extract relation (fallback)
        provenance = first_answer.get('provenance', {})
        if provenance:
            # checkisnohavemethod or parametersinformation
            method = provenance.get('method', '')
            if 'after_first' in method.lower() or 'direct' in method.lower():
                # from parameters or proof information in find relation
                proof = provenance.get('proof', {})
                if isinstance(proof, dict) and 'relation' in proof:
                    rel = proof.get('relation', '')
                    if isinstance(rel, str) and rel.strip():
                        print(f"📊 from provenance.proof extract relation: {rel}")
                        return rel.strip()
        
        print(f"⚠️ no from first subquestion answer extract valid relation")
        if first_answer:
            print(f"   first subquestion answer structure Details: {first_answer}")
        return None
        
    except Exception as e:
        print(f"❌ extract relation failed: {e}")
        return None


def refine_entity_selection_with_context(subq, topic_entities: List[Dict[str, Any]], 
                                       ctx: Dict[str, Any], max_entities: int = 15) -> List[Dict[str, Any]]:
    """
    refine entity select with abovebelow text information perform two times
    
    Args:
        subq: Subquestion
        topic_entities: Candidate entity list
        ctx: abovebelow text information
        max_entities: Maximum number of entities to keep
        
    Returns:
        refined after entity list
    """
    if not topic_entities:
        return []
    
    print(f"🔍 start abovebelow text two times refine: target count={max_entities}")
    
    # calculate each entity abovebelow text relevant number
    entity_scores = []
    
    for entity in topic_entities:
        entity_name = entity.get("name", "")
        base_score = entity.get("score", 0.0)  # original number
        context_score = 0.0
        
        # 1. check if no with Top answer relevant
        if ctx.get("answers"):
            for answer_key, answer_info in ctx["answers"].items():
                answer_entity = answer_info.get("entity", "")
                if answer_entity and entity_name:
                    # calculate with Topanswersemanticsimilar
                    similarity = calculate_semantic_similarity(entity_name, answer_entity)
                    context_score += similarity * 0.3
        
        # 2. checkisno with timeinformationrelevant
        if ctx.get("times"):
            for time_var, time_value in ctx["times"].items():
                if time_value in subq.text:
                    context_score += 0.2
        
        entity_type_score = calculate_entity_type_relevance(entity_name, subq.text)
        context_score += entity_type_score
        
        keyword_score = calculate_keyword_matching_score(entity_name, subq.text)
        context_score += keyword_score
        
        final_score = base_score + context_score
        
        entity_scores.append({
            'entity': entity,
            'name': entity_name,
            'base_score': base_score,
            'context_score': context_score,
            'final_score': final_score
        })
    
    entity_scores.sort(key=lambda x: x['final_score'], reverse=True)
    
    refined_entities = [item['entity'] for item in entity_scores[:max_entities]]
    
    print(f"📊 two times refine result statistics:")
    print(f"🏆 Top5 refine after entity:")
    for i, item in enumerate(entity_scores[:5]):
        print(f"  {i+1}. {item['name']} (final: {item['final_score']:.3f} = base: {item['base_score']:.3f} + abovebelow text: {item['context_score']:.3f})")
    
    return refined_entities


def calculate_entity_type_relevance(entity_name: str, subquestion: str) -> float:
    score = 0.0
    entity_lower = entity_name.lower()
    subq_lower = subquestion.lower()
    
    # person name relevant mode
    if any(pattern in subq_lower for pattern in ["who", "person", "people", "someone", "leader", "president", "minister"]):
        if any(indicator in entity_lower for indicator in ["_", "person", "president", "minister", "leader"]):
            score += 0.4
    
    # place name relevant mode
    if any(pattern in subq_lower for pattern in ["where", "country", "city", "place", "location"]):
        if any(indicator in entity_lower for indicator in ["country", "city", "state", "province"]):
            score += 0.4
    
    # organization relevant mode
    if any(pattern in subq_lower for pattern in ["organization", "company", "group", "party", "government"]):
        if any(indicator in entity_lower for indicator in ["organization", "party", "government", "company"]):
            score += 0.4
    
    # timerelevantmode
    if any(pattern in subq_lower for pattern in ["when", "time", "date", "year", "first", "last", "after", "before"]):
        score += 0.1  # time relevant questions，entity type not most important
    
    return score


def calculate_keyword_matching_score(entity_name: str, subquestion: str) -> float:
    """calculate keyword match number"""
    score = 0.0
    # normalize entity name：process below hyphen, special separator etc.
    # first normalize separator, again word
    normalized_entity = entity_name.lower().replace("_", " ").replace("-", " ")
    entity_words = set(normalized_entity.split())
    subq_words = set(subquestion.lower().split())
    
    # remove common stop words
    stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "from"}
    entity_words = entity_words - stop_words
    subq_words = subq_words - stop_words
    
    if not entity_words:
        return 0.0
    
    # calculate intersection ratio
    common_words = entity_words.intersection(subq_words)
    if common_words:
        score = len(common_words) / len(entity_words)
        # give completely match word higher number
        for word in common_words:
            if len(word) > 3:  # longer word more important
                score += 0.1
        full_match_name = " ".join(entity_words).replace(" ", "") if " " not in entity_words else entity_words
        original_entity_normalized = entity_name.lower().replace("_", "").replace("-", "").replace(" ", "")
        subq_normalized = subquestion.lower().replace(" ", "")
        if original_entity_normalized in subq_normalized:
            score += 0.2  # full name match reward    
        elif len(original_entity_normalized) > 8 and original_entity_normalized[:4] in subq_normalized:
            score += 0.1  # part Top suffix similar reward
            
    return min(score, 0.5)  # limit most large number


def intelligent_seed_selection(subq, topic_entities: List[Dict[str, Any]], ctx: Dict[str, Any], enhanced_prompt: str) -> List[int]:
    """
    intelligent select seed Entity, not limited to KG_Path, including single point extended KG_Path
    select标accurate:  
    1. based on Indicator edges and Subquestion text Entity match
    2. if no result, use LLM select most multiple two seeds
    """
    seeds = []
    filtered_entities = filter_relevant_entities_with_embedding(subq.text, topic_entities)
    original_entities = topic_entities
    print(f"Entity filtering: from {len(topic_entities)} Entity filtering to {len(filtered_entities)} relevant entities")
    
    if len(filtered_entities) == 1:
        return [filtered_entities[0]["id"]]
    topic_entities = filtered_entities
    
    # # 1. from Indicator edges in extract Entity as seed
    # for edge in subq.indicator.edges:
    #     subj = edge.subj
    #     obj = edge.obj
        
    #     # find Entity ID (exclude variable)
    #     for entity in topic_entities:
    #         entity_name = entity.get("name", "").lower()
    #         if (subj and subj != "?x" and subj != "?y" and subj.lower() == entity_name) or \
    #            (obj and obj != "?x" and obj != "?y" and obj.lower() == entity_name):
    #             if entity["id"] not in seeds:
    #                 seeds.append(entity["id"])
    
    # # 2. from Subquestion text in extract Entity
    # subq_text = subq.text.lower()
    # for entity in topic_entities:
    #     entity_name = entity.get("name", "").lower()
    #     # checkEntitynameisno inSubquestioninappear
    #     if entity_name in subq_text and entity["id"] not in seeds:
    #         seeds.append(entity["id"])
    
    # iffilterafter仍然have很multipleentity，perform二times精炼select
    if len(topic_entities) > 20:
        print(f"🎯 entity count still multiple({len(topic_entities)}), perform two times refine...")
        topic_entities = refine_entity_selection_with_context(subq, topic_entities, ctx)
        print(f"two times refine after keep {len(topic_entities)} entity")
    
    # 4. ifseedcountnot足，useLLMselect，mostmultipleretry3times
    if len(topic_entities) >1 or len(topic_entities) == 0:
        # print(f"seedcountnot足({len(seeds)})，attemptuseLLMselect...")
        if len(topic_entities) == 0:
            topic_entities = original_entities
        for attempt in range(3):
            try:
                print(f"LLMselectattempt {attempt + 1}/3")
                seed_append = llm_select_seeds(subq, topic_entities, max_seeds=2, enhanced_prompt=enhanced_prompt)
                if seed_append:
                    seeds.extend(seed_append)
                    seeds = list(set(seeds))  # deduplicate
                    print(f"LLMselectsuccess，obtained {len(seed_append)} seeds")
                    break
                else:
                    print(f"LLMselectattempt {attempt + 1} failed，return空result")
            except Exception as e:
                print(f"LLMselectattempt {attempt + 1} exception: {e}")
        
        if len(seeds) < 1:
            print("LLMselectfailed，execute3timesretry机制...")
            
            retry_count = 0
            max_retries = 3
            
            while retry_count < max_retries and len(seeds) < 1:
                retry_count += 1
                print(f"Seed Selectionretry {retry_count}/{max_retries}")
                
                try:
                    retry_seeds = llm_select_seeds_simplified(subq, topic_entities, max_seeds=2, enhanced_prompt=enhanced_prompt)
                    if retry_seeds:
                        seeds.extend(retry_seeds)
                        seeds = list(set(seeds))  # deduplicate
                        print(f"retry {retry_count} success，obtained {len(retry_seeds)} seeds")
                        break
                    else:
                        print(f"retry {retry_count} failed，继continuebelowonetimesretry")
                except Exception as e:
                    print(f"retry {retry_count} exception: {e}")
            
            if len(seeds) < 1:
                print("all retry failed，use top 3 high Entity as final backup")
                sorted_entities = sorted(topic_entities, key=lambda x: x.get("score", 0), reverse=True)
                top3_seeds = [entity["id"] for entity in sorted_entities[:3]]
                seeds.extend(top3_seeds)
                seeds = list(set(seeds))  # deduplicate
                print(f"Final use {len(top3_seeds)} high Entity as seed: {top3_seeds}")
    
    return seeds[:5]  # limitseedcount


def get_seeds_enhanced_prompt(question: str, question_type: str) -> str:
    """usetemplateenhance解questions"""

    try:
        classification_prompt = ""
        
        enhanced_prompt = ""
        print("✅ Seeds selection prompt enhanced successfully")
        return enhanced_prompt
    except Exception as e:
        print(f"❌ Classification test failed: {e}")
        return ""



def llm_select_seeds(subq, topic_entities: List[Dict[str, Any]], max_seeds: int = 3, enhanced_prompt: str = "None") -> List[int]:
    """
    useLLMselectseedEntity
    """
    try:
        # first attempt from enhance version unified knowledge storage get examples
        try:
            from .enhanced_unified_integration import get_seed_selection_enhanced
            # get question type (from abovebelow text or other place get)
            question_type = getattr(subq, 'question_type', None)
            if not question_type:
                # attempt from abovebelow text get
                question_type = None
            
            # use global experiment_setting
            global CURRENT_EXPERIMENT_SETTING
            examples = get_seed_selection_enhanced(
                given_subquestion=subq.text,
                topk=3,
                question_type=question_type,
                experiment_setting=CURRENT_EXPERIMENT_SETTING
            )
            
            if examples:
                print(f"✅ from unified knowledge storage get to {len(examples)} Seed Selection examples")
                # buildenhanceprompt
                enhanced_examples = "## Successful Examples for Seed Selection:\n\n"
                for i, example in enumerate(examples, 1):
                    enhanced_examples += f"Example {i}:\n"
                    enhanced_examples += f"Subquestion: {example['subquestion']}\n"
                    enhanced_examples += f"Entities: {example['entities']}\n"
                    enhanced_examples += f"Output: {example['output']}\n\n"
                
                # add enhance examples to original have prompt in
                enhanced_prompt = enhanced_examples + (enhanced_prompt if enhanced_prompt != "None" else "")
            else:
                print("📋 not foundsimilarSeed Selectionexamples")
        except Exception as e:
            print(f"⚠️ unified knowledge storage query failed: {e}, use original have prompt")
        
        # accurate backup simplified Entity list (only contains name and ID)
        ents = []
        for entity in topic_entities:
            ents.append(f"ID: {entity['id']}, Name: {entity['name']}")
        ents_text = "\n".join(ents)

        # call LLM
        response = LLM.call(LLM_SYSTEM_PROMPT, LLM_SEED_SELECT_PROMPT.format(subq=subq.text, entities=ents_text, enhanced_prompt=enhanced_prompt))
        
        # parse plain text result
        selected_ids = parse_seed_selection_response(response)
        
        # limit seed count and verify ID valid
        valid_ids = []
        for entity_id in selected_ids[:max_seeds]:
            # verify ID isn't in topic_entities in
            if any(entity["id"] == entity_id for entity in topic_entities):
                valid_ids.append(entity_id)
        
        return valid_ids
        
    except Exception as e:
        print(f"LLMSeed Selectionfailed: {e}")
        # if LLM failed，return empty list, let caller process
        return []


def llm_select_seeds_simplified(subq, topic_entities: List[Dict[str, Any]], max_seeds: int = 2, enhanced_prompt: str = "None") -> List[int]:
    """
    Simplified LLM Seed Selection method, for retry mechanism
    use simpler single prompt and fewer Entity information to improve high success rate
    """
    try:
        # only use top 10 Entity to reduce few LLM burden
        top_entities = sorted(topic_entities, key=lambda x: x.get("score", 0), reverse=True)[:10]
        
        # accurate backup simplified Entity list
        ents = []
        for entity in top_entities:
            ents.append(f"{entity['id']}: {entity['name']}")
        ents_text = "\n".join(ents)
        
        # use simpler prompt
        simplified_prompt = f"""
Given the question: "{subq.text}"
Select the most relevant entity IDs from the following list (maximum {max_seeds}):
for example:
{enhanced_prompt}
{ents_text}

Return only the IDs in format: [id1, id2]
"""
        
        # call LLM
        response = LLM.call(LLM_SYSTEM_PROMPT, simplified_prompt)
        
        # parse result
        selected_ids = parse_seed_selection_response(response)
        
        # limit seed count and verify ID valid
        valid_ids = []
        for entity_id in selected_ids[:max_seeds]:
            if any(entity["id"] == entity_id for entity in topic_entities):
                valid_ids.append(entity_id)
        
        return valid_ids
        
    except Exception as e:
        print(f"Simplified LLM Seed Selection failed: {e}")
        return []


def parse_seed_selection_response(response: str) -> List[int]:
    """
    parse LLM Seed Selection plain text response
    """
    try:
        # clean response text
        response = response.strip()
        
        # attempt direct parse as list format [1, 2]
        if response.startswith('[') and response.endswith(']'):
            # extract content in square brackets
            content = response[1:-1].strip()
            if not content:
                return []
            
            # split and parse ID
            ids = []
            for item in content.split(','):
                item = item.strip()
                try:
                    entity_id = int(item)
                    ids.append(entity_id)
                except ValueError:
                    continue
            return ids
        
        # attempt from text in extract number ID
        import re
        numbers = re.findall(r'\b(\d+)\b', response)
        ids = []
        for num in numbers:
            try:
                entity_id = int(num)
                ids.append(entity_id)
            except ValueError:
                continue
        return ids
        
    except Exception as e:
        print(f"Parse Seed Selection response failed: {e}")
        return []


def build_temporal_reasoning_path(reasoning_path: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    build Time increasing reasoning KG_Path
    """
    if not reasoning_path:
        return []
    
    # byTimeSort
    def parse_time(time_str):
        try:
            if "~" in time_str:
                time_str = time_str.split("~")[0]
            return time_str
        except:
            return "0000-00-00"
    
    sorted_path = sorted(reasoning_path, key=lambda x: parse_time(x.get("time", "")))
    
    # build complete reasoning KG_Path
    complete_path = []
    for i, edge in enumerate(sorted_path):
        path_info = {
            "step": i + 1,
            "subquestion": edge["subquestion"],
            "entity": edge["entity"],
            "time": edge["time"],
            "path": edge["path"],
            "provenance": edge["provenance"],
            "score": edge["score"],
            "LLM_selected_answer": edge["LLM_selected_answer"],
            "sufficiency_test": edge["sufficiency_test"]
        }
        # print(f"Edge: {edge}")
        complete_path.append(path_info)
    
    return complete_path

def generate_final_answer(original_question: str, reasoning_path: List[Dict[str, Any]]) -> str:
    """
    use complete reasoning KG_Path answer original questions
    """
    if not reasoning_path:
        return "No valid reasoning path found."
    
    # build reasoning KG_Path description
    path_description = []
    for step in reasoning_path:
        path_desc = f"Step {step['step']}: {step['entity']} at {step['time']}"
        if step.get("path"):
            # Handle new path format: [heads_list, relation, tails_list]
            if len(step['path']) > 1:
                relation = step['path'][1]
                path_desc += f" via {relation}"
            else:
                path_desc += " via unknown relation"
        path_description.append(path_desc)
    
    # use LLM generate final answer
    try:
        from .llm import LLM
        from .prompts import LLM_SYSTEM_PROMPT
        from ..config import TPKGConfig
        if TPKGConfig.DATASET == "TimeQuestions":
            inital_prompt = f"""
Based on the following reasoning path, answer the original question.
If Date or Time asking, It is only asking for the year.
            """
        else:
            inital_prompt = f"""
Based on the following reasoning path, answer the original question.
"""
        prompt = f"""


Original Question: {original_question}

Reasoning Path:
{chr(10).join(path_description)}

Please provide a clear and concise answer based on the reasoning path.
"""
        
        response = LLM.call(LLM_SYSTEM_PROMPT, prompt)
        return response.strip()
        
    except Exception as e:
        print(f"Error generating final answer: {e}")
        # Fallback: usemostafteroneAnswer
        last_step = reasoning_path[-1]
        return f"Based on the reasoning path: {last_step['entity']} at {last_step['time']}"


def record_successful_run(question: str, question_type: str, final_answer: str, trajectory: List[Dict[str, Any]], 
                         correct_answer: str = None) -> bool:
    """
    record success run and update template
    """
    try:
        # Template learnerremoved，skipverify
        
        # checkAnswerisnocorrect
        is_correct = False
        if correct_answer:
            # simple answer match check
            final_answer_lower = final_answer.lower()
            correct_answer_lower = correct_answer.lower()
            is_correct = correct_answer_lower in final_answer_lower or final_answer_lower in correct_answer_lower
        
        if is_correct:
            print(f"DetectedcorrectAnswer，recordsuccessmode...")
            
            # buildsuccess运行record
            successful_run = {
                'question': question,
                'question_type': question_type,
                'final_answer': final_answer,
                'correct_answer': correct_answer,
                'trajectory': trajectory,
                'timestamp': learner._get_timestamp()
            }
            
            # Updatetemplate
            learner.update_template_with_learning(question_type, successful_run)
            return True
        else:
            print(f"Answernotmatch，notrecordmode")
            return False
            
    except Exception as e:
        print(f"recordsuccess运行failed: {e}")
        return False




def get_toolkit_info() -> Dict[str, Any]:
    """getToolkitinformation"""
    from .kg_ops import KG
    return KG.get_available_tools()


def get_toolkit_methods_info() -> str:
    """getallToolkitmethoddetailedinformation"""
    methods_info = [
        "1. find_entities_by_name_pattern(pattern, limit=None)",
        "   - Find entities matching a name pattern",
        "   - Parameters: pattern (str), limit (int, optional)",
        "",
        "2. find_temporal_sequence(entity, relation, start_time, end_time)",
        "   - Find temporal sequence of events for an entity between two times",
        "   - Parameters: entity (str/int), relation (str), start_time (str), end_time (str)",
        "",
        "3. find_entities_after_time(time_point, limit=None)",
        "   - Find entities that appeared after a specific time",
        "   - Parameters: time_point (str), limit (int, optional)",
        "",
        "4. find_entities_before_time(time_point, limit=None)",
        "   - Find entities that appeared before a specific time",
        "   - Parameters: time_point (str), limit (int, optional)",
        "",
        "5. find_before_last(entity, reference_time, limit=None)",
        "   - Find the last entity to perform an action before a reference time",
        "   - Parameters: entity (str/int), reference_time (str), limit (int, optional)",
        "",
        "6. find_after_first(entity, reference_time, limit=None)",
        "   - Find the first entity to perform an action after a reference time",
        "   - Parameters: entity (str/int), reference_time (str), limit (int, optional)",
        "",
        "7. find_between_times(entity, start_time, end_time, limit=None)",
        "   - Find entities that performed actions between two times",
        "   - Parameters: entity (str/int), start_time (str), end_time (str), limit (int, optional)",
        "",
        "8. find_temporal_neighbors(entity, time_operation, limit=None)",
        "   - Find temporal neighbors of an entity",
        "   - Parameters: entity (str/int), time_operation (str), limit (int, optional)",
        "",
        "9. find_chronological_sequence(entities, relation, limit=None)",
        "   - Find chronological sequence of events for multiple entities",
        "   - Parameters: entities (list), relation (str), limit (int, optional)",
        "",
        "10. find_time_gaps(entity, min_gap_days=30, limit=None)",
        "    - Find time gaps in entity activities",
        "    - Parameters: entity (str/int), min_gap_days (int), limit (int, optional)",
        "",
        "11. events_on_day(date, limit=None)",
        "    - Find all events on a specific day",
        "    - Parameters: date (str), limit (int, optional)",
        "",
        "12. events_in_month(year, month, limit=None)",
        "    - Find all events in a specific month",
        "    - Parameters: year (int), month (int), limit (int, optional)",
        "",
        "13. find_entities(limit=None)",
        "    - Find all entities in the knowledge graph",
        "    - Parameters: limit (int, optional)",
        "",
        "14. entity_stats(entity)",
        "    - Get statistics for a specific entity",
        "    - Parameters: entity (str/int)"
    ]
    
    return "\n".join(methods_info)


def intelligent_toolkit_selection(subq, seeds: List[int], ctx: Dict[str, Any], question_type: str = None) -> Dict[str, Any]:
    """
    intelligentToolkit Selection：usenewintelligentToolkit Selection器
    """
    selector = get_intelligent_toolkit_selector()
    
    selected_toolkits = selector.select_toolkits(subq.text, seeds, ctx, question_type)
    
    if selected_toolkits:
        # select priority max Toolkit
        selected_toolkit = selected_toolkits[0]
        # print(f"intelligentselectToolkit: {selected_toolkit['toolkit_name']}")
        # print(f"select reason: {selected_toolkit['reasoning']}")
        
        return {
            "method_name": selected_toolkit['toolkit_name'],
            "parameters": selected_toolkit['parameters'],
            "description": f"agentice toolkit selection: {selected_toolkit['reasoning']}",
            "confidence": 0.9,
            "all_selections": selected_toolkits
        }
    else:
        print("agentice toolkit selection failed, using default toolkit...")
        return {
            "method_name": "retrieve_one_hop",
            "parameters": {"entity": seeds[0] if seeds else None},
            "description": "roll back to default toolkit: retrieve_one_hop",
            "confidence": 0.5
        }


def llm_select_toolkit(subq, seeds: List[int], ctx: Dict[str, Any]) -> Dict[str, Any]:
    """
    use LLM select most suitable Toolkit method
    """
    try:
        # first attempt from enhance version unified knowledge storage get examples
        enhanced_prompt = ""
        try:
            from .enhanced_unified_integration import get_toolkit_selection_enhanced
            # get question type (from abovebelow text or other place get)
            question_type = getattr(subq, 'question_type', None)
            if not question_type:
                question_type = None
            
            # buildIndicatorinformation
            indicator = {
                "edges": [{"subj": "?x", "rel": "unknown", "obj": "?y", "time_var": "t1"}],
                "constraints": []
            }
            
            # buildSeed info
            seed_info = []
            if seeds:
                for seed_id in seeds:
                    seed_info.append(f"ID: {seed_id}, Name: Entity_{seed_id}")
            
            # use global experiment_setting
            global CURRENT_EXPERIMENT_SETTING
            examples = get_toolkit_selection_enhanced(
                given_subquestion=subq.text,
                topk=10,
                question_type=question_type,
                similarity_threshold=0.1,
                experiment_setting=CURRENT_EXPERIMENT_SETTING
            )
            
            if examples:
                print(f"✅ from unified knowledge storage get to {len(examples)} Toolkit Selection examples")
                # buildenhanceprompt
                enhanced_examples = "## Successful Examples for Toolkit Selection:\n\n"
                for i, example in enumerate(examples, 1):
                    enhanced_examples += f"Example {i}:\n"
                    enhanced_examples += f"Subquestion: {example['subquestion']}\n"
                    enhanced_examples += f"Indicator: {example['indicator']}\n"
                    enhanced_examples += f"Seed_info: {example['seed_info']}\n"
                    enhanced_examples += f"Toolkit: {example['toolkit']}\n"
                    enhanced_examples += f"Parameters: {example['parameters']}\n"
                    # enhanced_examples += f"Context: {example['context']}\n"
                    enhanced_examples += f"Time_hints: {example['time_hints']}\n"
                    enhanced_examples += f"Reasoning: {example['reasoning']}\n\n"
                
                enhanced_prompt = enhanced_examples
            else:
                print("📋 not foundsimilarToolkit Selectionexamples")
        except Exception as e:
            print(f"⚠️ unified knowledge storage query failed: {e}, use original have prompt")
        
        # getToolkitmethodinformation
        toolkit_methods = get_toolkit_methods_info()
        
        # build abovebelow text information
        context_info = f"Times: {ctx.get('times', {})}, Answers: {list(ctx.get('answers', {}).keys())}"
        seeds_info = str(seeds) if seeds else "None"
        
        # call LLM
        prompt = LLM_TOOLKIT_SELECT_PROMPT.format(
            toolkit_methods=toolkit_methods,
            subquestion=subq.text,
            context=context_info,
            seeds=seeds_info,
            enhanced_prompt=enhanced_prompt
        )
        print(prompt)
        response = LLM.call(LLM_SYSTEM_PROMPT, prompt)
        print(response)
        # exit()
        # parse response
        toolkit_config = parse_toolkit_selection_response(response)
        
        return toolkit_config
        
    except Exception as e:
        print(f"LLMToolkit Selectionfailed: {e}")
        # return default configuration
        return {
            "method_name": "find_temporal_sequence",
            "parameters": {"entity": seeds[0] if seeds else None, "relation": "visit", "limit": 10},
            "description": "Default fallback method"
        }


def parse_toolkit_selection_response(response: str) -> Dict[str, Any]:
    """
    解析LLMToolkit Selection响应
    """
    try:
        # clean response text
        response = response.strip()
        
        # attempt parse [method_name, param1=value1, param2=value2, ...] format
        if response.startswith('[') and response.endswith(']'):
            content = response[1:-1].strip()
            if not content:
                raise ValueError("Empty method selection")
            
            # split parameters
            parts = [part.strip() for part in content.split(',')]
            if not parts:
                raise ValueError("No method specified")
            
            # first part is method name
            method_name = parts[0].strip()
            
            # parse parameters
            parameters = {}
            for part in parts[1:]:
                if '=' in part:
                    key, value = part.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # attempt convert data type
                    if value.lower() == 'none':
                        parameters[key] = None
                    elif value.isdigit():
                        parameters[key] = int(value)
                    elif value.startswith('[') and value.endswith(']'):
                        # listparameters
                        list_content = value[1:-1].strip()
                        if list_content:
                            parameters[key] = [item.strip() for item in list_content.split(',')]
                        else:
                            parameters[key] = []
                    else:
                        # string parameters, remove quotes
                        if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
                            value = value[1:-1]
                        parameters[key] = value
                else:
                    # no=parameters, maybe is location parameters
                    print(f"Warning: Ignoring parameter without '=': {part}")
            
            return {
                "method_name": method_name,
                "parameters": parameters,
                "description": f"Selected by LLM: {method_name}"
            }
        
        else:
            # attempt from text in extract method name
            import re
            method_match = re.search(r'(\w+)\(', response)
            if method_match:
                method_name = method_match.group(1)
                return {
                    "method_name": method_name,
                    "parameters": {"entity": None, "limit": 10},
                    "description": f"Extracted from text: {method_name}"
                }
            else:
                raise ValueError("Cannot parse method name from response")
        
    except Exception as e:
        print(f"Parse Toolkit Selection response failed: {e}")
        # return default configuration
        return {
            "method_name": "find_temporal_sequence",
            "parameters": {"entity": None, "relation": "visit", "limit": 10},
            "description": "Default fallback method"
        }


def execute_multiple_toolkit_requests(subq, seeds: List[int], ctx: Dict[str, Any], 
                                     toolkit_selections: List[Dict[str, Any]], 
                                     toolkit) -> Dict[str, Any]:
    """
    execute multiple Toolkit requests and perform debate vote
    """
    print(f"execute {len(toolkit_selections)} Toolkit请求...")
    
    # collect all Toolkit result
    toolkit_results = []
    all_candidates = []
    all_explanations = []
    successful_toolkits = []
    
    for i, selection in enumerate(toolkit_selections):
        try:
            method_name = selection['toolkit_name']
            original_name = selection.get('original_name', method_name)
            params = selection['parameters'].copy()
            
            # ensure seed entity parameters
            if seeds and 'entity' in params and params['entity'] is None:
                params['entity'] = seeds[0]
            elif seeds and 'entities' in params and params['entities'] is None:
                params['entities'] = seeds[:3]
            elif seeds and 'query' in params and params['query'] is None:
                params['query'] = seeds[0]
            
            # special handling for find_direct_connection method
            if method_name == "find_direct_connection":
                # ensure we have entity1 and entity2 parameters
                if 'entity1' in params and 'entity2' in params:
                    # Convert entity names to IDs if needed
                    from .kg_ops import KG
                    entity1_name = params['entity1']
                    entity2_name = params['entity2']
                    
                    # try to get entity IDs from names
                    entity1_id = KG.get_entity_id(entity1_name)
                    entity2_id = KG.get_entity_id(entity2_name)
                    
                    if entity1_id and entity2_id:
                        params['entity1'] = entity1_id
                        params['entity2'] = entity2_id
                        print(f"Direct connection search: {entity1_name}({entity1_id}) <-> {entity2_name}({entity2_id})")
                    else:
                        print(f"Warning: Could not find entity IDs for {entity1_name} or {entity2_name}")
                else:
                    print(f"Warning: find_direct_connection missing entity1 or entity2 parameters")
            
            # useunifiedIntelligent retrievalsystem
            print(f"🚀 useunifiedIntelligent retrievalsystem: {method_name}")
            # useIntelligent retrievalsystem
            intelligent_result = execute_intelligent_retrieval(subq, seeds, ctx, method_name, params, toolkit, True)
            
            # record Toolkit result for debate vote
            toolkit_result = {
                "toolkit_name": f"intelligent_{method_name}",
                "original_name": original_name,
                "parameters": params,
                "ok": intelligent_result.get('ok', False),
                "chosen": intelligent_result.get('chosen'),
                "candidates": intelligent_result.get('candidates', []),
                "explanations": intelligent_result.get('explanations', []),
                "top_paths": intelligent_result.get('top_paths', [])
            }
            toolkit_results.append(toolkit_result)
            
            if intelligent_result.get('ok'):
                all_candidates.extend(intelligent_result.get('candidates', []))
                all_explanations.extend(intelligent_result.get('explanations', []))
                successful_toolkits.append({
                    'toolkit_name': f"intelligent_{method_name}",
                    'original_name': original_name,
                    'parameters': params
                })
                print(f"Toolkit {i+1}/{len(toolkit_selections)}: intelligent_{method_name} Found {len(intelligent_result.get('candidates', []))} 候选")
            else:
                print(f"Toolkit {i+1}/{len(toolkit_selections)}: intelligent_{method_name} not found候选")
                
        except Exception as e:
            print(f"Toolkit {method_name} executefailed: {e}")
            # recordfailedToolkitresult
            toolkit_results.append({
                "toolkit_name": f"intelligent_{method_name}",
                "original_name": original_name,
                "parameters": params,
                "ok": False,
                "chosen": None,
                "candidates": [],
                "explanations": [f"executefailed: {e}"]
            })
            continue
    
    # performDebate Vote
    print(f"🗳️ startDebate Vote: {len(toolkit_results)} Toolkitresult")
    debate_system = DebateVoteSystem()
    vote_result = debate_system.conduct_debate_vote(subq.text, toolkit_results)
    
    # from vote result get winning answer
    winning_answer = vote_result.get("winning_answer", {})
    winning_toolkit = vote_result.get("winning_toolkit", 0)
    
    print(f"🏆 Debate Voteresult: winning Toolkit {winning_toolkit}")
    print(f"🎯 获胜answer: {winning_answer.get('entity', 'Unknown')} at {winning_answer.get('time', 'Unknown')}")
    
    # buildFinal answer
    chosen = None
    if winning_answer.get("entity") != "Unknown":
        # from winning Toolkit result in build chosen
        if winning_toolkit > 0 and winning_toolkit <= len(toolkit_results):
            winning_toolkit_result = toolkit_results[winning_toolkit - 1]
            if winning_toolkit_result.get("chosen"):
                chosen = winning_toolkit_result["chosen"]
                # update provenance information, add debate vote information
                chosen["provenance"]["debate_vote"] = {
                    "winning_toolkit": winning_toolkit,
                    "vote_reason": winning_answer.get("reason", ""),
                    "evaluation": vote_result.get("evaluation", {})
                }
            else:
                # if no chosen, from winning_answer build
                chosen = {
                    "entity": winning_answer.get("entity", "Unknown"),
                    "time": winning_answer.get("time", "Unknown"),
                    "path": winning_answer.get("path", []),
                    "provenance": {
                        "method": f"debate_vote_winner_{winning_toolkit}",
                        "similarity": winning_answer.get("score", 0),
                        "selection_reason": winning_answer.get("reason", ""),
                        "debate_vote": {
                            "winning_toolkit": winning_toolkit,
                            "vote_reason": winning_answer.get("reason", ""),
                            "evaluation": vote_result.get("evaluation", {})
                        }
                    }
                }
    
    # deduplicate and Sort all candidates (for display)
    unique_candidates = []
    seen_paths = set()
    for candidate in all_candidates:
        # process path in list, ensure can hash
        path = candidate["path"]
        if isinstance(path, list):
            # convert list in each element to string, then after create tuple
            path_key = tuple(str(item) for item in path)
        else:
            path_key = tuple(str(path))
        
        if path_key not in seen_paths:
            seen_paths.add(path_key)
            unique_candidates.append(candidate)
    
    # by priority Sort, then after by time Sort
    unique_candidates.sort(key=lambda x: (
        x["provenance"].get("priority", 999),
        x.get("time", "9999-12-31")
    ))
    
    print(f"merge after Found {len(unique_candidates)} unique candidates")
    
    # displayTop几候选result
    print("Top5 candidates result:")
    for i, candidate in enumerate(unique_candidates[:5]):
        # process path display, support list format
        path = candidate['path']
        if isinstance(path, list):
            path_str = ' -> '.join(str(item) for item in path)
        else:
            path_str = str(path)
        print(f"  {i+1}. {candidate['entity']} (Time: {candidate['time']}) - {path_str}")
    
    if chosen:
        # process chosen path display
        chosen_path = chosen['path']
        if isinstance(chosen_path, list):
            chosen_path_str = ' -> '.join(str(item) for item in chosen_path)
        else:
            chosen_path_str = str(chosen_path)
        print(f"🏆 Debate Voteselect candidate: {chosen['entity']} (Time: {chosen['time']}) - {chosen_path_str}")
        print(f"📊 winning Toolkit: {winning_toolkit}, reason: {winning_answer.get('reason', '')}")
    
    # Build answer
    answers = {}
    times = {}
    if chosen:
        if subq.indicator.edges and chosen.get("time"):
            time_var = subq.indicator.edges[0].time_var
            times[time_var] = chosen.get("time")
        
        answers[subq.sid] = {
            "entity": chosen.get("entity"),
            "time": chosen.get("time"),
            "score": chosen.get("provenance", {}).get("similarity", 0.9),
            "reason": f"Selected by debate vote (toolkit {winning_toolkit})",
            "proof": chosen.get("provenance", {}).get("proof", {}),
            "selection_reason": chosen.get("provenance", {}).get("selection_reason", ""),
            "debate_vote": chosen.get("provenance", {}).get("debate_vote", {})
        }
    
        # collect all Toolkit top_paths information
        all_top_paths = []
        for toolkit_result in toolkit_results:
            if toolkit_result.get("ok") and "top_paths" in toolkit_result:
                all_top_paths.extend(toolkit_result.get("top_paths", []))
        
        # deduplicate and keep Top3 most relevant path
        unique_top_paths = []
        seen_paths = set()
        for path in all_top_paths:
            if isinstance(path, dict):
                path_key = f"{path.get('heads_str', '')}->{path.get('relation', '')}->{path.get('tail', '')}"
                if path_key not in seen_paths:
                    seen_paths.add(path_key)
                    unique_top_paths.append(path)
                    if len(unique_top_paths) >= 3:
                        break
        
        return {
            "ok": bool(chosen),
            "chosen": chosen,
            "candidates": unique_candidates,
            "answers": answers,
            "times": times,
            "explanations": all_explanations,
            "toolkit_config": {
                "method_name": "debate_vote_multiple_toolkits",
                "parameters": {"selections": successful_toolkits, "vote_result": vote_result},
                "description": f"use {len(successful_toolkits)} Toolkit，通debate voteselectmost佳answer"
            },
            "debate_vote_result": vote_result,  # adddebate voteresult
            "top_paths": unique_top_paths  # addtop3pathinformation
        }


def execute_after_first_smart_search(subq, seeds: List[int], ctx: Dict[str, Any], toolkit) -> Dict[str, Any]:
    """
    according to after_first template execute Intelligent retrieval
    """
    try:
        print(f"🚀 according to after_first template execute Intelligent retrieval, Subquestion: {subq.sid}")
        
        # according to Subquestion type select Toolkit method
        if subq.sid == "s1":
            # first Subquestion：find reference Time
            print("executes1: find reference Time")
            return execute_find_reference_time(subq, seeds, ctx, toolkit)
        elif subq.sid == "s2":
            # second Subquestion：find first after - use Intelligent retrieval
            print("executes2: use Intelligent retrieval find first after")
            return execute_intelligent_after_first_search(subq, seeds, ctx, toolkit)
        else:
            # according to Subquestion content determine type
            if "when did" in subq.text.lower() or "when" in subq.text.lower():
                print("according to content determine ass1 type: find reference Time")
                return execute_find_reference_time(subq, seeds, ctx, toolkit)
            elif "after" in subq.text.lower() and "first" in subq.text.lower():
                print("according to content determine ass2 type: use Intelligent retrieval find first after")
                return execute_intelligent_after_first_search(subq, seeds, ctx, toolkit)
            else:
                print(f"no determine Subquestion type: {subq.sid}, fallback to basic retrieval")
                return execute_indicator(subq, seeds, ctx)
            
    except Exception as e:
        print(f"intelligent after_first retrieval failed: {e}")
        # Fallback to basic retrieval
        return execute_indicator(subq, seeds, ctx)


def execute_intelligent_after_first_search(subq, seeds: List[int], ctx: Dict[str, Any], toolkit) -> Dict[str, Any]:
    """
    use Intelligent retrieval system execute after_first search
    """
    try:
        # getIntelligent retrievalinstance
        intelligent_retrieval = get_intelligent_retrieval()
        
        # determineentity
        entity = seeds[0] if seeds else None
        if not entity:
            print("No entity specified for intelligent retrieval")
            return {"ok": False, "candidates": [], "answers": {}, "times": {}}
        
        # getentityname
        entity_name = KG.get_entity_name(entity)
        if not entity_name:
            print(f"nogetentityname: {entity}")
            return {"ok": False, "candidates": [], "answers": {}, "times": {}}
        
        # determinetimeconstraints
        time_constraint = None
        if ctx.get('times'):
            time_vars = list(ctx['times'].keys())
            if time_vars:
                time_constraint = ctx['times'][time_vars[0]]
        
        if not time_constraint:
            print("No time constraint found for intelligent retrieval")
            return {"ok": False, "candidates": [], "answers": {}, "times": {}}
        
        # buildSubquestion text
        subquestion_text = subq.text
        
        print(f"🚀 Intelligent retrieval: {entity_name} | {subquestion_text}")
        print(f"⏰ timeconstraints: {time_constraint} (after)")
        
        # executeIntelligent retrieval
        result = intelligent_retrieval.intelligent_retrieve(
            entity=entity_name,
            subquestion=subquestion_text,
            time_constraint=time_constraint,
            constraint_type="after"
        )
        
        if "error" in result:
            print(f"Intelligent retrievalfailed: {result['error']}")
            return {"ok": False, "candidates": [], "answers": {}, "times": {}}
        
        # processresult - keep with original have format compatible
        candidates = []
        if "selected_path" in result:
            selected_path = result["selected_path"]
            candidate = {
                "entity": selected_path.get('tail', 'Unknown'),
                "time": selected_path.get('time_start', 'Unknown'),
                "path": [
                    selected_path.get('head', 'Unknown'),
                    selected_path.get('relation', 'Unknown'),
                    selected_path.get('tail', 'Unknown')
                ],
                "provenance": {
                    "method": "intelligent_after_first",
                    "parameters": {"entity": entity_name, "time_constraint": time_constraint},
                    "similarity": selected_path.get('similarity', 0),
                    "selection_reason": selected_path.get('selection_reason', '')
                }
            }
            candidates.append(candidate)
        
        # buildanswer - keep with original have format compatible
        chosen = candidates[0] if candidates else None
        times = {}
        answers = {}
        
        if chosen:
            # Extract time information
            if subq.indicator.edges and chosen.get("time"):
                time_var = subq.indicator.edges[0].time_var
                times[time_var] = chosen.get("time")
            
            # Build answer
            answers[subq.sid] = {
                "entity": chosen.get("entity"),
                "time": chosen.get("time"),
                "score": chosen.get("provenance", {}).get("similarity", 0.9),
                "reason": f"Selected by intelligent after_first search"
            }
        
        print(f"✅ Intelligent retrievalcomplete: Found {len(candidates)} candidates")
        if chosen:
            print(f"select: {chosen.get('entity')} at {chosen.get('time')}")
            print(f"🎯 similar: {chosen.get('provenance', {}).get('similarity', 0):.3f}")
        
        return {
            "ok": bool(candidates),
            "chosen": chosen,
            "candidates": candidates,
            "answers": answers,
            "times": times,
            "explanations": ["Used intelligent after_first search with semantic pruning"],
            "toolkit_config": {
                "method_name": "intelligent_after_first",
                "description": "Intelligent after_first search with semantic pruning and LLM selection",
                "parameters": {"entity": entity_name, "time_constraint": time_constraint}
            }
        }
        
    except Exception as e:
        print(f"intelligent after_first retrieval execute failed: {e}")
        import traceback
        traceback.print_exc()
        # Fallback to basic retrieval
        return execute_indicator(subq, seeds, ctx)

def execute_find_reference_time(subq, seeds: List[int], ctx: Dict[str, Any], toolkit) -> Dict[str, Any]:
    """
    execute find reference Time (s1)
    """
    try:
        print("use basic 1-hop retrieval find reference Time...")
        
        # getseedEntityname
        entity_names = []
        for seed_id in seeds:
            entity_name = KG.get_entity_name(seed_id)
            if entity_name:
                entity_names.append(entity_name)
        
        if not entity_names:
            print("nogetEntityname")
            return {"ok": False, "error": "nogetEntityname"}
        
        print(f"retrieve Entity: {entity_names}")
        
        # use基本search
        all_candidates = []
        for entity_name in entity_names:
            try:
                result = KG.retrieve_paths_for_entity(entity_name)
                if result:
                    for path in result:
                        candidate = {
                            "entity": path.get("tail", "Unknown"),
                            "time": path.get("time", "Unknown"),
                            "path": [[path.get("head", "Unknown")], path.get("relation", "Unknown"), [path.get("tail", "Unknown")]],  # newformat: [heads_list, relation, tails_list]
                            "provenance": {"method": "basic_1hop", "source_entity": entity_name}
                        }
                        all_candidates.append(candidate)
            except Exception as e:
                print(f"retrieve {entity_name} failed: {e}")
        
        print(f"Found {len(all_candidates)} candidates KG_Path")
        
        # deduplicate
        unique_candidates = []
        seen_paths = set()
        for candidate in all_candidates:
            path_key = tuple(candidate["path"])
            if path_key not in seen_paths:
                seen_paths.add(path_key)
                unique_candidates.append(candidate)
        
        print(f"deduplicate after remaining {len(unique_candidates)} candidates KG_Path")
        
        target_entity = None
        
        if target_entity:
            print(f"priority select contains {target_entity} KG_Path...")
            target_candidates = [c for c in unique_candidates if target_entity in str(c["path"])]
            if target_candidates:
                print(f"Found {len(target_candidates)} contains {target_entity} candidates")
                unique_candidates = target_candidates
        
        # applyBERTfilter and LLMselect
        if len(unique_candidates) > 40:
            print("candidates KG_Path multiple, apply BERT filter...")
            # convert candidates to BERT filter format - support newpath structure [heads_list, relation, tails_list]
            bert_paths = []
            for candidate in unique_candidates:
                path = candidate["path"]
                if len(path) >= 2:
                    # process newpath structure
                    heads = path[0] if isinstance(path[0], list) else [path[0]]
                    relation = path[1]
                    tails = path[2] if len(path) > 2 and isinstance(path[2], list) else [path[2] if len(path) > 2 else 'Unknown']
                    
                    # each head-tail combination Create bert_path
                    for head in heads:
                        for tail in tails:
                            bert_path = {
                                "head": head,
                                "relation": relation, 
                                "tail": tail
                            }
                            bert_paths.append(bert_path)
            
            if bert_paths:
                filtered_bert_paths = KG.bert_filter_paths(bert_paths, {"text": subq.text})
                # convert filter after result to candidates format - support newpath structure
                filtered_candidates = []
                for bert_path in filtered_bert_paths:
                    for candidate in unique_candidates:
                        path = candidate["path"]
                        if len(path) >= 2:
                            # process newpath structure
                            heads = path[0] if isinstance(path[0], list) else [path[0]]
                            relation = path[1]
                            tails = path[2] if len(path) > 2 and isinstance(path[2], list) else [path[2] if len(path) > 2 else 'Unknown']
                            
                            # checkisnomatch
                            if (bert_path["head"] in heads and 
                                bert_path["relation"] == relation and
                                bert_path["tail"] in tails):
                                candidate["bert_score"] = bert_path.get("bert_score", 0.0)
                                filtered_candidates.append(candidate)
                                break
            else:
                filtered_candidates = unique_candidates
        else:
            filtered_candidates = unique_candidates
        
        # LLMselect
        if len(filtered_candidates) > 6:
            print("applyLLMselect...")
            # convert candidates to LLM select format
            llm_paths = []
            for candidate in filtered_candidates:
                path = candidate["path"]
                if len(path) >= 2:
                    # process newpath structure
                    heads = path[0] if isinstance(path[0], list) else [path[0]]
                    relation = path[1]
                    tails = path[2] if len(path) > 2 and isinstance(path[2], list) else [path[2] if len(path) > 2 else 'Unknown']
                    
                    # each head-tail combination Create llm_path
                    for head in heads:
                        for tail in tails:
                            llm_path = {
                                "head": head,
                                "relation": relation,
                                "tail": tail,
                                "time": candidate.get("time", "Unknown"),
                                "entity": candidate.get("entity", "Unknown")
                            }
                            llm_paths.append(llm_path)
            
            if llm_paths:
                selected_llm_paths = KG.llm_batch_select(llm_paths, subq.text, ctx)
                selected_candidates = []
                for llm_path in selected_llm_paths:
                    for candidate in filtered_candidates:
                        path = candidate["path"]
                        if len(path) >= 2:
                            heads = path[0] if isinstance(path[0], list) else [path[0]]
                            relation = path[1]
                            tails = path[2] if len(path) > 2 and isinstance(path[2], list) else [path[2] if len(path) > 2 else 'Unknown']
                            
                            # checkisnomatch
                            if (llm_path["head"] in heads and 
                                llm_path["relation"] == relation and
                                llm_path["tail"] in tails):
                                selected_candidates.append(candidate)
                                break
            else:
                selected_candidates = filtered_candidates
        else:
            selected_candidates = filtered_candidates
        
        # buildresult
        chosen = None
        times = {}
        answers = {}
        
        if selected_candidates:
            # Select first candidate answer
            chosen = selected_candidates[0]
            
            # Extract time information
            if subq.indicator.edges and chosen.get("time"):
                time_var = subq.indicator.edges[0].time_var
                times[time_var] = chosen.get("time")
            
            # Build answer
            answers[subq.sid] = {
                "entity": chosen.get("entity"),
                "time": chosen.get("time"),
                "score": 0.9,
                "reason": f"Selected by reference time search"
            }
        
        return {
            "ok": bool(selected_candidates),
            "chosen": chosen,
            "candidates": selected_candidates,
            "answers": answers,
            "times": times,
            "explanations": [f"Used reference time search"],
            "toolkit_config": {"method_name": "basic_1hop", "parameters": {"entities": entity_names}}
        }
        
    except Exception as e:
        print(f"find reference Time failed: {e}")
        return {"ok": False, "error": str(e)}


def execute_find_first_after(subq, seeds: List[int], ctx: Dict[str, Any], toolkit) -> Dict[str, Any]:
    """
    execute find first after (s2)
    """
    try:
        print("use Time Toolkit find first after...")
        
        # get参考Time
        reference_time = None
        if ctx.get("times"):
            time_vars = list(ctx["times"].keys())
            if time_vars:
                reference_time = ctx["times"][time_vars[0]]
        
        if not reference_time:
            print("no Found reference Time, fallback to basic retrieval")
            return execute_find_reference_time(subq, seeds, ctx, toolkit)
        
        print(f"reference Time: {reference_time}")
        
        # getseedEntityname, but priority select target Entity
        entity_names = []
        target_entity_name = None
        
        
        for seed_id in seeds:
            entity_name = KG.get_entity_name(seed_id)
            if entity_name:
                if target_entity_name and entity_name == target_entity_name:
                    # 优先add目标Entity
                    entity_names.insert(0, entity_name)
                else:
                    entity_names.append(entity_name)
        
        if not entity_names:
            print("nogetEntityname")
            return {"ok": False, "error": "nogetEntityname"}
        
        print(f"retrieve Entity: {entity_names}")
        
        all_candidates = []
        toolkit = KG.get_toolkit()
        
        # priority retrieve target Entity
        if target_entity_name and target_entity_name in entity_names:
            print(f"priority retrieve target Entity: {target_entity_name}")
            entity_names = [target_entity_name] + [e for e in entity_names if e != target_entity_name]
        
        for entity_name in entity_names:
            try:
                # use find_after_first Toolkit, only provide entity name
                if hasattr(toolkit, 'find_after_first'):
                    result = toolkit.find_after_first(
                        entity=entity_name,
                        reference_time=reference_time
                    )
                    if result and hasattr(result, 'edges'):
                        for edge in result.edges:
                            candidate = {
                                "entity": getattr(edge, 'tail', 'Unknown'),
                                "time": f"{getattr(edge, 'time_start', 'Unknown')}~{getattr(edge, 'time_end', 'Unknown')}" if getattr(edge, 'time_start', '') != getattr(edge, 'time_end', '') else getattr(edge, 'time_start', 'Unknown'),
                                "path": [getattr(edge, 'head', 'Unknown'), getattr(edge, 'relation', 'Unknown'), getattr(edge, 'tail', 'Unknown')],
                                "provenance": {"method": "find_after_first", "source_entity": entity_name, "reference_time": reference_time}
                            }
                            all_candidates.append(candidate)
                        print(f"TimeToolkitas {entity_name} Found {len(result.edges)} candidates")
                    else:
                        print(f"TimeToolkitas {entity_name} noFoundresult")
                else:
                    print("TimeToolkitnot available, fallback to basic retrieval")
                    # Fallback to basic retrieval
                    result = KG.retrieve_paths_for_entity(entity_name)
                    if result:
                        for path in result:
                            candidate = {
                                "entity": path.get("tail", "Unknown"),
                                "time": path.get("time", "Unknown"),
                                "path": [path.get("head", "Unknown"), path.get("relation", "Unknown"), path.get("tail", "Unknown")],
                                "provenance": {"method": "fallback_1hop", "source_entity": entity_name}
                            }
                            all_candidates.append(candidate)
            except Exception as e:
                print(f"TimeToolkit retrieve {entity_name} failed: {e}, fallback to basic retrieval")
                # Fallback to basic retrieval
                result = KG.retrieve_paths_for_entity(entity_name)
                if result:
                    for path in result:
                        candidate = {
                            "entity": path.get("tail", "Unknown"),
                            "time": path.get("time", "Unknown"),
                            "path": [path.get("head", "Unknown"), path.get("relation", "Unknown"), path.get("tail", "Unknown")],
                            "provenance": {"method": "fallback_1hop", "source_entity": entity_name}
                        }
                        all_candidates.append(candidate)
        
        print(f"Found {len(all_candidates)} candidates KG_Path")
        
        # deduplicate
        unique_candidates = []
        seen_paths = set()
        for candidate in all_candidates:
            path_key = tuple(candidate["path"])
            if path_key not in seen_paths:
                seen_paths.add(path_key)
                unique_candidates.append(candidate)
        
        print(f"deduplicate after remaining {len(unique_candidates)} candidates KG_Path")
        
        # applyBERTfilter and LLMselect
        if len(unique_candidates) > 40:
            print("candidates KG_Path multiple, apply BERT filter...")
            # convert candidates to BERT filter format
            bert_paths = []
            for candidate in unique_candidates:
                path = candidate["path"]
                if len(path) >= 3:
                    bert_path = {
                        "head": path[0],
                        "relation": path[1], 
                        "tail": path[2]
                    }
                    bert_paths.append(bert_path)
            
            if bert_paths:
                filtered_bert_paths = KG.bert_filter_paths(bert_paths, {"text": subq.text})
                # convert filter after result to candidates format
                filtered_candidates = []
                for bert_path in filtered_bert_paths:
                    for candidate in unique_candidates:
                        if (candidate["path"][0] == bert_path["head"] and 
                            candidate["path"][1] == bert_path["relation"] and
                            candidate["path"][2] == bert_path["tail"]):
                            candidate["bert_score"] = bert_path.get("bert_score", 0.0)
                            filtered_candidates.append(candidate)
                            break
            else:
                filtered_candidates = unique_candidates
        else:
            filtered_candidates = unique_candidates
        
        # LLMselect
        if len(filtered_candidates) > 6:
            print("applyLLMselect...")
            # convert candidates to LLM select format
            llm_paths = []
            for candidate in filtered_candidates:
                path = candidate["path"]
                if len(path) >= 2:
                    # process newpath structure
                    heads = path[0] if isinstance(path[0], list) else [path[0]]
                    relation = path[1]
                    tails = path[2] if len(path) > 2 and isinstance(path[2], list) else [path[2] if len(path) > 2 else 'Unknown']
                    
                    # each head-tail combination Create llm_path
                    for head in heads:
                        for tail in tails:
                            llm_path = {
                                "head": head,
                                "relation": relation,
                                "tail": tail,
                                "time": candidate.get("time", "Unknown"),
                                "entity": candidate.get("entity", "Unknown")
                            }
                            llm_paths.append(llm_path)
            
            if llm_paths:
                selected_llm_paths = KG.llm_batch_select(llm_paths, subq.text, ctx)
                # convert select result to candidates format
                selected_candidates = []
                for llm_path in selected_llm_paths:
                    for candidate in filtered_candidates:
                        path = candidate["path"]
                        if len(path) >= 2:
                            # process newpath structure
                            heads = path[0] if isinstance(path[0], list) else [path[0]]
                            relation = path[1]
                            tails = path[2] if len(path) > 2 and isinstance(path[2], list) else [path[2] if len(path) > 2 else 'Unknown']
                            
                            # checkisnomatch
                            if (llm_path["head"] in heads and 
                                llm_path["relation"] == relation and
                                llm_path["tail"] in tails):
                                selected_candidates.append(candidate)
                                break
            else:
                selected_candidates = filtered_candidates
        else:
            selected_candidates = filtered_candidates
        
        # buildresult
        chosen = None
        times = {}
        answers = {}
        
        if selected_candidates:
            # Select first candidate answer
            chosen = selected_candidates[0]
            
            # Extract time information
            if subq.indicator.edges and chosen.get("time"):
                time_var = subq.indicator.edges[0].time_var
                times[time_var] = chosen.get("time")
            
            # Build answer
            answers[subq.sid] = {
                "entity": chosen.get("entity"),
                "time": chosen.get("time"),
                "score": 0.9,
                "reason": f"Selected by first_after search"
            }
        
        return {
            "ok": bool(selected_candidates),
            "chosen": chosen,
            "candidates": selected_candidates,
            "answers": answers,
            "times": times,
            "explanations": [f"Used first_after search with reference time {reference_time}"],
            "toolkit_config": {"method_name": "find_after_first", "parameters": {"entities": entity_names, "reference_time": reference_time}}
        }
        
    except Exception as e:
        print(f"find first after failed: {e}")
        return {"ok": False, "error": str(e)}


def test_answer_sufficiency(subquestion: str, current_answer: Dict[str, Any], 
                          retrieved_info: List[Dict[str, Any]], 
                          context: Dict[str, Any], 
                          previous_subquestions: List[Dict[str, Any]] = None,
                          toolkit_info: Dict[str, Any] = None,
                          debate_vote_result: Dict[str, Any] = None,
                          top_paths: List[Dict[str, Any]] = None,
                          experiment_setting = None) -> Dict[str, Any]:
    """
    test Subquestion Answersufficientity
    
    Args:
        subquestion: Subquestion text
        current_answer: TopAnswer information
        retrieved_info: retrieve to information
            context: abovebelow text information
        previous_subquestions: TopSubquestion and Answer
        toolkit_info: use Toolkit information
        debate_vote_result: Debate voteresultinformation
            top_paths: LLMpathselectToptop3path
        experiment_setting: experiment setting
    
    Returns:
        sufficientity test result
    """
    try:
        # build evidence path information - only display LLM select one paths
        evidence_paths = []
        
        # from current_answer in extract LLM select path information
        answer_proof = current_answer.get('proof', {})
        if answer_proof:

            heads = answer_proof.get('heads', [])
            tails = answer_proof.get('tails', [])
            relation = answer_proof.get('relation', '')
            tail_str = answer_proof.get('tail_str', '')
            heads_count = answer_proof.get('heads_count', 1)
            tail_count = answer_proof.get('tail_count', 1)
            heads_str = ', '.join(heads) if len(heads) > 1 else heads[0] if heads else ''
            tails_str = ', '.join(tails) if len(tails) > 1 else tails[0] if tails else ''
            
            if heads_str and relation and tail_str:
                # build selected path string
                if len(heads) > 1 or len(tails) > 1:
                    path_str = f"[{heads_str}] - {relation} - [{tail_str}] (Multiple events happend simultaneously, Consider them same)"
                else:
                    path_str = f"[{heads_str}] - {relation} - [{tail_str}]"
                
                answer_entity = current_answer.get('entity', 'Unknown')
                answer_time = current_answer.get('time', 'Unknown')
                evidence_paths.append(f"{answer_entity} ({answer_time}): {path_str}")
        
        # if no proof information, attempt from retrieved_info in Found selected path
        if not evidence_paths and retrieved_info:
            # assume first is most important (selected by LLM)
            info = retrieved_info[0]
            entity = info.get('entity', 'Unknown')
            time = info.get('time', 'Unknown')
            path = info.get('path', [])
            
            if len(path) >= 2:
                # process newpath structure: [heads_list, relation, tails_list]
                heads = path[0] if isinstance(path[0], list) else [path[0]]
                relation = path[1]
                tails = path[2] if len(path) > 2 and isinstance(path[2], list) else [path[2] if len(path) > 2 else 'Unknown']
                
                # build path string, display aggregated information
                heads_str = ', '.join(heads) if len(heads) > 1 else heads[0]
                tails_str = ', '.join(tails) if len(tails) > 1 else tails[0]
                
                if len(heads) > 1 or len(tails) > 1:
                    path_str = f"[{heads_str}] - {relation} - [{tails_str}] (Multiple events happend simultaneously, Consider them same)"
                else:
                    path_str = f"{heads_str} - {relation} - {tails_str}"
                
                evidence_paths.append(f"{entity} ({time}): {path_str}")
        
        retrieved_text = "\n".join(evidence_paths) if evidence_paths else "No evidence paths found"
        
        # build concise answer information - support aggregated answer
        answer_entity = current_answer.get('entity', 'Unknown')
        answer_time = current_answer.get('time', 'Unknown')
        candidate_count = current_answer.get('candidate_count', 1)
        aggregated_entities = current_answer.get('aggregated_entities', None)
        
        # fromproofinextractpathinformation
        answer_proof = current_answer.get('proof', {})
        if answer_proof:
            heads_str = answer_proof.get('heads_str', '')
            relation = answer_proof.get('relation', '')
            tail = answer_proof.get('tail', '')
            heads_count = answer_proof.get('heads_count', 1)
            tail_count = answer_proof.get('tail_count', 1)
            
            if heads_str and relation and tail:
                # check if no have aggregated information
                if heads_count > 1 or tail_count > 1:
                    
                    answer_text = f"{answer_entity} ({answer_time})\nPath: {heads_str} - {relation} - {tail}\nNote: Multiple events happend simultaneously, Consider them same"
                else:
                    answer_text = f"{answer_entity} ({answer_time})\nPath: {heads_str} - {relation} - {tail}"
            else:
                # if no proof information, but have multiple candidates, display aggregated information
                if candidate_count > 1 and aggregated_entities:
                    answer_text = f"{answer_entity} ({answer_time})\nNote: Multiple events happend simultaneously, Consider them same"
                else:
                    answer_text = f"{answer_entity} ({answer_time})"
        else:
            # if no proof information, but have multiple candidates, display aggregated information
            if candidate_count > 1 and aggregated_entities:
                answer_text = f"{answer_entity} ({answer_time})\nNote: Multiple events happend simultaneously, Consider them same"
            else:
                answer_text = f"{answer_entity} ({answer_time})"
        
        # build abovebelow text summary
        context_text = f"solve Subquestion: {list(context.get('answers', {}).keys())}, Time variables: {list(context.get('times', {}).keys())}"
        
        # build concise TopStep summary
        previous_subq_text = "None"
        if previous_subquestions:
            previous_summary = []
            for i, prev_subq in enumerate(previous_subquestions, 1):
                prev_text = prev_subq.get('text', 'Unknown')
                prev_answer = prev_subq.get('answer', {})
                prev_entity = prev_answer.get('entity', 'Unknown')
                prev_time = prev_answer.get('time', 'Unknown')
                previous_summary.append(f"Step {i}: {prev_text}\nAnswer: {prev_entity} ({prev_time})")
            previous_subq_text = "\n\n".join(previous_summary)
        
        # buildDebate Voteresultinformation
        debate_vote_text = "No debate vote information"
        if debate_vote_result:
            winning_toolkit = debate_vote_result.get('winning_toolkit', 'Unknown')
            winning_answer = debate_vote_result.get('winning_answer', {})
            evaluation = debate_vote_result.get('evaluation', {})
            reasoning = evaluation.get('reasoning', 'No reasoning provided')
            
            debate_vote_text = f"Winning Toolkit: {winning_toolkit}\nWinning Answer: {winning_answer.get('entity', 'Unknown')} ({winning_answer.get('time', 'Unknown')})\nReasoning: {reasoning}"
        
        # buildTop 3pathinformation - support aggregated path, record time, not record similar
        top_paths_text = "No top paths information"
        if top_paths:
            paths_summary = []
            for i, path in enumerate(top_paths[:3], 1):
                if isinstance(path, dict):
                    # support newpath structure fields
                    heads_str = path.get('heads_str', path.get('head', 'Unknown'))
                    relation = path.get('relation', 'Unknown')
                    tail = path.get('tail', 'Unknown')
                    tails_str = path.get('tails_str', '')
                    time_start = path.get('time_start', 'Unknown')
                    count = path.get('count', 1)
                    head_count = path.get('head_count', 1)
                    tail_count = path.get('tail_count', 1)
                    
                    # build path description, display aggregated information and time, not display similar
                    if head_count > 1 or tail_count > 1:                        
                        # use tails_str if available, else use tail
                        tail_display = tails_str if tails_str else tail
                        path_desc = f"Path {i}: {heads_str} - {relation} - {tail_display} at {time_start} (Multiple events happend simultaneously, Consider them same)"
                    else:
                        path_desc = f"Path {i}: {heads_str} - {relation} - {tail} at {time_start}"
                    paths_summary.append(path_desc)
            top_paths_text = "\n".join(paths_summary) if paths_summary else "No valid paths found"
        
        # build new sufficientity test prompt format
        # extract search parameters information
        search_params = []
        if context.get('times'):
            time_vars = list(context['times'].keys())
            if time_vars:
                search_params.append(f"Time Range before: {{{context['times'][time_vars[0]]}}}")
        
        # build entity search information
        entity_info = []
        if retrieved_info:
            entities = set()
            for info in retrieved_info[:3]:
                path = info.get('path', [])
                if len(path) >= 2:
                    heads = path[0] if isinstance(path[0], list) else [path[0]]
                    tails = path[2] if len(path) > 2 and isinstance(path[2], list) else [path[2] if len(path) > 2 else 'Unknown']
                    entities.update(heads)
                    entities.update(tails)
            if entities:
                entity_info.append(f"Search on {{{', '.join(entities)}}}")
        
        # buildSortinformation
        sort_info = "Sort by Time {Descending}" if "last" in subquestion.lower() else "Sort by Time {Ascending}"  # 默认倒序，可以根据实际case调整
        
        # buildToolkitinformation
        toolkit_info_text = "{}"
        if toolkit_info:
            method_name = toolkit_info.get('method_name', 'Unknown')
            parameters = toolkit_info.get('parameters', {})
            toolkit_info_text = f"{{method: {method_name}, params: {parameters}}}"
        
        from config import TPKGConfig
        if TPKGConfig.DATASET == "TimeQuestions":
            inital_prompt = f"""
Based on the following reasoning path, answer the original question.
If Date or Time asking, It is only asking for the year.
            """
        else:
            inital_prompt = f"""
Based on the following reasoning path, answer the original question.
"""
        prompt = f"""
{inital_prompt}

    

Given question: {subquestion}

Answer: {answer_entity} ({answer_time})
Evidence paths: {retrieved_text}

Obtained by: {', '.join(search_params + entity_info)}, {sort_info}, Select Top 3,
Using toolkit (including parameters): {toolkit_info_text}

Top 3 Candidate Paths: 
{top_paths_text}


Check if the answer:
- Directly answers the given question
- Has complete information (entity, time, relationship)
- Is supported by the evidence paths and top candidate paths
- Fits logically with previous answers
- IF the answer is a list, the list should be the answer of the subquestion
Note: Multiple events happend simultaneously, Consider them same, IF question ask first or last, please consider this list of events as one event that happened at the same time (all be the first or last).

Note: the given path is already sorted by the time in first or last, so you don't need to consider the time.
you should consider the sematic information and reasonable. path may the the order, consider who is Active and who is Passive.

Note: If answer is insufficient, please chose one Action from ["Decompose", "Refine", "Retreval Again"], and return the action in the JSON. otherwise, return "".

Return JSON: {{"sufficient": true/false, "answer": "[answer]", "reason": "explanation", "action": "Decompose/Refine/Retreval Again"]}}

for example:

Q: When did the team with Crazy crab as their mascot win the world series?
Evidence paths:  
path 1 :  [Crazy Crab] - sports.mascot.team - [San Francisco Giants] - sports.sports_team.championships - [2010 World Series, 2012 World Series, 2014 World Series]
A:
sufficient: true.
answer: [2010 World Series, 2012 World Series, 2014 World Series]
reason: From the given path [Crazy Crab] - sports.mascot.team - [San Francisco Giants], San Francisco Giants is answer of the split question1, 
and from [San Francisco Giants] - sports.sports_team.championships - [2010 World Series, 2012 World Series, 2014 World Series],  the World Series won by the [San Francisco Giants] are [2010, 2012, 2014], 
therefore, the provided knowledge graph path is sufficient to answer the overall question, and the answer is [2010 World Series, 2012 World Series, 2014 World Series].

Q: who did tom hanks play in apollo 13?
Evidence paths:  
[Albert Einstein, Marie Curie, Isaac Newton] - physics.award.award_winner.awards_won - [Nobel Prize in Physics] at 1901
[Isaac Newton] - physics.award.award_winner.awards_won - [Nobel Prize in Physics] at 1903
[ Nikola Tesla, Richard Feynman] - physics.award.award_winner.awards_won - [Nobel Prize in Physics] at 1903
A:
sufficient: true.
answer: Albert Einstein, Marie Curie, Isaac Newton at 1901
reason: From the given path [Albert Einstein, Marie Curie, Isaac Newton] - physics.award.award_winner.awards_won - [Nobel Prize in Physics] at 1901, Albert Einstein, Marie Curie, Isaac Newton is answer of the question means they all the first person to win the Nobel Prize in Physics at 1901.


"""
        
        print("sufficientity test prompt:", prompt)
        
        # usesufficientity verify专用模型
        from config import TPKGConfig
        sufficiency_model = TPKGConfig.SUFFICIENCY_LLM_MODEL
        response = LLM.call(LLM_SYSTEM_PROMPT, prompt, model=sufficiency_model, temperature=0)
        print("sufficientity test response:", response)
        try:
            response_clean = response.strip()
            
            if "```json" in response_clean:
                start = response_clean.find("```json") + 7
                end = response_clean.find("```", start)
                if end != -1:
                    response_clean = response_clean[start:end].strip()
            elif "```" in response_clean:
                start = response_clean.find("```") + 3
                end = response_clean.find("```", start)
                if end != -1:
                    response_clean = response_clean[start:end].strip()
            
            # attempt Found JSON object start and end
            if "{" in response_clean and "}" in response_clean:
                start = response_clean.find("{")
                end = response_clean.rfind("}") + 1
                response_clean = response_clean[start:end]
            
            print(f"🔍 clean after LLM response: {response_clean[:200]}...")
            
            result = json.loads(response_clean)
            sufficiency_result = {
                "sufficient": result.get("sufficient", False),
                "reason": result.get("reason", "No reason provided"),
                "answer": result.get("answer", "No answer provided"),
                "raw_response": response
            }
            
            # if sufficientity test passed, record enhancement data
            if sufficiency_result["sufficient"]:
                try:
                    from .enhanced_unified_integration import (
                        record_seed_selection_enhancement,
                        record_toolkit_selection_enhancement,
                        map_function_to_toolkit
                    )
                    
                    # initialize selected_seeds variable
                    selected_seeds = []
                    
                    # recordSeed Selectionenhance（ifhaverelevantinformation）
                    # priority from toolkit_info in get most initial传递给Toolkitseed
                    if toolkit_info and toolkit_info.get('seeds'):
                        try:
                            # from toolkit_info in get most initial传递给Toolkitseed
                            original_seeds = toolkit_info.get('seeds', [])
                            # get available entity information
                            available_entities = []
                            if hasattr(context, 'get') and context.get('available_entities'):
                                available_entities = [e.get('name', '') for e in context.get('available_entities', [])]
                            
                            # convert seedID to entityname
                            for seed_id in original_seeds:
                                # attempt from available entity in Found corresponding name
                                seed_name = f"Entity_{seed_id}"
                                for entity in context.get('available_entities', []):
                                    if entity.get('id') == seed_id:
                                        seed_name = entity.get('name', seed_name)
                                        break
                                selected_seeds.append(seed_name)
                        except Exception as e:
                            print(f"⚠️ fromtoolkit_infoextractseedfailed: {e}")
                    
                    # alternative: from abovebelow text and when Topanswerinextract
                    elif hasattr(context, 'get') and context.get('available_entities'):
                        try:
                            # getSelectedseed
                            if current_answer.get('entity'):
                                entity_name = current_answer.get('entity')
                                # from available entity in Found corresponding ID
                                for entity in context.get('available_entities', []):
                                    if entity.get('name') == entity_name:
                                        selected_seeds.append(entity_name)
                                        break
                            
                            if selected_seeds:
                                # get available entity information
                                available_entities = []
                                if hasattr(context, 'get') and context.get('available_entities'):
                                    available_entities = [e.get('name', '') for e in context.get('available_entities', [])]
                                print(f"available_entities: {available_entities}")
                                print(f"selected_seeds: {selected_seeds}")

                                record_seed_selection_enhancement(
                                    subquestion=subquestion,
                                    available_entities=available_entities,
                                    selected_seeds=selected_seeds,
                                    llm_output=f"Selected based on sufficiency test: {sufficiency_result['answer']}",
                                    question_type=context.get('question_type', 'unknown'),
                                    experiment_setting=experiment_setting
                                )
                                print("✅ Seed Selectionenhancement datarecord")
                        except Exception as e:
                            print(f"⚠️ recordSeed Selectionenhancefailed: {e}")
                    
                    # recordToolkit Selectionenhance    (ifhaverelevantinformation)
                    if toolkit_info:
                        try:
                            pass
                            # method_name = toolkit_info.get('method_name', '')
                            # actual_function = method_name  # assume method_name is Actual function name
                            # toolkit_name = map_function_to_toolkit(actual_function)
                            
                            # # buildIndicatorinformation
                            # indicator = {
                            #     "edges": [{"subj": "?x", "rel": "unknown", "obj": "?y", "time_var": "t1"}],
                            #     "constraints": []
                            # }
                            
                            # # buildSeed info - fromtoolkit_infoinextractmost初传递给Toolkitseed
                            # seed_info = []
                            # if toolkit_info and toolkit_info.get('seeds'):
                            #     # fromtoolkit_infoingetmost初传递给Toolkitseed
                            #     original_seeds = toolkit_info.get('seeds', [])
                            #     for seed_id in original_seeds:
                            #         seed_info.append(f"ID: {seed_id}, Name: Entity_{seed_id}")
                            # elif current_answer.get('entity'):
                            #     # ifnooriginalSeed info，use当Topanswerentityasas备选
                            #     seed_info.append(f"ID: 0, Name: {current_answer.get('entity')}")
                            # print(f"seed_info: {seed_info}")
                            # print(f"indicator: {indicator}")
                            # print(f"toolkit_name: {toolkit_name}")
                            # print(f"actual_function: {actual_function}")
                            # print(f"parameters: {toolkit_info.get('parameters', {})}")
                            # # print(f"context: {context}")
                            # print(f"time_hints: ")
                            # print(f"reasoning:Toolkit selected based on sufficiency test success: {sufficiency_result['reason']}")
                            # print(f"llm_output: Toolkit selection successful: {toolkit_name}")
                            # print(f"question_type: {context.get('question_type', 'unknown')}")
                            # record_toolkit_selection_enhancement(
                            #     subquestion=subquestion,
                            #     indicator=indicator,
                            #     seed_info=seed_info,
                            #     toolkit_name=toolkit_name,
                            #     actual_function=actual_function,
                            #     parameters=toolkit_info.get('parameters', {}),
                            #     context=context,
                            #     time_hints={},
                            #     reasoning=f"Toolkit selected based on sufficiency test success: {sufficiency_result['reason']}",
                            #     llm_output=f"Toolkit selection successful: {toolkit_name}",
                            #     question_type=context.get('question_type', 'unknown'),
                            #     experiment_setting=experiment_setting
                            # )
                            # print("✅ Toolkit Selectionenhancement datarecord")
                            # exit()
                        except Exception as e:
                            print(f"⚠️ recordToolkit Selectionenhancefailed: {e}")
                            
                except Exception as e:
                    print(f"⚠️ recordenhancement datafailed: {e}")
            
            return sufficiency_result
        except json.JSONDecodeError as e:
            print(f"⚠️ JSON parsing failed: {e}")
            print(f"original response: {response[:200]}...")
            
            # attempt from text in extract information
            sufficient = "sufficient" in response.lower() and "true" in response.lower()
            
            # attempt from text in extract answer
            answer = "No answer provided"
            if '"answer":' in response:
                # find answer field
                start = response.find('"answer":')
                if start != -1:
                    start = response.find('"', start + 9) + 1
                    end = response.find('"', start)
                    if end != -1:
                        answer = response[start:end]
            
            return {
                "sufficient": sufficient,
                "reason": "JSON parsing failed, using text analysis",
                "answer": answer,
                "raw_response": response
            }
            
    except Exception as e:
        return {
            "sufficient": True,  # default assume sufficient, avoid blocking
            "reason": f"Error in sufficiency test: {str(e)}",
            # "suggestions": [],
            "answer": "No answer provided",
            "raw_response": ""
        }


def generate_fallback_answer(original_question: str, 
                           reasoning_path: List[Dict[str, Any]], 
                           all_info: List[Dict[str, Any]],
                           trajectory: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    when test_final_answer_sufficiencyreturnnotsufficient and reach most limit,
    generate one based on retrieval information predict answer
    """
    try:
        subquestions_details = []

        if trajectory:
            for i, traj in enumerate(trajectory, 1):
                subq_info = traj.get('subq', {})
                result_info = traj.get('result', {})
                subq_text = subq_info.get('text', f'Step {i}')
                # from resulting in extract chosen information
                chosen = result_info.get('chosen', {})
                # attempt from multiple position in get LLM_selected_answer
                LLM_selected_answer = traj.get('LLM_selected_answer', '')
                if not LLM_selected_answer and chosen:
                    LLM_selected_answer = chosen.get('LLM_selected_answer', '')
                if not LLM_selected_answer:
                    # from sufficiency_test in extract answer information
                    sufficiency_test = traj.get('sufficiency_test', {})
                    if sufficiency_test:
                        LLM_selected_answer = sufficiency_test.get('answer', '')
                
                LLM_selected_reason = traj.get('sufficiency_test', {}).get('reason', '')
                
                # if no chosen information, attempt from sufficiency_test in extract answer information
                if not chosen:
                    sufficiency_test = traj.get('sufficiency_test', {})
                    if sufficiency_test and sufficiency_test.get('sufficient'):
                        # fromsufficiency_testinextractanswerinformation
                        answer = sufficiency_test.get('answer', '')

                if chosen:
                    entity = chosen.get('entity', 'Unknown')
                    time = chosen.get('time', 'Unknown')
                    path = chosen.get('path', [])
                    proof = chosen.get('provenance', {}).get('proof', {})
                    selection_reason = chosen.get('provenance', {}).get('selection_reason', '')
                    debate_vote = chosen.get('provenance', {}).get('debate_vote', {})
                    
                    step_detail = f"Step {i} of {len(trajectory)}\n Sub-Question: {subq_text}"
                    step_detail += f"\nStep {i} Answer: {entity} at {time}"
                    step_detail += f"\nLLM Selected Answer: {LLM_selected_answer}"
                    if LLM_selected_reason:
                        step_detail += f"\nLLM Selected Reason: {LLM_selected_reason}"
                    
                    if path and len(path) >= 2:
                        heads = path[0] if isinstance(path[0], list) else [path[0]]
                        relation = path[1]
                        tails = path[2] if len(path) > 2 and isinstance(path[2], list) else [path[2] if len(path) > 2 else 'Unknown']
                        
                        heads_str = ",".join(heads) if len(heads) > 1 else (heads[0] if heads else 'Unknown')
                        tails_str = ",".join(tails) if len(tails) > 1 else (tails[0] if tails else 'Unknown')
                        
                        if len(heads) > 1 or len(tails) > 1:
                            step_detail += f"\nEvidence Path: [{heads_str}] - {relation} - [{tails_str}] at {time} (Multiple events happend simultaneously, Consider them same)"
                        else:
                            step_detail += f"\nEvidence Path: [{heads_str} - {relation} - {tails_str}] at {time}"
                    elif proof:
                        heads_str = proof.get('heads_str', '')
                        relation = proof.get('relation', '')
                        tail = proof.get('tail', '')
                        heads_count = proof.get('heads_count', 1)
                        tail_count = proof.get('tail_count', 1)
                        
                        if heads_str and relation and tail:
                            if heads_count > 1 or tail_count > 1:
                                step_detail += f"\nEvidence Path: [{heads_str}] - {relation} - [{tail}] at {time} (Multiple events happend simultaneously, Consider them same)"
                            else:
                                step_detail += f"\nEvidence Path: [{heads_str} - {relation} - {tail}] at {time}"
                    
                    # addRetrieval proof - gettop3 paths
                    candidates = result_info.get('candidates', [])
                    if candidates:
                        step_detail += f"\nRetrieval Proof (Top 3 paths):"
                        for j, candidate in enumerate(candidates[:3], 1):
                            candidate_entity = candidate.get('entity', 'Unknown')
                            candidate_time = candidate.get('time', 'Unknown')
                            candidate_path = candidate.get('path', [])
                            candidate_score = candidate.get('provenance', {}).get('llm_score', 0.0)
                            
                            if candidate_path and len(candidate_path) >= 2:
                                heads = candidate_path[0] if isinstance(candidate_path[0], list) else [candidate_path[0]]
                                relation = candidate_path[1]
                                tails = candidate_path[2] if len(candidate_path) > 2 and isinstance(candidate_path[2], list) else [candidate_path[2] if len(candidate_path) > 2 else 'Unknown']
                                
                                heads_str = ",".join(heads) if len(heads) > 1 else heads[0]
                                tails_str = ",".join(tails) if len(tails) > 1 else tails[0]
                                
                                step_detail += f"\n  Path {j}: [{heads_str} - {relation} - {tails_str}] at {candidate_time} (Score: {candidate_score:.3f})"
                            else:
                                step_detail += f"\n  Path {j}: {candidate_entity} at {candidate_time} (Score: {candidate_score:.3f})"
                    
                    step_detail += f"\nReasoning: {selection_reason}"
                    subquestions_details.append(step_detail)

        subquestions_text = "\n\n".join(subquestions_details)

        # buildprompt
        prompt = LLM_FALLBACK_ANSWER_PROMPT.format(
            original_question=original_question,
            subquestions_details=subquestions_text
        )

        print(prompt)
        response = LLM.call(LLM_SYSTEM_PROMPT, prompt)
        print(response)
        try:
            result = json.loads(response)
        except json.JSONDecodeError:
            reasoning = "No reasoning provided"
            llm_final_answer = "Unable to generate answer"
            if "so the answer is:" in response.lower():
                try:
                    import re
                    match = re.search(r"so the answer is:\s*(.+)", response, re.IGNORECASE)
                    if match:
                        llm_final_answer = match.group(1).strip()
                except:
                    pass
        else:
            reasoning = result.get("reasoning", "No reasoning provided")
            llm_final_answer = result.get("final_answer", "")
            
            if not llm_final_answer and trajectory:
                last_traj = trajectory[-1]
                llm_final_answer = last_traj.get('LLM_selected_answer', '')
                if not llm_final_answer:
                    llm_final_answer = last_traj.get('sufficiency_test', {}).get('answer', '')
            
            if trajectory:
                try:
                    last_traj = trajectory[-1]
                    last_llm_final_answer = last_traj.get('LLM_selected_answer', '')
                    if not last_llm_final_answer:
                        last_llm_final_answer = last_traj.get('sufficiency_test', {}).get('answer', '')
                    
                    # only in have extra answer when add
                    if last_llm_final_answer and last_llm_final_answer != llm_final_answer:
                        llm_final_answer = llm_final_answer + ", or " + last_llm_final_answer

                    # from resulting in extract chosen information
                    last_result = last_traj.get('result', {})
                    last_chosen = last_result.get('chosen', {})
                    last_entity = last_chosen.get('entity', '')
                    last_time = last_chosen.get('time', '')
                    if last_entity and last_time:
                        last_entity_and_time = f"[{last_entity}] on {last_time}"
                        if last_entity_and_time not in llm_final_answer:
                            llm_final_answer = llm_final_answer + ", or " + last_entity_and_time
                except Exception as e:
                    # if add extra answer failed, keep main answer
                    print(f"⚠️ addextraanswerinformation时出错: {e}")
                    pass

        return {
            "sufficient": False,  # mark as not sufficient
            "reason": reasoning,
            "suggestions": [],  # newformatnotagainusesuggestions
            "final_answer": llm_final_answer,
            "raw_response": response,
            "fallback": True  # mark this is fallback answer
        }
        
    except Exception as e:
        print(f"generatefallbackanswerfailed: {e}")
        return {
            "sufficient": False,
            "reason": f"Error generating fallback answer: {e}",
            "suggestions": [],
            "final_answer": "Unable to generate answer",
            "raw_response": "",
            "fallback": True
        }

def test_final_answer_sufficiency(original_question: str, 
                                reasoning_path: List[Dict[str, Any]], 
                                all_info: List[Dict[str, Any]],
                                trajectory: List[Dict[str, Any]] = None,
                                experiment_setting = None,
                                ctx: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    test FinalAnswer sufficientity
    
    Args:
        original_question: originalquestions
        final_answer: FinalAnswer
            reasoning_path: reasoning KG_Path
        all_info: all retrieval information
        trajectory: complete Trajectory information, contain each Subquestion detailed information
        experiment_setting: experiment setting
        ctx: abovebelow text information, contains question type etc.
    
    Returns:
        sufficientity test result
    """
    try:
        subquestions_details = []

        if trajectory:
            for i, traj in enumerate(trajectory, 1):
                subq_info = traj.get('subq', {})
                result_info = traj.get('result', {})
                subq_text = subq_info.get('text', f'Step {i}')
                # from resulting in extract chosen information
                chosen = result_info.get('chosen', {})
                # attempt from multiple position in get LLM_selected_answer
                LLM_selected_answer = traj.get('LLM_selected_answer', '')
                if not LLM_selected_answer and chosen:
                    LLM_selected_answer = chosen.get('LLM_selected_answer', '')
                if not LLM_selected_answer:
                    # from sufficiency_test in extract answer information
                    sufficiency_test = traj.get('sufficiency_test', {})
                    if sufficiency_test:
                        LLM_selected_answer = sufficiency_test.get('answer', '')
                LLM_selected_reason = traj.get('sufficiency_test', {}).get('reason', '')
                
                # if no chosen information, attempt from sufficiency_test in extract answer information
                if not chosen:
                    sufficiency_test = traj.get('sufficiency_test', {})
                    if sufficiency_test and sufficiency_test.get('sufficient'):
                        # fromsufficiency_testinextractanswerinformation
                        answer = sufficiency_test.get('answer', '')

                print(f"chosen: {chosen}")
                print(f"subq_text: {subq_text}")
                print(f"LLM_selected_answer: {LLM_selected_answer}")
                # exit()
                
                # initialize step_detail, ensure in all case below have definition
                step_detail = f"Step {i} of {len(trajectory)}\n Sub-Question: {subq_text}"
                
                if chosen:
                    entity = chosen.get('entity', 'Unknown')
                    time = chosen.get('time', 'Unknown')
                    path = chosen.get('path', [])
                    proof = chosen.get('provenance', {}).get('proof', {})
                    selection_reason = chosen.get('provenance', {}).get('selection_reason', '')
                    debate_vote = chosen.get('provenance', {}).get('debate_vote', {})
                    
                    # build detailed evidence information
                    step_detail += f"\nStep {i} Answer: {entity} at {time}"
                    step_detail += f"\nLLM Selected Answer: {LLM_selected_answer}"
                    if LLM_selected_reason:
                        step_detail += f"\nLLM Selected Reason: {LLM_selected_reason}"
                    # if time != 'Unknown':
                    #     step_detail += f" (Time: {time})"
                    
                    # addProof path - support newpath structure [heads_list, relation, tails_list]
                    if path and len(path) >= 2:
                        # process newpath structure
                        heads = path[0] if isinstance(path[0], list) else [path[0]]
                        relation = path[1]
                        tails = path[2] if len(path) > 2 and isinstance(path[2], list) else [path[2] if len(path) > 2 else 'Unknown']
                        
                        # build path string, display aggregated information
                        heads_str = ",".join(heads) if len(heads) > 1 else (heads[0] if heads else 'Unknown')
                        tails_str = ",".join(tails) if len(tails) > 1 else (tails[0] if tails else 'Unknown')
                        
                        if len(heads) > 1 or len(tails) > 1:
                            step_detail += f"\nEvidence Path: [{heads_str}] - {relation} - [{tails_str}] at {time} (Multiple events happend simultaneously, Consider them same)"
                        else:
                            step_detail += f"\nEvidence Path: [{heads_str} - {relation} - {tails_str}] at {time}"
                    elif proof:
                        # fallback to proof information
                        heads_str = proof.get('heads_str', '')
                        relation = proof.get('relation', '')
                        tail = proof.get('tail', '')
                        heads_count = proof.get('heads_count', 1)
                        tail_count = proof.get('tail_count', 1)
                        
                        if heads_str and relation and tail:
                            if heads_count > 1 or tail_count > 1:
                                aggregation_info = []
                                if heads_count > 1:
                                    aggregation_info.append(f"{heads_count} heads")
                                if tail_count > 1:
                                    aggregation_info.append(f"{tail_count} tails")
                                step_detail += f"\nEvidence Path: [{heads_str}] - {relation} - [{tail}] at {time} (Multiple events happend simultaneously, Consider them same)"
                            else:
                                step_detail += f"\nEvidence Path: [{heads_str}] - {relation} - [{tail}] at {time}"
                    
                    # addTop N candidate path information - from result top_paths get ( with Step sufficientity test one consistent)
                    top_paths = result_info.get('top_paths', [])
                    if not top_paths:
                        # fallback: attemptfromcandidatesget
                        top_paths = result_info.get('candidates', [])
                    
                    if top_paths:
                        step_detail += f"\n\nCandidate paths:"
                        for j, path_info in enumerate(top_paths[:3], 1):
                            # support two format: top_paths format and candidates format
                            if isinstance(path_info, dict):
                                # check is top_paths format or candidates format
                                if 'heads_str' in path_info or 'head' in path_info:
                                    # top_pathsformat
                                    heads_str = path_info.get('heads_str', path_info.get('head', 'Unknown'))
                                    relation = path_info.get('relation', 'Unknown')
                                    tail = path_info.get('tail', 'Unknown')
                                    tails_str = path_info.get('tails_str', tail)
                                    time_start = path_info.get('time_start', path_info.get('time', 'Unknown'))
                                    
                                    step_detail += f"\n{j}. [{heads_str}] -> {relation} -> [{tails_str}] at {time_start}"
                                else:
                                    # candidatesformat
                                    candidate_entity = path_info.get('entity', 'Unknown')
                                    candidate_time = path_info.get('time', 'Unknown')
                                    candidate_path = path_info.get('path', [])
                                    
                                    if candidate_path and len(candidate_path) >= 2:
                                        heads = candidate_path[0] if isinstance(candidate_path[0], list) else [candidate_path[0]]
                                        relation = candidate_path[1]
                                        tails = candidate_path[2] if len(candidate_path) > 2 and isinstance(candidate_path[2], list) else [candidate_path[2] if len(candidate_path) > 2 else 'Unknown']
                                        
                                        heads_str = ", ".join(heads) if len(heads) > 1 else heads[0] if heads else 'Unknown'
                                        tails_str = ", ".join(tails) if len(tails) > 1 else tails[0] if tails else 'Unknown'
                                        
                                        step_detail += f"\n{j}. [{heads_str}] -> {relation} -> [{tails_str}] at {candidate_time}"
                                    else:
                                        step_detail += f"\n{j}. {candidate_entity} at {candidate_time}"
                    
                    # add select reason
                    if selection_reason:
                        step_detail += f"\n\nReasoning: {selection_reason}"
                    
                    # add debate vote information (if have)
                    if debate_vote:
                        winning_toolkit = debate_vote.get('winning_toolkit', '')
                        vote_reason = debate_vote.get('vote_reason', '')
                        if winning_toolkit:
                            step_detail += f"\nSelected by: Toolkit {winning_toolkit}"
                        if vote_reason:
                            step_detail += f" - {vote_reason}"
                    
                    subquestions_details.append(step_detail)
                else:
                    step_detail += f"\nStep {i} Answer: No answer found"
                    subquestions_details.append(step_detail)
        else:
            subquestions_details = ["No step details available"]
        
        subquestions_text = "\n\n".join(subquestions_details)
        
        # simplify process, only keep core information
        
        prompt = LLM_FINAL_SUFFICIENT_TEST_PROMPT.format(
            original_question=original_question,
            subquestions_details=subquestions_text
        )
        print(prompt)
        from config import TPKGConfig
        final_answer_model = TPKGConfig.FINAL_ANSWER_LLM_MODEL
        response = LLM.call(LLM_SYSTEM_PROMPT, prompt, model=final_answer_model, temperature=0)
        print(response)

        # exit()
        try:
            result = json.loads(response)
            # new format contains reasoning and final_answer field
            reasoning = result.get("reasoning", "No reasoning provided")
            llm_final_answer = result.get("final_answer", "")
            if not llm_final_answer:
                last_traj = trajectory[-1]
                llm_final_answer = last_traj.get('LLM_selected_answer', '')
                print(f"llm_final_answer: {llm_final_answer}")
                if not llm_final_answer:
                    llm_final_answer = last_traj.get('sufficiency_test', {}).get('answer', '')
                    print(f"llm_final_answer: {llm_final_answer}")
            last_traj = trajectory[-1]
            last_llm_final_answer = last_traj.get('LLM_selected_answer', '')
            print(f"last_llm_final_answer: {last_llm_final_answer}")
            if not last_llm_final_answer:
                last_llm_final_answer = last_traj.get('sufficiency_test', {}).get('answer', '')
                print(f"last_llm_final_answer: {last_llm_final_answer}")
            llm_final_answer = llm_final_answer + ", or " + last_llm_final_answer

            # from resulting in extract chosen information
            last_result = last_traj.get('result', {})
            last_chosen = last_result.get('chosen', {})
            last_entity = last_chosen.get('entity', '')
            last_time = last_chosen.get('time', '')
            last_entity_and_time = f"[{last_entity}] on {last_time}".strip()
            # print(f"last_entity_and_time: {last_entity_and_time}")
            # print(f"last_chosen: {last_chosen}")
            llm_final_answer = llm_final_answer + ", or " + last_entity_and_time
            # print(f"llm_final_answer: {llm_final_answer}")

            sufficiency_result = {
                "sufficient": result.get("sufficient", False),
                "reason": reasoning,
                "suggestions": [],  # newformatnotagainusesuggestions
                "final_answer": llm_final_answer,
                "raw_response": response
            }
            
            # if Final sufficientity test pass, record questions enhancement data
            if sufficiency_result["sufficient"]:
                try:
                    from .enhanced_unified_integration import (
                        record_question_decomposition_enhancement,
                        record_seed_selection_enhancement,
                        record_toolkit_selection_enhancement,
                        record_experience_pool_enhancement,
                        map_function_to_toolkit
                    )
                    
                    print("\n🔍 extracttrajectory and answer_edgesinformation:")
                    print(f"Trajectorycount: {len(trajectory)}")
                    
                    # print trajectory information for debug
                    for i, traj in enumerate(trajectory):
                        print(f"\n--- Trajectory {i+1} ---")
                        print(f"Subq: {traj.get('subq', {}).get('text', 'N/A')}")
                        print(f"Selected seeds: {traj.get('selected_seed_names', [])}")
                        print(f"Available seeds: {traj.get('available_seeds', [])}")
                        print(f"Result keys: {list(traj.get('result', {}).keys())}")
                        print(f"Sufficiency test: {traj.get('sufficiency_test', {})}")
                    
                    # print answer_edges information
                    if 'answer_edges' in locals() or reasoning_path:
                        edges = reasoning_path if reasoning_path else []
                        print(f"\nAnswer edgescount: {len(edges)}")
                        for i, edge in enumerate(edges):
                            print(f"\n--- Answer Edge {i+1} ---")
                            print(f"Subquestion: {edge.get('subquestion', 'N/A')}")
                            print(f"Entity: {edge.get('entity', 'N/A')}")
                            print(f"Time: {edge.get('time', 'N/A')}")
                            print(f"Path: {edge.get('path', [])}")
                            print(f"Score: {edge.get('score', 0.0)}")
                            print(f"LLM answer: {edge.get('LLM_selected_answer', 'N/A')}")
                    
                    # from trajectory in extract information
                    if trajectory and len(trajectory) > 0:
                        # direct from ctx in get most initial determine question type
                        question_type = ctx.get('question_type', 'unknown') if ctx else 'unknown'
                        print(f"use most initial determine question type: {question_type}")
                        
                        # build result
                        subquestions = []
                        indicators = []
                        constraints = []
                        time_vars = []
                        
                        for traj in trajectory:
                            subq = traj.get('subq', {})
                            if subq.get('text'):
                                subquestions.append(subq['text'])
                            
                            # attempt to extract Indicator and constraints
                            if subq.get('indicator'):
                                indicator = subq['indicator']
                                if indicator.get('edges'):
                                    for edge in indicator['edges']:
                                        indicators.append(f"{edge.get('subj', '?x')} --[{edge.get('rel', 'unknown')}]--> {edge.get('obj', '?y')} ({edge.get('time_var', 't1')})")
                                if indicator.get('constraints'):
                                    constraints.extend(indicator['constraints'])
                        
                        # extract time variables
                        time_vars = list(set([f"t{i+1}" for i in range(len(trajectory))]))
                        
                        print(f"\n📋 extract questions information:")
                        print(f"questionstype: {question_type}")
                        print(f"Subquestion: {subquestions}")
                        print(f"Indicator: {indicators}")
                        print(f"constraints: {constraints}")
                        print(f"time variables: {time_vars}")
                        
                        # record questions enhancement data
                        decomposition_result = {
                            "subquestions": subquestions,
                            "indicators": indicators,
                            "constraints": constraints,
                            "time_vars": time_vars
                        }
                        print(f"decomposition_result: {decomposition_result}")
                        
                        record_question_decomposition_enhancement(
                            question=original_question,
                            question_type=question_type,
                            decomposition_result=decomposition_result,
                            llm_output=f"Final answer: {llm_final_answer}, Reasoning: {reasoning}",
                            experiment_setting=experiment_setting
                        )
                        print("✅ questions enhancement data record")
                        
                        # recordSeed Selection and Toolkit Selectionenhancement data
                        for i, traj in enumerate(trajectory):
                            try:
                                subq_text = traj.get('subq', {}).get('text', '')
                                available_seeds = traj.get('available_seeds', [])
                                result = traj.get('result', {})
                                selected_seeds = traj.get('selected_seed_names', [])
                                seeds_info_name_list = traj.get('seeds_info_name_list', [])
                                # print(f"step_answer_souce: {step_answer_souce}")
                                if traj.get('step_answer_souce') == "new_retrieval":
                                    step_answer_souce = "new_retrieval"
                                    print(f"step_answer_souce: {step_answer_souce}")
                                else:
                                    step_answer_souce = "unified_knowledge_store"
                                    print(f"step_answer_souce: {step_answer_souce}")
                                    continue
                                # # fromresultinextractSeed info
                                # selected_seeds = []
                                # if result:
                                #     # attemptfromchosenanswerinextractentity
                                #     chosen = result.get('chosen', {})
                                #     if chosen:
                                #         entity = chosen.get('entity', '')
                                #         if entity:
                                #             selected_seeds.append(entity)
                                    
                                #     # attemptfromanswersinextractentity
                                #     answers = result.get('answers', [])
                                #     if answers and not selected_seeds:
                                #         for answer in answers[:3]:  # 取Top3
                                #             if isinstance(answer, dict) and answer.get('entity'):
                                #                 selected_seeds.append(answer['entity'])
                                    
                                #     # attemptfromtop_candidatesinextractentity
                                #     top_candidates = result.get('top_candidates', [])
                                #     if top_candidates and not selected_seeds:
                                #         for candidate in top_candidates[:3]:  # 取Top3
                                #             if isinstance(candidate, dict) and candidate.get('entity'):
                                #                 selected_seeds.append(candidate['entity'])
                                
                                # recordSeed Selectionenhance
                                if available_seeds and subq_text:
                                    print(f"\n📋 recordSeed Selectionenhance (Trajectory {i+1}):")
                                    print(f"Subquestion: {subq_text}")
                                    print(f"Available seed: {available_seeds}")
                                    print(f"Selectedseed: {selected_seeds}")
                                    
                                    record_seed_selection_enhancement(
                                        subquestion=subq_text,
                                        available_entities=available_seeds,
                                        selected_seeds=selected_seeds,
                                        llm_output=f"Selected seeds based on trajectory {i+1}",
                                        question_type=question_type,
                                        experiment_setting=experiment_setting
                                    )
                                    print("✅ Seed Selectionenhancement datarecord")
                                
                                # recordToolkit Selectionenhance
                                detailed_toolkit_info = traj.get('detailed_toolkit_info', {})
                                if detailed_toolkit_info:
                                    method_name = detailed_toolkit_info.get('method', '') or detailed_toolkit_info.get('method_name', '')
                                    parameters = detailed_toolkit_info.get('parameters', {})
                                    seeds = detailed_toolkit_info.get('seeds', [])
                                    description = detailed_toolkit_info.get('description', '')
                                    all_selections = detailed_toolkit_info.get('all_selections', [])
                                    seed_info = [f"ID: {KG.get_entity_id(seed)}, Name: {seed}" for seed in selected_seeds]
                                    # seed_info = seeds
                                    if method_name:
                                        toolkit_name = map_function_to_toolkit(method_name)
                                        
                                        print(f"\n📋 recordToolkit Selectionenhance (Trajectory {i+1}):")
                                        print(f"Subquestion: {subq_text}")
                                        print(f"Toolkit: {toolkit_name}")
                                        print(f"Actual function: {method_name}")
                                        print(f"Description: {description}")
                                        print(f"parameters: {parameters}")
                                        print(f"seed: {seeds}")
                                        print(f"allselect: {all_selections}")
                                        
                                        # fromchosenanswerinbuildIndicatorinformation
                                        indicator = {
                                            "edges": [{"subj": "?x", "rel": "unknown", "obj": "?y", "time_var": "t1"}],
                                            "constraints": []
                                        }
                                        
                                        if result and result.get('chosen'):
                                            chosen = result['chosen']
                                            path = chosen.get('path', [])
                                            if path and len(path) >= 3:
                                                # frompathinextractrelationinformation
                                                subj = path[0][0] if path[0] else "?x"
                                                rel = path[1] if len(path) > 1 else "unknown"
                                                obj = path[2][0] if len(path) > 2 and path[2] else "?y"
                                                time_var = f"t{i+1}"
                                                
                                                indicator = {
                                                    "edges": [{"subj": subj, "rel": rel, "obj": obj, "time_var": time_var}],
                                                    "constraints": []
                                                }
                                        
                                        # # buildSeed info
                                        # seed_info = []
                                        # # 优先usedetailed_toolkit_infoinseeds，然afterisselected_seeds
                                        # final_seeds = seeds if seeds else selected_seeds
                                        # for seed in final_seeds:
                                        #     if isinstance(seed, (int, str)):
                                        #         seed_info.append(f"ID: {seed}, Name: Entity_{seed}")
                                        #     else:
                                        #         seed_info.append(f"ID: 0, Name: {seed}")
                                        
                                        # buildTime hints
                                        time_hints = {}
                                        if result and result.get('chosen') and result['chosen'].get('time'):
                                            time_hints = {"time": result['chosen']['time']}
                                        
                                        # build richer context
                                        context = {
                                            "chosen": result.get('chosen', {}) if result else {},
                                            "toolkit_description": description,
                                            "all_selections": all_selections,
                                            "question_type": detailed_toolkit_info.get('question_type', ''),
                                            "subq_index": detailed_toolkit_info.get('subq_index', 0)
                                        }
                                        print(f"context: {context}")
                                        print(f"time_hints: {time_hints}")
                                        print(f"indicator: {indicator}")
                                        print(f"seed_info: {seed_info}")
                                        # print(f"seeds_info_name_list: {seeds_info_name_list}")
                                        print(f"toolkit_name: {toolkit_name}")
                                        print(f"actual_function: {method_name}")
                                        # print(f"parameters: {parameters}")
                                        
                                        record_toolkit_selection_enhancement(
                                            subquestion=subq_text,
                                            indicator=indicator,
                                            seed_info=seed_info,
                                            toolkit_name=toolkit_name,
                                            actual_function=method_name,
                                            parameters=parameters,
                                            context=context,
                                            time_hints=time_hints,
                                            reasoning=f"Toolkit selected from trajectory {i+1}: {description}",
                                            llm_output=f"Toolkit selection from trajectory {i+1}: {toolkit_name} ({description})",
                                            question_type=question_type,
                                            experiment_setting=experiment_setting
                                        )
                                        print("✅ Toolkit Selectionenhancement datarecord")
                                        # exit()
                                        
                                        # recordExperience pooldata
                                        try:
                                            print(f"\n📋 recordExperience pooldata (Trajectory {i+1}):")
                                            print(f"Subquestion: {subq_text}")
                                            print(f"Evidence: {chosen.get('entity', 'N/A')} at {chosen.get('time', 'N/A')}")
                                            
                                            # accurate backup Toolkit information
                                            toolkit_info = {
                                                "method": method_name,
                                                "parameters": parameters,
                                                "question_type": question_type,
                                                "subq_index": i
                                            }
                                            
                                            # recordExperience pooldata
                                            record_experience_pool_enhancement(
                                                subquestion=subq_text,
                                                evidence=chosen,
                                                toolkit_info=toolkit_info,
                                                context={
                                                    "question_type": question_type, 
                                                    "subq_obj": traj.get('subq'),
                                                    "sufficiency_args": traj.get('sufficiency_args', {}),
                                                    "top_paths": traj.get('result', {}).get('top_paths', [])
                                                },
                                                experiment_setting=experiment_setting,
                                                enable_knowledge=True
                                            )
                                            print("✅ Experience pooldatarecord")
                                            
                                        except Exception as e:
                                            print(f"⚠️ recordExperience pooldatafailed: {e}")
                                            import traceback
                                            traceback.print_exc()
                                        
                            except Exception as e:
                                print(f"⚠️ recordtrajectory {i+1}enhancement datafailed: {e}")
                                import traceback
                                traceback.print_exc()
                        
                except Exception as e:
                    print(f"⚠️ record questions enhancement data failed: {e}")
                    import traceback
                    traceback.print_exc()
            
            return sufficiency_result
        except json.JSONDecodeError:
            # if JSON parsing failed, attempt from response in extract "So the answer is:" format answer
            sufficient = "sufficient" in response.lower() and "true" in response.lower()
            
            # attempt to extract "So the answer is:" format answer
            extracted_answer = llm_final_answer
            if "so the answer is:" in response.lower():
                try:
                    answer_start = response.lower().find("so the answer is:") + len("so the answer is:")
                    answer_part = response[answer_start:].strip()
                    # extract first line or to period as end content
                    if '\n' in answer_part:
                        extracted_answer = answer_part.split('\n')[0].strip()
                    elif '.' in answer_part:
                        extracted_answer = answer_part.split('.')[0].strip()
                    else:
                        extracted_answer = answer_part.strip()
                except:
                    pass
            
            return {
                "sufficient": sufficient,
                "reason": "JSON parsing failed, using text analysis",
                "suggestions": ["Improve final answer quality"],
                "final_answer": extracted_answer,
                "raw_response": response
            }
            
    except Exception as e:
        return {
            "sufficient": True,  # default assume sufficient, avoid blocking
            "reason": f"Error in final sufficiency test: {str(e)}",
            "suggestions": [],
            "raw_response": ""
        }


def regenerate_subquestion(original_subquestion: str, current_answer: Dict[str, Any],
                         gap_analysis: str, context: Dict[str, Any], 
                         original_question: str = None, 
                         previous_subquestions: List[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    regenerate new generate Subquestion
    
    Args:
        original_subquestion: original Subquestion
        current_answer: current Top not sufficient Answer
        gap_analysis: gap analysis
        context: abovebelow text information
        original_question: original questions
        previous_subquestions:    (previous subquestions and their answers)
    
    Returns:
        newSubquestioninformation
    """
    try:
        answer_text = f"Entity: {current_answer.get('entity', 'Unknown')}, Time: {current_answer.get('time', 'Unknown')}"
        context_text = f"Solve Subquestion: {list(context.get('answers', {}).keys())}, Time variables: {list(context.get('times', {}).keys())}"
        
        # buildTop面通Subquestioninformation
        previous_subq_text = "no Top Subquestion"
        if previous_subquestions:
            previous_summary = []
            for i, prev_subq in enumerate(previous_subquestions, 1):
                prev_text = prev_subq.get('text', f'Subquestion{i}')
                prev_answer = prev_subq.get('answer', {})
                prev_entity = prev_answer.get('entity', 'Unknown')
                prev_time = prev_answer.get('time', 'Unknown')
                previous_summary.append(f"Subquestion{i}: {prev_text} -> answer: {prev_entity} (time: {prev_time})")
            previous_subq_text = "\n".join(previous_summary)
        
        # if no provide original questions, use default value
        if not original_question:
            original_question = "original questions not provided"
        
        prompt = LLM_REGENERATE_SUBQUESTION_PROMPT.format(
            original_question=original_question,
            original_subquestion=original_subquestion,
            current_answer=answer_text,
            gap_analysis=gap_analysis,
            context=context_text,
            previous_subquestions=previous_subq_text
        )
        
        response = LLM.call(LLM_SYSTEM_PROMPT, prompt)
        
        try:
            # first attempt JSON parsing
            result = json.loads(response)
            return {
                "new_subquestion": result.get("new_subquestion", original_subquestion),
                "indicator": result.get("indicator", {}),
                "reasoning": result.get("reasoning", "No reasoning provided"),
                "raw_response": response
            }
        except json.JSONDecodeError:
            # JSON parsing failed, attempt from plain text in extract Subquestion
            print(f"JSON parsing failed, attempt from plain text in extract Subquestion: {response}")
            
            # clean response text
            cleaned_response = response.strip()
            
            # if response looks like is one questions (contains question mark), use it
            if "?" in cleaned_response and len(cleaned_response) > 10:
                return {
                    "new_subquestion": cleaned_response,
                    "indicator": {},
                    "reasoning": "Extracted from plain text response",
                    "raw_response": response
                }
            else:
                # if no extract valid questions, return original Subquestion
                print(f"no from response in extract valid Subquestion: {cleaned_response}")
                return {
                    "new_subquestion": original_subquestion,
                    "indicator": {},
                    "reasoning": "Could not extract valid subquestion from plain text",
                    "raw_response": response
                }
            
    except Exception as e:
        return {
            "new_subquestion": original_subquestion,
            "indicator": {},
            "reasoning": f"Error in regeneration: {str(e)}",
            "raw_response": ""
        }


def regenerate_final_question(original_question: str, final_answer: str,
                            gap_analysis: str, available_info: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    regenerate new generate Final questions
    
    Args:
        original_question: originalquestions
        final_answer: not sufficient Final Answer
        gap_analysis: gap analysis
        available_info: available information
    
    Returns:
        newquestionsinformation
    """
    try:
        info_summary = []
        for info in available_info[:10]:
            info_summary.append(f"- {info.get('entity', 'Unknown')} (Time: {info.get('time', 'Unknown')})")
        
        available_info_text = "\n".join(info_summary) if info_summary else "no可用information"
        
        prompt = LLM_REGENERATE_FINAL_QUESTION_PROMPT.format(
            original_question=original_question,
            final_answer=final_answer,
            gap_analysis=gap_analysis,
            available_info=available_info_text
        )
        
        response = LLM.call(LLM_SYSTEM_PROMPT, prompt)
        
        try:
            # first attempt JSON parsing
            result = json.loads(response)
            return {
                "new_question": result.get("new_question", original_question),
                "reasoning": result.get("reasoning", "No reasoning provided"),
                "raw_response": response
            }
        except json.JSONDecodeError:
            # JSON parsing failed, attempt from plain text in extract questions
            print(f"JSON parsing failed, attempt from plain text in extract questions: {response}")
            
            # clean response text
            cleaned_response = response.strip()
            
            # if response looks like is one questions (contains question mark), use it
            if "?" in cleaned_response and len(cleaned_response) > 10:
                return {
                    "new_question": cleaned_response,
                    "reasoning": "Extracted from plain text response",
                    "raw_response": response
                }
            else:
                # if no extract valid questions, return original questions
                print(f"no from response in extract valid questions: {cleaned_response}")
                return {
                    "new_question": original_question,
                    "reasoning": "Could not extract valid question from plain text",
                    "raw_response": response
                }
            
    except Exception as e:
        return {
            "new_question": original_question,
            "reasoning": f"Error in regeneration: {str(e)}",
            "raw_response": ""
        }

