#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
intelligent retrieval and selection system (integrate 8 "low-level toolkits")   
- through the same retrieval pipeline (seed / time filter / time sort / relation matching / scoring)
- use relation_select_mode + selection_mode etc. switches to adapt to different toolkits
"""

import sys
import os
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List, Dict, Any, Optional, Union, Tuple
import numpy as np
from datetime import datetime
import json
import re

# ========== Optional: Sentence Transformers ==========
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Warning: sentence-transformers not available. Using hashing fallback embedding.")

# ========== Relation Embedding Cache ==========
try:
    from kg_agent.embedding_cache import RelationEmbeddingCache
    EMBEDDING_CACHE_AVAILABLE = True
except ImportError:
    EMBEDDING_CACHE_AVAILABLE = False
    print("Warning: embedding_cache not available.")

# ========== Hybrid Retrieval System ==========
try:
    # try multiple import paths
    try:
        from memotime.kg_agent.hybrid_retrieval import HybridRetrieval, hybrid_retrieve_triples
    except ImportError:
        from kg_agent.hybrid_retrieval import HybridRetrieval, hybrid_retrieve_triples
    
    HYBRID_RETRIEVAL_AVAILABLE = True
    print("✅ Hybrid Retrieval System imported successfully")
except ImportError as e:
    HYBRID_RETRIEVAL_AVAILABLE = False
    print(f"Warning: hybrid_retrieval not available: {e}")

# ========== Your Dependencies ==========
from kg_agent.kg_ops import KG
from kg_agent.temporal_kg_toolkit import TemporalKGQuery, parse_time_to_range

# Import LLM Module
try:
    from kg_agent.llm import LLM
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    print("Warning: LLM module not available, using fallback selection method")

import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
class IntelligentRetrieval:
    """Intelligent Retrieval System (Refactored and Integrated Version)"""

    # -------- Hyperparameter Constants --------
    HASH_DIM = 2048
    NGRAM = 3

    REL_EXACT_BONUS = 0.5
    REL_PARTIAL_BONUS = 0.3
    REL_KEYWORD_BONUS = 0.1
    ACTION_MATCH_BONUS = 0.4

    WEIGHT_ENTITY = 0.3
    WEIGHT_REL = 0.7

    def __init__(self, db_path: str, embedding_model: str = "all-MiniLM-L6-v2", use_embedding_cache: bool = True, use_hybrid_retrieval: bool = False, raw_data_path: str = None):
        self.db_path = db_path
        self.kg_query = TemporalKGQuery(db_path)
        self.use_embedding_cache = use_embedding_cache
        self.use_hybrid_retrieval = use_hybrid_retrieval
        
        # Dynamic raw_data_path based on dataset
        if raw_data_path is None:
            try:
                from config import TPKGConfig
            except:
                try:
                    from ..config import TPKGConfig
                except:
                    TPKGConfig = None
            
            if TPKGConfig:
                # Get dataset config
                dataset_config = TPKGConfig.get_dataset_config()
                dataset_root = dataset_config['root']
                kg_format = dataset_config.get('kg_format', '4cols')
                
                if kg_format == '5cols':  # TimeQuestions
                    self.raw_data_path = os.path.join(dataset_root, 'full.txt')
                else:  # MultiTQ (4cols)
                    self.raw_data_path = os.path.join(dataset_root, 'full_fixed.txt')
                
                print(f"📁 Auto-set raw_data_path: {self.raw_data_path}")
            else:
                self.raw_data_path = str(Path(__file__).parent.parent.parent / "Data" / "full.txt")  # fallback
        else:
            self.raw_data_path = raw_data_path
        
        # relation embedding cache (dataset-specific)
        self.relation_cache = None
        if use_embedding_cache and EMBEDDING_CACHE_AVAILABLE:
            try:
                # Get dataset-specific cache directory
                try:
                    from config import TPKGConfig
                    dataset = TPKGConfig.DATASET
                    dataset_root = TPKGConfig.get_dataset_config()['root']
                    embedding_cache_dir = os.path.join(dataset_root, f"embedding_cache_{dataset.lower()}")
                    print(f"📁 Using dataset-specific embedding cache: {embedding_cache_dir}")
                except:
                    embedding_cache_dir = "embedding_cache"
                
                self.relation_cache = RelationEmbeddingCache(db_path, cache_dir=embedding_cache_dir)
                print("✅ Relation Embedding Cache System initialized successfully")
                
                # Check if pre-computation needed
                cache_info = self.relation_cache.get_cache_info()
                if cache_info['cached_relations'] == 0:
                    print("🔄 Start precomputing relation embeddings...")
                    self.relation_cache.precompute_all_embeddings()
                else:
                    print(f"✅ Found {cache_info['cached_relations']} cached relation embeddings")
            except Exception as e:
                print(f"❌ Relation Embedding Cache System initialization failed: {e}")
                self.relation_cache = None
        
        # traditional embedding model (as fallback)
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.embedding_model = SentenceTransformer(embedding_model)
                print(f"✅ Loaded embedding model: {embedding_model}")
            except Exception as e:
                print(f"❌ Loaded embedding model failed: {e}")
                self.embedding_model = None
        else:
            self.embedding_model = None
            print("⚠️ Using hashing trick as fallback embedding")
        
        # LLM
        if LLM_AVAILABLE:
            try:
                self.llm = LLM()
                print("✅ Loaded LLM model")
            except Exception as e:
                print(f"❌ Loaded LLM model failed: {e}")
                self.llm = None
        else:
            self.llm = None
            print("⚠️ Using fallback selection method")
        
        # Hybrid retrieval system
        self.hybrid_retrieval = None
        if use_hybrid_retrieval and HYBRID_RETRIEVAL_AVAILABLE:
            try:
                self.hybrid_retrieval = HybridRetrieval(db_path)
                print("✅ Hybrid Retrieval System initialized successfully")
            except Exception as e:
                print(f"⚠️ Hybrid Retrieval System initialization failed: {e}")
                self.hybrid_retrieval = None
        elif use_hybrid_retrieval:
            print("⚠️ Hybrid Retrieval System is not available, missing necessary dependencies")
    
    
    def analyze_retrieval_requirements(self, subquestion_obj: Any, method_name: str, params: Dict[str, Any], entity: Union[str, List[str]] = None) -> Dict[str, Any]:
        """
        Analyze the three core retrieval requirements of the question:
        1. Time dimension: before, after, between
        2. Sort dimension: first, last  
        3. Search key point: entity list
        
        Args:
            subquestion_obj: subquestion object
            method_name: toolkit method name
            params: toolkit parameters
            
        Returns:
            Dict: analysis result contains time, sort, entity information
        """
        analysis = {
            "time_dimension": {
                "type": None,  # "before", "after", "between"
                "value": None,  # specific time value
                "constraint": None  # time constraint type
            },
            "sort_dimension": {
                "type": None,  # "first", "last"
                "order": None  # "asc", "desc"
            },
            "entity_dimension": {
                "entities": [],  # entity list
                "primary_entity": None,  # primary entity
                "secondary_entities": []  # secondary entity
            },
            "toolkit_info": {
                "method": method_name,
                "params": params
            }
        }
        
        # 1. Analyze time dimension
        # first get time information from toolkit parameters
        time_value = params.get('reference_time') or params.get('after') or params.get('before') or params.get('between') or params.get('day') or params.get('month') or params.get('year')
        
        if time_value:
            # determine time constraint type based on toolkit type
            if method_name in ["find_after_first"]:
                analysis["time_dimension"]["type"] = "after"
                analysis["time_dimension"]["constraint"] = "after"
                analysis["time_dimension"]["value"] = time_value
            elif method_name in ["find_before_last"]:
                analysis["time_dimension"]["type"] = "before"
                analysis["time_dimension"]["constraint"] = "before"
                analysis["time_dimension"]["value"] = time_value
            elif method_name in ["find_between_range"]:
                analysis["time_dimension"]["type"] = "between"
                analysis["time_dimension"]["constraint"] = "between"
                analysis["time_dimension"]["value"] = time_value
            elif method_name in ["events_on_day"]:
                # for events_on_day, use equal constraint
                analysis["time_dimension"]["type"] = "equal"
                analysis["time_dimension"]["constraint"] = "equal"
                analysis["time_dimension"]["value"] = params.get('day') or params.get('date') or time_value
            elif method_name in ["events_in_month"]:
                # for events_in_month, use equal constraint
                analysis["time_dimension"]["type"] = "equal"
                analysis["time_dimension"]["constraint"] = "equal"
                analysis["time_dimension"]["value"] = params.get('month') or time_value
            elif method_name in ["events_in_year"]:
                # for events_in_year, use equal constraint
                analysis["time_dimension"]["type"] = "equal"
                analysis["time_dimension"]["constraint"] = "equal"
                analysis["time_dimension"]["value"] = params.get('year') or time_value
            elif method_name in ["find_direct_connection"]:
                # for direct connection, determine time constraint type based on parameters
                if params.get('after') and params.get('before'):
                    # if both after and before are provided, use between constraint
                    analysis["time_dimension"]["type"] = "between"
                    analysis["time_dimension"]["constraint"] = "between"
                    analysis["time_dimension"]["value"] = f"{params.get('after')}/{params.get('before')}"
                elif params.get('before') or params.get('reference_time'):
                    # if only before is provided, use before constraint
                    analysis["time_dimension"]["type"] = "before"
                    analysis["time_dimension"]["constraint"] = "before"
                    analysis["time_dimension"]["value"] = params.get('before') or params.get('reference_time')
                elif params.get('after'):
                    # if only after is provided, use after constraint
                    analysis["time_dimension"]["type"] = "after"
                    analysis["time_dimension"]["constraint"] = "after"
                    analysis["time_dimension"]["value"] = params.get('after')
                elif params.get('between'):
                    # between constraint
                    analysis["time_dimension"]["type"] = "between"
                    analysis["time_dimension"]["constraint"] = "between"
                    analysis["time_dimension"]["value"] = params.get('between')
                else:
                    # if there is no explicit time constraint parameter, but there is reference_time, infer from context
                    # usually reference_time is used for before_last scenario
                    analysis["time_dimension"]["type"] = "before"
                    analysis["time_dimension"]["constraint"] = "before"
                    analysis["time_dimension"]["value"] = time_value
        
        # if there is no time information in toolkit parameters, get from subquestion_obj
        if not analysis["time_dimension"]["value"] and hasattr(subquestion_obj, 'indicator') and hasattr(subquestion_obj.indicator, 'constraints'):
            constraints = subquestion_obj.indicator.constraints
            for constraint in constraints:
                if 'before(' in constraint:
                    analysis["time_dimension"]["type"] = "before"
                    analysis["time_dimension"]["constraint"] = "before"
                    analysis["time_dimension"]["value"] = params.get('before') or params.get('reference_time')
                elif 'after(' in constraint:
                    analysis["time_dimension"]["type"] = "after"
                    analysis["time_dimension"]["constraint"] = "after"
                    analysis["time_dimension"]["value"] = params.get('after') or params.get('reference_time')
                elif 'between(' in constraint:
                    analysis["time_dimension"]["type"] = "between"
                    analysis["time_dimension"]["constraint"] = "between"
                    analysis["time_dimension"]["value"] = params.get('between') or params.get('time_range')
                elif '=' in constraint:
                    # handle equal type constraint, like "t1 = 2007-01-14"
                    import re
                    equal_match = re.search(r't\d+\s*=\s*(.+)', constraint)
                    if equal_match:
                        time_value = equal_match.group(1).strip()
                        analysis["time_dimension"]["type"] = "equal"
                        analysis["time_dimension"]["constraint"] = "equal"
                        analysis["time_dimension"]["value"] = time_value
                        print(f"✅ extract equal constraint: {constraint} -> {time_value}")    
                elif '<' in constraint:
                    # handle before type constraint, like "t1 < 2008-10-22"
                    import re
                    before_match = re.search(r't\d+\s*<\s*(.+)', constraint)
                    if before_match:
                        time_value = before_match.group(1).strip()
                        analysis["time_dimension"]["type"] = "before"
                        analysis["time_dimension"]["constraint"] = "before"
                        analysis["time_dimension"]["value"] = time_value
                        print(f"✅ extract before constraint: {constraint} -> {time_value}")
                elif '>' in constraint:
                    # handle after type constraint, like "t1 > 2008-10-22"
                    import re
                    after_match = re.search(r't\d+\s*>\s*(.+)', constraint)
                    if after_match:
                        time_value = after_match.group(1).strip()
                        analysis["time_dimension"]["type"] = "after"
                        analysis["time_dimension"]["constraint"] = "after"
                        analysis["time_dimension"]["value"] = time_value
                        print(f"✅ extract after constraint: {constraint} -> {time_value}")
        
        # 2. Analyze sort dimension
        # first infer sort dimension from toolkit type
        if method_name in ["find_after_first"]:
            analysis["sort_dimension"]["type"] = "first"
            analysis["sort_dimension"]["order"] = "asc"
        elif method_name in ["find_before_last"]:
            analysis["sort_dimension"]["type"] = "last"
            analysis["sort_dimension"]["order"] = "desc"
        elif method_name in ["find_direct_connection"]:
            # for direct connection, if there is time parameter, default to last (get the latest connection)
            if time_value:
                analysis["sort_dimension"]["type"] = "last"
                analysis["sort_dimension"]["order"] = "desc"
        
        # if there is no sort information in toolkit parameters, get from subquestion_obj
        if not analysis["sort_dimension"]["type"] and hasattr(subquestion_obj, 'indicator') and hasattr(subquestion_obj.indicator, 'constraints'):
            constraints = subquestion_obj.indicator.constraints
            for constraint in constraints:
                if 'first(' in constraint:
                    analysis["sort_dimension"]["type"] = "first"
                    analysis["sort_dimension"]["order"] = "asc"
                elif 'last(' in constraint:
                    analysis["sort_dimension"]["type"] = "last"
                    analysis["sort_dimension"]["order"] = "desc"
        
        # 3. Analyze entity dimension
        # first use the passed entity parameter, otherwise get from params
        entity_to_analyze = entity if entity is not None else params.get('entity')
        if isinstance(entity_to_analyze, list):
            analysis["entity_dimension"]["entities"] = entity_to_analyze
            analysis["entity_dimension"]["primary_entity"] = entity_to_analyze[0] if entity_to_analyze else None
            analysis["entity_dimension"]["secondary_entities"] = entity_to_analyze[1:] if len(entity_to_analyze) > 1 else []
        else:
            analysis["entity_dimension"]["entities"] = [entity_to_analyze] if entity_to_analyze else []
            analysis["entity_dimension"]["primary_entity"] = entity_to_analyze
        
        # 4. Integrate toolkit information for comprehensive analysis
        analysis = self._integrate_toolkit_analysis(analysis, method_name, params)
        
        print(f"🔍 Basic retrieval requirement analysis:")
        print(f"    Toolkit parameters: {params}")
        print(f"    Entity: {entity}")
        # simplify analysis output, avoid over-complexity
        
        return analysis
    
    def _integrate_toolkit_analysis(self, analysis: Dict[str, Any], method_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Integrate toolkit information for comprehensive analysis
        """
        # supplement analysis based on toolkit type
        if method_name == "find_after_first":
            analysis["time_dimension"]["type"] = "after"
            analysis["sort_dimension"]["type"] = "first"
            analysis["sort_dimension"]["order"] = "asc"
        elif method_name == "find_before_last":
            analysis["time_dimension"]["type"] = "before"
            analysis["sort_dimension"]["type"] = "last"
            analysis["sort_dimension"]["order"] = "desc"
        elif method_name == "find_direct_connection":
            # direct connection needs two entities
            if len(analysis["entity_dimension"]["entities"]) >= 2:
                analysis["entity_dimension"]["primary_entity"] = analysis["entity_dimension"]["entities"][0]
                analysis["entity_dimension"]["secondary_entities"] = analysis["entity_dimension"]["entities"][1:]
        
        return analysis
    
    def llm_validate_retrieval_plan(self, analysis: Dict[str, Any], subquestion_text: str) -> Dict[str, Any]:
        """
        Use LLM to validate the retrieval plan, ensure the analysis is accurate
        """
        if not LLM_AVAILABLE:
            return analysis
        
        try:
            prompt = f"""
Please analyze whether the following retrieval requirements are accurate:

Question: {subquestion_text}

Current analysis:
- Time dimension: {analysis['time_dimension']}
- Sort dimension: {analysis['sort_dimension']}  
- Entity dimension: {analysis['entity_dimension']}

Please validate and optimize this analysis, return JSON format:
{{
    "time_dimension": {{"type": "before/after/between/none", "value": "specific time value or null", "constraint": "constraint type"}},
    "sort_dimension": {{"type": "first/last/none", "order": "asc/desc/none"}},
    "entity_dimension": {{"entities": ["entity list"], "primary_entity": "primary entity", "secondary_entities": ["secondary entity"]}},
    "confidence": 0.8,
    "suggestions": ["suggestion1", "suggestion2"]
}}

Note:
1. If the time dimension is not clear, please set type to "none", value to null
2. If the sort dimension is not clear, please set type to "none", order to "none"
3. The time value must be a specific date format (e.g. "2020-01-01") or null
4. Do not use fuzzy time description
"""
            
            response = LLM.call("You are a professional retrieval requirement analyst", prompt)
            
            # parse LLM response
            try:
                # try to extract JSON part
                response_clean = response.strip()
                
                # if the response contains ```json``` mark, extract the content
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
                
                # try to find the start and end of the JSON object
                if "{" in response_clean and "}" in response_clean:
                    start = response_clean.find("{")
                    end = response_clean.rfind("}") + 1
                    response_clean = response_clean[start:end]
                
                print(f"🔍 Cleaned LLM response: {response_clean[:200]}...")
                
                llm_analysis = json.loads(response_clean)
                
                # merge LLM analysis result - only update when LLM provides valid information
                for key in ["time_dimension", "sort_dimension", "entity_dimension"]:
                    if key in llm_analysis:
                        llm_value = llm_analysis[key]
                        # only update when LLM provides non-empty and valid information
                        if (llm_value.get("type") and llm_value.get("type") != "none") or \
                           (llm_value.get("value") and llm_value.get("value") is not None) or \
                           (llm_value.get("order") and llm_value.get("order") != "none"):
                            print(f"🔄 LLM update {key}: {llm_value}")
                            analysis[key].update(llm_value)
                        else:
                            print(f"⏭️ Skip LLM's {key} update (invalid value): {llm_value}")
                
                analysis["llm_validation"] = {
                    "confidence": llm_analysis.get("confidence", 0.8),
                    "suggestions": llm_analysis.get("suggestions", [])
                }
                
                print(f"🤖 LLM validation completed, confidence: {analysis['llm_validation']['confidence']}")
                
            except json.JSONDecodeError as e:
                print(f"⚠️ LLM response parsing failed: {e}")
                print(f"Original response: {response[:200]}...")
                print("Use original analysis")
            except Exception as e:
                print(f"⚠️ LLM response processing failed: {e}")
                print("Use original analysis")
                
        except Exception as e:
            print(f"⚠️ LLM validation failed: {e}")
        
        return analysis
    
    def intelligent_retrieve_with_analysis(self, entity: Union[str, List[str]], 
                                         subquestion: str,
                                         subquestion_obj: Optional[Any] = None,
                                         method_name: str = "retrieve_one_hop",
                                         **params) -> Dict[str, Any]:
        """
        Intelligent retrieval method with integrated analysis
        
        Args:
            entity: entity or entity list
            subquestion: subquestion text
            subquestion_obj: subquestion object
            method_name: toolkit method name
            **params: other parameters
            
        Returns:
            Dict: retrieval result and analysis information
        """
        print(f"🚀 start intelligent retrieval (integrated analysis)=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=")
        print(f"Entity: {entity} | Subquestion: {subquestion}")
        print(f"subq indicator: {subquestion_obj.indicator}")
        print(f"🚀 start intelligent retrieval (integrated analysis)=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=")
        
        # Step 1: analyze retrieval requirements
        analysis = self.analyze_retrieval_requirements(subquestion_obj, method_name, params, entity)
        
        # Step 2: skip LLM validation retrieval plan (disabled)
        # analysis = self.llm_validate_retrieval_plan(analysis, subquestion)
        print("⏭️ skip LLM intelligent analysis, use original parameters")
        
        # Step 3: determine retrieval strategy based on analysis result
        strategy = self._determine_retrieval_strategy(analysis)
        print(f"📋 retrieval strategy: {strategy}")
        
        # Step 4: execute intelligent retrieval
        result = self.intelligent_retrieve(
            entity=entity,
            subquestion=subquestion,
            time_constraint=analysis["time_dimension"].get("value"),
            constraint_type=analysis["time_dimension"].get("constraint", "after"),
            top_k=params.get('limit', 50),
            toolkit_type=method_name,
            subquestion_obj=subquestion_obj,
            time_order=analysis["sort_dimension"].get("order", "asc"),
            relation_select_mode="auto",
            selection_mode="auto"
        )
        
        # Step 5: append analysis result to return result
        if isinstance(result, dict):
            result["retrieval_analysis"] = analysis
            result["retrieval_strategy"] = strategy
        
        return result
    
    def _determine_retrieval_strategy(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Determine retrieval strategy based on analysis result
        """
        strategy = {
            "time_strategy": "none",
            "sort_strategy": "none", 
            "entity_strategy": "single",
            "relation_strategy": "auto"
        }
        
        # time strategy
        time_dim = analysis.get("time_dimension", {})
        if time_dim.get("type"):
            strategy["time_strategy"] = time_dim["type"]
        
        # sort strategy
        sort_dim = analysis.get("sort_dimension", {})
        if sort_dim.get("type"):
            strategy["sort_strategy"] = sort_dim["type"]
        
        # entity strategy
        entity_dim = analysis.get("entity_dimension", {})
        if len(entity_dim.get("entities", [])) > 1:
            strategy["entity_strategy"] = "multi"
        elif len(entity_dim.get("entities", [])) == 1:
            strategy["entity_strategy"] = "single"
        
        # relation strategy
        if strategy["entity_strategy"] == "multi":
            strategy["relation_strategy"] = "direct_connection"
        else:
            strategy["relation_strategy"] = "semantic_similarity"
        
        return strategy

    # ================== toolkit public entry (8 thin wrappers) ==================
    
    def execute_toolkit(self, method_name: str, **params) -> Dict[str, Any]:
        """
        Unified toolkit call interface, integrated problem analysis
        
        Args:
            method_name: toolkit method name
            **params: all parameters, including entity, limit, after, before etc.
        
        Returns:
            Dict: unified return format
        """
        # extract common parameters
        entity = params.get('entity')
        limit = params.get('limit', 100)
        subquestion_obj = params.get('subquestion_obj')
        
        # Step 1: analyze retrieval requirements
        print(f"🔍 start analyzing retrieval requirements...")
        analysis = self.analyze_retrieval_requirements(subquestion_obj, method_name, params)
        
        # Step 2: skip LLM validation retrieval plan (disabled)
        # if subquestion_obj and hasattr(subquestion_obj, 'text'):
        #     analysis = self.llm_validate_retrieval_plan(analysis, subquestion_obj.text)
        print("⏭️ skip LLM intelligent analysis, use original parameters")
        
        # Step 3: optimize parameters based on analysis result
        optimized_params = self._optimize_params_from_analysis(params, analysis)
        
        # Step 4: call the corresponding toolkit method
        result = self._call_toolkit_method(method_name, optimized_params)
        
        # Step 5: append analysis result to return result
        if isinstance(result, dict):
            result["retrieval_analysis"] = analysis
        
        return result
    
    def _optimize_params_from_analysis(self, params: Dict[str, Any], analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize parameters based on analysis result
        """
        optimized = params.copy()
        
        # optimize time parameters based on time dimension
        time_dim = analysis.get("time_dimension", {})
        if time_dim.get("type") == "before" and time_dim.get("value"):
            optimized["before"] = time_dim["value"]
        elif time_dim.get("type") == "after" and time_dim.get("value"):
            optimized["after"] = time_dim["value"]
        elif time_dim.get("type") == "between" and time_dim.get("value"):
            optimized["between"] = time_dim["value"]
        
        # optimize sort parameters based on sort dimension
        sort_dim = analysis.get("sort_dimension", {})
        if sort_dim.get("order"):
            optimized["time_order"] = sort_dim["order"]
        
        # optimize entity parameters based on entity dimension
        entity_dim = analysis.get("entity_dimension", {})
        if entity_dim.get("entities"):
            optimized["entity"] = entity_dim["entities"]
        
        print(f"🔧 parameters optimization completed: {list(optimized.keys())}")
        return optimized
    
    def _call_toolkit_method(self, method_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call the corresponding toolkit method
        """
        # extract common parameters
        entity = params.get('entity')
        limit = params.get('limit', 100)
        subquestion_obj = params.get('subquestion_obj')
        
        # call the corresponding toolkit method based on method name
        if method_name == "retrieve_one_hop":
            return self.OneHop(**params)
        elif method_name == "find_after_first":
            after = params.get('after') or params.get('reference_time')
            return self.AfterFirst(entity=entity, after=after, limit=limit, subquestion_obj=subquestion_obj)
        elif method_name == "find_before_last":
            before = params.get('before') or params.get('reference_time')
            return self.BeforeLast(entity=entity, before=before, limit=limit, subquestion_obj=subquestion_obj)
        elif method_name == "find_between_range":
            between = params.get('between') or params.get('time_range')
            return self.BetweenRange(entity=entity, between=between, limit=limit, subquestion_obj=subquestion_obj)
        elif method_name == "find_day_events":
            date = params.get('date') or params.get('same_day')
            return self.DayEvents(date=date, entity=entity, limit=limit, subquestion_obj=subquestion_obj)
        elif method_name == "find_month_events":
            month = params.get('month') or params.get('same_month')
            return self.MonthEvents(month=month, entity=entity, limit=limit, subquestion_obj=subquestion_obj)
        elif method_name == "find_direct_connection":
            entity1 = params.get('entity1') or (entity[0] if isinstance(entity, list) else entity)
            entity2 = params.get('entity2') or (entity[1] if isinstance(entity, list) and len(entity) > 1 else None)
            if not entity2:
                return {"error": "DirectConnection requires two entities"}
            return self.DirectConnection(entity1=entity1, entity2=entity2, limit=limit, subquestion_obj=subquestion_obj)
        elif method_name == "find_timeline":
            return self.Timeline(**params)
        else:
            return {"error": f"Unknown toolkit method: {method_name}"}

    def OneHop(self, entity: str,
               direction: str = "both",
               limit: int = 100,
               after: Optional[str] = None,
               before: Optional[str] = None,
               between: Optional[tuple] = None,
               same_day: Optional[str] = None,
               same_month: Optional[str] = None,
               sort_by_time: bool = True,
               subquestion_obj: Optional[Any] = None,
               **kwargs) -> Dict[str, Any]:
        """
        OneHop retrieval (optional time filtering)
        - relation_select_mode = 'all' (no top3 pruning, cover all)
        - selection_mode = 'none' (no forced selection, return top_k)
        """
        time_constraint, constraint_type, time_order = self._fold_time_args(
            after=after, before=before, between=between, same_day=same_day, same_month=same_month
        )
        return self.intelligent_retrieve(
            entity=entity,
            subquestion=f"OneHop {entity}",
            time_constraint=time_constraint,
            constraint_type=constraint_type,
            top_k=limit,
            toolkit_type="onehop",
            subquestion_obj=subquestion_obj,
            time_order=("asc" if sort_by_time else "desc"),
            relation_select_mode="all",
            selection_mode="none",
            limit=limit
        )

    def AfterFirst(self, entity: str, after: str, limit: int = 1, subquestion_obj: Optional[Any] = None, **kwargs) -> Dict[str, Any]:
        """
        AfterFirst retrieval (time strictly ascending select first; no LLM dependency)
        - relation_select_mode = 'all'
        - selection_mode = 'first'
        """
        return self.intelligent_retrieve(
            entity=entity,
            subquestion=f"First event after {after} for {entity}",
            time_constraint=after,
            constraint_type="after",
            top_k=max(limit, 1),
            toolkit_type="after_first",
            subquestion_obj=subquestion_obj,
            time_order="asc",
            relation_select_mode="auto",
            selection_mode="first",
            limit=limit
        )

    def BeforeLast(self, entity: str, before: str, limit: int = 1, subquestion_obj: Optional[Any] = None, **kwargs) -> Dict[str, Any]:
        """
        BeforeLast retrieval (time strictly descending select first; no LLM dependency)
        - relation_select_mode = 'all'
        - selection_mode = 'last'
        """
        return self.intelligent_retrieve(
            entity=entity,
            subquestion=f"Last event before {before} for {entity}",
            time_constraint=before,
            constraint_type="before",
            top_k=max(limit, 1),
            toolkit_type="before_last",
            subquestion_obj=subquestion_obj,
            time_order="desc",
            relation_select_mode="all",
            selection_mode="last",
            limit=limit
        )

    def BetweenRange(self, entity: str, between: tuple, limit: int = 50, subquestion_obj: Optional[Any] = None, **kwargs) -> Dict[str, Any]:
        """
        BetweenRange retrieval (ascending)
        - relation_select_mode = 'all'
        - selection_mode = 'none'
        """
        assert isinstance(between, (list, tuple)) and len(between) == 2, "between needs (start, end)"
        return self.intelligent_retrieve(
            entity=entity,
            subquestion=f"Events between {between} for {entity}",
            time_constraint=f"{between[0]}/{between[1]}",
            constraint_type="between",
            top_k=limit,
            toolkit_type="between_range",
            subquestion_obj=subquestion_obj,
            time_order="asc",
            relation_select_mode="all",
            selection_mode="none",
            limit=limit
        )

    def DayEvents(self, date: str, entity: Optional[str] = None, limit: int = 200, subquestion_obj: Optional[Any] = None, **kwargs) -> Dict[str, Any]:
        """
        DayEvents retrieval (equal type time constraint)
        - use intelligent retrieval system to process equal type time constraint
        """
        if entity:
            # use intelligent retrieval system to process equal type constraint
            return self.intelligent_retrieve(
                entity=entity,
                subquestion=f"Events on {date} for {entity}",
                time_constraint=date,
                constraint_type="equal",
                top_k=limit,
                toolkit_type="events_on_day",
                subquestion_obj=subquestion_obj,
                time_order="asc",
                relation_select_mode="all",
                selection_mode="none",
                limit=limit
            )
        if hasattr(self.kg_query, "events_on_day"):
            try:
                res = self.kg_query.events_on_day(day=date, limit=limit)
                return {"selected_path": {}, "top_3_paths": [], "total_paths": len(getattr(res, "edges", [])), "raw": res}
            except Exception as e:
                return {"error": f"events_on_day is not available: {e}"}
        return {"error": "DayEvents needs entity or underlying events_on_day support"}

    def MonthEvents(self, month: str, entity: Optional[str] = None, limit: int = 200, subquestion_obj: Optional[Any] = None, **kwargs) -> Dict[str, Any]:
        """
        MonthEvents retrieval (equal type time constraint)
        - use intelligent retrieval system to process equal type time constraint
        """
        if entity:
            # use intelligent retrieval system to process equal type constraint
            return self.intelligent_retrieve(
                entity=entity,
                subquestion=f"Events in {month} for {entity}",
                time_constraint=month,
                constraint_type="equal",
                top_k=limit,
                toolkit_type="events_in_month",
                subquestion_obj=subquestion_obj,
                time_order="asc",
                relation_select_mode="all",
                selection_mode="none",
                limit=limit
            )
        if hasattr(self.kg_query, "events_in_month"):
            try:
                res = self.kg_query.events_in_month(month=month, limit=limit)
                return {"selected_path": {}, "top_3_paths": [], "total_paths": len(getattr(res, "edges", [])), "raw": res}
            except Exception as e:
                return {"error": f"events_in_month is not available: {e}"}
        return {"error": "MonthEvents needs entity or underlying events_in_month support"}

    def YearEvents(self, year: str, entity: Optional[str] = None, limit: int = 200, subquestion_obj: Optional[Any] = None, **kwargs) -> Dict[str, Any]:
        """
        YearEvents retrieval (equal type time constraint)
        - use intelligent retrieval system to process equal type time constraint
        """
        if entity:
            # use intelligent retrieval system to process equal type constraint
            return self.intelligent_retrieve(
                entity=entity,
                subquestion=f"Events in {year} for {entity}",
                time_constraint=year,
                constraint_type="equal",
                top_k=limit,
                toolkit_type="events_in_year",
                subquestion_obj=subquestion_obj,
                time_order="asc",
                relation_select_mode="all",
                selection_mode="none",
                limit=limit
            )
        if hasattr(self.kg_query, "events_in_year"):
            try:
                res = self.kg_query.events_in_year(year=year, limit=limit)
                return {"selected_path": {}, "top_3_paths": [], "total_paths": len(getattr(res, "edges", [])), "raw": res}
            except Exception as e:
                return {"error": f"events_in_year is not available: {e}"}
        return {"error": "YearEvents needs entity or underlying events_in_year support"}

    def DirectConnection(self, entity1: str, entity2: str, limit: int = 200, direction: str = "both", subquestion_obj: Optional[Any] = None, **kwargs) -> Dict[str, Any]:
        """
        DirectConnection retrieval (two entities directly connected priority)
        - multiple seed retrieval channels (built-in direct connection priority + entity similarity filtering)
        - relation_select_mode = 'all'
        - selection_mode = 'none'
        """
        return self.intelligent_retrieve(
            entity=[entity1, entity2],
            subquestion=f"Direct connection between {entity1} and {entity2}",
            time_constraint=None,
            constraint_type="after",
            
            top_k=limit,
            toolkit_type="direct_connection",
            subquestion_obj=subquestion_obj,
            time_order="asc",
            relation_select_mode="all",
            selection_mode="none",
            limit=limit
        )

    def Timeline(self, entity: str, after: Optional[str] = None, before: Optional[str] = None, limit: int = 100, subquestion_obj: Optional[Any] = None, **kwargs) -> Dict[str, Any]:
        """
        Timeline retrieval (optional after / before), default ascending
        - relation_select_mode = 'all'
        - selection_mode = 'none'
        """
        time_constraint, constraint_type, _ = self._fold_time_args(after=after, before=before)
        return self.intelligent_retrieve(
            entity=entity,
            subquestion=f"Timeline for {entity}",
            time_constraint=time_constraint,
            constraint_type=constraint_type,
            top_k=limit,
            toolkit_type="timeline",
            time_order="asc",
            relation_select_mode="all",
            selection_mode="none",
            limit=limit
        )

    # ================== embedding ==================
    def _tokenize(self, text: str) -> List[str]:
        t = (text or "").lower()
        if len(t) <= self.NGRAM:
            return [t]
        return [t[i:i + self.NGRAM] for i in range(len(t) - self.NGRAM + 1)]

    def _sort_by_time(self, items: List[Dict[str, Any]],
                      key_fn=None,
                      order: str = "asc") -> List[Dict[str, Any]]:
        """Sort items by time using time_start_epoch (Unix timestamp)
        
        Works for both MultiTQ (precise timestamp) and TimeQuestions (year timestamp)
        """
        reverse = (order == "desc")
        
        def _key(x):
            t_epoch = x.get('time_start_epoch', 0)
            t_epoch = 0 if t_epoch is None else t_epoch
            return (t_epoch, str(x.get('relation', '')), str(x.get('tail', '')))
        
        return sorted(items, key=_key, reverse=reverse)

    def get_embedding(self, text: str) -> np.ndarray:
        if self.embedding_model:
            return self.embedding_model.encode(text)
        vec = np.zeros(self.HASH_DIM, dtype=np.float32)
        for tok in self._tokenize(text):
            idx = abs(hash(tok)) % self.HASH_DIM
            vec[idx] += 1.0
        n = np.linalg.norm(vec)
        return vec / (n + 1e-8)

    # ================== parse expected relations / score (keep original implementation) ==================
    def _extract_relations_from_subquestion_obj(self, subquestion_obj: Any) -> List[str]:
        relations = []
        try:
            if hasattr(subquestion_obj, 'indicator') and hasattr(subquestion_obj.indicator, 'edges'):

                for i, edge in enumerate(subquestion_obj.indicator.edges):
                    if hasattr(edge, 'rel') and edge.rel:
                        relations.append(edge.rel)
                        print(f"   Edge {i}: {edge.rel}")
                    else:
                        print(f"   Edge {i}: (no relation or problematic relation value)")
            elif hasattr(subquestion_obj, 'relations'):
                relations = list(subquestion_obj.relations)
                # print(f"🔍 DEBUG: extract from subquestion.relations: {relations}")
            elif isinstance(subquestion_obj, dict):
                indicator = subquestion_obj.get('indicator', {})
                edges = indicator.get('edges', [])
                # print(f"🔍 DEBUG: extract from subquestion.indicator.edges... ")
                for edge in edges:
                    if isinstance(edge, dict) and 'rel' in edge:
                        relations.append(edge['rel'])
                        print(f"   relation: {edge['rel']}")
                    elif hasattr(edge, 'rel'):
                        relations.append(edge.rel)
                        print(f"   relation: {edge.rel}")
        except Exception as e:
            print(f"⚠️ extract relations from subquestion object failed: {e}")
            
        print(f"🔍 DEBUG: _extract_relations_from_subquestion_obj extracted final relations: {relations}")
        return list(set([r for r in relations if r]))
    
    def _extract_indicators_from_subquestion(self, subquestion: str) -> List[str]:
        indicators = []
        action_words = [
            'want', 'express', 'negotiate', 'visit', 'meeting', 'talk',
            'discuss', 'request', 'demand', 'agree', 'sign', 'cooperate',
            'support', 'oppose', 'criticize', 'praise', 'help', 'aid', 'meet'
        ]
        q = (subquestion or "").lower()
        for w in action_words:
            if w in q:
                indicators.append(w)
        for w in (subquestion or "").split():
            lw = w.lower()
            if lw in ['negotiate', 'negotiating', 'negotiated']:
                indicators.append('negotiate')
            elif lw in ['visit', 'visited', 'visiting']:
                indicators.append('visit')
            elif lw in ['meet', 'meeting', 'met']:
                indicators.append('meet')
        return list(set(indicators))
    
    def _calculate_relationship_similarity(self, path: Dict[str, Any], 
                                        subquestion: str,
                                        expected_relations: List[str],
                                        question_embedding: np.ndarray) -> float:
        try:
            relation = path.get('relation', '')
            if not relation:
                return 0.0
            relation_emb = self.get_embedding(relation)
            denom = (np.linalg.norm(question_embedding) * np.linalg.norm(relation_emb) + 1e-8)
            relation_similarity = float(np.dot(question_embedding, relation_emb) / denom)
            relation_bonus = 0.0
            rel_lower = relation.lower()
            q_lower = (subquestion or '').lower()
            for exp_rel in expected_relations or []:
                exp_lower = exp_rel.lower()
                if exp_lower == rel_lower:
                    relation_bonus += self.REL_EXACT_BONUS
                elif exp_lower in rel_lower or rel_lower in exp_lower:
                    relation_bonus += self.REL_PARTIAL_BONUS
                for kw in exp_lower.split('_'):
                    if len(kw) > 3 and kw in q_lower:
                        relation_bonus += self.REL_KEYWORD_BONUS
                        break
            action_similarity = self.ACTION_MATCH_BONUS if rel_lower in q_lower else 0.0
            relationship_similarity = (relation_similarity * 0.4 + 
                                    action_similarity * 0.2 + 
                                    relation_bonus * 0.4)
            return float(min(max(relationship_similarity, 0.0), 1.0))
        except Exception as e:
            print(f"⚠️ relationship similarity calculation error: {e}")
            return 0.0
    
    def _calculate_relationship_indicator_similarity(self, relation, expected_relations, subquestion, embedding):
        max_score = 0.0
        
        # simplify debug information
        # print(f"🔍 similarity calculation: {relation}")
        # print(f"   expected relations: {expected_relations}")
        # print(f"   using embedding cache: {self.relation_cache is not None}")
        # print(f"   embedding available: {embedding is not None}")
        
        # method 1: using embedding cache (similarity between relations)
        if self.relation_cache is not None and expected_relations:
            best_cache_similarity = 0.0
            best_expected_rel = ""
            
            for exp_rel in expected_relations:
                similarity = self.relation_cache.calculate_similarity(relation, exp_rel)
                if similarity > best_cache_similarity:
                    best_cache_similarity = similarity
                    best_expected_rel = exp_rel
            
            if best_cache_similarity > 0:
                # print(f"   🚀 cache similarity {relation}: {best_cache_similarity:.4f}")
                max_score = max(max_score, best_cache_similarity)
        
        # method 2: using subquestion embedding (similarity between relation and subquestion)
        if embedding is not None:
            # better preprocess relation name to improve semantic similarity
            rel_text = relation.replace("_", " ").lower()
            
            # try multiple text variants to improve similarity
            rel_variants = [
                rel_text,  # original processed text
                relation.lower(),  # original lowercase
                relation.replace("_", " ").lower(),  # underscore replaced with space
                relation.replace("_", "").lower(),  # remove underscore
            ]
            
            # add more intelligent variants, handle common relation patterns
            if "_a_" in relation.lower():
                # handle "Make_a_visit" -> "make visit" etc.
                smart_variant = relation.lower().replace("_a_", " ").replace("_", " ")
                rel_variants.append(smart_variant)
            
            if "_" in relation.lower():
                # handle "Host_a_visit" -> "host visit" etc.
                smart_variant2 = relation.lower().replace("_a_", " ").replace("_", " ").strip()
                if smart_variant2 not in rel_variants:
                    rel_variants.append(smart_variant2)
            
            # remove duplicates
            rel_variants = list(dict.fromkeys(rel_variants))
            
            best_sim = 0.0
            best_variant = ""
            
            for variant in rel_variants:
                if variant:  # ensure variant is not empty
                    rel_emb = self.get_embedding(variant)
                    e_norm = np.linalg.norm(embedding)
                    r_norm = np.linalg.norm(rel_emb)
                    sim = float(embedding.dot(rel_emb) / (e_norm * r_norm + 1e-8)) if e_norm > 0 and r_norm > 0 else 0.0
                    
                    if sim > best_sim:
                        best_sim = sim
                        best_variant = variant
            
            # print(f"   🧠 BERT similarity {relation}: {best_sim:.4f}")
            max_score = max(max_score, best_sim)
        
        final_score = max(max_score, 0.01)
        return final_score

    # ================== hybrid retrieval interface ==================
    def _convert_hybrid_results_to_paths(self, hybrid_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        convert hybrid retrieval results to standard path format
        
        Args:
            hybrid_result: hybrid retrieval results
            
        Returns:
            list of standard path format
        """
        paths = []
        
        try:
            triples = hybrid_result.get('triples', [])
            facts = hybrid_result.get('facts', [])
            scores = hybrid_result.get('scores', [])
            
            for i, triple in enumerate(triples):
                # parse triple format: (head, relation, tail)
                if triple.startswith('(') and triple.endswith(')'):
                    content = triple[1:-1]  # remove parentheses
                    parts = [part.strip() for part in content.split(',')]
                    
                    if len(parts) == 3:
                        head, relation, tail = parts
                        
                        # extract time information from corresponding fact
                        time_start = "unknown"
                        if i < len(facts):
                            fact = facts[i]
                            # extract time from fact: "... at YYYY-MM-DD"
                            if " at " in fact:
                                time_start = fact.split(" at ")[-1].strip()
                        
                        # build standard path format
                        path = {
                            'head': head,
                            'relation': relation,
                            'tail': tail,
                            'time_start': time_start,
                            'similarity': scores[i] if i < len(scores) else 0.0,
                            'source': 'hybrid_retrieval'
                        }
                        
                        # try to parse timestamp
                        try:
                            from datetime import datetime
                            time_obj = datetime.strptime(time_start, "%Y-%m-%d")
                            path['time_start_epoch'] = int(time_obj.timestamp())
                        except:
                            path['time_start_epoch'] = 0
                        
                        paths.append(path)
            
            print(f"🔄 hybrid retrieval results conversion: {len(triples)} triples → {len(paths)} paths")
            
        except Exception as e:
            print(f"⚠️ hybrid retrieval results conversion failed: {e}")
        
        return paths
    async def hybrid_retrieve_for_question(self, question: str, top_k: int = 50, re_rank: bool = False) -> Dict[str, Any]:
        """
        use hybrid retrieval system to retrieve related triples for question
        
        Args:
            question: input question
            top_k: number of triples to return
            re_rank: whether to use time re-ranking
            
        Returns:
            dictionary containing related triples
        """
        if not self.hybrid_retrieval:
            return {'question': question, 'triples': [], 'error': 'Hybrid retrieval not available'}
        
        try:
            # ensure hybrid retrieval system is initialized
            if not self.hybrid_retrieval.is_loaded:
                print("🔄 initialize hybrid retrieval system...")
                await self.hybrid_retrieval.initialize()
            
            # execute hybrid retrieval
            result = await self.hybrid_retrieval.retrieve_triples(question, top_k, re_rank)
            
            print(f"🔍 hybrid retrieval completed: found {len(result.get('triples', []))} related triples")
            return result
            
        except Exception as e:
            print(f"❌ hybrid retrieval failed: {e}")
            return {'question': question, 'triples': [], 'error': str(e)}
    
    def get_hybrid_retrieval_stats(self) -> Dict[str, Any]:
        """get hybrid retrieval system statistics"""
        if not self.hybrid_retrieval:
            return {'available': False, 'reason': 'Hybrid retrieval not initialized'}
        
        stats = self.hybrid_retrieval.get_stats()
        stats['available'] = True
        return stats

    # ================== semantic similarity calculation ==================
    def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts using BERT"""
        try:
            embedding1 = self.get_text_embedding(text1)
            embedding2 = self.get_text_embedding(text2)
            
            # Calculate cosine similarity
            similarity = np.dot(embedding1, embedding2) / (
                np.linalg.norm(embedding1) * np.linalg.norm(embedding2) + 1e-8
            )
            return float(similarity)
        except Exception as e:
            print(f"Failed to calculate semantic similarity: {e}")
            return 0.0
    
    def get_text_embedding(self, text: str) -> np.ndarray:
        """Get text embedding vector using BERT model"""
        if self.embedding_model:
            return self.embedding_model.encode(text, convert_to_tensor=False)
        else:
            # Fallback: simple word frequency vector
            words = text.lower().replace("_", " ").split()
            unique_words = list(set(words))
            if not unique_words:
                return np.zeros(1)
            vector = np.zeros(len(unique_words))
            for word in words:
                if word in unique_words:
                    vector[unique_words.index(word)] += 1
            return vector / (np.linalg.norm(vector) + 1e-8)

    # ================== semantic filtering ==================
    def _2e_semantic_pruning(self, paths: List[Dict[str, Any]], entity_list: Any, score_threshold: float = None) -> List[Dict[str, Any]]:
        """
        semantic filtering based on two entities
        check if the head and tail of the path are both related to the query entities
        
        Args:
            paths: candidate path list
            entity_list: query entity list (length >= 2)
            score_threshold: similarity threshold (None for auto-detect based on dataset)
        
        Returns:
            filtered path list
        """
        # Auto-detect threshold based on dataset if not provided
        if score_threshold is None:
            try:
                from config import TPKGConfig
                if TPKGConfig.DATASET == "TimeQuestions":
                    score_threshold = 0.3  # Lower threshold for TimeQuestions (more recall)
                else:
                    score_threshold = 0.75  # Higher threshold for MultiTQ (better precision)
                print(f"🎯 Auto-selected threshold: {score_threshold} for dataset {TPKGConfig.DATASET}")
            except:
                score_threshold = 0.4  # Fallback default
                print(f"⚠️ Could not detect dataset, using default threshold: {score_threshold}")
        
        if not paths or not entity_list or len(entity_list) < 2:
            return paths
        
        try:
            # calculate the semantic similarity between each path and the query entities
            return_path = []
            for i, path in enumerate(paths):
                heads = path.get('heads', [path.get('head', '')])
                head_str = " ".join(heads) if isinstance(heads, list) else heads
                tails = path.get('tails', [path.get('tail', '')])
                tail_str = " ".join(tails) if isinstance(tails, list) else tails

                query1 = entity_list[0]
                query2 = entity_list[1]
                
                # calculate the similarity between the head and the two query entities, take the maximum value
                entity1_score = max(
                    self.calculate_semantic_similarity(heads, query1), 
                    self.calculate_semantic_similarity(heads, query2)
                )
                # calculate the similarity between the tail and the two query entities, take the maximum value
                entity2_score = max(
                    self.calculate_semantic_similarity(tails, query1), 
                    self.calculate_semantic_similarity(tails, query2)
                )
                # take the minimum value of the two scores (ensure head and tail are related)
                entity_score = min(entity1_score, entity2_score)

                # Filter based on dynamic threshold
                if entity_score >= score_threshold:
                    return_path.append(path)
                # Debug: Log filtered paths (first few)
                elif i < 5:
                    print(f"  ❌ Filtered: {head_str[:50]} -> {path.get('relation', '')} -> {tail_str[:50]} (score: {entity_score:.3f} < {score_threshold})")
            
            # If filtering is too aggressive (filtered > 90%), keep top 50% by score
            if len(return_path) < len(paths) * 0.1 and len(paths) > 10:
                print(f"⚠️ Filtering too aggressive ({len(return_path)}/{len(paths)} kept), keeping top 50% by score")
                # Calculate scores for all paths
                scored_paths = []
                for path in paths:
                    heads = path.get('heads', [path.get('head', '')])
                    tails = path.get('tails', [path.get('tail', '')])
                    query1, query2 = entity_list[0], entity_list[1]
                    entity1_score = max(
                        self.calculate_semantic_similarity(heads, query1),
                        self.calculate_semantic_similarity(heads, query2)
                    )
                    entity2_score = max(
                        self.calculate_semantic_similarity(tails, query1),
                        self.calculate_semantic_similarity(tails, query2)
                    )
                    score = min(entity1_score, entity2_score)
                    scored_paths.append((score, path))
                # Sort by score and keep top 50%
                scored_paths.sort(reverse=True, key=lambda x: x[0])
                return_path = [p for _, p in scored_paths[:max(5, len(scored_paths) // 2)]]
            
            return return_path if return_path else paths
            
        except Exception as e:
            print(f"⚠️ 2 entity semantic filtering failed: {e}")
            import traceback
            traceback.print_exc()
            return paths


    def _semantic_pruning(self, paths: List[Dict[str, Any]], subquestion_obj: Any, top: int = 100) -> List[Dict[str, Any]]:
        """
        semantic filtering based on subquestion indicator
        convert indicator to edge format, match with current path, select topN most related results
        """
        if not paths or not subquestion_obj:
            return paths
        
        try:
            # 1. extract indicator information from subquestion_obj
            indicator_edges = self._extract_indicator_edges(subquestion_obj)
            if not indicator_edges:
                print("🔍 no indicator information, skip semantic filtering")
                
            
            print(f"🔍 extracted {len(indicator_edges)} indicator edges")
            
            # # 2. convert indicator to standard edge format
            # indicator_standard_edges = self._convert_indicator_to_standard_edges(indicator_edges)
            # print(f"🔍 converted to {len(indicator_standard_edges)} standard indicator edges")
            
            # 3. calculate the semantic similarity between each path and the indicator
            path_scores = []
            for i, path in enumerate(paths):
                indicator_edge = indicator_edges[0]
                # print(indicator_edges)
                heads = path.get('heads', [path.get('head', '')])
                head_str = ", ".join(heads)
                tails = path.get('tails', [path.get('tail', '')])
                tail_str = ", ".join(tails)
                indicator_path = f"{indicator_edge.get('subj', '')} - {indicator_edge.get('rel', '')} - {indicator_edge.get('obj', '')}"
                path_path = f"{head_str} - {path.get('relation', '')} - {tail_str}"
                
                indicator_entity_path = f"{indicator_edge.get('subj', '')} - {indicator_edge.get('obj', '')}"
                path_entity_path = f"{head_str} - {tail_str}"


                entity_score = self.calculate_semantic_similarity(indicator_entity_path, path_entity_path)
                if entity_score < 0.5:
                #     # print(indicator_entity_path)
                #     # print(path_entity_path)
                #     # print(entity_score)
                    continue
                score = self.calculate_semantic_similarity(indicator_path, path_path)
               

                path_scores.append((i, score))
            
            # 4. sort by similarity, select topN
            path_scores.sort(key=lambda x: x[1], reverse=True)
            top_indices = [idx for idx, _ in path_scores[:top]]
            filtered_paths = [paths[i] for i in top_indices]
            
            print(f"🔍 semantic filtering: {len(paths)} -> {len(filtered_paths)} paths")
            
            # 5. display top5 similarity scores
            if path_scores:
                print(f"🔍 Top5 semantic similarity: {[(i, f'{score:.4f}') for i, score in path_scores[:5]]}")
            
            return filtered_paths
            
        except Exception as e:
            print(f"⚠️ semantic filtering failed: {e}")
            return paths
    def _semantic_filter(self, question: str, paths: List[Dict[str, Any]], top_k_value: int = 40) -> List[Dict[str, Any]]:

        if len(paths) >= top_k_value:
            model = SentenceTransformer('msmarco-distilbert-base-tas-b')
            candidate_sentences = []
            for path in paths:
                heads = path.get('heads', [path.get('head', '')])
                head_str = ", ".join(heads)
                tails = path.get('tails', [path.get('tail', '')])
                tail_str = ", ".join(tails)
                path_path = f"{head_str} - {path.get('relation', '')} - {tail_str}"
                candidate_sentences.append(path_path)

            print("none-LLM model loaded")

            candidate_embeddings = model.encode(candidate_sentences, batch_size=64, show_progress_bar=True)
            query_embedding = model.encode([question])

            similarities = cosine_similarity(query_embedding, candidate_embeddings)[0]

            top_k = top_k_value
            top_k_indices = similarities.argsort()[-top_k:][::-1]

            print(f"finish obtained the top {top_k} sentences.\n") 
            paths = [paths[i] for i in top_k_indices]
        return paths
    def _extract_indicator_edges(self, subquestion_obj: Any) -> List[Dict[str, Any]]:
        """extract indicator edges from subquestion_obj"""
        edges = []
        try:
            if hasattr(subquestion_obj, 'indicator') and hasattr(subquestion_obj.indicator, 'edges'):
                for edge in subquestion_obj.indicator.edges:
                    if hasattr(edge, 'rel') and edge.rel:
                        edge_dict = {
                            'subj': getattr(edge, 'subj', ''),
                            'rel': getattr(edge, 'rel', ''),
                            'obj': getattr(edge, 'obj', ''),
                            'time_var': getattr(edge, 'time_var', '')
                        }
                        edges.append(edge_dict)
                        print(f"🔍 extracted indicator edge: {edge_dict}")
        except Exception as e:
            print(f"⚠️ extracted indicator edges failed: {e}")
        return edges
    
    def _convert_indicator_to_standard_edges(self, indicator_edges: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """convert indicator edges to standard edge format"""
        standard_edges = []
        for edge in indicator_edges:
            # convert indicator format to standard edge format
            standard_edge = {
                'head': edge.get('subj', ''),
                'relation': edge.get('rel', ''),
                'tail': edge.get('obj', ''),
                'time_start': '',  # indicator usually has no specific time
                'time_start_epoch': 0,
                'similarity': 1.0,  # indicator similarity set to 1.0
                'is_indicator': True  # marked as indicator
            }
            standard_edges.append(standard_edge)
        return standard_edges
    

        
        
    
    def _calculate_relation_similarity(self, rel1: str, rel2: str) -> float:
        """calculate relation similarity"""
        if not rel1 or not rel2:
            return 0.0
        
        # use embedding cache to calculate relation similarity
        if self.relation_cache:
            return self.relation_cache.calculate_similarity(rel1, rel2)
        
        # fallback to string similarity
        rel1_norm = rel1.lower().replace("_", " ")
        rel2_norm = rel2.lower().replace("_", " ")
        
        if rel1_norm == rel2_norm:
            return 1.0
        elif rel1_norm in rel2_norm or rel2_norm in rel1_norm:
            return 0.5
        else:
            return 0.0
    
    def _calculate_entity_similarity(self, entity1: str, entity2: str) -> float:
        """calculate entity similarity"""
        if not entity1 or not entity2:
            return 0.0
        
        # fully match
        if entity1 == entity2:
            return 1.0
        
        # partial match
        entity1_norm = entity1.lower().replace("_", " ").replace("-", " ")
        entity2_norm = entity2.lower().replace("_", " ").replace("-", " ")
        
        if entity1_norm == entity2_norm:
            return 1.0
        elif entity1_norm in entity2_norm or entity2_norm in entity1_norm:
            return 0.7
        else:
            # use simple edit distance
            return self._calculate_edit_distance_similarity(entity1_norm, entity2_norm)
    
    def _calculate_time_similarity(self, time1: str, time2: str) -> float:
        """calculate time similarity"""
        if not time1 or not time2:
            return 0.0
        
        # time fully match
        if time1 == time2:
            return 1.0
        
        # time partial match (same year)
        if len(time1) >= 4 and len(time2) >= 4:
            if time1[:4] == time2[:4]:
                return 0.8
        
        return 0.0
    
    def _calculate_edit_distance_similarity(self, s1: str, s2: str) -> float:
        """calculate edit distance similarity"""
        if not s1 or not s2:
            return 0.0
        
        # simple edit distance calculation
        m, n = len(s1), len(s2)
        if m == 0:
            return 0.0 if n > 0 else 1.0
        if n == 0:
            return 0.0
        
        # create DP table
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # initialize
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        # fill DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i-1] == s2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
        
        # calculate similarity
        max_len = max(m, n)
        similarity = 1.0 - (dp[m][n] / max_len)
        return max(0.0, similarity)

    # ================== aggregate / time processing (keep and enhance sorting) ==================
    def aggregate_paths(self, paths: List[Dict[str, Any]], seed_entities: List[str] = None) -> List[Dict[str, Any]]:
        """
        aggregate path logic:
        1. undirected relation merge: [A, rel, B] and [B, rel, A] merge into one
        2. same tail merge: [A, rel, C] and [B, rel, C] merge into [[A,B], rel, C]
        3. standardization: seed entity put in head position, non-seed entity put in tail position
        """
        if not paths:
            return []
        
        print(f"🔍 aggregate start: {len(paths)} original paths")
        
        # get seed entity set
        if seed_entities is None:
            seed_entities = []
        print(f"🔍 aggregate start: {seed_entities}")
        # exit()
        seed_set = set(seed_entities)
        seed_first = set(seed_entities[0])
        
        # standardize path: ensure seed entity in head position
        normalized_paths = []
        for p in paths:
            head = p.get('head', '')
            tail = p.get('tail', '')
            relation = p.get('relation', '')
            
            # if tail is seed and head is not, swap position
            if tail in seed_set and head not in seed_set:
                normalized_path = p.copy()
                normalized_path['head'] = tail
                normalized_path['tail'] = head
                normalized_paths.append(normalized_path)
            elif head in seed_set and tail not in seed_set:
                normalized_path = p.copy()
                normalized_path['head'] = head
                normalized_path['tail'] = tail
                normalized_paths.append(normalized_path)
            else:
                if tail in seed_first and head not in seed_first:
                    normalized_path = p.copy()
                    normalized_path['head'] = tail
                    normalized_path['tail'] = head
                    normalized_paths.append(normalized_path)
                else:
                    normalized_paths.append(p)
        
        # aggregate logic: two steps
        # step 1: undirected relation merge [A,rel,B] and [B,rel,A]
        # step 2: same tail merge [A,rel,C] and [B,rel,C] -> [[A,B],rel,C]
        
        # step 1: undirected relation standardization
        undirected_aggregated = {}
        for p in normalized_paths:
            relation = p.get('relation', '')
            head = p.get('head', '')
            tail = p.get('tail', '')
            time_start = p.get('time_start', '')
            time_epoch = p.get('time_start_epoch', 0)
            similarity = p.get('similarity', 0.0)
            
            # create undirected relation standardization key
            entity_pair = tuple(sorted([head, tail]))
            key = (relation, entity_pair, time_start)
            
            if key not in undirected_aggregated:
                undirected_aggregated[key] = {
                    'relation': relation,
                    'head': head,
                    'tail': tail,
                    'time_start': time_start,
                    'time_start_epoch': time_epoch,
                    'similarities': [similarity],
                    'count': 1,
                    'original_paths': [p]
                }
            else:
                # merge undirected relation, keep the direction of the first path
                undirected_aggregated[key]['count'] += 1
                undirected_aggregated[key]['similarities'].append(similarity)
                undirected_aggregated[key]['original_paths'].append(p)
                
                # update earliest time
                if time_epoch > 0 and (undirected_aggregated[key]['time_start_epoch'] == 0 or time_epoch < undirected_aggregated[key]['time_start_epoch']):
                    undirected_aggregated[key]['time_start_epoch'] = time_epoch
                    undirected_aggregated[key]['time_start'] = time_start
        
        # step 2: comprehensive aggregate - aggregate all heads and tails by (relation, time)
        final_aggregated = {}
        for key, data in undirected_aggregated.items():
            relation = data['relation']
            head = data['head']
            tail = data['tail']
            time_start = data['time_start']
            
            # create final aggregate key: (relation, time) - aggregate paths with the same relation and time
            final_key = (head, relation, time_start)
            
            if final_key not in final_aggregated:
                final_aggregated[final_key] = {
                    'relation': relation,
                    'time_start': time_start,
                    'time_start_epoch': data['time_start_epoch'],
                    'similarities': data['similarities'].copy(),
                    'count': data['count'],
                    'heads': [head],
                    'tails': [tail],
                    'original_paths': data['original_paths'].copy()
                }
            else:
                # aggregate to the same relation and time
                final_aggregated[final_key]['count'] += data['count']
                final_aggregated[final_key]['similarities'].extend(data['similarities'])
                final_aggregated[final_key]['original_paths'].extend(data['original_paths'])
                
                # collect all different heads and tails
                if head not in final_aggregated[final_key]['heads']:
                    final_aggregated[final_key]['heads'].append(head)
                if tail not in final_aggregated[final_key]['tails']:
                    final_aggregated[final_key]['tails'].append(tail)
                
                # update earliest time
                if data['time_start_epoch'] > 0 and (final_aggregated[final_key]['time_start_epoch'] == 0 or data['time_start_epoch'] < final_aggregated[final_key]['time_start_epoch']):
                    final_aggregated[final_key]['time_start_epoch'] = data['time_start_epoch']
                    final_aggregated[final_key]['time_start'] = data['time_start']
        
        aggregated = final_aggregated

        # convert to final format
        agg_list = []
        for key, data in aggregated.items():
            relation = data['relation']
            all_heads = list(set(data['heads']))
            all_tails = list(set(data['tails']))
            
            # select primary head and tail
            # prioritize seed entity as head
            primary_head = None
            for h in all_heads:
                if h in seed_set:
                    primary_head = h
                    break
            if primary_head is None:
                primary_head = all_heads[0]
            
            # select primary tail
            primary_tail = None
            for t in all_tails:
                if t not in seed_set:  # prioritize non-seed entity as tail
                    primary_tail = t
                    break
            if primary_tail is None:
                primary_tail = all_tails[0]
            
            # calculate average similarity
            sims = data['similarities']
            if sims:
                avg_sim = sum(sims) / len(sims)
            else:
                avg_sim = 0.0
            
            agg_item = {
                'head': primary_head,
                'relation': relation,
                'tail': primary_tail,
                'time_start': data['time_start'],
                'time_start_epoch': data['time_start_epoch'],
                'similarity': float(avg_sim),
                'count': data['count'],
                'max_similarity': max(sims) if sims else 0.0,
                'min_similarity': min(sims) if sims else 0.0,
                # aggregate information
                'heads': all_heads,
                'head_count': len(all_heads),
                'heads_str': ', '.join(sorted(all_heads[:3])) + ('...' if len(all_heads) > 3 else ''),
                'tails': all_tails,
                'tail_count': len(all_tails),
                'tails_str': ', '.join(sorted(all_tails[:5])) + ('...' if len(all_tails) > 5 else ''),
                'original_paths': data['original_paths']
            }
                 
            agg_list.append(agg_item)


        print(f"📦 aggregate completed: {len(agg_list)} aggregate paths (originally {len(paths)} paths)")
        return agg_list

    def _parse_ref_epoch(self, time_constraint: Optional[str], subquestion_obj: Optional[Any] = None):
        if not time_constraint:
            return None, None, None, None, None
        
        # build context information to help smart parsing
        context_info = {}
        context_year = None
        
        if subquestion_obj and hasattr(subquestion_obj, 'indicator'):
            constraints = getattr(subquestion_obj.indicator, 'constraints', [])
            context_info['constraints'] = constraints
            
            # try to extract year information from constraints
            for constraint in constraints:
                # find constraints like "t1 = 2010-05"
                year_match = re.search(r'(\d{4})', str(constraint))
                if year_match:
                    context_year = int(year_match.group(1))
                    break
        
        try:
            from kg_agent.temporal_kg_toolkit import parse_time_to_range, smart_parse_time
            
            # first try smart parsing
            parsed_time = smart_parse_time(time_constraint, context_year, context_info)
            if parsed_time:
                print(f"✅ smart time parsing: '{time_constraint}' -> '{parsed_time}'")
                s, e, g, ref_epoch, ref_str = parse_time_to_range(parsed_time, context_year, context_info)
                return s, e, g, ref_epoch, ref_str
            else:
                # if smart parsing returns None, try direct parsing
                s, e, g, ref_epoch, ref_str = parse_time_to_range(time_constraint, context_year, context_info)
                return s, e, g, ref_epoch, ref_str
                
        except Exception as e:
            print(f"⚠️ time parsing failed: {time_constraint} ({e})")
            
            # try natural language time parsing (fallback)
            try:
                from kg_agent.natural_time_parser import parse_natural_time_to_iso
                iso_time = parse_natural_time_to_iso(time_constraint)
                if iso_time:
                    print(f"✅ natural language time parsing successful: '{time_constraint}' -> '{iso_time}'")
                    s, e, g, ref_epoch, ref_str = parse_time_to_range(iso_time, context_year, context_info)
                    return s, e, g, ref_epoch, ref_str
            except Exception as natural_e:
                pass  # silent failure, continue to next step
            
            # provide default time value for first_last type
            if subquestion_obj and hasattr(subquestion_obj, 'indicator'):
                constraints = getattr(subquestion_obj.indicator, 'constraints', [])
                for constraint in constraints:
                    if 'last(' in constraint:
                        # before_last tool, use 2025-01-01 as default time
                        print(f"🔄 use default time 2025-01-01 instead of {time_constraint}")
                        ref_epoch = int(datetime(2025, 1, 1).timestamp())
                        return None, None, None, ref_epoch, "2025-01-01"
                    elif 'first(' in constraint:
                        # after_first tool, use 1800-01-01 as default time
                        print(f"🔄 use default time 1800-01-01 instead of {time_constraint}")
                        ref_epoch = int(datetime(1800, 1, 1).timestamp())
                        return None, None, None, ref_epoch, "1800-01-01"
            
            return None, None, None, None, None

    def time_filtering_and_sorting(self, paths: List[Dict[str, Any]],
                                   time_constraint: str, 
                                   constraint_type: str = "after",
                                   subquestion_obj: Optional[Any] = None) -> List[Dict[str, Any]]:
        if not paths:
            return []
        s, e, g, ref_epoch, _ = self._parse_ref_epoch(time_constraint, subquestion_obj)
        if ref_epoch is None and constraint_type in ("after", "before"):
            return paths
        if s is None and e is None and constraint_type in ("equal", "between"):
            return paths
        
        # build context information
        context_info = {}
        context_year = None
        if subquestion_obj and hasattr(subquestion_obj, 'indicator'):
            constraints = getattr(subquestion_obj.indicator, 'constraints', [])
            context_info['constraints'] = constraints
            for constraint in constraints:
                year_match = re.search(r'(\d{4})', str(constraint))
                if year_match:
                    context_year = int(year_match.group(1))
                    break
        
        # directly use original time constraint to parse, get full range
        start_epoch = None
        end_epoch = None
        if time_constraint:
            try:
                from kg_agent.temporal_kg_toolkit import parse_time_to_range
                _, _, _, start_epoch, end_epoch = parse_time_to_range(time_constraint, context_year, context_info)
                print(f"🔍 time filtering range: {start_epoch} <= t <= {end_epoch} (constraint type: {constraint_type})")
            except Exception as ex:
                print(f"⚠️ time range conversion failed: {ex}")
                # if parsing fails, try using _parse_ref_epoch result
                if s is not None and e is not None:
                    try:
                        _, _, _, start_epoch, end_epoch = parse_time_to_range(s, context_year, context_info)
                    except:
                        # if all parsing fails, return all paths, skip time filtering
                        print(f"⚠️ time parsing completely failed, skip time filtering, return all {len(paths)} paths")
                        return paths
                else:
                    # if there is no valid time information, return all paths
                    print(f"⚠️ no valid time information, skip time filtering, return all {len(paths)} paths")
                    return paths
        
        filtered = []
        for p in paths:
            t = p.get('time_start_epoch', 0)
            if constraint_type == "after" and ref_epoch is not None and t > ref_epoch:
                filtered.append(p)
            elif constraint_type == "before" and ref_epoch is not None and t < ref_epoch:
                filtered.append(p)
            elif constraint_type == "between" and start_epoch is not None and end_epoch is not None and start_epoch <= t <= end_epoch:
                filtered.append(p)
            elif constraint_type == "equal" and start_epoch is not None and end_epoch is not None and start_epoch <= t <= end_epoch:
                filtered.append(p)
        if constraint_type == "after":
            filtered.sort(key=lambda x: x.get('time_start_epoch', 0))
        elif constraint_type == "before":
            filtered.sort(key=lambda x: x.get('time_start_epoch', 0), reverse=True)
        else:
            filtered.sort(key=lambda x: x.get('time_start_epoch', 0))
        return filtered

    # ================== semantic pruning (keep) ==================
    def semantic_pruning(self, paths: List[Dict[str, Any]],
                         subquestion: str,
                         top_k: int = 50,
                         subquestion_obj: Optional[Any] = None) -> List[Dict[str, Any]]:
        if not paths:
            return []
        if subquestion_obj is not None:
            expected_relations = self._extract_relations_from_subquestion_obj(subquestion_obj)
        else:
            expected_relations = self._extract_indicators_from_subquestion(subquestion)
        q_emb = self.get_embedding(subquestion)
        from collections import defaultdict
        rel2scores = defaultdict(list)
        for p in paths:
            rel = p.get('relation', 'Unknown')
            rel_sim = self._calculate_relationship_similarity(p, subquestion, expected_relations, q_emb)
            rel2scores[rel].append({'similarity': rel_sim, 'path': p})
        top_relations = {}
        for rel, lst in rel2scores.items():
            if not lst:
                continue
            best_item = max(lst, key=lambda x: x['similarity'])
            rel_score = best_item['similarity']
            if expected_relations:
                rel_lower = rel.lower()
                for exp in expected_relations:
                    exp_lower = exp.lower()
                    if exp_lower == rel_lower:
                        rel_score += self.REL_EXACT_BONUS
                    elif exp_lower in rel_lower or rel_lower in exp_lower:
                        rel_score += self.REL_PARTIAL_BONUS
            top_relations[rel] = {
                'score': rel_score,
                'best_path': best_item['path'],
                'similarity': best_item['similarity']
            }
        sorted_rel = sorted(top_relations.items(), key=lambda x: x[1]['score'], reverse=True)[:3]
        pruned = []
        for _, info in sorted_rel:
            bp = info['best_path'].copy()
            bp['relation_similarity'] = info['similarity']
            bp['relation_score'] = info['score']
            pruned.append(bp)
        return pruned

    # ================== retrieval (keep + direct connection priority & similarity filtering) ==================
    def _retrieve_single_seed_paths(self, entity: str, time_constraint: str = None):
        print(f"📊 1-hop: get related paths for seed {entity}")
        return self.kg_query.retrieve_one_hop(query=entity, direction="both", limit=None)

    def _retrieve_multi_seed_paths(self, entities: List[str], time_constraint: str = None):
        print(f"📊 multi-seed retrieval (seeds: {entities})")
        all_edges = []
        if len(entities) >= 2:
            print("🔗 prioritize direct connection between seeds")
            for i in range(len(entities)):
                for j in range(i + 1, len(entities)):
                    e1, e2 = entities[i], entities[j]
                    try:
                        direct = self.kg_query.find_direct_connection(entity1=e1, entity2=e2, direction="both", limit=None)
                        if hasattr(direct, 'edges') and direct.edges:
                            all_edges.extend(direct.edges)
                    except Exception as e:
                        print(f"⚠️ direct connection retrieval failed {e1} <-> {e2}: {e}")
        for ent in entities:
            try:
                res = self.kg_query.retrieve_one_hop(query=ent, direction="both", limit=None)
                if hasattr(res, 'edges'):
                    all_edges.extend(res.edges)
            except Exception as e:
                print(f"⚠️ seed {ent} 1-hop retrieval failed: {e}")
        if len(entities) >= 2:
            for i in range(len(entities)):
                for j in range(i + 1, len(entities)):
                    e1, e2 = entities[i], entities[j]
                    try:
                        direct_all = self.kg_query.find_direct_connection(entity1=e1, entity2=e2, direction="both", limit=None)
                        if hasattr(direct_all, 'edges') and direct_all.edges:
                            all_edges.extend(direct_all.edges)
                    except Exception as e:
                        print(f"⚠️ direct connection retrieval failed {e1} <-> {e2}: {e}")
        unique = self._prune_duplicate_paths(all_edges)
        prioritized_edges = self._prioritize_direct_connections(unique, entities)
        class MultiSeedResult:  # compatible return
            def __init__(self, edges): self.edges = edges
        return MultiSeedResult(prioritized_edges)

    def _prune_duplicate_paths(self, all_edges: List[Any]) -> List[Any]:
        if not all_edges:
            return []
        unique_edges, seen = [], set()
        for e in all_edges:
            try:
                if hasattr(e, 'head'):
                    head = getattr(e, 'head', '')
                    tail = getattr(e, 'tail', '')
                    relation = getattr(e, 'relation', '')
                    ts = getattr(e, 'time_start', '')
                    te = getattr(e, 'time_end', '')
                    t = f"{ts}" if ts == te else f"{ts}~{te}"
                    key = (str(head), str(relation), str(tail), str(t))
                elif isinstance(e, dict):
                    head = str(e.get('head', ''))
                    tail = str(e.get('tail', ''))
                    relation = str(e.get('relation', ''))
                    t = str(e.get('time', e.get('time_start', '')))
                    key = (head, relation, tail, t)
                else:
                    head = str(getattr(e, 'head', str(e)))
                    tail = str(getattr(e, 'tail', ''))
                    relation = str(getattr(e, 'relation', ''))
                    t = str(getattr(e, 'time', ''))
                    key = (head, relation, tail, t)
                if key not in seen:
                    seen.add(key)
                    unique_edges.append(e)
            except Exception as ex:
                print(f"⚠️ path deduplication failed: {ex}")
                continue
        return unique_edges

    def _find_all_direct_connections(self, entity_names: List[str], time_constraint: str = None, constraint_type: str = None, subquestion_obj: Optional[Any] = None) -> List[Dict[str, Any]]:
        """
        find direct connection edges between any two entities, with time constraint
        
        Args:
            entity_names: entity name list
            time_constraint: time constraint (e.g. "after" -> time parameter)
            constraint_type: constraint_type ("after", "before", "between")
            subquestion_obj: subquestion object, for extracting context information
            
        Returns:
            list of found direct connection edges
        """
        if len(entity_names) < 2:
            return []
        
        try:
            # import temporal_kg_pack database functions
            import sys
            import os
            
            # add correct path
            kg_agent_dir = os.path.dirname(os.path.abspath(__file__))
            tpd_root = os.path.dirname(os.path.dirname(kg_agent_dir))  # back to TPKG root directory
            sys.path.append(tpd_root)
            
            from Data.temporal_kg_pack import _connect, entity_id, entity_name
            from kg_agent.temporal_kg_toolkit import parse_time_to_range
            
            db_path = self.db_path
            conn = _connect(db_path)
            cur = conn.cursor()
            
            all_edges = []
            
            # build context information
            context_info = {}
            context_year = None
            if subquestion_obj and hasattr(subquestion_obj, 'indicator'):
                constraints = getattr(subquestion_obj.indicator, 'constraints', [])
                context_info['constraints'] = constraints
                for constraint in constraints:
                    year_match = re.search(r'(\d{4})', str(constraint))
                    if year_match:
                        context_year = int(year_match.group(1))
                        break
            
            # time filtering parameters
            q0 = q1 = None
            if time_constraint:
                try:
                    _, _, _, q0, q1 = parse_time_to_range(time_constraint, context_year, context_info)
                    print(f"time limit: {time_constraint} -> epoch range: [{q0}, {q1}]")
                except Exception as e:
                    print(f"time parsing failed: {e}")
                    # return []
            
            # iterate over all entity pairs
            for i in range(len(entity_names)):
                for j in range(i + 1, len(entity_names)):
                    entity1 = entity_names[i]
                    entity2 = entity_names[j]
                    
                    print(f"🔗 find direct connection between {entity1} and {entity2}")
                    
                    try:
                        # get entity ID
                        e1_id = entity_id(db_path, entity1)
                        e2_id = entity_id(db_path, entity2)
                        
                        # SQL query + time filtering
                        sql_base = """
                        SELECT e.id, e.head_id, e.relation, e.tail_id, e.t_start, e.t_end, 
                               e.t_start_epoch, e.t_end_epoch, e.granularity
                        FROM edges e
                        WHERE (e.head_id = ? AND e.tail_id = ?) 
                           OR (e.head_id = ? AND e.tail_id = ?)
                        """
                        
                        time_filter = ""
                        params = [e1_id, e2_id, e2_id, e1_id]
                        
                        if time_constraint and q0 and q1:
                            if constraint_type == "between":
                                time_filter = " AND NOT (e.t_end_epoch < ? OR e.t_start_epoch > ?)"
                                params.extend([q0, q1])
                            elif constraint_type == "before":
                                time_filter = " AND e.t_end_epoch <= ?"
                                params.append(q1)
                            elif constraint_type == "after":
                                time_filter = " AND e.t_start_epoch >= ?"
                                params.append(q0)
                        
                        sql = sql_base + time_filter + " ORDER BY e.t_start_epoch"
                        
                        cur.execute(sql, params)
                        rows = cur.fetchall()
                        
                        for row in rows:
                            edge_id, head_id, relation, tail_id, t_start, t_end, t_start_epoch, t_end_epoch, granularity = row
                            
                            # get entity name
                            try:
                                head_name = entity_name(db_path, head_id)
                                tail_name = entity_name(db_path, tail_id)
                            except:
                                head_name = str(head_id)
                                tail_name = str(tail_id)
                            
                            # create edge data - match system expected path format
                            edge_info = {
                                'head': head_name,
                                'relation': relation, 
                                'tail': tail_name,
                                'time_start': t_start,
                                'time': t_start,  # for compatibility with different variable names
                                'time_start_epoch': t_start_epoch,
                                'time_end_epoch': t_end_epoch,
                                'granularity': granularity,
                                'id': edge_id  # keep back edge ID
                            }
                            all_edges.append(edge_info)
                            # print(f"✅ find direct connection: {head_name} -> {relation} -> {tail_name} ({t_start})")
                        
                        print(f"📊 {entity1} <-> {entity2}: find {len(rows)} direct connections")
                        
                    except Exception as e:
                        print(f"⚠️ find direct connection between {entity1} and {entity2} failed: {e}")
                        continue
            
            cur.close()
            conn.close()
            
            print(f"🎯 direct connection summary: find {len(all_edges)} edges")
            return all_edges
            
        except Exception as e:
            print(f"❌ database direct connection query error: {e}")
            return []

    def _calculate_entity_similarity_to_seeds(self, entity_name: str, seed_entities: List[str]) -> float:
        if not entity_name or not seed_entities:
            return 0.0
        max_similarity = 0.0
        entity_embedding = self.get_embedding(entity_name)
        for seed in seed_entities:
            seed_embedding = self.get_embedding(seed)
            similarity = float(np.dot(entity_embedding, seed_embedding) / (
                np.linalg.norm(entity_embedding) * np.linalg.norm(seed_embedding) + 1e-8))
            max_similarity = max(max_similarity, similarity)
        return max_similarity

    def _filter_paths_by_entity_similarity(self, paths: List[Dict[str, Any]], seed_entities: List[str],
                                           similarity_threshold: float = 0.5) -> List[Dict[str, Any]]:
        if not paths or not seed_entities:
            return paths
        filtered = []
        for p in paths:
            head, tail = p.get('head', ''), p.get('tail', '')
            if head in seed_entities or tail in seed_entities:
                filtered.append(p); continue
            head_sim = self._calculate_entity_similarity_to_seeds(head, seed_entities)
            tail_sim = self._calculate_entity_similarity_to_seeds(tail, seed_entities)
            if max(head_sim, tail_sim) >= similarity_threshold:
                filtered.append(p)
        return filtered

    def _prioritize_direct_connections(self, all_edges: List[Any], seed_entities: List[str]) -> List[Any]:
        if not all_edges or len(seed_entities) < 2:
            return all_edges
        seed_set = set(seed_entities)
        direct, others = [], []
        for e in all_edges:
            if hasattr(e, 'head'):
                h, t = getattr(e, 'head', ''), getattr(e, 'tail', '')
            elif isinstance(e, dict):
                h, t = e.get('head', ''), e.get('tail', '')
            else:
                others.append(e); continue
            if h in seed_set and t in seed_set and h != t:
                direct.append(e)
            else:
                others.append(e)
        return direct + others

    def _normalize_path_direction(self, path: Dict[str, Any], subquestion_obj: Optional[Any] = None) -> Dict[str, Any]:
        head, tail, relation = path.get('head', ''), path.get('tail', ''), path.get('relation', '')
        # symmetric_relations = {'Engage_in_diplomatic_cooperation', 'Engage_in_negotiation',
        #                        'Make_agreement', 'Cooperate', 'Meet_with', 'Discuss_by_telephone'}
        try:
            # if relation in symmetric_relations and head > tail:
            #     n = path.copy(); n['head'], n['tail'] = tail, head; return n
            if subquestion_obj and hasattr(subquestion_obj, 'indicator'):
                indicator_entities = []
                for edge in subquestion_obj.indicator.edges:
                    if hasattr(edge, 'subj') and edge.subj and edge.subj not in ['?x', '?y']:
                        indicator_entities.append(edge.subj)
                    if hasattr(edge, 'obj') and edge.obj and edge.obj not in ['?x', '?y']:
                        indicator_entities.append(edge.obj)
                if indicator_entities:
                    first = indicator_entities[0]
                    if tail == first and head != first:
                        n = path.copy(); n['head'], n['tail'] = tail, head; return n
            return path
        except Exception as e:
            print(f"⚠️ path direction standardization failed: {e}")
            return path

    def _filter_self_loops(self, paths: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out, cnt = [], 0
        for p in paths:
            if p.get('head','') == p.get('tail',''):
                cnt += 1
                continue
            out.append(p)
        if cnt:
            print(f"🚫 filter out {cnt} self-loop paths")
        return out

    # ================== LLM selection (keep) ==================
    def llm_path_selection(self, paths: List[Dict[str, Any]],
                           subquestion: str,
                           context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Iterative LLM path selection: select top 1 from each batch of 20 paths,
        continue until we have exactly 1 final results
        """
        if not paths:
            return {"error": "No available paths"}, []
            
        print(f"🤖 LLM iterative path selection: {len(paths)} total paths")
        
        # Start with all paths
        remaining_paths = paths.copy()
        best_but_not_answer = []
        best_but_not_answer = remaining_paths.copy()
        
        # If 3 or fewer paths, select best one directly
        if len(remaining_paths) <= 3:
 
            if len(remaining_paths) == 1:
                final_path = remaining_paths[0]
                final_path['selection_reason'] = "single path available"
                return final_path, best_but_not_answer
            else:
                # Select best 1 from ≤ 3  remaining paths
                print(f"📋 Final selection from {len(remaining_paths)} paths")
                selected_indices = self._select_paths_batch(remaining_paths, subquestion, top_k=1)
                if selected_indices and 0 <= selected_indices[0] < len(remaining_paths):
                    final_path = remaining_paths[selected_indices[0]]
                    final_path['selection_reason'] = "final: best from ≤1"
                    return final_path, best_but_not_answer
                else:
                    final_path = remaining_paths[0]
                    final_path['selection_reason'] = "final: fallback to first"
                    return final_path, best_but_not_answer
        
        # Keep selecting until we have ≤3 paths
        while len(remaining_paths) > 3:
            batch_size = min(30, len(remaining_paths))
            batch_paths = remaining_paths[:batch_size]
            
            print(f"📋 Selecting top 1 from {batch_size} paths (iteration {max(3, (len(paths) - len(remaining_paths) // 40) + 1)})")
            
            selected_indices = self._select_paths_batch(batch_paths, subquestion, top_k=3)
            
            # Map indices back to selection and continue with selected paths
            selected_paths = []
            for idx in selected_indices:
                if 0 <= idx < len(batch_paths):
                    selected_paths.append(batch_paths[idx])
            
            # Update remaining paths - keep selected + rest of paths
            if len(remaining_paths) > 30:
                remaining_paths = selected_paths + remaining_paths[30:]
            else:
                remaining_paths = selected_paths
            remaining_paths = remaining_paths[:100]  # Limit to prevent infinite loops
            if len(remaining_paths) < len(best_but_not_answer) and len(remaining_paths) > 1:
                best_but_not_answer = remaining_paths.copy()
            # best_but_not_answer = remaining_paths
        # Handle final ≤3 paths  
        if not remaining_paths:
            # retry LLM path selection for 3 times
            for i in range(3):
                remaining_paths = self._select_paths_batch(best_but_not_answer, subquestion, top_k=100)
                if remaining_paths:
                    break
            if not remaining_paths:
                remaining_paths = best_but_not_answer
        
        # Final selection
        selected_indices = self._select_paths_batch(remaining_paths, subquestion, top_k=1)
        if selected_indices and 0 <= selected_indices[0] < len(remaining_paths):
            final_path = remaining_paths[selected_indices[0]]
            final_path['selection_reason'] = "final: iterative selection result"
            return final_path, best_but_not_answer
        else:
            final_path = remaining_paths[0]
            final_path['selection_reason'] = "final: fallback to first of ≤3"
            return final_path, best_but_not_answer
    
    def _select_paths_batch(self, batch_paths: List[Dict[str, Any]], 
                           subquestion: str, top_k: int = 1) -> List[int]:
        """Select top paths from a batch using LLM"""
        if not batch_paths:
            return []
            
        desc = []
        for i, p in enumerate(batch_paths):
            # Use aggregated tails information if available
            tail_display = p.get('tails_str', p.get('tail', ''))
            tail_count = p.get('tail_count', 1)
            if tail_count > 1:
                tail_display = f"{tail_display} ({tail_count} entities)"
            
            # check if there is aggregation information (multiple heads or tails)
            has_multiple_heads = p.get('head_count', 1) > 1
            has_multiple_tails = p.get('tail_count', 1) > 1
            has_aggregation = has_multiple_heads or has_multiple_tails > 1
            
            if has_aggregation:
                # display aggregation information
                if has_multiple_heads and 'heads_str' in p:
                    heads_display = p.get('heads_str', '')
                    if p.get('heads_count', 0) > 5:
                        heads_display += f" etc {p.get('heads_count', 0)} entities"
                else:
                    heads_display = p.get('head', '')
                
                line = f"{i+1}. [{heads_display}] -> {p.get('relation','')} -> [{tail_display}] at {p.get('time_start','')} (Multiple entities happend simultaneously, treat them same)"
            else:
                head = p.get('head', '') if 'head' in p else (p.get('heads', [''])[0] if 'heads' in p else '')
                line = f"{i+1}. [{head}] -> {p.get('relation','')} -> [{tail_display}] at {p.get('time_start','')}"
            
            desc.append(line)
        
        prompt = f"""You are a path selector. Select the **{top_k} most relevant** path numbers based on the subquestion.
Note: the given path is already sorted by the time in first or last, so you don't need to consider the time.
you should consider the sematic information and reasonable. path may the the order, consider who is Active and who is Passive.
Subquestion: {subquestion}

Candidate paths:
{chr(10).join(desc)}

Return JSON like: {{"selected_paths": [1, 3, 5], "reason": "brief explanation"}}"""
        
        if self.llm:
            try:
                print(prompt)
                # use path selection model
                from config import TPKGConfig
                path_selection_model = TPKGConfig.PATH_SELECTION_LLM_MODEL
                resp = self.llm.call("", prompt, model=path_selection_model)
                print(resp)
                # Extract JSON response
                m = re.search(r'\{.*\}', resp, re.S)
                if m:
                    try:
                        obj = json.loads(m.group(0))
                        selected = obj.get("selected_paths", [])
                        if isinstance(selected, list) and all(isinstance(x, int) for x in selected):
                            selected = [x-1 for x in selected if 1 <= x <= len(batch_paths)]  # Convert to 0-based
                            return selected[:top_k]
                    except json.JSONDecodeError:
                        pass
                
                # Fallback: try to extract numbers from response
                nums = re.findall(r'\b([1-9][0-9]*)\b', resp)
                selected = [int(n)-1 for n in nums if 1 <= int(n) <= len(batch_paths)][:top_k]  # Convert to 0-based
                return selected
                
            except Exception as e:
                print(f"⚠️ LLM parsing failed: {e}, using fallback")
        
        # Fallback: return top paths by similarity/score
        print(prompt)
        return list(range(min(top_k, len(batch_paths))))

    # ================== top-level pipeline (core) ==================
    def intelligent_retrieve(self, entity: Union[str, List[str]], 
                           subquestion: str,
                           time_constraint: str = None,
                           constraint_type: str = "after",
                           top_k: int = 50,
                           toolkit_type: str = "general",
                             subquestion_obj: Optional[Any] = None,
                             time_order: str = "asc",
                             relation_select_mode: str = "auto",  # 'auto'|'all'|'top3'
                             selection_mode: str = "llm",         # 'llm'|'first'|'last'|'none'
                             limit: Optional[int] = None) -> Dict[str, Any]:
        """
        unified pipeline (for 8 tool packages reuse):
        1) pull all entity relations
        2) relation selection (auto/top3/all)
        3) one-time pull all → relation filtering → self-loop filtering → direction standardization → (multiple seeds) entity similarity filtering
        4) time quick filtering
        5) aggregate before time sorting
        6) aggregate + aggregate after time sorting
        7) selection strategy (llm/first/last/none)
        """
        print(f"\n{'='*50} Retrieval Pipeline {'='*50}")
        print(f"entity: {entity} | subquestion: {subquestion}")
        print(f"time constraint: {time_constraint} ({constraint_type}) | toolkit: {toolkit_type}")
        print(f"subq indicator: {subquestion_obj.indicator}")
        print(f"\n{'='*50} Retrieval Pipeline {'='*50}")

        try:
            


            selected_paths = []
            selected_relations = []
            if isinstance(entity, list):
                original_entity = entity
            else:
                original_entity = [entity]
            # Check for direct connections first - only run once!
             # directly replace selected_paths, skip subsequent normal logic
                    
            # if direct connection cannot find enough path, or non-direct connection type, execute regular retrieval
            for entity in original_entity:
                if len(original_entity) > 1:
                    print(f"🔍 detect direct connection mode for entities: {original_entity}") 
                    all_edges = self._find_all_direct_connections(original_entity, time_constraint, constraint_type, subquestion_obj)
                    if len(all_edges) > 0:
                        selected_paths_2e = all_edges  # direct connection result, keep subsequent steps for aggregation
                        print(f"🔗 direct connection completed, find {len(selected_paths_2e)} edges, continue aggregation steps...")

                        # INSERT_YOUR_CODE
                        # if the question is first, select the first relation of each time as the answer; if the question is last, select the last relation
                        
                        # if "first" in subquestion.lower() or "last" in subquestion.lower():
                        #     # group by relation
                        #     from collections import defaultdict
                        #     rel_group = defaultdict(list)
                        #     for edge in selected_paths_2e:
                        #         rel = edge.get("relation", "")
                        #         rel_group[rel].append(edge)
                        #     pruned_selected_paths = []
                        #     for rel, edges in rel_group.items():
                        #         # sort by time
                        #         edges_sorted = sorted(
                        #             edges,
                        #             key=lambda x: x.get("time_start_epoch", 0)
                        #         )
                        #         if "last" in subquestion.lower() or "before_last" in subquestion.lower():
                        #             pruned_selected_paths.append(edges_sorted[-1])
                        #         else:  # "first" or "first_last"
                        #             pruned_selected_paths.append(edges_sorted[0])
                        #     selected_paths.extend(pruned_selected_paths)
                        #     print(f"🔝 first/last mode, each relation only take one, remaining {len(selected_paths)} paths")
                        #     # skip subsequent regular aggregation
                        #     break
                        if len(selected_paths_2e) > 50:
                            # step1: get all relations and count frequency
                            relation_totoal = set()
                            for edge in selected_paths_2e:
                                relation_totoal.add(edge['relation'])
                                # print(f"🔍 edge: {edge}")
                                # exit()

                            
                            print(f"🔍 relation_totoal: {len(relation_totoal)}")
                            relation_totoal = list(relation_totoal)
                            # step2: use hybrid method to select relations (first embedding select top10, then LLM select top3)
                            print(f"🔍 use hybrid method to select relations...")
                            selected_relations = self._select_relations_hybrid(relation_totoal, subquestion_obj, subquestion, top_k=10, final_k=3)
                            
                            # print(f"🔍 selected top3 relations: {selected_relations}")

                            pruned_selected_paths = []
                            # index = 3
                            sel_rel_set = set(selected_relations)
                            # while len(pruned_selected_paths) < 40 and index < len(selected_relations):
                            for edge in selected_paths_2e:
                                if edge['relation'] in sel_rel_set:
                                    pruned_selected_paths.append(edge)
                            # sel_rel_set = selected_relations[index]
                                # pruned_selected_paths= list(set(pruned_selected_paths))
                                # index += 1
                            # print(f"🔍 pruned_selected_paths: {pruned_selected_paths}")
                            
                            # sorted_paths = sorted(selected_paths, key=lambda x: (
                            #     rel_counts.get(x[1] if isinstance(x, (list, tuple)) and len(x) > 1 else x.get('relation', ''), 0)
                            # ), reverse=True)
                            # selected_paths = sorted_paths[:50]
                            selected_paths.extend(pruned_selected_paths)
                            # selected_paths = pruned_selected_paths
                            print(f"🔝 more than 20, sort by relation and select top50 paths, remaining {len(selected_paths)} paths")
                        else:
                            selected_paths.extend(selected_paths_2e)
                        print(f"🔍 toolkit_type: {toolkit_type}")
                        
                    else:
                        print("⚠️ Traditional retrieval paths are few, will rely on hybrid retrieval supplement")
                    if "direct" in toolkit_type.lower():
                            break

                # Step 1: get preliminary paths (using all relations)
                subquestion = subquestion_obj.text if subquestion_obj else subquestion
                print(f"🔍 subquestion: {subquestion}")
                relations_for_entity = self._get_relations_by_entity(entity, time_constraint, constraint_type)
                print(f"📊 relations total: {len(relations_for_entity)}")

                # Step 2: build preliminary paths (first without filtering relations, collect all paths)
                print(f"🔍 Step 2: build preliminary paths...")
                if relation_select_mode == "all":
                    # all mode directly use all relations
                    initial_relations = relations_for_entity
                else:
                    # other mode first quickly filter top20, reduce preliminary search cost
                    if len(relations_for_entity) > 20:
                        # quickly use embedding select top20
                        expected = self._extract_relations_from_subquestion_obj(subquestion_obj) if subquestion_obj else self._extract_indicators_from_subquestion(subquestion)
                        q_emb = self.get_embedding(subquestion)
                        relation_scoring = {rel: self._calculate_relationship_indicator_similarity(rel, expected, subquestion, q_emb) for rel in relations_for_entity}
                        sorted_relations = sorted(relation_scoring.items(), key=lambda x: x[1], reverse=True)
                        initial_relations = [rel for rel, _ in sorted_relations[:20]]
                        print(f"🔍 quickly filter top20 relations for preliminary search")
                    else:
                        initial_relations = relations_for_entity
                
                initial_paths = self._build_triplet_paths_for_relations(initial_relations, entity, time_constraint, constraint_type, subquestion_obj)
                print(f"📊 preliminary paths: {len(initial_paths)}")
                selected_paths.extend(initial_paths)

            relationship_paths = selected_paths
            subquestion = subquestion_obj.text if subquestion_obj else subquestion
            
            # Step 3: call hybrid retrieval system as supplement
            print(f"🔍 hybrid retrieval call check: HYBRID_RETRIEVAL_AVAILABLE={HYBRID_RETRIEVAL_AVAILABLE}, paths: {len(relationship_paths)}")
            if HYBRID_RETRIEVAL_AVAILABLE:
                try:
                    print(f"🔍 call hybrid retrieval system supplement result: {subquestion}")
                    
                    # use asynchronous call, optimize parameters
                    import asyncio
                    hybrid_result = asyncio.run(hybrid_retrieve_triples(
                        question=subquestion, 
                        db_path=self.db_path, 
                        raw_data_path=self.raw_data_path,
                        top_k=50,  # reduce number, improve speed
                        re_rank=False,  # not use re-ranking
                        use_gpu=False,  # use CPU, avoid GPU conflict
                        gpu_id=1  # use different GPU ID
                    ))
                    
                    if 'error' not in hybrid_result and hybrid_result.get('triples'):
                        # convert hybrid retrieval result to standard path format
                        hybrid_paths = self._convert_hybrid_results_to_paths(hybrid_result)
                        relationship_paths.extend(hybrid_paths)
                        print(f"🔍 hybrid retrieval supplement {len(hybrid_paths)} paths")
                    else:
                        print(f"⚠️ hybrid retrieval no result: {hybrid_result.get('error', 'No triples found')}")
                        
                except Exception as e:
                    print(f"⚠️ hybrid retrieval call failed: {e}")
            elif len(relationship_paths) >= 20:
                print(f"✅ traditional retrieval has enough paths({len(relationship_paths)}), skip hybrid retrieval")
            else:
                print("⚠️ hybrid retrieval system not available")

            # Step 4: extract relations from all paths and select hybrid
            if relation_select_mode != "all" and len(relationship_paths) > 0:
                print(f"🔍 Step 4: extract relations from all paths (including hybrid retrieval)...")
                # extract all unique relations
                all_relations_from_paths = set()
                for path in relationship_paths:
                    rel = path.get('relation', '')
                    if rel:
                        all_relations_from_paths.add(rel)
                
                all_relations_list = list(all_relations_from_paths)
                print(f"📊 extract {len(all_relations_list)} unique relations")
                
                # # use hybrid method to select top3 relations
                # if len(all_relations_list) > 3:
                #     selected_relations = self._select_relations_hybrid(
                #         all_relations_list, subquestion_obj, subquestion, 
                #         top_k=min(10, len(all_relations_list)), 
                #         final_k=3
                #     )
                #     print(f"🎯 hybrid method select top3 relations: {selected_relations}")
                    
                #     # filter paths by selected relations
                #     selected_rel_set = set(selected_relations)
                #     filtered_paths = [p for p in relationship_paths if p.get('relation', '') in selected_rel_set]
                #     print(f"🔍 relation filtered paths: {len(relationship_paths)} → {len(filtered_paths)}")
                #     relationship_paths = filtered_paths
                # else:
                #     print(f"✅ relation number≤3, skip filtering")
            
            # Step 5: Quick time filter
            if time_constraint:
                print(f"🔍 time constraint: {time_constraint}")
                # exit()
            relationship_paths = self._quick_time_filter(relationship_paths, time_constraint, constraint_type, subquestion_obj)
            print(f"⏰ time filtered: {len(relationship_paths)}")
            
            # Debug: Show all relations after time filtering
            if relationship_paths:
                relations_after_time_filter = {}
                for path in relationship_paths:
                    rel = path.get('relation', '')
                    tail = path.get('tail', '')
                    key = f"{rel} -> {tail}"
                    if key not in relations_after_time_filter:
                        relations_after_time_filter[key] = []
                    relations_after_time_filter[key].append(path.get('time_start', ''))
                
                print(f"📊 Relations after time filtering (top 15):")
                for i, (key, times) in enumerate(sorted(relations_after_time_filter.items())[:15]):
                    print(f"   {i+1}. {key}: {len(times)} occurrences, times: {times[:3]}...")
                    if 'Citizen' in key or 'North_Korea' in key:
                        print(f"      ⭐ FOUND POTENTIAL ANSWER: {key}")
            # else:
            time_sort = False
            if "first" in subquestion.lower():
                time_order = "asc"
                time_sort = True
            elif "last" in subquestion.lower():
                time_order = "desc"
                time_sort = True
            
            # # # step 5: sematic pruning based on the subquestion indicator path with current edge path
            # relationship_paths = self._semantic_pruning(relationship_paths, subquestion_obj, top =100)
            # print(f"🔍 semantic filtered: {len(relationship_paths)}")
            if len(original_entity) >= 2:
                paths_before_2e = len(relationship_paths)
            
                # Use None to auto-detect threshold based on dataset
                relationship_paths = self._2e_semantic_pruning(relationship_paths, original_entity, score_threshold=None)
                print(f"🔍 2 entity semantic filtered: {paths_before_2e} → {len(relationship_paths)}")

            print(f"🔍 time_order: {time_order}")
            # if time_sort:
            if False:
                # Step 5: aggregate before time sorting
                sorted_paths = self._sort_by_time(relationship_paths, order=time_order)

                if len(sorted_paths) > 80:
                    pruing_length = len(sorted_paths) // 2
                    sorted_paths = sorted_paths[:pruing_length]
                    print(f"🔍 aggregate before time sorting, paths number greater than 80, prune to {pruing_length} paths")
                else:
                    pruing_length = len(sorted_paths)
                    
                # Step 6: aggregate + aggregate after time sorting
                relationship_paths = sorted_paths
            else: 
                sorted_paths = relationship_paths
            relevant = relationship_paths
            print(f"🔍 prepare aggregate paths: {len(relevant)}")
            # for path in relevant[:10]:
            #     print(path)
            # exit()
            aggregated_paths = self.aggregate_paths(relevant, seed_entities=original_entity)
            print(f"🔍 aggregate paths: {len(aggregated_paths)}")

            # if original_entity number is one, each relation should only keep the first or last time of each relation
            # if original_entity number is greater than 1, each triple should only keep the first or last time of each triple
            # note we may have multiple tail_str, so we need to judge by tail_str, don't convert it to traditional triple
            if time_sort:
                if len(original_entity) == 1:
                    relation_groups = {}
                    for path in aggregated_paths:
                        relation = path.get('relation', '')
                        if relation not in relation_groups:
                            relation_groups[relation] = []
                        relation_groups[relation].append(path)
                    
                    # Debug: print relation distribution before filtering
                    print(f"📊 Relation distribution before First/Last filtering:")
                    for relation, group in sorted(relation_groups.items(), key=lambda x: -len(x[1]))[:10]:
                        print(f"   {relation}: {len(group)} paths")
                        # Show first path of each relation for debugging
                        if group:
                            first_path = group[0]
                            print(f"      Example: {first_path.get('head', '')} -> {relation} -> {first_path.get('tail', '')} ({first_path.get('time_start', '')})")
                    
                    # for each relation group, only keep the first or last time of each relation
                    filtered_paths = []
                    for relation, group in relation_groups.items():
                        if not group:
                            continue
                        # sort by time (using time_start_epoch timestamp for precise comparison)
                        sorted_group = sorted(
                            group, 
                            key=lambda x: x.get('time_start_epoch', 0),
                            reverse=(time_order == 'desc')
                        )
                        # only keep the first (the earliest or latest time)
                        filtered_paths.extend(sorted_group[:4])
                    
                    aggregated_paths = filtered_paths
                    print(f"🔍 First/Last mode filtered: {len(aggregated_paths)} paths (each relation keep top 4 per relation)")
                
                else:
                    # logic2: original_entity has multiple, group by (heads, relation, tails)
                    triple_groups = {}
                    for path in aggregated_paths:
                        relation = path.get('relation', '')
                        # get heads (may be list)
                        heads = path.get('heads', [path.get('head', '')])
                        # heads internal sort (standardization), but keep heads in the position of the triple
                        heads_key = tuple(sorted(heads)) if isinstance(heads, list) else (heads,)
                        
                        # get tails (may be list)
                        tails = path.get('tails', [path.get('tail', '')])
                        # tails internal sort (standardization), but keep tails in the position of the triple
                        tails_key = tuple(sorted(tails)) if isinstance(tails, list) else (tails,)
                        
                        # use (heads, relation, tails) as the group key
                        # heads and tails position has meaning, can't be swapped
                        group_key = (heads_key, relation, tails_key)
                        if group_key not in triple_groups:
                            triple_groups[group_key] = []
                        triple_groups[group_key].append(path)
                    
                    # for each triple group, only keep the first or last time of each triple
                    filtered_paths = []
                    for (heads_key, relation, tails_key), group in triple_groups.items():
                        if not group:
                            continue
                        # sort by time (using time_start_epoch timestamp for precise comparison)
                        sorted_group = sorted(
                            group, 
                            key=lambda x: x.get('time_start_epoch', 0),
                            reverse=(time_order == 'desc')
                        )
                        # only keep the first (the earliest or latest time)
                        filtered_paths.extend(sorted_group[:1])
                    
                    aggregated_paths = filtered_paths
                    print(f"🔍 First/Last mode filtered: {len(aggregated_paths)} paths (each triple keep one)")
            
            if time_sort:
                # Sort by time using t_start string (precise date comparison)
                aggregated_paths = self._sort_by_time(aggregated_paths, order=time_order)
                aggregated_paths = aggregated_paths[:80]
                print(f"🔍 Sorted by time ({time_order}), limited to top 80 paths")
            else:
                # step 5: semantic pruning based on the subquestion indicator path with current edge path
                # aggregated_paths = self._semantic_pruning(aggregated_paths, subquestion_obj, top =50)
                # only when the number of paths exceeds top_k_value, semantic filtering is performed, otherwise all paths are retained
                if len(aggregated_paths) > 80:
                    aggregated_paths = self._semantic_filter(subquestion, aggregated_paths, top_k_value=80)
                    print(f"🔍 multi e but using onehop semantic filtered: {len(aggregated_paths)}")
                else:
                    print(f"🔍 path number({len(aggregated_paths)}) not exceeds threshold(80), skip semantic filtering")


            print(f"📦 aggregate group number: {len(aggregated_paths)}")
            for path in aggregated_paths[:10]:
                # display aggregate heads and tails
                heads = path.get('heads', [path.get('head', '')])
                tails = path.get('tails', [path.get('tail', '')])
                
                # format display
                if len(heads) > 1:
                    head_display = f"[{', '.join(heads)}]"
                else:
                    head_display = heads[0] if heads else ''
                
                if len(tails) > 1:
                    tail_display = f"[{', '.join(tails)}]"
                else:
                    tail_display = tails[0] if tails else ''
                
                print(f"{head_display} {path.get('relation')} {tail_display} {path.get('time_start')}")
            
            if not aggregated_paths:
                return {"selected_path": {}, "top_3_paths": [], "error": "No valid paths", "total_paths": len(sorted_paths)}

            # Step 7: LLM final selection
            selected_path, best_but_not_answer = self.llm_path_selection(aggregated_paths, subquestion)

            # for path in aggregated_paths[:10]:
            #     # display aggregate heads and tails
            #     heads = path.get('heads', [path.get('head', '')])
            #     tails = path.get('tails', [path.get('tail', '')])
                
            #     # format display
            #     if len(heads) > 1:
            #         head_display = f"[{', '.join(heads)}]"
            #     else:
            #         head_display = heads[0] if heads else ''
                
            #     if len(tails) > 1:
            #         tail_display = f"[{', '.join(tails)}]"
            #     else:
            #         tail_display = tails[0] if tails else ''
                
            #     print(f"{head_display} {path.get('relation')} {tail_display} {path.get('time_start')}")
            
            
            # truncate topN (for toolkit limit)
            topN = best_but_not_answer
            
            return {
                "selected_path": selected_path if 'selected_path' in locals() else None,
                "top_3_paths": topN,
                "total_paths": len(sorted_paths),
                "aggregated_count": len(aggregated_paths),
                "relations_selected": len(selected_relations)
            }
        except Exception as e:
            print(f"❌ Retrieval failed: {e}")
            import traceback; traceback.print_exc()
            return {"error": str(e)}
    
    # ============= other reserved methods (_get_relations_by_entity / _select_* / _build_triplet_paths_* / _quick_time_filter / fallback) =============

    def _get_relations_by_entity(self, entity, time_constraint=None, constraint_type="after"):
        relations = set()
        seeds = entity if isinstance(entity, list) else [entity]
        for ent in seeds:
            try:
                res = self.kg_query.retrieve_one_hop(query=ent, direction="both", limit=None)
                if hasattr(res, 'edges'):
                    for edge in res.edges:
                        relations.add(getattr(edge, 'relation', ''))
            except Exception as e:
                print(f"query entity {ent} relations failed: {e}")
        return list(filter(bool, relations))
    # def _get_direct_connections(self, entity, time_constraint, constraint_type, subquestion_obj):
    
    def _find_most_similar_db_relation(self, indicator_relation: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        find the most similar relation in the database for the indicator relation
        directly use all relations in relation cache for comparison

        Args:
            indicator_relation: LLM generated indicator relation name
            top_k: return the top k most similar relations
            
        Returns:
            [(relation, similarity_score), ...] sorted by similarity score in descending order
        """
        if not self.relation_cache:
            return []
        
        # calculate embedding for the indicator relation (dynamic calculation, not cached)
        normalized_indicator = self.relation_cache.normalize_relation_text(indicator_relation)
        indicator_emb = self.relation_cache.get_openai_embedding(normalized_indicator)
        
        if indicator_emb is None:
            print(f"   ⚠️ cannot calculate embedding for '{indicator_relation}'")
            return []
        
        # directly use all relations embeddings in cache for comparison
        similarities = []
        norm1 = np.linalg.norm(indicator_emb)
        
        for cached_rel_normalized, cached_emb in self.relation_cache.embeddings.items():
            # calculate cosine similarity
            norm2 = np.linalg.norm(cached_emb)
            if norm1 > 0 and norm2 > 0:
                similarity = float(np.dot(indicator_emb, cached_emb) / (norm1 * norm2))
                # get original relation name from name mapping (preserve correct case and underscore)
                original_rel = self.relation_cache.normalized_to_original.get(
                    cached_rel_normalized, 
                    cached_rel_normalized.replace(" ", "_")  # fallback: simple replacement
                )
                similarities.append((original_rel, similarity))
        
        # sort by similarity score
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # print top k matching
        print(f"   🔍 Top {min(top_k, len(similarities))} similar relations (from {len(similarities)} cached relations):")
        for i, (rel, score) in enumerate(similarities[:top_k]):
            print(f"      {i+1}. {rel}: {score:.4f}")
        
        return similarities[:top_k]
    
    def _select_top3_relations_by_indicator(self, relations, subquestion_obj, subquestion):
        if not relations: return []
        expected = self._extract_relations_from_subquestion_obj(subquestion_obj)
        
        # 🆕 if the expected relation may not be in the database, try to use embedding to find the most matching relation
        mapped_expected = []
        if expected and self.relation_cache is not None:
            for exp_rel in expected:
                # check if the relation is in the database (by checking if there is a cached embedding)
                exp_emb = self.relation_cache.get_relation_embedding(exp_rel)
                if exp_emb is None:
                    # relation not in database, calculate its embedding and find the most similar database relation
                    print(f"🔍 detected indicator relation '{exp_rel}' not in database, automatically map to the most similar relation...")
                    similar_rels = self._find_most_similar_db_relation(exp_rel, top_k=5)
                    if similar_rels:
                        best_match, best_score = similar_rels[0]
                        print(f"   ✅ mapping: '{exp_rel}' → '{best_match}' (similarity: {best_score:.4f})")
                        if best_score > 0.6:  # only map if the similarity is high enough
                            mapped_expected.append(best_match)
                        else:
                            print(f"   ⚠️ best match similarity too low(<0.6), keep original relation for fuzzy matching")
                            mapped_expected.append(exp_rel)
                    else:
                        mapped_expected.append(exp_rel)
                else:
                    # relation in database, directly use
                    mapped_expected.append(exp_rel)
        else:
            mapped_expected = expected
        
        # use mapped expected relations for scoring
        relation_scoring = {}
        q_emb = self.get_embedding(subquestion)
        
        for rel in relations:
            sc = self._calculate_relationship_indicator_similarity(rel, mapped_expected, subquestion, q_emb)
            relation_scoring[rel] = sc
        
        # sort by similarity, take top3
        sorted_relations = sorted(relation_scoring.items(), key=lambda x: x[1], reverse=True)
        top3 = [rel for rel, _ in sorted_relations[:3]]
        print(f"🔍 similarity sorted result: {[(rel, f'{score:.4f}') for rel, score in sorted_relations[:5]]}")
        return top3
    
    def _select_top3_relations_by_text(self, relations, subquestion):
        if not relations: return []
        expected = self._extract_indicators_from_subquestion(subquestion)
        print(f"🔍 extracted expected relations: {expected}")
        q_emb = self.get_embedding(subquestion)
        relation_scoring = {}
        for rel in relations:
            sc = self._calculate_relationship_indicator_similarity(rel, expected, subquestion, q_emb)
            relation_scoring[rel] = sc  # remove sc > 0 restriction, keep all relations
            print(f"🔍 relation {rel} similarity: {sc:.4f}")
        
        # sort by similarity, take top3
        sorted_relations = sorted(relation_scoring.items(), key=lambda x: x[1], reverse=True)
        top3 = [rel for rel, _ in sorted_relations[:5]]
        print(f"🔍 similarity sorted result: {[(rel, f'{score:.4f}') for rel, score in sorted_relations[:5]]}")
        print(f"🔍 selected top3 relations: {top3}")
        return top3
    
    def _select_relations_hybrid(self, relations, subquestion_obj, subquestion, top_k=10, final_k=3):
        """
        hybrid relation selection: first use embedding to quickly select top10, then use LLM to select top3
        
        Args:
            relations: candidate relation list
            subquestion_obj: subquestion object (may contain indicator)
            subquestion: subquestion text
            top_k: number of relations selected in the first step (default 10)
            final_k: number of relations selected in the final step (default 3)
        
        Returns:
            final selected relation list
        """
        if not relations:
            return []
        
        print(f"🔄 hybrid relation selection: from {len(relations)} relations...")
        
        # Step 1: use embedding to quickly select top10
        if subquestion_obj and hasattr(subquestion_obj, 'indicator'):
            expected = self._extract_relations_from_subquestion_obj(subquestion_obj)
        else:
            expected = self._extract_indicators_from_subquestion(subquestion)
        
        q_emb = self.get_embedding(subquestion)
        relation_scoring = {}
        
        for rel in relations:
            sc = self._calculate_relationship_indicator_similarity(rel, expected, subquestion, q_emb)
            relation_scoring[rel] = sc
        
        # sort and select top_k
        sorted_relations = sorted(relation_scoring.items(), key=lambda x: x[1], reverse=True)
        top10_relations = [rel for rel, score in sorted_relations[:top_k]]
        
        print(f"📊 Embedding selected Top{top_k}: {[(rel, f'{score:.4f}') for rel, score in sorted_relations[:top_k]]}")
        
        # if candidate relation less than final_k, return directly
        if len(top10_relations) <= final_k:
            print(f"✅ candidate relation≤{final_k}, return directly")
            return top10_relations
        
        # Step 2: use LLM from top10 to select top3
        try:
            from .llm import LLM
            
            prompt = f"""Given a subquestion and a list of candidate relations, select the {final_k} most relevant relations.

Subquestion: {subquestion}

Candidate Relations:
{chr(10).join([f"{i+1}. {rel}" for i, rel in enumerate(top10_relations)])}

Please select exactly {final_k} relations that are most relevant to answering the subquestion.
Return ONLY a JSON object with the selected relation numbers (1-{len(top10_relations)}).

Example: {{"selected": [1, 3, 5]}}"""

            response = LLM.call("You are a relation selection expert.", prompt, temperature=0)
            
            # parse LLM response
            import json
            import re
            
            # try to extract JSON
            json_match = re.search(r'\{[^}]*"selected"[^}]*\}', response)
            if json_match:
                result = json.loads(json_match.group())
                selected_indices = result.get("selected", [])
                
                # convert index to relation
                final_relations = []
                for idx in selected_indices[:final_k]:
                    if 1 <= idx <= len(top10_relations):
                        final_relations.append(top10_relations[idx-1])
                
                if final_relations:
                    print(f"🤖 LLM selected Top{final_k}: {final_relations}")
                    return final_relations
            
            print(f"⚠️ LLM selection failed, use embedding Top{final_k}")
            return top10_relations[:final_k]
            
        except Exception as e:
            print(f"⚠️ LLM relation selection failed: {e}, use embedding Top{final_k}")
            return top10_relations[:final_k]
    
    def _build_triplet_paths_for_relations(self, relations, entity, time_constraint, constraint_type, subquestion_obj=None):
        paths_list = []
        seeds = entity if isinstance(entity, list) else [entity]
        all_edges = []
        if len(seeds) >= 2:
            try:
                ms = self._retrieve_multi_seed_paths(seeds, time_constraint)
                if hasattr(ms, 'edges'): all_edges.extend(ms.edges)
            except Exception as e:
                print(f"⚠️ multiple seed retrieval failed: {e}")
        for ent in seeds:
            try:
                res = self._retrieve_single_seed_paths(ent, time_constraint)
                if hasattr(res, 'edges'): all_edges.extend(res.edges)
            except Exception as e:
                print(f"⚠️ single seed retrieval failed {ent}: {e}")
        # Deduplication
        all_edges = self._prune_duplicate_paths(all_edges)
        # Relation filter
        # Create relation set for filtering
        rel_set = set(relations) if relations else set()
        relation_filtered_paths = []
        relation_number = 0
        
        print(f"🔍 used relation set: {rel_set}")
        print(f"🔍 total edge number: {len(all_edges)}")
        
        for edge in all_edges:
            rel = getattr(edge, 'relation', '') if hasattr(edge, 'relation') else (edge.get('relation', '') if isinstance(edge, dict) else '')
            if (not rel_set) or (rel in rel_set):
                triplet = {
                    'head': getattr(edge, 'head', '') if hasattr(edge, 'head') else (edge.get('head', '') if isinstance(edge, dict) else ''),
                    'relation': rel,
                    'tail': getattr(edge, 'tail', '') if hasattr(edge, 'tail') else (edge.get('tail', '') if isinstance(edge, dict) else ''),
                    'time_start': getattr(edge, 'time_start', '') if hasattr(edge, 'time_start') else (edge.get('time_start', '') if isinstance(edge, dict) else ''),
                    'time_start_epoch': getattr(edge, 'time_start_epoch', 0) if hasattr(edge, 'time_start_epoch') else (edge.get('time_start_epoch', 0) if isinstance(edge, dict) else 0)
                }
                relation_filtered_paths.append(triplet)
                
            # limit path number
            # if len(relation_filtered_paths) >= 40 and relation_number >= len(rel_set_List):
            #     break
        # self loop filtering + direction standardization
        print(f"relation_filtered_paths length: {len(relation_filtered_paths)}")
        print(f"Final used relation number: {len(relations)}")
        no_self = self._filter_self_loops(relation_filtered_paths)
        # if len(no_self)
        normalized = [self._normalize_path_direction(p, subquestion_obj) for p in no_self]
        # multiple seed: do entity similarity filtering to avoid generalization pollution
        if len(seeds) > 1:
            normalized = self._filter_paths_by_entity_similarity(normalized, seeds, similarity_threshold=0.5)
        return normalized
    
    def _quick_time_filter(self, paths, time_constraint, constraint_type, subquestion_obj=None):
        if not paths or not time_constraint:
            return paths
        s, e, g, ref_epoch, _ = self._parse_ref_epoch(time_constraint, subquestion_obj)
        
        # build context information
        context_info = {}
        context_year = None
        if subquestion_obj and hasattr(subquestion_obj, 'indicator'):
            constraints = getattr(subquestion_obj.indicator, 'constraints', [])
            context_info['constraints'] = constraints
            for constraint in constraints:
                year_match = re.search(r'(\d{4})', str(constraint))
                if year_match:
                    context_year = int(year_match.group(1))
                    break
        
        # directly use original time constraint to parse, get full range
        start_epoch = None
        end_epoch = None
        if time_constraint:
            try:
                from kg_agent.temporal_kg_toolkit import parse_time_to_range
                _, _, _, start_epoch, end_epoch = parse_time_to_range(time_constraint, context_year, context_info)
                print(f"🔍 time filtering range: {start_epoch} <= t <= {end_epoch} (constraint type: {constraint_type})")
            except Exception as ex:
                print(f"⚠️ time range conversion failed: {ex}")
                # If parsing fails, try using _parse_ref_epoch result
                if s is not None and e is not None:
                    try:
                        _, _, _, start_epoch, end_epoch = parse_time_to_range(s, context_year, context_info)
                    except:
                        # if all parsing failed, return all paths, skip time filtering
                        print(f"⚠️ time parsing completely failed, skip time filtering, return all {len(paths)} paths")
                        return paths
                else:
                    # if there is no valid time information, return all paths, skip time filtering
                    print(f"⚠️ no valid time information, skip time filtering, return all {len(paths)} paths")
                    return paths
        
        # ✅ Convert epoch seconds to years for comparison (database stores years)
        def epoch_to_year(epoch_seconds):
            """Convert epoch seconds to year"""
            if epoch_seconds is None:
                return None
            # Epoch seconds to year: divide by seconds per year (approximately)
            # 1970 epoch = 0, so year = 1970 + (epoch / seconds_per_year)
            return int(1970 + (epoch_seconds / (365.25 * 24 * 3600)))
        
        start_year = epoch_to_year(start_epoch) if start_epoch else None
        end_year = epoch_to_year(end_epoch) if end_epoch else None
        ref_year = epoch_to_year(ref_epoch) if ref_epoch else None
        
        print(f"🔍 Converted to years: start={start_year}, end={end_year}, ref={ref_year}")
        
        # Simplified time filtering using time_start_epoch (Unix timestamp)
        # Works for both MultiTQ (precise day) and TimeQuestions (year)
        filtered = []
        for p in paths:
            t_epoch = p.get('time_start_epoch', 0)
            
            if constraint_type == 'before' and ref_epoch is not None and t_epoch < ref_epoch:
                filtered.append(p)
            elif constraint_type == 'after' and ref_epoch is not None and t_epoch > ref_epoch:
                filtered.append(p)
            elif constraint_type == 'between' and start_epoch is not None and end_epoch is not None:
                if start_epoch <= t_epoch <= end_epoch:
                    filtered.append(p)
            elif constraint_type == 'equal' and start_epoch is not None and end_epoch is not None:
                if start_epoch <= t_epoch <= end_epoch:
                    filtered.append(p)
        
        return filtered
    
    def fallback_path_selection(self, aggregated_paths: List[Dict[str, Any]]) -> Dict[str, Any]:
        best = aggregated_paths[0]
        best['selection_reason'] = "fallback: aggregated top-1"
        return best

    # ================== Utilities ==================
    def _fold_time_args(self, after: Optional[str] = None, before: Optional[str] = None,
                        between: Optional[tuple] = None, same_day: Optional[str] = None,
                        same_month: Optional[str] = None) -> tuple:
        """
        fold OneHop style time parameters into (time_constraint, constraint_type, time_order)
        """
        if same_day:
            return same_day, "equal", "asc"  # use equal type, match specific date
        if same_month:
            # use equal type, match specific month
            return same_month, "equal", "asc"
        if between:
            assert isinstance(between, (list, tuple)) and len(between) == 2
            return f"{between[0]}/{between[1]}", "between", "asc"
        if after and before:
            # if both are provided, convert to between
            return f"{after}/{before}", "between", "asc"
        if after:
            return after, "after", "asc"
        if before:
            return before, "before", "desc"
        return None, "after", "asc"
