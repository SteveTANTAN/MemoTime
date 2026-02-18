
# =============================
# file: kg_agent/kg_ops.py
# =============================
import sys
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from .registry import TemplateCard

# import temporal knowledge graph toolkit
try:
    from .temporal_kg_toolkit import TemporalKGToolkit, create_toolkit
    TOOLKIT_AVAILABLE = True
except ImportError:
    print("Warning: temporal_kg_toolkit not available, using stub functions")
    TOOLKIT_AVAILABLE = False
    
    class StubToolkit:
        def retrieve_one_hop(self, *args, **kwargs):
            return type('QueryResult', (), {'edges': [], 'total_count': 0, 'query_params': {}})()
        def events_on_day(self, *args, **kwargs):
            return type('QueryResult', (), {'edges': [], 'total_count': 0, 'query_params': {}})()
        def events_in_month(self, *args, **kwargs):
            return type('QueryResult', (), {'edges': [], 'total_count': 0, 'query_params': {}})()
        def events_in_year(self, *args, **kwargs):
            return type('QueryResult', (), {'edges': [], 'total_count': 0, 'query_params': {}})()
        def find_entities_by_name_pattern(self, *args, **kwargs):
            return []
        def get_entity_statistics(self, *args, **kwargs):
            return {}
    
    def create_toolkit(db_path):
        return StubToolkit()

class KG:
    # database path configuration (dynamic from TPKGConfig)
    _toolkit = None
    _last_db_path = None
    
    @classmethod
    def get_db_path(cls):
        """get current dataset database path"""
        try:
            from config import TPKGConfig
            return TPKGConfig.DB_PATH
        except:
            # fallback to default path
            return str(Path(__file__).parent.parent.parent / "Data" / "tempkg_wt.db")
    
    @classmethod
    def get_toolkit(cls):
        """get toolkit instance (automatically detect database path change and reinitialize)"""
        current_db_path = cls.get_db_path()
        
        # if database path changed, reinitialize toolkit
        if cls._toolkit is None or cls._last_db_path != current_db_path:
            cls._toolkit = create_toolkit(current_db_path)
            cls._last_db_path = current_db_path
            print(f"🔄 KG toolkit initialized: {current_db_path}")
        
        return cls._toolkit
    
    @classmethod
    def get_entity_name(cls, entity_id: int) -> str:
        """get entity name by entity ID"""
        try:
            import sqlite3
            db_path = cls.get_db_path()
            conn = sqlite3.connect(db_path)
            cur = conn.cursor()
            cur.execute("SELECT name FROM entities WHERE id = ?", (entity_id,))
            entity_row = cur.fetchone()
            conn.close()
            
            if entity_row:
                return entity_row[0]
            else:
                return None
        except Exception as e:
            print(f"get entity name failed (DB: {cls.get_db_path()}): {e}")
            return None
    

    
    @classmethod
    def get_entity_id(cls, entity_name: str) -> int:
        """get entity ID by entity name"""
        try:
            import sqlite3
            db_path = cls.get_db_path()
            conn = sqlite3.connect(db_path)
            cur = conn.cursor()
            cur.execute("SELECT id FROM entities WHERE name = ?", (entity_name,))
            entity_row = cur.fetchone()
            conn.close()
            
            if entity_row:
                return entity_row[0]
            else:
                return None
        except Exception as e:
            print(f"get entity ID failed: {e}")
            return None
    
    @staticmethod
    def entity_link(question: str, topic_hint: Optional[str] = None) -> Dict[str, Any]:
        # TODO: implement your EL if needed
        return {"anchor_entity": None, "linked_entities": []}

    @staticmethod
    def retrieve_paths_for_entity(entity_name: str, seeds: List[int] = None, 
                                 time_constraints: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        retrieve one hop path for given entity
        return format: [{"head": str, "relation": str, "tail": str, "time": str, "granularity": str, "score": float}]
        """
        try:
            # directly query database, because structure does not match temporal_kg_pack
            import sqlite3
            conn = sqlite3.connect(cls.get_db_path())
            cur = conn.cursor()
            
            # get entity ID
            cur.execute("SELECT id FROM entities WHERE name = ?", (entity_name,))
            entity_row = cur.fetchone()
            if not entity_row:
                conn.close()
                return []
            
            entity_id = entity_row[0]
            
            # query one hop path, increase limit to ensure all related relations are included
            cur.execute("""
                SELECT h.name, e.relation, t.name, e.t_start, e.t_end, e.granularity
                FROM edges e
                JOIN entities h ON h.id = e.head_id
                JOIN entities t ON t.id = e.tail_id
                WHERE e.head_id = ? OR e.tail_id = ?
                ORDER BY e.t_start_epoch DESC
                LIMIT 2000
            """, (entity_id, entity_id))
            
            rows = cur.fetchall()
            conn.close()
            
            # convert to standard format and normalize path
            formatted_paths = []
            for head, relation, tail, t_start, t_end, granularity in rows:
                time_range = f"{t_start}~{t_end}" if t_start != t_end else t_start
                
                # normalize path: ensure path format consistent, no directionality
                normalized_path = KG._normalize_path(head, relation, tail, entity_name)
                
                formatted_paths.append({
                    "head": normalized_path["head"],
                    "relation": normalized_path["relation"],
                    "tail": normalized_path["tail"],
                    "time": time_range,
                    "granularity": granularity,
                    "score": 1.0,  # initial score, subsequent by BERT and LLM
                    "original_head": head,
                    "original_tail": tail
                })
            
            return formatted_paths
        except Exception as e:
            print(f"Error retrieving paths for {entity_name}: {e}")
            return []

    @staticmethod
    def _normalize_path(head: str, relation: str, tail: str, seed_entity: str) -> Dict[str, str]:
        """
        normalize path format, ensure path no directionality
        according to the position of seed_entity in the path, but the relation name remains unchanged
        """
        # if seed_entity is head, keep original direction
        if head == seed_entity:
            return {"head": head, "relation": relation, "tail": tail}
        # if seed_entity is tail, reverse direction but keep relation name unchanged
        elif tail == seed_entity:
            return {"head": tail, "relation": relation, "tail": head}
        else:
            # if seed_entity is not in the path, keep original direction
            return {"head": head, "relation": relation, "tail": tail}

    @staticmethod
    def bert_filter_paths(paths: List[Dict[str, Any]], indicator: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        use BERT model to pre-filter paths, select the top 40 paths most similar to indicator
        """
        if not paths:
            return []
        
        try:
            # try to import transformers
            from transformers import AutoTokenizer, AutoModel
            import torch
            import numpy as np
            from sklearn.metrics.pairwise import cosine_similarity
            
            # load BERT model
            model_name = "bert-base-uncased"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)
            
            # build indicator text
            indicator_text = KG._build_indicator_text(indicator)
            
            # get indicator embedding
            indicator_tokens = tokenizer(indicator_text, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                indicator_embedding = model(**indicator_tokens).last_hidden_state.mean(dim=1)
            
            # calculate similarity between each path and indicator
            path_scores = []
            for i, path in enumerate(paths):
                path_text = f"{path['head']} {path['relation']} {path['tail']}"
                path_tokens = tokenizer(path_text, return_tensors="pt", truncation=True, padding=True)
                with torch.no_grad():
                    path_embedding = model(**path_tokens).last_hidden_state.mean(dim=1)
                
                # calculate cosine similarity
                similarity = float(cosine_similarity(
                    indicator_embedding.numpy(), 
                    path_embedding.numpy()
                )[0][0])
                
                path_scores.append((i, similarity))
            
            # sort by similarity, select top 40 paths
            path_scores.sort(key=lambda x: x[1], reverse=True)
            top_40_indices = [idx for idx, _ in path_scores[:40]]
            
            # return top 40 paths, and update BERT score
            filtered_paths = []
            for idx in top_40_indices:
                path = paths[idx].copy()
                path["bert_score"] = path_scores[idx][1]
                filtered_paths.append(path)
            
            print(f"BERT filtered {len(paths)} paths to {len(filtered_paths)} paths")
            return filtered_paths
            
        except ImportError:
            print("Transformers not available, using simple text matching")
            return KG._simple_text_filter(paths, indicator)
        except Exception as e:
            print(f"Error in BERT filtering: {e}")
            return paths[:40]  # return top 40 paths as fallback

    @staticmethod
    def _build_indicator_text(indicator: Dict[str, Any]) -> str:
        """build indicator text representation"""
        edges = indicator.get("edges", [])
        constraints = indicator.get("constraints", [])
        
        edge_texts = []
        for edge in edges:
            subj = edge.get("subj", "")
            rel = edge.get("rel", "")
            obj = edge.get("obj", "")
            edge_texts.append(f"{subj} {rel} {obj}")
        
        indicator_text = " ".join(edge_texts)
        if constraints:
            indicator_text += " " + " ".join(constraints)
        
        return indicator_text

    @staticmethod
    def _simple_text_filter(paths: List[Dict[str, Any]], indicator: Dict[str, Any]) -> List[Dict[str, Any]]:
        """simple text matching filter, when BERT is not available"""
        indicator_text = KG._build_indicator_text(indicator).lower()
        
        path_scores = []
        for i, path in enumerate(paths):
            path_text = f"{path['head']} {path['relation']} {path['tail']}".lower()
            
            # simple word overlap calculation
            indicator_words = set(indicator_text.split())
            path_words = set(path_text.split())
            overlap = len(indicator_words.intersection(path_words))
            score = overlap / max(len(indicator_words), 1)
            
            path_scores.append((i, score))
        
        path_scores.sort(key=lambda x: x[1], reverse=True)
        top_40_indices = [idx for idx, _ in path_scores[:40]]
        
        filtered_paths = []
        for idx in top_40_indices:
            path = paths[idx].copy()
            path["bert_score"] = float(path_scores[idx][1])
            filtered_paths.append(path)
        
        return filtered_paths

    @staticmethod
    def llm_batch_select(paths: List[Dict[str, Any]], subquestion: str, 
                        context: Dict[str, Any] = None, batch_size: int = 20, 
                        select_per_batch: int = 3, total_select: int = 6) -> List[Dict[str, Any]]:
        """
        LLM batch selection: select 3 from 20 each time, finally get 6 candidate answers
        include retry mechanism and BERT fallback
        """
        if not paths:
            return []
        
        selected_paths = []
        remaining_paths = paths.copy()
        
        while len(selected_paths) < total_select and remaining_paths:
            # get current batch
            current_batch = remaining_paths[:batch_size]
            remaining_paths = remaining_paths[batch_size:]
            
            if not current_batch:
                break
            
            # build current batch path description
            paths_text = "\n".join([
                f"Path {i+1}: {p['head']} --[{p['relation']}]--> {p['tail']} (Time: {p['time']})"
                for i, p in enumerate(current_batch)
            ])
            
            # use new simplified prompt
            from .prompts import LLM_PATH_SELECT_PROMPT
            prompt = LLM_PATH_SELECT_PROMPT.format(
                subquestion=subquestion,
                paths=paths_text
            )
            
            # retry mechanism:最多重试3次
            batch_selected = False
            for attempt in range(3):
                try:
                    from .llm import LLM
                    from .prompts import LLM_SYSTEM_PROMPT
                    import json
                    
                    response = LLM.call(LLM_SYSTEM_PROMPT, prompt)
                    print(f"LLM Response (attempt {attempt + 1}): {response[:200]}...")
                    
                    # try to parse JSON
                    try:
                        result = json.loads(response)
                        selected_ids = result.get("selected_paths", [])
                        
                        if not selected_ids or len(selected_ids) == 0:
                            raise ValueError("No paths selected")
                        
                        # select current batch paths
                        for path_id in selected_ids[:select_per_batch]:
                            if 1 <= path_id <= len(current_batch):
                                selected_path = current_batch[path_id - 1].copy()  # convert to 0-based index
                                selected_path["llm_score"] = 0.8  # default high score
                                selected_path["reason"] = f"Selected by LLM (attempt {attempt + 1})"
                                selected_paths.append(selected_path)
                                
                                if len(selected_paths) >= total_select:
                                    break
                        
                        batch_selected = True
                        break  # success, break retry loop
                        
                    except (json.JSONDecodeError, ValueError) as e:
                        print(f"JSON parse error (attempt {attempt + 1}): {e}")
                        if attempt < 2:  # not last attempt
                            continue
                        else:
                            raise e
                            
                except Exception as e:
                    print(f"LLM selection error (attempt {attempt + 1}): {e}")
                    if attempt < 2:  # not last attempt
                        continue
                    else:
                        # last attempt failed, use BERT fallback
                        print("All LLM attempts failed, using BERT fallback")
                        break
            
            # if LLM completely failed, use BERT to select top 3
            if not batch_selected:
                print("Using BERT fallback for path selection")
                bert_selected = KG._bert_fallback_select(current_batch, select_per_batch)
                selected_paths.extend(bert_selected)
        
        return selected_paths
    
    @staticmethod
    def _bert_fallback_select(paths: List[Dict[str, Any]], select_count: int) -> List[Dict[str, Any]]:
        """
        BERT fallback: when LLM failed, use BERT score to select top paths
        """
        # sort by BERT score
        sorted_paths = sorted(paths, key=lambda x: x.get("bert_score", 0.0), reverse=True)
        
        selected = []
        for i in range(min(select_count, len(sorted_paths))):
            path = sorted_paths[i].copy()
            path["llm_score"] = 0.6  # lower than LLM selected score
            path["reason"] = "Selected by BERT fallback"
            selected.append(path)
        
        return selected

    @staticmethod
    def llm_prune_paths(paths: List[Dict[str, Any]], subquestion: str, 
                       context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        use BERT + LLM to prune and score paths
        """
        if not paths:
            return []
        
        # first use BERT to pre-filter to top 40
        bert_filtered = KG.bert_filter_paths(paths, context.get("indicator", {}))
        # for candidate in bert_filtered[:5]:
        #     print("bert_filter_paths: ", candidate)
        # # exit()

        # then use LLM batch selection, finally get 6 candidate answers
        final_candidates = KG.llm_batch_select(bert_filtered, subquestion, context)
        # for candidate in final_candidates:
        #     print("candidate: ", candidate)
        # exit()
        return final_candidates

    @staticmethod
    def run_workflow(card: TemplateCard, question: str, linked: Dict[str, Any]) -> Dict[str, Any]:
        """Execute KG-only logic according to a template card or indicator.
        Return: {items: [{entity, time, path, provenance}], explanations: [...], verification: {...}}
        """
        try:
            # get seed entities
            seeds = linked.get("seeds", [])
            context = linked.get("ctx", {})
            indicator = linked.get("indicator", {})
            
            all_candidates = []
            all_paths = []
            
            # retrieve paths for each seed entity
            for seed_id in seeds:
                try:
                    # directly query database to get entity name
                    import sqlite3
                    conn = sqlite3.connect(KG.get_db_path())
                    cur = conn.cursor()
                    cur.execute("SELECT name FROM entities WHERE id = ?", (seed_id,))
                    entity_row = cur.fetchone()
                    conn.close()
                    
                    if not entity_row:
                        print(f"Entity ID {seed_id} not found in database")
                        continue
                    
                    entity_name_str = entity_row[0]
                    
                    # retrieve one hop path
                    paths = KG.retrieve_paths_for_entity(entity_name_str, [seed_id])
                    
                    if not paths:
                        print(f"No paths found for entity {entity_name_str}")
                        continue
                    
                    # collect all paths
                    all_paths.extend(paths)
                        
                except Exception as e:
                    print(f"Error processing seed {seed_id}: {e}")
                    continue
            
            if not all_paths:
                return {
                    "items": [],
                    "explanations": ["No paths found for any seed entities"],
                    "verification": {"passed": False, "details": ["No paths retrieved"]},
                }
            
            # sort paths by indicator
            sorted_paths = KG._sort_paths_by_indicator(all_paths, indicator)
            
            # use BERT + LLM to prune and select
            final_candidates = KG.llm_prune_paths(sorted_paths, question, context)
            
            # convert to candidate answer format
            for path in final_candidates:
                candidate = {
                    "entity": path["tail"],
                    "time": path["time"],
                    "path": [path["head"], path["relation"], path["tail"]],
                    "provenance": {
                        "source_entity": path["head"],
                        "relation": path["relation"],
                        "bert_score": path.get("bert_score", 0.0),
                        "llm_score": path.get("llm_score", 0.0),
                        "reason": path.get("reason", "")
                    }
                }
                all_candidates.append(candidate)
            
            # sort by LLM score
            all_candidates.sort(key=lambda x: x["provenance"]["llm_score"], reverse=True)
            
            # return result
            return {
                "items": all_candidates,
                "explanations": [
                    f"Retrieved {len(all_paths)} total paths from {len(seeds)} seed entities",
                    f"BERT filtered to top 40 paths",
                    f"LLM selected {len(all_candidates)} final candidates",
                    f"Top candidate: {all_candidates[0]['entity'] if all_candidates else 'None'}"
                ],
                "verification": {
                    "passed": len(all_candidates) > 0,
                    "details": [f"Found {len(all_candidates)} high-quality candidates"]
                },
            }
            
        except Exception as e:
            print(f"Error in run_workflow: {e}")
            return {
                "items": [],
                "explanations": [f"Executed {card.workflow_id} (error: {str(e)})"],
                "verification": {"passed": False, "details": [f"Error: {str(e)}"]},
            }

    @staticmethod
    def _sort_paths_by_indicator(paths: List[Dict[str, Any]], indicator: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        sort paths by indicator, ensure indicator entities are in the front of paths
        """
        if not indicator or not indicator.get("edges"):
            return paths
        
        # extract indicator entities
        indicator_entities = set()
        for edge in indicator.get("edges", []):
            subj = edge.get("subj", "")
            obj = edge.get("obj", "")
            if subj and subj != "?x":
                indicator_entities.add(subj)
            if obj and obj != "?x":
                indicator_entities.add(obj)
        
        def path_priority(path):
            """calculate path priority, paths with indicator entities in the front have higher priority"""
            head = path.get("head", "")
            tail = path.get("tail", "")
            
            # if path head is in indicator, higher priority
            if head in indicator_entities:
                return 0
            # if path tail is in indicator, medium priority
            elif tail in indicator_entities:
                return 1
            # other paths have lowest priority
            else:
                return 2
        
        # sort by priority
        sorted_paths = sorted(paths, key=path_priority)
        return sorted_paths

    @staticmethod
    def query_with_toolkit(query_type: str, **kwargs) -> Dict[str, Any]:
        """
        use toolkit to query
        
        Args:
            query_type: query type ("one_hop", "events_on_day", "events_in_month", "find_entities", "entity_stats")
            **kwargs: query parameters
        
        Returns:
            query result
        """
        try:
            toolkit = KG.get_toolkit()
            
            if query_type == "one_hop":
                result = toolkit.retrieve_one_hop(**kwargs)
                return {
                    "success": True,
                    "query_type": "one_hop",
                    "edges": [
                        {
                            "head": edge.head,
                            "relation": edge.relation,
                            "tail": edge.tail,
                            "time_start": edge.time_start,
                            "time_end": edge.time_end,
                            "granularity": edge.granularity,
                            "head_id": edge.head_id,
                            "tail_id": edge.tail_id
                        }
                        for edge in result.edges
                    ],
                    "total_count": result.total_count,
                    "query_params": result.query_params
                }
            
            elif query_type == "events_on_day":
                result = toolkit.events_on_day(**kwargs)
                return {
                    "success": True,
                    "query_type": "events_on_day",
                    "edges": [
                        {
                            "head": edge.head,
                            "relation": edge.relation,
                            "tail": edge.tail,
                            "time_start": edge.time_start,
                            "time_end": edge.time_end,
                            "granularity": edge.granularity
                        }
                        for edge in result.edges
                    ],
                    "total_count": result.total_count,
                    "query_params": result.query_params
                }
            
            elif query_type == "events_in_month":
                result = toolkit.events_in_month(**kwargs)
                return {
                    "success": True,
                    "query_type": "events_in_month",
                    "edges": [
                        {
                            "head": edge.head,
                            "relation": edge.relation,
                            "tail": edge.tail,
                            "time_start": edge.time_start,
                            "time_end": edge.time_end,
                            "granularity": edge.granularity
                        }
                        for edge in result.edges
                    ],
                    "total_count": result.total_count,
                    "query_params": result.query_params
                }
            
            elif query_type == "events_in_year":
                result = toolkit.events_in_year(**kwargs)
                return {
                    "success": True,
                    "query_type": "events_in_year",
                    "edges": [
                        {
                            "head": edge.head,
                            "relation": edge.relation,
                            "tail": edge.tail,
                            "time_start": edge.time_start,
                            "time_end": edge.time_end,
                            "granularity": edge.granularity
                        }
                        for edge in result.edges
                    ],
                    "total_count": result.total_count,
                    "query_params": result.query_params
                }
            
            elif query_type == "find_entities":
                entities = toolkit.find_entities_by_name_pattern(**kwargs)
                return {
                    "success": True,
                    "query_type": "find_entities",
                    "entities": entities,
                    "total_count": len(entities)
                }
            
            elif query_type == "entity_stats":
                stats = toolkit.get_entity_statistics(**kwargs)
                return {
                    "success": True,
                    "query_type": "entity_stats",
                    "statistics": stats
                }
            
            elif query_type == "before_last":
                result = toolkit.advanced.find_before_last(**kwargs)
                return {
                    "success": True,
                    "query_type": "before_last",
                    "edges": [
                        {
                            "head": edge.head,
                            "relation": edge.relation,
                            "tail": edge.tail,
                            "time_start": edge.time_start,
                            "time_end": edge.time_end,
                            "granularity": edge.granularity
                        }
                        for edge in result.edges
                    ],
                    "total_count": result.total_count,
                    "query_params": result.query_params
                }
            
            elif query_type == "after_first":
                result = toolkit.advanced.find_after_first(**kwargs)
                return {
                    "success": True,
                    "query_type": "after_first",
                    "edges": [
                        {
                            "head": edge.head,
                            "relation": edge.relation,
                            "tail": edge.tail,
                            "time_start": edge.time_start,
                            "time_end": edge.time_end,
                            "granularity": edge.granularity
                        }
                        for edge in result.edges
                    ],
                    "total_count": result.total_count,
                    "query_params": result.query_params
                }
            
            elif query_type == "between_times":
                result = toolkit.advanced.find_between_times(**kwargs)
                return {
                    "success": True,
                    "query_type": "between_times",
                    "edges": [
                        {
                            "head": edge.head,
                            "relation": edge.relation,
                            "tail": edge.tail,
                            "time_start": edge.time_start,
                            "time_end": edge.time_end,
                            "granularity": edge.granularity
                        }
                        for edge in result.edges
                    ],
                    "total_count": result.total_count,
                    "query_params": result.query_params
                }
            
            elif query_type == "temporal_neighbors":
                result = toolkit.advanced.find_temporal_neighbors(**kwargs)
                return {
                    "success": True,
                    "query_type": "temporal_neighbors",
                    "edges": [
                        {
                            "head": edge.head,
                            "relation": edge.relation,
                            "tail": edge.tail,
                            "time_start": edge.time_start,
                            "time_end": edge.time_end,
                            "granularity": edge.granularity
                        }
                        for edge in result.edges
                    ],
                    "total_count": result.total_count,
                    "query_params": result.query_params
                }
            
            elif query_type == "chronological_sequence":
                result = toolkit.advanced.find_chronological_sequence(**kwargs)
                return {
                    "success": True,
                    "query_type": "chronological_sequence",
                    "edges": [
                        {
                            "head": edge.head,
                            "relation": edge.relation,
                            "tail": edge.tail,
                            "time_start": edge.time_start,
                            "time_end": edge.time_end,
                            "granularity": edge.granularity
                        }
                        for edge in result.edges
                    ],
                    "total_count": result.total_count,
                    "query_params": result.query_params
                }
            
            elif query_type == "time_gaps":
                gaps = toolkit.advanced.find_time_gaps(**kwargs)
                return {
                    "success": True,
                    "query_type": "time_gaps",
                    "gaps": gaps,
                    "total_count": len(gaps)
                }
            
            else:
                return {
                    "success": False,
                    "error": f"Unknown query type: {query_type}",
                    "available_types": [
                        "one_hop", "events_on_day", "events_in_month", "events_in_year", "find_entities", "entity_stats",
                        "before_last", "after_first", "between_times", "temporal_neighbors", 
                        "chronological_sequence", "time_gaps"
                    ]
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "query_type": query_type
            }

    @staticmethod
    def get_available_tools() -> Dict[str, Any]:
        """get available tool information"""
        try:
            toolkit = KG.get_toolkit()
            if hasattr(toolkit, 'get_toolkit_info'):
                return toolkit.get_toolkit_info()
            else:
                return {
                    "available_tools": [
                        "retrieve_one_hop",
                        "events_on_day", 
                        "events_in_month",
                        "events_in_year",
                    "find_entities_by_name_pattern",
                    "get_entity_statistics"
                ],
                "db_path": KG.get_db_path()
            }
        except Exception as e:
            return {
                "error": str(e),
                "available_tools": ["stub_functions_only"]
            }

