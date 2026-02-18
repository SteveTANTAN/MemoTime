#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Knowledge Store for TPKG System
Unified knowledge store: merge experience pool and template learning data
"""

import os
import sqlite3
import json
import hashlib
import time
import pickle
from typing import Optional, Dict, Any, List, Tuple, Union
from datetime import datetime
from collections import defaultdict
import numpy as np
from dataclasses import dataclass, asdict

# Import storage manager
from .storage_manager import get_storage_manager, ExperimentSetting, StorageMode

# Optional sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
    _SBERT = SentenceTransformer("all-MiniLM-L6-v2")
    SBERT_AVAILABLE = True
except Exception:
    _SBERT = None
    SBERT_AVAILABLE = False

@dataclass
class KnowledgeEntry:
    """Unified knowledge entry"""
    # Basic information
    key_hash: str
    query_text: str
    query_type: str  # 'main_question' or 'subquestion'
    norm_text: str
    
    # Experience pool data
    entities_json: str
    time_constraint_json: str
    indicators_json: str
    evidence_json: str
    toolkit_params_json: str
    
    # Template learning data
    question_type: Optional[str] = None
    template_data_json: Optional[str] = None
    decomposition_data_json: Optional[str] = None
    execution_data_json: Optional[str] = None
    
    # Metadata
    success_rate: float = 0.0
    hit_count: int = 0
    last_hit_at: Optional[str] = None
    created_at: str = ""
    updated_at: str = ""
    
    # Cache and performance
    access_frequency: int = 0
    last_access: Optional[str] = None

class LRUCache:
    """LRU cache implementation"""
    
    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self.cache = {}
        self.access_order = []
    
    def get(self, key: str) -> Optional[Any]:
        if key in self.cache:
            # Update access order
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        return None
    
    def put(self, key: str, value: Any):
        if key in self.cache:
            # Update existing item
            self.access_order.remove(key)
        elif len(self.cache) >= self.capacity:
            # Delete the least recently used item
            oldest = self.access_order.pop(0)
            del self.cache[oldest]
        
        self.cache[key] = value
        self.access_order.append(key)
    
    def remove(self, key: str):
        if key in self.cache:
            del self.cache[key]
            self.access_order.remove(key)

class UnifiedKnowledgeStore:
    """
        Unified knowledge store system
    
    Merge experience pool and template learning data, provide a unified query interface
    Support LRU cache and buffer management
    """
    
    def __init__(self, db_path: str = None, experiment_setting: Optional[ExperimentSetting] = None,
                 cache_size: int = 1000, buffer_size: int = 100):
        """
        Initialize unified knowledge store
        
        Args:
            db_path: Database path
            experiment_setting: Experiment setting
            cache_size: LRU cache size
            buffer_size: Buffer size
        """
        # Initialize storage manager
        self.storage_manager = get_storage_manager()
        if experiment_setting:
            self.storage_manager.set_experiment_setting(experiment_setting)
        
        if db_path is None:
            base_path = self.storage_manager.get_storage_path("unified_knowledge", setting=experiment_setting)
            os.makedirs(base_path, exist_ok=True)
            self.db_path = os.path.join(base_path, "unified_knowledge.db")
        else:
            # Ensure database directory exists
            db_dir = os.path.dirname(db_path)
            if db_dir and not os.path.exists(db_dir):
                os.makedirs(db_dir, exist_ok=True)
            self.db_path = db_path
        
        # Initialize cache and buffer
        self.cache = LRUCache(cache_size)
        self.buffer = []
        self.buffer_size = buffer_size
        self.buffer_lock = False
        
        # Initialize database
        self._init_db()
        
        # Initialize embedding model
        self.sentence_transformer = _SBERT if SBERT_AVAILABLE else None
        
        # Initialize vector index (lazy loading)
        self.vector_index = None
        self._use_vector_index = True  # Default enable vector index
    
    def _init_db(self):
        """Initialize database table structure"""
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)
            
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        
        # Create unified knowledge table
        cur.execute("""
        CREATE TABLE IF NOT EXISTS unified_knowledge (
          key_hash TEXT PRIMARY KEY,
          query_text TEXT NOT NULL,
          query_type TEXT NOT NULL,  -- 'main_question' or 'subquestion'
          norm_text TEXT NOT NULL,
          
          -- Experience pool data
          entities_json TEXT,
          time_constraint_json TEXT,
          indicators_json TEXT,
          evidence_json TEXT,
          toolkit_params_json TEXT,
          
          -- Template learning data
          question_type TEXT,
          template_data_json TEXT,
          decomposition_data_json TEXT,
          execution_data_json TEXT,
          
          -- Sufficiency test parameters
          sufficiency_args_json TEXT,
          
          -- Metadata
          success_rate REAL DEFAULT 0.0,
          hit_count INTEGER DEFAULT 0,
          last_hit_at TEXT,
          access_frequency INTEGER DEFAULT 0,
          last_access TEXT,
          created_at TEXT NOT NULL,
          updated_at TEXT NOT NULL
        )""")
        
        # Check and add missing columns (for existing database compatibility)
        cur.execute("PRAGMA table_info(unified_knowledge)")
        existing_columns = [row[1] for row in cur.fetchall()]
        
        if 'sufficiency_args_json' not in existing_columns:
            cur.execute("ALTER TABLE unified_knowledge ADD COLUMN sufficiency_args_json TEXT")
        
        if 'validation_failed' not in existing_columns:
            cur.execute("ALTER TABLE unified_knowledge ADD COLUMN validation_failed INTEGER DEFAULT 0")
        
            # Create index
        cur.execute("CREATE INDEX IF NOT EXISTS idx_query_text ON unified_knowledge(query_text)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_query_type ON unified_knowledge(query_type)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_norm_text ON unified_knowledge(norm_text)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_question_type ON unified_knowledge(question_type)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_hit_count ON unified_knowledge(hit_count)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_access_freq ON unified_knowledge(access_frequency)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_updated_at ON unified_knowledge(updated_at)")
        
        conn.commit()
        conn.close()
    
    def _normalize_text(self, text: str, entities: List[str] = None, time_constraint: Dict[str, Any] = None) -> str:
        """Normalize text"""
        def canon(s: str) -> str:
            return s.lower().replace("_", " ").strip()
        
        # Normalize entity list
        ent = sorted([canon(e) for e in entities or []])
        
        # Normalize time constraint
        tkey = json.dumps(time_constraint or {}, sort_keys=True)
        
        # Combine into normalized representation
        base = f"{canon(text)} || ENT:{'|'.join(ent)} || T:{tkey}"
        return base
    
    def _generate_key_hash(self, query_text: str, query_type: str, entities: List[str] = None, 
                          time_constraint: Dict[str, Any] = None) -> str:
        """Generate key hash"""
        norm_text = self._normalize_text(query_text, entities, time_constraint)
        combined = f"{query_type}:{norm_text}"
        return hashlib.sha1(combined.encode("utf-8")).hexdigest()
    
    def _embed_text(self, text: str) -> List[float]:
        """Text embedding"""
        if self.sentence_transformer is not None:
            try:
                vector = self.sentence_transformer.encode([text])[0]
                return vector.astype(np.float32).tolist()
            except Exception as e:
                print(f"Embedding failed: {e}")
        
        # Degradation: word frequency vector
        toks = text.split()
        freq = {}
        for w in toks:
            freq[w] = freq.get(w, 0) + 1
        top = sorted(freq.items(), key=lambda x: (-x[1], x[0]))[:32]
        return [float(v) for _, v in top] + [0.0] * max(0, 32 - len(top))
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity"""
        if not a or not b:
            return 0.0
        
        import math
        la = math.sqrt(sum(x*x for x in a)) or 1.0
        lb = math.sqrt(sum(x*x for x in b)) or 1.0
        return sum(x*y for x, y in zip(a, b)) / (la * lb)
    
    def _get_vector_index(self):
        """Get or initialize vector index"""
        if not self._use_vector_index:
            return None
        
        if self.vector_index is None:
            try:
                from .vector_index import get_vector_index
                self.vector_index = get_vector_index(self.db_path)
                # Check if index needs to be built/updated
                self.vector_index.build_index()
            except Exception as e:
                print(f"⚠️  Vector index initialization failed: {e}")
                self._use_vector_index = False
                return None
        
        return self.vector_index
    
    def _search_across_individual_databases(self,
                                           query_text: str,
                                           query_type: str = None,
                                           entities: List[str] = None,
                                           time_constraint: Dict[str, Any] = None,
                                           k: int = 5,
                                           sim_threshold: float = 0.7,
                                           include_template_data: bool = True,
                                           include_experience_data: bool = True) -> List[Dict[str, Any]]:
        """
        In shared mode, search all individual databases.
        Aggregate all individual database results, sort by similarity and return top-k.
        In shared mode, search all individual databases.
        Aggregate all individual database results, sort by similarity and return top-k.
        
        Args:
            Same as lookup_knowledge
        
        Returns:
            Aggregated knowledge entry list
        """
        print(f"🔍 Shared mode: search all individual databases...")
        
        # Get data storage base path
        base_data_path = os.path.dirname(self.db_path)  # e.g., Data/unified_knowledge
        parent_data_path = os.path.dirname(base_data_path)  # e.g., Data/
        
        # Find all individual database directories
        individual_db_dirs = []
        if os.path.exists(parent_data_path):
            for entry in os.listdir(parent_data_path):
                entry_path = os.path.join(parent_data_path, entry)
                if os.path.isdir(entry_path) and entry.startswith("unified_knowledge_"):
                    db_file = os.path.join(entry_path, "unified_knowledge.db")
                    if os.path.exists(db_file):
                        individual_db_dirs.append(db_file)
        
        print(f"    Find {len(individual_db_dirs)} individual databases")
        
        # Aggregate all results
        all_results = []
        
        for db_file in individual_db_dirs:
            try:
                # Directly query each database (using SQL, because vector index may not exist)
                results = self._query_single_database(
                    db_file, query_text, query_type, entities, time_constraint,
                    k * 2,  # Each database takes more candidates
                    sim_threshold, include_template_data, include_experience_data
                )
                all_results.extend(results)
            except Exception as e:
                print(f"⚠️  Query database {db_file} failed: {e}")
                continue
        
        # Sort by similarity and remove duplicates
        all_results.sort(key=lambda x: x.get('similarity', 0), reverse=True)
        
        # Remove duplicates (based on key_hash)
        seen_hashes = set()
        unique_results = []
        for result in all_results:
            key_hash = result.get('key_hash')
            if key_hash and key_hash not in seen_hashes:
                seen_hashes.add(key_hash)
                unique_results.append(result)
        
        # Return top-k
        final_results = unique_results[:k]
        print(f"   🎯 Return {len(final_results)} results after aggregation")
        
        return final_results
    
    def _query_single_database(self,
                               db_path: str,
                               query_text: str,
                               query_type: str = None,
                               entities: List[str] = None,
                               time_constraint: Dict[str, Any] = None,
                               k: int = 10,
                               sim_threshold: float = 0.7,
                               include_template_data: bool = True,
                               include_experience_data: bool = True) -> List[Dict[str, Any]]:
        """
        Query a single database (without using vector index, using text similarity)
        """
        # Execute query
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        
        # Check if the database has validation_failed field
        cur.execute("PRAGMA table_info(unified_knowledge)")
        columns = [row[1] for row in cur.fetchall()]
        has_validation_failed = 'validation_failed' in columns
        
        # Build SQL query
        where_conditions = []
        params = []
        
        # If there is validation_failed field, filter out failed items
        if has_validation_failed:
            where_conditions.append("(validation_failed IS NULL OR validation_failed = 0)")
        
        # Query type filter
        if query_type:
            where_conditions.append("query_type = ?")
            params.append(query_type)
        
        # If there is no condition, add a always true condition
        if not where_conditions:
            where_conditions.append("1=1")
        
        where_clause = " AND ".join(where_conditions)
        
        cur.execute(f"""
            SELECT * FROM unified_knowledge 
            WHERE {where_clause}
            ORDER BY updated_at DESC
            LIMIT ?
        """, params + [k * 5])  # Get more candidates for similarity calculation
        
        rows = cur.fetchall()
        conn.close()
        
        # Use text similarity calculation (word overlap + containment)
        query_words = set(query_text.lower().split())
        
        results = []
        for row in rows:
            row_text = row['query_text'].lower()
            row_words = set(row_text.split())
            
            # Calculate similarity (combined Jaccard and containment)
            if query_words and row_words:
                intersection = len(query_words & row_words)
                union = len(query_words | row_words)
                jaccard = intersection / union if union > 0 else 0.0
                
                # Containment: how many proportion of query words appear in the result
                containment = intersection / len(query_words) if len(query_words) > 0 else 0.0
                
                # Comprehensive similarity: take the higher one, because we want to find related questions
                similarity = max(jaccard, containment * 0.8)  # Containment weight slightly lower
            else:
                similarity = 0.0
            
            # For cross-database search, lower the threshold to get more candidates
            effective_threshold = sim_threshold * 0.5 if sim_threshold > 0.5 else sim_threshold
            
            if similarity < effective_threshold:
                continue
            
            result = {
                'key_hash': row['key_hash'],
                'query_text': row['query_text'],
                'query_type': row['query_type'],
                'similarity': similarity,
                'hit_count': row['hit_count'],
                'success_rate': row['success_rate'],
                'source_db': db_path  # Mark source database
            }
            
            # Add experience pool data
            if include_experience_data:
                try:
                    result.update({
                        'entities': json.loads(row['entities_json'] or '[]'),
                        'time_constraint': json.loads(row['time_constraint_json'] or '{}'),
                        'indicators': json.loads(row['indicators_json'] or '{}'),
                        'evidence': json.loads(row['evidence_json'] or '{}'),
                        'toolkit_params': json.loads(row['toolkit_params_json'] or '{}')
                    })
                except (ValueError, json.JSONDecodeError) as e:
                    result.update({
                        'entities': [],
                        'time_constraint': {},
                        'indicators': {},
                        'evidence': {},
                        'toolkit_params': {}
                    })
            
            # Add template learning data
            if include_template_data:
                try:
                    result.update({
                        'question_type': row['question_type'],
                        'template_data': json.loads(row['template_data_json'] or '{}'),
                        'decomposition_data': json.loads(row['decomposition_data_json'] or '{}'),
                        'execution_data': json.loads(row['execution_data_json'] or '{}'),
                        'sufficiency_args': json.loads(row['sufficiency_args_json'] or '{}')
                    })
                except (ValueError, json.JSONDecodeError) as e:
                    result.update({
                        'question_type': row['question_type'],
                        'template_data': {},
                        'decomposition_data': {},
                        'execution_data': {},
                        'sufficiency_args': {}
                    })
            
            results.append(result)
        
        return results
    
    def store_knowledge(self, 
                       query_text: str,
                       query_type: str,
                       entities: List[str] = None,
                       time_constraint: Dict[str, Any] = None,
                       indicators: Dict[str, Any] = None,
                       evidence: Dict[str, Any] = None,
                       toolkit_params: Dict[str, Any] = None,
                       question_type: str = None,
                       template_data: Dict[str, Any] = None,
                       decomposition_data: Dict[str, Any] = None,
                       execution_data: Dict[str, Any] = None,
                       sufficiency_args: Dict[str, Any] = None,
                       success_rate: float = 0.0) -> str:
        """
        Store knowledge entry
        
        Returns:
            Generated key hash
        """
        # Generate key and normalize text
        key_hash = self._generate_key_hash(query_text, query_type, entities, time_constraint)
        norm_text = self._normalize_text(query_text, entities, time_constraint)
        
        # Prepare data
        now = datetime.utcnow().isoformat()
        
        # Check if it already exists
        conn = sqlite3.connect(self.db_path)
        print("DB_PATH: ", self.db_path)
        cur = conn.cursor()
        
        cur.execute("SELECT key_hash FROM unified_knowledge WHERE key_hash = ?", (key_hash,))
        exists = cur.fetchone() is not None
        
        if exists:
            # Update existing entry
            cur.execute("""
            UPDATE unified_knowledge SET
              query_text = ?, norm_text = ?,
              entities_json = ?, time_constraint_json = ?, indicators_json = ?,
              evidence_json = ?, toolkit_params_json = ?,
              question_type = ?, template_data_json = ?, decomposition_data_json = ?,
              execution_data_json = ?, sufficiency_args_json = ?, success_rate = ?, updated_at = ?
            WHERE key_hash = ?
            """, (
                query_text, norm_text,
                json.dumps(entities or []), json.dumps(time_constraint or {}),
                json.dumps(indicators or {}), json.dumps(evidence or {}),
                json.dumps(toolkit_params or {}),
                question_type, json.dumps(template_data or {}),
                json.dumps(decomposition_data or {}), json.dumps(execution_data or {}),
                json.dumps(sufficiency_args or {}), success_rate, now, key_hash
            ))
        else:
            # Insert new entry
            cur.execute("""
            INSERT INTO unified_knowledge (
              key_hash, query_text, query_type, norm_text,
              entities_json, time_constraint_json, indicators_json,
              evidence_json, toolkit_params_json,
              question_type, template_data_json, decomposition_data_json,
              execution_data_json, sufficiency_args_json, success_rate,
              hit_count, access_frequency, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                key_hash, query_text, query_type, norm_text,
                json.dumps(entities or []), json.dumps(time_constraint or {}),
                json.dumps(indicators or {}), json.dumps(evidence or {}),
                json.dumps(toolkit_params or {}),
                question_type, json.dumps(template_data or {}),
                json.dumps(decomposition_data or {}), json.dumps(execution_data or {}),
                json.dumps(sufficiency_args or {}), success_rate, 0, 0, now, now
            ))
        
        conn.commit()
        conn.close()
        
        # Update vector index (only for new records)
        if not exists and self._use_vector_index:
            try:
                vector_index = self._get_vector_index()
                if vector_index:
                    vector_index.add_record(key_hash, query_text)
            except Exception as e:
                print(f"⚠️  Vector index update failed: {e}")
        
        # Update cache
        entry = KnowledgeEntry(
            key_hash=key_hash,
            query_text=query_text,
            query_type=query_type,
            norm_text=norm_text,
            entities_json=json.dumps(entities or []),
            time_constraint_json=json.dumps(time_constraint or {}),
            indicators_json=json.dumps(indicators or {}),
            evidence_json=json.dumps(evidence or {}),
            toolkit_params_json=json.dumps(toolkit_params or {}),
            question_type=question_type,
            template_data_json=json.dumps(template_data or {}),
            decomposition_data_json=json.dumps(decomposition_data or {}),
            execution_data_json=json.dumps(execution_data or {}),
            success_rate=success_rate,
            created_at=now,
            updated_at=now
        )
        
        self.cache.put(key_hash, entry)
        
        return key_hash
    
    def lookup_knowledge(self, 
                        query_text: str,
                        query_type: str = None,
                        entities: List[str] = None,
                        time_constraint: Dict[str, Any] = None,
                        k: int = 5,
                        sim_threshold: float = 0.7,
                        include_template_data: bool = True,
                        include_experience_data: bool = True,
                        search_all_individual: bool = False) -> List[Dict[str, Any]]:
        """
        Query knowledge entry
        
        Args:
            query_text: Query text
            query_type: Query type filter ('main_question', 'subquestion', None means all)
            entities: Entity list
            time_constraint: Time constraint
            k: Return result number
            sim_threshold: Similarity threshold
            include_template_data: Whether to include template learning data
            include_experience_data: Whether to include experience pool data
            search_all_individual: Whether to search all individual databases (only valid in shared mode)
        
        Returns:
            Matched knowledge entry list
        """
        # If in shared mode and require to search all individual databases
        if search_all_individual and self.storage_manager.storage_mode == StorageMode.SHARED:
            return self._search_across_individual_databases(
                query_text, query_type, entities, time_constraint, 
                k, sim_threshold, include_template_data, include_experience_data
            )
        
        # Try to use vector index for fast retrieval
        vector_index = self._get_vector_index()
        
        if vector_index and not entities:  # Vector index does not support entity filtering, if there are entities, use traditional method
            # Use vector index to retrieve top-K candidates
            top_candidates = vector_index.search(
                query_text=query_text,
                top_k=k * 2,  # Get more candidates, then filter by threshold
                query_type=query_type
            )
            
            if not top_candidates:
                return []
            
            # Get full record from database
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            
            # Batch query (filter out failed items)
            key_hashes = [kh for kh, _ in top_candidates]
            placeholders = ','.join('?' * len(key_hashes))
            cur.execute(f"""
                SELECT * FROM unified_knowledge 
                WHERE key_hash IN ({placeholders})
                AND (validation_failed IS NULL OR validation_failed = 0)
            """, key_hashes)
            
            rows = cur.fetchall()
            conn.close()
            
            # Build mapping from key_hash to row
            row_map = {row['key_hash']: row for row in rows}
            
            # Build results by similarity order
            results = []
            for key_hash, similarity in top_candidates:
                if similarity < sim_threshold:
                    continue
                
                row = row_map.get(key_hash)
                if not row:
                    continue
                # Build result entry
                result = {
                    'key_hash': row['key_hash'],
                    'query_text': row['query_text'],
                    'query_type': row['query_type'],
                    'similarity': similarity,
                    'hit_count': row['hit_count'],
                    'success_rate': row['success_rate']
                }
                # Add experience pool data
                if include_experience_data:
                    import json
                    try:
                        result.update({
                            'entities': json.loads(row['entities_json'] or '[]'),
                            'time_constraint': json.loads(row['time_constraint_json'] or '{}'),
                            'indicators': json.loads(row['indicators_json'] or '{}'),
                            'evidence': json.loads(row['evidence_json'] or '{}'),
                            'toolkit_params': json.loads(row['toolkit_params_json'] or '{}')
                        })
                    except (ValueError, json.JSONDecodeError) as e:
                        print(f"Warning: Failed to parse experience data for {row['key_hash']}: {e}")
                        result.update({
                            'entities': [],
                            'time_constraint': {},
                            'indicators': {},
                            'evidence': {},
                            'toolkit_params': {}
                        })
                    # Add template learning data
                if include_template_data:
                    import json
                    try:
                        result.update({
                            'question_type': row['question_type'],
                            'template_data': json.loads(row['template_data_json'] or '{}'),
                            'decomposition_data': json.loads(row['decomposition_data_json'] or '{}'),
                            'execution_data': json.loads(row['execution_data_json'] or '{}'),
                            'sufficiency_args': json.loads(row['sufficiency_args_json'] or '{}')
                        })
                    except (ValueError, json.JSONDecodeError) as e:
                        print(f"Warning: Failed to parse template data for {row['key_hash']}: {e}")
                        result.update({
                            'question_type': row['question_type'],
                            'template_data': {},
                            'decomposition_data': {},
                            'execution_data': {},
                            'sufficiency_args': {}
                        })
                results.append(result)
            
            # Sort by similarity and limit result number
            results.sort(key=lambda x: x['similarity'], reverse=True)
            results = results[:k]
            
            # Update hit statistics
            for result in results:
                self._update_hit_stats(result['key_hash'])
            
            return results
        
        # Fallback: use traditional SQL query method (when vector index is not available or there is entity filtering)
        norm_text = self._normalize_text(query_text, entities, time_constraint)
        query_embedding = self._embed_text(query_text)
        
        # Build SQL query
        where_conditions = []
        params = []
        
        # Filter out failed items
        where_conditions.append("(validation_failed IS NULL OR validation_failed = 0)")
        
        # Basic conditions
        where_conditions.append("(query_text LIKE ? OR norm_text LIKE ?)")
        params.extend([f"%{query_text}%", f"%{norm_text}%"])
        
        # Query type filter
        if query_type:
            where_conditions.append("query_type = ?")
            params.append(query_type)
        
        # Entity filtering - change to more宽松的匹配
        if entities:
            # Create a LIKE condition for each entity, using OR to connect
            entity_conditions = []
            for entity in entities:
                entity_conditions.append("entities_json LIKE ?")
                params.append(f"%{entity}%")
            if entity_conditions:
                where_conditions.append(f"({' OR '.join(entity_conditions)})")
        
        where_clause = " AND ".join(where_conditions)
        
        # Execute query
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        
        cur.execute(f"""
        SELECT * FROM unified_knowledge 
        WHERE {where_clause}
        ORDER BY hit_count DESC, access_frequency DESC
        LIMIT ?
        """, params + [k * 3])  # Get more candidates for similarity calculation
        
        candidates = cur.fetchall()
        conn.close()
        
        # Calculate similarity and filter
        results = []
        for idx, row in enumerate(candidates):
            # Calculate text similarity
            candidate_embedding = self._embed_text(row['query_text'])
            similarity = self._cosine_similarity(query_embedding, candidate_embedding)
            if similarity >= sim_threshold:
                # Build result entry
                result = {
                    'key_hash': row['key_hash'],
                    'query_text': row['query_text'],
                    'query_type': row['query_type'],
                    'similarity': similarity,
                    'hit_count': row['hit_count'],
                    'success_rate': row['success_rate']
                }
                # Add experience pool data
                if include_experience_data:
                    import json
                    try:
                        result.update({
                            'entities': json.loads(row['entities_json'] or '[]'),
                            'time_constraint': json.loads(row['time_constraint_json'] or '{}'),
                            'indicators': json.loads(row['indicators_json'] or '{}'),
                            'evidence': json.loads(row['evidence_json'] or '{}'),
                            'toolkit_params': json.loads(row['toolkit_params_json'] or '{}')
                        })
                    except (ValueError, json.JSONDecodeError) as e:
                        print(f"Warning: Failed to parse experience data for {row['key_hash']}: {e}")
                        result.update({
                            'entities': [],
                            'time_constraint': {},
                            'indicators': {},
                            'evidence': {},
                            'toolkit_params': {}
                        })
                # Add template learning data
                if include_template_data:
                    import json
                    try:
                        result.update({
                            'question_type': row['question_type'],
                            'template_data': json.loads(row['template_data_json'] or '{}'),
                            'decomposition_data': json.loads(row['decomposition_data_json'] or '{}'),
                            'execution_data': json.loads(row['execution_data_json'] or '{}'),
                            'sufficiency_args': json.loads(row['sufficiency_args_json'] or '{}')
                        })
                    except (ValueError, json.JSONDecodeError) as e:
                        print(f"Warning: Failed to parse template data for {row['key_hash']}: {e}")
                        result.update({
                            'question_type': row['question_type'],
                            'template_data': {},
                            'decomposition_data': {},
                            'execution_data': {},
                            'sufficiency_args': {}
                        })
                results.append(result)
        
        # Sort by similarity and limit result number
        results.sort(key=lambda x: x['similarity'], reverse=True)
        results = results[:k]
        
        # Update hit statistics
        for result in results:
            self._update_hit_stats(result['key_hash'])
        
        return results
    
    def _update_hit_stats(self, key_hash: str):
        """Update hit statistics"""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        
        now = datetime.utcnow().isoformat()
        cur.execute("""
        UPDATE unified_knowledge SET
          hit_count = hit_count + 1,
          access_frequency = access_frequency + 1,
          last_hit_at = ?,
          last_access = ?
        WHERE key_hash = ?
        """, (now, now, key_hash))
        
        conn.commit()
        conn.close()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics"""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        
        # Overall statistics
        cur.execute("SELECT COUNT(*) FROM unified_knowledge")
        total_entries = cur.fetchone()[0]
        
        # Query type statistics
        cur.execute("SELECT query_type, COUNT(*) FROM unified_knowledge GROUP BY query_type")
        type_stats = dict(cur.fetchall())
        
        # Question type statistics
        cur.execute("SELECT question_type, COUNT(*) FROM unified_knowledge WHERE question_type IS NOT NULL GROUP BY question_type")
        question_type_stats = dict(cur.fetchall())
        
        # Hit statistics
        cur.execute("SELECT SUM(hit_count), AVG(hit_count), MAX(hit_count) FROM unified_knowledge")
        hit_stats = cur.fetchone()
        
        conn.close()
        
        return {
            'total_entries': total_entries,
            'type_distribution': type_stats,
            'question_type_distribution': question_type_stats,
            'hit_statistics': {
                'total_hits': hit_stats[0] or 0,
                'average_hits': hit_stats[1] or 0,
                'max_hits': hit_stats[2] or 0
            },
            'cache_size': len(self.cache.cache),
            'buffer_size': len(self.buffer)
        }
    
    def cleanup_old_entries(self, days: int = 30, min_hits: int = 2):
        """Clean old, low-hit entries"""
        cutoff_date = (datetime.utcnow().timestamp() - days * 24 * 3600)
        cutoff_iso = datetime.fromtimestamp(cutoff_date).isoformat()
        
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        
        cur.execute("""
        DELETE FROM unified_knowledge 
        WHERE updated_at < ? AND hit_count < ?
        """, (cutoff_iso, min_hits))
        
        deleted_count = cur.rowcount
        conn.commit()
        conn.close()
        
        return deleted_count
    
    def mark_failed_knowledge(self, key_hash: str):
        """
        Mark failed knowledge entries
        
        Args:
            key_hash: Hash key of knowledge entry
        """
        try:
            # Remove from cache (so next query will not get from cache)
            if key_hash in self.cache.cache:
                self.cache.remove(key_hash)
            
            # Remove from buffer
            if key_hash in self.buffer:
                del self.buffer[key_hash]
            
            # Mark as failed in database
            conn = sqlite3.connect(self.db_path)
            cur = conn.cursor()
            
            cur.execute("""
            UPDATE unified_knowledge 
            SET validation_failed = 1, updated_at = ?
            WHERE key_hash = ?
            """, (datetime.utcnow().isoformat(), key_hash))
            
            marked = cur.rowcount > 0
            conn.commit()
            conn.close()
            
            if marked:
                print(f"⚠️ Marked failed knowledge entry: {key_hash[:16]}...")
            
            return marked
            
        except Exception as e:
            print(f"❌ Error marking failed knowledge entry: {e}")
            return False

# Global instance management
_unified_store = None
_current_setting = None
_current_db_path = None

def get_unified_knowledge_store(db_path: str = None, experiment_setting: Optional[ExperimentSetting] = None) -> UnifiedKnowledgeStore:
    """Get global unified knowledge store instance"""
    global _unified_store, _current_setting, _current_db_path
    
    # Compare experiment_setting content rather than object reference
    setting_changed = False
    if _current_setting is None and experiment_setting is not None:
        setting_changed = True
    elif _current_setting is not None and experiment_setting is None:
        setting_changed = True
    elif _current_setting is not None and experiment_setting is not None:
        # Compare setting_id rather than object reference
        current_id = _current_setting.get_setting_id()
        new_id = experiment_setting.get_setting_id()
        if current_id != new_id:
            setting_changed = True
    
    # Check if need to create new instance
    if (_unified_store is None or 
        setting_changed or
        _current_db_path != db_path):
        _unified_store = UnifiedKnowledgeStore(db_path, experiment_setting)
        _current_setting = experiment_setting
        _current_db_path = db_path
    
    return _unified_store

def get_shared_unified_knowledge_store(db_path: str = None) -> UnifiedKnowledgeStore:
    """Get shared unified knowledge store instance"""
    storage_manager = get_storage_manager()
    original_mode = storage_manager.storage_mode
    storage_manager.storage_mode = StorageMode.SHARED
    
    try:
        shared_store = UnifiedKnowledgeStore(db_path, None)
        return shared_store
    finally:
        storage_manager.storage_mode = original_mode

if __name__ == "__main__":
    # Test code
    store = UnifiedKnowledgeStore("/tmp/test_unified.db")
    
    # Store test data
    key1 = store.store_knowledge(
        query_text="When did Japan visit China?",
        query_type="subquestion",
        entities=["Japan", "China"],
        indicators={"edges": [{"subj": "Japan", "rel": "visit", "obj": "China"}]},
        evidence={"entity": "Japan", "time": "2008-10-22"},
        question_type="temporal",
        template_data={"pattern": "temporal_relation"}
    )
    
    # Query test
    results = store.lookup_knowledge(
        query_text="When did Japan visit China?",
        k=5
    )
    
    print(f"Store key: {key1}")
    print(f"Query result: {len(results)} entries")
    print(f"Statistics: {store.get_stats()}")
