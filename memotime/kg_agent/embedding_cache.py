"""
# =============================
# file: kg_agent/embedding_cache.py
# =============================
relation embedding precomputation and caching system
"""

import os
import json
import pickle
import numpy as np
from typing import Dict, List, Optional, Tuple
import sqlite3
from datetime import datetime

# OpenAI embedding
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: openai not available. Using fallback embedding.")

class RelationEmbeddingCache:
    """relation embedding cache system"""
    
    def __init__(self, db_path: str, cache_dir: str = "embedding_cache"):
        self.db_path = db_path
        self.cache_dir = cache_dir
        self.embeddings_file = os.path.join(cache_dir, "relation_embeddings.pkl")
        self.metadata_file = os.path.join(cache_dir, "embedding_metadata.json")
        self.name_mapping_file = os.path.join(cache_dir, "relation_name_mapping.json")
        
        # ensure cache directory exists
        os.makedirs(cache_dir, exist_ok=True)
        
        # initialize OpenAI client
        self.openai_client = None
        if OPENAI_AVAILABLE:
            try:
                from kg_agent.llm import DEFAULT_OPENAI_API_KEY
                # use environment variable or directly set API key
                api_key = os.getenv('OPENAI_API_KEY', DEFAULT_OPENAI_API_KEY)
                self.openai_client = openai.OpenAI(api_key=api_key)
                print("✅ OpenAI client initialized successfully")
            except Exception as e:
                print(f"❌ OpenAI client initialization failed: {e}")
                self.openai_client = None
        
        # load cached embeddings
        self.embeddings = self._load_embeddings()
        self.metadata = self._load_metadata()
        # load normalized name to original name mapping
        self.normalized_to_original = self._load_name_mapping()
    
    def _load_embeddings(self) -> Dict[str, np.ndarray]:
        """load cached embeddings"""
        if os.path.exists(self.embeddings_file):
            try:
                with open(self.embeddings_file, 'rb') as f:
                    embeddings = pickle.load(f)
                print(f"✅ loaded {len(embeddings)} cached relation embeddings")
                return embeddings
            except Exception as e:
                print(f"⚠️ load embedding cache failed: {e}")
        return {}
    
    def _load_metadata(self) -> Dict[str, any]:
        """load cached metadata"""
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                return metadata
            except Exception as e:
                print(f"⚠️ load metadata failed: {e}")
        return {}
    
    def _load_name_mapping(self) -> Dict[str, str]:
        """load normalized name to original name mapping"""
        if os.path.exists(self.name_mapping_file):
            try:
                with open(self.name_mapping_file, 'r', encoding='utf-8') as f:
                    mapping = json.load(f)
                print(f"✅ loaded {len(mapping)} relation name mappings")
                return mapping
            except Exception as e:
                print(f"⚠️ load name mapping failed: {e}")
        return {}
    
    def _save_name_mapping(self, mapping: Dict[str, str]):
        """save normalized name to original name mapping"""
        try:
            with open(self.name_mapping_file, 'w', encoding='utf-8') as f:
                json.dump(mapping, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"❌ save name mapping failed: {e}")
    
    def _save_embeddings(self, embeddings: Dict[str, np.ndarray]):
        """save embeddings to cache"""
        try:
            with open(self.embeddings_file, 'wb') as f:
                pickle.dump(embeddings, f)
            print(f"✅ saved {len(embeddings)} relation embeddings to cache")
        except Exception as e:
            print(f"❌ save embedding cache failed: {e}")
    
    def _save_metadata(self, metadata: Dict[str, any]):
        """save metadata to cache"""
        try:
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"❌ save metadata failed: {e}")
    
    def get_openai_embedding(self, text: str) -> Optional[np.ndarray]:
        """use OpenAI to get embedding"""
        if not self.openai_client:
            return None
        
        try:
            response = self.openai_client.embeddings.create(
                model="text-embedding-3-small",  # use the latest embedding model
                input=text
            )
            embedding = np.array(response.data[0].embedding, dtype=np.float32)
            return embedding
        except Exception as e:
            print(f"❌ OpenAI embedding failed: {e}")
            return None
    
    def normalize_relation_text(self, relation: str) -> str:
        """normalize relation text"""
        # 将下划线替换为空格，转换为小写
        normalized = relation.replace("_", " ").lower().strip()
        return normalized
    
    def get_all_relations_from_db(self) -> List[str]:
        """get all unique relations from database"""
        relations = set()
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # query all relations
            cursor.execute("SELECT DISTINCT relation FROM edges WHERE relation IS NOT NULL AND relation != ''")
            results = cursor.fetchall()
            
            for (relation,) in results:
                if relation and relation.strip():
                    relations.add(relation.strip())
            
            conn.close()
            print(f"✅ got {len(relations)} unique relations from database")
            return list(relations)
            
        except Exception as e:
            print(f"❌ get relations from database failed: {e}")
            return []
    
    def precompute_all_embeddings(self, force_recompute: bool = False):
        """precompute all relations embeddings"""
        print("🚀 start precomputing relation embeddings...")
        
        # check if need to recompute
        if not force_recompute and self.embeddings and self.metadata:
            last_update = self.metadata.get('last_update')
            if last_update:
                print(f"✅ found existing cache, last updated: {last_update}")
                return
        
        # get all relations
        all_relations = self.get_all_relations_from_db()
        if not all_relations:
            print("❌ no relations found, cannot precompute")
            return
        
        # check OpenAI availability
        if not self.openai_client:
            print("❌ OpenAI client unavailable, cannot precompute")
            return
        
        # precompute embedding and name mapping
        new_embeddings = {}
        new_name_mapping = {}
        processed_count = 0
        
        for relation in all_relations:
            normalized_text = self.normalize_relation_text(relation)
            
            # save normalized name to original name mapping
            new_name_mapping[normalized_text] = relation
            
            # if already cached and not forced to recompute, skip
            if not force_recompute and normalized_text in self.embeddings:
                new_embeddings[normalized_text] = self.embeddings[normalized_text]
                continue
            
            # compute embedding
            embedding = self.get_openai_embedding(normalized_text)
            if embedding is not None:
                new_embeddings[normalized_text] = embedding
                processed_count += 1
                
                if processed_count % 100 == 0:
                    print(f"📊 processed {processed_count}/{len(all_relations)} relations")
        
        # save results
        self.embeddings = new_embeddings
        self.normalized_to_original = new_name_mapping
        self._save_embeddings(new_embeddings)
        self._save_name_mapping(new_name_mapping)
        
        # update metadata
        metadata = {
            'last_update': datetime.now().isoformat(),
            'total_relations': len(all_relations),
            'processed_relations': processed_count,
            'model': 'text-embedding-3-small',
            'normalization': 'underscore_to_space_lowercase'
        }
        self.metadata = metadata
        self._save_metadata(metadata)
        
        print(f"✅ precomputation completed! processed {processed_count} relations")
    
    def get_relation_embedding(self, relation: str) -> Optional[np.ndarray]:
        """get relation embedding"""
        normalized_text = self.normalize_relation_text(relation)
        return self.embeddings.get(normalized_text)
    
    def calculate_similarity(self, relation1: str, relation2: str) -> float:
        """calculate similarity between two relations"""
        emb1 = self.get_relation_embedding(relation1)
        emb2 = self.get_relation_embedding(relation2)
        
        if emb1 is None or emb2 is None:
            return 0.0
        
        # calculate cosine similarity
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = float(np.dot(emb1, emb2) / (norm1 * norm2))
        return similarity
    
    def find_most_similar_relations(self, target_relation: str, relations: List[str], top_k: int = 5) -> List[Tuple[str, float]]:
        """find most similar relations to target relation"""
        target_emb = self.get_relation_embedding(target_relation)
        if target_emb is None:
            return []
        
        similarities = []
        for relation in relations:
            similarity = self.calculate_similarity(target_relation, relation)
            similarities.append((relation, similarity))
        
        # sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def get_cache_info(self) -> Dict[str, any]:
        """get cache information"""
        # Get total relations from database
        total_relations = 0
        try:
            conn = sqlite3.connect(self.db_path)
            cur = conn.cursor()
            cur.execute("SELECT COUNT(DISTINCT relation) FROM edges")
            total_relations = cur.fetchone()[0]
            conn.close()
        except:
            pass
        
        return {
            'cached_relations': len(self.embeddings),
            'total_relations': total_relations,
            'metadata': self.metadata,
            'cache_files_exist': {
                'embeddings': os.path.exists(self.embeddings_file),
                'metadata': os.path.exists(self.metadata_file)
            }
        }
