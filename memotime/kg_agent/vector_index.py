#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vector Index for Unified Knowledge Store
Vector index based on BERT embedding, for fast similarity retrieval
"""

import os
import pickle
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from sentence_transformers import SentenceTransformer
import sqlite3


class VectorIndex:
    """
    Vector index system
    - Use BERT embedding to convert query text to vector
    - Build offline index to accelerate similarity retrieval
    - Support incremental update and persistence
    """
    
    def __init__(self, db_path: str, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        """
        Initialize vector index
        
        Args:
            db_path: Database path
            model_name: BERT model name
        """
        self.db_path = db_path
        self.model_name = model_name
        
        # Index file path (same directory as database)
        db_dir = os.path.dirname(db_path)
        self.index_path = os.path.join(db_dir, "vector_index.pkl")
        
        # Set CUDA device (use GPU 7 by default)
        import torch
        if torch.cuda.is_available():
            # If environment variable is not set, use GPU 7 by default
            if 'CUDA_VISIBLE_DEVICES' not in os.environ:
                os.environ['CUDA_VISIBLE_DEVICES'] = '7'
            device = 'cuda'
            print(f"🔧 Use GPU: {os.environ.get('CUDA_VISIBLE_DEVICES', 'default')}")
        else:
            device = 'cpu'
            print(f"🔧 Use CPU")
        
        # Load BERT model
        print(f"🔧 Load BERT model: {model_name}")
        self.model = SentenceTransformer(model_name, device=device)
        
        # Vector index data
        self.key_hashes: List[str] = []  # key_hash list
        self.embeddings: Optional[np.ndarray] = None  # embedding matrix (N, D)
        self.last_update_time: Optional[str] = None
        
        # Try to load existing index
        self._load_index()
    
    def _load_index(self):
        """Load index from disk"""
        if os.path.exists(self.index_path):
            try:
                with open(self.index_path, 'rb') as f:
                    data = pickle.load(f)
                    self.key_hashes = data['key_hashes']
                    self.embeddings = data['embeddings']
                    self.last_update_time = data.get('last_update_time')
                print(f"✅ Load vector index: {len(self.key_hashes)} entries")
            except Exception as e:
                print(f"⚠️ Load index failed: {e}, will rebuild")
                self.key_hashes = []
                self.embeddings = None
    
    def _save_index(self):
        """Save index to disk"""
        try:
            os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
            with open(self.index_path, 'wb') as f:
                pickle.dump({
                    'key_hashes': self.key_hashes,
                    'embeddings': self.embeddings,
                    'last_update_time': self.last_update_time,
                    'model_name': self.model_name
                }, f)
            print(f"💾 Save vector index: {len(self.key_hashes)} entries")
        except Exception as e:
            print(f"⚠️ Save index failed: {e}")
    
    def build_index(self, force_rebuild: bool = False):
        """
        Build or update vector index
        
        Args:
            force_rebuild: Whether to force rebuild the entire index
        """
        # Check if database has new data
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        
        # Get the latest update time from database
        cur.execute("SELECT MAX(updated_at) FROM unified_knowledge")
        db_last_update = cur.fetchone()[0]
        
        # Check if need to update
        if not force_rebuild and self.embeddings is not None:
            if db_last_update and self.last_update_time:
                if db_last_update <= self.last_update_time:
                    print(f"✅ Vector index is up to date, no need to update")
                    conn.close()
                    return
        
        print(f"🔨 Build vector index...")
        
        # Get all records
        cur.execute("""
            SELECT key_hash, query_text, query_type 
            FROM unified_knowledge 
            ORDER BY created_at
        """)
        rows = cur.fetchall()
        conn.close()
        
        if not rows:
            print(f"⚠️ Database is empty, cannot build index")
            return
        
        # Extract data
        key_hashes = [row[0] for row in rows]
        query_texts = [row[1] for row in rows]
        
        # Batch generate embeddings
        print(f"📊 Generating embeddings for {len(query_texts)} records...")
        embeddings = self.model.encode(
            query_texts, 
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # Update index
        self.key_hashes = key_hashes
        self.embeddings = embeddings
        self.last_update_time = db_last_update
        
        # Save to disk
        self._save_index()
        
        print(f"✅ Vector index built: {len(self.key_hashes)} records")
    
    def search(self, query_text: str, top_k: int = 10, query_type: Optional[str] = None) -> List[Tuple[str, float]]:
        """
        Vector similarity search
        
        Args:
            query_text: Query text
            top_k: Return top-K most similar results
            query_type: Optional query type filter
        
        Returns:
            [(key_hash, similarity_score), ...] sorted by similarity score
        """
        if self.embeddings is None or len(self.key_hashes) == 0:
            print(f"⚠️ Vector index is empty, trying to build...")
            self.build_index()
            if self.embeddings is None:
                return []
        
        # Generate query vector
        query_embedding = self.model.encode([query_text], convert_to_numpy=True)[0]
        
        # If query_type is specified, need to filter first
        if query_type:
            # Get key_hash from database
            conn = sqlite3.connect(self.db_path)
            cur = conn.cursor()
            cur.execute("SELECT key_hash FROM unified_knowledge WHERE query_type = ?", (query_type,))
            valid_key_hashes = set(row[0] for row in cur.fetchall())
            conn.close()
            
            # Find the position of these key_hash in the index
            valid_indices = [i for i, kh in enumerate(self.key_hashes) if kh in valid_key_hashes]
            
            if not valid_indices:
                return []
            
            # Only calculate the similarity of these records
            valid_embeddings = self.embeddings[valid_indices]
            similarities = self._cosine_similarity_batch(query_embedding, valid_embeddings)
            
            # Get top-K
            top_k_local_indices = np.argsort(similarities)[::-1][:top_k]
            top_k_indices = [valid_indices[i] for i in top_k_local_indices]
            top_k_similarities = similarities[top_k_local_indices]
        else:
            # Calculate the similarity of all records
            similarities = self._cosine_similarity_batch(query_embedding, self.embeddings)
            
            # Get top-K
            top_k_indices = np.argsort(similarities)[::-1][:top_k]
            top_k_similarities = similarities[top_k_indices]
        
        # Build results
        results = [
            (self.key_hashes[idx], float(sim)) 
            for idx, sim in zip(top_k_indices, top_k_similarities)
        ]
        
        return results
    
    def _cosine_similarity_batch(self, query_vec: np.ndarray, candidate_vecs: np.ndarray) -> np.ndarray:
        """
        Batch calculate cosine similarity
        
        Args:
            query_vec: Query vector (D,)
            candidate_vecs: Candidate vector matrix (N, D)
        
        Returns:
            Similarity array (N,)
        """
        # Normalize
        query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
        candidate_norms = candidate_vecs / (np.linalg.norm(candidate_vecs, axis=1, keepdims=True) + 1e-10)
        
        # Dot product
        similarities = np.dot(candidate_norms, query_norm)
        
        return similarities
    
    def add_record(self, key_hash: str, query_text: str):
        """
        Incrementally add a single record to the index
        
        Args:
            key_hash: Record key_hash
            query_text: Query text
        """
        # Generate embedding
        embedding = self.model.encode([query_text], convert_to_numpy=True)[0]
        
        # Add to index
        if self.embeddings is None:
            self.embeddings = embedding.reshape(1, -1)
            self.key_hashes = [key_hash]
        else:
            self.embeddings = np.vstack([self.embeddings, embedding.reshape(1, -1)])
            self.key_hashes.append(key_hash)
        
        # Save periodically (every 100 records)
        if len(self.key_hashes) % 100 == 0:
            self._save_index()
    
    def remove_record(self, key_hash: str):
        """
        Remove record from index
        
        Args:
            key_hash: Record key_hash
        """
        if key_hash in self.key_hashes:
            idx = self.key_hashes.index(key_hash)
            self.key_hashes.pop(idx)
            if self.embeddings is not None:
                self.embeddings = np.delete(self.embeddings, idx, axis=0)
            self._save_index()


# Global index instance management
_vector_index_instances: Dict[str, VectorIndex] = {}


def get_vector_index(db_path: str) -> VectorIndex:
    """
    Get vector index instance (singleton pattern)
    
    Args:
        db_path: Database path
    
    Returns:
        VectorIndex instance
    """
    if db_path not in _vector_index_instances:
        _vector_index_instances[db_path] = VectorIndex(db_path)
    return _vector_index_instances[db_path]

