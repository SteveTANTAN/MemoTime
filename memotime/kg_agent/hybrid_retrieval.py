#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hybrid retrieval system - semantic retrieval based on BGE-M3
integrated into TPKG system, provide retrieval interface from question to triples
"""

import os
import json
import time
import asyncio
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
import torch
from concurrent.futures import ThreadPoolExecutor

# optional dependencies
try:
    import faiss
    import faiss.contrib.torch_utils
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("Warning: FAISS not available. Hybrid retrieval will be disabled.")

try:
    from FlagEmbedding import BGEM3FlagModel, FlagReranker
    BGE_AVAILABLE = True
except ImportError:
    BGE_AVAILABLE = False
    print("Warning: FlagEmbedding not available. Hybrid retrieval will be disabled.")

try:
    import spacy
    import en_core_web_sm
    nlp = en_core_web_sm.load()
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    print("Warning: spaCy not available. Date extraction will be disabled.")


def parse_date(date_str):
    """parse date string"""
    formats = [
        "%Y-%m-%d",
        "%d %B %Y", 
        "%B %Y"
    ]
    for fmt in formats:
        try:
            date_obj = datetime.strptime(date_str, fmt).date()
            return date_obj
        except ValueError:
            pass
    return None


def extract_dates(text):
    """extract dates from text"""
    if not SPACY_AVAILABLE:
        return None
        
    doc = nlp(text)
    dates = ""
    for ent in doc.ents:
        if ent.label_ == "DATE":
            dates += ent.text + " "
    dates = dates.strip()
    processed_dates = parse_date(dates)
    return processed_dates


class HybridRetrieval:
    """
    hybrid retrieval system - semantic retrieval based on BGE-M3
    provide retrieval interface from question to triples
    """
    
    def __init__(self, 
                 db_path: str,
                 model_name: str = 'BAAI/bge-m3',
                 embedding_size: int = 1024,
                 use_gpu: bool = True,
                 gpu_id: int = 0,
                 index_path: str = None,
                 raw_data_path: str = None):
        """
        initialize hybrid retrieval system
        
        Args:
            db_path: database path
            model_name: BGE model name
            embedding_size: embedding vector dimension
            use_gpu: whether to use GPU
            gpu_id: GPU device ID
            index_path: FAISS index save path
            raw_data_path: original data file path (if provided, will load data from this file)
        """
        self.db_path = db_path
        self.device = f'cuda:{gpu_id}' if torch.cuda.is_available() and use_gpu else 'cpu'
        self.model = None
        self.model_name = model_name
        self.embedding_size = embedding_size
        self.use_gpu = use_gpu
        self.gpu_id = gpu_id
        self.raw_data_path = raw_data_path
        
        # data storage
        self.triple_list = []
        self.fact_list = []
        self.time_list = []
        self.triplet_embeddings = None
        self.index = None
        
        # get cache directory (isolated by dataset, no longer use GPU ID suffix)
        from config import TPKGConfig
        cache_dir = TPKGConfig.HYBRID_CACHE_DIR or "."
        
        # index path (no longer contains GPU ID)
        self.index_path = index_path or os.path.join(cache_dir, "hybrid_index.bin")
        
        # data cache path (no longer contains GPU ID)
        self.data_cache_path = os.path.join(cache_dir, "hybrid_data_cache.pkl")
        self.embedding_cache_path = os.path.join(cache_dir, "hybrid_embeddings.npy")
        
        # initialize state
        self.is_loaded = False
        
        print(f"✅ hybrid retrieval system initialized: device={self.device}, model={model_name}")
    
    def load_triples_from_raw_file(self):
        """load triples from raw file"""
        if not self.raw_data_path or not os.path.exists(self.raw_data_path):
            print("❌ original data file does not exist")
            return False
        
        try:
            print(f"🔄 load data from raw file: {self.raw_data_path}")
            
            self.triple_list = []
            self.fact_list = []
            self.time_list = []
            
            with open(self.raw_data_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    if line_num % 100000 == 0:
                        print(f"🔄 processed {line_num} lines...")
                    
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split('\t')
                    if len(parts) != 4:
                        continue
                    
                    head, relation, tail, time_start = parts
                    
                    # build triple
                    triple = f"({head}, {relation}, {tail})"
                    fact = f"{head} {relation.replace('_', ' ')} {tail} at {time_start}"
                    
                    self.triple_list.append(triple)
                    self.fact_list.append(fact)
                    
                    # parse time
                    try:
                        time_obj = datetime.strptime(time_start, "%Y-%m-%d").date()
                        self.time_list.append(time_obj)
                    except:
                        self.time_list.append(None)
            
            print(f"✅ loaded {len(self.triple_list)} triples from raw file")
            
            # save data cache
            self.save_data_cache()
            return True
            
        except Exception as e:
            print(f"❌ failed to load triples from raw file: {e}")
            return False
    
    def save_data_cache(self):
        """save data cache"""
        try:
            import pickle
            cache_data = {
                'triple_list': self.triple_list,
                'fact_list': self.fact_list,
                'time_list': self.time_list
            }
            with open(self.data_cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
            print(f"✅ data cache saved: {self.data_cache_path}")
        except Exception as e:
            print(f"⚠️ failed to save data cache: {e}")
    
    def load_data_cache(self):
        """load data cache"""
        try:
            import pickle
            if os.path.exists(self.data_cache_path):
                with open(self.data_cache_path, 'rb') as f:
                    cache_data = pickle.load(f)
                
                self.triple_list = cache_data['triple_list']
                self.fact_list = cache_data['fact_list']
                self.time_list = cache_data['time_list']
                
                print(f"✅ loaded {len(self.triple_list)} triples from cache")
                return True
        except Exception as e:
            print(f"⚠️ failed to load data cache: {e}")
        
        return False
    
    def load_triples_from_db(self):
        """load triples from database"""
        try:
            import sqlite3
            
            # directly use sqlite3 to connect database
            conn = sqlite3.connect(self.db_path)
            
            # query all edges
            query = """
            SELECT head, relation, tail, time_start 
            FROM edges 
            ORDER BY time_start_epoch
            """
            
            cursor = conn.cursor()
            cursor.execute(query)
            results = cursor.fetchall()
            
            self.triple_list = []
            self.fact_list = []
            self.time_list = []
            
            for row in results:
                head, relation, tail, time_start = row
                
                # build triple
                triple = f"({head}, {relation}, {tail})"
                fact = f"{head} {relation.replace('_', ' ')} {tail} at {time_start}"
                
                self.triple_list.append(triple)
                self.fact_list.append(fact)
                
                # parse time
                try:
                    time_obj = datetime.strptime(time_start, "%Y-%m-%d").date()
                    self.time_list.append(time_obj)
                except:
                    self.time_list.append(None)
            
            conn.close()
            print(f"✅ loaded {len(self.triple_list)} triples from database")
            return True
            
        except Exception as e:
            print(f"❌ failed to load triples from database: {e}")
            return False
    
    async def load_model_and_index(self):
        """async load model and build index"""
        if not BGE_AVAILABLE or not FAISS_AVAILABLE:
            print("❌ missing necessary dependencies, cannot load hybrid retrieval model")
            return False
        
        try:
            print("🔄 start loading BGE-M3 model...")
            self.model = BGEM3FlagModel(
                self.model_name, 
                use_fp16=True, 
                devices=[self.device]
            )
            print("✅ BGE-M3 model loaded successfully")
            
            # 检查是否存在预构建的索引
            if os.path.exists(self.index_path):
                print(f"🔄 load prebuilt FAISS index: {self.index_path}")
                self.index = faiss.read_index(self.index_path)
                print("✅ FAISS index loaded successfully")
            else:
                print("🔄 build new FAISS index...")
                await self.build_embeddings_and_index()
                print("✅ FAISS index built successfully")
            
            self.is_loaded = True
            return True
            
        except Exception as e:
            print(f"❌ failed to load model and index: {e}")
            return False
    
    def save_embeddings_cache(self):
        """save embeddings cache"""
        try:
            if self.triplet_embeddings is not None:
                np.save(self.embedding_cache_path, self.triplet_embeddings)
                print(f"✅ embeddings cache saved: {self.embedding_cache_path}")
        except Exception as e:
            print(f"⚠️ failed to save embeddings cache: {e}")
    
    def load_embeddings_cache(self):
        """load embeddings cache"""
        try:
            if os.path.exists(self.embedding_cache_path):
                self.triplet_embeddings = np.load(self.embedding_cache_path)
                print(f"✅ loaded embeddings cache: {self.triplet_embeddings.shape}")
                return True
        except Exception as e:
            print(f"⚠️ failed to load embeddings cache: {e}")
        
        return False
    
    async def build_embeddings_and_index(self):
        """build embeddings and FAISS index"""
        if not self.fact_list:
            print("❌ no available fact data")
            return
        
        # try to load embeddings cache
        if self.load_embeddings_cache():
            print("✅ use cached embeddings")
        else:
            print(f"🔄 start encoding {len(self.fact_list)} facts...")
            
            # 编码所有事实
            self.triplet_embeddings = self.model.encode_corpus(
                self.fact_list,
                convert_to_numpy=True,
                batch_size=512,  # decrease batch size to avoid memory problem
                return_dense=True,
                return_sparse=False,  # temporarily only use dense vectors
                return_colbert_vecs=False
            )
            
            self.triplet_embeddings = self.triplet_embeddings['dense_vecs']
            self.triplet_embeddings = self.triplet_embeddings.astype(np.float32)
            
            print(f"✅ embeddings shape: {self.triplet_embeddings.shape}")
            
            # save embeddings cache
            self.save_embeddings_cache()
        
        # build FAISS index
        self.index = self.build_faiss_index()
        
        if not self.index.is_trained:
            print("🔄 train FAISS index...")
            self.index.train(self.triplet_embeddings)
        
        print("🔄 add vectors to index...")
        self.index.add(self.triplet_embeddings)
        
        # 保存索引
        print(f"🔄 save index to: {self.index_path}")
        faiss.write_index(self.index, self.index_path)
        print("✅ index saved successfully")
    
    def build_faiss_index(self, n_clusters=500, nprobe=60):
        """build FAISS index"""
        quantizer = faiss.IndexFlatIP(self.embedding_size)
        index = faiss.IndexIVFFlat(quantizer, self.embedding_size, n_clusters, faiss.METRIC_INNER_PRODUCT)
        index.nprobe = nprobe
        
        if self.use_gpu and torch.cuda.is_available():
            try:
                ngpu = 1
                resources = [faiss.StandardGpuResources() for _ in range(ngpu)]
                vres = faiss.GpuResourcesVector()
                vdev = faiss.Int32Vector()
                for i, res in zip(range(ngpu), resources):
                    vdev.push_back(self.gpu_id)
                    vres.push_back(res)
                index_gpu = faiss.index_cpu_to_gpu_multiple(vres, vdev, index)
                print(f"✅ use GPU index: cuda:{self.gpu_id}")
                return index_gpu
            except Exception as e:
                print(f"⚠️ failed to create GPU index, use CPU: {e}")
                return index
        else:
            print("✅ use CPU index")
            return index
    
    async def get_embedding(self, corpus_list):
        """get embedding for query"""
        result = await asyncio.to_thread(
            self.model.encode_queries,
            corpus_list,
            convert_to_numpy=True,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False
        )
        return result['dense_vecs']
    
    async def compute_similarity(self, question: str, n: int = 100):
        """compute similarity between question and triples"""
        if not self.is_loaded:
            print("❌ model not loaded, please call load() method")
            return None, None
        
        question_embedding = await self.get_embedding([question])
        distances, corpus_ids = self.index.search(question_embedding, n)
        return distances[0], corpus_ids[0]
    
    async def retrieve_triples(self, 
                             question: str, 
                             top_k: int = 50,
                             re_rank: bool = False,
                             time_weight: float = 0.6) -> Dict[str, Any]:
        """
        retrieve related triples based on question
        
        Args:
            question: input question
            top_k: number of triples to return
            re_rank: whether to use time re-ranking
            time_weight: time weight (for re-ranking)
            
        Returns:
            dictionary containing related triples
        """
        if not self.is_loaded:
            await self.initialize()
        
        try:
            # compute similarity
            distances, corpus_ids = await self.compute_similarity(question, top_k * 2)
            
            if distances is None:
                return {'question': question, 'triples': [], 'error': 'Similarity computation failed'}
            
            # get result
            if re_rank:
                result = await self.re_rank_result(question, distances, corpus_ids, time_weight)
            else:
                result = await self.basic_result(question, distances, corpus_ids)
            
            # limit return number
            result['triples'] = result['triples'][:top_k]
            result['facts'] = result['facts'][:top_k]
            result['scores'] = result['scores'][:top_k]
            
            return result
            
        except Exception as e:
            print(f"❌ failed to retrieve: {e}")
            return {'question': question, 'triples': [], 'error': str(e)}
    
    async def re_rank_result(self, question: str, distances, corpus_ids, time_weight: float = 0.6):
        """re-rank result based on time information"""
        target_time = extract_dates(question)
        time_scores = [10 for _ in range(len(self.time_list))]
        
        if target_time and target_time != "None":
            target_time = datetime.strptime(str(target_time), "%Y-%m-%d").date()
            self.adjust_time_scores(question, target_time, time_scores)
        
        result = {'question': question}
        hits = []
        
        for id, score in zip(corpus_ids, distances):
            if id < len(self.triple_list):
                final_score = score * (1 - time_weight) - time_scores[id] * time_weight
                hits.append({
                    'corpus_id': id,
                    'score': float(score),
                    'time_score': time_scores[id],
                    'final_score': final_score
                })
        
        # sort by final score
        hits = sorted(hits, key=lambda x: x['final_score'], reverse=True)
        
        result['scores'] = [hit['score'] for hit in hits]
        result['time_scores'] = [hit['time_score'] for hit in hits]
        result['final_scores'] = [hit['final_score'] for hit in hits]
        result['triples'] = [self.triple_list[hit['corpus_id']] for hit in hits]
        result['facts'] = [self.fact_list[hit['corpus_id']] for hit in hits]
        
        return result
    
    def adjust_time_scores(self, question: str, target_time, time_scores: List[float]):
        """adjust time scores based on time information in question"""
        for idx, t in enumerate(self.time_list):
            if t is None:
                continue
                
            time_difference = target_time - t
            days_difference = time_difference.days
            
            if 'before' in question.lower():
                if 0 < days_difference < 16:
                    time_scores[idx] = days_difference / 15
            elif 'after' in question.lower():
                if -16 < days_difference < 0:
                    time_scores[idx] = -days_difference / 15
            elif 'in' in question.lower() and days_difference == 0:
                time_scores[idx] = 0
    
    async def basic_result(self, question: str, distances, corpus_ids):
        """basic result (no re-ranking)"""
        result = {'question': question}
        hits = []
        
        for id, score in zip(corpus_ids, distances):
            if id < len(self.triple_list):
                hits.append({
                    'corpus_id': id,
                    'score': float(score)
                })
        
        # sort by score
        hits = sorted(hits, key=lambda x: x['score'], reverse=True)
        
        result['scores'] = [hit['score'] for hit in hits]
        result['triples'] = [self.triple_list[hit['corpus_id']] for hit in hits]
        result['facts'] = [self.fact_list[hit['corpus_id']] for hit in hits]
        
        return result
    
    async def initialize(self):
        """initialize hybrid retrieval system"""
        if self.is_loaded:
            return True
        
        print("🔄 initialize hybrid retrieval system...")
        
        # 1. load triples data (priority: cache > raw file > database)
        data_loaded = False
        
        # first try to load from cache
        if self.load_data_cache():
            data_loaded = True
        # if there is raw file, load from raw file
        elif self.raw_data_path:
            data_loaded = self.load_triples_from_raw_file()
        # finally try to load from database
        else:
            data_loaded = self.load_triples_from_db()
        
        if not data_loaded:
            print("❌ failed to load triples data")
            return False
        
        # 2. load model and index
        if not await self.load_model_and_index():
            return False
        
        print("✅ hybrid retrieval system initialized successfully")
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """get system statistics"""
        return {
            'is_loaded': self.is_loaded,
            'total_triples': len(self.triple_list),
            'model_name': self.model_name,
            'device': self.device,
            'embedding_size': self.embedding_size,
            'index_path': self.index_path
        }


# global instance
_hybrid_retrieval_instance = None


def get_hybrid_retrieval(db_path: str, raw_data_path: str = None, **kwargs) -> HybridRetrieval:
    """get global instance of hybrid retrieval system"""
    global _hybrid_retrieval_instance
    
    if _hybrid_retrieval_instance is None:
        _hybrid_retrieval_instance = HybridRetrieval(db_path, raw_data_path=raw_data_path, **kwargs)
    
    return _hybrid_retrieval_instance


async def hybrid_retrieve_triples(question: str, 
                                db_path: str,
                                top_k: int = 50,
                                re_rank: bool = False,
                                raw_data_path: str = None,
                                **kwargs) -> Dict[str, Any]:
    """
    hybrid retrieval interface function
    
    Args:
        question: input question
        db_path: database path
        top_k: number of triples to return
        re_rank: whether to use time re-ranking
        raw_data_path: raw data file path
        **kwargs: other parameters
        
    Returns:
        dictionary containing related triples
    """
    retrieval = get_hybrid_retrieval(db_path, raw_data_path=raw_data_path, **kwargs)
    
    if not retrieval.is_loaded:
        await retrieval.initialize()
    
    return await retrieval.retrieve_triples(question, top_k, re_rank)

