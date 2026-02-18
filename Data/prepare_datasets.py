#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import sys
import json
import sqlite3
import pickle
import asyncio
from datetime import datetime
from typing import List, Tuple, Dict, Any
from tqdm import tqdm

# Add current directory and parent directory to path for imports
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))  # Add project root for memotime module
sys.path.insert(0, str(Path(__file__).parent))
try:
    from new import cmd_build_index, semantic_candidates_by_keywords
    from entity_try import combined_keywords
except ImportError as e:
    print("Warning: Could not import helper modules, will use alternative methods...")

# Check hybrid retrieval dependencies
try:
    import numpy as np
    import torch
    import faiss
    from FlagEmbedding import BGEM3FlagModel
    HYBRID_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Hybrid retrieval dependencies missing: {e}")
    print("Will skip hybrid index build...")
    HYBRID_AVAILABLE = False

# Dataset configuration
DATASETS = {
    "TimeQuestions": {
        "root": str(Path(__file__).parent / "TimeQuestions"),
        "kg_file": "full.txt",
        "questions_file": "test.json",
        "kg_format": "5cols",  # subject, relation, object, start_time, end_time
        "question_id_field": "Id",
        "question_text_field": "Question",
        "answer_field": "Answer",
        "hybrid_cache_dir": str(Path(__file__).parent / "TimeQuestions" / "hybrid_cache"),
        "embedding_cache_dir": str(Path(__file__).parent / "TimeQuestions" / "embedding_cache_timequestions")
    },
    "MultiTQ": {
        "root": str(Path(__file__).parent / "MultiTQ"),
        "kg_file": "full_fixed.txt",
        "questions_file": "test.json",
        "kg_format": "4cols",  # subject, relation, object, time
        "question_id_field": "quid",
        "question_text_field": "question",
        "answer_field": "answers",
        "hybrid_cache_dir": str(Path(__file__).parent / "MultiTQ" / "hybrid_cache"),
        "embedding_cache_dir": str(Path(__file__).parent / "MultiTQ" / "embedding_cache_multitq")
    }
}


class DatasetPreprocessor:
    """dataset preprocessor"""
    
    def __init__(self, dataset_name: str):
        """Initializepreprocess"""
        if dataset_name not in DATASETS:
            raise ValueError(f"Unknown Dataset: {dataset_name}. Supported Datasets: {list(DATASETS.keys())}")
        
        self.dataset_name = dataset_name
        self.config = DATASETS[dataset_name]
        
        # path settings
        self.root = self.config["root"]
        self.kg_file = os.path.join(self.root, self.config["kg_file"])
        self.questions_file = os.path.join(self.root, self.config["questions_file"])
        self.db_path = os.path.join(self.root, f"tempkg_{dataset_name.lower()}.db")
        self.index_dir = os.path.join(self.root, f"entity_index_{dataset_name.lower()}")
        self.output_file = os.path.join(self.root, f"questions_with_candidates_{dataset_name.lower()}.json")
        self.hybrid_cache_dir = self.config.get("hybrid_cache_dir", 
                                                 os.path.join(self.root, "hybrid_cache"))
        self.embedding_cache_dir = self.config.get("embedding_cache_dir",
                                                     os.path.join(self.root, f"embedding_cache_{dataset_name.lower()}"))
        
        # Create cache directories
        os.makedirs(self.hybrid_cache_dir, exist_ok=True)
        os.makedirs(self.embedding_cache_dir, exist_ok=True)
        
        print(f"\n{'='*80}")
        print(f"Initializing {dataset_name} dataset preprocessor")
        print(f"{'='*80}")
        print(f"KGFile: {self.kg_file}")
        print(f"questionFile: {self.questions_file}")
        print(f"Database: {self.db_path}")
        print(f"indexDir: {self.index_dir}")
        print(f"OutputFile: {self.output_file}")
        print(f"Hybridcache: {self.hybrid_cache_dir}")
    
    def _parse_time(self, time_str: str) -> int:
        """Parse time string to integer (year)
        
        Supported formats:
        - Pure number: "2005" -> 2005
        - Date format: "2005-01-01" -> 2005 (extract year)
        - Special: "+40200000" -> None (ignore)
        """
        if not time_str or not time_str.strip():
            return None
        
        time_str = time_str.strip()
        
        try:
            # Process date format (MultiTQ: 2005-01-01)
            if '-' in time_str:
                year_str = time_str.split('-')[0]
                return int(year_str)
            
            # Process pure number (TimeQuestions: 2005)
            if time_str.isdigit():
                return int(time_str)
            
            # Ignore special values starting with +
            if time_str.startswith('+'):
                return None
            
            # Try direct conversion
            return int(time_str)
        except:
            return None
    
    def _parse_time_to_timestamp(self, time_str: str) -> int:
        """Parse time string to Unix timestamp for precise comparison
        
        Supported formats:
        - Date format: "2005-01-01" -> Unix timestamp
        - Year only: "2005" -> timestamp for 2005-01-01
        
        Returns:
            Unix timestamp in seconds, or None if parsing fails
        """
        if not time_str or not time_str.strip():
            return None
        
        time_str = time_str.strip()
        
        try:
            if '-' in time_str:
                # YYYY-MM-DD format (MultiTQ)
                parts = time_str.split('-')
                if len(parts) >= 3:
                    year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
                elif len(parts) == 2:
                    year, month, day = int(parts[0]), int(parts[1]), 1
                else:
                    year, month, day = int(parts[0]), 1, 1
                
                dt = datetime(year, month, day)
                return int(dt.timestamp())
            elif time_str.isdigit():
                # Pure year (TimeQuestions)
                year = int(time_str)
                dt = datetime(year, 1, 1)
                return int(dt.timestamp())
            else:
                return None
        except:
            return None
    
    def create_database(self, force_rebuild: bool = False):
        """Create SQLite database from KG file
        
        Args:
            force_rebuild: If True, rebuild database even if it exists
        """
        print(f"\n{'='*80}")
        print(f"Step 1: Create SQLite database ({self.dataset_name})")
        print(f"{'='*80}")
        
        if os.path.exists(self.db_path) and not force_rebuild:
            print(f"✅ Database exists: {self.db_path}")
            # Validate table structure
            conn = sqlite3.connect(self.db_path)
            cur = conn.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='edges'")
            has_edges = cur.fetchone() is not None
            conn.close()
            
            if not has_edges:
                print(f"⚠️ Database table structure incorrect (missing edges table), rebuilding...")
                os.remove(self.db_path)
            else:
                return
        
        if force_rebuild and os.path.exists(self.db_path):
            print(f"🔄 Force rebuild: removing existing database...")
            os.remove(self.db_path)
        
        if not os.path.exists(self.kg_file):
            print(f"❌ KG file does not exist: {self.kg_file}")
            return
        
        
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA temp_store=MEMORY")
        conn.execute("PRAGMA foreign_keys=ON")
        
        cur = conn.cursor()
        cur.executescript("""
            CREATE TABLE IF NOT EXISTS entities(
                id   INTEGER PRIMARY KEY,
                name TEXT NOT NULL UNIQUE
            );
            CREATE TABLE IF NOT EXISTS edges(
                id            INTEGER PRIMARY KEY,
                head_id       INTEGER NOT NULL,
                relation      TEXT NOT NULL,
                tail_id       INTEGER NOT NULL,
                t_start       TEXT NOT NULL,
                t_end         TEXT NOT NULL,
                t_start_epoch INTEGER NOT NULL,
                t_end_epoch   INTEGER NOT NULL,
                granularity   TEXT NOT NULL DEFAULT 'year',
                FOREIGN KEY(head_id) REFERENCES entities(id),
                FOREIGN KEY(tail_id) REFERENCES entities(id),
                UNIQUE(head_id, relation, tail_id, t_start, t_end)
            );
            CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name);
            CREATE INDEX IF NOT EXISTS idx_edges_head ON edges(head_id);
            CREATE INDEX IF NOT EXISTS idx_edges_tail ON edges(tail_id);
            CREATE INDEX IF NOT EXISTS idx_edges_tstart ON edges(t_start_epoch);
            CREATE INDEX IF NOT EXISTS idx_edges_tend ON edges(t_end_epoch);
            CREATE INDEX IF NOT EXISTS idx_edges_tstart_str ON edges(t_start);
        """)
        
        print("📊 Parsing KG file and inserting data...")
        
        # Collect entities and edges
        entities = {}  # name -> id
        edges_buffer = []
        
        with open(self.kg_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Reading KG"):
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split('\t')
                
                if self.config["kg_format"] == "5cols":
                    if len(parts) >= 5:
                        head, relation, tail, time = parts[0], parts[1], parts[2], parts[3]
                elif self.config["kg_format"] == "4cols":
                    if len(parts) >= 4:
                        head, relation, tail, time = parts[:4]
                else:
                    continue
                
                # Clean strings
                head, relation, tail, time = head.strip(), relation.strip(), tail.strip(), time.strip()
                
                # Skip invalid data
                if not all([head, relation, tail, time]):
                    continue
                
                # Get or create entity ID
                if head not in entities:
                    entities[head] = len(entities) + 1
                if tail not in entities:
                    entities[tail] = len(entities) + 1
                
                # Parse time to Unix timestamp for precise comparison
                # This works for both YYYY-MM-DD (MultiTQ) and YYYY (TimeQuestions)
                time_epoch = self._parse_time_to_timestamp(time)
                if time_epoch is None:
                    time_epoch = 0
                
                # Store both t_start string and t_start_epoch timestamp
                # Both enable precise date comparison and sorting
                # Compatible with both MultiTQ (YYYY-MM-DD) and TimeQuestions (YYYY)
                edges_buffer.append((
                    entities[head], relation, entities[tail],
                    time, time, time_epoch, time_epoch, 'day' if '-' in time else 'year'
                ))
        
        # First insert all entities (must be done before edges due to foreign keys)
        print(f"💾 Inserting {len(entities)} entities...")
        for name, eid in tqdm(entities.items(), desc="Inserting entities"):
            cur.execute("INSERT INTO entities (id, name) VALUES (?, ?)", (eid, name))
        conn.commit()
        
        # Then insert edges (batch processing for efficiency)
        print(f"💾 Inserting edge data (batch processing)...")
        batch_size_insert = 50000
        for i in tqdm(range(0, len(edges_buffer), batch_size_insert), desc="Batch inserting edges"):
            batch = edges_buffer[i:i+batch_size_insert]
            cur.executemany(
                "INSERT OR IGNORE INTO edges(head_id, relation, tail_id, t_start, t_end, t_start_epoch, t_end_epoch, granularity) VALUES (?,?,?,?,?,?,?,?)",
                batch
            )
            conn.commit()
        
        # Get actual number of inserted edges
        cur.execute("SELECT COUNT(*) FROM edges")
        n_edges = cur.fetchone()[0]
        
        conn.commit()
        conn.close()
        
        print(f"✅ Database creation completed: {self.db_path}")
        print(f"   - Number of entities: {len(entities)}")
        print(f"   - Number of edges: {n_edges}")
        print(f"   - Time storage format:")
        print(f"     • t_start: Original string (YYYY-MM-DD for MultiTQ, YYYY for TimeQuestions)")
        print(f"     • t_start_epoch: Year as integer (for backward compatibility)")
        print(f"   📝 Note: Use t_start string for precise date comparison")
    
    def build_entity_index(self):
        """Build entity index (for semantic retrieval)"""
        print(f"\n{'='*80}")
        print(f"Step 2: Build entity index ({self.dataset_name})")
        print(f"{'='*80}")
        
        if os.path.exists(os.path.join(self.index_dir, "entity_hnsw.index")):
            print(f"✅ Index already exists: {self.index_dir}")
            return
        
        if not os.path.exists(self.db_path):
            print(f"❌ Database does not exist: {self.db_path}")
            return
        
        print(f"🔨 Building entity index to: {self.index_dir}")
        
        try:
            # Use index building method from new.py
            cmd_build_index(
                db_path=self.db_path,
                out_dir=self.index_dir,
                fp16=False,
                build_hnsw=True,
                M=64,
                ef_construction=400,
                ef_search=128
            )
            print("✅ Entity index build completed")
        except Exception as e:
            print(f"❌ Index build failed: {e}")
            print("   Skipping index build")
    
    def generate_candidates(self, num_questions: int = None):
        """Generate candidate entities for questions"""
        print(f"\n{'='*80}")
        print(f"Step 3: Generate candidate entities ({self.dataset_name})")
        print(f"{'='*80}")
        
        if not os.path.exists(self.questions_file):
            print(f"❌ Question file does not exist: {self.questions_file}")
            return
        
        if not os.path.exists(self.db_path):
            print(f"❌ Database does not exist: {self.db_path}")
            return
        
        print(f"📁 Reading question file: {self.questions_file}")
        with open(self.questions_file, 'r', encoding='utf-8') as f:
            questions_all = json.load(f)
        
        total_available = len(questions_all)
        if num_questions:
            questions = questions_all[:num_questions]
            print(f"📊 Processing first {num_questions} questions (out of {total_available} total)")
            print(f"   💡 Tip: Use --num-questions N or -n N to control the number of questions")
        else:
            questions = questions_all
            print(f"📊 Processing all {len(questions)} questions")
            print(f"   💡 Tip: Use --num-questions N or -n N to process fewer questions for testing")
        
        # Read entity name mapping
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("SELECT id, name FROM entities")
        id_name = {int(eid): nm for eid, nm in cur.fetchall()}
        conn.close()
        
        # Process questions
        output_data = []
        topk = 200
        per_kw_topk = 500
        agg = "max"
        
        question_id_field = self.config["question_id_field"]
        question_text_field = self.config["question_text_field"]
        answer_field = self.config["answer_field"]
        
        for q in tqdm(questions, desc="Generating candidate entities"):
            question_text = q.get(question_text_field, "")
            qid = q.get(question_id_field)
            
            # Extract keywords
            try:
                keywords = combined_keywords(question_text)
            except Exception as e:
                # If keyword extraction fails, use simple tokenization
                keywords = [w for w in question_text.lower().split() if len(w) > 2]
                if not keywords:
                    keywords = [question_text]
            
            # Generate candidate entities
            candidates = []
            try:
                # Check if index exists
                if os.path.exists(self.index_dir):
                    cands = semantic_candidates_by_keywords(
                        db_path=self.db_path,
                        index_dir=self.index_dir,
                        keywords=keywords,
                        topk=topk,
                        per_kw_topk=per_kw_topk,
                        agg=agg,
                        exclude_ids=[],
                        use_faiss=True
                    )
                    candidates = [
                        {"id": int(eid), "name": id_name.get(int(eid), ""), "score": float(score)}
                        for eid, _nm, score in cands if score > 0.65
                    ]
            except Exception as e:
                print(f"Warning: Candidate generation failed for question {qid}: {e}")
            
            # Build output format (keep all raw fields)
            result = q.copy()  # Copy all raw fields
            result["keywords_used"] = keywords
            result["candidates"] = candidates
            
            output_data.append(result)
        
        # Save results
        print(f"💾 Saving results to: {self.output_file}")
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Candidate entity generation completed, processed {len(output_data)} questions")
        
        # Show statistics
        total_candidates = sum(len(q["candidates"]) for q in output_data)
        avg_candidates = total_candidates / len(output_data) if output_data else 0
        print(f"   - Average candidates per question: {avg_candidates:.1f}")
        
        # Show example
        if output_data:
            print(f"\nSample question:")
            example = output_data[0]
            print(f"  Question ID: {example.get(question_id_field)}")
            print(f"  Question: {example.get(question_text_field)}")
            print(f"  Keywords: {example.get('keywords_used', [])}")
            print(f"  Candidate count: {len(example.get('candidates', []))}")
            if example.get('candidates'):
                print(f"  Top 3 candidates:")
                for cand in example['candidates'][:3]:
                    print(f"    - {cand['name']} (score: {cand['score']:.3f})")
    
    def build_hybrid_index(self, gpu_id: int = 0, use_gpu: bool = True):
        """Build Hybrid retrieval index (BGE-M3 + FAISS)
        
        Args:
            gpu_id: GPU device ID (for selecting GPU, but does not affect file name)
            use_gpu: is buseGPU（if available）
        """
        print(f"\n{'='*80}")
        print(f"Step4: Build Hybrid retrieval index ({self.dataset_name})")
        print(f"{'='*80}")
        
        if not HYBRID_AVAILABLE:
            print("❌ Hybrid retrieval dependency missing, Skip indexBuild")
            print("   Please install: pip install faiss-gpu torch FlagEmbedding")
            return
        
        if not os.path.exists(self.db_path):
            print(f"❌ Database does not exist: {self.db_path}")
            print("   Please run create_database()")
            return
        
        # define cache file path (no use GPU ID suffix)
        index_path = os.path.join(self.hybrid_cache_dir, "hybrid_index.bin")
        data_cache_path = os.path.join(self.hybrid_cache_dir, "hybrid_data_cache.pkl")
        embedding_cache_path = os.path.join(self.hybrid_cache_dir, "hybrid_embeddings.npy")
        
        # check if exists
        if os.path.exists(index_path) and os.path.exists(data_cache_path) and os.path.exists(embedding_cache_path):
            print(f"✅ Hybridindexexists: {self.hybrid_cache_dir}")
            print(f"   - index: {index_path}")
            print(f"   - data cache: {data_cache_path}")
            print(f"   - embedding cache: {embedding_cache_path}")
            return
        
        print(f"🔨 StartingBuildHybridindex...")
        print(f"   cacheDir: {self.hybrid_cache_dir}")
        
        # Step1: from database Load triples
        print("\n📊 Step1: from database Load triples...")
        triple_list, fact_list, time_list = self._load_triples_from_db()
        
        if not triple_list:
            print("❌ Cannot Load triples data")
            return
        
        print(f"✅ Loaddone {len(triple_list)} triples")
        
        # Save数据cache
        print("\n💾 Save data cache...")
        cache_data = {
            'triple_list': triple_list,
            'fact_list': fact_list,
            'time_list': time_list
        }
        with open(data_cache_path, 'wb') as f:
            pickle.dump(cache_data, f)
        print(f"✅ Data cache saved: {data_cache_path}")
        
        # Step2: LoadBGE-M3 model
        print("\n🔄 Step2: LoadBGE-M3 model...")
        device = f'cuda:{gpu_id}' if torch.cuda.is_available() and use_gpu else 'cpu'
        print(f"   Device: {device}")
        
        try:
            model = BGEM3FlagModel(
                'BAAI/bge-m3',
                use_fp16=True,
                devices=[device]
            )
            print("✅ BGE-M3 model LoadSuccess")
        except Exception as e:
            print(f"❌ Model LoadFailed: {e}")
            return
        
        # Step3: Encode All facts
        print(f"\n🔄 Step3: Encode {len(fact_list)} facts...")
        try:
            embeddings = model.encode_corpus(
                fact_list,
                convert_to_numpy=True,
                batch_size=512,
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False
            )
            
            embeddings = embeddings['dense_vecs']
            embeddings = embeddings.astype(np.float32)
            
            print(f"✅ Embedding vector shape: {embeddings.shape}")
            
            # Save embedding vector cache
            np.save(embedding_cache_path, embeddings)
            print(f"✅ Embedding vector cache saved: {embedding_cache_path}")
            
        except Exception as e:
            print(f"❌ Encoding Failed: {e}")
            return
        
        # Step4: BuildFAISSindex
        print("\n🔄 Step4: BuildFAISSindex...")
        try:
            embedding_size = embeddings.shape[1]
            n_clusters = 500
            nprobe = 60
            
            # Createindex
            quantizer = faiss.IndexFlatIP(embedding_size)
            index = faiss.IndexIVFFlat(quantizer, embedding_size, n_clusters, faiss.METRIC_INNER_PRODUCT)
            index.nprobe = nprobe
            
            # GPU acceleration (if available)
            if use_gpu and torch.cuda.is_available():
                try:
                    print("   Trying to use GPU acceleration...")
                    res = faiss.StandardGpuResources()
                    index = faiss.index_cpu_to_gpu(res, gpu_id, index)
                    print("   ✅ GPU acceleration enabled")
                except Exception as e:
                    print(f"   ⚠️ GPU acceleration Failed, useCPU: {e}")
            
            # Train index
            if not index.is_trained:
                print("   Training index...")
                index.train(embeddings)
                print("   ✅ index training completed")
            
            # Add vectors
            print("   Adding vectors to index...")
            index.add(embeddings)
            print(f"   ✅ Added {index.ntotal} vectors")
            
            # return to CPU and Save
            if use_gpu and torch.cuda.is_available():
                try:
                    index = faiss.index_gpu_to_cpu(index)
                except:
                    pass
            
            # Saveindex
            faiss.write_index(index, index_path)
            print(f"✅ index saved: {index_path}")
            
        except Exception as e:
            print(f"❌ index build Failed: {e}")
            import traceback
            traceback.print_exc()
            return
        
        print(f"\n{'='*80}")
        print(f"✅ Hybrid index build completed!")
        print(f"{'='*80}")
        print(f"cacheDir: {self.hybrid_cache_dir}")
        print(f"  - indexFile: hybrid_index.bin ({os.path.getsize(index_path) / 1024 / 1024:.1f} MB)")
        print(f"  - Data cache: hybrid_data_cache.pkl ({os.path.getsize(data_cache_path) / 1024 / 1024:.1f} MB)")
        print(f"  - Embedding cache: hybrid_embeddings.npy ({os.path.getsize(embedding_cache_path) / 1024 / 1024:.1f} MB)")
    
    def build_relation_embeddings(self):
        """
        Build relation embedding cache for the dataset
        Uses OpenAI text-embedding-3-small API to compute embeddings for all unique relations
        """
        print(f"\n{'='*80}")
        print(f"Building Relation Embedding Cache for {self.dataset_name}")
        print(f"{'='*80}")
        
        # Check if embedding cache module is available
        try:
            sys.path.insert(0, str(Path(__file__).parent.parent / 'memotime'))
            from kg_agent.embedding_cache import RelationEmbeddingCache
        except ImportError as e:
            print(f"❌ Cannot import RelationEmbeddingCache: {e}")
            print("   Please ensure TPKG/kg_agent/embedding_cache.py is available")
            return
        
        # Initialize RelationEmbeddingCache with dataset-specific directory
        try:
            print(f"📁 Embedding cache directory: {self.embedding_cache_dir}")
            cache = RelationEmbeddingCache(self.db_path, cache_dir=self.embedding_cache_dir)
            
            # Get cache info
            cache_info = cache.get_cache_info()
            print(f"📊 Current cache status:")
            print(f"   - Cached relations: {cache_info.get('cached_relations', 0)}")
            print(f"   - Total relations in DB: {cache_info.get('total_relations', 0)}")
            
            # Precompute all embeddings
            if cache_info.get('cached_relations', 0) < cache_info.get('total_relations', 0):
                print(f"\n🔄 Precomputing relation embeddings...")
                cache.precompute_all_embeddings()
                print(f"✅ Relation embeddings precomputed successfully!")
            else:
                print(f"✅ All relation embeddings already cached")
            
            # Show final stats
            final_info = cache.get_cache_info()
            print(f"\n📊 Final cache statistics:")
            print(f"   - Total relations: {final_info.get('total_relations', 0)}")
            print(f"   - Cached relations: {final_info.get('cached_relations', 0)}")
            print(f"   - Cache directory: {self.embedding_cache_dir}")
            
        except Exception as e:
            print(f"❌ Relation embedding building failed: {e}")
            import traceback
            traceback.print_exc()
    
    def _load_triples_from_db(self):
        """from database Load triples data (use edges table structure)
        
        Returns:
            tuple: (triple_list, fact_list, time_list)
        """
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Query All edge (use t_start field)
            query = """
            SELECT 
                h.name as head,
                e.relation,
                t.name as tail,
                e.t_start
            FROM edges e
            JOIN entities h ON h.id = e.head_id
            JOIN entities t ON t.id = e.tail_id
            ORDER BY e.t_start_epoch
            """
            
            cursor = conn.cursor()
            cursor.execute(query)
            results = cursor.fetchall()
            conn.close()
            
            triple_list = []
            fact_list = []
            time_list = []
            
            for row in tqdm(results, desc="Load triples"):
                head, relation, tail, t_start = row
                
                # Build triples
                triple = f"({head}, {relation}, {tail})"
                fact = f"{head} {relation.replace('_', ' ')} {tail}"
                if t_start:
                    fact += f" at {t_start}"
                
                triple_list.append(triple)
                fact_list.append(fact)
                
                # Parse time
                try:
                    if t_start:
                        # Support multiple time format
                        if '-' in t_start:
                            # YYYY-MM-DDformat
                            year = int(t_start.split('-')[0])
                        elif t_start.isdigit():
                            # Pure year
                            year = int(t_start)
                        else:
                            time_list.append(None)
                            continue
                        
                        time_obj = datetime(year, 1, 1).date()
                        time_list.append(time_obj)
                    else:
                        time_list.append(None)
                except:
                    time_list.append(None)
            
            return triple_list, fact_list, time_list
            
        except Exception as e:
            print(f"❌ from database Load triples Failed: {e}")
            import traceback
            traceback.print_exc()
            return [], [], []
    
    def run_all_steps(self, num_questions: int = None, build_hybrid: bool = False, build_embeddings: bool = False, gpu_id: int = 0):
        """Run all preprocessing steps
        
        Args:
            num_questions: Number of questions to process (None means all)
            build_hybrid: Whether to build hybrid index
            build_embeddings: Whether to build relation embeddings
            gpu_id: GPU device ID
        """
        print(f"\n{'='*80}")
        print(f"{self.dataset_name} Dataset Preprocessing")
        print(f"{'='*80}")
        
        # Step1: Create database
        self.create_database()
        
        # Step2: Build entity index
        self.build_entity_index()
        
        # Step3: Generate candidate entities
        self.generate_candidates(num_questions=num_questions)
        
        # Step4: Build hybrid index (optional)
        if build_hybrid:
            self.build_hybrid_index(gpu_id=gpu_id)
        
        # Step5: Build relation embeddings (optional)
        if build_embeddings:
            self.build_relation_embeddings()
        
        print(f"\n{'='*80}")
        print(f"✅ {self.dataset_name} preprocessing completed!")
        print(f"{'='*80}")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Dataset preprocess script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process only first 10 questions for TimeQuestions
  python prepare_datasets.py --dataset TimeQuestions --num-questions 10
  
  # Quick test with 5 questions
  python prepare_datasets.py --test --dataset TimeQuestions
  
  # Process first 100 questions and build embeddings
  python prepare_datasets.py --dataset TimeQuestions --num-questions 100 --build-embeddings
  
  # Skip database and index, only generate candidates for 50 questions
  python prepare_datasets.py --skip-db --skip-index --num-questions 50
        """
    )
    parser.add_argument("--dataset", type=str, choices=["TimeQuestions", "MultiTQ", "all"],
                       default="all", help="Dataset to process")
    parser.add_argument("--test", action="store_true", help="Test mode (only process first 5 questions)")
    parser.add_argument("--num-questions", "-n", type=int, default=None,
                       help="Number of questions to process (default: process all). Use -n 10, -n 50, -n 100 for quick tests")
    parser.add_argument("--skip-db", action="store_true", help="Skip database creation")
    parser.add_argument("--force-rebuild-db", action="store_true", help="Force rebuild database even if it exists")
    parser.add_argument("--skip-index", action="store_true", help="Skip index building")
    parser.add_argument("--skip-candidates", action="store_true", help="Skip candidate generation")
    parser.add_argument("--build-hybrid", action="store_true", help="Build hybrid retrieval index")
    parser.add_argument("--build-embeddings", action="store_true", help="Build relation embedding cache")
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU device ID (default: 0)")
    
    args = parser.parse_args()
    
    # Test mode
    if args.test:
        args.num_questions = 5
    
    # Determine which Dataset to process
    if args.dataset == "all":
        datasets_to_process = ["TimeQuestions", "MultiTQ"]
    else:
        datasets_to_process = [args.dataset]
    
    # Show configuration
    print(f"\n{'='*80}")
    print("📋 CONFIGURATION SUMMARY")
    print(f"{'='*80}")
    print(f"Datasets to process: {', '.join(datasets_to_process)}")
    if args.num_questions:
        print(f"Questions per dataset: {args.num_questions} (limited)")
    else:
        print(f"Questions per dataset: ALL")
    print(f"Skip database: {args.skip_db}")
    print(f"Skip index: {args.skip_index}")
    print(f"Skip candidates: {args.skip_candidates}")
    print(f"Build hybrid index: {args.build_hybrid}")
    print(f"Build embeddings: {args.build_embeddings}")
    if args.build_hybrid:
        print(f"GPU ID: {args.gpu_id}")
    print(f"{'='*80}\n")
    
    # process per Dataset
    for dataset_name in datasets_to_process:
        try:
            preprocessor = DatasetPreprocessor(dataset_name)
            
            if not args.skip_db:
                preprocessor.create_database(force_rebuild=args.force_rebuild_db)
            
            if not args.skip_index:
                preprocessor.build_entity_index()
            
            if not args.skip_candidates:
                preprocessor.generate_candidates(num_questions=args.num_questions)
            
            if args.build_hybrid:
                preprocessor.build_hybrid_index(gpu_id=args.gpu_id)
            
            if args.build_embeddings:
                preprocessor.build_relation_embeddings()
            
            print(f"\n✅ {dataset_name} processing completed!\n")
            
        except Exception as e:
            print(f"\n❌ {dataset_name} processFailed: {e}\n")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*80}")
    print("All Dataset Processing completed!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

