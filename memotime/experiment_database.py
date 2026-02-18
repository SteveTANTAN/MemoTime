#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sqlite3
import json
import datetime
import hashlib
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from functools import wraps

def db_retry(max_retries: int = 20, delay: float = 10.0):
    """
    Database operation retry decorator
    
    Args:
        max_retries: Maximum number of retries
        delay: Retry interval (seconds)
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except sqlite3.OperationalError as e:
                    if "locked" in str(e).lower() and attempt < max_retries:
                        print(f"⏳ Database locked, retry {attempt + 1}/{max_retries} times, wait {delay}s...")
                        time.sleep(delay)
                        continue
                    else:
                        raise
                except Exception as e:
                    raise
            return None
        return wrapper
    return decorator

class ExperimentDatabase:
    """Experiment database manager - save the best result according to the configuration"""
    
    def __init__(self, db_path: str = None, dataset: str = None):
        """
        Initialize experiment database  
        
        Args:
            db_path: Database path, if None, generate according to dataset
            dataset: Dataset name (TimeQuestions/MultiTQ), used to generate isolated database files
        """
        if db_path is None:
            # ✅ Generate isolated database path according to dataset
            if dataset is None:
                # Try to get from configuration
                try:
                    from config import TPKGConfig
                    dataset = TPKGConfig.DATASET
                except:
                    dataset = 'MultiTQ'  # Default
            
            # Generate dataset-specific database file name
            db_filename = f"experiments_{dataset.lower()}.db"
            db_path = Path(__file__).parent.parent / "experiments" / db_filename
            print(f"📊 Use dataset-specific experiment database: {db_path}")
        
        self.db_path = db_path
        self.dataset = dataset
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
    
    def _compute_config_hash(self, config: Dict[str, Any]) -> str:
        """Compute configuration hash (used as unique identifier)"""
        # Only use key configurations that affect experiment results
        key_config = {
            'dataset': config.get('Dataset configuration', {}).get('dataset', 'MultiTQ'),  # ✅ Add dataset
            'max_retries': config.get('Problem solving configuration', {}).get('max_retries', 1),  # ✅ Add max_retries  
            'max_depth': config.get('Problem solving configuration', {}).get('max_depth', 1),  # ✅ Add max_depth
            'max_branches': config.get('Problem solving configuration', {}).get('max_total_branches', 5),  # ✅ Add max_branches 
            'use_hybrid': config.get('Retrieval configuration', {}).get('use_hybrid_retrieval', True),  # ✅ Add use_hybrid
            'use_pool': config.get('Experience pool configuration', {}).get('use_experience_pool', True),  # ✅ Add use_pool
            'use_template': config.get('Template Learning configuration', {}).get('use_template_learning', True),
            'model': config.get('LLM configuration', {}).get('default_model', 'gpt-4o-mini'),  # ✅ Add model
            # Added: storage mode parameter (different storage modes should be treated as different configurations)
            'storage_mode': config.get('Storage configuration', {}).get('storage_mode', 'shared'),
            'enable_shared_fallback': config.get('Storage configuration', {}).get('enable_shared_fallback', False),
        }
        
        config_str = json.dumps(key_config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:12]
    
    def _init_database(self):
        """Initialize database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Configuration table (each configuration has one record)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS configurations (
            config_id TEXT PRIMARY KEY,
            config_name TEXT NOT NULL,
            config_hash TEXT UNIQUE NOT NULL,
            config_json TEXT NOT NULL,
            description TEXT,
            tags TEXT,
            created_at TEXT NOT NULL,
            last_updated TEXT NOT NULL,
            run_count INTEGER DEFAULT 0,
            best_success_rate REAL DEFAULT 0.0,
            best_run_at TEXT
        )
        """)
        
        # Run record table (each configuration may have multiple runs)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS runs (
            run_id INTEGER PRIMARY KEY AUTOINCREMENT,
            config_id TEXT NOT NULL,
            run_at TEXT NOT NULL,
            total_questions INTEGER DEFAULT 0,
            success_count INTEGER DEFAULT 0,
            fail_count INTEGER DEFAULT 0,
            success_rate REAL DEFAULT 0.0,
            is_best INTEGER DEFAULT 0,
            FOREIGN KEY (config_id) REFERENCES configurations(config_id)
        )
        """)
        
        # Question result table (only save the best run result)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS question_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            config_id TEXT NOT NULL,
            qid TEXT NOT NULL,
            question TEXT NOT NULL,
            question_type TEXT,
            final_answer TEXT,
            is_correct INTEGER,
            processing_time REAL,
            saved_at TEXT NOT NULL,
            answer_type TEXT,
            time_level TEXT,
            golden_answers TEXT,
            qlabel TEXT,
            topics TEXT,
            FOREIGN KEY (config_id) REFERENCES configurations(config_id),
            UNIQUE(config_id, qid)
        )
        """)
        
        # Trajectory table (only save the best run)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS trajectories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            config_id TEXT NOT NULL,
            qid TEXT NOT NULL,
            step_index INTEGER NOT NULL,
            subquestion TEXT NOT NULL,
            subquestion_obj TEXT,
            selected_seeds TEXT,
            available_seeds TEXT,
            result TEXT,
            sufficiency_test TEXT,
            saved_at TEXT NOT NULL,
            FOREIGN KEY (config_id) REFERENCES configurations(config_id),
            UNIQUE(config_id, qid, step_index)
        )
        """)
        
        # Evidence Edges table (only save the best run)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS evidence_edges (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            config_id TEXT NOT NULL,
            qid TEXT NOT NULL,
            step_index INTEGER,
            head_entity TEXT NOT NULL,
            relation TEXT NOT NULL,
            tail_entity TEXT NOT NULL,
            time_info TEXT,
            score REAL,
            source TEXT,
            is_used INTEGER DEFAULT 1,
            metadata TEXT,
            saved_at TEXT NOT NULL,
            FOREIGN KEY (config_id) REFERENCES configurations(config_id)
        )
        """)
        
        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_question_config ON question_results(config_id, qid)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_trajectory_config ON trajectories(config_id, qid)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_edges_config ON evidence_edges(config_id, qid)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_question_answer_type ON question_results(answer_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_question_qtype ON question_results(question_type)")
        
        # Add new columns (if not exist) - used to upgrade existing database
        new_columns = [
            ("answer_type", "TEXT"),
            ("time_level", "TEXT"),
            ("golden_answers", "TEXT"),
            ("qlabel", "TEXT"),
            ("topics", "TEXT")
        ]
        
        for column_name, column_type in new_columns:
            try:
                cursor.execute(f"ALTER TABLE question_results ADD COLUMN {column_name} {column_type}")
                print(f"✅ Add column: {column_name}")
            except sqlite3.OperationalError as e:
                if "duplicate column name" in str(e).lower():
                    pass  # Column already exists, ignore
                else:
                    raise
        
        conn.commit()
        conn.close()
    
    @db_retry(max_retries=20, delay=10.0)
    def get_or_create_config(self, 
                            config_name: str,
                            config: Dict[str, Any],
                            description: str = "",
                            tags: List[str] = None) -> str:
        """
        Get or create configuration
        Return config_id
        """
        config_hash = self._compute_config_hash(config)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check if it already exists
        cursor.execute("SELECT config_id FROM configurations WHERE config_hash = ?", (config_hash,))
        row = cursor.fetchone()
        
        if row:
            config_id = row[0]
            # Update configuration name and run count
            cursor.execute("""
            UPDATE configurations 
            SET config_name = ?,
                run_count = run_count + 1,
                last_updated = ?
            WHERE config_id = ?
            """, (config_name, datetime.datetime.now().isoformat(), config_id))
            conn.commit()
            conn.close()
            print(f"📌 Use existing configuration: {config_id} (update name to: {config_name})")
            return config_id
        
        # Create new configuration
        config_id = f"cfg_{config_hash}"
        
        cursor.execute("""
        INSERT INTO configurations
        (config_id, config_name, config_hash, config_json, description, tags, 
         created_at, last_updated, run_count)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1)
        """, (
            config_id,
            config_name,
            config_hash,
            json.dumps(config, ensure_ascii=False),
            description,
            json.dumps(tags or [], ensure_ascii=False),
            datetime.datetime.now().isoformat(),
            datetime.datetime.now().isoformat()
        ))
        
        conn.commit()
        conn.close()
        print(f"✅ Create new configuration: {config_id}")
        return config_id
    
    @db_retry(max_retries=20, delay=10.0)
    def record_run(self, 
                  config_id: str,
                  total_questions: int,
                  success_count: int,
                  fail_count: int) -> int:
        """Record one run, return run_id"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        success_rate = success_count / total_questions if total_questions > 0 else 0.0
        
        cursor.execute("""
        INSERT INTO runs
        (config_id, run_at, total_questions, success_count, fail_count, success_rate)
        VALUES (?, ?, ?, ?, ?, ?)
        """, (
            config_id,
            datetime.datetime.now().isoformat(),
            total_questions,
            success_count,
            fail_count,
            success_rate
        ))
        
        run_id = cursor.lastrowid
        
        # Check if it is the best result
        cursor.execute("""
        SELECT best_success_rate FROM configurations WHERE config_id = ?
        """, (config_id,))
        
        row = cursor.fetchone()
        current_best = row[0] if row else 0.0
        
        if success_rate > current_best:
            # Update best record
            cursor.execute("""
            UPDATE runs SET is_best = 0 WHERE config_id = ?
            """, (config_id,))
            
            cursor.execute("""
            UPDATE runs SET is_best = 1 WHERE rowid = ?
            """, (run_id,))
            
            cursor.execute("""
            UPDATE configurations 
            SET best_success_rate = ?,
                best_run_at = ?
            WHERE config_id = ?
            """, (success_rate, datetime.datetime.now().isoformat(), config_id))
            
            print(f"🏆 New best result! Success rate: {success_rate*100:.2f}%")
        else:
            print(f"📊 Current run success rate: {success_rate*100:.2f}% (best: {current_best*100:.2f}%)")
        
        conn.commit()
        conn.close()
        return run_id
    
    @db_retry(max_retries=20, delay=10.0)
    def get_question_result(self, config_id: str, qid: str) -> Optional[Dict[str, Any]]:
        """
        Get the saved result of a question
        
        Args:
            config_id: Configuration ID
            qid: Question ID
            
        Returns:
        Question result dictionary, if not exist return None
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
        SELECT qid, question, question_type, final_answer, is_correct, 
               answer_type, time_level, golden_answers, qlabel, saved_at
        FROM question_results 
        WHERE config_id = ? AND qid = ?
        """, (config_id, qid))
        
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return None
        
        return {
            'qid': row['qid'],
            'question': row['question'],
            'question_type': row['question_type'],
            'final_answer': row['final_answer'],
            'is_correct': bool(row['is_correct']) if row['is_correct'] is not None else None,
            'answer_type': row['answer_type'],
            'time_level': row['time_level'],
            'golden_answers': row['golden_answers'],
            'qlabel': row['qlabel'],
            'saved_at': row['saved_at']
        }
    
    @db_retry(max_retries=20, delay=10.0)
    def should_save_question(self, config_id: str, qid: str, is_correct: bool) -> bool:

        if is_correct is None:
            # No correctness information, always save
            return True
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
        SELECT is_correct FROM question_results 
        WHERE config_id = ? AND qid = ?
        """, (config_id, qid))
        
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            # Previously not answered, save
            return True
        
        previous_correct = row[0]
        
        if previous_correct is None:
            # No correctness information before, save
            return True
        
        previous_correct = bool(previous_correct)
        
        if not previous_correct and is_correct:
            # Previously wrong, this time correct, replace
            return True
        
        if previous_correct and is_correct:
            # Previously correct, this time also correct, replace (save the updated data)
            return True
        
        if previous_correct and not is_correct:
            # Previously correct, this time wrong, do not save
            return False
        
        if not previous_correct and not is_correct:
            # Previously wrong, this time also wrong, do not save
            return False
        
        return False
    
    def _load_question_metadata(self, qid: str) -> Dict[str, Any]:
        """Load question metadata from test.json"""
        try:
            from ..config import TPKGConfig
            # Use raw test data path (test.json) not the candidates file
            test_data_path = TPKGConfig.get_raw_test_data_path()
        except:
            # Fallback: try to determine from dataset
            test_data_path = str(Path(__file__).parent.parent / "Data" / "MultiTQ" / "test.json")
        
        try:
            with open(test_data_path, 'r', encoding='utf-8') as f:
                test_data_list = json.load(f)
            
            qid_int = int(qid)
            for test_item in test_data_list:
                if test_item.get('quid') == qid_int:
                    return {
                        'answer_type': test_item.get('answer_type', None),
                        'time_level': test_item.get('time_level', None),
                        'golden_answers': json.dumps(test_item.get('answers', []), ensure_ascii=False),
                        'qlabel': test_item.get('qlabel', None),
                        'topics': json.dumps(test_item.get('topics', []), ensure_ascii=False),
                        'qtype': test_item.get('qtype', None)  # Add question type
                    }
        except Exception as e:
            print(f"⚠️  Load question {qid} metadata failed: {e}")
        
        return {
            'answer_type': None,
            'time_level': None,
            'golden_answers': None,
            'qlabel': None,
            'topics': None,
            'qtype': None  # Add question type
        }
    
    @db_retry(max_retries=20, delay=10.0)
    def save_best_results(self,
                         config_id: str,
                         qid: str,
                         question: str,
                         question_type: str = None,
                         final_answer: str = None,
                         is_correct: bool = None,
                         trajectory: List[Dict[str, Any]] = None,
                         evidence_edges: List[Dict[str, Any]] = None):
        """Save best results (overwrite old data)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        now = datetime.datetime.now().isoformat()
        
        # Load question metadata
        metadata = self._load_question_metadata(qid)
        
        # 1. Save question result
        # Prioritize the incoming question_type, if None use the qtype loaded from test.json
        effective_question_type = question_type if question_type is not None else metadata['qtype']
        
        cursor.execute("""
        INSERT OR REPLACE INTO question_results
        (config_id, qid, question, question_type, final_answer, is_correct, saved_at,
         answer_type, time_level, golden_answers, qlabel, topics)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            config_id, qid, question, effective_question_type, final_answer,
            1 if is_correct else 0 if is_correct is not None else None,
            now,
            metadata['answer_type'],
            metadata['time_level'],
            metadata['golden_answers'],
            metadata['qlabel'],
            metadata['topics']
        ))
        
        # 2. Delete old trajectory
        cursor.execute("DELETE FROM trajectories WHERE config_id = ? AND qid = ?", (config_id, qid))
        
        # 3. Save new trajectory
        if trajectory:
            for step_idx, step in enumerate(trajectory):
                subq = step.get('subq', {})
                cursor.execute("""
                INSERT INTO trajectories
                (config_id, qid, step_index, subquestion, subquestion_obj,
                 selected_seeds, available_seeds, result, sufficiency_test, saved_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    config_id, qid, step_idx,
                    subq.get('text', ''),
                    json.dumps(subq, ensure_ascii=False),
                    json.dumps(step.get('selected_seed_names', []), ensure_ascii=False),
                    json.dumps(step.get('available_seeds', []), ensure_ascii=False),
                    json.dumps(step.get('result', {}), ensure_ascii=False),
                    json.dumps(step.get('sufficiency_test', {}), ensure_ascii=False),
                    now
                ))
        
        # 4. Delete old edges
        cursor.execute("DELETE FROM evidence_edges WHERE config_id = ? AND qid = ?", (config_id, qid))
        
        # 5. Save new edges
        if evidence_edges:
            for edge in evidence_edges:
                cursor.execute("""
                INSERT INTO evidence_edges
                (config_id, qid, step_index, head_entity, relation, tail_entity,
                 time_info, score, source, is_used, metadata, saved_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    config_id, qid, edge.get('step_index'),
                    edge.get('head_entity', ''),
                    edge.get('relation', ''),
                    edge.get('tail_entity', ''),
                    edge.get('time_info'),
                    edge.get('score'),
                    edge.get('source'),
                    1 if edge.get('is_used', True) else 0,
                    json.dumps(edge.get('metadata', {}), ensure_ascii=False),
                    now
                ))
        
        conn.commit()
        conn.close()
    
    def get_trajectory(self, config_id: str, qid: str) -> List[Dict[str, Any]]:
        """Get trajectory"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
        SELECT * FROM trajectories 
        WHERE config_id = ? AND qid = ?
        ORDER BY step_index
        """, (config_id, qid))
        
        rows = cursor.fetchall()
        conn.close()
        
        trajectory = []
        for row in rows:
            step = dict(row)
            step['subquestion_obj'] = json.loads(step['subquestion_obj'])
            step['selected_seeds'] = json.loads(step['selected_seeds'])
            step['available_seeds'] = json.loads(step['available_seeds'])
            step['result'] = json.loads(step['result'])
            step['sufficiency_test'] = json.loads(step['sufficiency_test'])
            trajectory.append(step)
        
        return trajectory
    
    def get_evidence_edges(self, config_id: str, qid: str) -> List[Dict[str, Any]]:
        """Get evidence edges"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
        SELECT * FROM evidence_edges 
        WHERE config_id = ? AND qid = ?
        ORDER BY step_index, id
        """, (config_id, qid))
        
        rows = cursor.fetchall()
        conn.close()
        
        edges = []
        for row in rows:
            edge = dict(row)
            edge['metadata'] = json.loads(edge['metadata'])
            edge['is_used'] = bool(edge['is_used'])
            edges.append(edge)
        
        return edges
    
    def list_configurations(self) -> List[Dict[str, Any]]:
        """List all configurations"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
        SELECT * FROM configurations 
        ORDER BY best_success_rate DESC, last_updated DESC
        """)
        
        rows = cursor.fetchall()
        conn.close()
        
        configs = []
        for row in rows:
            cfg = dict(row)
            cfg['config_json'] = json.loads(cfg['config_json'])
            cfg['tags'] = json.loads(cfg['tags'])
            configs.append(cfg)
        
        return configs
    
    def get_config_stats(self, config_id: str) -> Dict[str, Any]:
        """Get configuration statistics"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Basic information
        cursor.execute("SELECT * FROM configurations WHERE config_id = ?", (config_id,))
        config = dict(cursor.fetchone())
        config['config_json'] = json.loads(config['config_json'])
        
        # Run history
        cursor.execute("""
        SELECT run_at, success_rate, is_best
        FROM runs
        WHERE config_id = ?
        ORDER BY run_at DESC
        """, (config_id,))
        runs = [dict(row) for row in cursor.fetchall()]
        
        # Data statistics
        cursor.execute("""
        SELECT 
            COUNT(DISTINCT qid) as total_questions,
            AVG(CASE WHEN is_correct = 1 THEN 1.0 ELSE 0.0 END) as avg_success_rate
        FROM question_results
        WHERE config_id = ?
        """, (config_id,))
        
        stats = dict(cursor.fetchone())
        
        conn.close()
        
        return {
            'config': config,
            'runs': runs,
            'stats': stats
        }
    
    def export_config_data(self, config_id: str, output_path: str):
        """Export all data of the configuration"""
        config_info = self.get_config_stats(config_id)
        
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Get all questions
        cursor.execute("SELECT * FROM question_results WHERE config_id = ?", (config_id,))
        questions = [dict(row) for row in cursor.fetchall()]
        
        # Get trajectory and edges for each question
        for q in questions:
            qid = q['qid']
            q['trajectory'] = self.get_trajectory(config_id, qid)
            q['evidence_edges'] = self.get_evidence_edges(config_id, qid)
        
        conn.close()
        
        export_data = {
            'config_info': config_info,
            'questions': questions
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Configuration data exported: {output_path}")


# Global singleton
_db_instance = None
_last_dataset = None

def get_experiment_db(dataset: str = None) -> ExperimentDatabase:
    """
    Get global database instance (switch automatically according to dataset)
    
    Args:
        dataset: Dataset name, if None get from TPKGConfig
    
    Returns:
        ExperimentDatabase instance
    """
    global _db_instance, _last_dataset
    
    # Get current dataset
    if dataset is None:
        try:
            from config import TPKGConfig
            dataset = TPKGConfig.DATASET
        except:
            dataset = 'MultiTQ'
    
    # If dataset changes or instance does not exist, recreate
    if _db_instance is None or _last_dataset != dataset:
        print(f"🔄 Switch experiment database to dataset: {dataset}")
        _db_instance = ExperimentDatabase(dataset=dataset)
        _last_dataset = dataset
    
    return _db_instance


if __name__ == "__main__":
    db = ExperimentDatabase()
    
    # Test
    test_config = {
        'Problem solving configuration': {'max_retries': 2},
        'Retrieval configuration': {'use_hybrid_retrieval': True},
        'Experience pool configuration': {'use_experience_pool': True}
    }
    
    config_id = db.get_or_create_config("test", test_config, "Test configuration")
    print(f"Config ID: {config_id}")
    
    # Record run
    run_id = db.record_run(config_id, 10, 8, 2)
    print(f"Run ID: {run_id}")
    
    # Save data
    db.save_best_results(
        config_id, "q123", "Test question", "equal", "Answer", True,
        trajectory=[{'subq': {'text': 'Subquestion'}, 'selected_seed_names': ['A']}],
        evidence_edges=[{'head_entity': 'A', 'relation': 'R', 'tail_entity': 'B'}]
    )
    
    # Query
    configs = db.list_configurations()
    print(f"\nConfiguration list: {len(configs)}")
    for cfg in configs:
        print(f"  {cfg['config_id']}: {cfg['config_name']} (Best success rate: {cfg['best_success_rate']*100:.1f}%)")



