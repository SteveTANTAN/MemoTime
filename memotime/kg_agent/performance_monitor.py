#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Performance Monitor for TPKG System
-----------------------------------
monitor performance metrics of each stage in the system:
- time consumption
- memory usage
- branch and depth of question decomposition
"""

import time
import psutil
import json
import sqlite3
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import threading
from contextlib import contextmanager

@dataclass
class StageMetrics:
    """performance metrics of a single stage"""
    stage_name: str
    start_time: float
    end_time: float
    duration: float
    llm_calls: int = 0
    llm_tokens_input: int = 0
    llm_tokens_output: int = 0
    memory_start_mb: float = 0.0
    memory_end_mb: float = 0.0
    memory_peak_mb: float = 0.0
    additional_info: Dict[str, Any] = None

@dataclass
class DecompositionInfo:
    """question decomposition information"""
    original_question: str
    subquestions: List[str]
    branch_count: int
    depth: int
    decomposition_method: str = "unknown"

@dataclass
class QuestionRecord:
    """complete question record"""
    qid: str
    question: str
    question_type: str
    is_correct: bool
    final_answer: str
    answer_path: List[Dict[str, Any]]
    
    # performance metrics
    total_duration: float
    total_llm_calls: int
    total_llm_tokens_input: int
    total_llm_tokens_output: int
    total_memory_peak_mb: float
    
    # metrics of each stage
    classification_metrics: Optional[StageMetrics] = None
    decomposition_metrics: Optional[StageMetrics] = None
    retrieval_metrics: Optional[StageMetrics] = None
    pruning_metrics: Optional[StageMetrics] = None
    qa_metrics: Optional[StageMetrics] = None
    
    # decomposition information
    decomposition_info: Optional[DecompositionInfo] = None
    
    # metadata
    timestamp: str = ""
    version: str = "1.0"

class PerformanceMonitor:
    """performance monitor"""
    
    def __init__(self, db_path: str = str(Path(__file__).parent.parent / "data" / "performance.db")):
        self.db_path = db_path
        self.current_question = None
        self.current_stage = None
        self.stage_start_time = None
        self.stage_start_memory = None
        self.llm_calls = 0
        self.llm_tokens_input = 0
        self.llm_tokens_output = 0
        self.stage_metrics = {}
        self.lock = threading.Lock()
        
        # initialize database
        self._init_database()
    
    def _init_database(self):
        """initialize database tables"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # create question record table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS question_records (
                    qid TEXT PRIMARY KEY,
                    question TEXT NOT NULL,
                    question_type TEXT,
                    is_correct BOOLEAN,
                    final_answer TEXT,
                    answer_path TEXT,  -- JSON string
                    
                    -- overall performance metrics
                    total_duration REAL,
                    total_llm_calls INTEGER,
                    total_llm_tokens_input INTEGER,
                    total_llm_tokens_output INTEGER,
                    total_memory_peak_mb REAL,
                    
                    -- decomposition information
                    decomposition_info TEXT,  -- JSON string
                    
                    -- metadata
                    timestamp TEXT,
                    version TEXT,
                    
                    -- metrics of each stage (JSON strings)
                    classification_metrics TEXT,
                    decomposition_metrics TEXT,
                    retrieval_metrics TEXT,
                    pruning_metrics TEXT,
                    qa_metrics TEXT
                )
            """)
            
            # create stage metrics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS stage_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    qid TEXT NOT NULL,
                    stage_name TEXT NOT NULL,
                    duration REAL,
                    llm_calls INTEGER,
                    llm_tokens_input INTEGER,
                    llm_tokens_output INTEGER,
                    memory_start_mb REAL,
                    memory_end_mb REAL,
                    memory_peak_mb REAL,
                    additional_info TEXT,  -- JSON string
                    timestamp TEXT,
                    FOREIGN KEY (qid) REFERENCES question_records (qid)
                )
            """)
            
            conn.commit()
    
    def start_question(self, qid: str, question: str, question_type: str = "unknown"):
        """start monitoring a new question"""
        with self.lock:
            self.current_question = {
                'qid': qid,
                'question': question,
                'question_type': question_type,
                'start_time': time.time(),
                'start_memory': self._get_memory_usage(),
                'stages': {}
            }
            print(f"🔍 start monitoring question {qid}: {question[:50]}...")
    
    @contextmanager
    def monitor_stage(self, stage_name: str):
        """monitor context manager for a single stage"""
        if not self.current_question:
            raise ValueError("No active question monitoring. Call start_question() first.")
        
        with self.lock:
            self.current_stage = stage_name
            self.stage_start_time = time.time()
            self.stage_start_memory = self._get_memory_usage()
            self.llm_calls = 0
            self.llm_tokens_input = 0
            self.llm_tokens_output = 0
        
        try:
            yield self
        finally:
            with self.lock:
                self._end_stage()
    
    def _end_stage(self):
        """end current stage"""
        if not self.current_stage or not self.current_question:
            return
        
        end_time = time.time()
        end_memory = self._get_memory_usage()
        
        metrics = StageMetrics(
            stage_name=self.current_stage,
            start_time=self.stage_start_time,
            end_time=end_time,
            duration=end_time - self.stage_start_time,
            llm_calls=self.llm_calls,
            llm_tokens_input=self.llm_tokens_input,
            llm_tokens_output=self.llm_tokens_output,
            memory_start_mb=self.stage_start_memory,
            memory_end_mb=end_memory,
            memory_peak_mb=max(self.stage_start_memory, end_memory)
        )

        self.current_question['stages'][self.current_stage] = metrics
        print(f"✅ stage {self.current_stage} completed: {metrics.duration:.3f}s, {self.llm_calls} LLM calls, {self.llm_tokens_input + self.llm_tokens_output} tokens")
        
        self.current_stage = None
    
    def record_llm_call(self, input_tokens: int = 0, output_tokens: int = 0):
        """record LLM call"""
        with self.lock:
            self.llm_calls += 1
            self.llm_tokens_input += input_tokens
            self.llm_tokens_output += output_tokens
    
    def set_decomposition_info(self, original_question: str, subquestions: List[str], 
                              decomposition_method: str = "unknown"):
        """设置问题分解信息"""
        if not self.current_question:
            return
        
        with self.lock:
            self.current_question['decomposition_info'] = DecompositionInfo(
                original_question=original_question,
                subquestions=subquestions,
                branch_count=len(subquestions),
                depth=self._calculate_depth(subquestions),
                decomposition_method=decomposition_method
            )
    
    def _calculate_depth(self, subquestions: List[str]) -> int:
        """Calculate decomposition depth (simplified version)"""
        # here we can calculate depth according to the actual decomposition logic
        # currently simplified to 1, representing the first layer of decomposition
        return 1
    
    def end_question(self, is_correct: bool, final_answer: str, 
                    answer_path: List[Dict[str, Any]] = None):
        """End question monitoring and save record"""
        if not self.current_question:
            return
        
        with self.lock:
            end_time = time.time()
            total_duration = end_time - self.current_question['start_time']
            total_memory_peak = max([stage.memory_peak_mb for stage in self.current_question['stages'].values()] + [self.current_question['start_memory']])
            
            # calculate total LLM metrics
            total_llm_calls = sum(stage.llm_calls for stage in self.current_question['stages'].values())
            total_llm_tokens_input = sum(stage.llm_tokens_input for stage in self.current_question['stages'].values())
            total_llm_tokens_output = sum(stage.llm_tokens_output for stage in self.current_question['stages'].values())
            
            # create question record
            record = QuestionRecord(
                qid=self.current_question['qid'],
                question=self.current_question['question'],
                question_type=self.current_question['question_type'],
                is_correct=is_correct,
                final_answer=final_answer,
                answer_path=answer_path or [],
                total_duration=total_duration,
                total_llm_calls=total_llm_calls,
                total_llm_tokens_input=total_llm_tokens_input,
                total_llm_tokens_output=total_llm_tokens_output,
                total_memory_peak_mb=total_memory_peak,
                classification_metrics=self.current_question['stages'].get('classification'),
                decomposition_metrics=self.current_question['stages'].get('decomposition'),
                retrieval_metrics=self.current_question['stages'].get('retrieval'),
                pruning_metrics=self.current_question['stages'].get('pruning'),
                qa_metrics=self.current_question['stages'].get('qa'),
                decomposition_info=self.current_question.get('decomposition_info'),
                timestamp=datetime.now().isoformat()
            )
            
            # save to database
            self._save_question_record(record)
            
            print(f"📊 question {self.current_question['qid']} monitoring completed:")
            print(f"   total time: {total_duration:.3f}s")
            print(f"   Token consumption: {total_llm_tokens_input + total_llm_tokens_output}")
            print(f"   Memory peak: {total_memory_peak:.1f}MB")
            print(f"   correctness: {'✅' if is_correct else '❌'}")
            
            self.current_question = None
    
    def _save_question_record(self, record: QuestionRecord):
        """Save question record to database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # check if record already exists
            cursor.execute("SELECT qid FROM question_records WHERE qid = ?", (record.qid,))
            exists = cursor.fetchone()
            
            if exists:
                # if previous was wrong, now is correct, then update
                cursor.execute("SELECT is_correct FROM question_records WHERE qid = ?", (record.qid,))
                old_correct = cursor.fetchone()[0]
                
                if not old_correct and record.is_correct:
                    print(f"🔄 update question {record.qid} record (from wrong to correct)")
                    self._update_question_record(cursor, record)
                else:
                    print(f"⏭️ skip question {record.qid} (already exists and state not changed)")
            else:
                # new record
                print(f"💾 save new question {record.qid} record")
                self._insert_question_record(cursor, record)
            
            conn.commit()
    
    def _insert_question_record(self, cursor, record: QuestionRecord):
        """Insert new question record"""
        cursor.execute("""
            INSERT INTO question_records (
                qid, question, question_type, is_correct, final_answer, answer_path,
                total_duration, total_llm_calls, total_llm_tokens_input, total_llm_tokens_output, total_memory_peak_mb,
                decomposition_info, timestamp, version,
                classification_metrics, decomposition_metrics, retrieval_metrics, pruning_metrics, qa_metrics
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            record.qid, record.question, record.question_type, record.is_correct, 
            json.dumps(record.final_answer), json.dumps(record.answer_path),
            record.total_duration, record.total_llm_calls, record.total_llm_tokens_input, 
            record.total_llm_tokens_output, record.total_memory_peak_mb,
            json.dumps(asdict(record.decomposition_info)) if record.decomposition_info else None,
            record.timestamp, record.version,
            json.dumps(asdict(record.classification_metrics)) if record.classification_metrics else None,
            json.dumps(asdict(record.decomposition_metrics)) if record.decomposition_metrics else None,
            json.dumps(asdict(record.retrieval_metrics)) if record.retrieval_metrics else None,
            json.dumps(asdict(record.pruning_metrics)) if record.pruning_metrics else None,
            json.dumps(asdict(record.qa_metrics)) if record.qa_metrics else None
        ))
    
    def _update_question_record(self, cursor, record: QuestionRecord):
        """Update question record"""
        cursor.execute("""
            UPDATE question_records SET
                is_correct = ?, final_answer = ?, answer_path = ?,
                total_duration = ?, total_llm_calls = ?, total_llm_tokens_input = ?, 
                total_llm_tokens_output = ?, total_memory_peak_mb = ?,
                decomposition_info = ?, timestamp = ?, version = ?,
                classification_metrics = ?, decomposition_metrics = ?, retrieval_metrics = ?, 
                pruning_metrics = ?, qa_metrics = ?
            WHERE qid = ?
        """, (
            record.is_correct, json.dumps(record.final_answer), json.dumps(record.answer_path),
            record.total_duration, record.total_llm_calls, record.total_llm_tokens_input, 
            record.total_llm_tokens_output, record.total_memory_peak_mb,
            json.dumps(asdict(record.decomposition_info)) if record.decomposition_info else None,
            record.timestamp, record.version,
            json.dumps(asdict(record.classification_metrics)) if record.classification_metrics else None,
            json.dumps(asdict(record.decomposition_metrics)) if record.decomposition_metrics else None,
            json.dumps(asdict(record.retrieval_metrics)) if record.retrieval_metrics else None,
            json.dumps(asdict(record.pruning_metrics)) if record.pruning_metrics else None,
            json.dumps(asdict(record.qa_metrics)) if record.qa_metrics else None,
            record.qid
        ))
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage (MB)"""
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except:
            return 0.0
    
    def get_question_record(self, qid: str) -> Optional[QuestionRecord]:
        """Get question record"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM question_records WHERE qid = ?", (qid,))
            row = cursor.fetchone()
            
            if not row:
                return None
            
            # rebuild record object
            return self._row_to_question_record(row)
    
    def _row_to_question_record(self, row) -> QuestionRecord:
        """Convert database row to QuestionRecord object"""
        # here we need to implement according to the actual database structure
        # simplified version
    pass
    
    def get_all_records(self) -> List[QuestionRecord]:
        """Get all question records"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM question_records ORDER BY timestamp DESC")
            rows = cursor.fetchall()
            
            records = []
            for row in rows:
                record = self._row_to_question_record(row)
                if record:
                    records.append(record)
            
            return records

# global monitor instance
_global_monitor = None

def get_performance_monitor() -> PerformanceMonitor:
    """Get global performance monitor instance"""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = PerformanceMonitor()
    return _global_monitor

# convenient function
def start_question_monitoring(qid: str, question: str, question_type: str = "unknown"):
    """Start question monitoring"""
    monitor = get_performance_monitor()
    monitor.start_question(qid, question, question_type)

def end_question_monitoring(is_correct: bool, final_answer: str, answer_path: List[Dict[str, Any]] = None):
    """End question monitoring"""
    monitor = get_performance_monitor()
    monitor.end_question(is_correct, final_answer, answer_path)

def record_llm_call(input_tokens: int = 0, output_tokens: int = 0):
    """Record LLM call"""
    monitor = get_performance_monitor()
    monitor.record_llm_call(input_tokens, output_tokens)

def set_decomposition_info(original_question: str, subquestions: List[str], decomposition_method: str = "unknown"):
    """Set decomposition information"""
    monitor = get_performance_monitor()
    monitor.set_decomposition_info(original_question, subquestions, decomposition_method)