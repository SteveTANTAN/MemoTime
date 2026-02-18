#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TPKG System Unified Configuration
TPKG system unified configuration file - convenient for experiments and tuning
"""

import os
from typing import Dict, Any
from pathlib import Path

class TPKGConfig:
    """TPKG system unified configuration"""
    
    # Project root: MemoTime folder (one level up from memotime/ folder)
    PROJECT_ROOT = Path(__file__).parent.parent.absolute()
    DATA_ROOT = PROJECT_ROOT / "Data"
    
    # ========== dataset configuration ==========  
    
    # Current used dataset: TimeQuestions or MultiTQ
    DATASET = os.getenv("TPKG_DATASET", "MultiTQ")
    
    # Dataset configuration mapping (using relative paths from project root)
    DATASET_CONFIGS = {
        "TimeQuestions": {
            "root": str(DATA_ROOT / "TimeQuestions"),
            "db_path": str(DATA_ROOT / "TimeQuestions" / "tempkg_timequestions.db"),
            "test_data_path": str(DATA_ROOT / "TimeQuestions" / "questions_with_candidates_timequestions.json"),
            "raw_test_data_path": str(DATA_ROOT / "TimeQuestions" / "test.json"),
            "entity_index_dir": str(DATA_ROOT / "TimeQuestions" / "entity_index_timequestions"),
            "question_id_field": "Id",
            "question_text_field": "Question",
            "answer_field": "Answer",
            "kg_format": "5cols"  # subject, relation, object, start_time, end_time
        },
        "MultiTQ": {
            "root": str(DATA_ROOT / "MultiTQ"),
            "db_path": str(DATA_ROOT / "MultiTQ" / "tempkg_multitq.db"),
            "test_data_path": str(DATA_ROOT / "MultiTQ" / "questions_with_candidates_multitq.json"),
            "raw_test_data_path": str(DATA_ROOT / "MultiTQ" / "test.json"),
            "entity_index_dir": str(DATA_ROOT / "MultiTQ" / "entity_index_multitq"),
            "question_id_field": "quid",
            "question_text_field": "question",
            "answer_field": "answers",
            "kg_format": "4cols"  # subject, relation, object, time
        }
    }
    
    # Get configuration according to current dataset
    @classmethod
    def get_dataset_config(cls):
        """Get configuration according to current dataset"""
        if cls.DATASET not in cls.DATASET_CONFIGS:
            raise ValueError(f"Unknown dataset: {cls.DATASET}. Supported datasets: {list(cls.DATASET_CONFIGS.keys())}")
        return cls.DATASET_CONFIGS[cls.DATASET]
    
    # ========== basic configuration (dynamic according to dataset) ==========
    
    # Database path (dynamic)
    @classmethod
    def get_db_path(cls):
        return cls.get_dataset_config()["db_path"]
    
    # Test data path (dynamic)
    @classmethod
    def get_test_data_path(cls):
        return cls.get_dataset_config()["test_data_path"]
    
    # Raw test data path (without candidates)
    @classmethod
    def get_raw_test_data_path(cls):
        return cls.get_dataset_config()["raw_test_data_path"]
    
    # Entity index directory (dynamic)
    @classmethod
    def get_entity_index_dir(cls):
        return cls.get_dataset_config()["entity_index_dir"]
    
    # Hybrid retrieval cache directory (dynamic, isolated by dataset)
    @classmethod
    def get_hybrid_cache_dir(cls):
        """Get Hybrid retrieval cache directory (isolated by dataset)"""
        dataset_root = cls.get_dataset_config()["root"]
        return os.path.join(dataset_root, "hybrid_cache")  # use independent hybrid_cache directory
    
    # For backward compatibility, keep the original attribute names
    DB_PATH = None  # Will be dynamically set in apply_config
    TEST_DATA_PATH = None  # Will be dynamically set in apply_config
    ENTITY_INDEX_DIR = None  # Will be dynamically set in apply_config
    HYBRID_CACHE_DIR = None  # Will be dynamically set in apply_config
    
    # ========== problem solving configuration ==========
    
    # Maximum retries (original question level)
    MAX_RETRIES = int(os.getenv("MAX_RETRIES", "2"))
    
    # Maximum depth (subproblem recursion depth)
    MAX_DEPTH = int(os.getenv("MAX_DEPTH", "3"))
    
    # Maximum total branches (maximum branches of problem decomposition)
    MAX_TOTAL_BRANCHES = int(os.getenv("MAX_TOTAL_BRANCHES", "20"))
    
    # Maximum retries of subproblems
    MAX_SUBQ_RETRIES = int(os.getenv("MAX_SUBQ_RETRIES", "2"))
    
    # ========== LLM configuration ==========
    
    # Default LLM model
    DEFAULT_LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
    
    # LLM model used for classification
    CLASSIFICATION_LLM_MODEL = os.getenv("CLASSIFICATION_LLM_MODEL", DEFAULT_LLM_MODEL)
    
    # LLM model used for decomposition
    DECOMPOSITION_LLM_MODEL = os.getenv("DECOMPOSITION_LLM_MODEL", DEFAULT_LLM_MODEL)
    
    # LLM model used for seed selection
    SEED_SELECTION_LLM_MODEL = os.getenv("SEED_SELECTION_LLM_MODEL", DEFAULT_LLM_MODEL)
    
    # LLM model used for toolkit selection
    TOOLKIT_SELECTION_LLM_MODEL = os.getenv("TOOLKIT_SELECTION_LLM_MODEL", DEFAULT_LLM_MODEL)
    
    # LLM model used for path selection
    PATH_SELECTION_LLM_MODEL = os.getenv("PATH_SELECTION_LLM_MODEL", DEFAULT_LLM_MODEL)
    
    # LLM model used for sufficiency verification
    SUFFICIENCY_LLM_MODEL = os.getenv("SUFFICIENCY_LLM_MODEL", DEFAULT_LLM_MODEL)
    
    # LLM model used for final answer generation
    FINAL_ANSWER_LLM_MODEL = os.getenv("FINAL_ANSWER_LLM_MODEL", DEFAULT_LLM_MODEL)
    
    # ========== retrieval configuration ==========
    
    # Whether to use hybrid retrieval
    USE_HYBRID_RETRIEVAL = os.getenv("USE_HYBRID_RETRIEVAL", "true").lower() == "true"
    
    # Hybrid retrieval model
    HYBRID_RETRIEVAL_MODEL = os.getenv("HYBRID_RETRIEVAL_MODEL", "BAAI/bge-m3")
    
    # Semantic embedding model
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    
    # Retrieval result number limit
    RETRIEVAL_LIMIT = int(os.getenv("RETRIEVAL_LIMIT", "200"))
    

    
    # ========== storage mode configuration ==========
    
    # Storage mode: shared(shared) or individual(independent)
    STORAGE_MODE = os.getenv("TPKG_STORAGE_MODE", "shared")
    
    # Whether to enable shared storage fallback (also access shared data in independent mode)
    ENABLE_SHARED_FALLBACK = os.getenv("ENABLE_SHARED_FALLBACK", "false").lower() == "true"
    
    # Configuration name (used for database path)
    CONFIG_NAME = os.getenv("TPKG_CONFIG_NAME", None)
    
    # Data directory
    TPKG_DATA_DIR = os.getenv("TPKG_DATA_DIR", str(DATA_ROOT))
    
    # ========== unified knowledge store configuration ==========    
    
    # Whether to enable unified knowledge store
    USE_UNIFIED_KNOWLEDGE_STORE = os.getenv("USE_UNIFIED_KNOWLEDGE_STORE", "true").lower() == "true"
    
    # Unified knowledge store database path (dynamic according to dataset)
    @classmethod
    def get_unified_knowledge_path(cls, config_name=None):
        """Get unified knowledge store path (isolated by dataset)"""
        dataset_root = cls.get_dataset_config()["root"]
        if config_name:
            return os.path.join(dataset_root, f"unified_knowledge_{config_name}", "unified_knowledge.db")
        else:
            return os.path.join(dataset_root, "unified_knowledge", "unified_knowledge.db")
    
    UNIFIED_KNOWLEDGE_DB = None  # Will be dynamically set in apply_config
    
    # Unified knowledge store similarity threshold
    UNIFIED_KNOWLEDGE_SIM_THRESHOLD = float(os.getenv("UNIFIED_KNOWLEDGE_SIM_THRESHOLD", "0.7"))
    
    # Unified knowledge store maximum candidates
    UNIFIED_KNOWLEDGE_MAX_CANDIDATES = int(os.getenv("UNIFIED_KNOWLEDGE_MAX_CANDIDATES", "5"))
    
    # Unified knowledge store cache size
    UNIFIED_KNOWLEDGE_CACHE_SIZE = int(os.getenv("UNIFIED_KNOWLEDGE_CACHE_SIZE", "1000"))
    
    # Unified knowledge store buffer size
    UNIFIED_KNOWLEDGE_BUFFER_SIZE = int(os.getenv("UNIFIED_KNOWLEDGE_BUFFER_SIZE", "100"))
    
    # Unified knowledge store cleanup strategy
    UNIFIED_KNOWLEDGE_CLEANUP_DAYS = int(os.getenv("UNIFIED_KNOWLEDGE_CLEANUP_DAYS", "30"))
    UNIFIED_KNOWLEDGE_MIN_HITS = int(os.getenv("UNIFIED_KNOWLEDGE_MIN_HITS", "2"))
    
        # ========== performance monitoring configuration ==========
    
    # Performance database path
    PERFORMANCE_DB = os.getenv("PERFORMANCE_DB", 
                              str(PROJECT_ROOT / "memotime" / "data" / "performance.db"))
    
    # Whether to enable performance monitoring
    ENABLE_PERFORMANCE_MONITORING = os.getenv("ENABLE_PERFORMANCE_MONITORING", "true").lower() == "true"
    
    # ========== experiment configuration ==========
    
    # Number of questions to process
    MAX_QUESTIONS = int(os.getenv("MAX_QUESTIONS", "200"))
    
    # Skip the first N questions
    SKIP_QUESTIONS = int(os.getenv("SKIP_QUESTIONS", "0"))
    
    # Filter question type (empty means process all types)
    FILTER_QUESTION_TYPE = os.getenv("FILTER_QUESTION_TYPE", "")  # 例如: "after_first", "equal", "before_last"
    
    # Maximum number of candidate entities
    MAX_CANDIDATE_ENTITIES = int(os.getenv("MAX_CANDIDATE_ENTITIES", "10"))
    
    # ========== debug configuration ==========
    
    # Whether to enable detailed logging
    VERBOSE = os.getenv("VERBOSE", "false").lower() == "true"
    
    # Whether to enable debug mode
    DEBUG = os.getenv("DEBUG", "false").lower() == "false"
    
    # ========== tool methods ==========
    
    @classmethod
    def get_all_config(cls) -> Dict[str, Any]:
        """Get all configurations"""
        return {
            "basic configuration": {
                "db_path": cls.DB_PATH,
                "test_data_path": cls.TEST_DATA_PATH,
            },
            "problem solving configuration": {
                "max_retries": cls.MAX_RETRIES,
                "max_depth": cls.MAX_DEPTH,
                "max_total_branches": cls.MAX_TOTAL_BRANCHES,
                "max_subq_retries": cls.MAX_SUBQ_RETRIES,
            },
            "LLM configuration": {
                "default_model": cls.DEFAULT_LLM_MODEL,
                "classification_model": cls.CLASSIFICATION_LLM_MODEL,
                "decomposition_model": cls.DECOMPOSITION_LLM_MODEL,
                "seed_selection_model": cls.SEED_SELECTION_LLM_MODEL,
                "toolkit_selection_model": cls.TOOLKIT_SELECTION_LLM_MODEL,
                "path_selection_model": cls.PATH_SELECTION_LLM_MODEL,
                "sufficiency_model": cls.SUFFICIENCY_LLM_MODEL,
                "final_answer_model": cls.FINAL_ANSWER_LLM_MODEL,
            },
            "retrieval configuration": {
                "use_hybrid_retrieval": cls.USE_HYBRID_RETRIEVAL,
                "hybrid_model": cls.HYBRID_RETRIEVAL_MODEL,
                "embedding_model": cls.EMBEDDING_MODEL,
                "retrieval_limit": cls.RETRIEVAL_LIMIT,
            },
            # Template Learning and experience pool configuration removed, using unified knowledge store
            "unified knowledge store configuration": {
                "use_unified_knowledge_store": cls.USE_UNIFIED_KNOWLEDGE_STORE,
                "db_path": cls.UNIFIED_KNOWLEDGE_DB,
                "sim_threshold": cls.UNIFIED_KNOWLEDGE_SIM_THRESHOLD,
                "max_candidates": cls.UNIFIED_KNOWLEDGE_MAX_CANDIDATES,
                "cache_size": cls.UNIFIED_KNOWLEDGE_CACHE_SIZE,
                "buffer_size": cls.UNIFIED_KNOWLEDGE_BUFFER_SIZE,
                "cleanup_days": cls.UNIFIED_KNOWLEDGE_CLEANUP_DAYS,
                "min_hits": cls.UNIFIED_KNOWLEDGE_MIN_HITS,
            },
            "performance monitoring configuration": {
                "enable_monitoring": cls.ENABLE_PERFORMANCE_MONITORING,
                "db_path": cls.PERFORMANCE_DB,
            },
            "storage configuration": {
                "storage_mode": cls.STORAGE_MODE,
                "enable_shared_fallback": cls.ENABLE_SHARED_FALLBACK,
                "tpkg_data_dir": cls.TPKG_DATA_DIR,
            },
            "experiment configuration": {
                "max_questions": cls.MAX_QUESTIONS,
                "skip_questions": cls.SKIP_QUESTIONS,
                "filter_question_type": cls.FILTER_QUESTION_TYPE,
                "max_candidate_entities": cls.MAX_CANDIDATE_ENTITIES,
            },
            "debug configuration": {
                "verbose": cls.VERBOSE,
                "debug": cls.DEBUG,
            }
        }
    
    @classmethod
    def print_config(cls):
        """Print all configurations"""
        print("=" * 80)
        print("TPKG system configuration")
        print("=" * 80)
        
        all_config = cls.get_all_config()
        for category, configs in all_config.items():
            print(f"\n📋 {category}:")
            for key, value in configs.items():
                print(f"   {key}: {value}")
    
    @classmethod
    def save_config_to_file(cls, filepath: str = "current_config.json"):
        """Save current configuration to file"""
        import json
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(cls.get_all_config(), f, ensure_ascii=False, indent=2)
        print(f"✅ Configuration saved to {filepath}")
    
    @classmethod
    def create_experiment_config(cls, 
                                 experiment_name: str,
                                 **kwargs) -> Dict[str, Any]:
        """
        Create experiment configuration
        
        Args:
            experiment_name: Experiment name
            **kwargs: Configuration items to override
            
        Returns:
            Experiment configuration dictionary
        """
        config = cls.get_all_config()
        
        # Apply overrides
        for key, value in kwargs.items():
            for category in config.values():
                if key in category:
                    category[key] = value
        
        config["experiment_name"] = experiment_name
        config["timestamp"] = __import__('datetime').datetime.now().isoformat()
        
        return config

# ========== predefined experiment configuration ==========

class ExperimentPresets:
    """Predefined experiment configuration"""
    
    @staticmethod
    def baseline():
        """Baseline configuration: no optimization"""
        return {
            "use_hybrid_retrieval": False,
            "use_template_learning": False,
            "use_experience_pool": False,
            "max_retries": 0,
            "max_depth": 1,
        }
    
    @staticmethod
    def full_optimization():
        """Full optimization: enable all features"""
        return {
            "use_hybrid_retrieval": True,
            "use_template_learning": True,
            "use_experience_pool": True,
            "max_retries": 1,
            "max_depth": 2,
        }
    
    @staticmethod
    def fast_mode():
        """Fast mode: prioritize speed"""
        return {
            "use_hybrid_retrieval": False,
            "use_template_learning": True,
            "use_experience_pool": True,
            "retrieval_limit": 100,
            "max_retries": 0,
        }
    
    @staticmethod
    def accuracy_mode():
        """Accuracy mode: prioritize accuracy"""
        return {
            "use_hybrid_retrieval": True,
            "use_template_learning": True,
            "use_experience_pool": True,
            "retrieval_limit": 300,
            "max_retries": 2,
            "max_depth": 2,
        }
    
    @staticmethod
    def equal_type_test():
        """Equal type problem test"""
        return {
            "filter_question_type": "equal",
            "use_hybrid_retrieval": True,
            "use_experience_pool": True,
            "max_questions": 30,
        }
    
    @staticmethod
    def experience_pool_test():
        """Experience pool effect test"""
        return {
            "use_experience_pool": True,
            "exp_pool_sim_threshold": 0.83,
            "max_questions": 100,
            "skip_questions": 0,
        }

    # ========== convenient functions ==========

def apply_config(config_dict: Dict[str, Any]):
    """
    Apply configuration dictionary to TPKGConfig class
    
    Args:
        config_dict: Configuration dictionary
    """
    # Special handling: dataset configuration (must be processed first, because other paths depend on it)
    if 'dataset' in config_dict:
        dataset = config_dict['dataset']
        if dataset not in TPKGConfig.DATASET_CONFIGS:
            raise ValueError(f"Unknown dataset: {dataset}. Supported datasets: {list(TPKGConfig.DATASET_CONFIGS.keys())}")
        TPKGConfig.DATASET = dataset
        print(f"✅ Switch to dataset: {dataset}")
        
        # Update all dataset-related paths
        TPKGConfig.DB_PATH = TPKGConfig.get_db_path()
        TPKGConfig.TEST_DATA_PATH = TPKGConfig.get_test_data_path()
        TPKGConfig.ENTITY_INDEX_DIR = TPKGConfig.get_entity_index_dir()
        TPKGConfig.HYBRID_CACHE_DIR = TPKGConfig.get_hybrid_cache_dir()
        print(f"   - Database: {TPKGConfig.DB_PATH}")
        print(f"   - Question file: {TPKGConfig.TEST_DATA_PATH}")
        print(f"   - Entity index: {TPKGConfig.ENTITY_INDEX_DIR}")
        print(f"   - Hybrid cache: {TPKGConfig.HYBRID_CACHE_DIR}")
    
    for key, value in config_dict.items():
        if hasattr(TPKGConfig, key.upper()):
            setattr(TPKGConfig, key.upper(), value)
    
    # Special handling: if DEFAULT_LLM_MODEL is modified, update all other LLM configurations
    if 'default_llm_model' in config_dict:
        new_model = config_dict['default_llm_model']
        TPKGConfig.DEFAULT_LLM_MODEL = new_model
        TPKGConfig.CLASSIFICATION_LLM_MODEL = new_model
        TPKGConfig.DECOMPOSITION_LLM_MODEL = new_model
        TPKGConfig.SEED_SELECTION_LLM_MODEL = new_model
        TPKGConfig.TOOLKIT_SELECTION_LLM_MODEL = new_model
        TPKGConfig.PATH_SELECTION_LLM_MODEL = new_model
        TPKGConfig.SUFFICIENCY_LLM_MODEL = new_model
        TPKGConfig.FINAL_ANSWER_LLM_MODEL = new_model
        print(f"✅ All LLM configurations updated to: {new_model}")
    
    # Special handling: storage mode configuration
    if 'storage_mode' in config_dict:
        from kg_agent.storage_manager import set_storage_mode, StorageMode
        mode = StorageMode.SHARED if config_dict['storage_mode'] == 'shared' else StorageMode.INDIVIDUAL
        set_storage_mode(mode)
    
    if 'enable_shared_fallback' in config_dict:
        TPKGConfig.ENABLE_SHARED_FALLBACK = config_dict['enable_shared_fallback']
    
    if 'config_name' in config_dict:
        TPKGConfig.CONFIG_NAME = config_dict['config_name']
    
    # Update unified knowledge store path (isolated by dataset)
    config_name = config_dict.get('config_name', TPKGConfig.CONFIG_NAME)
    TPKGConfig.UNIFIED_KNOWLEDGE_DB = TPKGConfig.get_unified_knowledge_path(config_name)
    
    # Set storage manager
    if any(key in config_dict for key in ['storage_mode', 'enable_shared_fallback', 'config_name', 'preset', 'max_retries', 'max_depth']):
        from kg_agent.storage_manager import set_current_experiment_setting, create_experiment_setting_from_config
        experiment_setting = create_experiment_setting_from_config(config_dict)
        set_current_experiment_setting(experiment_setting)
    
    # Special handling: unified knowledge store configuration
    if 'use_unified_knowledge_store' in config_dict:
        TPKGConfig.USE_UNIFIED_KNOWLEDGE_STORE = config_dict['use_unified_knowledge_store']

def load_preset(preset_name: str):
    """
    Load predefined configuration
    
    Args:
        preset_name: Preset name
    """
    presets = {
        # Basic configuration
        "baseline": ExperimentPresets.baseline(),
        "full_optimization": ExperimentPresets.full_optimization(),
        "fast_mode": ExperimentPresets.fast_mode(),
        "accuracy_mode": ExperimentPresets.accuracy_mode(),
        "equal_type_test": ExperimentPresets.equal_type_test(),
        "experience_pool_test": ExperimentPresets.experience_pool_test(),
        
        # New descriptive configuration
        "retry2_hybrid_pool_template_unified_shared_1q": {
            "max_retries": 2,
            "use_hybrid_retrieval": True,
            "use_experience_pool": True,
            "use_template_learning": True,
            "use_unified_knowledge_store": True,
            "storage_mode": "shared",
            "max_questions": 1,
            "skip_questions": 0,
            "max_depth": 1,
            "max_total_branches": 5,
            "default_llm_model": "gpt-4o-mini"
        },
        
        "retry2_hybrid_pool_template_individual_1q": {
            "max_retries": 2,
            "use_hybrid_retrieval": True,
            "use_experience_pool": True,
            "use_template_learning": True,
            "use_unified_knowledge_store": True,
            "storage_mode": "individual",
            "max_questions": 1,
            "skip_questions": 0,
            "max_depth": 1,
            "max_total_branches": 5,
            "default_llm_model": "gpt-4o-mini"
        },
        
        "retry3_hybrid_pool_template_unified_shared_5q": {
            "max_retries": 3,
            "use_hybrid_retrieval": True,
            "use_experience_pool": True,
            "use_template_learning": True,
            "use_unified_knowledge_store": True,
            "storage_mode": "shared",
            "max_questions": 5,
            "skip_questions": 0,
            "max_depth": 2,
            "max_total_branches": 10,
            "default_llm_model": "gpt-4o-mini"
        },
        
        "retry1_fast_mode_shared_1q": {
            "max_retries": 1,
            "use_hybrid_retrieval": False,
            "use_experience_pool": False,
            "use_template_learning": False,
            "use_unified_knowledge_store": False,
            "storage_mode": "shared",
            "max_questions": 1,
            "skip_questions": 0,
            "max_depth": 1,
            "max_total_branches": 3,
            "default_llm_model": "gpt-4o-mini"
        },
        
        "retry2_accuracy_mode_individual_10q": {
            "max_retries": 2,
            "use_hybrid_retrieval": True,
            "use_experience_pool": True,
            "use_template_learning": True,
            "use_unified_knowledge_store": True,
            "storage_mode": "individual",
            "max_questions": 10,
            "skip_questions": 0,
            "max_depth": 3,
            "max_total_branches": 15,
            "default_llm_model": "gpt-4o"
        }
    }
    
    if preset_name in presets:
        apply_config(presets[preset_name])
        print(f"✅ Preset configuration loaded: {preset_name}")
        return presets[preset_name]
    else:
        print(f"❌ Unknown preset: {preset_name}")
        print(f"Available presets: {list(presets.keys())}")
        return None

if __name__ == "__main__":
    # Display current configuration
    TPKGConfig.print_config()
    
    # Display available presets
    print("\n" + "=" * 80)
    print("Available experiment presets:")
    print("=" * 80)
    presets = ["baseline", "full_optimization", "fast_mode", "accuracy_mode", 
               "equal_type_test", "experience_pool_test"]
    for preset in presets:
        print(f"   - {preset}")


