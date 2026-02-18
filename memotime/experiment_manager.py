#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TPKG Experiment Manager
Automatically record experiment configuration, results and logs
"""

import os
import json
import datetime
import shutil
from pathlib import Path
from typing import Dict, Any, Optional

class ExperimentManager:
    """Experiment manager - automatically save configuration and results"""
    
    def __init__(self, experiments_root: str = None):
        """
        Initialize experiment manager
        
        Args:
            experiments_root: Experiment root directory, default is ./experiments
        """
        if experiments_root is None:
            experiments_root = os.path.join(
                os.path.dirname(__file__), 
                "..", 
                "experiments"
            )
        
        self.experiments_root = Path(experiments_root)
        self.experiments_root.mkdir(parents=True, exist_ok=True)
        
        self.current_experiment = None
        self.experiment_dir = None
        self.experiment_id = None
    
    def create_experiment(self, 
                         name: str = None,
                         description: str = "",
                         tags: list = None,
                         config: Dict[str, Any] = None) -> str:
        """
        Create new experiment
        
        Args:
            name: Experiment name (optional, default is timestamp)
            description: Experiment description
            tags: Experiment label list
            config: Configuration dictionary
            
        Returns:
            Experiment ID
        """
        # Generate experiment ID
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if name:
            # Clean name, remove special characters
            clean_name = "".join(c if c.isalnum() or c in "_-" else "_" for c in name)
            self.experiment_id = f"{timestamp}_{clean_name}"
        else:
            self.experiment_id = timestamp
        
        # Create experiment directory
        self.experiment_dir = self.experiments_root / self.experiment_id
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.experiment_dir / "logs").mkdir(exist_ok=True)
        (self.experiment_dir / "results").mkdir(exist_ok=True)
        (self.experiment_dir / "data").mkdir(exist_ok=True)
        
        # Save experiment metadata
        self.current_experiment = {
            "id": self.experiment_id,
            "name": name or timestamp,
            "description": description,
            "tags": tags or [],
            "created_at": datetime.datetime.now().isoformat(),
            "status": "running",
            "config": config or {},
        }
        
        self._save_metadata()
        
        print(f"✅ Create experiment: {self.experiment_id}")
        print(f"   Directory: {self.experiment_dir}")
        
        return self.experiment_id
    
    def save_config(self, config: Dict[str, Any]):
        """Save experiment configuration"""
        if not self.experiment_dir:
            raise RuntimeError("Please create experiment first")
        
        config_file = self.experiment_dir / "config.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        
        self.current_experiment["config"] = config
        self._save_metadata()
        
        print(f"💾 Configuration saved: {config_file}")
    
    def save_results(self, results: Dict[str, Any]):
        """Save experiment results"""
        if not self.experiment_dir:
            raise RuntimeError("Please create experiment first")
        
        results_file = self.experiment_dir / "results" / "summary.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"📊 Results saved: {results_file}")
    
    def save_detailed_results(self, question_id: str, result: Dict[str, Any]):
        """Save detailed results of a single question"""
        if not self.experiment_dir:
            raise RuntimeError("Please create experiment first")
        
        result_file = self.experiment_dir / "results" / f"q_{question_id}.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    
    def get_log_file(self) -> str:
        """Get log file path"""
        if not self.experiment_dir:
            raise RuntimeError("Please create experiment first")
        
        log_file = self.experiment_dir / "logs" / "experiment.log"
        return str(log_file)
    
    def complete_experiment(self, 
                           success: bool = True,
                           summary: Dict[str, Any] = None):
        """Mark experiment completed"""
        if not self.current_experiment:
            return
        
        self.current_experiment["status"] = "completed" if success else "failed"
        self.current_experiment["completed_at"] = datetime.datetime.now().isoformat()
        
        if summary:
            self.current_experiment["summary"] = summary
        
        self._save_metadata()
        
        print(f"✅ Experiment completed: {self.experiment_id}")
        print(f"   Status: {self.current_experiment['status']}")
    
    def _save_metadata(self):
        """Save experiment metadata"""
        if not self.experiment_dir:
            return
        
        metadata_file = self.experiment_dir / "experiment.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.current_experiment, f, ensure_ascii=False, indent=2)
    
    def copy_database(self, db_path: str, name: str = "database_snapshot.db"):
        """Copy database snapshot"""
        if not self.experiment_dir:
            raise RuntimeError("Please create experiment first")
        
        if os.path.exists(db_path):
            dest = self.experiment_dir / "data" / name
            shutil.copy2(db_path, dest)
            print(f"💾 Database snapshot saved: {dest}")
    
    @classmethod
    def list_experiments(cls, experiments_root: str = None) -> list:
        """List all experiments"""
        if experiments_root is None:
            experiments_root = os.path.join(
                os.path.dirname(__file__), 
                "..", 
                "experiments"
            )
        
        experiments_root = Path(experiments_root)
        if not experiments_root.exists():
            return []
        
        experiments = []
        for exp_dir in sorted(experiments_root.iterdir(), reverse=True):
            if exp_dir.is_dir():
                metadata_file = exp_dir / "experiment.json"
                if metadata_file.exists():
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        experiments.append(json.load(f))
        
        return experiments
    
    @classmethod
    def load_experiment(cls, experiment_id: str, experiments_root: str = None) -> Dict[str, Any]:
        """Load experiment data"""
        if experiments_root is None:
            experiments_root = os.path.join(
                os.path.dirname(__file__), 
                "..", 
                "experiments"
            )
        
        exp_dir = Path(experiments_root) / experiment_id
        if not exp_dir.exists():
            raise ValueError(f"Experiment not found: {experiment_id}")
        
        metadata_file = exp_dir / "experiment.json"
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # Load configuration
        config_file = exp_dir / "config.json"
        if config_file.exists():
            with open(config_file, 'r', encoding='utf-8') as f:
                metadata["config"] = json.load(f)
        
        # Load results
        results_file = exp_dir / "results" / "summary.json"
        if results_file.exists():
            with open(results_file, 'r', encoding='utf-8') as f:
                metadata["results"] = json.load(f)
        
        return metadata
    
    @classmethod
    def compare_experiments(cls, 
                           exp_ids: list, 
                           experiments_root: str = None) -> Dict[str, Any]:
        """Compare multiple experiments"""
        experiments = []
        for exp_id in exp_ids:
            try:
                exp = cls.load_experiment(exp_id, experiments_root)
                experiments.append(exp)
            except Exception as e:
                print(f"⚠️ Cannot load experiment {exp_id}: {e}")
        
        if not experiments:
            return {}
        
        # Extract key metrics for comparison
        comparison = {
            "experiments": [],
            "configs": {},
            "results": {}
        }
        
        for exp in experiments:
            exp_id = exp["id"]
            comparison["experiments"].append({
                "id": exp_id,
                "name": exp.get("name"),
                "created_at": exp.get("created_at"),
                "status": exp.get("status"),
            })
            
            # Compare configuration
            config = exp.get("config", {})
            for key, value in config.items():
                if key not in comparison["configs"]:
                    comparison["configs"][key] = {}
                comparison["configs"][key][exp_id] = value
            
            # Compare results
            summary = exp.get("summary", {})
            for key, value in summary.items():
                if key not in comparison["results"]:
                    comparison["results"][key] = {}
                comparison["results"][key][exp_id] = value
        
        return comparison


def print_experiment_summary(experiment_id: str):
    """Print experiment summary"""
    try:
        exp = ExperimentManager.load_experiment(experiment_id)
        
        print("\n" + "=" * 80)
        print(f"Experiment summary: {exp['name']} ({exp['id']})")
        print("=" * 80)
        
        print(f"\nCreated time: {exp.get('created_at')}")
        print(f"Status: {exp.get('status')}")
        if exp.get('description'):
            print(f"Description: {exp['description']}")
        if exp.get('tags'):
            print(f"Tags: {', '.join(exp['tags'])}")
        
        print("\nConfiguration:")
        config = exp.get('config', {})
        for key, value in sorted(config.items()):
            print(f"  {key}: {value}")
        
        if 'summary' in exp:
            print("\nResults summary:")
            for key, value in exp['summary'].items():
                print(f"  {key}: {value}")
        
        print("=" * 80 + "\n")
        
    except Exception as e:
        print(f"❌ Cannot load experiment: {e}")


def list_all_experiments():
    """List all experiments"""
    experiments = ExperimentManager.list_experiments()
    
    if not experiments:
        print("📭 No experiment records")
        return
    
    print("\n" + "=" * 80)
    print(f"Experiment list (共 {len(experiments)} 个)")
    print("=" * 80)
    
    for exp in experiments:
        status_emoji = "✅" if exp.get('status') == 'completed' else "🔄"
        print(f"\n{status_emoji} {exp['id']}")
        print(f"   Name: {exp.get('name', 'N/A')}")
        print(f"   Time: {exp.get('created_at', 'N/A')}")
        print(f"   Status: {exp.get('status', 'unknown')}")
        if exp.get('tags'):
            print(f"   Tags: {', '.join(exp['tags'])}")
        
        # Show key configuration
        config = exp.get('config', {})
        if config:
            key_params = ['max_retries', 'use_hybrid_retrieval', 'use_experience_pool']
            params_str = ", ".join([f"{k}={config.get(k)}" for k in key_params if k in config])
            if params_str:
                print(f"   Configuration: {params_str}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "list":
            list_all_experiments()
        
        elif command == "show" and len(sys.argv) > 2:
            print_experiment_summary(sys.argv[2])
        
        elif command == "compare" and len(sys.argv) > 2:
            exp_ids = sys.argv[2:]
            comparison = ExperimentManager.compare_experiments(exp_ids)
            print(json.dumps(comparison, ensure_ascii=False, indent=2))
        
        else:
            print("Usage:") 
            print("  python experiment_manager.py list              # List all experiments")
            print("  python experiment_manager.py show <exp_id>     # Show experiment details")
            print("  python experiment_manager.py compare <id1> <id2> ...  # Compare experiments")
    else:
        # Demo
        print("Experiment management system demo")
        list_all_experiments()


