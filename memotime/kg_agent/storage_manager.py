#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Storage Manager for TPKG System
Storage Manager for TPKG System
"""

import os
from pathlib import Path
import json
import hashlib
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum

class StorageMode(Enum):
    """Storage Mode"""
    SHARED = "shared"      # 共享模式：所有实验setting共用存储
    INDIVIDUAL = "individual"  # 独立模式：每个实验setting有独立的存储

@dataclass
class ExperimentSetting:
    """Experiment Setting Identifier"""
    preset: Optional[str] = None
    max_retries: Optional[int] = None
    max_depth: Optional[int] = None
    max_total_branches: Optional[int] = None
    use_hybrid_retrieval: Optional[bool] = None
    use_experience_pool: Optional[bool] = None
    use_template_learning: Optional[bool] = None
    default_llm_model: Optional[str] = None
    max_candidate_entities: Optional[int] = None
    config_name: Optional[str] = None  # configuration name, for readable path
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "preset": self.preset,
            "max_retries": self.max_retries,
            "max_depth": self.max_depth,
            "max_total_branches": self.max_total_branches,
            "use_hybrid_retrieval": self.use_hybrid_retrieval,
            "use_experience_pool": self.use_experience_pool,
            "use_template_learning": self.use_template_learning,
            "default_llm_model": self.default_llm_model,
            "max_candidate_entities": self.max_candidate_entities
        }
    
    def get_setting_id(self) -> str:
        """Get setting ID (for different experiment configurations)"""
        # if config_name is provided, use it as ID
        if self.config_name:
            return self.config_name
        
        # otherwise, generate hash ID based on configuration parameters
        # filter out None values and config_name field and sort
        filtered_dict = {k: v for k, v in self.to_dict().items() 
                        if v is not None and k != 'config_name'}
        
        # create stable string representation
        setting_str = json.dumps(filtered_dict, sort_keys=True)
        
        # generate short hash as ID
        return hashlib.md5(setting_str.encode()).hexdigest()[:12]

class StorageManager:
    """Storage Manager"""
    
    def __init__(self, base_dir: str = None, 
                 storage_mode: StorageMode = StorageMode.SHARED):
        """
        Initialize storage manager
        
        Args:
            base_dir: base data directory
            storage_mode: storage mode
        """
        if base_dir is None:
            # Use relative path from this file
            base_dir = str(Path(__file__).parent.parent.parent / "Data")
        self.base_dir = base_dir
        self.storage_mode = storage_mode
        self.current_setting: Optional[ExperimentSetting] = None
        
        # ensure base directory exists
        os.makedirs(base_dir, exist_ok=True)
    
    def set_experiment_setting(self, setting: ExperimentSetting):
        """Set current experiment setting"""
        self.current_setting = setting
    
    def get_storage_path(self, component: str, setting: Optional[ExperimentSetting] = None) -> str:
        """
        Get storage path
        
        Args:
            component: component name ('experience_pool', 'template_learning', 'learning_records')
            setting: experiment setting, if None, use current setting
            
        Returns:
            storage path
        """
        if setting is None:
            setting = self.current_setting
        
        if self.storage_mode == StorageMode.SHARED:
            # shared mode: all experiments use the same path
            return os.path.join(self.base_dir, component)
        else:
            # individual mode: each setting uses independent path
            if setting is None:
                # if no setting, use default path
                setting_id = "default"
            else:
                setting_id = setting.get_setting_id()
            
            return os.path.join(self.base_dir, f"{component}_{setting_id}")
    
    def get_experience_pool_path(self, setting: Optional[ExperimentSetting] = None) -> str:
        """Get experience pool storage path"""
        return self.get_storage_path("experience_pool", setting)
    
    def get_template_learning_path(self, setting: Optional[ExperimentSetting] = None) -> str:
        """Get template learning storage path"""
        return self.get_storage_path("template_learning", setting)
    
    def get_learning_records_path(self, setting: Optional[ExperimentSetting] = None) -> str:
        """Get learning records storage path"""
        return self.get_storage_path("learning_records", setting)
    
    def get_shared_storage_path(self, component: str) -> str:
        """Get shared storage path (regardless of current mode)"""
        return os.path.join(self.base_dir, component)
    
    def list_all_settings(self, component: str) -> List[str]:
        """
        List all setting IDs
        
        Args:
            component: component name
            
        Returns:
            setting IDs list
        """
        if self.storage_mode == StorageMode.SHARED:
            # shared mode has only one path
            path = self.get_shared_storage_path(component)
            if os.path.exists(path):
                return ["shared"]
            return []
        else:
            # individual mode: scan all setting directories
            settings = []
            base_path = self.base_dir
            if os.path.exists(base_path):
                for item in os.listdir(base_path):
                    if item.startswith(f"{component}_"):
                        setting_id = item[len(f"{component}_"):]
                        settings.append(setting_id)
            return settings
    
    def get_storage_info(self) -> Dict[str, Any]:
        """Get storage information"""
        info = {
            "storage_mode": self.storage_mode.value,
            "base_dir": self.base_dir,
            "current_setting": self.current_setting.to_dict() if self.current_setting else None
        }
        
        # count the data of each component
        components = ["experience_pool", "template_learning", "learning_records"]
        for component in components:
            settings = self.list_all_settings(component)
            info[f"{component}_settings"] = settings
            info[f"{component}_count"] = len(settings)
        
        return info

# global storage manager instance
_storage_manager = None
_global_storage_mode = None  # global storage mode setting

def get_storage_manager() -> StorageManager:
    """Get global storage manager instance"""
    global _storage_manager, _global_storage_mode
    if _storage_manager is None:
        # read the configuration from environment variables
        base_dir = os.getenv("TPKG_DATA_DIR", str(Path(__file__).parent.parent.parent / "Data"))
        
        # use the global storage mode setting, otherwise use the environment variable
        if _global_storage_mode is not None:
            storage_mode = _global_storage_mode
        else:
            storage_mode_str = os.getenv("TPKG_STORAGE_MODE", "shared")
            storage_mode = StorageMode.SHARED if storage_mode_str == "shared" else StorageMode.INDIVIDUAL
        
        _storage_manager = StorageManager(base_dir=base_dir, storage_mode=storage_mode)
    return _storage_manager

def set_storage_mode(mode: StorageMode):
    """Set storage mode"""
    global _storage_manager, _global_storage_mode
    
    # set global storage mode
    _global_storage_mode = mode
    
    # if the storage manager exists, update its storage mode
    if _storage_manager is not None:
        _storage_manager.storage_mode = mode
    else:
        # if the storage manager does not exist, force to recreate
        _storage_manager = None

def set_current_experiment_setting(setting: ExperimentSetting):
    """Set current experiment setting"""
    global _storage_manager
    if _storage_manager is None:
        _storage_manager = get_storage_manager()
    _storage_manager.set_experiment_setting(setting)

def create_experiment_setting_from_config(config_dict: Dict[str, Any]) -> ExperimentSetting:
    """Create experiment setting from configuration dictionary"""
    return ExperimentSetting(
        preset=config_dict.get("preset"),
        max_retries=config_dict.get("max_retries"),
        max_depth=config_dict.get("max_depth"),
        max_total_branches=config_dict.get("max_total_branches"),
        use_hybrid_retrieval=config_dict.get("use_hybrid_retrieval"),
        use_experience_pool=config_dict.get("use_experience_pool"),
        use_template_learning=config_dict.get("use_template_learning"),
        default_llm_model=config_dict.get("default_llm_model"),
        max_candidate_entities=config_dict.get("max_candidate_entities")
    )

if __name__ == "__main__":
    # test code
    manager = get_storage_manager()
    
    # test shared mode
    manager.storage_mode = StorageMode.SHARED
    setting = ExperimentSetting(preset="baseline", max_retries=1)
    manager.set_experiment_setting(setting)
    
    print("Shared mode test:")
    print(f"Experience pool path: {manager.get_experience_pool_path()}")
    print(f"Template learning path: {manager.get_template_learning_path()}")
    
    # test individual mode
    manager.storage_mode = StorageMode.INDIVIDUAL
    print("\nIndividual mode test:")
    print(f"Experience pool path: {manager.get_experience_pool_path()}")
    print(f"Template learning path: {manager.get_template_learning_path()}")
    
    print(f"\nStorage information: {json.dumps(manager.get_storage_info(), indent=2)}")
