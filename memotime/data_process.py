import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
from config import TPKGConfig

def load_questions(file_path: str = None) -> Tuple[List[Dict[str, Any]], str, str, str]:
    """
    Load questions and candidates from a JSON file.
    
    Automatically load questions and candidates from a JSON file based on current dataset configuration.

    Args:
        file_path (str, optional): Path to the JSON file. 
                                   If None, use the path configured in TPKGConfig.

    Returns:
        tuple: (data, question_field, id_field, candidates_field)
            - data: Question list
            - question_field: Question text field name
            - id_field: Question ID field name
            - candidates_field: Candidate entity field name
    """
    # Get current dataset configuration
    dataset_config = TPKGConfig.get_dataset_config()
    
    # Use configured path (if not specified)
    if file_path is None:
        file_path = TPKGConfig.TEST_DATA_PATH
        
    # Special handling: backward compatibility
    if file_path == "MultiQA":
        file_path = str(Path(__file__).parent.parent / "Data" / "question_identi_V1.json")
    
    print(f"📁 Load questions file: {file_path}")
    
    # Load data
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Return correct field names based on dataset
    question_field = dataset_config["question_text_field"]
    id_field = dataset_config["question_id_field"]
    candidates_field = "candidates"
    
    print(f"✅ Successfully loaded {len(data)} questions")
    print(f"   Dataset: {TPKGConfig.DATASET}")  
    print(f"   Question field: {question_field}")
    print(f"   ID field: {id_field}")
    print(f"   Candidates field: {candidates_field}")
    
    return data, question_field, id_field, candidates_field
