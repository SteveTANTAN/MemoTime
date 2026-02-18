#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Installation Verification Script for MemoTime
Checks if all required dependencies are properly installed
"""

import sys
from typing import Dict, List, Tuple

def check_python_version() -> bool:
    """Check if Python version is 3.9 or higher"""
    version = sys.version_info
    if version.major >= 3 and version.minor >= 9:
        print(f"✓ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"✗ Python {version.major}.{version.minor}.{version.micro} (requires 3.9+)")
        return False

def check_dependencies() -> Dict[str, bool]:
    """Check all dependencies"""
    results = {}
    
    # Core dependencies
    dependencies = [
        ("numpy", "NumPy"),
        ("torch", "PyTorch"),
        ("openai", "OpenAI"),
        ("tqdm", "tqdm"),
        ("psutil", "psutil"),
    ]
    
    # NLP & Embeddings
    dependencies.extend([
        ("sentence_transformers", "Sentence Transformers"),
        ("transformers", "Transformers"),
        ("spacy", "spaCy"),
    ])
    
    # Vector search
    dependencies.extend([
        ("faiss", "FAISS"),
    ])
    
    # FlagEmbedding (special case)
    dependencies.append(("FlagEmbedding", "FlagEmbedding"))
    
    # Machine Learning
    dependencies.append(("sklearn", "scikit-learn"))
    
    # Optional but recommended
    optional_deps = [
        ("google.generativeai", "Google AI"),
        ("pandas", "Pandas"),
        ("datasets", "HuggingFace Datasets"),
    ]
    
    print("\n" + "="*60)
    print("REQUIRED DEPENDENCIES")
    print("="*60)
    
    for module_name, display_name in dependencies:
        try:
            module = __import__(module_name.split('.')[0])
            version = getattr(module, '__version__', 'installed')
            print(f"✓ {display_name:<30} {version}")
            results[module_name] = True
        except ImportError:
            print(f"✗ {display_name:<30} NOT INSTALLED")
            results[module_name] = False
    
    print("\n" + "="*60)
    print("OPTIONAL DEPENDENCIES")
    print("="*60)
    
    for module_name, display_name in optional_deps:
        try:
            module = __import__(module_name.split('.')[0])
            version = getattr(module, '__version__', 'installed')
            print(f"✓ {display_name:<30} {version}")
            results[module_name] = True
        except ImportError:
            print(f"○ {display_name:<30} Not installed (optional)")
            results[module_name] = False
    
    return results

def check_spacy_model() -> bool:
    """Check if spaCy English model is installed"""
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        print("✓ spaCy en_core_web_sm model")
        return True
    except:
        print("✗ spaCy en_core_web_sm model NOT INSTALLED")
        print("  Install with: python -m spacy download en_core_web_sm")
        return False

def check_gpu_support() -> None:
    """Check GPU support"""
    print("\n" + "="*60)
    print("GPU SUPPORT")
    print("="*60)
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.version.cuda}")
            print(f"✓ GPU device: {torch.cuda.get_device_name(0)}")
            print(f"✓ Number of GPUs: {torch.cuda.device_count()}")
        else:
            print("○ CUDA not available (using CPU)")
    except:
        print("✗ Cannot check CUDA status")
    
    try:
        import faiss
        # Try to check if GPU version is installed
        try:
            res = faiss.StandardGpuResources()
            print("✓ FAISS GPU support available")
        except:
            print("○ FAISS GPU support not available (using CPU version)")
    except:
        pass

def check_api_keys() -> None:
    """Check if API keys are set"""
    import os
    
    print("\n" + "="*60)
    print("API KEYS")
    print("="*60)
    
    keys_to_check = [
        ("OPENAI_API_KEY", "OpenAI"),
        ("GOOGLE_API_KEY", "Google AI"),
    ]
    
    for env_var, name in keys_to_check:
        if os.getenv(env_var):
            print(f"✓ {name:<30} Set")
        else:
            print(f"○ {name:<30} Not set (may use default)")

def print_summary(results: Dict[str, bool]) -> bool:
    """Print summary and return overall status"""
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    required_modules = [
        "numpy", "torch", "openai", "tqdm", "psutil",
        "sentence_transformers", "transformers", "spacy",
        "faiss", "FlagEmbedding", "sklearn"
    ]
    
    missing_required = [m for m in required_modules if not results.get(m, False)]
    
    if not missing_required:
        print("\n✓ All required dependencies are installed!")
        print("\nYou can now:")
        print("1. Set your API keys:")
        print("   export OPENAI_API_KEY='your-api-key'")
        print("\n2. Test the installation:")
        print("   cd Data")
        print("   python prepare_datasets.py --test --dataset TimeQuestions")
        return True
    else:
        print(f"\n✗ Missing {len(missing_required)} required dependencies:")
        for module in missing_required:
            print(f"   - {module}")
        print("\nTo install missing dependencies:")
        print("   pip install -r requirements.txt")
        print("   python -m spacy download en_core_web_sm")
        return False

def main():
    """Main verification function"""
    print("\n" + "="*60)
    print("MEMOTIME INSTALLATION VERIFICATION")
    print("="*60)
    
    # Check Python version
    print("\nPython Version:")
    python_ok = check_python_version()
    
    if not python_ok:
        print("\n✗ Python version too old. Please upgrade to Python 3.9 or higher.")
        sys.exit(1)
    
    # Check dependencies
    results = check_dependencies()
    
    # Check spaCy model
    print("\n" + "="*60)
    print("SPACY MODEL")
    print("="*60)
    spacy_ok = check_spacy_model()
    results["spacy_model"] = spacy_ok
    
    # Check GPU support
    check_gpu_support()
    
    # Check API keys
    check_api_keys()
    
    # Print summary
    all_ok = print_summary(results)
    
    if all_ok and spacy_ok:
        print("\n" + "="*60)
        print("✓ INSTALLATION VERIFIED SUCCESSFULLY!")
        print("="*60 + "\n")
        sys.exit(0)
    else:
        print("\n" + "="*60)
        print("✗ INSTALLATION INCOMPLETE")
        print("="*60 + "\n")
        sys.exit(1)

if __name__ == "__main__":
    main()




