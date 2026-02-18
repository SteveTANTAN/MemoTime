#!/bin/bash
# ================================
# MemoTime Dependency Setup Script
# ================================

set -e  # Exit on error

echo "================================"
echo "MemoTime Dependency Setup"
echo "================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python version
echo "Checking Python version..."
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
echo "Python version: $PYTHON_VERSION"

if ! python -c 'import sys; assert sys.version_info >= (3,9)' 2>/dev/null; then
    echo -e "${RED}Error: Python 3.9 or higher is required${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python version OK${NC}"
echo ""

# Ask user for installation type
echo "Select installation type:"
echo "1) Full installation (recommended)"
echo "2) Minimal installation (core features only)"
echo "3) GPU-enabled installation"
read -p "Enter choice [1-3]: " INSTALL_TYPE
echo ""

case $INSTALL_TYPE in
    1)
        echo "Installing full dependencies..."
        pip install -r requirements.txt
        ;;
    2)
        echo "Installing minimal dependencies..."
        pip install -r requirements-minimal.txt
        ;;
    3)
        echo "Installing GPU-enabled dependencies..."
        echo "Installing PyTorch with CUDA support..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
        
        # Install other dependencies
        pip install -r requirements.txt
        
        # Replace faiss-cpu with faiss-gpu
        echo "Replacing faiss-cpu with faiss-gpu..."
        pip uninstall faiss-cpu -y 2>/dev/null || true
        pip install faiss-gpu
        ;;
    *)
        echo -e "${RED}Invalid choice${NC}"
        exit 1
        ;;
esac

echo ""
echo "================================"
echo "Installing spaCy language model..."
echo "================================"
python -m spacy download en_core_web_sm

echo ""
echo "================================"
echo "Verifying installation..."
echo "================================"

# Verify core dependencies
python -c "
import sys
try:
    import torch
    print('✓ PyTorch:', torch.__version__)
except ImportError as e:
    print('✗ PyTorch: Not installed')
    sys.exit(1)

try:
    import openai
    print('✓ OpenAI: OK')
except ImportError:
    print('✗ OpenAI: Not installed')
    sys.exit(1)

try:
    import faiss
    print('✓ FAISS: OK')
except ImportError:
    print('✗ FAISS: Not installed')
    sys.exit(1)

try:
    import spacy
    print('✓ spaCy: OK')
except ImportError:
    print('✗ spaCy: Not installed')
    sys.exit(1)

try:
    from FlagEmbedding import BGEM3FlagModel
    print('✓ FlagEmbedding: OK')
except ImportError:
    print('✗ FlagEmbedding: Not installed')
    sys.exit(1)

try:
    from sentence_transformers import SentenceTransformer
    print('✓ sentence-transformers: OK')
except ImportError:
    print('✗ sentence-transformers: Not installed')
    sys.exit(1)

print('')
print('All core dependencies installed successfully!')
"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}"
    echo "================================"
    echo "✓ Installation Complete!"
    echo "================================"
    echo -e "${NC}"
    echo ""
    echo "Next steps:"
    echo "1. Set your OpenAI API key:"
    echo "   export OPENAI_API_KEY='your-api-key-here'"
    echo ""
    echo "2. Test the installation:"
    echo "   cd Data"
    echo "    python prepare_datasets.py --build-hybrid --build-embeddings -n 5 --dataset TimeQuestions"
    echo ""
    echo "3. Read the documentation:"
    echo "   cat README.md"
    echo "   cat QUICKSTART.md"
    echo ""
else
    echo -e "${RED}"
    echo "================================"
    echo "✗ Installation failed"
    echo "================================"
    echo -e "${NC}"
    echo "Please check the error messages above and try again."
    exit 1
fi




