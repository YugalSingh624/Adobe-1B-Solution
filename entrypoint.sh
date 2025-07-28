#!/bin/bash
set -e

echo "🚀 Advanced Document Section Selection System (CPU-ONLY)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Verify CPU-only PyTorch (CRITICAL for submission requirements)
python -c "
import torch
print(f'🔧 PyTorch version: {torch.__version__}')
print(f'💻 CUDA available: {torch.cuda.is_available()}')
print(f'🎯 Device: CPU-only ✅' if not torch.cuda.is_available() else '❌ CUDA detected!')
assert not torch.cuda.is_available(), 'ERROR: CUDA detected - should be CPU-only!'
" || {
    echo "❌ CPU-only verification failed!"
    exit 1
}

# Verify model availability
if [ -d "/app/models/bge-small-en-v1.5" ]; then
    echo "✅ BGE model: Ready"
else
    echo "❌ BGE model: Not found"
    exit 1
fi

# Verify Python environment
python -c "from src.embedder import PersonaEmbedder; print('✅ Application: Ready')" 2>/dev/null || {
    echo "❌ Application: Import failed"
    exit 1
}

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📁 Directories:"
echo "  • /app/docs/     - Input PDF documents"
echo "  • /app/outputs/  - Processing results"
echo "  • /app/cache/    - Performance cache"
echo ""
echo "🔧 Commands:"
echo "  • python run_pipeline.py - Run processing pipeline"
echo "  • cat config.json        - View configuration"
echo ""
echo "Ready to process documents! 🎯"

exec "$@"
