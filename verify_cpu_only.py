#!/usr/bin/env python3
"""
CPU-Only Verification Script
Ensures the system is running in CPU-only mode as required for submission.
"""

import sys
import torch
import numpy as np
from sentence_transformers import SentenceTransformer

def verify_cpu_only():
    """Verify that the system is running in CPU-only mode."""
    
    print("🔍 CPU-Only System Verification")
    print("=" * 50)
    
    # Check PyTorch
    print(f"📦 PyTorch version: {torch.__version__}")
    print(f"💻 CUDA available: {torch.cuda.is_available()}")
    print(f"🎯 Device count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}")
    
    if torch.cuda.is_available():
        print("❌ ERROR: CUDA is available - this violates CPU-only requirements!")
        return False
    else:
        print("✅ VERIFIED: CPU-only mode confirmed")
    
    # Test tensor operations on CPU
    print("\n🧮 Testing CPU tensor operations...")
    try:
        x = torch.randn(100, 100)
        y = torch.mm(x, x.t())
        print(f"✅ CPU tensor operations successful (device: {x.device})")
    except Exception as e:
        print(f"❌ CPU tensor operations failed: {e}")
        return False
    
    # Test sentence transformers on CPU
    print("\n🤖 Testing SentenceTransformer on CPU...")
    try:
        # Test with a small model first
        model = SentenceTransformer('all-MiniLM-L6-v2')
        test_text = "This is a CPU test."
        embedding = model.encode(test_text)
        print(f"✅ SentenceTransformer works on CPU (embedding shape: {embedding.shape})")
        
        # Verify the device
        if hasattr(model, '_modules'):
            for name, module in model._modules.items():
                if hasattr(module, 'device'):
                    device = next(module.parameters()).device
                    print(f"✅ Model '{name}' is on device: {device}")
        
    except Exception as e:
        print(f"❌ SentenceTransformer test failed: {e}")
        return False
    
    print("\n🎉 ALL VERIFICATIONS PASSED - SYSTEM IS CPU-ONLY COMPLIANT!")
    return True

if __name__ == "__main__":
    success = verify_cpu_only()
    sys.exit(0 if success else 1)
