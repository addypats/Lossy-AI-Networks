#!/usr/bin/env python3
'''
Quick test script for Megatron-LM + Lossy Network integration.
Run this to verify everything is working correctly.
'''

import torch
import sys
import os

def test_basic_imports():
    print("🔍 Testing basic imports...")
    
    try:
        import torch.distributed as dist
        print("✅ PyTorch distributed")
    except ImportError as e:
        print(f"❌ PyTorch distributed: {e}")
        return False
    
    try:
        from transformers import AutoModel, AutoTokenizer
        print("✅ Transformers")
    except ImportError as e:
        print(f"❌ Transformers: {e}")
        return False
    
    try:
        import wandb
        print("✅ Weights & Biases")
    except ImportError as e:
        print(f"❌ Weights & Biases: {e}")
        return False
    
    return True

def test_megatron_imports():
    print("🔍 Testing Megatron-LM imports...")
    
    try:
        from megatron.core.tensor_parallel import mappings
        print("✅ Megatron tensor parallel mappings")
        megatron_available = True
    except ImportError as e:
        print(f"⚠️ Megatron tensor parallel mappings: {e}")
        megatron_available = False
    
    try:
        from megatron.core.models.gpt import GPTModel
        print("✅ Megatron GPT model")
    except ImportError as e:
        print(f"⚠️ Megatron GPT model: {e}")
        if megatron_available:
            megatron_available = False
    
    return megatron_available

def test_your_modules():
    print("🔍 Testing your existing modules...")
    
    try:
        from comms import LossyNetwork, GillbertElliotLossyNetwork
        print("✅ Your lossy network classes")
    except ImportError as e:
        print(f"❌ Your lossy network classes: {e}")
        return False
    
    try:
        from data import get_dataset
        print("✅ Your data loading")
    except ImportError as e:
        print(f"❌ Your data loading: {e}")
        return False
    
    try:
        from parallel_layers_gpt import RowParallelLinear
        print("✅ Your parallel layers")
    except ImportError as e:
        print(f"❌ Your parallel layers: {e}")
        return False
    
    return True

def test_integration():
    print("🔍 Testing integration...")
    
    # Test basic lossy network
    try:
        class MockArgs:
            loss_rate = 0.1
        
        args = MockArgs()
        from comms import LossyNetwork
        network = LossyNetwork(args)
        
        # Test with dummy tensor
        test_tensor = torch.randn(10, 10)
        mask = network.send(test_tensor)
        result = network.receive(test_tensor, mask)
        
        print("✅ Basic lossy network functionality")
        return True
    except Exception as e:
        print(f"❌ Basic lossy network functionality: {e}")
        return False

def main():
    print("🧪 Running integration test...")
    print()
    
    # Test components
    basic_ok = test_basic_imports()
    megatron_ok = test_megatron_imports()
    modules_ok = test_your_modules()
    integration_ok = test_integration()
    
    print()
    print("📊 Test Results:")
    print(f"   Basic imports: {'✅' if basic_ok else '❌'}")
    print(f"   Megatron-LM: {'✅' if megatron_ok else '⚠️'}")
    print(f"   Your modules: {'✅' if modules_ok else '❌'}")
    print(f"   Integration: {'✅' if integration_ok else '❌'}")
    
    if basic_ok and modules_ok and integration_ok:
        if megatron_ok:
            print("🎉 All tests passed! You can use Megatron-LM integration.")
            print("💡 Run: python src/pytorch_train_tp_gpt_megatron.py --help")
        else:
            print("⚠️ Basic functionality works, but Megatron-LM is not available.")
            print("💡 You can still use your existing custom tensor parallelism.")
            print("💡 To install Megatron-LM: python setup_megatron_simple.py install")
    else:
        print("❌ Some tests failed. Please check your environment setup.")
    
    print()
    print("🔗 Next steps:")
    print("1. Make sure your existing code works: python src/pytorch_train_tp_gpt.py --help")
    print("2. Test the Megatron integration: python src/pytorch_train_tp_gpt_megatron.py --help")
    print("3. Run experiments: bash run_experiments_megatron.sh")

if __name__ == "__main__":
    main()
