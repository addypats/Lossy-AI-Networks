#!/usr/bin/env python3
"""
Simple Setup Script for Megatron-LM Integration
This script helps you set up Megatron-LM for use with your existing lossy network research.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, check=True, cwd=None):
    """Run a shell command and return the result."""
    print(f"🔧 Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=cwd)
    if check and result.returncode != 0:
        print(f"❌ Error running command: {cmd}")
        print(f"Error output: {result.stderr}")
        return False
    return True

def check_megatron_availability():
    """Check if Megatron-LM is already installed."""
    try:
        import subprocess
        result = subprocess.run([sys.executable, "-c", "from megatron.core.tensor_parallel import mappings"], 
                               capture_output=True)
        return result.returncode == 0
    except:
        return False

def install_megatron_lm():
    """Install Megatron-LM from source."""
    print("📦 Installing Megatron-LM...")
    
    # Check if already installed
    if check_megatron_availability():
        print("✅ Megatron-LM is already installed!")
        return True
    
    # Install prerequisites
    print("📋 Installing prerequisites...")
    prerequisites = [
        "torch>=1.13.0",
        "transformers>=4.20.0",
        "datasets",
        "tensorboard", 
        "wandb",
        "packaging",
        "ninja",
    ]
    
    for package in prerequisites:
        if not run_command(f"{sys.executable} -m pip install {package}", check=False):
            print(f"⚠️ Warning: Failed to install {package}")
    
    # Clone Megatron-LM if not exists
    megatron_dir = "Megatron-LM"
    if not os.path.exists(megatron_dir):
        print("📥 Cloning Megatron-LM repository...")
        if not run_command("git clone https://github.com/NVIDIA/Megatron-LM.git"):
            return False
    
    # Install Megatron-LM
    print("🔧 Installing Megatron-LM...")
    if not run_command(f"{sys.executable} -m pip install -e .", cwd=megatron_dir):
        return False
    
    # Verify installation
    if check_megatron_availability():
        print("✅ Megatron-LM installation successful!")
        return True
    else:
        print("❌ Megatron-LM installation verification failed")
        return False

def create_quick_test():
    """Create a quick test script to verify everything works."""
    test_script = """#!/usr/bin/env python3
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
"""
    
    with open("test_integration.py", "w") as f:
        f.write(test_script)
    
    # Make executable
    os.chmod("test_integration.py", 0o755)
    print("📝 Created test_integration.py")

def create_readme():
    """Create a simple README for the integration."""
    readme_content = """# Megatron-LM + Lossy Network Integration

## Quick Start

1. **Test your existing setup:**
   ```bash
   python test_integration.py
   ```

2. **Install Megatron-LM (optional but recommended):**
   ```bash
   python setup_megatron_simple.py install
   ```

3. **Run experiments:**
   ```bash
   # Test Megatron availability
   bash run_experiments_megatron.sh test-megatron
   
   # Run full experiments
   bash run_experiments_megatron.sh run
   
   # Analyze results
   bash run_experiments_megatron.sh analyze
   ```

## Files

- `pytorch_train_tp_gpt_megatron.py` - Main training script with Megatron integration
- `run_experiments_megatron.sh` - Experimental script for batch runs
- `test_integration.py` - Quick test to verify everything works
- `setup_megatron_simple.py` - This setup script

## What This Does

This integration gives you:

✅ **Production-scale tensor parallelism** from Megatron-LM  
✅ **Your existing lossy network research** preserved  
✅ **Your existing experimental setup** maintained  
✅ **Fallback to custom TP** if Megatron is not available  
✅ **Direct comparison** between implementations  

## Research Benefits

- Test lossy communication on larger models that require tensor parallelism
- Compare optimized vs custom tensor parallel implementations
- Study loss effects at production scale
- Maintain all your existing experimental configurations

Your research setup remains exactly the same - just with better performance and scalability!
"""
    
    with open("README_Integration.md", "w") as f:
        f.write(readme_content)
    
    print("📝 Created README_Integration.md")

def main():
    """Main setup function."""
    print("🚀 Megatron-LM + Lossy Network Setup")
    print("=" * 50)
    
    action = input("What would you like to do?\n"
                  "1. Install Megatron-LM\n"
                  "2. Test current setup\n"
                  "3. Create test files only\n"
                  "4. All of the above\n"
                  "Enter choice (1-4): ").strip()
    
    if action in ['1', '4']:
        print("\n📦 Installing Megatron-LM...")
        success = install_megatron_lm()
        if not success:
            print("❌ Megatron-LM installation failed")
            print("💡 You can still use your existing custom tensor parallelism")
    
    if action in ['2', '4']:
        print("\n🧪 Testing current setup...")
        if check_megatron_availability():
            print("✅ Megatron-LM is available!")
        else:
            print("⚠️ Megatron-LM is not available, will use custom tensor parallelism")
    
    if action in ['3', '4']:
        print("\n📝 Creating helper files...")
        create_quick_test()
        create_readme()
    
    print("\n🎉 Setup completed!")
    print("\n📋 Next steps:")
    print("1. Run: python test_integration.py")
    print("2. Test: bash run_experiments_megatron.sh test-megatron") 
    print("3. Experiment: bash run_experiments_megatron.sh run")
    print("4. Check README_Integration.md for detailed instructions")

if __name__ == "__main__":
    main()
