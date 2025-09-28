#!/usr/bin/env python3
"""
Test script to verify Bangla pre-training setup before running full training.
This script tests:
1. Tokenizer merging
2. Dataset loading
3. Basic configuration validation
"""

import os
import sys
from pathlib import Path

def test_tokenizer_merging():
    """Test tokenizer merging functionality"""
    print("=" * 50)
    print("Testing tokenizer merging...")
    
    try:
        import sys
        sys.path.append('pre_training')
        from tokenizer_merger import create_merged_tokenizer_config
        
        # Test parameters
        titulm_path = "hishab/titulm-llama-3.2-3b-v2.0"
        smollm_path = "HuggingFaceTB/SmolLM2-135M"
        output_path = "./test_merged_tokenizer"
        
        print(f"TituLM tokenizer: {titulm_path}")
        print(f"SmolLM2 tokenizer: {smollm_path}")
        print(f"Output path: {output_path}")
        print("\nTituLM contains: LLaMA-32K + 48K new Bangla tokens = ~170K total")
        print("SmolLM2 contains: English tokens = 49K total")
        print("Target: Extract the 48K unique Bangla tokens from TituLM")
        
        # Use your existing tokenizer merger
        print("\nMerging tokenizers using your existing approach...")
        tokenizer = create_merged_tokenizer_config(
            titulm_tokenizer_path=titulm_path,
            smollm_tokenizer_path=smollm_path,
            output_path=output_path
        )
        print(f"✅ Tokenizer merging successful! Vocab size: {len(tokenizer)}")
        
        # Clean up test directory
        import shutil
        if os.path.exists(output_path):
            shutil.rmtree(output_path)
            print("🧹 Cleaned up test directory")
            
        return True
        
    except Exception as e:
        print(f"❌ Tokenizer merging test failed: {e}")
        return False

def test_dataset_loading():
    """Test dataset loading functionality"""
    print("=" * 50)
    print("Testing dataset loading...")
    
    try:
        from datasets import load_dataset
        
        # Test loading the Bangla corpus
        print("Loading titulm_bangla_corpus...")
        dataset = load_dataset("hishab/titulm-bangla-corpus")
        print(f"✅ Dataset loaded successfully!")
        print(f"   Train samples: {len(dataset['train'])}")
        if 'validation' in dataset:
            print(f"   Validation samples: {len(dataset['validation'])}")
        
        # Test loading a subset
        print("\nLoading common_crawl subset...")
        subset = load_dataset("hishab/titulm-bangla-corpus", data_dir="common_crawl")
        print(f"✅ Subset loaded successfully!")
        print(f"   Common crawl samples: {len(subset['train'])}")
        
        return True
        
    except Exception as e:
        print(f"❌ Dataset loading test failed: {e}")
        return False

def test_llamafactory_config():
    """Test LLaMA-Factory configuration files"""
    print("=" * 50)
    print("Testing LLaMA-Factory configuration...")
    
    config_files = [
        "examples/train_lora/smollm2_bangla_pretrain.yaml",
        "examples/train_full/smollm2_bangla_pretrain_full.yaml"
    ]
    
    for config_file in config_files:
        if os.path.exists(config_file):
            print(f"✅ Configuration file exists: {config_file}")
        else:
            print(f"❌ Configuration file missing: {config_file}")
            return False
    
    # Test dataset_info.json
    dataset_info_path = "data/dataset_info.json"
    if os.path.exists(dataset_info_path):
        print(f"✅ Dataset info file exists: {dataset_info_path}")
        
        # Check if our Bangla dataset entries are present
        import json
        with open(dataset_info_path, 'r') as f:
            dataset_info = json.load(f)
        
        bangla_datasets = [
            "titulm_bangla_corpus",
            "titulm_bangla_common_crawl", 
            "titulm_bangla_translated",
            "titulm_bangla_romanized"
        ]
        
        for dataset_name in bangla_datasets:
            if dataset_name in dataset_info:
                print(f"✅ Dataset entry found: {dataset_name}")
            else:
                print(f"❌ Dataset entry missing: {dataset_name}")
                return False
    else:
        print(f"❌ Dataset info file missing: {dataset_info_path}")
        return False
    
    return True

def test_environment():
    """Test environment setup"""
    print("=" * 50)
    print("Testing environment...")
    
    try:
        import torch
        print(f"✅ PyTorch version: {torch.__version__}")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   CUDA version: {torch.version.cuda}")
            print(f"   GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
        
        import transformers
        print(f"✅ Transformers version: {transformers.__version__}")
        
        import datasets
        print(f"✅ Datasets version: {datasets.__version__}")
        
        # Test LLaMA-Factory import
        try:
            import llamafactory
            print(f"✅ LLaMA-Factory available")
        except ImportError:
            print("❌ LLaMA-Factory not available")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Environment test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Bangla Pre-training Setup Test")
    print("=" * 50)
    
    tests = [
        ("Environment", test_environment),
        ("LLaMA-Factory Config", test_llamafactory_config),
        ("Dataset Loading", test_dataset_loading),
        ("Tokenizer Merging", test_tokenizer_merging),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
        print()
    
    # Summary
    print("=" * 50)
    print("📊 Test Results Summary:")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20} {status}")
        if result:
            passed += 1
    
    print("=" * 50)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! You're ready to run Bangla pre-training.")
        print("\nNext steps:")
        print("1. Run: bash examples/train_lora/smollm2_bangla_pretrain.sh")
        print("2. Monitor: tail -f saves/smollm2-135m/lora/bangla_pretrain/logs/train_*.log")
        print("3. Check wandb dashboard for training metrics")
    else:
        print("⚠️  Some tests failed. Please fix the issues before running training.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
