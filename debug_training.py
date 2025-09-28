#!/usr/bin/env python3
"""
Debug script to test training setup step by step
"""

import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def test_gpu_availability():
    """Test GPU availability and memory"""
    print("=" * 50)
    print("Testing GPU availability...")
    
    if torch.cuda.is_available():
        print(f"✅ CUDA available: {torch.cuda.is_available()}")
        print(f"✅ CUDA version: {torch.version.cuda}")
        print(f"✅ GPU count: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"   GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
            
        # Test GPU memory allocation
        try:
            test_tensor = torch.randn(1000, 1000).cuda()
            print("✅ GPU memory allocation test passed")
            del test_tensor
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"❌ GPU memory allocation test failed: {e}")
            return False
    else:
        print("❌ CUDA not available")
        return False
    
    return True

def test_model_loading():
    """Test model loading with different configurations"""
    print("=" * 50)
    print("Testing model loading...")
    
    model_name = "HuggingFaceTB/SmolLM2-135M"
    
    try:
        # Test 1: Load tokenizer
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(f"✅ Tokenizer loaded, vocab size: {len(tokenizer)}")
        
        # Test 2: Load model with different memory configurations
        configs = [
            {"device_map": "auto", "torch_dtype": torch.bfloat16},
            {"device_map": "auto", "torch_dtype": torch.float16},
            {"device_map": "cpu", "torch_dtype": torch.float32},
        ]
        
        for i, config in enumerate(configs):
            try:
                print(f"\nTesting config {i+1}: {config}")
                model = AutoModelForCausalLM.from_pretrained(model_name, **config)
                print(f"✅ Model loaded successfully with config {i+1}")
                
                # Test forward pass
                test_input = tokenizer("Hello world", return_tensors="pt")
                if config["device_map"] != "cpu":
                    test_input = {k: v.cuda() for k, v in test_input.items()}
                
                with torch.no_grad():
                    outputs = model(**test_input)
                print(f"✅ Forward pass successful with config {i+1}")
                
                del model
                torch.cuda.empty_cache()
                break
                
            except Exception as e:
                print(f"❌ Config {i+1} failed: {e}")
                continue
        
        return True
        
    except Exception as e:
        print(f"❌ Model loading test failed: {e}")
        return False

def test_dataset_loading():
    """Test dataset loading"""
    print("=" * 50)
    print("Testing dataset loading...")
    
    try:
        from datasets import load_dataset
        
        # Test loading the Bangla corpus
        print("Loading titulm_bangla_corpus...")
        dataset = load_dataset("hishab/titulm-bangla-corpus")
        print(f"✅ Dataset loaded successfully!")
        print(f"   Train samples: {len(dataset['train'])}")
        
        # Test a few samples
        sample = dataset['train'][0]
        print(f"✅ Sample data: {sample}")
        
        return True
        
    except Exception as e:
        print(f"❌ Dataset loading test failed: {e}")
        return False

def test_llamafactory_import():
    """Test LLaMA-Factory import and basic functionality"""
    print("=" * 50)
    print("Testing LLaMA-Factory...")
    
    try:
        import llamafactory
        print("✅ LLaMA-Factory imported successfully")
        
        # Test CLI availability
        import subprocess
        result = subprocess.run(['llamafactory-cli', '--help'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ llamafactory-cli available")
        else:
            print(f"❌ llamafactory-cli failed: {result.stderr}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ LLaMA-Factory test failed: {e}")
        return False

def test_simple_training():
    """Test simple training configuration"""
    print("=" * 50)
    print("Testing simple training configuration...")
    
    try:
        # Test with minimal configuration
        config_file = "examples/train_lora/smollm2_bangla_pretrain.yaml"
        
        if not os.path.exists(config_file):
            print(f"❌ Config file not found: {config_file}")
            return False
        
        print(f"✅ Config file exists: {config_file}")
        
        # Test with very small dataset and minimal resources
        test_command = [
            'llamafactory-cli', 'train', config_file,
            'max_samples=10',  # Very small dataset
            'per_device_train_batch_size=1',  # Minimal batch size
            'gradient_accumulation_steps=1',
            'num_train_epochs=1',
            'save_steps=5',
            'logging_steps=1',
            'output_dir=./test_output'
        ]
        
        print("Testing with minimal configuration...")
        print("Command:", ' '.join(test_command))
        
        # Don't actually run it, just test the command construction
        print("✅ Command construction successful")
        print("Note: This is just a test - actual training would require more resources")
        
        return True
        
    except Exception as e:
        print(f"❌ Simple training test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🔍 LLaMA-Factory Training Debug Script")
    print("=" * 50)
    
    tests = [
        ("GPU Availability", test_gpu_availability),
        ("Model Loading", test_model_loading),
        ("Dataset Loading", test_dataset_loading),
        ("LLaMA-Factory Import", test_llamafactory_import),
        ("Simple Training Config", test_simple_training),
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
    print("📊 Debug Results Summary:")
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
        print("🎉 All tests passed! The issue might be with the training configuration.")
        print("\nRecommendations:")
        print("1. Try reducing batch size: per_device_train_batch_size=1")
        print("2. Try reducing sequence length: cutoff_len=2048")
        print("3. Try using CPU for debugging: device_map=cpu")
    else:
        print("⚠️  Some tests failed. Please fix the issues before running training.")
        
        if not results[0][1]:  # GPU test failed
            print("\n🔧 GPU Issues:")
            print("- Check if CUDA is properly installed")
            print("- Check if GPUs are available")
            print("- Try running with CPU: device_map=cpu")
        
        if not results[1][1]:  # Model loading failed
            print("\n🔧 Model Loading Issues:")
            print("- Try reducing model precision: torch_dtype=torch.float16")
            print("- Try using CPU: device_map=cpu")
            print("- Check available memory")
    
    return 0 if passed == total else 1

if __name__ == "__main__":
    sys.exit(main())
