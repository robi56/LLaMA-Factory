#!/usr/bin/env bash

# Step-by-step testing script
set -euo pipefail

echo "🔍 Step-by-Step Training Debug"
echo "================================"

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate llamafactory

echo "Step 1: Test with c4_demo dataset (local, small)"
echo "------------------------------------------------"
llamafactory-cli train examples/train_lora/smollm2_bangla_pretrain_debug.yaml \
    max_samples=10 \
    per_device_train_batch_size=1 \
    gradient_accumulation_steps=1 \
    num_train_epochs=1 \
    save_steps=5 \
    logging_steps=1 \
    output_dir=./test_c4_output

if [ $? -eq 0 ]; then
    echo "✅ Step 1 PASSED: c4_demo training successful"
else
    echo "❌ Step 1 FAILED: c4_demo training failed"
    exit 1
fi

echo ""
echo "Step 2: Create small Bangla dataset"
echo "-----------------------------------"
python create_small_bangla_dataset.py

if [ $? -eq 0 ]; then
    echo "✅ Step 2 PASSED: Small Bangla dataset created"
else
    echo "❌ Step 2 FAILED: Could not create small Bangla dataset"
    exit 1
fi

echo ""
echo "Step 3: Test with small Bangla dataset"
echo "--------------------------------------"
llamafactory-cli train examples/train_lora/smollm2_bangla_small.yaml \
    max_samples=20 \
    per_device_train_batch_size=1 \
    gradient_accumulation_steps=1 \
    num_train_epochs=1 \
    save_steps=10 \
    logging_steps=1 \
    output_dir=./test_bangla_small_output

if [ $? -eq 0 ]; then
    echo "✅ Step 3 PASSED: Small Bangla dataset training successful"
    echo ""
    echo "🎉 All tests passed! You can now try the full training."
    echo ""
    echo "Next steps:"
    echo "1. Try with larger max_samples: max_samples=1000"
    echo "2. Try with original titulm_bangla_corpus dataset"
    echo "3. Increase batch size and other parameters gradually"
else
    echo "❌ Step 3 FAILED: Small Bangla dataset training failed"
    exit 1
fi
