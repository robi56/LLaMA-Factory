#!/usr/bin/env bash

# Debug training test script - use c4_demo dataset first
set -euo pipefail

echo "🧪 Testing with c4_demo dataset (smaller, local dataset)..."

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate llamafactory

# Test with c4_demo dataset (local, small dataset)
echo "Running debug training test with c4_demo..."
llamafactory-cli train examples/train_lora/smollm2_bangla_pretrain_debug.yaml \
    max_samples=10 \
    per_device_train_batch_size=1 \
    gradient_accumulation_steps=1 \
    num_train_epochs=1 \
    save_steps=5 \
    logging_steps=1 \
    output_dir=./test_debug_output

echo "✅ Debug training test completed!"
