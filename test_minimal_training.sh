#!/usr/bin/env bash

# Minimal training test script
set -euo pipefail

echo "🧪 Testing minimal training configuration..."

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate llamafactory

# Test with minimal configuration
echo "Running minimal training test..."
llamafactory-cli train examples/train_lora/smollm2_bangla_pretrain_minimal.yaml \
    max_samples=10 \
    per_device_train_batch_size=1 \
    gradient_accumulation_steps=1 \
    num_train_epochs=1 \
    save_steps=5 \
    logging_steps=1 \
    output_dir=./test_minimal_output

echo "✅ Minimal training test completed!"
