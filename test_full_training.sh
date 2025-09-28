#!/usr/bin/env bash

# Test full fine-tuning (without LoRA) with small Bangla dataset
set -euo pipefail

echo "🧪 Testing Full Fine-tuning (No LoRA) with Small Bangla Dataset"
echo "================================================================"

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate llamafactory

echo "Running full fine-tuning test..."
llamafactory-cli train examples/train_full/smollm2_bangla_pretrain_test.yaml \
    max_samples=20 \
    per_device_train_batch_size=1 \
    gradient_accumulation_steps=1 \
    num_train_epochs=1 \
    save_steps=5 \
    logging_steps=1 \
    output_dir=./test_full_output

if [ $? -eq 0 ]; then
    echo "✅ Full fine-tuning test PASSED!"
    echo ""
    echo "🎉 Full fine-tuning works! You can now:"
    echo "1. Try with larger datasets"
    echo "2. Try with the original titulm_bangla_corpus"
    echo "3. Increase batch size and other parameters"
    echo "4. Use tokenizer merging for better Bangla support"
else
    echo "❌ Full fine-tuning test FAILED!"
    echo "This suggests the issue might be with:"
    echo "1. Memory constraints (full fine-tuning uses more memory)"
    echo "2. Model loading issues"
    echo "3. Dataset processing issues"
    exit 1
fi
