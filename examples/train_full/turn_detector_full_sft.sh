#!/usr/bin/env bash

# Turn Detector Full SFT Training Script
# Based on your original training script but adapted for LLaMA Factory

set -euo pipefail

# Configuration
export CUDA_VISIBLE_DEVICES=0,1,2,3  # Adjust based on your available GPUs
export HF_HOME="${HF_HOME:-./hf_cache}"
export WANDB_API_KEY="${WANDB_API_KEY:-your_wandb_key_here}"

# Training parameters
OUTPUT_DIR="./saves/turn-detector/full/sft"
MODEL_NAME="hishab/titulm-llama-3.2-1b-v1.1"
DATASET_NAME="turn_detector_demo"

echo "Starting Turn Detector Full SFT Training"
echo "Model: $MODEL_NAME"
echo "Dataset: $DATASET_NAME"
echo "Output: $OUTPUT_DIR"

# Create output directory
mkdir -p "${OUTPUT_DIR}/logs"
NOHUP_LOG="${OUTPUT_DIR}/logs/train_$(date +%Y%m%d_%H%M%S).log"

# Run training with LLaMA Factory
llamafactory-cli train \
    examples/train_full/turn_detector_full_sft.yaml \
    --output_dir="${OUTPUT_DIR}" \
    --model_name_or_path="${MODEL_NAME}" \
    --dataset="${DATASET_NAME}" \
    --cache_dir="${HF_HOME}" \
    --max_samples=1000 \
    --per_device_train_batch_size=2 \
    --gradient_accumulation_steps=8 \
    --learning_rate=2e-5 \
    --num_train_epochs=20 \
    --cutoff_len=512 \
    --preprocessing_num_workers=16 \
    --bf16=true \
    --ddp_find_unused_parameters=false \
    --dataloader_pin_memory=true \
    --logging_steps=10 \
    --save_steps=500 \
    --report_to=wandb \
    > "$NOHUP_LOG" 2>&1 &

PID=$!
echo "Training started in background. PID=${PID}"
echo "Logs: $NOHUP_LOG"
echo "Tail logs with: tail -f \"$NOHUP_LOG\""
echo "Monitor with: watch -n 1 'tail -n 20 \"$NOHUP_LOG\"'"
