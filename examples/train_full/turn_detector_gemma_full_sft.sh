#!/usr/bin/env bash

# Turn Detector Full-Parameter SFT Training Script (Qwen2.5-1.5B)

set -euo pipefail

# Configuration
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export HF_HOME="${HF_HOME:-./hf_cache}"

# Training parameters
OUTPUT_DIR="./saves/turn-detector/full/sft-gemma3-270m-refined50k"
MODEL_NAME="google/gemma-3-270m"
DATASET_NAME="rnnandi/bn-turn-detection-dataset-train-v3-refined-50k"

echo "Starting Turn Detector FULL SFT Training"
echo "Model: $MODEL_NAME"
echo "Dataset: $DATASET_NAME"
echo "Output: $OUTPUT_DIR"

mkdir -p "${OUTPUT_DIR}/logs"
NOHUP_LOG="${OUTPUT_DIR}/logs/train_$(date +%Y%m%d_%H%M%S).log"

llamafactory-cli train \
  examples/train_full/turn_detector_gemma_full_sft.yaml \
  output_dir="${OUTPUT_DIR}" \
  model_name_or_path="${MODEL_NAME}" \
  dataset="${DATASET_NAME}" \
  cache_dir="${HF_HOME}" \
  per_device_train_batch_size=1 \
  gradient_accumulation_steps=2 \
  learning_rate=2e-5 \
  num_train_epochs=4 \
  cutoff_len=128 \
  preprocessing_num_workers=16 \
  bf16=true \
  ddp_find_unused_parameters=false \
  dataloader_pin_memory=true \
  logging_steps=10 \
  save_steps=20000 \
  report_to=none \
  > "$NOHUP_LOG" 2>&1 &

PID=$!
echo "Training started in background. PID=${PID}"
echo "Logs: $NOHUP_LOG"
echo "Tail logs with: tail -f \"$NOHUP_LOG\""

