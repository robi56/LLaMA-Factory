#!/usr/bin/env bash

# SmolLM2-135M Bangla Pre-training with Full Fine-tuning using LLaMA-Factory
# This script adapts your original pre-training approach to use LLaMA-Factory

set -euo pipefail

# Configuration
CONDA_ENV_NAME=${CONDA_ENV_NAME:-llamafactory}
OUTPUT_DIR=${OUTPUT_DIR:-./saves/smollm2-135m/full/bangla_pretrain}
MODEL_NAME=${MODEL_NAME:-HuggingFaceTB/SmolLM2-135M}
TITULM_TOKENIZER=${TITULM_TOKENIZER:-hishab/titulm-llama-3.2-3b-v2.0}
DATASET_NAME=${DATASET_NAME:-titulm_bangla_corpus}

# Use all available GPUs (comment out to use specific GPUs)
# export CUDA_VISIBLE_DEVICES=0,1,2,3  # Uncomment and modify to use specific GPUs

echo "[1/4] Setting up environment: ${CONDA_ENV_NAME}"

# Initialize conda if not already done
if ! command -v conda &> /dev/null; then
    echo "Conda not found. Please install Anaconda/Miniconda first."
    exit 1
fi

# Initialize conda for this shell session
eval "$(conda shell.bash hook)"

if ! conda env list | grep -q "^${CONDA_ENV_NAME} "; then
    echo "Creating new conda environment with Python 3.10..."
    conda create -n ${CONDA_ENV_NAME} python=3.10 -y
    conda activate ${CONDA_ENV_NAME}
    echo "Installing LLaMA-Factory and dependencies..."
    pip install -e .
    pip install wandb
else
    echo "Activating existing conda environment: ${CONDA_ENV_NAME}"
    conda activate ${CONDA_ENV_NAME}
fi

echo "[2/4] Merging tokenizer (TituLM 48K Bangla tokens -> SmolLM2)"
python - <<'PYCODE'
import os
import sys
sys.path.append('pre_training')
from tokenizer_merger import create_merged_tokenizer_config

output_dir = os.environ.get("OUTPUT_DIR", "./saves/smollm2-135m/full/bangla_pretrain")
merge_dir = os.path.join(output_dir, "merged_tokenizer")

titulm = os.environ.get("TITULM_TOKENIZER", "hishab/titulm-llama-3.2-3b-v2.0")
model = os.environ.get("MODEL_NAME", "HuggingFaceTB/SmolLM2-135M")

print("TituLM contains: LLaMA-32K + 48K new Bangla tokens = ~170K total")
print("SmolLM2 contains: English tokens = 49K total")
print("Target: Extract the 48K unique Bangla tokens from TituLM")

print("\nMerging tokenizers...")
tok = create_merged_tokenizer_config(
    titulm_tokenizer_path=titulm,
    smollm_tokenizer_path=model,
    output_path=merge_dir
)
print("Merged tokenizer vocab size:", len(tok))
PYCODE

echo "[3/4] Starting pre-training with LLaMA-Factory (Full Fine-tuning)"
mkdir -p "${OUTPUT_DIR}/logs"
NOHUP_LOG="${OUTPUT_DIR}/logs/train_$(date +%Y%m%d_%H%M%S).log"

# Set wandb environment variables
export WANDB_PROJECT="smollm2-bangla-pretraining"
export WANDB_RUN_NAME="smollm2_bangla_full_$(date +%Y%m%d_%H%M%S)"

# Check if wandb key is set, prompt if not
if [ -z "${WANDB_API_KEY:-}" ]; then
    echo "⚠️  WANDB_API_KEY not set. You can:"
    echo "   1. Export it: export WANDB_API_KEY=your_key_here"
    echo "   2. Login interactively: wandb login"
    echo "   3. Continue without wandb (will use 'none' instead)"
    echo ""
    read -p "Do you want to login to wandb now? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        wandb login
    else
        echo "Continuing without wandb logging..."
        WANDB_REPORT_TO="none"
    fi
else
    echo "✅ WANDB_API_KEY is set"
    WANDB_REPORT_TO="wandb"
fi

# Run training with LLaMA-Factory
nohup llamafactory-cli train examples/train_full/smollm2_bangla_pretrain_full.yaml \
    output_dir="${OUTPUT_DIR}" \
    model_name_or_path="${MODEL_NAME}" \
    dataset="${DATASET_NAME}" \
    max_samples=20000 \
    per_device_train_batch_size=4 \
    gradient_accumulation_steps=8 \
    learning_rate=2e-4 \
    num_train_epochs=2 \
    cutoff_len=4096 \
    bf16=true \
    logging_steps=50 \
    save_steps=1000 \
    report_to="${WANDB_REPORT_TO:-wandb}" \
    > "$NOHUP_LOG" 2>&1 &

PID=$!
echo "Training started in background. PID=${PID}"
echo "Logs: $NOHUP_LOG"
echo "Tail logs with: tail -f \"$NOHUP_LOG\""
echo "Monitor with: watch -n 1 'tail -n 20 \"$NOHUP_LOG\"'"

echo "Done. Outputs (and logs) saved under ${OUTPUT_DIR}"
echo ""
echo "To monitor training:"
echo "  tail -f \"$NOHUP_LOG\""
echo "  wandb login  # if not already logged in"
echo "  # Then check your wandb dashboard"
