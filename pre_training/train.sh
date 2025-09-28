#!/usr/bin/env bash

set -euo pipefail

# Configuration
CONDA_ENV_NAME=${CONDA_ENV_NAME:-llm_pretraining}
OUTPUT_DIR=${OUTPUT_DIR:-./outputs/smollm2_bangla_continual}
MODEL_NAME=${MODEL_NAME:-HuggingFaceTB/SmolLM2-135M}
TITULM_TOKENIZER=${TITULM_TOKENIZER:-hishab/titulm-llama-3.2-3b-v2.0}
DATASET_NAME=${DATASET_NAME:-hishab/titulm-bangla-corpus}

# Use GPUs 1-4 (0-indexed, so 1,2,3,4)
export CUDA_VISIBLE_DEVICES=1,2,3

echo "[1/4] Initializing conda and setting up environment: ${CONDA_ENV_NAME}"

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
    echo "Installing PyTorch with CUDA support..."
    conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
else
    echo "Activating existing conda environment: ${CONDA_ENV_NAME}"
    conda activate ${CONDA_ENV_NAME}
fi

echo "[2/4] Installing additional dependencies"
pip install --upgrade pip
pip install "transformers>=4.44.0" datasets accelerate tensorboard sentencepiece wandb

echo "[3/4] Merging tokenizer (TituLM -> SmolLM2)"
python - <<'PYCODE'
import os
from tokenizer_merger import create_merged_tokenizer_config

output_dir = os.environ.get("OUTPUT_DIR", "./outputs/smollm2_bangla_continual")
merge_dir = os.path.join(output_dir, "merged_tokenizer")

titulm = os.environ.get("TITULM_TOKENIZER", "hishab/titulm-llama-3.2-3b-v2.0")
model = os.environ.get("MODEL_NAME", "HuggingFaceTB/SmolLM2-135M")

tok = create_merged_tokenizer_config(
    titulm_tokenizer_path=titulm,
    smollm_tokenizer_path=model,
    output_path=merge_dir,
)
print("Merged tokenizer vocab size:", len(tok))
PYCODE

echo "[4/4] Starting continual pretraining on GPUs 1-4 (nohup)"
mkdir -p "${OUTPUT_DIR}/logs"
NOHUP_LOG="${OUTPUT_DIR}/logs/train_$(date +%Y%m%d_%H%M%S).log"
nohup python train_run.py --use_multiple_subsets --common_crawl_ratio 0.2 --other_datasets_ratio 0.3 --max_samples 20000 > "$NOHUP_LOG" 2>&1 &
PID=$!
echo "Training started in background. PID=${PID}"
echo "Logs: $NOHUP_LOG"
echo "Tail logs with: tail -f \"$NOHUP_LOG\""

echo "Done. Outputs (and logs) saved under ${OUTPUT_DIR}"

