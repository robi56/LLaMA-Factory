# Bangla Pre-training with LLaMA-Factory

This guide shows you how to adapt your existing pre-training scripts to use LLaMA-Factory for continual pre-training of SmolLM2-135M on Bangla corpus data.

## Overview

Your original pre-training setup includes:
- **Model**: SmolLM2-135M (HuggingFaceTB/SmolLM2-135M)
- **Dataset**: TituLM Bangla Corpus (hishab/titulm-bangla-corpus)
- **Tokenizer**: Merged tokenizer (SmolLM2 + TituLM Bangla tokens)
- **Training**: Continual pre-training with mixed precision

## Files Created

### 1. Tokenizer Merger (`tokenizer_merger.py`)
- Merges SmolLM2 tokenizer with TituLM Bangla tokens
- Handles vocabulary expansion for Bangla language support
- Creates merged tokenizer configuration

### 2. Dataset Configuration (`data/dataset_info.json`)
Added entries for:
- `titulm_bangla_corpus` - Full Bangla corpus
- `titulm_bangla_common_crawl` - Common crawl subset
- `titulm_bangla_translated` - Translated subset  
- `titulm_bangla_romanized` - Romanized subset

### 3. LLaMA-Factory Configuration Files
- `examples/train_lora/smollm2_bangla_pretrain.yaml` - LoRA pre-training config
- `examples/train_full/smollm2_bangla_pretrain_full.yaml` - Full fine-tuning config

### 4. Training Scripts
- `examples/train_lora/smollm2_bangla_pretrain.sh` - LoRA training script
- `examples/train_full/smollm2_bangla_pretrain_full.sh` - Full training script

## Usage

### Option 1: LoRA Pre-training (Recommended)

```bash
# Make script executable
chmod +x examples/train_lora/smollm2_bangla_pretrain.sh

# Run LoRA pre-training
bash examples/train_lora/smollm2_bangla_pretrain.sh
```

### Option 2: Full Fine-tuning

```bash
# Make script executable
chmod +x examples/train_full/smollm2_bangla_pretrain_full.sh

# Run full fine-tuning
bash examples/train_full/smollm2_bangla_pretrain_full.sh
```

### Option 3: Direct LLaMA-Factory CLI

```bash
# LoRA pre-training
llamafactory-cli train examples/train_lora/smollm2_bangla_pretrain.yaml

# Full fine-tuning
llamafactory-cli train examples/train_full/smollm2_bangla_pretrain_full.yaml
```

## Configuration Options

### Key Parameters

| Parameter | LoRA | Full | Description |
|-----------|------|------|-------------|
| `per_device_train_batch_size` | 8 | 4 | Batch size per device |
| `gradient_accumulation_steps` | 4 | 8 | Gradient accumulation |
| `learning_rate` | 2e-4 | 2e-4 | Learning rate |
| `cutoff_len` | 4096 | 4096 | Sequence length |
| `max_samples` | 20000 | 20000 | Dataset size limit |
| `num_train_epochs` | 2 | 2 | Training epochs |

### Dataset Selection

You can use different dataset subsets by modifying the `dataset` parameter:

```yaml
# Full corpus
dataset: titulm_bangla_corpus

# Specific subsets
dataset: titulm_bangla_common_crawl
dataset: titulm_bangla_translated
dataset: titulm_bangla_romanized
```

### Multi-GPU Training

Set the `CUDA_VISIBLE_DEVICES` environment variable:

```bash
# Use GPUs 0,1,2,3
export CUDA_VISIBLE_DEVICES=0,1,2,3
bash examples/train_lora/smollm2_bangla_pretrain.sh

# Use specific GPUs
export CUDA_VISIBLE_DEVICES=1,2,3,4
bash examples/train_lora/smollm2_bangla_pretrain.sh
```

## Monitoring Training

### 1. Log Files
Training logs are saved to:
```
{OUTPUT_DIR}/logs/train_YYYYMMDD_HHMMSS.log
```

Monitor with:
```bash
tail -f saves/smollm2-135m/lora/bangla_pretrain/logs/train_*.log
```

### 2. Weights & Biases
- Project: `smollm2-bangla-pretraining`
- Automatic run naming with timestamps
- Metrics: loss, learning rate, perplexity

### 3. TensorBoard
```bash
tensorboard --logdir saves/smollm2-135m/lora/bangla_pretrain
```

## Customization

### 1. Modify Training Parameters

Edit the YAML configuration files:

```yaml
### train
per_device_train_batch_size: 8        # Adjust based on GPU memory
gradient_accumulation_steps: 4        # Adjust effective batch size
learning_rate: 2.0e-4                 # Learning rate
num_train_epochs: 2                   # Number of epochs
cutoff_len: 4096                      # Sequence length
max_samples: 20000                    # Dataset size limit
```

### 2. Use Different Datasets

Create custom dataset entries in `data/dataset_info.json`:

```json
{
  "my_custom_dataset": {
    "hf_hub_url": "username/dataset-name",
    "columns": {
      "prompt": "text"
    }
  }
}
```

### 3. Adjust Tokenizer Merging

Modify `tokenizer_merger.py` for different merging strategies:

```python
# Use simple merging
create_simple_merged_tokenizer(titulm_path, smollm_path, output_path)

# Use advanced merging with custom parameters
create_merged_tokenizer_config(
    titulm_tokenizer_path=titulm_path,
    smollm_tokenizer_path=smollm_path,
    output_path=output_path,
    max_vocab_size=50000  # Custom vocab size limit
)
```

## Comparison: Original vs LLaMA-Factory

| Feature | Original Script | LLaMA-Factory |
|---------|----------------|---------------|
| **Setup** | Manual environment setup | Automated conda environment |
| **Tokenizer Merging** | Custom implementation | Integrated with training |
| **Training Loop** | Custom Trainer | LLaMA-Factory Trainer |
| **Logging** | Manual wandb setup | Built-in wandb integration |
| **Configuration** | Command-line args | YAML configuration |
| **Multi-GPU** | Manual setup | Automatic DDP |
| **Checkpointing** | Custom implementation | Built-in checkpointing |
| **Evaluation** | Manual setup | Built-in evaluation |

## Testing Your Setup

Before running the full training, test your setup:

```bash
# Run the test script
python test_bangla_pretraining.py
```

This will verify:
- ✅ Environment setup (PyTorch, CUDA, LLaMA-Factory)
- ✅ Configuration files
- ✅ Dataset loading
- ✅ Tokenizer merging

## Troubleshooting

### 1. Command Line Argument Errors
If you see errors like:
```
ValueError: Some keys are not used by the HfArgumentParser: ['--bf16', '--cutoff_len', ...]
```

**Solution**: The scripts have been fixed to use the correct argument format (`key=value` instead of `--key value`).

### 2. Tokenizer Merging Issues
If tokenizer merging shows "0 Bangla-specific tokens":

**Solution**: The improved tokenizer merger now uses a **targeted approach**:
- **TituLM contains**: LLaMA-32K + 48K new Bangla tokens = ~170K total
- **SmolLM2 contains**: English tokens = 49K total  
- **Target**: Extract the 48K unique Bangla tokens from TituLM
- Uses scoring system to identify the most likely Bangla tokens
- Falls back to simple merging if targeted approach fails
- Includes comprehensive Bangla character detection

### 3. Memory Issues
- Reduce `per_device_train_batch_size`
- Increase `gradient_accumulation_steps`
- Use LoRA instead of full fine-tuning
- Reduce `cutoff_len`

### 4. Dataset Loading Issues
- Verify dataset is accessible on HuggingFace Hub
- Check dataset_info.json configuration
- Test with smaller `max_samples` first

### 5. GPU Issues
- Check `CUDA_VISIBLE_DEVICES` setting
- Verify GPU memory availability
- Use `nvidia-smi` to monitor GPU usage

## Next Steps

After pre-training:

1. **Evaluate the model** on Bangla benchmarks
2. **Fine-tune for specific tasks** (instruction following, etc.)
3. **Merge LoRA adapters** if using LoRA
4. **Deploy the model** for inference

## Example Commands

```bash
# Quick start with LoRA
bash examples/train_lora/smollm2_bangla_pretrain.sh

# Monitor training
tail -f saves/smollm2-135m/lora/bangla_pretrain/logs/train_*.log

# Check wandb dashboard
wandb login
# Visit: https://wandb.ai/your-username/smollm2-bangla-pretraining

# Resume training from checkpoint
llamafactory-cli train examples/train_lora/smollm2_bangla_pretrain.yaml \
    --resume_from_checkpoint saves/smollm2-135m/lora/bangla_pretrain/checkpoint-1000
```

This setup provides a robust, scalable solution for Bangla pre-training using LLaMA-Factory while maintaining the key features of your original implementation.
