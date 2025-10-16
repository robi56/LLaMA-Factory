# Turn Detector Training with LLaMA Factory

This guide shows you how to train a turn detector model using LLaMA Factory, converted from your original Hugging Face training script.

## Overview

Your original training setup has been converted to work with LLaMA Factory:
- **Model**: hishab/titulm-llama-3.2-1b-v1.1 (your original model)
- **Dataset**: Turn detector conversations in ShareGPT format
- **Training**: Supervised Fine-Tuning (SFT) with LoRA or Full fine-tuning
- **Format**: ChatML template compatible with your original Livekit format

## Files Created

### 1. Dataset Configuration (`data/dataset_info.json`)
Added entry for:
- `turn_detector_demo` - Demo dataset with turn detection conversations

### 2. LLaMA Factory Configuration Files
- `examples/train_lora/turn_detector_lora_sft.yaml` - LoRA SFT config
- `examples/train_full/turn_detector_full_sft.yaml` - Full SFT config

### 3. Training Scripts
- `examples/train_lora/turn_detector_lora_sft.sh` - LoRA training script
- `examples/train_full/turn_detector_full_sft.sh` - Full training script

### 4. Chat Utilities (`chat_utils.py`)
- Functions to convert your original chat format to ShareGPT format
- Compatible with your original Livekit ChatML template

### 5. Example Dataset (`data/turn_detector_demo.json`)
- Sample conversations about turn detection
- Formatted in ShareGPT format for LLaMA Factory

## Usage

### Option 1: LoRA Training (Recommended)

```bash
# Make script executable
chmod +x examples/train_lora/turn_detector_lora_sft.sh

# Set your environment variables
export WANDB_API_KEY="your_wandb_key_here"
export HF_HOME="./hf_cache"

# Run LoRA training
bash examples/train_lora/turn_detector_lora_sft.sh
```

### Option 2: Full Fine-tuning

```bash
# Make script executable
chmod +x examples/train_full/turn_detector_full_sft.sh

# Set your environment variables
export WANDB_API_KEY="your_wandb_key_here"
export HF_HOME="./hf_cache"

# Run full fine-tuning
bash examples/train_full/turn_detector_full_sft.sh
```

### Option 3: Direct LLaMA Factory CLI

```bash
# LoRA training
llamafactory-cli train examples/train_lora/turn_detector_lora_sft.yaml

# Full training
llamafactory-cli train examples/train_full/turn_detector_full_sft.yaml
```

## Configuration Details

### Key Differences from Your Original Script

| Feature | Original Script | LLaMA Factory |
|---------|----------------|---------------|
| **Model Loading** | Manual AutoModelForCausalLM | Automatic with LLaMA Factory |
| **Tokenizer Setup** | Manual special tokens | Automatic template handling |
| **Data Processing** | Custom format_chat_and_tokenize | Built-in ShareGPT processing |
| **Training Loop** | Custom Trainer | LLaMA Factory Trainer |
| **Configuration** | Command-line arguments | YAML configuration |
| **Multi-GPU** | Manual device_map | Automatic DDP |

### Training Parameters

The configuration maintains your original training parameters:
- **Learning Rate**: 2e-5 (same as original)
- **Batch Size**: 2 per device (same as original)
- **Epochs**: 20 (same as original)
- **Sequence Length**: 512 (same as original)
- **Mixed Precision**: bf16 (equivalent to your fp16)

### Chat Template Compatibility

Your original Livekit ChatML template:
```
{% for message in messages %}{{'<|im_start|>' + '<|' + message['role'] + '|>' + message['content'] + '<|im_end|>'}}{% endfor %}
```

Is automatically handled by LLaMA Factory's llama3 template, which produces the same format.

## Customizing for Your Dataset

### 1. Replace Demo Dataset

To use your actual turn detector dataset:

1. **Convert your dataset to ShareGPT format**:
```python
from chat_utils import convert_to_sharegpt_format

# Your original conversations
your_conversations = [
    {
        "conversations": [
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."}
        ]
    }
]

# Convert to ShareGPT format
sharegpt_data = convert_to_sharegpt_format(your_conversations)
```

2. **Update dataset configuration**:
```json
{
  "your_turn_detector_dataset": {
    "hf_hub_url": "your_username/your_dataset",
    "formatting": "sharegpt",
    "columns": {
      "messages": "conversations"
    }
  }
}
```

3. **Update YAML configs**:
```yaml
dataset: your_turn_detector_dataset
```

### 2. Adjust Training Parameters

Edit the YAML files to match your needs:

```yaml
### train
per_device_train_batch_size: 4        # Increase if you have more GPU memory
gradient_accumulation_steps: 4        # Adjust effective batch size
learning_rate: 1.0e-5                 # Lower learning rate for fine-tuning
num_train_epochs: 10                  # Adjust based on your dataset size
cutoff_len: 1024                      # Increase for longer conversations
max_samples: 50000                    # Limit dataset size if needed
```

### 3. Add Evaluation

Uncomment and configure evaluation in the YAML files:

```yaml
### eval
eval_dataset: turn_detector_demo
val_size: 0.1
per_device_eval_batch_size: 2
eval_strategy: steps
eval_steps: 500
```

## Monitoring Training

### 1. Weights & Biases
```bash
# Set your WANDB API key
export WANDB_API_KEY="your_key_here"

# Training will automatically log to WANDB
```

### 2. TensorBoard
```bash
tensorboard --logdir saves/turn-detector/lora/sft
```

### 3. Log Files
```bash
# Monitor training logs
tail -f saves/turn-detector/lora/sft/logs/train_*.log
```

## Model Inference

After training, you can use the model for inference:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load your trained model
model_path = "saves/turn-detector/lora/sft"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# Use for turn detection inference
def detect_turns(conversation_text):
    # Your inference logic here
    pass
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce `per_device_train_batch_size` or increase `gradient_accumulation_steps`
2. **Dataset Loading Errors**: Check your dataset format matches ShareGPT structure
3. **Template Errors**: Ensure your model supports the llama3 template

### Getting Help

- Check LLaMA Factory documentation: https://github.com/hiyouga/LLaMA-Factory
- Review your original training script for reference
- Use the chat_utils.py functions to debug data formatting

## Next Steps

1. **Test with demo dataset**: Start with the provided demo to ensure everything works
2. **Prepare your dataset**: Convert your turn detector conversations to ShareGPT format
3. **Run training**: Choose LoRA or full fine-tuning based on your needs
4. **Evaluate results**: Test the trained model on your turn detection tasks
5. **Deploy**: Integrate the trained model into your turn detection system
