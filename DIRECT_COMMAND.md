# Direct Command Equivalent

## Your Original Command:
```bash
nohup python train.py \
    --model_name "google/gemma-3-270m" \
    --output_dir "models/gemma-3-270m-train-v3-16-10" \
    --dataset_name "rnnandi/bn-turn-detection-dataset-train-v3" \
    --num_epochs 40 \
    --per_device_batch_size 8 \
    --hf_token "" \
    > logs/gemma-3-270m-16-10.log 2>&1 &
```

## Equivalent LLaMA Factory Command:

### Option 1: Direct Command (One-liner)
```bash
nohup llamafactory-cli train examples/train_lora/turn_detector_lora_sft.yaml --model_name_or_path="google/gemma-3-270m" --output_dir="models/gemma-3-270m-train-v3-16-10" --dataset="rnnandi/bn-turn-detection-dataset-train-v3" --num_train_epochs=40 --per_device_train_batch_size=8 --hf_token="" --cutoff_len=512 --preprocessing_num_workers=16 --bf16=true --ddp_find_unused_parameters=false --dataloader_pin_memory=true --logging_steps=10 --save_steps=500 --report_to=none > logs/gemma-3-270m-16-10.log 2>&1 &
```

### Option 2: Using the Script
```bash
chmod +x run_turn_detector_training.sh
./run_turn_detector_training.sh
```

### Option 3: Multi-line Command (More Readable)
```bash
nohup llamafactory-cli train \
    examples/train_lora/turn_detector_lora_sft.yaml \
    --model_name_or_path="google/gemma-3-270m" \
    --output_dir="models/gemma-3-270m-train-v3-16-10" \
    --dataset="rnnandi/bn-turn-detection-dataset-train-v3" \
    --num_train_epochs=40 \
    --per_device_train_batch_size=8 \
    --hf_token="" \
    --cutoff_len=512 \
    --preprocessing_num_workers=16 \
    --bf16=true \
    --ddp_find_unused_parameters=false \
    --dataloader_pin_memory=true \
    --logging_steps=10 \
    --save_steps=500 \
    --report_to=none \
    > logs/gemma-3-270m-16-10.log 2>&1 &
```

## Key Differences:

| Parameter | Original Script | LLaMA Factory |
|-----------|----------------|---------------|
| **Model** | `--model_name` | `--model_name_or_path` |
| **Dataset** | `--dataset_name` | `--dataset` |
| **Epochs** | `--num_epochs` | `--num_train_epochs` |
| **Batch Size** | `--per_device_batch_size` | `--per_device_train_batch_size` |
| **Token** | `--hf_token` | `--hf_token` (same) |
| **Output** | `--output_dir` | `--output_dir` (same) |

## Additional LLaMA Factory Benefits:

1. **Automatic Multi-GPU**: No need to manually set device_map
2. **Built-in Logging**: Better logging and monitoring
3. **Checkpointing**: Automatic checkpoint saving
4. **Template Handling**: Automatic chat template setup
5. **Data Processing**: Optimized data loading and preprocessing

## Environment Setup:

Make sure you have the required environment variables:
```bash
export HF_TOKEN=""
export HF_HOME="./hf_cache"  # Optional, for cache location
```

## Monitoring:

```bash
# Check if training is running
ps aux | grep llamafactory

# Monitor logs
tail -f logs/gemma-3-270m-16-10.log

# Monitor with watch
watch -n 1 'tail -n 20 logs/gemma-3-270m-16-10.log'
```
