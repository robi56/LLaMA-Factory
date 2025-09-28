"""
Continual pretraining script for `HuggingFaceTB/SmolLM2-135M` on the
`hishab/titulm-bangla-corpus` dataset using a merged tokenizer that
extends SmolLM2 with Bangla tokens from `hishab/titulm-llama-3.2-3b-v2.0`.

Features:
- device_map="auto" with Accelerate integration
- mixed precision (bf16 preferred, fallback fp16)
- gradient checkpointing, gradient accumulation
- dynamic packing via group_texts
- save/resume checkpoints, push_to_hub optional
"""

import os
import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import wandb
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    set_seed,
)

from tokenizer_merger import create_merged_tokenizer_config


@dataclass
class ScriptArgs:
    model_name_or_path: str = field(default="HuggingFaceTB/SmolLM2-135M")
    titulm_tokenizer_ref: str = field(default="hishab/titulm-llama-3.2-3b-v2.0")
    dataset_name: str = field(default="hishab/titulm-bangla-corpus")
    text_column: str = field(default="text")
    subset: Optional[str] = field(default=None)  # e.g., "common_crawl"
    max_samples: Optional[int] = field(default=None)  # Limit total samples
    use_multiple_subsets: bool = field(default=False)  # Use multiple subsets
    common_crawl_ratio: float = field(default=0.5)  # Ratio of common_crawl to use
    other_datasets_ratio: float = field(default=1.0)  # Ratio of other datasets to use
    cache_dir: Optional[str] = field(default=None)
    output_dir: str = field(default="./outputs/smollm2_bangla_continual")
    seed: int = field(default=42)
    block_size: int = field(default=4096)  # Optimal: use more of 8192 limit
    packing: bool = field(default=True)
    bf16: bool = field(default=True)
    fp16: bool = field(default=False)
    learning_rate: float = field(default=2e-4)
    weight_decay: float = field(default=0.1)
    warmup_ratio: float = field(default=0.03)
    num_train_epochs: int = field(default=2)
    per_device_train_batch_size: int = field(default=12)  # Reduced for 7 GPUs
    gradient_accumulation_steps: int = field(default=4)  # Reduced for memory
    logging_steps: int = field(default=50)
    save_steps: int = field(default=1000)
    save_total_limit: int = field(default=3)
    eval_ratio: float = field(default=0.0001)  # small eval sample
    push_to_hub: bool = field(default=False)
    use_wandb: bool = field(default=True)  # Enable wandb logging
    wandb_project: str = field(default="smollm2-bangla-pretraining")  # wandb project name
    wandb_run_name: Optional[str] = field(default=None)  # Custom run name


def main():
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Continual pretraining script')
    parser.add_argument('--use_multiple_subsets', action='store_true', help='Use multiple subsets')
    parser.add_argument('--common_crawl_ratio', type=float, default=0.5, help='Ratio of common_crawl to use')
    parser.add_argument('--other_datasets_ratio', type=float, default=1.0, help='Ratio of other datasets to use')
    parser.add_argument('--max_samples', type=int, default=None, help='Maximum number of samples')
    parser.add_argument('--subset', type=str, default=None, help='Specific subset to use')
    parser.add_argument('--use_wandb', action='store_true', help='Enable wandb logging')
    parser.add_argument('--wandb_project', type=str, default='smollm2-bangla-pretraining', help='wandb project name')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='Custom wandb run name')

    cmd_args = parser.parse_args()

    # Create args object
    args = ScriptArgs()

    # Override with command line arguments
    if cmd_args.use_multiple_subsets:
        args.use_multiple_subsets = True
    if cmd_args.common_crawl_ratio:
        args.common_crawl_ratio = cmd_args.common_crawl_ratio
    if cmd_args.other_datasets_ratio:
        args.other_datasets_ratio = cmd_args.other_datasets_ratio
    if cmd_args.max_samples:
        args.max_samples = cmd_args.max_samples
    if cmd_args.subset:
        args.subset = cmd_args.subset
    if cmd_args.use_wandb:
        args.use_wandb = True
    if cmd_args.wandb_project:
        args.wandb_project = cmd_args.wandb_project
    if cmd_args.wandb_run_name:
        args.wandb_run_name = cmd_args.wandb_run_name

    set_seed(args.seed)

    # Set wandb API key if not already set
    if args.use_wandb and not os.environ.get("WANDB_API_KEY"):
        # Replace with your actual wandb API key
        os.environ["WANDB_API_KEY"] = "d5b3c750c70f5b1c8d317165a8c50a808101648e"
        print("Wandb API key set from code")

    # Initialize wandb if enabled
    if args.use_wandb:
        # Generate run name if not provided
        if args.wandb_run_name is None:
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            args.wandb_run_name = f"smollm2_bangla_{timestamp}"

        # Initialize wandb
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={
                "model_name": args.model_name_or_path,
                "dataset": args.dataset_name,
                "common_crawl_ratio": args.common_crawl_ratio,
                "other_datasets_ratio": args.other_datasets_ratio,
                "max_samples": args.max_samples,
                "block_size": args.block_size,
                "per_device_batch_size": args.per_device_train_batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "learning_rate": args.learning_rate,
                "num_epochs": args.num_train_epochs,
                "bf16": args.bf16,
                "packing": args.packing,
            }
        )
        print(f"Wandb initialized: {args.wandb_project}/{args.wandb_run_name}")

    # 1) Build merged tokenizer (SmolLM2 base + TituLM additions)
    tokenizer_dir = os.path.join(args.output_dir, "merged_tokenizer")
    tokenizer = create_merged_tokenizer_config(
        titulm_tokenizer_path=args.titulm_tokenizer_ref,
        smollm_tokenizer_path=args.model_name_or_path,
        output_path=tokenizer_dir,
    )

    # Ensure a pad token exists for batching/collation
    added_tokens = 0
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            added_tokens += tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    # Save tokenizer with pad token for reproducibility
    try:
        tokenizer.save_pretrained(tokenizer_dir)
    except Exception:
        pass

    # 2) Load model with device_map auto and mixed precision
    dtype = torch.bfloat16 if args.bf16 and torch.cuda.is_available() else (
        torch.float16 if args.fp16 and torch.cuda.is_available() else None
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        device_map = "auto",
        dtype=dtype,
        )

    # If tokenizer grew, resize embeddings
    if hasattr(model, "resize_token_embeddings"):
        model.resize_token_embeddings(len(tokenizer))
    # Set pad token id on config to avoid warnings/errors
    if getattr(model.config, "pad_token_id", None) is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id

    # Enable gradient checkpointing for memory efficiency
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    # 3) Load dataset
    load_kwargs = {"cache_dir": args.cache_dir} if args.cache_dir else {}

    if args.use_multiple_subsets:
        print(f"Loading multiple subsets: {args.common_crawl_ratio*100}% common_crawl + {args.other_datasets_ratio*100}% other datasets")
        print(f"Max samples limit: {args.max_samples}")

        # Load common_crawl with specified ratio
        common_crawl = load_dataset(args.dataset_name, data_dir="common_crawl", **load_kwargs)
        common_crawl_size = int(len(common_crawl["train"]) * args.common_crawl_ratio)
        common_crawl["train"] = common_crawl["train"].select(range(common_crawl_size))
        print(f"Using {common_crawl_size} samples from common_crawl ({args.common_crawl_ratio*100}%)")

        # Load translated with specified ratio
        translated = load_dataset(args.dataset_name, data_dir="translated", **load_kwargs)
        translated_size = int(len(translated["train"]) * args.other_datasets_ratio)
        translated["train"] = translated["train"].select(range(translated_size))
        print(f"Using {translated_size} samples from translated ({args.other_datasets_ratio*100}%)")

        # Load romanized with specified ratio
        romanized = load_dataset(args.dataset_name, data_dir="romanized", **load_kwargs)
        romanized_size = int(len(romanized["train"]) * args.other_datasets_ratio)
        romanized["train"] = romanized["train"].select(range(romanized_size))
        print(f"Using {romanized_size} samples from romanized ({args.other_datasets_ratio*100}%)")

        # Combine datasets
        from datasets import concatenate_datasets
        combined_train = concatenate_datasets([
            common_crawl["train"],
            translated["train"],
            romanized["train"]
        ])

        # Create validation split
        split_count = max(1, int(len(combined_train) * args.eval_ratio))
        dataset = combined_train.train_test_split(test_size=split_count, seed=args.seed)
        dataset = {"train": dataset["train"], "validation": dataset["test"]}

        print(f"Combined dataset: {len(dataset['train'])} train, {len(dataset['validation'])} validation")

    elif args.subset:
        dataset = load_dataset(args.dataset_name, data_dir=args.subset, **load_kwargs)
    else:
        dataset = load_dataset(args.dataset_name, **load_kwargs)

    # Limit dataset size if specified
    if args.max_samples:
        print(f"Limiting dataset to {args.max_samples} samples")
        if "train" in dataset:
            dataset["train"] = dataset["train"].select(range(min(args.max_samples, len(dataset["train"]))))
        if "validation" in dataset:
            dataset["validation"] = dataset["validation"].select(range(min(args.max_samples // 10, len(dataset["validation"]))))

    # Some variants provide only train split. Create a tiny eval split if absent.
    if "validation" not in dataset:
        split_count = max(1, int(len(dataset["train"]) * args.eval_ratio))
        dataset = dataset["train"].train_test_split(test_size=split_count, seed=args.seed)
        dataset = {"train": dataset["train"], "validation": dataset["test"]}

    # 4) Tokenization + packing
    def tokenize_fn(batch):
        # Truncate text at character level first (more efficient)
        texts = []
        for text in batch[args.text_column]:
            # Rough estimate: 4 chars per token, so truncate at block_size * 4
            max_chars = args.block_size * 4
            if len(text) > max_chars:
                text = text[:max_chars]
            texts.append(text)

        return tokenizer(
            texts,
            truncation=True,
            max_length=args.block_size,
            padding=False
        )

    with_dataset = {}
    for split in ["train", "validation"]:
        ds = dataset[split]
        ds = ds.remove_columns([c for c in ds.column_names if c != args.text_column])
        ds = ds.map(tokenize_fn, batched=True, num_proc=os.cpu_count() or 1, remove_columns=[args.text_column])

        if args.packing:
            def group_texts(examples):
                # More robust packing that handles variable lengths
                concatenated = {k: sum(examples[k], []) for k in examples.keys()}
                total_len = len(concatenated["input_ids"])

                # Create blocks of exactly block_size
                result = {k: [] for k in concatenated.keys()}
                for i in range(0, total_len, args.block_size):
                    if i + args.block_size <= total_len:  # Only complete blocks
                        for k in concatenated.keys():
                            result[k].append(concatenated[k][i:i + args.block_size])

                result["labels"] = result["input_ids"].copy()
                return result

            ds = ds.map(group_texts, batched=True, num_proc=os.cpu_count() or 1)
        else:
            # No packing - sequences are already truncated to block_size
            pass

        with_dataset[split] = ds

    # 5) Collator for causal LM
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # 6) Training args
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.num_train_epochs,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=args.bf16,
        fp16=(not args.bf16) and args.fp16,
        report_to=["tensorboard", "wandb"] if args.use_wandb else ["tensorboard"],
        lr_scheduler_type="cosine",
        push_to_hub=args.push_to_hub,
        do_eval=True,
    )

    # 7) Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=with_dataset["train"],
        eval_dataset=with_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=collator,
    )

    train_result = trainer.train(resume_from_checkpoint=False)
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)

    metrics = train_result.metrics
    metrics["train_samples"] = len(with_dataset["train"]) if with_dataset.get("train") is not None else 0
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    eval_metrics = trainer.evaluate()
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)

    # Finish wandb run
    if args.use_wandb:
        wandb.finish()
        print("Wandb run completed")


if __name__ == "__main__":
    main()

