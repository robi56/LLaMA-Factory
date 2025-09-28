"""
Tokenizer merging utility for TituLM Bangla corpus and SmolLM2-135M

This merges the vocabularies by adding missing tokens from the TituLM tokenizer
into the SmolLM2 tokenizer using the supported `add_tokens`/`add_special_tokens`
APIs. We do NOT mutate internal vocab tables directly because different tokenizer
implementations (e.g., SentencePiece/BPE) have distinct constraints.
"""

import os
from typing import List, Set
from transformers import AutoTokenizer, PreTrainedTokenizerBase


def merge_tokenizers(
    titulm_tokenizer_path: str,
    smollm_tokenizer_path: str,
    output_path: str,
    max_added_tokens: int = 48000,
) -> PreTrainedTokenizerBase:
    """
    Merge TituLM and SmolLM2 tokenizers to create a unified tokenizer.
    
    Args:
        titulm_tokenizer_path: Path to TituLM tokenizer
        smollm_tokenizer_path: Path to SmolLM2 tokenizer  
        output_path: Path to save merged tokenizer
        max_vocab_size: Maximum vocabulary size for merged tokenizer
        
    Returns:
        Merged PreTrainedTokenizer
    """
    
    print("Loading base (SmolLM2) tokenizer...")
    base_tokenizer = AutoTokenizer.from_pretrained(smollm_tokenizer_path, use_fast=True)
    print("Loading TituLM tokenizer (source of additional tokens)...")
    titulm_tokenizer = AutoTokenizer.from_pretrained(titulm_tokenizer_path, use_fast=True)

    base_vocab: Set[str] = set(base_tokenizer.get_vocab().keys())
    titulm_vocab: Set[str] = set(titulm_tokenizer.get_vocab().keys())

    # Collect special tokens from TituLM that may be missing
    titulm_special = set()
    for key, value in titulm_tokenizer.special_tokens_map.items():
        if isinstance(value, list):
            titulm_special.update(value)
        else:
            titulm_special.add(value)

    missing_special: List[str] = [tok for tok in titulm_special if tok and tok not in base_vocab]

    # Add missing special tokens first (if any)
    special_added = 0
    if missing_special:
        special_added = base_tokenizer.add_special_tokens({"additional_special_tokens": missing_special})
        print(f"Added {special_added} special tokens from TituLM to base tokenizer")

    # Add regular tokens up to max_added_tokens
    regular_candidates: List[str] = [tok for tok in titulm_vocab if tok not in base_vocab and tok not in missing_special]
    if max_added_tokens > 0 and regular_candidates:
        # Keep deterministic order for reproducibility
        regular_candidates.sort()
        to_add = regular_candidates[:max_added_tokens]
        num_added = base_tokenizer.add_tokens(to_add)
        print(f"Added {num_added} regular tokens from TituLM (requested {len(to_add)})")

    os.makedirs(output_path, exist_ok=True)
    base_tokenizer.save_pretrained(output_path)
    print(f"Merged tokenizer saved to: {output_path}")

    return base_tokenizer


def create_merged_tokenizer_config(
    titulm_tokenizer_path: str = "hishab/titulm-llama-3.2-3b-v2.0",
    smollm_tokenizer_path: str = "HuggingFaceTB/SmolLM2-135M",
    output_path: str = "./merged_tokenizer"
) -> PreTrainedTokenizerBase:
    """
    Create merged tokenizer configuration for TituLM + SmolLM2.
    
    Args:
        titulm_tokenizer_path: HuggingFace path to TituLM tokenizer
        smollm_tokenizer_path: HuggingFace path to SmolLM2 tokenizer
        output_path: Local path to save merged tokenizer
        
    Returns:
        Merged PreTrainedTokenizer
    """
    
    try:
        return merge_tokenizers(
            titulm_tokenizer_path=titulm_tokenizer_path,
            smollm_tokenizer_path=smollm_tokenizer_path,
            output_path=output_path,
        )
    except Exception as e:
        print(f"Error merging tokenizers: {e}")
        print("Falling back to base SmolLM2 tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(smollm_tokenizer_path, use_fast=True)
        os.makedirs(output_path, exist_ok=True)
        tokenizer.save_pretrained(output_path)
        return tokenizer


if __name__ == "__main__":
    # Create merged tokenizer
    merged_tokenizer = create_merged_tokenizer_config()
    print("Tokenizer merging completed!")
