"""
Tokenizer merger utility for combining SmolLM2 and TituLM tokenizers.
This script merges the vocabulary from TituLM (which has Bangla tokens) 
into the SmolLM2 tokenizer to create a merged tokenizer for Bangla pre-training.
"""

import os
import json
from typing import Dict, List, Set, Optional
from transformers import AutoTokenizer, PreTrainedTokenizer
import shutil


def create_merged_tokenizer_config(
    titulm_tokenizer_path: str,
    smollm_tokenizer_path: str,
    output_path: str,
    max_vocab_size: Optional[int] = None
) -> PreTrainedTokenizer:
    """
    Create a merged tokenizer by combining SmolLM2 base with TituLM Bangla tokens.
    
    Args:
        titulm_tokenizer_path: Path to TituLM tokenizer (source of Bangla tokens)
        smollm_tokenizer_path: Path to SmolLM2 tokenizer (base tokenizer)
        output_path: Path to save the merged tokenizer
        max_vocab_size: Maximum vocabulary size (optional)
    
    Returns:
        Merged tokenizer
    """
    print(f"Loading TituLM tokenizer from: {titulm_tokenizer_path}")
    titulm_tokenizer = AutoTokenizer.from_pretrained(titulm_tokenizer_path)
    
    print(f"Loading SmolLM2 tokenizer from: {smollm_tokenizer_path}")
    smollm_tokenizer = AutoTokenizer.from_pretrained(smollm_tokenizer_path)
    
    print(f"TituLM vocab size: {len(titulm_tokenizer)}")
    print(f"SmolLM2 vocab size: {len(smollm_tokenizer)}")
    
    # Get vocabularies
    titulm_vocab = set(titulm_tokenizer.get_vocab().keys())
    smollm_vocab = set(smollm_tokenizer.get_vocab().keys())
    
    # Find new tokens in TituLM that are not in SmolLM2
    new_tokens = titulm_vocab - smollm_vocab
    print(f"Found {len(new_tokens)} new tokens in TituLM")
    
    # Filter for likely Bangla tokens (basic heuristic)
    bangla_tokens = []
    for token in new_tokens:
        # Simple heuristic: tokens containing Bangla characters or common Bangla patterns
        if any('\u0980' <= char <= '\u09FF' for char in token):  # Bangla Unicode range
            bangla_tokens.append(token)
        elif any(char in token for char in ['ং', 'ঃ', 'ঁ', '়', 'া', 'ি', 'ী', 'ু', 'ূ', 'ৃ', 'ে', 'ৈ', 'ো', 'ৌ']):
            bangla_tokens.append(token)
    
    print(f"Identified {len(bangla_tokens)} Bangla-specific tokens")
    
    # Create merged tokenizer by copying SmolLM2 and adding new tokens
    merged_tokenizer = AutoTokenizer.from_pretrained(smollm_tokenizer_path)
    
    # Add new Bangla tokens
    if bangla_tokens:
        print(f"Adding {len(bangla_tokens)} Bangla tokens to merged tokenizer...")
        added_tokens = merged_tokenizer.add_tokens(bangla_tokens)
        print(f"Successfully added {added_tokens} tokens")
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Save merged tokenizer
    merged_tokenizer.save_pretrained(output_path)
    
    # Save merge info
    merge_info = {
        "base_tokenizer": smollm_tokenizer_path,
        "source_tokenizer": titulm_tokenizer_path,
        "original_vocab_size": len(smollm_tokenizer),
        "new_tokens_count": len(bangla_tokens),
        "final_vocab_size": len(merged_tokenizer),
        "new_tokens": bangla_tokens[:100]  # Save first 100 for reference
    }
    
    with open(os.path.join(output_path, "merge_info.json"), "w", encoding="utf-8") as f:
        json.dump(merge_info, f, indent=2, ensure_ascii=False)
    
    print(f"Merged tokenizer saved to: {output_path}")
    print(f"Final vocab size: {len(merged_tokenizer)}")
    
    return merged_tokenizer


def create_simple_merged_tokenizer(
    titulm_tokenizer_path: str,
    smollm_tokenizer_path: str,
    output_path: str
) -> PreTrainedTokenizer:
    """
    Simplified version that just copies TituLM tokenizer if it's compatible.
    Use this if the above method doesn't work well.
    """
    print(f"Creating simple merged tokenizer...")
    
    # Try to load TituLM tokenizer first
    try:
        tokenizer = AutoTokenizer.from_pretrained(titulm_tokenizer_path)
        print(f"Using TituLM tokenizer directly (vocab size: {len(tokenizer)})")
    except Exception as e:
        print(f"Failed to load TituLM tokenizer: {e}")
        print("Falling back to SmolLM2 tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(smollm_tokenizer_path)
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Save tokenizer
    tokenizer.save_pretrained(output_path)
    
    print(f"Tokenizer saved to: {output_path}")
    return tokenizer


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Merge tokenizers for Bangla pre-training")
    parser.add_argument("--titulm_path", default="hishab/titulm-llama-3.2-3b-v2.0", 
                       help="Path to TituLM tokenizer")
    parser.add_argument("--smollm_path", default="HuggingFaceTB/SmolLM2-135M", 
                       help="Path to SmolLM2 tokenizer")
    parser.add_argument("--output_path", default="./merged_tokenizer", 
                       help="Output path for merged tokenizer")
    parser.add_argument("--simple", action="store_true", 
                       help="Use simple merging strategy")
    
    args = parser.parse_args()
    
    if args.simple:
        create_simple_merged_tokenizer(args.titulm_path, args.smollm_path, args.output_path)
    else:
        create_merged_tokenizer_config(args.titulm_path, args.smollm_path, args.output_path)
