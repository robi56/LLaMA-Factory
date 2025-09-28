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
    
    TituLM contains: LLaMA-32K + 48K new Bangla tokens = ~170K total
    SmolLM2 contains: English tokens = 49K total
    We need to extract the unique Bangla tokens from TituLM.
    
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
    print(f"Found {len(new_tokens)} new tokens in TituLM (not in SmolLM2)")
    
    # Filter for Bangla tokens using multiple strategies
    bangla_tokens = []
    
    # Strategy 1: Check for Bangla Unicode range (U+0980 to U+09FF)
    bangla_unicode_tokens = []
    for token in new_tokens:
        if any('\u0980' <= char <= '\u09FF' for char in token):
            bangla_unicode_tokens.append(token)
    
    print(f"Tokens with Bangla Unicode characters: {len(bangla_unicode_tokens)}")
    bangla_tokens.extend(bangla_unicode_tokens)
    
    # Strategy 2: Check for common Bangla characters and diacritics
    bangla_chars = [
        # Vowels and diacritics
        'া', 'ি', 'ী', 'ু', 'ূ', 'ৃ', 'ে', 'ৈ', 'ো', 'ৌ', 'ং', 'ঃ', 'ঁ', '়',
        # Consonants
        'ক', 'খ', 'গ', 'ঘ', 'ঙ', 'চ', 'ছ', 'জ', 'ঝ', 'ঞ', 'ট', 'ঠ', 'ড', 'ঢ', 'ণ',
        'ত', 'থ', 'দ', 'ধ', 'ন', 'প', 'ফ', 'ব', 'ভ', 'ম', 'য', 'র', 'ল', 'শ', 'ষ', 'স', 'হ',
        # Additional characters
        'ড়', 'ঢ়', 'য়', 'ৎ', 'ক্ষ', 'জ্ঞ'
    ]
    
    bangla_char_tokens = []
    for token in new_tokens:
        if any(char in token for char in bangla_chars):
            if token not in bangla_tokens:  # Avoid duplicates
                bangla_char_tokens.append(token)
    
    print(f"Tokens with Bangla characters: {len(bangla_char_tokens)}")
    bangla_tokens.extend(bangla_char_tokens)
    
    # Strategy 3: Include tokens that are likely Bangla words (more permissive)
    # This catches tokens that might be Bangla but don't match the above patterns
    additional_tokens = []
    for token in new_tokens:
        if (token not in bangla_tokens and  # Not already included
            len(token) > 1 and  # Not single characters
            not token.startswith('<') and  # Not special tokens
            not token.startswith('[') and  # Not special tokens
            not token.startswith('##') and  # Not BERT-style subwords
            not token.isdigit() and  # Not pure numbers
            not all(c.isascii() for c in token)):  # Contains non-ASCII characters
            additional_tokens.append(token)
    
    # Limit additional tokens to avoid including too many non-Bangla tokens
    max_additional = min(10000, len(additional_tokens))  # Limit to 10K additional tokens
    additional_tokens = additional_tokens[:max_additional]
    
    print(f"Additional likely Bangla tokens: {len(additional_tokens)}")
    bangla_tokens.extend(additional_tokens)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_bangla_tokens = []
    for token in bangla_tokens:
        if token not in seen:
            seen.add(token)
            unique_bangla_tokens.append(token)
    
    bangla_tokens = unique_bangla_tokens
    print(f"Total unique Bangla tokens identified: {len(bangla_tokens)}")
    
    # Show some examples
    if bangla_tokens:
        print("Sample Bangla tokens:")
        for i, token in enumerate(bangla_tokens[:10]):
            print(f"  {i+1}: '{token}'")
        if len(bangla_tokens) > 10:
            print(f"  ... and {len(bangla_tokens) - 10} more")
    
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
    Simplified version that extracts only the new Bangla tokens from TituLM.
    This approach is more conservative and focuses on the 48K new tokens.
    """
    print(f"Creating simple merged tokenizer...")
    
    # Load both tokenizers
    titulm_tokenizer = AutoTokenizer.from_pretrained(titulm_tokenizer_path)
    smollm_tokenizer = AutoTokenizer.from_pretrained(smollm_tokenizer_path)
    
    print(f"TituLM vocab size: {len(titulm_tokenizer)}")
    print(f"SmolLM2 vocab size: {len(smollm_tokenizer)}")
    
    # Get vocabularies
    titulm_vocab = set(titulm_tokenizer.get_vocab().keys())
    smollm_vocab = set(smollm_tokenizer.get_vocab().keys())
    
    # Find tokens that are in TituLM but not in SmolLM2
    new_tokens = titulm_vocab - smollm_vocab
    print(f"Found {len(new_tokens)} tokens in TituLM that are not in SmolLM2")
    
    # Filter for Bangla tokens more conservatively
    bangla_tokens = []
    
    # Only include tokens that contain Bangla Unicode characters
    for token in new_tokens:
        if any('\u0980' <= char <= '\u09FF' for char in token):
            bangla_tokens.append(token)
    
    print(f"Conservative Bangla token count: {len(bangla_tokens)}")
    
    # Create merged tokenizer
    merged_tokenizer = AutoTokenizer.from_pretrained(smollm_tokenizer_path)
    
    # Add Bangla tokens
    if bangla_tokens:
        added_count = merged_tokenizer.add_tokens(bangla_tokens)
        print(f"Added {added_count} Bangla tokens to merged tokenizer")
    else:
        print("No Bangla tokens found, using SmolLM2 tokenizer as-is")
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Save tokenizer
    merged_tokenizer.save_pretrained(output_path)
    
    print(f"Final merged tokenizer vocab size: {len(merged_tokenizer)}")
    print(f"Tokenizer saved to: {output_path}")
    
    return merged_tokenizer


def create_targeted_merged_tokenizer(
    titulm_tokenizer_path: str,
    smollm_tokenizer_path: str,
    output_path: str,
    target_bangla_tokens: int = 48000
) -> PreTrainedTokenizer:
    """
    Create a merged tokenizer targeting the specific 48K new Bangla tokens.
    This is the most precise approach for your use case.
    """
    print(f"Creating targeted merged tokenizer (targeting {target_bangla_tokens} Bangla tokens)...")
    
    # Load both tokenizers
    titulm_tokenizer = AutoTokenizer.from_pretrained(titulm_tokenizer_path)
    smollm_tokenizer = AutoTokenizer.from_pretrained(smollm_tokenizer_path)
    
    print(f"TituLM vocab size: {len(titulm_tokenizer)}")
    print(f"SmolLM2 vocab size: {len(smollm_tokenizer)}")
    
    # Get vocabularies
    titulm_vocab = set(titulm_tokenizer.get_vocab().keys())
    smollm_vocab = set(smollm_tokenizer.get_vocab().keys())
    
    # Find tokens that are in TituLM but not in SmolLM2
    new_tokens = titulm_vocab - smollm_vocab
    print(f"Found {len(new_tokens)} tokens in TituLM that are not in SmolLM2")
    
    # Sort tokens by likelihood of being Bangla
    bangla_candidates = []
    
    for token in new_tokens:
        bangla_score = 0
        
        # Score based on Bangla Unicode characters
        bangla_chars = sum(1 for char in token if '\u0980' <= char <= '\u09FF')
        bangla_score += bangla_chars * 10
        
        # Score based on common Bangla characters
        common_bangla = ['া', 'ি', 'ী', 'ু', 'ূ', 'ৃ', 'ে', 'ৈ', 'ো', 'ৌ', 'ং', 'ঃ', 'ঁ', '়',
                        'ক', 'খ', 'গ', 'ঘ', 'ঙ', 'চ', 'ছ', 'জ', 'ঝ', 'ঞ', 'ট', 'ঠ', 'ড', 'ঢ', 'ণ',
                        'ত', 'থ', 'দ', 'ধ', 'ন', 'প', 'ফ', 'ব', 'ভ', 'ম', 'য', 'র', 'ল', 'শ', 'ষ', 'স', 'হ']
        bangla_score += sum(1 for char in token if char in common_bangla) * 5
        
        # Penalize tokens that look like English
        if token.isascii() and not any('\u0980' <= char <= '\u09FF' for char in token):
            bangla_score -= 5
        
        # Penalize special tokens
        if token.startswith('<') or token.startswith('['):
            bangla_score -= 10
        
        if bangla_score > 0:
            bangla_candidates.append((token, bangla_score))
    
    # Sort by score (highest first)
    bangla_candidates.sort(key=lambda x: x[1], reverse=True)
    
    # Take the top tokens up to the target
    selected_tokens = [token for token, score in bangla_candidates[:target_bangla_tokens]]
    
    print(f"Selected {len(selected_tokens)} Bangla tokens (target: {target_bangla_tokens})")
    
    # Show some examples
    if selected_tokens:
        print("Sample selected Bangla tokens:")
        for i, token in enumerate(selected_tokens[:10]):
            print(f"  {i+1}: '{token}'")
        if len(selected_tokens) > 10:
            print(f"  ... and {len(selected_tokens) - 10} more")
    
    # Create merged tokenizer
    merged_tokenizer = AutoTokenizer.from_pretrained(smollm_tokenizer_path)
    
    # Add selected Bangla tokens
    if selected_tokens:
        added_count = merged_tokenizer.add_tokens(selected_tokens)
        print(f"Added {added_count} Bangla tokens to merged tokenizer")
    else:
        print("No Bangla tokens selected, using SmolLM2 tokenizer as-is")
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Save tokenizer
    merged_tokenizer.save_pretrained(output_path)
    
    print(f"Final merged tokenizer vocab size: {len(merged_tokenizer)}")
    print(f"Tokenizer saved to: {output_path}")
    
    return merged_tokenizer


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
