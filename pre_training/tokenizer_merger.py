
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
    Merge TituLM and SmolLM2 tokenizers to create a unified tokenizer,
    using TituLM as the base.

    Args:
        titulm_tokenizer_path: Path to TituLM tokenizer (used as base)
        smollm_tokenizer_path: Path to SmolLM2 tokenizer (source of additional tokens)
        output_path: Path to save merged tokenizer
        max_added_tokens: Maximum number of regular tokens to add from SmolLM2

    Returns:
        Merged PreTrainedTokenizer
    """

    print("Loading base (TituLM) tokenizer...")
    base_tokenizer = AutoTokenizer.from_pretrained(titulm_tokenizer_path, use_fast=True)
    print(f"Base (TituLM) tokenizer vocab size: {len(base_tokenizer)}")

    print("Loading additional tokens source (SmolLM2) tokenizer...")
    other_tokenizer = AutoTokenizer.from_pretrained(smollm_tokenizer_path, use_fast=True)
    print(f"Additional tokens source (SmolLM2) tokenizer vocab size: {len(other_tokenizer)}")

    # Save original tokenizers
    base_tokenizer.save_pretrained(os.path.join(output_path, "titulm_tokenizer"))
    print(f"TituLM tokenizer saved to: {os.path.join(output_path, 'titulm_tokenizer')}")
    other_tokenizer.save_pretrained(os.path.join(output_path, "smollm_tokenizer"))
    print(f"SmolLM2 tokenizer saved to: {os.path.join(output_path, 'smollm_tokenizer')}")


    base_vocab: Set[str] = set(base_tokenizer.get_vocab().keys())
    other_vocab: Set[str] = set(other_tokenizer.get_vocab().keys())

    # Collect special tokens from SmolLM2 that may be missing in TituLM
    other_special = set()
    for key, value in other_tokenizer.special_tokens_map.items():
        if isinstance(value, list):
            other_special.update(value)
        else:
            other_special.add(value)

    missing_special: List[str] = [tok for tok in other_special if tok and tok not in base_vocab]

    # Add missing special tokens first (if any)
    special_added = 0
    if missing_special:
        special_added = base_tokenizer.add_special_tokens({"additional_special_tokens": missing_special})
        print(f"Added {special_added} special tokens from SmolLM2 to base tokenizer")

    # Add regular tokens up to max_added_tokens
    regular_candidates: List[str] = [tok for tok in other_vocab if tok not in base_vocab and tok not in missing_special]
    newly_added_tokens = []
    if max_added_tokens > 0 and regular_candidates:
        # Keep deterministic order for reproducibility
        regular_candidates.sort()
        to_add = regular_candidates[:max_added_tokens]
        num_added = base_tokenizer.add_tokens(to_add)
        newly_added_tokens.extend(to_add)
        print(f"Added {num_added} regular tokens from SmolLM2 (requested {len(to_add)})")


    os.makedirs(output_path, exist_ok=True)
    base_tokenizer.save_pretrained(output_path)
    print(f"Merged tokenizer saved to: {output_path}")
    print(f"Merged tokenizer vocab size: {len(base_tokenizer)}")


    # Save newly added tokens to a file
    new_tokens_file = os.path.join(output_path, "newly_added_tokens_from_smollm.txt")
    with open(new_tokens_file, "w", encoding="utf-8") as f:
        for token in newly_added_tokens:
            f.write(f"{token}")
    print(f"Newly added tokens from SmolLM2 saved to: {new_tokens_file}")

    # Extract and save Bangla tokens from TituLM tokenizer (which is the base now)
    bangla_tokens = [token for token in base_vocab if any("ঀ" <= char <= "৿" for char in token)]
    bangla_tokens_file = os.path.join(output_path, "titulm_bangla_tokens.txt")
    with open(bangla_tokens_file, "w", encoding="utf-8") as f:
        for token in bangla_tokens:
            f.write(f"{token}")
    print(f"Bangla tokens from TituLM saved to: {bangla_tokens_file}")


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
        # Call the revised merge_tokenizers function
        return merge_tokenizers(
            titulm_tokenizer_path=titulm_tokenizer_path,
            smollm_tokenizer_path=smollm_tokenizer_path,
            output_path=output_path,
        )
    except Exception as e:
        print(f"Error merging tokenizers: {e}")
        print("Falling back to base SmolLM2 tokenizer...")
        # Fallback logic (using SmolLM2 as base, which was the original strategy)
        tokenizer = AutoTokenizer.from_pretrained(smollm_tokenizer_path, use_fast=True)
        os.makedirs(output_path, exist_ok=True)
        tokenizer.save_pretrained(output_path)
        return tokenizer

if __name__ == "__main__":
    # Create merged tokenizer
    merged_tokenizer = create_merged_tokenizer_config()
    print("Tokenizer merging completed!")