#!/usr/bin/env python3
"""
Create a small Bangla dataset for testing purposes
"""

import json
import os
from datasets import load_dataset

def create_small_bangla_dataset():
    """Create a small Bangla dataset for testing"""
    
    print("Loading titulm_bangla_corpus...")
    try:
        # Load a small subset of the dataset
        dataset = load_dataset("hishab/titulm-bangla-corpus", streaming=True)
        
        # Take only the first 100 samples
        small_samples = []
        for i, sample in enumerate(dataset['train']):
            if i >= 100:  # Limit to 100 samples
                break
            small_samples.append(sample)
        
        print(f"Collected {len(small_samples)} samples")
        
        # Save to local file
        output_file = "data/small_bangla_demo.jsonl"
        os.makedirs("data", exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in small_samples:
                json.dump(sample, f, ensure_ascii=False)
                f.write('\n')
        
        print(f"✅ Small Bangla dataset saved to: {output_file}")
        return output_file
        
    except Exception as e:
        print(f"❌ Failed to create small Bangla dataset: {e}")
        return None

if __name__ == "__main__":
    create_small_bangla_dataset()
