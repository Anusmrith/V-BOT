"""
================================================================================
Dataset Builder & Preprocessor for V-BOT
================================================================================
Merges, formats, and analyzes datasets for training your transformer model.
================================================================================
"""

import os
import sys

def analyze_file(filepath):
    """Analyzes character and vocabulary stats for a given text file."""
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return None
    
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
    
    unique_chars = sorted(list(set(text)))
    lines = text.count('\n') + 1
    words = len(text.split())
    
    return {
        "chars": len(text),
        "vocab_size": len(unique_chars),
        "lines": lines,
        "words": words,
        "unique_chars": "".join(unique_chars[:30]) + ("..." if len(unique_chars) > 30 else "")
    }

def merge_datasets(output_file="combined_train.txt", repeat_qa=10):
    """
    Combines Q&A dataset with domain texts, repeating high-value Q&A pairs
    so the model develops strong conversational habits.
    """
    qa_path = "qa_dataset.txt"
    data_path = "data.txt"
    
    combined = []
    
    if os.path.exists(qa_path):
        with open(qa_path, 'r', encoding='utf-8') as f:
            qa_text = f.read().strip()
        print(f"Loaded {qa_path} ({len(qa_text):,} chars). Repeating {repeat_qa}x to reinforce assistant behavior.")
        for _ in range(repeat_qa):
            combined.append(qa_text)
            combined.append("\n\n")
    
    if os.path.exists(data_path):
        with open(data_path, 'r', encoding='utf-8') as f:
            domain_text = f.read().strip()
        print(f"Loaded {data_path} ({len(domain_text):,} chars).")
        for _ in range(repeat_qa // 2):
            combined.append(domain_text)
            combined.append("\n\n")
            
    final_text = "".join(combined).strip()
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(final_text)
        
    print(f"\nSuccessfully generated: {output_file}")
    stats = analyze_file(output_file)
    print(f"Total Characters: {stats['chars']:,}")
    print(f"Total Words:      {stats['words']:,}")
    print(f"Vocabulary Size:  {stats['vocab_size']} unique characters")
    return output_file

if __name__ == "__main__":
    print("=" * 60)
    print("V-BOT DATASET BUILDER")
    print("=" * 60)
    
    print("\n[Current Files Analysis]:")
    for fname in ["qa_dataset.txt", "data.txt", "input.txt"]:
        if os.path.exists(fname):
            s = analyze_file(fname)
            print(f" • {fname:<15} : {s['chars']:,} chars | {s['words']:,} words | {s['vocab_size']} vocab")
    
    print("\nBuilding reinforced training corpus...")
    out = merge_datasets("combined_train.txt", repeat_qa=15)
    print(f"\nReady to train! You can upload '{out}' to Google Colab or use it in llm.py.")
