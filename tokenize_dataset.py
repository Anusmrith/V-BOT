"""
================================================================================
V-BOT: Dataset Tokenization Pipeline using Byte Pair Encoding (BPE)
================================================================================
Tokenizes raw text datasets using the trained BPE tokenizer, calculates
sequence compression metrics, splits into training/validation partitions,
and saves PyTorch tensors ready for immediate training.

Usage:
    python tokenize_dataset.py
    python tokenize_dataset.py --input combined_train.txt --vocab_size 2048
    python tokenize_dataset.py --input input.txt --output_prefix shakespeare
================================================================================
"""

import os
import sys
import json
import argparse
import time
import torch

from tokenizer import BPETokenizer, get_tokenizer

def tokenize_file(
    input_path: str = "combined_train.txt",
    tokenizer_path: str = "bpe_tokenizer.json",
    target_vocab_size: int = 2048,
    train_split_ratio: float = 0.9,
    output_prefix: str = "combined_train",
    save_tensors: bool = True,
    retrain_tokenizer: bool = False
):
    """
    Loads text, encodes with BPE tokenizer, partitions into train/val splits,
    and saves .pt tensor artifacts.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input dataset file not found: {input_path}")

    print("=" * 70)
    print("V-BOT DATASET BPE TOKENIZATION PIPELINE")
    print("=" * 70)
    print(f"Dataset Source:     {input_path}")
    print(f"Target Vocab Size:  {target_vocab_size}")
    print(f"Tokenizer Model:    {tokenizer_path}")
    print(f"Train/Val Split:    {int(train_split_ratio * 100)}% / {int((1 - train_split_ratio) * 100)}%")
    print("-" * 70)

    # 1. Load raw text
    t0 = time.time()
    with open(input_path, "r", encoding="utf-8") as f:
        raw_text = f.read().replace("\r\n", "\n").replace("\r", "")

    char_count = len(raw_text)
    byte_count = len(raw_text.encode("utf-8"))
    print(f"[Dataset] Loaded {char_count:,} characters ({byte_count:,} UTF-8 bytes)")

    # 2. Get or Train Tokenizer
    if os.path.exists(tokenizer_path) and not retrain_tokenizer:
        print(f"[Tokenizer] Loading existing BPE tokenizer from: {tokenizer_path}")
        tokenizer = BPETokenizer.load(tokenizer_path)
    else:
        print(f"[Tokenizer] Training fresh BPE tokenizer on {input_path} (vocab size: {target_vocab_size})...")
        tokenizer = BPETokenizer(target_vocab_size=target_vocab_size)
        tokenizer.train(raw_text, verbose=True)
        tokenizer.save(tokenizer_path)

    # 3. Tokenize dataset
    print(f"[Tokenization] Encoding text with BPE vocabulary ({tokenizer.vocab_size} tokens)...")
    enc_t0 = time.time()
    token_ids = tokenizer.encode(raw_text)
    token_count = len(token_ids)
    enc_time = time.time() - enc_t0

    # 4. Metrics & Validation
    compression_ratio = char_count / max(token_count, 1)
    bytes_per_token = byte_count / max(token_count, 1)

    print("-" * 70)
    print("TOKENIZATION SUMMARY & COMPRESSION METRICS")
    print("-" * 70)
    print(f"Total Raw Characters:   {char_count:,}")
    print(f"Total BPE Tokens:       {token_count:,}")
    print(f"Compression Ratio:      {compression_ratio:.2f}x  (chars per token)")
    print(f"Bytes Per Token:        {bytes_per_token:.2f} bytes/token")
    print(f"Tokenization Speed:     {token_count / max(enc_time, 0.001):,.0f} tokens/sec")
    print(f"Context Extension:      With block_size=128, model now sees ~{int(128 * compression_ratio)} characters of context!")

    # Verify round-trip sample fidelity
    sample_len = min(200, len(token_ids))
    decoded_sample = tokenizer.decode(token_ids[:sample_len])
    assert raw_text.startswith(decoded_sample), "Decoded token sample does not match raw text!"
    print(f"[Verification] Round-trip decode test passed successfully! (Lossless)")

    # 5. Convert to PyTorch Tensor & Partition
    data_tensor = torch.tensor(token_ids, dtype=torch.long)
    split_idx = int(train_split_ratio * len(data_tensor))
    train_data = data_tensor[:split_idx]
    val_data = data_tensor[split_idx:]

    print("-" * 70)
    print(f"Training Tokens:    {len(train_data):,} ({train_split_ratio * 100:.1f}%)")
    print(f"Validation Tokens:  {len(val_data):,} ({(1 - train_split_ratio) * 100:.1f}%)")

    # 6. Save Artifacts
    output_files = {}
    if save_tensors:
        train_file = "train_tokens.pt"
        val_file = "val_tokens.pt"
        full_file = f"{output_prefix}_tokens.pt"

        torch.save(train_data, train_file)
        torch.save(val_data, val_file)
        torch.save(data_tensor, full_file)

        output_files["train_tokens"] = train_file
        output_files["val_tokens"] = val_file
        output_files["full_tokens"] = full_file

        print(f"[Output] Saved training tensor:   {train_file}")
        print(f"[Output] Saved validation tensor: {val_file}")
        print(f"[Output] Saved full tensor:       {full_file}")

    # Save metadata JSON
    meta = {
        "dataset_file": input_path,
        "tokenizer_file": tokenizer_path,
        "vocab_size": tokenizer.vocab_size,
        "total_characters": char_count,
        "total_bytes": byte_count,
        "total_tokens": token_count,
        "train_tokens": len(train_data),
        "val_tokens": len(val_data),
        "compression_ratio": round(compression_ratio, 3),
        "bytes_per_token": round(bytes_per_token, 3),
        "train_split_ratio": train_split_ratio,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    meta_path = "tokenized_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    output_files["metadata"] = meta_path
    print(f"[Output] Saved tokenization metadata: {meta_path}")
    print("=" * 70)
    print(f"ALL DONE! Ready to train V-BOT with 'python llm.py'!")
    print("=" * 70)

    return {
        "train_data": train_data,
        "val_data": val_data,
        "tokenizer": tokenizer,
        "metadata": meta,
        "files": output_files
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="V-BOT Dataset BPE Tokenizer Pipeline")
    parser.add_argument("--input", type=str, default="combined_train.txt", help="Input dataset path (default: combined_train.txt)")
    parser.add_argument("--tokenizer", type=str, default="bpe_tokenizer.json", help="Tokenizer model path (default: bpe_tokenizer.json)")
    parser.add_argument("--vocab_size", type=int, default=2048, help="Target vocabulary size (default: 2048)")
    parser.add_argument("--train_split", type=float, default=0.9, help="Train split ratio (default: 0.9)")
    parser.add_argument("--output_prefix", type=str, default="combined_train", help="Prefix for saved token tensors")
    parser.add_argument("--retrain", action="store_true", help="Force retrain the BPE tokenizer on the dataset")
    args = parser.parse_args()

    tokenize_file(
        input_path=args.input,
        tokenizer_path=args.tokenizer,
        target_vocab_size=args.vocab_size,
        train_split_ratio=args.train_split,
        output_prefix=args.output_prefix,
        retrain_tokenizer=args.retrain
    )
