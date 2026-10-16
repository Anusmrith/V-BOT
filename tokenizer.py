"""
================================================================================
V-BOT: Pure Python Byte-Level Byte Pair Encoding (BPE) Tokenizer
================================================================================
A self-contained, zero-dependency Byte-Level BPE tokenizer built from scratch.
Adheres to V-BOT's pure PyTorch / no-wrapper architecture principles.

Features:
- 256 Base Byte Tokens (0..255) ensures 100% UTF-8 coverage without [UNK] tokens
- Special token support (<|endoftext|>, <|pad|>, and reserved tokens)
- GPT-2 style regex pre-tokenization for clean word & whitespace boundaries
- Ultra-fast word-frequency weighted pair counting and merging
- Lossless encode/decode round-trip
- JSON serialization for checkpoint & inference portability
================================================================================
"""

import os
import sys
import re
import json
import argparse
from collections import Counter, defaultdict

# GPT-2 style regex splitting pattern (pure Python re compatible)
# Splits on contractions, words with optional leading space, punctuation, or spaces
GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\w+| ?[^\s\w]+|\s+(?!\S)|\s+"""

class BPETokenizer:
    """
    Byte-Level Byte Pair Encoding (BPE) Tokenizer.
    
    Attributes:
        target_vocab_size (int): Desired vocabulary size (e.g. 2048).
        merges (dict): Maps adjacent token pairs (p0, p1) -> new_token_id.
        vocab (dict): Maps token_id (int) -> bytes representation.
        special_tokens (dict): Maps special string (e.g. '<|endoftext|>') -> token_id.
    """
    def __init__(self, target_vocab_size=2048):
        self.target_vocab_size = target_vocab_size
        self.pattern = re.compile(GPT2_SPLIT_PATTERN)
        self.special_tokens = {"<|endoftext|>": 256}
        self.inverse_special_tokens = {256: "<|endoftext|>"}
        self.merges = {}   # tuple(int, int) -> int
        self.vocab = {}    # int -> bytes
        self._init_base_vocab()

    def _init_base_vocab(self):
        """Initializes the base vocabulary with 256 bytes and default special tokens."""
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.vocab[256] = b"<|endoftext|>"

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def train(self, text: str, min_frequency: int = 2, pad_to_target: bool = True, verbose: bool = True):
        """
        Learns BPE merges from input text until target_vocab_size is reached or
        no recurring pairs remain.
        """
        self._init_base_vocab()
        self.merges = {}

        if not text:
            raise ValueError("Cannot train BPE tokenizer on empty text.")

        # Pre-tokenize text using regex
        words = self.pattern.findall(text)
        word_counts = Counter(words)
        
        # Represent words as lists of byte integers, indexed by word_id
        word_tuples = [list(w.encode("utf-8")) for w in word_counts.keys()]
        word_freqs = list(word_counts.values())

        # Inverted index: pair -> set of word_ids, and pair -> total frequency
        pair_counts = defaultdict(int)
        pair_to_words = defaultdict(set)
        for wid, (w, freq) in enumerate(zip(word_tuples, word_freqs)):
            for p in zip(w, w[1:]):
                pair_counts[p] += freq
                pair_to_words[p].add(wid)

        base_size = len(self.vocab)
        max_merges = max(0, self.target_vocab_size - base_size)

        if verbose:
            print(f"[BPETokenizer] Training on {len(words):,} words ({len(word_tuples):,} unique). Target: {self.target_vocab_size} tokens.")

        for i in range(max_merges):
            if not pair_counts:
                if verbose:
                    print(f"[BPETokenizer] All words fully merged at iteration {i}.")
                break

            best_pair = max(pair_counts, key=pair_counts.get)
            count = pair_counts[best_pair]

            if count < min_frequency:
                if verbose:
                    print(f"[BPETokenizer] Best pair frequency ({count}) < threshold ({min_frequency}) at iteration {i}.")
                break

            new_id = base_size + i
            self.merges[best_pair] = new_id
            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]

            affected_words = list(pair_to_words[best_pair])
            del pair_counts[best_pair]
            del pair_to_words[best_pair]

            p0, p1 = best_pair
            for wid in affected_words:
                w = word_tuples[wid]
                freq = word_freqs[wid]

                # Decrement old pairs of this word
                for p in zip(w, w[1:]):
                    pair_counts[p] -= freq
                    if pair_counts[p] <= 0:
                        pair_counts.pop(p, None)
                    pair_to_words[p].discard(wid)
                    if not pair_to_words[p]:
                        pair_to_words.pop(p, None)

                # Merge p0, p1 in w
                new_w = []
                j = 0
                n = len(w)
                while j < n:
                    if j < n - 1 and w[j] == p0 and w[j + 1] == p1:
                        new_w.append(new_id)
                        j += 2
                    else:
                        new_w.append(w[j])
                        j += 1
                word_tuples[wid] = new_w

                # Add new pairs
                for p in zip(new_w, new_w[1:]):
                    pair_counts[p] += freq
                    pair_to_words[p].add(wid)

        # If training stopped before target_vocab_size and pad_to_target is requested,
        # populate the remaining slots with reserved tokens to ensure fixed embedding dimensions.
        if pad_to_target and len(self.vocab) < self.target_vocab_size:
            pad_count = 0
            for idx in range(len(self.vocab), self.target_vocab_size):
                tag = f"<|reserved_{pad_count}|>"
                self.vocab[idx] = tag.encode("utf-8")
                self.special_tokens[tag] = idx
                self.inverse_special_tokens[idx] = tag
                pad_count += 1
            if verbose and pad_count > 0:
                print(f"[BPETokenizer] Added {pad_count} reserved tokens to guarantee exact vocab size {self.target_vocab_size}.")

        if verbose:
            print(f"[BPETokenizer] Complete: {self.vocab_size} tokens ({len(self.merges)} learned merges).")

    def _encode_chunk(self, byte_list: bytes) -> list:
        """Applies learned merges greedily to a single regex chunk of bytes."""
        tokens = list(byte_list)
        while len(tokens) >= 2:
            min_pair = None
            min_rank = float("inf")
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                rank = self.merges.get(pair)
                if rank is not None and rank < min_rank:
                    min_rank = rank
                    min_pair = pair

            if min_pair is None:
                break

            new_tokens = []
            i = 0
            p0, p1 = min_pair
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == p0 and tokens[i + 1] == p1:
                    new_tokens.append(min_rank)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        return tokens

    def encode(self, text: str, allowed_special: bool = True) -> list:
        """
        Encodes an input string into a list of integer token IDs.
        Handles both regular text and optional special tokens like <|endoftext|>.
        """
        if not text:
            return []

        # Handle special tokens if enabled
        if allowed_special:
            for spec_str, spec_id in self.special_tokens.items():
                if spec_str in text:
                    parts = text.split(spec_str)
                    tokens = []
                    for i, part in enumerate(parts):
                        if part:
                            tokens.extend(self.encode(part, allowed_special=False))
                        if i < len(parts) - 1:
                            tokens.append(spec_id)
                    return tokens

        tokens = []
        for match in self.pattern.finditer(text):
            chunk = match.group(0).encode("utf-8")
            tokens.extend(self._encode_chunk(chunk))
        return tokens

    def decode(self, tokens: list) -> str:
        """Decodes a sequence of integer token IDs back into a UTF-8 string."""
        if not tokens:
            return ""

        parts = []
        for t in tokens:
            b = self.vocab.get(t)
            if b is not None:
                parts.append(b)
        return b"".join(parts).decode("utf-8", errors="replace")

    def get_stoi(self) -> dict:
        """Returns string/representation to ID mapping for backward compatibility."""
        return {self.decode([i]): i for i in range(self.vocab_size)}

    def get_itos(self) -> dict:
        """Returns ID to string/representation mapping for backward compatibility."""
        return {i: self.decode([i]) for i in range(self.vocab_size)}

    def save(self, filepath: str):
        """Saves the trained BPE tokenizer state to a JSON file."""
        merges_serializable = [[p[0], p[1], new_id] for p, new_id in self.merges.items()]
        vocab_serializable = {str(k): v.hex() for k, v in self.vocab.items()}
        data = {
            "target_vocab_size": self.target_vocab_size,
            "vocab_size": self.vocab_size,
            "special_tokens": self.special_tokens,
            "merges": merges_serializable,
            "vocab": vocab_serializable
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"[BPETokenizer] Successfully saved tokenizer model to: {filepath}")

    @classmethod
    def load(cls, filepath: str):
        """Loads a BPE tokenizer model from a JSON file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Tokenizer checkpoint not found: {filepath}")

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        tok = cls(target_vocab_size=data.get("target_vocab_size", 2048))
        tok.special_tokens = data.get("special_tokens", {"<|endoftext|>": 256})
        tok.inverse_special_tokens = {v: k for k, v in tok.special_tokens.items()}
        tok.merges = {(p0, p1): new_id for p0, p1, new_id in data["merges"]}
        tok.vocab = {int(k): bytes.fromhex(v) for k, v in data["vocab"].items()}
        return tok


def get_tokenizer(
    model_path: str = "bpe_tokenizer.json",
    corpus_path: str = "combined_train.txt",
    target_vocab_size: int = 2048
) -> BPETokenizer:
    """
    Convenience function: loads a saved BPE tokenizer if available,
    otherwise trains a fresh tokenizer from corpus_path and saves it.
    """
    if os.path.exists(model_path):
        try:
            tok = BPETokenizer.load(model_path)
            return tok
        except Exception as e:
            print(f"[BPETokenizer] Failed to load '{model_path}': {e}. Retraining...")

    if not os.path.exists(corpus_path):
        for candidate in ["combined_train.txt", "input.txt", "data.txt"]:
            if os.path.exists(candidate):
                corpus_path = candidate
                break

    if not os.path.exists(corpus_path):
        raise FileNotFoundError(f"Cannot train BPE tokenizer: corpus '{corpus_path}' not found!")

    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus_text = f.read()

    tokenizer = BPETokenizer(target_vocab_size=target_vocab_size)
    tokenizer.train(corpus_text, verbose=True)
    tokenizer.save(model_path)
    return tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="V-BOT Byte Pair Encoding (BPE) Tokenizer CLI")
    parser.add_argument("--train", action="store_true", help="Train a new BPE tokenizer on a text corpus")
    parser.add_argument("--corpus", type=str, default="combined_train.txt", help="Path to training text corpus")
    parser.add_argument("--vocab_size", type=int, default=2048, help="Target vocabulary size")
    parser.add_argument("--save", type=str, default="bpe_tokenizer.json", help="Path to save tokenizer JSON model")
    parser.add_argument("--test", type=str, default="User: What is Vishwas AI?\nAssistant: Trust is built on evidence.", help="Test text to encode & decode")
    args = parser.parse_args()

    if args.train:
        print(f"Loading corpus from: {args.corpus}")
        with open(args.corpus, "r", encoding="utf-8") as f:
            text = f.read()
        tok = BPETokenizer(target_vocab_size=args.vocab_size)
        tok.train(text, verbose=True)
        tok.save(args.save)
    else:
        tok = get_tokenizer(model_path=args.save, corpus_path=args.corpus, target_vocab_size=args.vocab_size)

    # Run quick test
    print("\n--- Tokenizer Test Run ---")
    test_input = args.test
    encoded = tok.encode(test_input)
    decoded = tok.decode(encoded)
    print(f"Original Text: {test_input!r} ({len(test_input)} chars)")
    print(f"Encoded IDs:   {encoded} ({len(encoded)} tokens)")
    print(f"Decoded Text:  {decoded!r}")
    print(f"Lossless Match: {decoded == test_input}")
    print(f"Compression:    {len(test_input) / max(len(encoded), 1):.2f}x chars/token")
