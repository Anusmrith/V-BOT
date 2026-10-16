"""
================================================================================
Single-Line Prompt CLI for your GPT Model
================================================================================
Allows you to ask questions directly from PowerShell with one command:
Example:
    python ask.py Vishwas AI is
    python ask.py Trust means
================================================================================
"""

import sys
import os
# pyrefly: ignore [missing-import]
import torch

from chat import load_trained_model, ask_model

def main():
    if len(sys.argv) > 1:
        # Combine all command line words into a single prompt string
        raw_prompt = " ".join(sys.argv[1:]).strip().replace('\\n', '\n')
    else:
        # Default prompt if none provided
        raw_prompt = "What is your name?"

    # Distinguish between theatrical dialogue prompts and conversational user questions
    is_theatrical = any(tag in raw_prompt for tag in [":\n", "Citizen:", "ROMEO:", "JULIET:", "HAMLET:", "All:", "BRUTUS:"]) or raw_prompt.endswith(":")
    
    if is_theatrical:
        prompt = raw_prompt
        stop_seqs = ["<|endoftext|>"]
        prefix = ""
    elif not raw_prompt.startswith("User:") and "Assistant:" not in raw_prompt:
        prompt = f"User: {raw_prompt}\nAssistant:"
        stop_seqs = ["\nUser:", "\n\nUser:", "<|endoftext|>"]
        prefix = "Assistant: "
    else:
        prompt = raw_prompt
        stop_seqs = ["\nUser:", "\n\nUser:", "<|endoftext|>"]
        prefix = "Assistant: " if not prompt.endswith("Assistant:") else ""

    model = load_trained_model("model.pt")
    
    print(f"\n[Prompt]:\n{prompt.strip()}")
    print("-" * 50)
    answer = ask_model(
        model, 
        prompt, 
        max_new_tokens=150, 
        temperature=0.4, 
        top_k=10, 
        stop_sequences=stop_seqs,
        return_full=False
    )
    print(f"{prefix}{answer}")
    print("-" * 50)

if __name__ == "__main__":
    main()
