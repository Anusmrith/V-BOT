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
        prompt = " ".join(sys.argv[1:]).strip()
    else:
        # Default prompt if none provided
        prompt = "Vishwas AI is "

    model = load_trained_model("model.pt")
    
    print(f"\n[Prompt]: {prompt}")
    print("[AI Continuation]:")
    print("-" * 50)
    output = ask_model(model, prompt, max_new_tokens=150, temperature=0.7)
    print(output)
    print("-" * 50)

if __name__ == "__main__":
    main()
