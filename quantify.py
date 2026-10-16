"""
================================================================================
V-BOT Token Probability Quantification CLI
================================================================================
Quantifies the exact mathematical relationship between the ground-truth target
tokens and the predicted probability distribution produced by V-BOT:
- Softmax predicted probabilities over the 2,048-token vocabulary
- Ground truth target token probability
- Cross-entropy loss: -log(P(target))
- Perplexity: exp(Cross-Entropy)
- Top-1 token prediction accuracy
- Token-by-token comparison table
================================================================================
"""

import sys
import os
import math
import torch
from torch.nn import functional as F

from tokenizer import BPETokenizer
from chat import load_trained_model
from llm import decode, encode, device

def quantify_prompt(model, prompt_text: str):
    # Ensure correct conversational formatting if prompt is a question
    is_theatrical = any(tag in prompt_text for tag in [":\n", "Citizen:", "ROMEO:", "JULIET:", "HAMLET:", "All:", "BRUTUS:"]) or prompt_text.endswith(":")
    if not is_theatrical and not prompt_text.startswith("User:") and "Assistant:" not in prompt_text:
        # Load known answer from dataset if available
        qa_file = "qa_dataset.txt"
        known_answer = None
        if os.path.exists(qa_file):
            with open(qa_file, "r", encoding="utf-8") as f:
                content = f.read().replace("\r\n", "\n")
            for block in content.split("\n\n"):
                if f"User: {prompt_text.strip()}" in block and "Assistant:" in block:
                    known_answer = block.split("Assistant:", 1)[1].strip()
                    break
        
        if known_answer:
            full_text = f"User: {prompt_text.strip()}\nAssistant: {known_answer} <|endoftext|>"
        else:
            full_text = f"User: {prompt_text.strip()}\nAssistant:"
    else:
        full_text = prompt_text

    tokens = encode(full_text)
    if len(tokens) < 2:
        print("Prompt is too short to quantify token transitions.")
        return

    tokens = tokens[:model.block_size]
    x = torch.tensor([tokens[:-1]], dtype=torch.long, device=device)
    y = torch.tensor([tokens[1:]], dtype=torch.long, device=device)

    model.eval()
    with torch.no_grad():
        logits, loss = model(x, y)
        probs = F.softmax(logits[0], dim=-1)

    y_tokens = y[0].tolist()
    x_tokens = x[0].tolist()
    
    print("\n" + "=" * 85)
    print("V-BOT TOKEN PROBABILITY QUANTIFICATION")
    print(f"Sequence: {repr(full_text[:75])}...")
    print("=" * 85)
    print(f"{'Step':<4} | {'Target Token':<18} | {'P(Target)':<11} | {'Top-1 Predicted':<18} | {'P(Top-1)':<11} | {'Match?'}")
    print("-" * 85)

    matches = 0
    target_probs_sum = 0.0
    total_tokens = len(y_tokens)

    for i, target_id in enumerate(y_tokens):
        target_token_str = repr(decode([target_id]))
        target_p = probs[i, target_id].item() * 100.0
        target_probs_sum += target_p

        top1_p, top1_id = torch.max(probs[i], dim=-1)
        top1_p_val = top1_p.item() * 100.0
        top1_token_str = repr(decode([top1_id.item()]))

        is_match = (target_id == top1_id.item())
        if is_match:
            matches += 1
            match_status = "✓ MATCH"
        else:
            match_status = "✗ DIFF"

        print(f"{i+1:<4} | {target_token_str:<18} | {target_p:>9.2f}% | {top1_token_str:<18} | {top1_p_val:>9.2f}% | {match_status}")

    acc = (matches / max(total_tokens, 1)) * 100.0
    avg_target_p = target_probs_sum / max(total_tokens, 1)
    ppl = math.exp(loss.item())

    print("-" * 85)
    print("STATISTICAL QUANTIFICATION SUMMARY:")
    print(f" • Cross-Entropy Loss:          {loss.item():.4f}")
    print(f" • Perplexity:                  {ppl:.2f}  (Ideal = 1.0, Random Uniform = 2048.0)")
    print(f" • Top-1 Next-Token Accuracy:   {acc:.1f}% ({matches}/{total_tokens} tokens correct)")
    print(f" • Mean Target Probability:     {avg_target_p:.2f}%  (Probability assigned to ground truth)")
    print("=" * 85 + "\n")

def main():
    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:]).strip().replace('\\n', '\n')
    else:
        prompt = "What is your name?"

    model = load_trained_model("model.pt")
    quantify_prompt(model, prompt)

if __name__ == "__main__":
    main()
