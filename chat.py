

import os
import sys
import re

# Ensure UTF-8 console encoding on Windows to prevent charmap UnicodeEncodeErrors
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import torch.nn as nn
# pyrefly: ignore [missing-import]
from torch.nn import functional as F

# Import model architecture & configurations from llm.py
from llm import (
    GPTLanguageModel,
    load_text_data,
    encode,
    decode,
    stoi,
    itos,
    vocab_size,
    block_size,
    device,
    n_embd,
    n_head,
    n_layer
)

def load_trained_model(checkpoint_path="model.pt"):
    """Loads model weights from checkpoint into evaluation mode, auto-detecting dimensions."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint '{checkpoint_path}' not found! Run 'python llm.py' first to train.")
    
    ckpt = torch.load(checkpoint_path, map_location=device)
    
    # Auto-remap Colab / NanoGPT keys if present
    if 'tok_emb.weight' in ckpt:
        ckpt['token_embedding_table.weight'] = ckpt.pop('tok_emb.weight')
    if 'pos_emb.weight' in ckpt:
        ckpt['position_embedding_table.weight'] = ckpt.pop('pos_emb.weight')
    if 'head.weight' in ckpt:
        ckpt['lm_head.weight'] = ckpt.pop('head.weight')
    if 'head.bias' in ckpt:
        ckpt['lm_head.bias'] = ckpt.pop('head.bias')

    # Auto-detect dimensions from checkpoint tensors
    v_size, embd_dim = ckpt['token_embedding_table.weight'].shape
    b_size = ckpt['position_embedding_table.weight'].shape[0]
    num_layers = len([k for k in ckpt.keys() if 'blocks.' in k and '.ln1.weight' in k])
    num_heads = len([k for k in ckpt.keys() if 'blocks.0.sa.heads.' in k and '.key.weight' in k])
    
    print(f"[Model Config] Vocab: {v_size} | Embd: {embd_dim} | Context: {b_size} | Layers: {num_layers} | Heads: {num_heads}")
    if v_size != vocab_size:
        print(f"[Notice] Checkpoint vocabulary size ({v_size}) differs from current BPE tokenizer ({vocab_size}).")
        print("         Run 'python llm.py' to update checkpoint weights to current BPE vocabulary.")
    
    model = GPTLanguageModel(
        vocab_size=v_size,
        n_embd=embd_dim,
        block_size=b_size,
        n_head=num_heads,
        n_layer=num_layers
    ).to(device)
    
    model.load_state_dict(ckpt)
    model.eval()
    return model

def ask_model(model, prompt_text, max_new_tokens=150, temperature=0.4, top_k=10, stop_sequences=None, return_full=False):
    """
    Encodes the user prompt and autoregressively generates the continuation.
    Stops generating as soon as a stop sequence (e.g. '\nUser:' or '<|endoftext|>') is reached,
    preventing the model from generating multiple answers or hallucinating subsequent user questions.
    """
    if stop_sequences is None:
        stop_sequences = ["\nUser:", "\n\nUser:", "<|endoftext|>"]

    tokens = encode(prompt_text)
    if len(tokens) == 0:
        tokens = [0]
    
    context = torch.tensor([tokens], dtype=torch.long, device=device)
    new_token_ids = []
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            idx_cond = context[:, -model.block_size:]
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-4)
            
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
                
            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)
            token_id = next_idx.item()
            
            # Stop token check (<|endoftext|>)
            if token_id == 256:
                break
                
            context = torch.cat((context, next_idx), dim=1)
            new_token_ids.append(token_id)
            
            # Substring stop check
            curr_text = decode(new_token_ids)
            if any(stop in curr_text for stop in stop_sequences):
                break
            # In conversational mode, also stop if a theatrical character header appears
            if "\nUser:" in stop_sequences and re.search(r'\n[A-Z]{3,}[\sA-Z]*:', curr_text):
                break

    answer_text = decode(new_token_ids)
    for stop in stop_sequences:
        if stop in answer_text:
            answer_text = answer_text.split(stop)[0]

    # In conversational mode, truncate at any theatrical character turn
    if "\nUser:" in stop_sequences:
        theatrical_match = re.search(r'\n[A-Z]{3,}[\sA-Z]*:', answer_text)
        if theatrical_match:
            answer_text = answer_text[:theatrical_match.start()]

    # Clean up excessive newlines or horizontal gaps
    answer_text = re.sub(r'\n{3,}', '\n\n', answer_text)
    answer_text = re.sub(r'[ \t]{3,}', ' ', answer_text)
    answer_text = answer_text.strip()
    if return_full:
        return prompt_text + " " + answer_text
    return answer_text

def main():
    print("=" * 65)
    print("VISHWAS AI - INTERACTIVE GPT INFERENCE CONSOLE")
    print("=" * 65)
    print(f"Loading trained weights from 'model.pt' on [{device.upper()}]...")
    
    try:
        model = load_trained_model("model.pt")
        print("Model loaded successfully!\n")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print("HOW TO PROMPT THIS BASE MODEL:")
    print("-> Base LLMs are next-token predictors (they complete text).")
    print("-> For best results, give a starting prefix, e.g.:")
    print("     'Vishwas AI is '")
    print("     'Trust means '")
    print("     'When a model gives an answer, '")
    print("Type 'quit' or 'exit' to exit.")
    print("-" * 65)

    while True:
        try:
            prompt = input("\nEnter prompt > ").strip()
            if prompt.lower() in ['quit', 'exit']:
                print("Exiting console. Goodbye!")
                break
            if not prompt:
                continue

            print("\nGenerating continuation...")
            output = ask_model(model, prompt, max_new_tokens=400, temperature=0.7)
            print("-" * 40)
            print(output)
            print("-" * 40)
        except KeyboardInterrupt:
            print("\nExiting console.")
            break

if __name__ == "__main__":
    main()
