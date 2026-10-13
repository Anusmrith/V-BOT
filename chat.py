

import os
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

def ask_model(model, prompt_text, max_new_tokens=150, temperature=0.7, top_k=10):
    """Encodes the user prompt and autoregressively generates the continuation."""
    tokens = encode(prompt_text)
    if len(tokens) == 0:
        tokens = [0]
    
    context = torch.tensor([tokens], dtype=torch.long, device=device)
    with torch.no_grad():
        generated_ids = model.generate(
            context, 
            max_new_tokens=max_new_tokens, 
            temperature=temperature, 
            top_k=top_k
        )[0].tolist()
    
    return decode(generated_ids)

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
            output = ask_model(model, prompt, max_new_tokens=150, temperature=0.7)
            print("-" * 40)
            print(output)
            print("-" * 40)
        except KeyboardInterrupt:
            print("\nExiting console.")
            break

if __name__ == "__main__":
    main()
