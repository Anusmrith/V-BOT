"""
================================================================================
V-BOT Hyperparameter Tuning Engine
================================================================================
Performs systematic grid search across AdamW hyperparameters:
- Learning Rate (max_lr)
- Weight Decay (decoupled for 2D matrices)
- Warmup Iterations
- Betas & Dropout

Quantifies token probabilities between target outputs and predicted distributions
using Cross-Entropy Loss, Perplexity, Top-1 Accuracy, and Mean Target Probability.
================================================================================
"""

import os
import sys
import math
import time
import json
import torch
import torch.nn as nn
from torch.nn import functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tokenizer import BPETokenizer

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load Tokenizer
tok = BPETokenizer.load("bpe_tokenizer.json")
vocab_size = tok.vocab_size

# Load Datasets
with open("qa_dataset.txt", "r", encoding="utf-8") as f:
    qa_raw = f.read().replace('\r\n', '\n')
qa_pairs = [p.strip() for p in qa_raw.split("\n\n") if p.strip()]

train_tokens = torch.load("train_tokens.pt", map_location='cpu')
val_tokens   = torch.load("val_tokens.pt", map_location='cpu')

block_size = 128
pad_token_id = 256

# Prepare aligned Q&A samples with prompt masking
qa_samples = []
for pair in qa_pairs:
    if "Assistant:" not in pair:
        continue
    user_part, assistant_part = pair.split("Assistant:", 1)
    prompt_str = user_part + "Assistant:"
    answer_str = assistant_part.strip() + " <|endoftext|>"
    
    prompt_ids = tok.encode(prompt_str)
    answer_ids = tok.encode(" " + answer_str.lstrip())
    
    full_ids = prompt_ids + answer_ids
    if len(full_ids) > block_size:
        full_ids = full_ids[:block_size]
    
    x = full_ids[:-1]
    y = full_ids[1:]
    
    prompt_len_in_y = len(prompt_ids) - 1
    y_masked = [-100] * prompt_len_in_y + y[prompt_len_in_y:]
    
    pad_len = block_size - len(x)
    if pad_len > 0:
        x = x + [pad_token_id] * pad_len
        y_masked = y_masked + [-100] * pad_len
        
    qa_samples.append((torch.tensor(x, dtype=torch.long), torch.tensor(y_masked, dtype=torch.long)))

# Split into 85% train QA, 15% val QA
qa_train_split = int(0.85 * len(qa_samples))
qa_train = qa_samples[:qa_train_split]
qa_val   = qa_samples[qa_train_split:]

print(f"Loaded {len(qa_train)} train QA, {len(qa_val)} val QA samples.")
print(f"Loaded {len(train_tokens):,} train tokens, {len(val_tokens):,} val tokens.")

from llm import GPTLanguageModel

def get_batch(split: str, batch_size=16):
    half = batch_size // 2
    qa_pool = qa_train if split == 'train' else qa_val
    data_pool = train_tokens if split == 'train' else val_tokens
    
    # 50% from Q&A
    ix_qa = torch.randint(len(qa_pool), (half,))
    x_qa = torch.stack([qa_pool[i][0] for i in ix_qa])
    y_qa = torch.stack([qa_pool[i][1] for i in ix_qa])
    
    # 50% from Continuous Corpus
    high = len(data_pool) - block_size - 1
    ix_sh = torch.randint(high, (half,))
    x_sh = torch.stack([data_pool[i:i + block_size] for i in ix_sh])
    y_sh = torch.stack([data_pool[i + 1:i + block_size + 1] for i in ix_sh])
    
    x = torch.cat([x_qa, x_sh], dim=0)
    y = torch.cat([y_qa, y_sh], dim=0)
    return x.to(device), y.to(device)

def quantify_probabilities(logits: torch.Tensor, targets: torch.Tensor):
    """
    Quantifies the token probability distribution between the target output tokens
    and the predicted probabilities produced by V-BOT:
    1. Cross-Entropy Loss
    2. Perplexity = exp(Loss)
    3. Top-1 Token Accuracy (% where argmax == target)
    4. Top-5 Token Coverage (% where target in top-5)
    5. Mean Target Probability (% confidence assigned to ground truth targets)
    6. Mean Top-1 Predicted Probability (% confidence of the model's top choice)
    """
    B, T, V = logits.shape
    logits_flat = logits.view(B * T, V)
    targets_flat = targets.view(B * T)
    
    valid_mask = (targets_flat != -100)
    valid_logits = logits_flat[valid_mask]
    valid_targets = targets_flat[valid_mask]
    
    if len(valid_targets) == 0:
        return {"loss": 0.0, "ppl": 1.0, "top1_acc": 0.0, "top5_acc": 0.0, "target_prob": 0.0, "top1_prob": 0.0}
    
    # Cross-entropy loss
    loss = F.cross_entropy(valid_logits, valid_targets).item()
    ppl = math.exp(min(loss, 20.0))
    
    # Probabilities via softmax
    probs = F.softmax(valid_logits, dim=-1)
    
    # Target Token Probability: exact probability assigned to correct target
    target_probs = probs.gather(1, valid_targets.unsqueeze(1)).squeeze(1)
    mean_target_prob = target_probs.mean().item() * 100.0
    
    # Top-1 accuracy and probability
    top1_probs, top1_indices = torch.max(probs, dim=-1)
    top1_acc = (top1_indices == valid_targets).float().mean().item() * 100.0
    mean_top1_prob = top1_probs.mean().item() * 100.0
    
    # Top-5 coverage
    top5_indices = torch.topk(probs, k=min(5, V), dim=-1).indices
    top5_acc = (top5_indices == valid_targets.unsqueeze(1)).any(dim=1).float().mean().item() * 100.0
    
    return {
        "loss": loss,
        "ppl": ppl,
        "top1_acc": top1_acc,
        "top5_acc": top5_acc,
        "target_prob": mean_target_prob,
        "top1_prob": mean_top1_prob
    }

def configure_adamw(model, weight_decay, learning_rate, betas=(0.9, 0.95)):
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, eps=1e-8)

def get_cosine_lr(it, max_iters, warmup_iters, max_lr, min_lr):
    if it < warmup_iters:
        return max_lr * (it + 1) / (warmup_iters + 1)
    if it > max_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / max(1, max_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

def evaluate_metrics(model, eval_iters=15):
    model.eval()
    all_metrics = []
    with torch.no_grad():
        for _ in range(eval_iters):
            xb, yb = get_batch('val')
            logits, _ = model(xb, yb)
            metrics = quantify_probabilities(logits, yb)
            all_metrics.append(metrics)
    model.train()
    
    avg_res = {}
    for k in all_metrics[0].keys():
        avg_res[k] = sum(m[k] for m in all_metrics) / len(all_metrics)
    return avg_res

def run_trial(trial_id, lr, wd, warmup, iters=200):
    print(f"\n[Trial {trial_id}] Testing: LR={lr:.4f} | WeightDecay={wd:.3f} | Warmup={warmup}")
    torch.manual_seed(42)
    
    model = GPTLanguageModel(
        vocab_size=vocab_size,
        n_embd=128,
        block_size=block_size,
        n_head=4,
        n_layer=4
    ).to(device)
    
    optimizer = configure_adamw(model, weight_decay=wd, learning_rate=lr)
    min_lr = lr * 0.1
    
    t0 = time.time()
    for it in range(iters):
        curr_lr = get_cosine_lr(it, iters, warmup_iters=warmup, max_lr=lr, min_lr=min_lr)
        for pg in optimizer.param_groups:
            pg['lr'] = curr_lr
            
        xb, yb = get_batch('train')
        logits, loss = model(xb, yb)
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
    metrics = evaluate_metrics(model)
    elapsed = time.time() - t0
    print(f" -> Val Loss: {metrics['loss']:.4f} | PPL: {metrics['ppl']:.2f} | "
          f"Top-1 Acc: {metrics['top1_acc']:.2f}% | Target Prob: {metrics['target_prob']:.2f}% "
          f"[{elapsed:.1f}s]")
    
    return {
        "trial_id": trial_id,
        "lr": lr,
        "weight_decay": wd,
        "warmup_iters": warmup,
        "metrics": metrics,
        "elapsed": elapsed
    }

def main():
    print("=" * 70)
    print("STARTING HYPERPARAMETER GRID TUNING FOR V-BOT")
    print("=" * 70)
    
    # Define candidate grid
    lr_candidates = [8e-4, 1.2e-3, 1.5e-3, 2.0e-3]
    wd_candidates = [0.01, 0.05, 0.1]
    warmup_candidates = [20, 40]
    
    # Representative subset of hyperparameter combinations for fast tuning
    configs = [
        (1.2e-3, 0.05, 40),
        (1.5e-3, 0.05, 40),
        (2.0e-3, 0.05, 40),
        (1.5e-3, 0.01, 40),
        (1.5e-3, 0.10, 40),
        (1.5e-3, 0.05, 20),
    ]
    
    results = []
    for idx, (lr, wd, warmup) in enumerate(configs, 1):
        res = run_trial(idx, lr, wd, warmup, iters=250)
        results.append(res)
        
    # Sort by validation loss (lower is better) and target probability (higher is better)
    results.sort(key=lambda x: (x["metrics"]["loss"], -x["metrics"]["target_prob"]))
    
    best = results[0]
    print("\n" + "=" * 70)
    print("HYPERPARAMETER TUNING WINNER:")
    print("=" * 70)
    print(f"Best Configuration: Trial #{best['trial_id']}")
    print(f" • Learning Rate:     {best['lr']:.4f}")
    print(f" • Weight Decay:     {best['weight_decay']:.3f}")
    print(f" • Warmup Steps:     {best['warmup_iters']}")
    print(f" • Val Loss:         {best['metrics']['loss']:.4f}")
    print(f" • Perplexity:       {best['metrics']['ppl']:.2f}")
    print(f" • Top-1 Accuracy:   {best['metrics']['top1_acc']:.2f}%")
    print(f" • Target Prob:      {best['metrics']['target_prob']:.2f}%")
    print("=" * 70)
    
    # Save hyperparameter tuning results to JSON
    with open("tuning_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("Saved tuning analysis to 'tuning_results.json'")

if __name__ == "__main__":
    main()
