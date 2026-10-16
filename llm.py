"""
================================================================================
Decoder-Only Large Language Model (GPT Architecture) from Scratch in PyTorch
================================================================================
Author: AI & Deep Learning Systems
Framework: Pure PyTorch (No Hugging Face AutoModel / LangChain / high-level wrappers)

Architectural Overview:
1. Byte-Level Byte Pair Encoding (BPE) Tokenizer (UTF-8 byte base + learned subwords)
2. Scaled Dot-Product Causal Self-Attention Head (with lower triangular mask)
3. Multi-Head Attention (combining parallel heads with linear projection)
4. Position-wise Feed-Forward Network (2-layer linear projection + GELU)
5. Pre-LayerNorm Residual Transformer Block
6. Full Decoder-Only GPT Language Model (Token & Positional Embeddings + Blocks + LM Head)
7. Production Optimizer: Decoupled AdamW (weight decay parameter grouping)
8. Learning Rate Schedule: Linear Warmup + Cosine Annealing Decay
9. Gradient Regularization: Gradient Norm Clipping (max_norm=1.0)
10. Dual-Domain Balanced Training: Prompt-Masked Aligned SFT + Continuous Corpus
================================================================================
"""

import os
import sys
import math
import time

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

# ==============================================================================
# 0. Global Hyperparameters & Device Setup
# ==============================================================================
torch.manual_seed(1337)

# Hardware-agnostic: automatically use CUDA GPU if available, otherwise fall back to CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Model dimensions & training configuration
batch_size    = 16      # Number of independent sequence batches processed in parallel
block_size    = 128     # Maximum context window length for self-attention (T)
max_iters     = 600     # Total training iterations (converges in ~2 min on CPU)
eval_interval = 50      # Frequency of evaluating and logging loss & probability quantification
learning_rate = 2.0e-3  # Peak tuned learning rate for AdamW (from hyperparameter grid search)
min_lr        = 2.0e-4  # Minimum learning rate after cosine decay
warmup_iters  = 40      # Iterations for linear warmup
weight_decay  = 0.05    # Weight decay for 2D parameter matrices (linear layers and embeddings)
max_norm      = 1.0     # Gradient norm clipping threshold for gradient descent stability
eval_iters    = 25      # Number of batches to average loss over during evaluation
n_embd        = 128     # Embedding dimension (C)
n_head        = 4       # Number of multi-head attention heads
n_layer       = 4       # Number of stacked transformer blocks
dropout       = 0.1     # Dropout probability for regularization


# ==============================================================================
# 1. Dataset Loading & Byte-Level Byte Pair Encoding (BPE) Tokenizer
# ==============================================================================
from tokenizer import BPETokenizer, get_tokenizer

# Default embedded fallback dataset if 'data.txt' is absent
FALLBACK_TEXT = """
Vishwas AI is a system built to bring trust to artificial intelligence.
Trust is not the same as truth. Trust means the output is supported by evidence.
When a model gives an answer, Vishwas AI checks every single claim.
If a claim is supported, Vishwas AI marks it as verified.
If a claim is contradicted, Vishwas AI warns the user.
Trustworthy artificial intelligence requires verification, validation, and transparency.
Every deep learning model produces tokens based on learned patterns and statistical associations.
A reliable reasoning engine must ground its statements in verifiable knowledge sources.
Self-attention mechanisms allow every token to look back at previous tokens and build context.
Multi-head attention enables the network to attend to information from different representation subspaces.
Feed-forward networks apply non-linear transformations to refine token representations.
Residual connections and layer normalization stabilize training across deep architectures.
By systematically evaluating assertions against ground truth, Vishwas AI creates confidence.
Trust is built step by step, token by token, through rigorous verification and continuous learning.
"""

DATA_FILE = "combined_train.txt" if os.path.exists("combined_train.txt") else ("input.txt" if os.path.exists("input.txt") else "data.txt")

def load_text_data(filepath=None):
    """
    Loads text from local file if available; otherwise uses the fallback dataset.
    Prioritizes combined_train.txt if present, then input.txt, then data.txt.
    """
    if filepath is None:
        filepath = DATA_FILE

    candidates = [
        filepath,
        "combined_train.txt",
        "input.txt",
        "data.txt",
        "data.txt.txt",
        os.path.join(os.path.dirname(__file__), filepath) if filepath else "combined_train.txt"
    ]
    for path in candidates:
        if path and os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
                if len(content.strip()) > 0:
                    # Sanitize carriage returns (\r) to prevent terminal display overwrite glitches
                    content = content.replace("\r\n", "\n").replace("\r", "")
                    print(f"[Data] Successfully loaded dataset from: {path} ({len(content):,} characters)")
                    return content
    print("[Data] Neither 'combined_train.txt', 'input.txt' nor 'data.txt' found. Using default embedded fallback dataset.")
    return FALLBACK_TEXT.strip().replace("\r\n", "\n").replace("\r", "")

raw_text = load_text_data()

# Initialize or load Byte-Level BPE Tokenizer
TOKENIZER_FILE = "bpe_tokenizer.json"
tokenizer = get_tokenizer(
    model_path=TOKENIZER_FILE,
    corpus_path=DATA_FILE,
    target_vocab_size=2048
)
vocab_size = tokenizer.vocab_size

def encode(s: str):
    """Encodes a string of text into a list of BPE integer token IDs."""
    return tokenizer.encode(s)

def decode(indices):
    """Decodes a list of BPE integer token IDs back into a text string."""
    return tokenizer.decode(indices)

# Backward-compatible mappings
stoi = tokenizer.get_stoi()
itos = tokenizer.get_itos()

# Load pre-tokenized training tensors if available, otherwise tokenize in-memory
if os.path.exists("train_tokens.pt") and os.path.exists("val_tokens.pt"):
    train_data = torch.load("train_tokens.pt", map_location='cpu')
    val_data   = torch.load("val_tokens.pt", map_location='cpu')
    print(f"[Tokenizer] BPE Tokenizer active (vocab size: {vocab_size} tokens)")
    print(f"[Dataset] Loaded pre-tokenized BPE tensors: Train={len(train_data):,} | Validation={len(val_data):,} tokens")
else:
    data_tensor = torch.tensor(encode(raw_text), dtype=torch.long)
    split_idx = int(0.9 * len(data_tensor))
    train_data = data_tensor[:split_idx]
    val_data   = data_tensor[split_idx:]
    print(f"[Tokenizer] BPE Tokenizer active (vocab size: {vocab_size} tokens)")
    print(f"[Dataset] Train tokens: {len(train_data):,} | Validation tokens: {len(val_data):,}")

# Prepare Prompt-Masked Aligned SFT Samples for Q&A
QA_FILE = "qa_dataset.txt"
qa_samples = []
if os.path.exists(QA_FILE):
    with open(QA_FILE, "r", encoding="utf-8") as f:
        qa_raw_text = f.read().replace("\r\n", "\n").replace("\r", "")
    qa_pairs = [p.strip() for p in qa_raw_text.split("\n\n") if p.strip()]
    for pair in qa_pairs:
        if "Assistant:" not in pair:
            continue
        user_part, assistant_part = pair.split("Assistant:", 1)
        prompt_str = user_part + "Assistant:"
        answer_str = assistant_part.strip() + " <|endoftext|>"
        
        prompt_ids = encode(prompt_str)
        answer_ids = encode(" " + answer_str.lstrip())
        
        full_ids = prompt_ids + answer_ids
        if len(full_ids) > block_size:
            full_ids = full_ids[:block_size]
        
        x = full_ids[:-1]
        y = full_ids[1:]
        
        # Mask prompt tokens with -100 so loss exclusively trains on generating the Assistant's answer
        prompt_len_in_y = len(prompt_ids) - 1
        y_masked = [-100] * prompt_len_in_y + y[prompt_len_in_y:]
        
        # Pad to block_size with endoftext (token 256)
        pad_len = block_size - len(x)
        if pad_len > 0:
            x = x + [256] * pad_len
            y_masked = y_masked + [-100] * pad_len
            
        qa_samples.append((
            torch.tensor(x, dtype=torch.long),
            torch.tensor(y_masked, dtype=torch.long)
        ))
    print(f"[Dataset] Prepared {len(qa_samples)} prompt-masked aligned Q&A samples for high-accuracy training.")


# ==============================================================================
# 2. Dual-Domain Balanced Batching Pipeline
# ==============================================================================
def get_batch(split: str):
    """
    Dual-Domain Balanced Batch Loader:
    Combines 50% aligned Q&A instruction pairs with 50% continuous corpus tokens.
    Guarantees the model learns clean, crisp answers while retaining rich general/Shakespeare knowledge.
    """
    half = batch_size // 2
    data = train_data if split == 'train' else val_data
    
    if len(qa_samples) > 0 and half > 0:
        qa_idx = torch.randint(len(qa_samples), (half,))
        x_qa = torch.stack([qa_samples[i][0] for i in qa_idx])
        y_qa = torch.stack([qa_samples[i][1] for i in qa_idx])
        
        high = len(data) - block_size - 1
        sh_idx = torch.randint(high, (half,))
        x_sh = torch.stack([data[i:i + block_size] for i in sh_idx])
        y_sh = torch.stack([data[i + 1:i + block_size + 1] for i in sh_idx])
        
        x = torch.cat([x_qa, x_sh], dim=0)
        y = torch.cat([y_qa, y_sh], dim=0)
    else:
        high = len(data) - block_size - 1
        ix = torch.randint(high, (batch_size,))
        x = torch.stack([data[i:i + block_size] for i in ix])
        y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
        
    return x.to(device), y.to(device)


# ==============================================================================
# 3. Neural Network Architecture Components
# ==============================================================================

class Head(nn.Module):
    """
    A single head of Scaled Dot-Product Self-Attention with Causal Masking.
    """
    def __init__(self, head_size: int, n_embd: int = n_embd, block_size: int = block_size):
        super().__init__()
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        k = self.key(x)    # (B, T, head_size)
        q = self.query(x)  # (B, T, head_size)
        v = self.value(x)  # (B, T, head_size)
        
        scale = 1.0 / (k.shape[-1] ** 0.5)
        wei = (q @ k.transpose(-2, -1)) * scale
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention: Combines parallel attention heads.
    """
    def __init__(self, num_heads: int, head_size: int, n_embd: int = n_embd, block_size: int = block_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, n_embd, block_size) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network: 2-layer MLP with 4x expansion and GELU.
    """
    def __init__(self, n_embd: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """
    Pre-LayerNorm Transformer Block.
    """
    def __init__(self, n_embd: int, n_head: int, block_size: int = block_size):
        super().__init__()
        head_size = n_embd // n_head
        self.sa   = MultiHeadAttention(n_head, head_size, n_embd, block_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1  = nn.LayerNorm(n_embd)
        self.ln2  = nn.LayerNorm(n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):
    """
    Complete Decoder-Only GPT Language Model.
    Supports variable dimensions for loading checkpoints with different sizes.
    """
    def __init__(self, vocab_size: int, n_embd: int = n_embd, block_size: int = block_size, n_head: int = n_head, n_layer: int = n_layer):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_embd = n_embd
        
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[TransformerBlock(n_embd, n_head=n_head, block_size=block_size) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)                                    # (B, T, n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))   # (T, n_embd)
        x = tok_emb + pos_emb                                                        # (B, T, n_embd)
        x = self.blocks(x)                                                           # (B, T, n_embd)
        x = self.ln_f(x)                                                             # (B, T, n_embd)
        logits = self.lm_head(x)                                                     # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits_flat  = logits.view(B * T, C)
            targets_flat = targets.view(B * T)
            # ignore_index=-100 ignores prompt tokens and padding in loss computation
            loss = F.cross_entropy(logits_flat, targets_flat, ignore_index=-100)

        return logits, loss

    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: int = None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            
            if temperature > 0.0:
                logits = logits / temperature
            else:
                logits = logits / 0.01
            
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
                
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            
            # Stop token check (<|endoftext|>)
            if idx_next.item() == 256:
                break
            
        return idx


# ==============================================================================
# 4. Production Optimizer & Learning Rate Schedule
# ==============================================================================

def configure_optimizers(model: nn.Module, weight_decay: float = 0.05, learning_rate: float = 1.5e-3, betas=(0.9, 0.95)):
    """
    Constructs an AdamW optimizer with decoupled parameter grouping:
    - 2D parameters (weights of linear layers and embeddings) receive weight_decay (0.05).
    - 1D parameters (biases, LayerNorm scale and shift parameters) receive 0.0 weight decay.
    This prevents LayerNorm scale shrinkage and preserves training stability across deep layers.
    """
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    print(f"[Optimizer] Parameter groups: {len(decay_params)} decay ({num_decay_params:,} params) | "
          f"{len(nodecay_params)} non-decay ({num_nodecay_params:,} params)")
    
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, eps=1e-8)
    return optimizer

def get_lr(it: int, max_iters: int, warmup_iters: int = 40, max_lr: float = 1.5e-3, min_lr: float = 1e-4) -> float:
    """
    Linear warmup for warmup_iters, followed by Cosine Annealing decay down to min_lr.
    """
    if it < warmup_iters:
        return max_lr * (it + 1) / (warmup_iters + 1)
    if it > max_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / max(1, max_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


# ==============================================================================
# 5. Model Training Pipeline
# ==============================================================================

def quantify_probabilities(logits: torch.Tensor, targets: torch.Tensor):
    """
    Quantifies the token probability distribution between the ground-truth target
    tokens and the predicted probabilities produced by V-BOT:
    1. Cross-Entropy Loss: -E[log P(y_target)]
    2. Perplexity: exp(Loss)
    3. Top-1 Token Accuracy: % of tokens where argmax(P_pred) == y_target
    4. Top-5 Token Accuracy: % of tokens where y_target in top-5 predicted
    5. Mean Target Token Probability: Average confidence assigned to the target tokens
    6. Mean Top-1 Probability: Average confidence of the model's highest prediction
    """
    B, T, V = logits.shape
    logits_flat = logits.view(B * T, V)
    targets_flat = targets.view(B * T)
    
    valid_mask = (targets_flat != -100)
    valid_logits = logits_flat[valid_mask]
    valid_targets = targets_flat[valid_mask]
    
    if len(valid_targets) == 0:
        return {"loss": 0.0, "ppl": 1.0, "top1_acc": 0.0, "top5_acc": 0.0, "target_prob": 0.0, "top1_prob": 0.0}
    
    loss = F.cross_entropy(valid_logits, valid_targets).item()
    ppl = math.exp(min(loss, 20.0))
    
    probs = F.softmax(valid_logits, dim=-1)
    target_probs = probs.gather(1, valid_targets.unsqueeze(1)).squeeze(1)
    mean_target_prob = target_probs.mean().item() * 100.0
    
    top1_probs, top1_indices = torch.max(probs, dim=-1)
    top1_acc = (top1_indices == valid_targets).float().mean().item() * 100.0
    mean_top1_prob = top1_probs.mean().item() * 100.0
    
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

@torch.no_grad()
def estimate_loss(model: nn.Module):
    """
    Evaluates model across multiple batches for both train and validation splits
    to compute an accurate, low-variance estimate of loss and probability metrics.
    """
    out = {}
    model.eval()
    for split in ['train', 'val']:
        batch_metrics = []
        for k in range(eval_iters):
            xb, yb = get_batch(split)
            logits, _ = model(xb, yb)
            metrics = quantify_probabilities(logits, yb)
            batch_metrics.append(metrics)
        
        avg_split = {}
        for metric_name in batch_metrics[0].keys():
            avg_split[metric_name] = sum(m[metric_name] for m in batch_metrics) / len(batch_metrics)
        out[split] = avg_split
    model.train()
    return out

def print_probability_quantification_report(model: nn.Module, sample_text: str = None):
    """
    Renders an explicit token-by-token quantification table comparing
    ground-truth target tokens against V-BOT's predicted token probabilities.
    """
    if sample_text is None:
        sample_text = "User: What is your name?\nAssistant: My name is V-BOT, an intelligent assistant built on the Vishwas AI neural architecture."
    
    model.eval()
    tokens = encode(sample_text)
    if len(tokens) > block_size:
        tokens = tokens[:block_size]
    
    x = torch.tensor([tokens[:-1]], dtype=torch.long, device=device)
    y = torch.tensor([tokens[1:]], dtype=torch.long, device=device)
    
    with torch.no_grad():
        logits, loss = model(x, y)
        probs = F.softmax(logits[0], dim=-1)
    
    print("\n" + "=" * 85)
    print("TOKEN PROBABILITY QUANTIFICATION REPORT (TARGET OUTPUT VS. PREDICTED PROBABILITIES)")
    print("=" * 85)
    print(f"{'Step':<4} | {'Target Token':<18} | {'P(Target)':<11} | {'Top-1 Predicted':<18} | {'P(Top-1)':<11} | {'Match?'}")
    print("-" * 85)
    
    y_tokens = y[0].tolist()
    matches = 0
    total_tokens = len(y_tokens)
    target_probs_sum = 0.0
    
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
            match_symbol = "✓ MATCH"
        else:
            match_symbol = "✗ DIFF"
            
        print(f"{i+1:<4} | {target_token_str:<18} | {target_p:>9.2f}% | {top1_token_str:<18} | {top1_p_val:>9.2f}% | {match_symbol}")
    
    acc = (matches / max(total_tokens, 1)) * 100.0
    avg_target_p = target_probs_sum / max(total_tokens, 1)
    print("-" * 85)
    print(f"Overall Cross-Entropy Loss: {loss.item():.4f} | Perplexity: {math.exp(loss.item()):.2f}")
    print(f"Token Prediction Accuracy:  {acc:.1f}% ({matches}/{total_tokens} tokens)")
    print(f"Mean Target Probability:    {avg_target_p:.2f}% (Average confidence on ground truth)")
    print("=" * 85 + "\n")
    model.train()

def main():
    # Allow overriding iterations via command line (e.g. 'python llm.py 600')
    iters = max_iters
    for arg in sys.argv[1:]:
        if arg.isdigit():
            iters = int(arg)
            break

    print("=" * 70)
    print("INITIALIZING CUSTOM DECODER-ONLY GPT MODEL WITH OPTIMIZER")
    print(f"Device: {device.upper()} | PyTorch Version: {torch.__version__}")
    print("=" * 70)
    
    # Instantiate Model
    model = GPTLanguageModel(
        vocab_size=vocab_size,
        n_embd=n_embd,
        block_size=block_size,
        n_head=n_head,
        n_layer=n_layer
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Model Parameters: {total_params:,}")
    print(f"Architecture: {n_layer} layers | {n_head} heads | {n_embd} embedding dim | {block_size} context length")
    
    # Checkpoint resuming: continue training from previous weights if compatible
    checkpoint_path = "model.pt"
    if os.path.exists(checkpoint_path):
        try:
            ckpt = torch.load(checkpoint_path, map_location=device)
            ckpt_v = ckpt.get('token_embedding_table.weight', ckpt.get('tok_emb.weight', None))
            ckpt_e = ckpt_v.shape[1] if ckpt_v is not None else None
            if ckpt_v is not None and ckpt_v.shape[0] == vocab_size and ckpt_e == n_embd:
                model.load_state_dict(ckpt)
                print(f"[Checkpoint] Resuming from existing '{checkpoint_path}'. Weights preserved!")
            else:
                ckpt_size = ckpt_v.shape[0] if ckpt_v is not None else "unknown"
                print(f"[Checkpoint] Existing checkpoint dimensions differ. Initializing optimized model weights.")
        except Exception as e:
            print(f"[Checkpoint] Initializing fresh model weights ({e}).")
    print("-" * 70)

    # Configure advanced Decoupled AdamW Optimizer
    optimizer = configure_optimizers(
        model, 
        weight_decay=weight_decay, 
        learning_rate=learning_rate, 
        betas=(0.9, 0.95)
    )

    print(f"Beginning Training Loop ({iters} iterations)...")
    print(f"Optimizer: Decoupled AdamW (weight_decay={weight_decay}, betas=(0.9, 0.95))")
    print(f"LR Schedule: Linear Warmup ({warmup_iters} iters) -> Cosine Annealing (decay to {min_lr})")
    print(f"Gradient Clipping: max_norm={max_norm}")
    print("-" * 70)

    t0 = time.time()
    for iter_num in range(iters):
        # Update dynamic learning rate
        lr = get_lr(iter_num, iters, warmup_iters=warmup_iters, max_lr=learning_rate, min_lr=min_lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Periodically evaluate loss & probability metrics on train and validation splits
        if iter_num % eval_interval == 0 or iter_num == iters - 1:
            losses = estimate_loss(model)
            elapsed = time.time() - t0
            print(f"Iter {iter_num:4d}/{iters} | Train Loss: {losses['train']['loss']:.4f} (PPL: {losses['train']['ppl']:.1f}) | "
                  f"Val Loss: {losses['val']['loss']:.4f} (PPL: {losses['val']['ppl']:.1f}) | "
                  f"Val Acc: {losses['val']['top1_acc']:.1f}% | Target Prob: {losses['val']['target_prob']:.2f}% | "
                  f"LR: {lr:.6f} | Elapsed: {elapsed:.1f}s")

        # Sample a balanced batch of training sequences
        xb, yb = get_batch('train')

        # Forward pass & loss calculation
        logits, loss = model(xb, yb)

        # Backward pass & parameter optimization with gradient clipping
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
        optimizer.step()

    print("-" * 70)
    # Save the trained model checkpoint
    checkpoint_path = "model.pt"
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Model checkpoint successfully saved to: {checkpoint_path}")
    print("=" * 70)

    # ==============================================================================
    # 6. Token Probability Quantification Report
    # ==============================================================================
    print_probability_quantification_report(model)

    # ==============================================================================
    # 6. Text Generation Demonstration
    # ==============================================================================
    print("\n" + "=" * 70)
    print("AUTOREGRESSIVE TEXT GENERATION ENGINE")
    print("=" * 70)

    test_prompts = [
        "User: What is your name?\nAssistant:",
        "User: What is Vishwas AI?\nAssistant:",
        "User: Who is Romeo?\nAssistant:",
        "User: Who is William Shakespeare?\nAssistant:",
        "First Citizen:\n"
    ]

    for p in test_prompts:
        seed_tokens = encode(p)
        if len(seed_tokens) == 0:
            seed_tokens = [0]
        context = torch.tensor([seed_tokens], dtype=torch.long, device=device)
        with torch.no_grad():
            generated_ids = model.generate(context, max_new_tokens=80, temperature=0.4, top_k=10)[0].tolist()
        generated_text = decode(generated_ids)
        continuation = generated_text[len(p):]
        for stop in ["\nUser:", "<|endoftext|>"]:
            if stop in continuation:
                continuation = continuation.split(stop)[0]
        print(f"\n--- Continuation for prompt: '{p.strip()}' ---")
        try:
            print(continuation.strip())
        except Exception:
            print(continuation.strip().encode('utf-8', errors='replace').decode('latin-1', errors='replace'))

    print("=" * 70)

if __name__ == "__main__":
    main()
