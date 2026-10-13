"""
================================================================================
Decoder-Only Large Language Model (GPT Architecture) from Scratch in PyTorch
================================================================================
Author: AI & Deep Learning Systems
Framework: Pure PyTorch (No Hugging Face AutoModel / LangChain / high-level wrappers)

Architectural Overview:
1. Character-level Tokenizer (bi-directional mapping: char <-> int)
2. Scaled Dot-Product Causal Self-Attention Head (with lower triangular mask)
3. Multi-Head Attention (combining parallel heads with linear projection)
4. Position-wise Feed-Forward Network (2-layer linear projection + GELU)
5. Pre-LayerNorm Residual Transformer Block
6. Full Decoder-Only GPT Language Model (Token & Positional Embeddings + Blocks + LM Head)
7. Autoregressive Text Generation Engine (sampling via torch.multinomial)
================================================================================
"""

import os
import sys
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
block_size    = 64      # Maximum context window length for self-attention (T)
max_iters     = 600     # Total training iterations (fast convergence on CPU/GPU)
eval_interval = 100     # Frequency of evaluating and logging loss
learning_rate = 1e-3    # Learning rate for AdamW
eval_iters    = 25      # Number of batches to average loss over during evaluation
n_embd        = 64      # Embedding dimension (C)
n_head        = 4       # Number of multi-head attention heads
n_layer       = 4       # Number of stacked transformer blocks
dropout       = 0.1     # Dropout probability for regularization


# ==============================================================================
# 1. Dataset Loading & Character-level Tokenizer
# ==============================================================================
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

DATA_FILE = "input.txt" if os.path.exists("input.txt") else "data.txt"

def load_text_data(filepath=None):
    """
    Loads text from local file if available; otherwise uses the fallback dataset.
    Prioritizes input.txt if present, then data.txt.
    """
    if filepath is None:
        filepath = DATA_FILE

    candidates = [
        filepath,
        "input.txt",
        "data.txt",
        "data.txt.txt",
        os.path.join(os.path.dirname(__file__), filepath) if filepath else "input.txt"
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
    print("[Data] Neither 'input.txt' nor 'data.txt' found. Using default embedded fallback dataset.")
    return FALLBACK_TEXT.strip().replace("\r\n", "\n").replace("\r", "")

# Load corpus (prioritizing input.txt)
raw_text = load_text_data()

# Ensure sufficient length for splitting and batch sampling
if len(raw_text) < 1000:
    # Repeat text to ensure rich statistical sampling for both train and val splits
    multiplier = (1000 // len(raw_text)) + 2
    raw_text = raw_text * multiplier

# Build character-level vocabulary
chars = sorted(list(set(raw_text)))
vocab_size = len(chars)

# Bi-directional mapping between characters and integer indices
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(s: str):
    """Encodes a string of text into a list of integers."""
    return [stoi[c] for c in s if c in stoi]

def decode(indices):
    """Decodes a list of integers back into a text string."""
    return ''.join([itos[i] for i in indices])

# Convert entire dataset into a 1D PyTorch tensor of token IDs
data_tensor = torch.tensor(encode(raw_text), dtype=torch.long)

# 90% train, 10% validation split
split_idx = int(0.9 * len(data_tensor))
train_data = data_tensor[:split_idx]
val_data   = data_tensor[split_idx:]

print(f"[Tokenizer] Vocabulary size: {vocab_size} unique characters")
print(f"[Dataset] Train tokens: {len(train_data)} | Validation tokens: {len(val_data)}")


# ==============================================================================
# 2. Batching Pipeline
# ==============================================================================
def get_batch(split: str):
    """
    Extracts random input contexts (x) and next-token targets (y).
    x shape: (batch_size, block_size)
    y shape: (batch_size, block_size) where y[b, t] is the target token following x[b, t].
    """
    data = train_data if split == 'train' else val_data
    high = len(data) - block_size
    if high <= 0:
        raise ValueError(f"Split '{split}' has fewer tokens ({len(data)}) than block_size ({block_size}).")
    
    # Generate random starting indices for sequences
    ix = torch.randint(high, (batch_size,))
    
    # Extract input context chunk and shifted target chunk
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    
    # Transfer to target compute device (GPU or CPU)
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

        tok_emb = self.token_embedding_table(idx)                               # (B, T, n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T, n_embd)
        x = tok_emb + pos_emb                                                   # (B, T, n_embd)
        x = self.blocks(x)                                                      # (B, T, n_embd)
        x = self.ln_f(x)                                                        # (B, T, n_embd)
        logits = self.lm_head(x)                                                # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits_flat  = logits.view(B * T, C)
            targets_flat = targets.view(B * T)
            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss

    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: int = None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            
            if temperature > 0.0:
                logits = logits / temperature
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
                
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            
        return idx


# ==============================================================================
# 4. Model Training Pipeline
# ==============================================================================

@torch.no_grad()
def estimate_loss(model: nn.Module):
    """
    Evaluates model across multiple batches for both train and validation splits
    to compute an accurate, low-variance estimate of loss.
    """
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb, yb = get_batch(split)
            _, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

def main():
    # Allow overriding iterations via command line (e.g. 'python llm.py 3000')
    iters = max_iters
    for arg in sys.argv[1:]:
        if arg.isdigit():
            iters = int(arg)
            break

    print("=" * 70)
    print("INITIALIZING CUSTOM DECODER-ONLY GPT MODEL")
    print(f"Device: {device.upper()} | PyTorch Version: {torch.__version__}")
    print("=" * 70)
    
    # Instantiate Model
    model = GPTLanguageModel(vocab_size=vocab_size).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Model Parameters: {total_params:,}")
    print(f"Architecture: {n_layer} layers | {n_head} heads | {n_embd} embedding dim | {block_size} context length")
    
    # Checkpoint resuming: continue training from previous weights if available
    checkpoint_path = "model.pt"
    if os.path.exists(checkpoint_path):
        try:
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            print(f"[Checkpoint] Resuming from existing '{checkpoint_path}'. Weights preserved!")
        except Exception:
            print("[Checkpoint] Initializing fresh model weights.")
    print("-" * 70)

    # Optimizer: AdamW with weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    print(f"Beginning Training Loop ({iters} iterations)...")
    for iter_num in range(iters):
        # Periodically evaluate loss on train and validation splits
        if iter_num % eval_interval == 0 or iter_num == iters - 1:
            losses = estimate_loss(model)
            print(f"Iteration {iter_num:4d}/{iters} | Train Loss: {losses['train']:.4f} | Val Loss: {losses['val']:.4f}")

        # Sample a batch of training sequences
        xb, yb = get_batch('train')

        # Forward pass & loss calculation
        logits, loss = model(xb, yb)

        # Backward pass & parameter optimization
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    print("-" * 70)
    # Save the trained model checkpoint
    checkpoint_path = "model.pt"
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Model checkpoint successfully saved to: {checkpoint_path}")
    print("=" * 70)

    # ==============================================================================
    # 5. Text Generation Demonstration
    # ==============================================================================
    print("\n" + "=" * 70)
    print("AUTOREGRESSIVE TEXT GENERATION ENGINE")
    print("=" * 70)

    # Test prompt conditioning adapted to dataset
    if "Citizen" in raw_text or "ROMEO" in raw_text or "All:" in raw_text:
        test_prompts = [
            "First Citizen:\n",
            "Before we proceed "
        ]
    else:
        test_prompts = [
            "Vishwas AI ",
            "Trust is "
        ]

    for p in test_prompts:
        seed_tokens = encode(p)
        if len(seed_tokens) == 0:
            seed_tokens = [0]
        context = torch.tensor([seed_tokens], dtype=torch.long, device=device)
        generated_ids = model.generate(context, max_new_tokens=200, temperature=0.7)[0].tolist()
        generated_text = decode(generated_ids)
        print(f"\n--- Continuation for prompt: '{p}' ---")
        print(generated_text)

    print("=" * 70)

if __name__ == "__main__":
    main()
