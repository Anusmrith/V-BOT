# ⚡ V-BOT: Neural Transformer Language Model

```text
 ██╗   ██╗      ██████╗   ██████╗ ████████╗
 ██║   ██║      ██╔══██╗ ██╔═══██╗╚══██╔══╝
 ██║   ██║█████╗██████╔╝ ██║   ██║   ██║   
 ╚██╗ ██╔╝╚════╝██╔══██╗ ██║   ██║   ██║   
  ╚████╔╝       ██████╔╝ ╚██████╔╝   ██║   
   ╚═══╝        ╚═════╝   ╚═════╝    ╚═╝   
    ⚡ NEURAL TRANSFORMER TERMINAL INTERFACE ⚡
```

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/Status-Under%20Active%20Training-orange.svg?style=flat)]()
[![Architecture](https://img.shields.io/badge/Architecture-Decoder--Only%20GPT-brightgreen.svg)]()
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey.svg)]()

> [!IMPORTANT]
> ### 🚧 Project Status: Under Active Training
> **V-BOT** is currently undergoing an active training and experimental development phase.
> * The neural model weights (`model.pt`), vocabulary, and training corpus are actively evolving.
> * Generated outputs represent ongoing training checkpoints and will continue to improve as loss converges across further iterations.

**V-BOT** is a Decoder-Only Transformer (GPT architecture) built from scratch in PyTorch. It features an autoregressive text generation engine, model training pipelines, and a cyberpunk-styled **Terminal User Interface (TUI)** with real-time character-level token streaming, live telemetry, and automatic model checkpoint hot-reloading.

---

## 🌟 Key Features

* **Pure PyTorch Architecture**: Built from fundamental deep learning principles without high-level wrapper libraries (no Hugging Face `AutoModel` or LangChain abstractions).
* **Byte-Level Byte Pair Encoding (BPE)**: Custom subword tokenizer (`tokenizer.py`) built from scratch with 256 base byte tokens (100% UTF-8 coverage, zero `[UNK]` tokens), achieving **~5.2x sequence compression** over character-level tokenization.
* **Production-Grade Transformer Optimizer**:
  * **Decoupled AdamW**: Decoupled weight decay parameter grouping (weight decay `0.05` on 2D weight matrices; `0.0` on 1D biases and LayerNorm parameters) to prevent normalization shrinkage and training instability.
  * **Learning Rate Schedule**: Linear Warmup (first 40 steps) transitioning into Cosine Annealing decay down to `min_lr = 1e-4`.
  * **Gradient Regularization**: Gradient norm clipping (`max_norm = 1.0`) preventing attention logit spikes.
  * **Dual-Domain Balanced SFT Batching**: 50/50 balanced sampling between prompt-masked aligned Q&A pairs (`ignore_index = -100`) and continuous Shakespeare corpus, ensuring 100% focused answer generation without mixing domains.
* **Cyberpunk Terminal UI (`vbot_ui.py`)**:
  * Real-time token streaming with fluid typewriter cadence.
  * Live inference metrics (token count, elapsed time, and tokens/sec throughput).
  * In-terminal parameter controls (`/temp`, `/tokens`).
  * **Automatic Hot-Reloading**: Automatically detects when a new `model.pt` checkpoint is saved and reloads weights on the fly without closing the terminal.
* **One-Click Native Terminal Launcher**:
  * Run [**`run_vbot.bat`**](run_vbot.bat) directly in Windows Terminal or PowerShell outside any IDE.
* **Flexible CLI Interfaces**:
  * `vbot_ui.py`: Full interactive terminal interface.
  * `ask.py`: Fast single-line prompt execution directly from your shell.
  * `chat.py`: Lightweight inference console.
* **Dataset Preprocessing & Tokenization Pipeline**:
  * `tokenize_dataset.py`: Fast BPE dataset tokenization and PyTorch tensor generation (`train_tokens.pt`, `val_tokens.pt`).
* **Training & Fine-Tuning**:
  * Complete local training loop in `llm.py` with multi-head self-attention and Pre-LayerNorm residual blocks.
  * Google Colab & high-quality GPU training notebooks included (`train_colab.ipynb`, `train_max_quality.ipynb`).

---

## 🧠 Model Architecture

| Parameter | Specification | Description |
| :--- | :--- | :--- |
| **Model Type** | Decoder-Only Transformer | Causal next-token autoregressive predictor |
| **Tokenizer** | Byte-Level BPE (`tokenizer.py`) | 2,048 tokens (256 base bytes + special + subword merges) |
| **Sequence Compression** | ~5.21x chars/token | 128 context window sees ~666 characters of dialogue |
| **Attention Mechanism** | Multi-Head Self-Attention | Scaled Dot-Product with lower triangular causal mask |
| **Normalization** | Pre-LayerNorm | Layer normalization before attention and feed-forward blocks |
| **Activations** | GELU | Non-linear activations across feed-forward networks |
| **Parameters** | 1,334,528 (CPU) / ~4.7M (GPU) | 4 layers, 4 heads, 128 embedding dim, 128 context |
| **Embedding Dimension ($d_{model}$)** | 128 (CPU) / 256 (GPU) | Configurable token representation space |
| **Optimizer** | Decoupled AdamW | `weight_decay=0.05` on 2D weights, `0.0` on biases & LayerNorm |
| **LR Scheduler** | Linear Warmup + Cosine Decay | `max_lr=1.5e-3`, `min_lr=1e-4`, `warmup_iters=40` |
| **Gradient Clipping** | $\ell_2$ norm $\le 1.0$ | `clip_grad_norm_` for attention logit stabilization |
| **Context Window ($T$)** | 128 tokens | Maximum sequence context length (fits complete Q&A turns) |

---

## 🚀 Quick Start

### 1. Prerequisites & Installation

Ensure you have Python 3.10+ installed. Install the PyTorch dependency:

```bash
pip install torch
```

### 2. Run V-BOT in Terminal

#### Option A: One-Click Windows Launcher (Fastest)
Double-click `run_vbot.bat` or run:
```cmd
run_vbot.bat
```

#### Option B: Run via Python
```bash
python vbot_ui.py
```

#### Option C: Single-Line Prompt CLI
Ask quick questions or test prompts directly:
```bash
python ask.py "Vishwas AI is "
```

---

## 🎮 Terminal UI Commands

Inside `vbot_ui.py`, you can run runtime commands without restarting:

| Command | Example | Description |
| :--- | :--- | :--- |
| `/temp <value>` | `/temp 0.8` | Adjust generation temperature (randomness vs determinism) |
| `/tokens <value>` | `/tokens 250` | Adjust the maximum number of tokens to generate |
| `/reload` | `/reload` | Force reload the latest trained weights from `model.pt` |
| `/info` | `/info` | Display active model architecture & hardware specs |
| `/clear` | `/clear` | Refresh and clear the terminal console |
| `/help` | `/help` | Display the list of available commands |
| `/exit` | `/exit` | Gracefully quit the application |

---

## 🏋️‍♂️ Tokenizing & Training the Model

### Step 1: Tokenize Datasets using Byte Pair Encoding (BPE)
To tokenize `combined_train.txt` (or custom dataset) into BPE token tensors:
```bash
python tokenize_dataset.py
```
This generates:
- `bpe_tokenizer.json`: The trained Byte-Level BPE tokenizer model (2,048 tokens).
- `train_tokens.pt`: PyTorch training tensor (90% split).
- `val_tokens.pt`: PyTorch validation tensor (10% split).
- `tokenized_metadata.json`: Sequence compression analytics and statistics.

### Step 2: Train V-BOT
Run the transformer training loop:
```bash
python llm.py
```
*(Tip: You can specify custom training iterations, e.g. `python llm.py 1000`)*

Once training completes:
1. Updated neural weights are saved to `model.pt`.
2. If `vbot_ui.py` is open in another terminal, it will **automatically hot-reload** the new weights!

---

## 📂 Project Structure

```text
V-BOT/
├── tokenizer.py             # Pure Python Byte-Level BPE Tokenizer engine
├── tokenize_dataset.py      # Dataset tokenization & tensor preparation pipeline
├── bpe_tokenizer.json       # Trained BPE tokenizer model (vocab & merges)
├── train_tokens.pt          # Pre-tokenized training tensor
├── val_tokens.pt            # Pre-tokenized validation tensor
├── model.pt                 # Saved trained PyTorch model weights
├── vbot_ui.py               # Interactive cyberpunk Terminal User Interface
├── run_vbot.bat             # One-click Windows native terminal launcher
├── llm.py                   # GPT model architecture & training pipeline
├── chat.py                  # Standard inference console
├── ask.py                   # Single-line prompt CLI
├── build_dataset.py         # Merges & reinforces Q&A datasets
├── combined_train.txt       # Merged training corpus for V-BOT
├── qa_dataset.txt           # Structured Vishwas AI & transformer Q&A pairs
├── data.txt                 # Specialized domain knowledge text
├── input.txt                # Raw training corpus (Shakespeare)
├── train_colab.ipynb        # Google Colab GPU training notebook
├── train_max_quality.ipynb  # Extended high-quality training notebook
├── .gitignore               # Ignored cache & temporary files
└── README.md                # Project documentation
```

---

## 📄 License

This project is open-source under the MIT License.
