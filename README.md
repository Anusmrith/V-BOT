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
[![Architecture](https://img.shields.io/badge/Architecture-Decoder--Only%20GPT-brightgreen.svg)]()
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey.svg)]()

**V-BOT** is a Decoder-Only Transformer (GPT architecture) built from scratch in PyTorch. It features an autoregressive text generation engine, model training pipelines, and a cyberpunk-styled **Terminal User Interface (TUI)** with real-time character-level token streaming, live telemetry, and automatic model checkpoint hot-reloading.

---

## 🌟 Key Features

* **Pure PyTorch Architecture**: Built from fundamental deep learning principles without high-level wrapper libraries (no Hugging Face `AutoModel` or LangChain abstractions).
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
* **Training & Fine-Tuning**:
  * Complete local training loop in `llm.py` with multi-head self-attention and Pre-LayerNorm residual blocks.
  * Google Colab & high-quality GPU training notebooks included (`train_colab.ipynb`, `train_max_quality.ipynb`).

---

## 🧠 Model Architecture

| Parameter | Specification | Description |
| :--- | :--- | :--- |
| **Model Type** | Decoder-Only Transformer | Causal next-token autoregressive predictor |
| **Attention Mechanism** | Multi-Head Self-Attention | Scaled Dot-Product with lower triangular causal mask |
| **Normalization** | Pre-LayerNorm | Layer normalization before attention and feed-forward blocks |
| **Activations** | GELU | Non-linear activations across feed-forward networks |
| **Parameters** | ~4,776,001 | Checkpoint dependent (configurable in `llm.py`) |
| **Embedding Dimension ($d_{model}$)** | 256 | Configurable token representation space |
| **Transformer Layers ($N$)** | 6 | Stacked transformer blocks |
| **Attention Heads ($h$)** | 6 | Parallel attention heads |
| **Context Window ($T$)** | 128 tokens | Maximum sequence context length |

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

## 🏋️‍♂️ Training the Model

To train or fine-tune V-BOT on your own dataset:

1. Place your training text in `input.txt` or `data.txt`.
2. Run the training script:
   ```bash
   python llm.py
   ```
3. Once training finishes, the updated weights will be saved to `model.pt`.
4. If `vbot_ui.py` is open in another terminal, it will **automatically hot-reload** the new weights!

---

## 📂 Project Structure

```text
V-BOT/
├── model.pt                 # Saved trained PyTorch model weights
├── vbot_ui.py               # Interactive cyberpunk Terminal User Interface
├── run_vbot.bat             # One-click Windows native terminal launcher
├── llm.py                   # GPT model architecture & training pipeline
├── chat.py                  # Standard inference console
├── ask.py                   # Single-line prompt CLI
├── data.txt                 # Specialized dataset / domain text
├── input.txt                # Raw training corpus
├── train_colab.ipynb        # Google Colab training notebook
├── train_max_quality.ipynb  # Extended high-quality training notebook
├── .gitignore               # Ignored cache & temporary files
└── README.md                # Project documentation
```

---

## 📄 License

This project is open-source under the MIT License.
