"""
================================================================================
V-BOT: Terminal User Interface (TUI) for Transformer Language Model
================================================================================
Provides a modern, cyberpunk-inspired terminal UI for interactive inference
with real-time token streaming, parameter controls, and formatted panels.
================================================================================
"""

import os
import sys
import time
import math
# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import torch.nn as nn
# pyrefly: ignore [missing-import]
from torch.nn import functional as F

# Enable Windows ANSI Virtual Terminal Processing
if os.name == 'nt':
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        h_stdout = kernel32.GetStdHandle(-11)
        mode = ctypes.c_ulong()
        kernel32.GetConsoleMode(h_stdout, ctypes.byref(mode))
        mode.value |= 0x0004  # ENABLE_VIRTUAL_TERMINAL_PROCESSING
        kernel32.SetConsoleMode(h_stdout, mode)
    except Exception:
        pass

# Import architecture & tokenizer from llm.py
from llm import (
    GPTLanguageModel,
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

# ==============================================================================
# UI Styling & ANSI Color Codes
# ==============================================================================
RESET     = "\033[0m"
BOLD      = "\033[1m"
DIM       = "\033[2m"
ITALIC    = "\033[3m"
UNDERLINE = "\033[4m"

# 24-bit / 256 Color Palette
CYAN      = "\033[38;5;51m"
PURPLE    = "\033[38;5;141m"
PINK      = "\033[38;5;207m"
GREEN     = "\033[38;5;84m"
YELLOW    = "\033[38;5;220m"
ORANGE    = "\033[38;5;208m"
RED       = "\033[38;5;196m"
WHITE     = "\033[97m"
MUTED     = "\033[38;5;244m"
DARK_GRAY = "\033[38;5;238m"

BG_DARK   = "\033[48;5;235m"
BG_PURPLE = "\033[48;5;54m"

BANNER_ART = f"""{CYAN}{BOLD}
 ██╗   ██╗      ██████╗   ██████╗ ████████╗
 ██║   ██║      ██╔══██╗ ██╔═══██╗╚══██╔══╝
 ██║   ██║█████╗██████╔╝ ██║   ██║   ██║   
 ╚██╗ ██╔╝╚════╝██╔══██╗ ██║   ██║   ██║   
  ╚████╔╝       ██████╔╝ ╚██████╔╝   ██║   
   ╚═══╝        ╚═════╝   ╚═════╝    ╚═╝   {RESET}
{PURPLE}{BOLD}    ⚡ NEURAL TRANSFORMER TERMINAL INTERFACE ⚡{RESET}
"""

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header(model_info):
    print(BANNER_ART)
    dev_color = GREEN if "cuda" in device.lower() else YELLOW
    
    info_line = (
        f" {MUTED}Model:{RESET} {BOLD}{CYAN}V-BOT{RESET}  │  "
        f"{MUTED}Device:{RESET} {dev_color}{device.upper()}{RESET}  │  "
        f"{MUTED}Params:{RESET} {GREEN}{model_info['params']:,}{RESET}  │  "
        f"{MUTED}Context:{RESET} {PURPLE}{model_info['block_size']}{RESET}  │  "
        f"{MUTED}Layers:{RESET} {WHITE}{model_info['layers']}{RESET}  │  "
        f"{MUTED}Vocab:{RESET} {WHITE}{model_info['vocab']}{RESET}"
    )
    
    print(f"{DARK_GRAY}╭─────────────────────────────────────────────────────────────────────────────╮{RESET}")
    print(f"{DARK_GRAY}│{RESET}{info_line} {DARK_GRAY}│{RESET}")
    print(f"{DARK_GRAY}├─────────────────────────────────────────────────────────────────────────────┤{RESET}")
    print(f"{DARK_GRAY}│{RESET} {YELLOW}{BOLD}🚧 STATUS: UNDER ACTIVE TRAINING{RESET} {MUTED}(Weights & generations are experimental){RESET} {DARK_GRAY}│{RESET}")
    print(f"{DARK_GRAY}├─────────────────────────────────────────────────────────────────────────────┤{RESET}")
    print(f"{DARK_GRAY}│{RESET} {MUTED}Commands:{RESET} {CYAN}/temp <val>{RESET}  {MUTED}•{RESET}  {CYAN}/tokens <val>{RESET}  {MUTED}•{RESET}  {GREEN}/reload{RESET}  {MUTED}•{RESET}  {CYAN}/clear{RESET}  {MUTED}•{RESET}  {CYAN}/info{RESET}  {MUTED}•{RESET}  {RED}/exit{RESET}   {DARK_GRAY}│{RESET}")
    print(f"{DARK_GRAY}╰─────────────────────────────────────────────────────────────────────────────╯{RESET}")
    print()

def load_vbot_model(checkpoint_path="model.pt"):
    """Loads weights and extracts model architecture metadata."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint '{checkpoint_path}' not found! Run training first."
        )
    
    ckpt = torch.load(checkpoint_path, map_location=device)
    v_size, embd_dim = ckpt['token_embedding_table.weight'].shape
    b_size = ckpt['position_embedding_table.weight'].shape[0]
    num_layers = len([k for k in ckpt.keys() if 'blocks.' in k and '.ln1.weight' in k])
    num_heads = len([k for k in ckpt.keys() if 'blocks.0.sa.heads.' in k and '.key.weight' in k])
    
    model = GPTLanguageModel(
        vocab_size=v_size,
        n_embd=embd_dim,
        block_size=b_size,
        n_head=num_heads,
        n_layer=num_layers
    ).to(device)
    
    model.load_state_dict(ckpt)
    model.eval()
    
    total_params = sum(p.numel() for p in model.parameters())
    
    info = {
        "vocab": v_size,
        "embd": embd_dim,
        "block_size": b_size,
        "layers": num_layers,
        "heads": num_heads,
        "params": total_params
    }
    return model, info

def stream_vbot_response(model, prompt_text, max_new_tokens=150, temperature=0.7, top_k=10):
    """
    Streams characters one-by-one with realistic generation cadence,
    returning the full generated text and generation statistics.
    """
    tokens = encode(prompt_text)
    if len(tokens) == 0:
        tokens = [0]
    
    context = torch.tensor([tokens], dtype=torch.long, device=device)
    generated_chars = []
    
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            idx_cond = context[:, -model.block_size:]
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :]
            
            if temperature > 0.0:
                logits = logits / temperature
            else:
                temperature = 0.01
                logits = logits / temperature
            
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)
            context = torch.cat((context, next_idx), dim=1)
            
            char = decode([next_idx.item()])
            generated_chars.append(char)
            
            # Real-time stdout stream
            sys.stdout.write(f"{WHITE}{char}{RESET}")
            sys.stdout.flush()
            
            # Subtle visual pacing if generating too fast on GPU
            time.sleep(0.005)
            
    elapsed = max(time.time() - start_time, 0.001)
    tok_per_sec = len(generated_chars) / elapsed
    
    return "".join(generated_chars), len(generated_chars), elapsed, tok_per_sec

def main():
    clear_screen()
    print(f"\n{CYAN}Initializing V-BOT Neural Inference Engine...{RESET}")
    print(f"{MUTED}Device target: [{device.upper()}]{RESET}")
    
    try:
        model, model_info = load_vbot_model("model.pt")
    except Exception as e:
        print(f"\n{RED}{BOLD}Failed to load model:{RESET} {e}")
        input(f"\n{MUTED}Press Enter to exit...{RESET}")
        return

    # Default runtime generation parameters
    temperature = 0.7
    max_tokens  = 160
    top_k       = 10

    clear_screen()
    print_header(model_info)

    print(f"{YELLOW}{BOLD}🚧 NOTICE:{RESET} {MUTED}V-BOT is currently under active training. Outputs are evolving & experimental.{RESET}")
    print(f"{PURPLE}{BOLD}💡 PROMPT TIP:{RESET} {MUTED}V-BOT is a causal autoregressive transformer.{RESET}")
    print(f"{MUTED}Start with phrases like: {CYAN}\"Vishwas AI is \"{MUTED}, {CYAN}\"Trust means \"{MUTED}, or {CYAN}\"To be or not \"{RESET}\n")

    checkpoint_file = "model.pt"
    last_mtime = os.path.getmtime(checkpoint_file) if os.path.exists(checkpoint_file) else 0

    while True:
        try:
            # Check for newly trained model weights on disk and auto-reload
            if os.path.exists(checkpoint_file):
                current_mtime = os.path.getmtime(checkpoint_file)
                if current_mtime > last_mtime:
                    print(f"\n{YELLOW}{BOLD}⚡ [AUTO-RELOAD]{RESET} {CYAN}Detected newly trained '{checkpoint_file}' on disk!{RESET}")
                    print(f"{MUTED}Hot-reloading fresh neural weights into V-BOT...{RESET}")
                    try:
                        model, model_info = load_vbot_model(checkpoint_file)
                        last_mtime = current_mtime
                        print(f"{GREEN}✓ Successfully loaded new weights! (Params: {model_info['params']:,}){RESET}\n")
                    except Exception as reload_err:
                        print(f"{YELLOW}⚠ Checkpoint file is being written or locked: {reload_err}{RESET}\n")

            # Styled User Prompt Box
            badge = f"{DARK_GRAY}[{CYAN}temp:{temperature:.2f}{DARK_GRAY} | {PURPLE}tokens:{max_tokens}{DARK_GRAY}]{RESET}"
            print(f"{DARK_GRAY}╭──{RESET} {BOLD}{CYAN}👤 You{RESET} {badge}")
            prompt = input(f"{DARK_GRAY}╰─❯{RESET} {BOLD}").strip()
            print(f"{RESET}", end="")

            if not prompt:
                continue

            # Command Handling
            cmd = prompt.lower()
            if cmd in ['/exit', '/quit', 'exit', 'quit']:
                print(f"\n{GREEN}Shutting down V-BOT. See you next time!{RESET}\n")
                break

            if cmd in ['/reload', 'reload']:
                print(f"\n{CYAN}Reloading model weights from '{checkpoint_file}'...{RESET}")
                try:
                    model, model_info = load_vbot_model(checkpoint_file)
                    last_mtime = os.path.getmtime(checkpoint_file) if os.path.exists(checkpoint_file) else 0
                    print(f"{GREEN}✓ Model reloaded successfully! (Params: {model_info['params']:,}){RESET}\n")
                except Exception as e:
                    print(f"{RED}Failed to reload model: {e}{RESET}\n")
                continue

            if cmd == '/clear':
                clear_screen()
                print_header(model_info)
                continue

            if cmd == '/info':
                print(f"\n{DARK_GRAY}┌── {BOLD}{PURPLE}V-BOT ARCHITECTURE DETAILS{RESET}")
                print(f"{DARK_GRAY}│{RESET} Parameters:    {GREEN}{model_info['params']:,}{RESET}")
                print(f"{DARK_GRAY}│{RESET} Vocab Size:    {WHITE}{model_info['vocab']}{RESET}")
                print(f"{DARK_GRAY}│{RESET} Embedding Dim: {WHITE}{model_info['embd']}{RESET}")
                print(f"{DARK_GRAY}│{RESET} Context Block: {WHITE}{model_info['block_size']} tokens{RESET}")
                print(f"{DARK_GRAY}│{RESET} Attention:     {WHITE}{model_info['heads']} heads, {model_info['layers']} layers{RESET}")
                print(f"{DARK_GRAY}│{RESET} Hardware:      {YELLOW}{device.upper()}{RESET}")
                print(f"{DARK_GRAY}└────────────────────────────────────────{RESET}\n")
                continue

            if cmd.startswith('/temp'):
                parts = prompt.split()
                if len(parts) > 1:
                    try:
                        new_temp = float(parts[1])
                        if 0.01 <= new_temp <= 2.5:
                            temperature = new_temp
                            print(f"{GREEN}✓ Temperature set to {temperature:.2f}{RESET}\n")
                        else:
                            print(f"{YELLOW}⚠ Please specify temperature between 0.05 and 2.5{RESET}\n")
                    except ValueError:
                        print(f"{RED}⚠ Invalid number for temperature.{RESET}\n")
                else:
                    print(f"{MUTED}Current temperature: {temperature:.2f} (usage: /temp 0.8){RESET}\n")
                continue

            if cmd.startswith('/tokens'):
                parts = prompt.split()
                if len(parts) > 1:
                    try:
                        new_tokens = int(parts[1])
                        if 10 <= new_tokens <= 1000:
                            max_tokens = new_tokens
                            print(f"{GREEN}✓ Max tokens set to {max_tokens}{RESET}\n")
                        else:
                            print(f"{YELLOW}⚠ Max tokens must be between 10 and 1000{RESET}\n")
                    except ValueError:
                        print(f"{RED}⚠ Invalid integer for max tokens.{RESET}\n")
                else:
                    print(f"{MUTED}Current max tokens: {max_tokens} (usage: /tokens 200){RESET}\n")
                continue

            if cmd == '/help':
                print(f"\n{DARK_GRAY}┌── {BOLD}{CYAN}AVAILABLE COMMANDS{RESET}")
                print(f"{DARK_GRAY}│{RESET} {CYAN}/temp <val>{RESET}   : Adjust sampling randomness (e.g. /temp 0.5)")
                print(f"{DARK_GRAY}│{RESET} {CYAN}/tokens <val>{RESET} : Adjust continuation length (e.g. /tokens 250)")
                print(f"{DARK_GRAY}│{RESET} {GREEN}/reload{RESET}       : Force reload latest weights from model.pt")
                print(f"{DARK_GRAY}│{RESET} {CYAN}/clear{RESET}        : Clear terminal console")
                print(f"{DARK_GRAY}│{RESET} {CYAN}/info{RESET}         : Inspect model architecture")
                print(f"{DARK_GRAY}│{RESET} {RED}/exit{RESET}         : Quit V-BOT")
                print(f"{DARK_GRAY}└────────────────────────────────────────{RESET}\n")
                continue

            # V-BOT Response Box
            print(f"{DARK_GRAY}╭──{RESET} {BOLD}{PURPLE}🤖 V-BOT{RESET} {MUTED}(generating...){RESET}")
            print(f"{DARK_GRAY}│{RESET} {CYAN}{prompt}{RESET}", end="")
            
            output, count, duration, tps = stream_vbot_response(
                model=model,
                prompt_text=prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k
            )
            
            print()
            stats = f"{MUTED}⏱ {count} tokens in {duration:.2f}s ({tps:.1f} tok/s){RESET}"
            print(f"{DARK_GRAY}╰─────────────────────────────────────────────────── {stats}\n")

        except KeyboardInterrupt:
            print(f"\n\n{YELLOW}Session paused. Type /exit or press Ctrl+C again to quit.{RESET}\n")
        except Exception as e:
            print(f"\n{RED}Error during inference: {e}{RESET}\n")

if __name__ == "__main__":
    main()
