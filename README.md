# EuroMoE Workbench

Interactive LLM inference explorer for Mixture-of-Experts models. Step through token generation, inspect expert routing, timing breakdown, attention patterns, and hidden state evolution in real-time.

## Features

- **Step-by-step generation** — Generate one token at a time, inspect everything before continuing
- **Expert routing** — See which 8 of 64 experts the router selects per layer. Click to force/disable
- **Timing breakdown** — Per-operation timing: attention vs MoE experts vs overhead
- **Attention visualization** — Which previous tokens the current token attends to
- **Hidden state trajectory** — Chart showing token magnitude across all 24 layers
- **Token forcing** — Override the model's choice with custom tokens
- **Multi-turn chat** — Full conversation with context

## Quick Start

```bash
# Install dependencies
pip install torch transformers flask

# Run the server (requires GPU with model loaded)
python server.py

# Open in browser
open http://localhost:5000
```

## Architecture

- **Backend** (`server.py`): Flask server with PyTorch hooks on every layer. Captures routing, attention, hidden states, timing.
- **Frontend** (`index.html`): Single-file dashboard. Dark mode, orange accent. No external dependencies except Three.js CDN.

## Requirements

- Python 3.10+
- PyTorch with CUDA
- Transformers (HuggingFace)
- Flask
- GPU with 8GB+ VRAM
- EuroMoE-2.6B-A0.6B model (or any Mixtral-style MoE model)
