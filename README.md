# Model Belief Exploration - Web GUI

Interactive web interface for exploring model identity through systematic interrogation with token probability visualization and logit lens analysis.

## Quick Start

### 1. Create Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Server

```bash
python3 web_viewer.py
```

Server starts at: **http://localhost:5000**

## Usage

1. **Browse Experiments**: View all completed experiments on the home page
2. **New Experiment**: Click "+ New Experiment" to start an interactive session
3. **Configure**: Select model, enable logit lens, set system prompt/prefill
4. **Chat**: Ask questions and observe token probabilities
5. **Finish**: Mark identity as revealed/concealed and save results

## Features

- 🎲 **Token Probabilities**: View confidence for each generated token
- 🔬 **Logit Lens**: Heatmap showing layer-by-layer predictions
- 🏷️ **Labels**: Tag experiments (important, weird, cool, etc.)
- 🗑️ **Management**: Delete individual experiments or clear all
- 💾 **Auto-save**: GPU memory cleanup between experiments

## Models

Default models available:
- `google/gemma-2-2b-it`
- `google/gemma-2-9b-it`
- `google/gemma-2-27b-it`
- `google/gemma-3-12b-it` (default)

First run downloads model (~5-10 min for 12B model, ~24GB).

## File Structure

```
.
├── web_viewer.py              # Flask server
├── agent_transformerlens.py   # Model interaction
├── logger.py                  # Logging utility
├── templates/
│   ├── index.html            # Experiments browser
│   ├── run_experiment.html   # New experiment form
│   ├── chat.html             # Interactive chat
│   └── experiment.html       # Detailed results view
└── identity-experiments/      # Saved results (auto-created)
```

## Requirements

- Python 3.8+
- CUDA-compatible GPU (recommended, 24GB+ VRAM for 12B model)
- ~30GB disk space for model cache
