# Optimizing Pallet Configuration

MIT GCLOG Capstone Project — 3D bin packing for robotic palletization using Deep Reinforcement Learning and heuristic baselines.

## Problem

Items arrive sequentially on a conveyor belt and must be placed in real time into a fixed-size bin (25×32×30 units) to maximize space utilization. The algorithm looks ahead `k` items to make better decisions.

## Methods

| Method | Type | Description |
|--------|------|-------------|
| `rl`   | Deep Reinforcement Learning | CNN-based DQN that learns placement policies |
| `bl`   | Heuristic | Bottom-Left |
| `baf`  | Heuristic | Best Area Fit |
| `bssf` | Heuristic | Best Shortest Side Fit |
| `blsf` | Heuristic | Best Longest Side Fit |

## Project Structure

```
├── src/            # Core Python source code
├── notebooks/      # Experiments and results
│   ├── 01_heuristics.ipynb       # Heuristic baselines evaluation
│   ├── 02_rl_training.ipynb      # RL model training and testing
│   └── 03_rl_vs_heuristics.ipynb # Method comparison
├── data/
│   ├── input_synthetic.txt   # 1000 synthetic items (6–12 units each dim)
│   └── input_worst_case.txt  # Real worst-case dataset (scaled ÷10)
├── models/         # Trained RL models (not tracked in git — stored in OneDrive)
└── requirements.txt
```

## Installation

Requires Python 3.10 and TensorFlow 2.10.

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Run a heuristic method
python src/deeppack3d.py bl 10 --data=file --path=data/input_worst_case.txt

# Run RL inference
python src/deeppack3d.py rl 10 --data=file --path=data/input_worst_case.txt
```

## Models

Pre-trained models are stored in OneDrive and not tracked in this repository due to file size (~81MB each). To regenerate, run notebook `02_rl_training.ipynb`.

| Model file | Lookahead | Training data |
|------------|-----------|---------------|
| `k=5.h5`  | 5  | Synthetic (original repo) |
| `k=10.h5` | 10 | Synthetic (original repo) |
| `bin_25x32x30_k10_mis_datos.h5` | 10 | Worst-case dataset |

## Reference

Tsang, Y. P., Mo, D. Y., Chung, K. T., & Lee, C. K. M. (2025). A deep reinforcement learning approach for online and concurrent 3D bin packing optimisation with bin replacement strategies. *Computers in Industry*, *164*, 104202.
