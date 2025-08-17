# Circuit Discovery with Edge Attribution Patching (EAP)

This repository implements a simplified Edge Attribution Patching (EAP) workflow using TransformerLens and GPT-2 small to discover circuits for the Indirect Object Identification (IOI) task.

## Getting Started

### 1) Create environment and install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

### 2) Run the basic notebook

```bash
jupyter lab notebooks/01_basic_eap.ipynb
```

## Project Structure

```
eap-circuit-discovery/
├── src/
│   ├── eap_algorithm.py           # Main EAP implementation
│   ├── ioi_task.py                # IOI task setup and data
│   ├── circuit_finder.py          # Core circuit discovery logic
│   └── visualization.py           # Plot results
├── notebooks/
│   ├── 01_basic_eap.ipynb         # Getting started tutorial
│   ├── 02_ioi_discovery.ipynb     # IOI circuit discovery
│   └── 03_results_analysis.ipynb  # Analyze and visualize results
├── data/
│   └── ioi_examples.json          # IOI task examples
├── results/
│   ├── circuits/                  # Discovered circuits (saved tensors/JSON)
│   └── plots/                     # Generated visualizations
├── requirements.txt
├── .gitignore
└── README.md
```

## Notes
- This is a minimal, didactic implementation. For rigorous research use, you may want to incorporate batched caching, more robust loss definitions, and careful token position alignment for IOI.
- GPU strongly recommended.
