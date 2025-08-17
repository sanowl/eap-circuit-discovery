from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns


def plot_head_importance(importance: torch.Tensor, title: str = "Head Importance (EAP)", save_path: Optional[str | Path] = None):
    imp = importance.detach().cpu().numpy()
    plt.figure(figsize=(12, 6))
    ax = sns.heatmap(imp, cmap="viridis", cbar=True)
    ax.set_xlabel("Head")
    ax.set_ylabel("Layer")
    ax.set_title(title)
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
    return ax


def plot_top_heads_graph(importance: torch.Tensor, top_k: int = 10, save_path: Optional[str | Path] = None):
    # Minimal placeholder graph: bars of top-k head scores annotated with LxHy labels
    imp = importance.detach().cpu().numpy()
    layers, heads = imp.shape
    flat = imp.reshape(-1)
    top_idx = np.argsort(-flat)[:top_k]
    scores = flat[top_idx]
    labels = [f"L{idx // heads}H{idx % heads}" for idx in top_idx]
    plt.figure(figsize=(10, 4))
    plt.bar(range(top_k), scores)
    plt.xticks(range(top_k), labels, rotation=45, ha="right")
    plt.ylabel("Importance")
    plt.title(f"Top-{top_k} Heads by EAP Importance")
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
