from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import numpy as np

from .eap_algorithm import EdgeAttributionPatching, EAPConfig

try:
    from transformer_lens import HookedTransformer
except Exception:  # pragma: no cover
    HookedTransformer = None  # type: ignore


@dataclass
class CircuitDiscoveryResult:
    head_importance: torch.Tensor  # (layers, heads)
    baseline_delta: float
    metadata: Dict


def run_eap_on_ioi(
    model: "HookedTransformer",
    clean_input_ids: torch.Tensor,
    corrupted_input_ids: torch.Tensor,
    target_token_idx: int,
    device: str | None = None,
) -> CircuitDiscoveryResult:
    config = EAPConfig(target_token_idx=target_token_idx, device=device or ("cuda" if torch.cuda.is_available() else "cpu"))
    eap = EdgeAttributionPatching(model, config)
    head_importance = eap.compute_head_importance(clean_input_ids, corrupted_input_ids)

    # Compute baseline delta for reference
    with torch.no_grad():
        logits_clean = model(clean_input_ids.to(config.device))
        logits_corr = model(corrupted_input_ids.to(config.device))
        batch = torch.arange(clean_input_ids.size(0), device=config.device)
        pos = target_token_idx % clean_input_ids.size(1)
        target_clean = clean_input_ids.to(config.device)[batch, pos]
        target_corr = corrupted_input_ids.to(config.device)[batch, pos]
        logp_clean = torch.log_softmax(logits_clean[batch, pos, :], dim=-1)[batch, target_clean]
        logp_corr = torch.log_softmax(logits_corr[batch, pos, :], dim=-1)[batch, target_corr]
        baseline_delta = float(((-logp_corr) - (-logp_clean)).mean().item())

    return CircuitDiscoveryResult(
        head_importance=head_importance.detach().cpu(),
        baseline_delta=baseline_delta,
        metadata={"target_token_idx": target_token_idx},
    )
