from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable, Sequence

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


def _collect_hook_names(model: "HookedTransformer") -> List[str]:
    return [f"blocks.{layer}.attn.hook_z" for layer in range(model.cfg.n_layers)]


@torch.no_grad()
def ablate_specific_heads(
    model: "HookedTransformer",
    input_ids: torch.Tensor,
    heads_to_ablate: Sequence[tuple[int, int]],
    target_token_idx: int,
) -> float:
    """Ablate specific (layer, head) pairs by zeroing their `hook_z` output and
    return the average NLL at `target_token_idx`.

    heads_to_ablate: sequence of (layer_index, head_index)
    Returns: mean NLL over the batch at the target position
    """
    device = model.cfg.device
    input_ids = input_ids.to(device)

    # Group heads by layer for efficient masking
    layer_to_heads: Dict[int, List[int]] = {}
    for layer_index, head_index in heads_to_ablate:
        layer_to_heads.setdefault(layer_index, []).append(head_index)

    def make_layer_mask_hook(head_indices: List[int]):
        def hook_fn(z: torch.Tensor, hook):
            # z: (batch, seq, n_heads, d_head)
            z = z.clone()
            z[:, :, head_indices, :] = 0.0
            return z
        return hook_fn

    fwd_hooks = []
    for layer_index, head_indices in layer_to_heads.items():
        fwd_hooks.append((f"blocks.{layer_index}.attn.hook_z", make_layer_mask_hook(head_indices)))

    logits = model.run_with_hooks(input_ids, fwd_hooks=fwd_hooks)

    batch = torch.arange(input_ids.size(0), device=device)
    pos = target_token_idx % input_ids.size(1)
    target = input_ids[batch, pos]
    log_probs = torch.log_softmax(logits[batch, pos, :], dim=-1)
    nll = -log_probs[batch, target]
    return float(nll.mean().item())


@torch.no_grad()
def evaluate_ablation_plan(
    model: "HookedTransformer",
    clean_input_ids: torch.Tensor,
    corrupted_input_ids: torch.Tensor,
    target_token_idx: int,
    heads_to_ablate: Sequence[tuple[int, int]],
) -> Dict[str, float]:
    """Evaluate the effect of ablating chosen heads on clean vs corrupted NLL.

    Returns a dict with baseline and ablated NLLs and deltas.
    """
    device = model.cfg.device
    clean_input_ids = clean_input_ids.to(device)
    corrupted_input_ids = corrupted_input_ids.to(device)

    def nll(inputs: torch.Tensor) -> float:
        logits = model(inputs)
        batch = torch.arange(inputs.size(0), device=device)
        pos = target_token_idx % inputs.size(1)
        target = inputs[batch, pos]
        log_probs = torch.log_softmax(logits[batch, pos, :], dim=-1)
        nll_vals = -log_probs[batch, target]
        return float(nll_vals.mean().item())

    baseline_clean = nll(clean_input_ids)
    baseline_corr = nll(corrupted_input_ids)

    ablated_clean = ablate_specific_heads(model, clean_input_ids, heads_to_ablate, target_token_idx)
    ablated_corr = ablate_specific_heads(model, corrupted_input_ids, heads_to_ablate, target_token_idx)

    return {
        "baseline_clean_nll": baseline_clean,
        "baseline_corrupted_nll": baseline_corr,
        "ablated_clean_nll": ablated_clean,
        "ablated_corrupted_nll": ablated_corr,
        "baseline_delta": baseline_corr - baseline_clean,
        "ablated_delta": ablated_corr - ablated_clean,
    }
