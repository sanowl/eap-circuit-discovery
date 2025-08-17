from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import torch
from torch import Tensor

try:
    from transformer_lens import HookedTransformer
except Exception as e:  # pragma: no cover - handled at runtime
    HookedTransformer = None  # type: ignore


@dataclass
class EAPConfig:
    """Configuration for Edge Attribution Patching.

    Attributes
    ----------
    ablation_value: float
        Value to use when ablating an edge/component (e.g., replace activation with this value).
    batch_size: int
        Number of examples to process per forward pass.
    device: str
        Torch device string, e.g., "cuda" or "cpu".
    target_token_idx: int
        Index of the target token in the sequence to evaluate the task loss on.
    head_mask_value: float
        Value to multiply attention head outputs with when ablating (0.0 -> full ablation).
    seed: int
        Random seed for reproducibility.
    """

    ablation_value: float = 0.0
    batch_size: int = 16
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    target_token_idx: int = -1
    head_mask_value: float = 0.0
    seed: int = 42


class EdgeAttributionPatching:
    """Simplified Edge Attribution Patching implementation for HookedTransformer.

    This class estimates the importance of attention heads by measuring the change
    in task performance when their outputs are replaced/ablated (patched) between
    clean and corrupted runs.
    """

    def __init__(
        self,
        model: "HookedTransformer",
        config: Optional[EAPConfig] = None,
    ) -> None:
        if model is None:
            raise ValueError("A HookedTransformer model instance is required")
        self.model = model
        self.config = config or EAPConfig()
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)
        self.model.to(self.config.device)
        self.model.eval()

    @staticmethod
    def _iter_attention_head_hooks(model: "HookedTransformer") -> Iterable[Tuple[int, str]]:
        """Yield pairs of (head_index, hook_name) for attention head outputs.

        Hook names follow TransformerLens convention: 'blocks.{layer}.attn.hook_z'.
        We'll index heads within each layer.
        """
        for layer in range(model.cfg.n_layers):
            yield layer, f"blocks.{layer}.attn.hook_z"

    @torch.no_grad()
    def compute_head_importance(
        self,
        clean_input_ids: Tensor,
        corrupted_input_ids: Tensor,
        target_token_idx: Optional[int] = None,
        loss_fn: Optional[Callable[[Tensor, Tensor], Tensor]] = None,
    ) -> Tensor:
        """Compute importance scores for each attention head via EAP.

        Parameters
        ----------
        clean_input_ids: Tensor
            Tokenized clean prompts (batch, seq_len)
        corrupted_input_ids: Tensor
            Tokenized corrupted prompts (batch, seq_len)
        target_token_idx: Optional[int]
            Index of token whose logits are evaluated. Defaults to config.target_token_idx.
        loss_fn: Optional[Callable]
            Function to compute scalar loss from logits and labels. If None, uses negative logprob
            of the correct token at target_token_idx.

        Returns
        -------
        Tensor
            Importance scores shaped (n_layers, n_heads)
        """
        device = self.config.device
        clean_input_ids = clean_input_ids.to(device)
        corrupted_input_ids = corrupted_input_ids.to(device)
        target_pos = target_token_idx if target_token_idx is not None else self.config.target_token_idx

        # Helper: compute baseline loss difference between clean and corrupted
        def compute_loss(input_ids: Tensor) -> Tensor:
            logits = self.model(input_ids)
            if loss_fn is not None:
                return loss_fn(logits, input_ids)
            # Default: negative logprob of the next token at target_pos
            # Shifted next-token prediction: label is token at target_pos
            batch_indices = torch.arange(input_ids.size(0), device=device)
            pos = target_pos % input_ids.size(1)
            target_token = input_ids[batch_indices, pos]
            logits_at_pos = logits[batch_indices, pos, :]
            log_probs = torch.log_softmax(logits_at_pos, dim=-1)
            nll = -log_probs[batch_indices, target_token]
            return nll.mean()

        baseline_clean = compute_loss(clean_input_ids)
        baseline_corrupted = compute_loss(corrupted_input_ids)
        baseline_delta = (baseline_corrupted - baseline_clean).detach()

        n_layers = self.model.cfg.n_layers
        n_heads = self.model.cfg.n_heads
        importance = torch.zeros((n_layers, n_heads), device=device)

        # For each layer, ablate each head and measure delta restoration
        for layer_idx, hook_name in self._iter_attention_head_hooks(self.model):
            def ablate_head_z(z: Tensor, hook, head_index: int) -> Tensor:
                # z shape: (batch, seq, n_heads, d_head)
                z = z.clone()
                z[:, :, head_index, :] *= self.config.head_mask_value
                return z

            for head_index in range(n_heads):
                # Run corrupted with patching for this head from clean (EAP style)
                # We'll capture z from clean, and inject into corrupted at this head.
                clean_cache = {}
                def save_clean_z(z: Tensor, hook):
                    clean_cache["z"] = z.detach()

                _ = self.model.run_with_hooks(
                    clean_input_ids,
                    fwd_hooks=[(hook_name, save_clean_z)],
                )

                def patch_corrupted_z(z: Tensor, hook):
                    # Replace only the target head stream with clean head output
                    z = z.clone()
                    if "z" in clean_cache:
                        z[:, :, head_index, :] = clean_cache["z"][:, :, head_index, :]
                    return z

                logits_corrupted_patched = self.model.run_with_hooks(
                    corrupted_input_ids,
                    fwd_hooks=[(hook_name, patch_corrupted_z)],
                )

                # Measure restoration in loss delta
                if loss_fn is not None:
                    patched_loss = loss_fn(logits_corrupted_patched, corrupted_input_ids)
                else:
                    batch_indices = torch.arange(corrupted_input_ids.size(0), device=device)
                    pos = target_pos % corrupted_input_ids.size(1)
                    target_token = corrupted_input_ids[batch_indices, pos]
                    logits_at_pos = logits_corrupted_patched[batch_indices, pos, :]
                    log_probs = torch.log_softmax(logits_at_pos, dim=-1)
                    nll = -log_probs[batch_indices, target_token]
                    patched_loss = nll.mean()

                restoration = (baseline_corrupted - patched_loss).detach()
                importance[layer_idx, head_index] = restoration

        return importance


__all__ = [
    "EAPConfig",
    "EdgeAttributionPatching",
]
