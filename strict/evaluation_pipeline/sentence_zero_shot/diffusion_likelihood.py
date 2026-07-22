"""LLaDA-style Monte Carlo conditional log-likelihood for masked diffusion LMs.

Ported from https://github.com/ML-GSAI/LLaDA/blob/main/get_log_likelihood.py
(Eq. 6 in the LLaDA paper): mask only the answer region with a fixed mask count,
weight token CE by 1 / (l / L_answer), and average over Monte Carlo samples.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def forward_process(
    batch: Tensor,
    prompt_index: Tensor,
    mask_id: int,
) -> tuple[Tensor, Tensor]:
    """Mask a random subset of answer tokens; keep the prompt intact.

    Args:
        batch: Token ids, shape [B, L] (typically MC copies of one sequence).
        prompt_index: Bool mask of length L; True = prompt (never masked).
        mask_id: Mask token id.

    Returns:
        noisy_batch: Masked token ids [B, L].
        p_mask: Per-position mask probability used for loss weighting [B, L].
    """
    batch_size, seq_len = batch.shape
    device = batch.device

    target_positions = torch.where(~prompt_index)[0]
    target_len = target_positions.numel()
    if target_len < 1:
        raise ValueError("Answer region is empty; cannot estimate diffusion likelihood.")

    # Fixed mask counts spaced across [1, target_len] (LLaDA low-variance form).
    k = torch.randint(1, target_len + 1, (), device=device)
    x = torch.round(
        torch.linspace(
            float(k),
            k + (batch_size - 1) * (target_len / batch_size),
            steps=batch_size,
            device=device,
        )
    ).long()
    x = ((x - 1) % target_len) + 1

    indices = torch.arange(target_len, device=device).repeat(batch_size, 1)
    is_mask = indices < x.unsqueeze(1)
    for i in range(batch_size):
        is_mask[i] = is_mask[i][torch.randperm(target_len, device=device)]

    target_is_mask = is_mask
    is_mask = torch.zeros(
        batch_size,
        seq_len,
        dtype=torch.bool,
        device=device,
    )
    is_mask[:, target_positions] = target_is_mask
    noisy_batch = torch.where(is_mask, mask_id, batch)
    p_mask = (x.float() / target_len).unsqueeze(1).expand(batch_size, seq_len)
    return noisy_batch, p_mask


def _pad_stack(tensors: list[Tensor], pad_value: int | float) -> Tensor:
    """Pad 2D tensors along dim=1 and stack on dim=0 → [N, L_max]."""
    max_len = max(t.shape[1] for t in tensors)
    padded = []
    for t in tensors:
        if t.shape[1] < max_len:
            pad = t.new_full((t.shape[0], max_len - t.shape[1]), pad_value)
            t = torch.cat([t, pad], dim=1)
        padded.append(t)
    return torch.cat(padded, dim=0)


@torch.no_grad()
def get_log_likelihoods(
    model: torch.nn.Module,
    tokens: Tensor,
    attention_mask: Tensor,
    prompt_index: Tensor,
    *,
    mask_id: int,
    mc_num: int = 128,
    mc_batch_size: int = 16,
    temperature: float = 1.0,
) -> Tensor:
    """Monte Carlo conditional log-likelihoods for a batch of sequences.

    Effective GPU batch size is ``tokens.shape[0] * mc_batch_size`` (examples ×
    MC copies). Choose those so the product fits in memory.

    Args:
        model: Mask predictor (MaskedLM / LLaDAMDLM); uses ``.logits``.
        tokens: Token ids [B, L] (may be right-padded).
        attention_mask: 1 = real token, 0 = pad [B, L].
        prompt_index: Bool [B, L]; True where tokens are conditioned on (not scored).
            Padding positions should be True (treated as prompt / never masked).
        mask_id: Mask token id.
        mc_num: Number of Monte Carlo samples (LLaDA uses 128 for multi-token).
        mc_batch_size: Mini-batch size over MC samples *per example*.
        temperature: Logit temperature (BabyLM temperature sweep).

    Returns:
        Estimated log-likelihoods [B] (higher is better).
    """
    if tokens.ndim != 2:
        raise ValueError(f"Expected 2D tokens [B, L], got shape {tuple(tokens.shape)}")
    if attention_mask.shape != tokens.shape or prompt_index.shape != tokens.shape:
        raise ValueError("attention_mask and prompt_index must match tokens shape")

    batch_size = tokens.shape[0]
    device = tokens.device
    mc_batch_size = min(mc_batch_size, mc_num)
    if mc_num % mc_batch_size != 0:
        raise ValueError(
            f"mc_num ({mc_num}) must be divisible by mc_batch_size ({mc_batch_size})"
        )

    lengths = attention_mask.sum(dim=1).long()
    if (lengths < 1).any():
        raise ValueError("All sequences must have length >= 1")

    # Per-example unpadded prompt masks (answer region must be non-empty).
    prompt_indices: list[Tensor] = []
    for i in range(batch_size):
        length = int(lengths[i].item())
        pi = prompt_index[i, :length]
        if int((~pi).sum().item()) < 1:
            raise ValueError(
                f"Example {i}: answer region is empty; cannot estimate diffusion likelihood."
            )
        prompt_indices.append(pi)

    temp = max(float(temperature), 1e-6)
    # Accumulate sum of per-example MC losses; average at the end.
    loss_sums = torch.zeros(batch_size, device=device, dtype=torch.float64)
    n_mc_steps = mc_num // mc_batch_size

    for _ in range(n_mc_steps):
        noisy_chunks: list[Tensor] = []
        clean_chunks: list[Tensor] = []
        p_mask_chunks: list[Tensor] = []
        attn_chunks: list[Tensor] = []

        for i in range(batch_size):
            length = int(lengths[i].item())
            seq = tokens[i, :length]
            seq_batch = seq.unsqueeze(0).repeat(mc_batch_size, 1)
            perturbed_seq, p_mask = forward_process(
                seq_batch, prompt_indices[i], mask_id
            )
            noisy_chunks.append(perturbed_seq)
            clean_chunks.append(seq_batch)
            p_mask_chunks.append(p_mask)
            attn_chunks.append(torch.ones_like(perturbed_seq))

        # [B * mc_batch_size, L_max]
        perturbed = _pad_stack(noisy_chunks, pad_value=mask_id)
        clean = _pad_stack(clean_chunks, pad_value=0)
        p_mask = _pad_stack(p_mask_chunks, pad_value=1.0)
        attn = _pad_stack(attn_chunks, pad_value=0)

        outputs = model(input_ids=perturbed, attention_mask=attn)
        logits = outputs.logits if not isinstance(outputs, tuple) else outputs[0]
        logits = logits / temp

        mask_index = (perturbed == mask_id) & attn.bool()
        # Flattened CE over all masked positions, then reduce per example.
        token_loss = F.cross_entropy(
            logits[mask_index],
            clean[mask_index],
            reduction="none",
        ) / p_mask[mask_index]

        # Map flat masked positions back to example indices.
        # mask_index is [B * mc_batch_size, L_max]; row // mc_batch_size → example.
        flat_rows = torch.arange(
            perturbed.shape[0], device=device
        ).unsqueeze(1).expand_as(mask_index)
        example_ids = (flat_rows[mask_index] // mc_batch_size).long()
        # Sum token losses per example, then divide by mc_batch_size (same as
        # single-seq path: token_loss.sum() / mc_batch_size).
        per_example = torch.zeros(batch_size, device=device, dtype=token_loss.dtype)
        per_example.scatter_add_(0, example_ids, token_loss)
        loss_sums += (per_example / mc_batch_size).double()

    return -(loss_sums / n_mc_steps).float()


@torch.no_grad()
def get_log_likelihood(
    model: torch.nn.Module,
    seq: Tensor,
    prompt_index: Tensor,
    *,
    mask_id: int,
    mc_num: int = 128,
    mc_batch_size: int = 16,
    temperature: float = 1.0,
) -> float:
    """Monte Carlo estimate of conditional log-likelihood for one sequence.

    Args:
        model: Mask predictor (MaskedLM / LLaDAMDLM); uses ``.logits``.
        seq: 1D token ids [L] for prompt+answer (no padding).
        prompt_index: Bool [L]; True where tokens are conditioned on (not scored).
        mask_id: Mask token id.
        mc_num: Number of Monte Carlo samples (LLaDA uses 128 for multi-token).
        mc_batch_size: Mini-batch size over Monte Carlo samples.
        temperature: Logit temperature (BabyLM temperature sweep).

    Returns:
        Estimated log-likelihood (higher is better). Negation of the mean
        weighted CE over masked answer positions.
    """
    if seq.ndim != 1:
        raise ValueError(f"Expected 1D sequence, got shape {tuple(seq.shape)}")
    if prompt_index.shape != seq.shape:
        raise ValueError("prompt_index must match seq shape")

    scores = get_log_likelihoods(
        model,
        seq.unsqueeze(0),
        torch.ones(1, seq.shape[0], dtype=torch.long, device=seq.device),
        prompt_index.unsqueeze(0),
        mask_id=mask_id,
        mc_num=mc_num,
        mc_batch_size=mc_batch_size,
        temperature=temperature,
    )
    return float(scores[0].item())
