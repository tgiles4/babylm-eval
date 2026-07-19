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

    prompt_len = int(prompt_index.sum().item())
    target_len = seq_len - prompt_len
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

    is_mask = torch.cat(
        (
            torch.zeros(batch_size, prompt_len, dtype=torch.bool, device=device),
            is_mask,
        ),
        dim=1,
    )
    noisy_batch = torch.where(is_mask, mask_id, batch)
    p_mask = (x.float() / target_len).unsqueeze(1).expand(batch_size, seq_len)
    return noisy_batch, p_mask


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
        mc_batch_size: Mini-batch size over MC samples.
        temperature: Logit temperature (BabyLM temperature sweep).

    Returns:
        Estimated log-likelihood (higher is better). Negation of the mean
        weighted CE over masked answer positions.
    """
    if seq.ndim != 1:
        raise ValueError(f"Expected 1D sequence, got shape {tuple(seq.shape)}")
    if prompt_index.shape != seq.shape:
        raise ValueError("prompt_index must match seq shape")

    answer_len = int((~prompt_index).sum().item())
    if answer_len < 1:
        raise ValueError("Answer region is empty; cannot estimate diffusion likelihood.")

    device = seq.device
    mc_batch_size = min(mc_batch_size, mc_num)
    if mc_num % mc_batch_size != 0:
        raise ValueError(f"mc_num ({mc_num}) must be divisible by mc_batch_size ({mc_batch_size})")

    seq_batch = seq.unsqueeze(0).repeat(mc_batch_size, 1)
    losses: list[float] = []
    temp = max(float(temperature), 1e-6)

    for _ in range(mc_num // mc_batch_size):
        perturbed_seq, p_mask = forward_process(seq_batch, prompt_index, mask_id)
        mask_index = perturbed_seq == mask_id
        attention_mask = torch.ones_like(perturbed_seq)
        outputs = model(input_ids=perturbed_seq, attention_mask=attention_mask)
        logits = outputs.logits if not isinstance(outputs, tuple) else outputs[0]
        logits = logits / temp

        token_loss = F.cross_entropy(
            logits[mask_index],
            seq_batch[mask_index],
            reduction="none",
        ) / p_mask[mask_index]
        losses.append((token_loss.sum() / mc_batch_size).item())

    return -sum(losses) / len(losses)
