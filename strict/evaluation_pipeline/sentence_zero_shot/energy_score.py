"""EDLM transition-energy scoring for BabyLM preference ranking.

Averages E(x_t, x_0) over Monte Carlo forward masks of the answer region.
Lower energy is better (matches NCE); we return -mean(E) so callers can
rank with the existing argmax path used for log-likelihood backends.
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor

from evaluation_pipeline.sentence_zero_shot.diffusion_likelihood import forward_process


@torch.no_grad()
def get_energy_score(
    model: torch.nn.Module,
    seq: Tensor,
    prompt_index: Tensor,
    *,
    mask_id: int,
    mc_num: int = 128,
    mc_batch_size: int = 16,
) -> float:
    """Monte Carlo mean transition energy, negated for higher-is-better ranking.

    Args:
        model: EDLM (or any module with ``energy(xt_ids, x0_ids, attention_mask)``).
        seq: Clean token ids x_0, shape [L] (no padding).
        prompt_index: Bool [L]; True = prompt (never masked).
        mask_id: Mask token id.
        mc_num: Number of Monte Carlo forward masks.
        mc_batch_size: Mini-batch size over MC samples.

    Returns:
        -mean_m E(x_t^{(m)}, x_0). Higher means lower energy / better under NCE.
    """
    if not hasattr(model, "energy"):
        raise TypeError(
            "Energy backend requires a model with an .energy(xt, x0, attention_mask) "
            "method (EDLM). AutoModelForMaskedLM is not enough — load EDLM weights."
        )
    if seq.ndim != 1:
        raise ValueError(f"Expected 1D sequence, got shape {tuple(seq.shape)}")
    if prompt_index.shape != seq.shape:
        raise ValueError("prompt_index must match seq shape")

    answer_len = int((~prompt_index).sum().item())
    if answer_len < 1:
        raise ValueError("Answer region is empty; cannot estimate energy.")

    mc_batch_size = min(mc_batch_size, mc_num)
    if mc_num % mc_batch_size != 0:
        raise ValueError(
            f"mc_num ({mc_num}) must be divisible by mc_batch_size ({mc_batch_size})"
        )

    seq_batch = seq.unsqueeze(0).repeat(mc_batch_size, 1)
    energies: list[float] = []

    for _ in range(mc_num // mc_batch_size):
        xt_ids, _p_mask = forward_process(seq_batch, prompt_index, mask_id)
        attention_mask = torch.ones_like(xt_ids)
        e = model.energy(xt_ids, seq_batch, attention_mask)  # [B]
        energies.append(e.mean().item())

    return -sum(energies) / len(energies)


def _ensure_ebdlm_on_path(ebdlm_root: str | Path | None) -> None:
    import sys

    root = Path(ebdlm_root) if ebdlm_root else Path(
        r"C:\Users\tgiles\dev\ebdlm-babylm"
    )
    root_str = str(root.resolve())
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


def _resolve_hf_load_path(
    model_path_or_name: str,
    revision: str | None,
) -> tuple[str | Path, dict]:
    """Return (load_path, from_pretrained kwargs) for a local HF export layout."""
    load_kwargs: dict = {
        "trust_remote_code": True,
        "attn_implementation": "sdpa",
    }
    if revision is not None:
        load_kwargs["revision"] = revision

    path = Path(model_path_or_name)
    if path.is_dir() and revision and (path / revision).is_dir():
        load_path: str | Path = path / revision
        load_kwargs.pop("revision", None)
    else:
        load_path = path if path.exists() else model_path_or_name
    return load_path, load_kwargs


def load_lladamdlm(
    model_path_or_name: str,
    *,
    revision: str | None = None,
    dtype: torch.dtype | None = None,
    ebdlm_root: str | Path | None = None,
) -> torch.nn.Module:
    """Load a LLaDAMDLM HF export (same path training uses for generation)."""
    _ensure_ebdlm_on_path(ebdlm_root)
    from models.ebdlm import LLaDAMDLM  # noqa: WPS433

    load_path, load_kwargs = _resolve_hf_load_path(model_path_or_name, revision)
    if dtype is not None:
        load_kwargs["torch_dtype"] = dtype
    model = LLaDAMDLM.from_pretrained(load_path, **load_kwargs)
    model.eval()
    return model


def load_edlm(
    model_path_or_name: str,
    *,
    revision: str | None = None,
    dtype: torch.dtype | None = None,
    ebdlm_root: str | Path | None = None,
) -> torch.nn.Module:
    """Load an EDLM checkpoint including the energy head.

    ``EDLM.__init__`` expects a live LLaDAMDLM, so we load the backbone with
    ``LLaDAMDLM.from_pretrained``, wrap it, then restore ``energy_head`` weights
    from the checkpoint state dict.
    """
    backbone = load_lladamdlm(
        model_path_or_name,
        revision=revision,
        dtype=dtype,
        ebdlm_root=ebdlm_root,
    )
    from models.ebdlm import EDLM  # noqa: WPS433

    model = EDLM(backbone)

    load_path, _ = _resolve_hf_load_path(model_path_or_name, revision)
    state = _load_state_dict(load_path)
    energy_state = {
        k[len("energy_head.") :]: v
        for k, v in state.items()
        if k.startswith("energy_head.")
    }
    if not energy_state:
        raise ValueError(
            f"No energy_head.* weights found in {load_path}. "
            "This checkpoint is not an EDLM export."
        )
    model.energy_head.load_state_dict(energy_state, strict=True)
    model.eval()
    return model


def _load_state_dict(load_path: str | Path) -> dict[str, Tensor]:
    load_path = Path(load_path)
    safetensors_path = load_path / "model.safetensors"
    bin_path = load_path / "pytorch_model.bin"
    if safetensors_path.is_file():
        from safetensors.torch import load_file

        return load_file(str(safetensors_path))
    if bin_path.is_file():
        return torch.load(bin_path, map_location="cpu", weights_only=True)
    raise FileNotFoundError(
        f"No model.safetensors or pytorch_model.bin under {load_path}"
    )
