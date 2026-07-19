"""Shared helpers to load tokenizers from local ebdlm HF exports."""
from __future__ import annotations

import json
import pathlib
from typing import Any


def resolve_local_hf_dir(
    model_path_or_name: str, revision_name: str | None
) -> pathlib.Path:
    """Return the concrete HF export directory (e.g. runs/.../hf/last).

    ``revision=`` in transformers is a git ref, not a subdirectory. Our exports
    live at ``hf/<revision>/``.
    """
    base = pathlib.Path(model_path_or_name)
    candidates: list[pathlib.Path] = []
    if revision_name:
        candidates.append(base / revision_name)
    candidates.append(base / "last")
    candidates.append(base)
    for path in candidates:
        if path.is_dir() and (
            (path / "tokenizer.json").is_file() or (path / "config.json").is_file()
        ):
            return path
    raise FileNotFoundError(
        f"No HF export with tokenizer.json/config.json under {model_path_or_name!r} "
        f"(revision={revision_name!r})"
    )


def _apply_tokenizer_config(tokenizer: Any, config_path: pathlib.Path) -> None:
    if not config_path.is_file():
        return
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    for key in (
        "bos_token",
        "eos_token",
        "unk_token",
        "sep_token",
        "pad_token",
        "cls_token",
        "mask_token",
    ):
        value = cfg.get(key)
        if isinstance(value, dict):
            value = value.get("content")
        if value is not None and getattr(tokenizer, key, None) is None:
            setattr(tokenizer, key, value)
    if cfg.get("model_max_length") is not None:
        tokenizer.model_max_length = int(cfg["model_max_length"])


def load_tokenizer(
    model_path_or_name: str,
    revision_name: str | None = None,
    *,
    padding_side: str = "right",
):
    """Load a local export tokenizer without Auto*/from_pretrained.

    Exports saved with transformers 5.x set tokenizer_class to
    ``TokenizersBackend``, which 4.x cannot load via from_pretrained.
    """
    import evaluation_pipeline  # noqa: F401 — TokenizersBackend shim
    from transformers import PreTrainedTokenizerFast

    export_dir = resolve_local_hf_dir(model_path_or_name, revision_name)
    tok_json = export_dir / "tokenizer.json"
    if not tok_json.is_file():
        raise FileNotFoundError(f"Missing tokenizer.json in {export_dir}")

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(tok_json),
        padding_side=padding_side,
    )
    _apply_tokenizer_config(tokenizer, export_dir / "tokenizer_config.json")
    tokenizer.padding_side = padding_side
    return tokenizer
