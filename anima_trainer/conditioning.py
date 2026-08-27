from __future__ import annotations

import contextlib
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import Qwen2Tokenizer, Qwen3Config, Qwen3Model, T5TokenizerFast

from .loader import adapter_state_from_checkpoint
from .model import LLMAdapter


QWEN3_CONFIG = dict(
    vocab_size=151936,
    hidden_size=1024,
    intermediate_size=3072,
    num_hidden_layers=28,
    num_attention_heads=16,
    num_key_value_heads=8,
    head_dim=128,
    max_position_embeddings=32768,
    rms_norm_eps=1e-6,
    rope_theta=1_000_000.0,
)
COND_PAD_LEN = 512
QWEN_TOKENIZER_ID = "Qwen/Qwen3-0.6B"
T5_TOKENIZER_ID = "google/t5-v1_1-xxl"


def _find_tokenizer_dir(name: str, local_only: bool = True) -> str:
    from huggingface_hub import snapshot_download

    patterns = ["tokenizer*", "vocab*", "merges*", "spiece.model", "special_tokens_map.json", "config.json", "added_tokens.json"]
    try:
        return snapshot_download(name, allow_patterns=patterns, local_files_only=local_only)
    except Exception:
        if local_only:
            return snapshot_download(name, allow_patterns=patterns)
        raise


@contextlib.contextmanager
def _default_dtype(dtype: torch.dtype):
    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


class AnimaConditioner:
    """Frozen native Qwen3 -> T5-token LLMAdapter conditioning path.

    Token ids replicate the production Anima text path: raw Qwen ids without
    BOS/EOS, T5 ids including </s>, adapter output zero-padded to >= 512.
    """

    def __init__(
        self,
        text_encoder_path: str | Path,
        dit_checkpoint: str | Path,
        *,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.qwen_tokenizer = Qwen2Tokenizer.from_pretrained(_find_tokenizer_dir(QWEN_TOKENIZER_ID), local_files_only=True)
        self.t5_tokenizer = T5TokenizerFast.from_pretrained(_find_tokenizer_dir(T5_TOKENIZER_ID), local_files_only=True)

        with _default_dtype(dtype):
            qwen = Qwen3Model(Qwen3Config(**QWEN3_CONFIG))
        qwen_state = {
            key[len("model.") :] if key.startswith("model.") else key: value
            for key, value in load_file(str(text_encoder_path), device="cpu").items()
        }
        incompatible = qwen.load_state_dict(qwen_state, strict=False, assign=True)
        missing = [key for key in incompatible.missing_keys if not key.startswith("lm_head")]
        if missing or incompatible.unexpected_keys:
            raise RuntimeError(
                f"Qwen checkpoint mismatch: missing={missing[:5]} "
                f"unexpected={incompatible.unexpected_keys[:5]}"
            )
        del qwen_state

        with _default_dtype(dtype):
            adapter = LLMAdapter()
        adapter_state = adapter_state_from_checkpoint(dit_checkpoint)
        incompatible = adapter.load_state_dict(adapter_state, strict=True, assign=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError("LLMAdapter checkpoint mismatch")
        del adapter_state

        self.device = torch.device(device)
        self.dtype = dtype
        self.qwen = qwen.to(device=self.device, dtype=dtype).eval().requires_grad_(False)
        self.adapter = adapter.to(device=self.device, dtype=dtype).eval().requires_grad_(False)

    @torch.inference_mode()
    def encode(self, prompt: str) -> torch.Tensor:
        text = prompt if prompt.strip() else "."
        qwen_ids = self.qwen_tokenizer(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=COND_PAD_LEN,
            return_tensors="pt",
        ).input_ids.to(self.device)
        if qwen_ids.shape[1] == 0:
            qwen_ids = self.qwen_tokenizer(
                ".",
                add_special_tokens=False,
                truncation=True,
                max_length=COND_PAD_LEN,
                return_tensors="pt",
            ).input_ids.to(self.device)
        t5_ids = self.t5_tokenizer(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=COND_PAD_LEN,
            return_tensors="pt",
        ).input_ids.to(self.device)
        hidden = self.qwen(qwen_ids).last_hidden_state
        cond = self.adapter(hidden.to(self.dtype), t5_ids)
        if cond.shape[1] < COND_PAD_LEN:
            cond = F.pad(cond, (0, 0, 0, COND_PAD_LEN - cond.shape[1]))
        elif cond.shape[1] > COND_PAD_LEN:
            cond = cond[:, :COND_PAD_LEN]
        return cond[0].detach().to("cpu", self.dtype).contiguous()

