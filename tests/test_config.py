from pathlib import Path

import pytest

from anima_trainer.config import load_config


def test_config_resolves_paths_and_selected_scope(tmp_path):
    config_path = tmp_path / "run.yaml"
    config_path.write_text(
        """
model:
  checkpoint: model.safetensors
  vae: vae.safetensors
  text_encoder: te.safetensors
data:
  manifest: manifest.jsonl
  cache_dir: cache
concept:
  mode: character
  trigger: '@alice'
train:
  scope:
    kind: selected_blocks_lora
    blocks: [8, 9]
    components: [cross_attn, mlp]
""",
        encoding="utf-8",
    )
    config = load_config(config_path)
    assert config.model.checkpoint == (tmp_path / "model.safetensors").resolve()
    assert config.train.scope.blocks == (8, 9)
    assert config.concept.trigger == "@alice"


def test_config_rejects_unknown_keys(tmp_path):
    config_path = tmp_path / "bad.yaml"
    config_path.write_text(
        """
model:
  checkpoint: model.safetensors
  vae: vae.safetensors
  text_encoder: te.safetensors
  typo: true
data:
  manifest: manifest.jsonl
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown model keys"):
        load_config(config_path)

