import torch
from pathlib import Path

from torch import nn

from anima_trainer.config import ScopeConfig
from anima_trainer.lora import LoRALinear, kohya_state_dict, named_lora_modules
from anima_trainer.scopes import apply_train_scope


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(4, 4, bias=False)
        self.k_proj = nn.Linear(4, 4, bias=False)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(4, 8, bias=False)
        self.layer2 = nn.Linear(8, 4, bias=False)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = Attention()
        self.cross_attn = Attention()
        self.mlp = MLP()
        self.adaln_modulation_mlp = nn.Sequential(nn.SiLU(), nn.Linear(4, 12, bias=False))


class ToyAnima(nn.Module):
    def __init__(self):
        super().__init__()
        self.x_embedder = nn.Linear(4, 4, bias=False)
        self.blocks = nn.ModuleList([Block(), Block()])
        self.final_layer = nn.Linear(4, 4, bias=False)
        self.llm_adapter = nn.Linear(4, 4, bias=False)


def test_lora_scope_is_zero_init_and_excludes_adapter():
    torch.manual_seed(0)
    model = ToyAnima()
    x = torch.randn(2, 4)
    base = model.blocks[0].self_attn.q_proj(x).detach()
    scope = ScopeConfig(
        kind="selected_blocks_lora",
        blocks=(0,),
        components=("self_attn",),
        rank=2,
        alpha=2,
        trainable_dtype="float32",
    )
    report = apply_train_scope(model, scope)
    assert report.trainable_parameters > 0
    assert isinstance(model.blocks[0].self_attn.q_proj, LoRALinear)
    assert not isinstance(model.blocks[1].self_attn.q_proj, LoRALinear)
    assert torch.allclose(model.blocks[0].self_attn.q_proj(x), base)
    assert not model.llm_adapter.weight.requires_grad


def test_kohya_export_uses_anima_comfyui_names():
    model = ToyAnima()
    apply_train_scope(
        model,
        ScopeConfig(
            kind="selected_blocks_lora",
            blocks=(1,),
            components=("mlp",),
            rank=2,
            alpha=2,
        ),
    )
    state = kohya_state_dict(model)
    assert "lora_unet_blocks_1_mlp_layer1.lora_down.weight" in state
    assert "lora_unet_blocks_1_mlp_layer1.lora_up.weight" in state
    assert all("llm_adapter" not in key for key in state)


def test_full_selected_scope_can_include_all_block_components():
    model = ToyAnima()
    report = apply_train_scope(
        model,
        ScopeConfig(kind="selected_blocks_full", blocks=(1,), components=("all",)),
    )
    assert report.trainable_parameters > 0
    assert all(not parameter.requires_grad for parameter in model.blocks[0].parameters())
    assert all(parameter.requires_grad for parameter in model.blocks[1].parameters())
    assert not model.llm_adapter.weight.requires_grad



def test_rank_overrides_hetero_ranks_and_drops():
    model = ToyAnima()
    scope = ScopeConfig(
        kind="selected_blocks_lora",
        blocks=(0, 1),
        components=("self_attn", "mlp"),
        rank=4,
        alpha=4,
        rank_overrides={
            "mlp": {"rank": 2},
            "self_attn": {"rank": 0},
            "blocks.1.mlp.layer2": {"rank": 3, "alpha": 6},
        },
        trainable_dtype="float32",
    )
    report = apply_train_scope(model, scope)
    assert report.trainable_parameters > 0
    for idx in (0, 1):
        block = model.blocks[idx]
        for attn_proj in (block.self_attn.q_proj, block.self_attn.k_proj):
            assert not isinstance(attn_proj, LoRALinear)
    layer1_0 = model.blocks[0].mlp.layer1
    layer2_0 = model.blocks[0].mlp.layer2
    layer2_1 = model.blocks[1].mlp.layer2
    assert isinstance(layer1_0, LoRALinear) and layer1_0.rank == 2 and layer1_0.alpha == 2
    assert isinstance(layer2_0, LoRALinear) and layer2_0.rank == 2 and layer2_0.alpha == 2
    assert isinstance(layer2_1, LoRALinear) and layer2_1.rank == 3 and layer2_1.alpha == 6


def test_inject_lora_all_dropped_raises():
    from anima_trainer.lora import inject_lora

    model = ToyAnima()
    try:
        inject_lora(
            model,
            ("blocks.0.self_attn.q_proj", "blocks.0.self_attn.k_proj"),
            rank=4,
            alpha=4,
            rank_overrides={"self_attn": {"rank": 0}},
        )
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError when every module is dropped")


def test_config_rank_override_validation(tmp_path):
    import pytest as _pytest

    from anima_trainer.config import load_config

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
train:
  scope:
    kind: selected_blocks_lora
    blocks: [0]
    components: [self_attn, mlp]
    rank: 8
    alpha: 8
    rank_overrides:
      mlp: {rank: 2}
      blocks.1.mlp.layer2: {rank: 3, alpha: 6}
""",
        encoding="utf-8",
    )
    cfg = load_config(config_path)
    assert cfg.train.scope.rank_overrides["mlp"]["rank"] == 2.0
    assert cfg.train.scope.rank_overrides["blocks.1.mlp.layer2"]["alpha"] == 6.0

    bad_path = tmp_path / "bad.yaml"
    bad_path.write_text(
        """
model:
  checkpoint: model.safetensors
  vae: vae.safetensors
  text_encoder: te.safetensors
data:
  manifest: manifest.jsonl
  cache_dir: cache
train:
  scope:
    rank_overrides:
      self_attn: {foo: 1}
""",
        encoding="utf-8",
    )
    with _pytest.raises(ValueError):
        load_config(bad_path)
