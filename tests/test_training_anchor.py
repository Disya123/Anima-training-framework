import contextlib

import torch
from torch import nn

from anima_trainer.lora import inject_lora, named_lora_modules
from anima_trainer.training import AnimaTrainer


class FakeDit(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp_layer1 = nn.Linear(4, 4, bias=False)
        inject_lora(self, ("mlp_layer1",), rank=2, alpha=2)

    def forward_latent(self, noisy_latents, sigmas, cond, checkpoint_fn=None, checkpoint_components=None, checkpoint_group_size=1, transport=None, boundary_hooks=None, checkpoint_context_fn=None):
        b, c, h, w = noisy_latents.shape
        x = noisy_latents.flatten(2).transpose(1, 2) + cond.mean(dim=1, keepdim=True)
        y = self.mlp_layer1(x)
        return y.transpose(1, 2).reshape(b, c, h, w)


def make_trainer(lora_nonzero=False):
    trainer = object.__new__(AnimaTrainer)
    trainer.device = torch.device("cpu")
    trainer.model_dtype = torch.float32
    trainer.model = FakeDit()
    if lora_nonzero:
        for _, m in named_lora_modules(trainer.model):
            with torch.no_grad():
                m.lora_down.weight.normal_(0, 0.5)
                m.lora_up.weight.normal_(0, 0.5)
    train_ns = type("T", (), {})()
    train_ns.timestep_sampling = "uniform"
    train_ns.sigmoid_scale = 1.0
    train_ns.sigma_shift = 1.0
    train_ns.gradient_checkpointing = False
    train_ns.checkpoint_mode = "off"
    config = type("C", (), {})()
    config.train = train_ns
    trainer.config = config
    trainer._ckpt_fn = None
    trainer._ckpt_comps = None
    trainer._ckpt_group = 1
    trainer._ckpt_ctx = None
    trainer._transport = frozenset()

    def _autocast():
        return contextlib.nullcontext()

    trainer._autocast = _autocast
    return trainer


def make_batch(seed=7):
    g = torch.Generator().manual_seed(seed)
    return {
        "latents": torch.randn(1, 4, 8, 8, generator=g),
        "cond": torch.randn(1, 6, generator=g),
        "cond_no_trigger": torch.randn(1, 6, generator=g),
        "weights": torch.ones(1),
    }


def test_sequential_matches_direct_computation():
    tr = make_trainer()
    batch = make_batch(11)

    class FakeFlow:
        pass

    saved_build = AnimaTrainer._build_step

    def fixed_build(self, b):
        latents = b["latents"]
        sigmas = torch.tensor([0.5])
        noisy = (1 - sigmas) * latents + sigmas * torch.zeros_like(latents)
        flow = FakeFlow()
        flow.noisy_latents = noisy
        flow.target_velocity = torch.zeros_like(latents) - latents
        return b["cond"], b["cond_no_trigger"], b["weights"], sigmas, flow

    AnimaTrainer._build_step = fixed_build
    try:
        stats = tr._sequential_step(batch, anchor_weight=0.0, inv_accum=1.0, prior_batch=None, prior_weight=0.0)
    finally:
        AnimaTrainer._build_step = saved_build

    # manual: pred = W(noisy + mean(cond)), target = -x0
    x0 = batch["latents"]
    noisy = 0.5 * x0
    inp = noisy.flatten(2).transpose(1, 2) + batch["cond"].mean(dim=1, keepdim=True)
    W = tr.model.mlp_layer1.base_layer.weight
    pred_manual = (inp @ W.T).transpose(1, 2).reshape(x0.shape)
    expected = (((pred_manual - (-x0)) ** 2).mean())
    assert abs(stats["target_loss"] - float(expected)) < 1e-5


def test_grads_flow_from_both_components_when_anchor_on():
    tr = make_trainer(lora_nonzero=True)
    stats = tr._sequential_step(make_batch(3), anchor_weight=8.0, inv_accum=1.0, prior_batch=None, prior_weight=0.0)
    assert stats["anchor_loss"] > 0.0
    grads = [p.grad for p in tr.model.parameters() if p.grad is not None]
    assert grads, "no gradients after sequential backward"


def test_anchor_weight_scales_contribution():
    tr_a = make_trainer(lora_nonzero=True)
    tr_b = make_trainer(lora_nonzero=True)
    # same weights
    tr_b.model.load_state_dict(tr_a.model.state_dict())
    s_a = tr_a._sequential_step(make_batch(5), anchor_weight=0.0, inv_accum=1.0, prior_batch=None, prior_weight=0.0)
    s_b = tr_b._sequential_step(make_batch(5), anchor_weight=4.0, inv_accum=1.0, prior_batch=None, prior_weight=0.0)
    ga = sum(p.grad.abs().sum().item() for p in tr_a.model.parameters() if p.grad is not None)
    gb = sum(p.grad.abs().sum().item() for p in tr_b.model.parameters() if p.grad is not None)
    assert s_a["anchor_loss"] is None and s_b["anchor_loss"] > 0.0
    assert gb > ga


def test_zero_lora_anchor_loss_is_zero():
    tr = make_trainer()
    stats = tr._sequential_step(make_batch(9), anchor_weight=8.0, inv_accum=1.0, prior_batch=None, prior_weight=0.0)
    assert stats["anchor_loss"] < 1e-10
