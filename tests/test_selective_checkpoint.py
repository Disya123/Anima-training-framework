import torch

from anima_trainer.model import Block, build_anima_dit


def test_block_sections_equivalent_eager_vs_whole_vs_selective():
    torch.manual_seed(0)
    block = Block(x_dim=32, context_dim=16, num_heads=4)
    block = block.double()
    B, T, H, W, D = 1, 2, 2, 2, 32
    x = torch.randn(B, T, H, W, D, dtype=torch.float64, generator=torch.Generator().manual_seed(1))
    emb = torch.randn(B, T, D, dtype=torch.float64, generator=torch.Generator().manual_seed(2))
    ctx = torch.randn(B, 5, 16, dtype=torch.float64, generator=torch.Generator().manual_seed(3))
    adaln = torch.randn(B, T, 3 * D, dtype=torch.float64, generator=torch.Generator().manual_seed(4))

    eager = block(x, emb, ctx, adaln_lora_B_T_3D=adaln)

    def passthrough(fn, *args):
        return fn(*args)

    spy_calls: list[int] = []

    def spy(fn, *args):
        spy_calls.append(1)
        return fn(*args)

    spy.components = {"self_attn", "mlp"}

    selective = block(x, emb, ctx, adaln_lora_B_T_3D=adaln, ckpt=spy)
    assert len(spy_calls) == 2  # self_attn + mlp wrapped; cross_attn left eager

    whole = block(x, emb, ctx, adaln_lora_B_T_3D=adaln, ckpt=passthrough)

    assert torch.allclose(eager, whole, atol=1e-10)
    assert torch.allclose(eager, selective, atol=1e-10)


def test_checkpoint_components_equivalent_to_block_mode_on_mini_dit():
    from torch.utils.checkpoint import checkpoint as ckpt_fn

    model = build_anima_dit(num_blocks=1, model_channels=64, num_heads=4, crossattn_emb_channels=32, adaln_lora_dim=16)
    model = model.double()
    x = torch.randn(1, 16, 1, 8, 8, dtype=torch.float64, generator=torch.Generator().manual_seed(5))
    timesteps = torch.tensor([0.5], dtype=torch.float64)
    ctx = torch.randn(1, 7, 32, dtype=torch.float64, generator=torch.Generator().manual_seed(6))

    saved_grad = torch.is_grad_enabled()
    try:
        torch.set_grad_enabled(True)
        out_block = model.forward_latent(x, timesteps, ctx, checkpoint_fn=ckpt_fn)
        base_block = model.forward_latent(x, timesteps, ctx)
        out_sections_mlp_sa = model.forward_latent(x, timesteps, ctx, checkpoint_components=("self_attn", "mlp"))
        out_sections_cross_only = model.forward_latent(x, timesteps, ctx, checkpoint_components=("cross_attn",))
        out_off = model.forward_latent(x, timesteps, ctx, checkpoint_components=())
    finally:
        torch.set_grad_enabled(saved_grad)

    ref = base_block.detach()
    for name, out in [
        ("block_mode", out_block),
        ("sections_sa_mlp", out_sections_mlp_sa),
        ("sections_cross", out_sections_cross_only),
        ("sections_off", out_off),
    ]:
        assert torch.allclose(ref, out.detach(), atol=1e-9), name


def test_coarse_group_equivalence_and_closure_hygiene():
    model = build_anima_dit(num_blocks=5, model_channels=64, num_heads=4, crossattn_emb_channels=32, adaln_lora_dim=16)
    model = model.double()
    x = torch.randn(1, 16, 1, 8, 8, dtype=torch.float64, generator=torch.Generator().manual_seed(7))
    timesteps = torch.tensor([0.3], dtype=torch.float64)
    ctx = torch.randn(1, 7, 32, dtype=torch.float64, generator=torch.Generator().manual_seed(8))

    saved = torch.is_grad_enabled()
    try:
        torch.set_grad_enabled(True)
        eager = model.forward_latent(x, timesteps, ctx)
        # grouped sizes over 5 blocks: 1(legacy whole-block), 2, 4, 7 (covers all)
        for g in (1, 2, 4, 7):
            out = model.forward_latent(x, timesteps, ctx, checkpoint_group_size=g)
            assert torch.allclose(eager.detach(), out.detach(), atol=1e-9), f"group={g}"
    finally:
        torch.set_grad_enabled(saved)

