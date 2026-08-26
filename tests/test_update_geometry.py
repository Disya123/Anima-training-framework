import math

import torch

from anima_trainer.update_geometry import (
    adapter_factors_from_file,
    deltas_from_lora_state,
    effective_rank,
    factor_spectra,
    layer_spectrum,
    rank_for_energy,
    spectra_report,
    truncated_lora_state,
)


def test_rank_for_energy_exact_spectrum():
    s = torch.tensor([3.0, 2.0, 1.0])
    assert rank_for_energy(s, 0.5) == 1
    assert rank_for_energy(s, 0.9) == 2
    assert rank_for_energy(s, 0.999) == 3


def test_effective_rank_uniform_vs_spiked():
    uniform = torch.ones(8)
    spiked = torch.tensor([10.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    assert effective_rank(uniform) > 7.0
    assert effective_rank(spiked) < 2.0


def test_layer_spectrum_matches_svd():
    torch.manual_seed(0)
    delta = torch.randn(64, 32) * torch.linspace(8, 0.1, 32)
    spectrum = layer_spectrum("test", delta)
    assert spectrum.shape == (64, 32)
    assert abs(spectrum.frobenius - delta.norm().item()) < 1e-4
    assert spectrum.ranks_for_energy["r99"] <= 32


def test_factor_spectra_match_dense_svd():
    torch.manual_seed(1)
    rank = 4
    A = torch.randn(rank, 96)
    B = torch.randn(48, rank)
    dense = torch.linalg.svdvals((B @ A).T if False else B @ A)
    from anima_trainer.update_geometry import factor_spectrum_pair

    fast, _ = factor_spectrum_pair(A, B)
    assert fast.shape[0] == rank
    assert torch.allclose(fast, dense[:rank], rtol=1e-4, atol=1e-5)
    assert dense[rank:].abs().max() < 1e-4


def test_truncated_lora_state_reconstruction():
    torch.manual_seed(2)
    delta = torch.randn(48, 96)
    state = truncated_lora_state({"blocks.0.mlp.layer1": delta}, lambda name, m: 6)
    up = state["lora_unet_blocks_0_mlp_layer1.lora_up.weight"]
    down = state["lora_unet_blocks_0_mlp_layer1.lora_down.weight"]
    assert up.shape == (48, 6) and down.shape == (6, 96)
    recon = up @ down
    full = torch.linalg.svdvals(delta)
    tail = full[6:].square().sum().sqrt().item()
    err = (delta - recon).norm().item()
    assert err <= tail * 1.01 + 1e-5
    assert state["lora_unet_blocks_0_mlp_layer1.alpha"].item() == 6.0


def test_deltas_from_lora_state_scaling():
    down = torch.randn(4, 16)
    up = torch.randn(32, 4)
    state = {
        "blocks.1.self_attn.q_proj.lora_down.weight": down,
        "blocks.1.self_attn.q_proj.lora_up.weight": up,
    }
    deltas = deltas_from_lora_state(state, alpha=8.0, rank=4)
    expected = up @ down * 2.0
    assert torch.allclose(deltas["blocks.1.self_attn.q_proj"], expected, atol=1e-6)


def test_spectra_report_aggregates():
    torch.manual_seed(3)
    spectra = {}
    for block in (0, 1):
        for comp, shape in (("self_attn.q_proj", (64, 32)), ("mlp.layer1", (128, 32))):
            name = f"blocks.{block}.{comp}"
            spectra[name] = layer_spectrum(name, torch.randn(*shape))
    report = spectra_report(spectra)
    assert report["n_layers"] == 4
    assert set(report["energy_by_component"]) == {"self_attn", "mlp"}
    assert set(report["energy_by_block"]) == {"0", "1"}
    assert 0 <= report["mean_effective_rank"]
