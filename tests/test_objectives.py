import torch

from anima_trainer.objectives import bell_timestep_weights, make_flow_batch, shift_sigmas, weighted_flow_mse


def test_flow_construction_and_target():
    latents = torch.ones(2, 1, 1, 2, 2)
    noise = torch.zeros_like(latents)
    sigmas = torch.tensor([0.0, 1.0])
    flow = make_flow_batch(latents, sigmas, noise)
    assert torch.equal(flow.noisy_latents[0], latents[0])
    assert torch.equal(flow.noisy_latents[1], noise[1])
    assert torch.equal(flow.target_velocity, noise - latents)


def test_weighted_mse_is_per_sample_weighted():
    prediction = torch.tensor([[[[0.0]]], [[[2.0]]]])
    target = torch.zeros_like(prediction)
    loss = weighted_flow_mse(prediction, target, torch.tensor([3.0, 1.0]))
    assert loss.item() == 2.0


def test_weights_do_not_cancel_at_batch_one():
    """Regression for the batch-normalization no-op: at B=1 the weight must
    actually scale the loss, not cancel against its own sum."""
    prediction = torch.tensor([[[[2.0]]]])
    target = torch.zeros_like(prediction)
    plain = weighted_flow_mse(prediction, target, None)
    weighted = weighted_flow_mse(prediction, target, torch.tensor([4.0]))
    assert abs(weighted.item() - 4.0 * plain.item()) < 1e-6


def test_bell_weights_are_mean_normalized():
    grid = torch.linspace(0.0, 1.0, 4096)
    weights = bell_timestep_weights(grid)
    assert weights.min() >= 0.0
    assert abs(weights.mean().item() - 1.0) < 0.01
    peak = bell_timestep_weights(torch.tensor([0.5])).item()
    assert 1.3 < peak < 1.9


def test_fast_reduction_matches_exact_within_noise():
    torch.manual_seed(4)
    prediction = torch.randn(2, 4, 8, 8, dtype=torch.bfloat16)
    target = torch.randn_like(prediction) * 0.5
    weights = torch.tensor([1.0, 2.0])
    exact = weighted_flow_mse(prediction, target, weights)
    fast = weighted_flow_mse(prediction, target, weights, fast=True)
    rel = abs(exact.item() - fast.item()) / (abs(exact.item()) + 1e-12)
    assert rel < 0.03, f"exact={exact.item()} fast={fast.item()}"


def test_sigma_shift_matches_native_formula():
    values = torch.tensor([0.0, 0.5, 1.0])
    shifted = shift_sigmas(values, 3.0)
    assert torch.allclose(shifted, torch.tensor([0.0, 0.75, 1.0]))

