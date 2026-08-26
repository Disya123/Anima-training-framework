import torch

from anima_trainer.objectives import make_flow_batch, shift_sigmas, weighted_flow_mse


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
    assert loss.item() == 1.0


def test_sigma_shift_matches_native_formula():
    values = torch.tensor([0.0, 0.5, 1.0])
    shifted = shift_sigmas(values, 3.0)
    assert torch.allclose(shifted, torch.tensor([0.0, 0.75, 1.0]))

