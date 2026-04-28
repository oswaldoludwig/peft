import torch
import torch.nn as nn
from peft.helpers import KappaTuneSelector, find_kappa_target_modules


class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 5)
        with torch.no_grad():
            # Deterministic low-κ matrix (well-conditioned)
            eps = 0.05
            u, _, v = torch.linalg.svd(torch.randn(10, 20), full_matrices=False)
            s = 1 + eps * torch.randn(min(10, 20))
            self.fc1.weight.data = u @ torch.diag_embed(s) @ v

            # Deterministic high-κ matrix (poorly conditioned)
            u2, _, v2 = torch.linalg.svd(torch.randn(20, 5), full_matrices=False)
            s2 = torch.tensor([1000.0, 1.0, 1.0, 1.0, 1.0])
            self.fc2.weight.data = u2 @ torch.diag_embed(s2) @ v2


def test_selector_basic():
    """top_p=0.5 should always return only the well-conditioned fc1."""
    torch.manual_seed(42)
    model = SimpleMLP()
    selector = KappaTuneSelector(model)
    targets = selector.get_best_targets(top_p=0.5)
    assert len(targets) == 1
    assert targets[0] == "fc1"


def test_one_liner():
    """Test the new dict return format of find_kappa_target_modules."""
    torch.manual_seed(42)
    model = SimpleMLP()
    result = find_kappa_target_modules(model, top_p=1.0)

    assert isinstance(result, dict)
    assert "target_modules" in result
    assert "target_parameters" in result
    assert isinstance(result["target_modules"], list)
    assert isinstance(result["target_parameters"], list)

    # With top_p=1.0 we should get both linear layers
    assert len(result["target_modules"]) == 2
    assert set(result["target_modules"]) == {"fc1", "fc2"}


def test_num_modules():
    """num_modules=1 should return the best one (fc1)."""
    torch.manual_seed(42)
    model = SimpleMLP()
    targets = KappaTuneSelector(model).get_best_targets(num_modules=1)
    assert len(targets) == 1
    assert targets[0] == "fc1"


def test_kappatune_with_moe_layers():
    """Test support for fused MoE 3D parameters (target_parameters)."""
    class DummyMoE(nn.Module):
        def __init__(self):
            super().__init__()
            # Simulate fused MoE weights (num_experts, out_features, in_features)
            self.gate_up_proj = nn.Parameter(torch.randn(8, 4096, 11008))
            self.down_proj = nn.Parameter(torch.randn(8, 11008, 4096))

    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([DummyMoE() for _ in range(2)])

    model = DummyModel()

    # Test selector directly
    selector = KappaTuneSelector(model)
    target_params = selector.get_best_target_parameters(top_p=0.5)
    assert len(target_params) > 0
    assert any("gate_up_proj" in name or "down_proj" in name for name in target_params)

    # Test convenience function
    result = find_kappa_target_modules(model, top_p=0.5)
    assert len(result["target_parameters"]) > 0
    assert any("gate_up_proj" in name or "down_proj" in name for name in result["target_parameters"])
