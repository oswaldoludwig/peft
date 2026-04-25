# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
from typing import List, Optional, Dict, Tuple

try:
    import bitsandbytes as bnb
except ImportError:
    bnb = None


class KappaTuneSelector:
    """
    Lightweight utility to compute per-module / per-parameter condition numbers
    (κ = σ_max / σ_min) and return the best LoRA targets.

    Supports:
    - Classic nn.Linear modules (target_modules in LoraConfig)
    - Modern fused MoE weights stored as 3D nn.Parameter (gate_up_proj / down_proj,
      gate_proj / up_proj, etc.) used in Llama-4, Qwen2_MoE, Qwen3_MoE, Mixtral,
      OLMoE and similar models. These are returned via target_parameters.
    """

    def __init__(
        self,
        model: nn.Module,
        max_dim_size_to_analyze: int = 16384,
        moe_param_suffixes: Optional[Tuple[str, ...]] = None,
    ):
        self.model = model
        self.max_dim_size_to_analyze = max_dim_size_to_analyze
        self.moe_param_suffixes = moe_param_suffixes or (
            ".gate_up_proj",
            ".down_proj",
            ".gate_proj",
            ".up_proj",
        )
        self._condition_numbers: Optional[Dict[str, float]] = None
        self._parameter_condition_numbers: Optional[Dict[str, float]] = None

    def _compute_kappas(self) -> None:
        if self._condition_numbers is not None:
            return

        # === 1. nn.Linear modules ===
        condition_numbers: Dict[str, float] = {}
        for module_name, module in self.model.named_modules():
            if not isinstance(module, nn.Linear):
                continue

            weight = module.weight
            if bnb is not None:
                if hasattr(weight, "quant_state"):  # 4-bit
                    w = bnb.functional.dequantize_4bit(
                        weight.data, weight.quant_state
                    ).float()
                elif hasattr(weight, "state") and hasattr(weight.state, "CB"):  # int8
                    w = bnb.functional.int8_vectorwise_dequant(
                        weight.state.CB, weight.state.SCB
                    ).float()
                else:
                    w = weight.data.detach().float()
            else:
                w = weight.data.detach().float()

            if any(dim > self.max_dim_size_to_analyze for dim in w.shape):
                continue

            S = torch.linalg.svdvals(w.view(w.size(0), -1))
            kappa = (S[0] / (S[-1] + 1e-8)).item()
            condition_numbers[module_name] = kappa

        self._condition_numbers = condition_numbers

        # === 2. fused MoE parameters (3D nn.Parameter) ===
        parameter_condition_numbers: Dict[str, float] = {}
        for param_name, param in self.model.named_parameters():
            if not any(param_name.endswith(s) for s in self.moe_param_suffixes):
                continue
            if param.dim() != 3:
                continue  # MoE fused weights are always 3D

            w = param.data.detach().float()
            num_experts, *expert_shape = w.shape

            # Check size of the actual 2D matrix per expert
            if any(dim > self.max_dim_size_to_analyze for dim in expert_shape):
                continue

            # Compute κ per expert and average (most representative for the block)
            kappas = []
            for expert_idx in range(num_experts):
                expert_w = w[expert_idx]  # 2D slice
                S = torch.linalg.svdvals(expert_w)
                kappa = (S[0] / (S[-1] + 1e-8)).item()
                kappas.append(kappa)

            kappa = sum(kappas) / len(kappas)
            parameter_condition_numbers[param_name] = kappa

        self._parameter_condition_numbers = parameter_condition_numbers

    def get_best_targets(
        self,
        top_p: Optional[float] = None,
        num_modules: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> list[str]:
        """
        Return the best MoE parameter targets according to one of three mutually-exclusive strategies.

        Args:
            top_p: Return the top best parameters (e.g. 0.2 = paper default).
            num_modules: Return exactly this many best parameters (fixed budget).
            threshold: Return every parameter with κ ≤ threshold (quality cutoff: lower κ = more
                isotropic weight matrix → higher flexibility for fine-tuning / higher
                differential entropy).

        Returns:
            List of parameter names (e.g. [model.layers.0.block_sparse_moe.experts.down_proj, ...])

        """
        self._compute_kappas()
        if not self._condition_numbers:
            return []

        sorted_modules = sorted(self._condition_numbers.items(), key=lambda x: x[1])

        if num_modules is not None:
            k = min(num_modules, len(sorted_modules))
            return [name for name, _ in sorted_modules[:k]]
        if top_p is not None:
            k = max(1, int(len(sorted_modules) * top_p))
            return [name for name, _ in sorted_modules[:k]]
        if threshold is not None:
            return [name for name, kappa in sorted_modules if kappa <= threshold]

        return [name for name, _ in sorted_modules]

    def get_best_target_parameters(
        self,
        top_p: Optional[float] = None,
        num_modules: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> List[str]:
        """Return best MoE parameters for LoraConfig.target_parameters."""
        self._compute_kappas()
        if not self._parameter_condition_numbers:
            return []

        sorted_params = sorted(self._parameter_condition_numbers.items(), key=lambda x: x[1])

        if num_modules is not None:
            k = min(num_modules, len(sorted_params))
            return [name for name, _ in sorted_params[:k]]
        if top_p is not None:
            k = max(1, int(len(sorted_params) * top_p))
            return [name for name, _ in sorted_params[:k]]
        if threshold is not None:
            return [name for name, kappa in sorted_params if kappa <= threshold]

        return [name for name, _ in sorted_params]


def find_kappa_target_modules(
    model: nn.Module,
    top_p: float = 0.2,
    max_dim_size_to_analyze: int = 16384,
    moe_param_suffixes: Optional[Tuple[str, ...]] = None,
) -> Dict[str, List[str]]:
    """
    One-liner convenience function.

    Returns both target_modules and target_parameters so you can do:

        targets = find_kappa_target_modules(model)
        config = LoraConfig(
            target_modules=targets["target_modules"],
            target_parameters=targets["target_parameters"],
            ...
        )
    """
    selector = KappaTuneSelector(
        model,
        max_dim_size_to_analyze=max_dim_size_to_analyze,
        moe_param_suffixes=moe_param_suffixes,
    )
    return {
        "target_modules": selector.get_best_targets(top_p=top_p),
        "target_parameters": selector.get_best_target_parameters(top_p=top_p),
    }
