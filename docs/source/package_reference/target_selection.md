# KappaTuneSelector

**Automatic LoRA target module selection based on matrix condition numbers (κ)**

`KappaTuneSelector` implements the condition-number-based target selection strategy from the [KappaTune paper](https://arxiv.org/abs/2506.16289).  
It scans every nn.Linear module and, for models where MoE expert weights are stored as fused 3D nn.Parameter tensors (e.g. Llama-4, Qwen3-MoE), also those parameters, computes the matrix condition number κ = σ_max / σ_min for each, and selects the most isotropic layers (lowest κ). These layers are the most flexible for LoRA adaptation and help mitigate catastrophic forgetting on downstream datasets.

The selector fully supports **4-bit and int8 quantized models** (bitsandbytes).

## Quick one-liner (recommended)

```python
from peft.helpers import find_kappa_target_modules

target_modules = find_kappa_target_modules(model, top_p=0.2)
```

## API reference

::: peft.helpers.KappaTuneSelector
    options:
      heading_level: 3

::: peft.helpers.find_kappa_target_modules
