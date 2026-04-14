# KappaTune Experiment

This script compares different fine-tuning strategies on a downstream task (IMDB sentiment) while measuring **catastrophic forgetting** on a general-knowledge control dataset (WikiText).

- **Baseline**: No adaptation (full 4-bit model)
- **LoRA_Global**: Classic LoRA on common projections (`q_proj`, `v_proj`, `up_proj`, `down_proj`)
- **KappaTune_LoRA**: The new `KappaTuneSelector` with relative selection (`top_p=0.2`)

The goal is to show that KappaTune achieves similar task adaptation **while forgetting less** of the original pre-trained knowledge.

### How to run

cd examples/KappaTune
Recommended: run in a clean environment with a GPU
python experiments_SA_kappatune_peft.py

### Key hyperparameters to play with

| Hyperparameter                | Location                          | Default          | What it controls                                      | Recommendation                                      |
|-------------------------------|-----------------------------------|------------------|-------------------------------------------------------|-----------------------------------------------------|
| `top_p`                       | KappaTune block                   | `0.2`            | Fraction of best (lowest κ) modules selected          | 0.1–0.3 (lower = more conservative)                 |
| `num_modules`                 | KappaTune block (alternative)     | `None`           | Fixed number of modules                               | Use instead of `top_p` for strict budget            |
| `r` (rank)                    | Both LoRA configs                 | `16` / `190`     | LoRA rank (controls trainable parameters)             | Keep total trainable params similar between runs    |
| `LR` (learning rate)          | Top of script                     | `2e-4`           | Training speed and stability                          | 1e-4 – 5e-4                                         |
| `num_train_epochs`            | Top of script                     | `10`             | Total training steps                                  | Increase for stronger adaptation                    |
| `MODEL_ID`                    | Top of script                     | DeepSeek-V2-Lite | Base model                                            | Try Mistral, Qwen, etc. MoE yields the best results |
| `max_dim_size_to_analyze`     | `KappaTuneSelector`               | `16384`          | Max matrix size for SVD (memory / speed trade-off)    | Increase only if you have very high VRAM            |
