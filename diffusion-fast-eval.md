# Diffusion fast eval (LLaDA Eq. 6)

Backend: `diffusion` (Monte Carlo mask-prediction likelihood, `mc_num=128`, `mc_batch_size=16`)  
Reading / eye-tracking: not run (not implemented for diffusion)  
Data: `evaluation_data/fast_eval`

## `worldly-resonance-19`

Checkpoint: `hf/last`  
Results: `strict/results/worldly-resonance-19/last/last/zero_shot/diffusion/`

| Checkpoint | BLiMP | Supp | EWoK | Entity |
|---|---:|---:|---:|---:|
| last | 63.98 | 58.00 | 51.09 | 19.35 |

## `213248`

Checkpoint: `hf/last`  
Results: `strict/results/213248/last/last/zero_shot/diffusion/`

| Checkpoint | BLiMP | Supp | EWoK | Entity |
|---|---:|---:|---:|---:|
| last | 66.57 | 57.60 | 48.00 | 21.20 |
