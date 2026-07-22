# Diffusion fast eval (LLaDA Eq. 6)

Backend: `diffusion` (Monte Carlo mask-prediction likelihood, `mc_num=128`)  
Reading / eye-tracking: available for `good-cloud-26`; not run for the earlier entries  
Data: `evaluation_data/fast_eval`

## `worldly-resonance-19`

Checkpoint: `hf/last`  
Results: `strict/results/worldly-resonance-19/last/last/zero_shot/diffusion/`
Notes: FFN Width 2560, LR 0.0035, Batch Size 192, Seq Len 512
| Checkpoint | BLiMP | Supp | EWoK | Entity |
|---|---:|---:|---:|---:|
| last | 63.98 | 58.00 | 51.09 | 19.35 |

## `213248`

Checkpoint: `hf/last`  
Results: `strict/results/213248/last/last/zero_shot/diffusion/`
Notes: FFN Width 1152, LR 0.007, Batch Size 128, Seq Len 512
| Checkpoint | BLiMP | Supp | EWoK | Entity |
|---|---:|---:|---:|---:|
| last | 66.57 | 57.60 | 48.00 | 21.20 |

## `good-cloud-26`

Checkpoint: `hf/last`  
Run configuration: `mc_num=128`, `mc_batch_size=128`, `batch_size=64`
Notes: FFN Width 2560, LR 0.007, Batch Size 192, Seq Len 512
| Checkpoint | BLiMP | Supp | EWoK | Entity | Eye tracking | Self-paced reading |
|---|---:|---:|---:|---:|---:|---:|
| last | 63.80 | 56.00 | 50.45 | 19.43 | 0.95 | 0.01 |

## `avid-gorge-25`

Checkpoint: `hf/last`  
Run configuration: `mc_num=128`, `mc_batch_size=128`, `batch_size=64`
Notes: FFN Width 2560, LR 0.007, Batch Size 192, Seq Len 512
| Checkpoint | BLiMP | Supp | EWoK | Entity | Eye tracking | Self-paced reading |
|---|---:|---:|---:|---:|---:|---:|
| last | 59.40 | 55.20 | 49.55 | — | — | — |

Entity tracking and reading results were not present in job `8977029` output.
