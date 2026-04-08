# Diffusion Repair Improvement History

This file records every tested algorithm variant and its metrics.

## Metric Definitions
- before_viol_rate: baseline violation rate before repair
- after_viol_rate: violation rate after repair
- viol_rate_delta: after_viol_rate - before_viol_rate (negative is better)
- efficacy: fraction of originally-unsafe samples repaired
- side_effect_mse: MSE shift on originally-safe samples

## Run Log

### v0_default_e150
- timestamp: 2026-04-08T15:56:41Z
- change_summary: original settings with 20 repair steps
- config_overrides: {"beta_alm": 1.5, "ddim_steps_repair": 20, "mu_0": 5.0, "penalty_update_every": 10, "repair_batch_seeds": 200, "repair_epochs": 150, "repair_lr": 0.001}
- before_viol_rate: 0.4340
- after_viol_rate: 0.8480
- viol_rate_delta: 0.4140
- before_mean_viol: 0.5147
- after_mean_viol: 4.8716
- efficacy: 0.0000
- side_effect_mse: 19.5744
- max_weight_delta_linf: 0.0059

### v1_match_steps_e150
- timestamp: 2026-04-08T15:57:19Z
- change_summary: match repair rollout steps with eval steps (50)
- config_overrides: {"beta_alm": 1.5, "ddim_steps_repair": 50, "mu_0": 5.0, "penalty_update_every": 10, "repair_batch_seeds": 200, "repair_epochs": 150, "repair_lr": 0.001}
- before_viol_rate: 0.4340
- after_viol_rate: 0.8800
- viol_rate_delta: 0.4460
- before_mean_viol: 0.5147
- after_mean_viol: 5.2265
- efficacy: 0.0000
- side_effect_mse: 28.1803
- max_weight_delta_linf: 0.0070

### v2_fullseed_slowmu_e150
- timestamp: 2026-04-08T15:58:11Z
- change_summary: full seed batch + slower ALM penalty growth
- config_overrides: {"beta_alm": 1.1, "ddim_steps_repair": 20, "mu_0": 2.0, "penalty_update_every": 25, "repair_batch_seeds": 500, "repair_epochs": 150, "repair_lr": 0.001}
- before_viol_rate: 0.4340
- after_viol_rate: 0.7960
- viol_rate_delta: 0.3620
- before_mean_viol: 0.5147
- after_mean_viol: 4.5299
- efficacy: 0.0000
- side_effect_mse: 14.2300
- max_weight_delta_linf: 0.0060

### v3_noclamp_fullseed_e150
- timestamp: 2026-04-08T15:59:51Z
- change_summary: no-clamp eval consistency + 50 repair steps + full seed batch + slow mu
- config_overrides: {"beta_alm": 1.1, "ddim_steps_repair": 50, "mu_0": 2.0, "penalty_update_every": 25, "repair_batch_seeds": 500, "repair_epochs": 150, "repair_lr": 0.001}
- before_viol_rate: 0.4340
- after_viol_rate: 0.0000
- viol_rate_delta: -0.4340
- before_mean_viol: 0.4920
- after_mean_viol: 0.0000
- efficacy: 1.0000
- side_effect_mse: 0.4911
- max_weight_delta_linf: 0.0064

### v4_noclamp_seed1
- timestamp: 2026-04-08T16:51:31Z
- change_summary: robustness check of best config on seed=1
- config_overrides: {"beta_alm": 1.1, "ddim_steps_repair": 50, "mu_0": 2.0, "penalty_update_every": 25, "repair_batch_seeds": 500, "repair_epochs": 150, "repair_lr": 0.001, "seed": 1}
- before_viol_rate: 0.4260
- after_viol_rate: 0.0040
- viol_rate_delta: -0.4220
- before_mean_viol: 0.5022
- after_mean_viol: 0.0006
- efficacy: 0.9906
- side_effect_mse: 0.4771
- max_weight_delta_linf: 0.0074
