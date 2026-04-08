# Diffusion Repair Results Summary
Auto-generated from `results/metrics_*.json` using rank key: (after_viol_rate asc, side_effect_mse asc).
## Best Run
- tag: v3_noclamp_fullseed_e150
- seed: 0
- before_viol_rate: 0.4340
- after_viol_rate: 0.0000
- efficacy: 1.0000
- side_effect_mse: 0.4911
- change_summary: no-clamp eval consistency + 50 repair steps + full seed batch + slow mu

## Leaderboard
| rank | tag | seed | before | after | delta | efficacy | side_effect_mse |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | v3_noclamp_fullseed_e150 | 0 | 0.4340 | 0.0000 | -0.4340 | 1.0000 | 0.4911 |
| 2 | v4_noclamp_seed1 | 1 | 0.4260 | 0.0040 | -0.4220 | 0.9906 | 0.4771 |
| 3 | v2_fullseed_slowmu_e150 | 0 | 0.4340 | 0.7960 | 0.3620 | 0.0000 | 14.2300 |
| 4 | v0_default_e150 | 0 | 0.4340 | 0.8480 | 0.4140 | 0.0000 | 19.5744 |
| 5 | v1_match_steps_e150 | 0 | 0.4340 | 0.8800 | 0.4460 | 0.0000 | 28.1803 |
