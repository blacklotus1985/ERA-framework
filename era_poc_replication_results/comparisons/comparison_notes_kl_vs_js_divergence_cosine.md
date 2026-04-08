# ERA Comparison: KL/Cosine vs JS-Divergence/Cosine

## Run Folders
- Baseline: `era_poc_replication_results/kl_cosine/`
- New metric: `era_poc_replication_results/js_divergence_cosine/`

## Summary Metrics

| metric | kl_cosine | js_divergence_cosine | delta_js_minus_kl |
|---|---:|---:|---:|
| l1_mean_kl | 0.004821923401 | 0.001350030449 | -3.471892951787e-03 |
| l2_mean_kl | 2.037126727578 | 0.106981094019 | -1.930145633559e+00 |
| l3_mean_delta | 0.000017114939 | 0.000017114939 | 0.000000000000e+00 |
| alignment_score | 119026.234762704087 | 6250.743578958211 | -1.127754911837e+05 |

## Interpretation
- L1/L2 are directly affected by the chosen distribution drift metric (`kl` vs `js_divergence`).
- L3 is unchanged here because both runs use `l3_metric=cosine`.
- Alignment Score changes as a function of L2/L3 under the selected drift metric.