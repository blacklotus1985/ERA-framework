# ERA Comparison: KL/Cosine vs JS-Divergence/Cosine vs K-Normalized/Cosine

## Run Folders
- Baseline: `era_poc_replication_results/kl_cosine/`
- Comparator 1: `era_poc_replication_results/js_divergence_cosine/`
- Comparator 2: `era_poc_replication_results/k_divergence_normalized_cosine/`

## Summary Metrics

| metric | kl_cosine | js_divergence_cosine | k_divergence_normalized_cosine | delta_js_minus_kl | delta_k_norm_minus_kl | delta_k_norm_minus_js |
|---|---:|---:|---:|---:|---:|---:|
| l1_mean_kl | 0.004821923401 | 0.001350030449 | 0.002202335216 | -3.471892951787e-03 | -2.619588184361e-03 | 8.523047674258e-04 |
| l2_mean_kl | 2.037126727578 | 0.106981094019 | 0.157402547358 | -1.930145633559e+00 | -1.879724180221e+00 | 5.042145333805e-02 |
| l3_mean_delta | 0.000017114939 | 0.000017114939 | 0.000017114939 | 0.000000000000e+00 | 0.000000000000e+00 | 0.000000000000e+00 |
| alignment_score | 119026.234762704087 | 6250.743578958211 | 9196.792865359754 | -1.127754911837e+05 | -1.098294418973e+05 | 2.946049286402e+03 |

## Interpretation
- L1/L2 vary with the selected distribution drift metric (KL vs JS divergence vs normalized Lin K).
- L3 remains unchanged because all three runs use `l3_metric=cosine`.
- Alignment Score follows the L2/L3 ratio, so it changes with the chosen L1/L2 metric.