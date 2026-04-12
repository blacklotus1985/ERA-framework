# Metric Conventions (Source of Truth)

This file defines the metric conventions used by the executable code.

## What controls experiment outputs

For the replication workflow, the operative entry point is:
- `run_era_implicit_gender_experiment.py`

That script runs `ERAAnalyzer`, writes CSV/JSON outputs, and should be treated as the source of truth for generated results.

## L1 and L2 (distribution drift)

Supported metrics in code:
- `kl`
- `k_divergence`
- `k_divergence_normalized` (Lin K scaled to [0, 1] by dividing by `ln(2)`)
- `js_divergence`
- `js_distance`

Configured through `ERAAnalyzer(distribution_metric=...)`.

## L3 (representational drift)

Supported pairwise metrics in code:
- `cosine`
- `euclidean`

Configured through `ERAAnalyzer(l3_metric=...)`.

The L3 aggregate in summary is always the mean absolute delta of the configured metric:
- cosine run: `mean(abs(delta_cosine))`
- euclidean run: `mean(abs(delta_euclidean))`

## Output folder convention

Replication outputs are stored under:
- `era_poc_replication_results/<distribution_metric>_<l3_metric>/`

Examples:
- `era_poc_replication_results/kl_cosine/`
- `era_poc_replication_results/k_divergence_cosine/`
- `era_poc_replication_results/k_divergence_normalized_cosine/`
- `era_poc_replication_results/js_divergence_cosine/`
- `era_poc_replication_results/js_divergence_euclidean/`

## Why this file exists

Some paper/note drafts discuss Euclidean centroid formulations for L3. Those are valid methodological variants, but may not match the currently active implementation. When in doubt, trust the executable configuration and saved `run_config.json`.
