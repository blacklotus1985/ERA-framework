# Changelog

All notable changes to ERA Framework will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Configurable distribution drift metrics in core analysis:
  - `kl`
  - `js_divergence`
  - `js_distance`
- Configurable L3 representational metric:
  - `cosine`
  - `euclidean`
- New metric helper functions exported from package API:
  - `compute_js_divergence`
  - `compute_js_distance`
  - `compute_distribution_drift`
  - `compute_euclidean_distance`
- Metric conventions document: `docs/METRIC_CONVENTIONS.md`

### Changed
- `ERAAnalyzer` now accepts `distribution_metric` and `l3_metric` while preserving backward-compatible defaults.
- Replication workflow now writes results into metric-specific subfolders:
  - `era_poc_replication_results/<distribution_metric>_<l3_metric>/`
- Replication workflow now writes `run_config.json` for reproducibility.

## [1.0.1] - 2024-11-30

### Added
- **Graph-based genealogy tracking** (`era/graph.py`)
  - `ModelGraph` class for tracking model evolution
  - `ModelNode`, `ModelEdge`, `RelationType` for graph structure
  - Lineage analysis and drift tracking across generations
  - Save/load genealogy graphs to JSON
- **Graph visualization** (`era/graph_viz.py`)
  - `visualize_graph()` for full genealogy visualization
  - `visualize_lineage()` for metric evolution across lineage
  - `visualize_metric_comparison()` for cross-model comparison
- **Complete genealogy example** (`examples/genealogy_analysis.ipynb`)
  - Legal AI model family demonstration
  - Integration of graph + L1/L2/L3 analysis
- **Graph module tests** (`tests/test_graph.py`)
  - Comprehensive unit tests for graph operations

### Changed
- Updated `README.md` to emphasize graph-based genealogy capabilities
- Enhanced package exports to include graph classes
- Added `networkx` to requirements for graph visualization

### Fixed
- Made `tqdm` and `torch` imports optional with graceful fallbacks
- Improved error handling in graph save/load operations

## [1.0.0] - 2024-11-30

### Added
- Initial release of ERA Framework
- Three-level drift analysis (L1: Behavioral, L2: Probabilistic, L3: Representational)
- `ERAAnalyzer` class for systematic bias evaluation
- `HuggingFaceWrapper` for model abstraction
- Metrics: KL divergence, cosine similarity, alignment score
- Visualization tools for L1/L2/L3 results
- Proof-of-concept with GPT-Neo-125M on gender bias
- Comprehensive documentation and examples
- MIT License
- Python 3.8+ support
