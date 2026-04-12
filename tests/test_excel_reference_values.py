"""Validation tests backed by the saved ERA CSV/JSON outputs.

These are the kinds of numbers that can be checked in Excel or LibreOffice:
the test suite recomputes them and verifies that the Python implementation
matches the exported files.
"""

from __future__ import annotations

import ast
import csv
import json
from pathlib import Path

import pytest

from era.metrics import compute_alignment_score, compute_distribution_drift


ROOT = Path(__file__).resolve().parents[1]
DOCS_AUDIT_CSV = ROOT / "docs" / "era_metric_audit_report.csv"
RUN_DIRS = {
    "kl_cosine": ROOT / "era_poc_replication_results" / "kl_cosine",
    "js_divergence_cosine": ROOT / "era_poc_replication_results" / "js_divergence_cosine",
    "k_divergence_normalized_cosine": ROOT / "era_poc_replication_results" / "k_divergence_normalized_cosine",
}


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _renormalize_union(p_dist: dict[str, float], q_dist: dict[str, float]) -> tuple[dict[str, float], dict[str, float]]:
    union_tokens = set(p_dist) | set(q_dist)
    p_probs = {token: p_dist.get(token, 0.0) for token in union_tokens}
    q_probs = {token: q_dist.get(token, 0.0) for token in union_tokens}

    p_sum = sum(p_probs.values()) or 1e-12
    q_sum = sum(q_probs.values()) or 1e-12

    p_norm = {token: value / p_sum for token, value in p_probs.items()}
    q_norm = {token: value / q_sum for token, value in q_probs.items()}
    return p_norm, q_norm


def _audit_rows_by_run() -> dict[str, dict[str, str]]:
    rows = _load_csv_rows(DOCS_AUDIT_CSV)
    return {row["run"]: row for row in rows}


def _summary_for(run_name: str) -> dict:
    return _load_json(RUN_DIRS[run_name] / "era_summary.json")


@pytest.mark.parametrize("run_name", list(RUN_DIRS))
def test_excel_audit_report_matches_saved_summary(run_name: str):
    """The Excel-friendly audit CSV should match each saved summary JSON."""
    audit_row = _audit_rows_by_run()[run_name]
    summary = _summary_for(run_name)

    assert float(audit_row["l1_mean_kl"]) == pytest.approx(summary["l1_mean_kl"], rel=1e-12, abs=1e-12)
    assert float(audit_row["l2_mean_kl"]) == pytest.approx(summary["l2_mean_kl"], rel=1e-12, abs=1e-12)
    assert float(audit_row["l3_mean_delta"]) == pytest.approx(summary["l3_mean_delta"], rel=1e-12, abs=1e-12)
    assert float(audit_row["alignment_score"]) == pytest.approx(summary["alignment_score"], rel=1e-12, abs=1e-12)


@pytest.mark.parametrize("run_name", list(RUN_DIRS))
def test_excel_recomputed_means_match_summary_and_formula(run_name: str):
    """Recompute the same means from CSV rows, as one would do in Excel."""
    run_dir = RUN_DIRS[run_name]
    summary = _summary_for(run_name)

    l1_rows = _load_csv_rows(run_dir / "era_l1_behavioral_drift.csv")
    l2_rows = _load_csv_rows(run_dir / "era_l2_probabilistic_drift.csv")
    l3_rows = _load_csv_rows(run_dir / "era_l3_representational_drift.csv")

    l1_mean = _mean([float(row["kl_divergence"]) for row in l1_rows])
    l2_mean = _mean([float(row["kl_divergence"]) for row in l2_rows])
    l3_mean = _mean([abs(float(row["delta_cosine"])) for row in l3_rows])
    alignment_score = compute_alignment_score(l2_mean_kl=l2_mean, l3_mean_delta=l3_mean)

    assert l1_mean == pytest.approx(summary["l1_mean_kl"], rel=1e-12, abs=1e-12)
    assert l2_mean == pytest.approx(summary["l2_mean_kl"], rel=1e-12, abs=1e-12)
    assert l3_mean == pytest.approx(summary["l3_mean_delta"], rel=1e-12, abs=1e-12)
    assert alignment_score == pytest.approx(summary["alignment_score"], rel=1e-12, abs=1e-12)


def test_saved_l1_row_matches_real_distribution_function():
    """A real L1 row should reproduce via the same configured drift function used by the analyzer."""
    first_row = _load_csv_rows(RUN_DIRS["kl_cosine"] / "era_l1_behavioral_drift.csv")[0]
    base_probs = ast.literal_eval(first_row["base_probs"])
    finetuned_probs = ast.literal_eval(first_row["finetuned_probs"])

    recomputed = compute_distribution_drift(base_probs, finetuned_probs, method="kl")

    assert recomputed == pytest.approx(float(first_row["kl_divergence"]), rel=1e-12, abs=1e-12)


def test_saved_l2_row_matches_real_distribution_function():
    """A real L2 row should reproduce after top-k union and renormalization, like the analyzer does."""
    first_row = _load_csv_rows(RUN_DIRS["kl_cosine"] / "era_l2_probabilistic_drift.csv")[0]
    base_topk = ast.literal_eval(first_row["base_topk"])
    finetuned_topk = ast.literal_eval(first_row["finetuned_topk"])
    base_norm, finetuned_norm = _renormalize_union(base_topk, finetuned_topk)

    recomputed = compute_distribution_drift(base_norm, finetuned_norm, method="kl")

    assert recomputed == pytest.approx(float(first_row["kl_divergence"]), rel=1e-12, abs=1e-12)