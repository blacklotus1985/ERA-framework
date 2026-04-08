#!/usr/bin/env python
"""Utility script for ERA regression checks.

Usage examples:

1) Create a baseline hash snapshot:
   python tools/regression_check.py snapshot --out _baseline_hashes.json

2) Compare current artifacts with a hash snapshot:
   python tools/regression_check.py compare-hash --baseline _baseline_hashes.json

3) Recompute KL/cosine metrics and compare with era_summary.json:
   python tools/regression_check.py compare-metrics --tolerance 1e-9
"""

from __future__ import annotations

import argparse
import json
import hashlib
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Ensure local package imports work when the script is run as a file.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from era import ERAAnalyzer, HuggingFaceWrapper


DEFAULT_FILES = [
    "era_poc_replication_results/era_summary.json",
    "era_poc_replication_results/era_l1_behavioral_drift.csv",
    "era_poc_replication_results/era_l2_probabilistic_drift.csv",
    "era_poc_replication_results/era_l3_representational_drift.csv",
    "era_poc_replication_results/si_results.json",
    "finetuned_gpt_neo_poc/model.safetensors",
    "finetuned_gpt_neo_poc/training_info.json",
]


def sha256_of_file(path: Path) -> str:
    if not path.exists():
        return "MISSING"
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def build_hash_snapshot(root: Path, rel_files: List[str]) -> List[Dict[str, str]]:
    rows = []
    for rel in rel_files:
        rows.append({"path": rel, "sha256": sha256_of_file(root / rel)})
    return rows


def cmd_snapshot(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    out = Path(args.out).resolve()
    snapshot = build_hash_snapshot(root, DEFAULT_FILES)
    out.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    print(f"Wrote hash snapshot to: {out}")
    return 0


def cmd_compare_hash(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    baseline = Path(args.baseline).resolve()
    rows = json.loads(baseline.read_text(encoding="utf-8"))

    all_match = True
    for row in rows:
        # Backward-compatible key handling for older snapshots.
        rel = row.get("path", row.get("Path"))
        expected = row.get("sha256", row.get("SHA256"))
        if rel is None or expected is None:
            print("DIFF\t<invalid baseline row>")
            print(f"  row={row}")
            all_match = False
            continue
        current = sha256_of_file(root / rel)
        match = expected == current
        all_match = all_match and match
        status = "MATCH" if match else "DIFF"
        print(f"{status}\t{rel}")
        if not match:
            print(f"  expected={expected}")
            print(f"  current ={current}")

    print("HASH_STATUS:", "PASS" if all_match else "FAIL")
    return 0 if all_match else 1


def cmd_compare_metrics(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    results_dir = root / "era_poc_replication_results"
    finetuned_dir = root / "finetuned_gpt_neo_poc"

    l1_df = pd.read_csv(results_dir / "era_l1_behavioral_drift.csv")
    l3_df = pd.read_csv(results_dir / "era_l3_representational_drift.csv")
    base_summary = json.loads((results_dir / "era_summary.json").read_text(encoding="utf-8"))
    si = json.loads((results_dir / "si_results.json").read_text(encoding="utf-8"))

    contexts = l1_df["context"].tolist()
    target_tokens = list(dict.fromkeys(si["token_sets"]["male_tokens"] + si["token_sets"]["female_tokens"]))
    concept_tokens = sorted(set(l3_df["token_a"].tolist()) | set(l3_df["token_b"].tolist()))

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
    base_model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")
    finetuned_model = AutoModelForCausalLM.from_pretrained(str(finetuned_dir))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_wrapper = HuggingFaceWrapper(base_model, tokenizer, device=device)
    finetuned_wrapper = HuggingFaceWrapper(finetuned_model, tokenizer, device=device)

    analyzer = ERAAnalyzer(
        base_model=base_wrapper,
        finetuned_model=finetuned_wrapper,
        device=device,
        distribution_metric="kl",
        l3_metric="cosine",
    )
    result = analyzer.analyze(
        test_contexts=contexts,
        target_tokens=target_tokens,
        concept_tokens=concept_tokens,
        topk_semantic=50,
    )

    recomputed = {
        "l1_mean_kl": float(result.summary["l1_mean_kl"]),
        "l2_mean_kl": float(result.summary["l2_mean_kl"]),
        "l3_mean_delta": float(result.summary["l3_mean_delta"]),
        "alignment_score": float(result.alignment_score),
    }

    all_match = True
    tol = float(args.tolerance)
    for k, new_val in recomputed.items():
        old_val = float(base_summary[k])
        delta = new_val - old_val
        match = abs(delta) <= tol
        all_match = all_match and match
        status = "MATCH" if match else "DIFF"
        print(f"{status}\t{k}: baseline={old_val:.12f} current={new_val:.12f} delta={delta:.12e}")

    print("METRIC_STATUS:", "PASS" if all_match else "FAIL")
    return 0 if all_match else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ERA regression checks")
    parser.add_argument("--root", default=".", help="Repository root path (default: current directory)")

    sub = parser.add_subparsers(dest="command", required=True)

    p_snapshot = sub.add_parser("snapshot", help="Write baseline hash snapshot")
    p_snapshot.add_argument("--out", default="_baseline_hashes.json", help="Output JSON path")
    p_snapshot.set_defaults(func=cmd_snapshot)

    p_hash = sub.add_parser("compare-hash", help="Compare current files vs baseline hash snapshot")
    p_hash.add_argument("--baseline", default="_baseline_hashes.json", help="Baseline JSON path")
    p_hash.set_defaults(func=cmd_compare_hash)

    p_metrics = sub.add_parser("compare-metrics", help="Recompute KL/cosine metrics and compare")
    p_metrics.add_argument("--tolerance", type=float, default=1e-9, help="Absolute tolerance")
    p_metrics.set_defaults(func=cmd_compare_metrics)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
