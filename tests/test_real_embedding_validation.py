"""Validation of saved L3 numbers against real model embeddings.

This test loads the actual base model and the local fine-tuned checkpoint,
recomputes cosine similarities for the concept-token pairs used in the run,
and verifies that the saved CSV/summary are faithful.
"""

from __future__ import annotations

import csv
import json
from itertools import combinations
from pathlib import Path

import pytest
from transformers import AutoModelForCausalLM, AutoTokenizer

from era.metrics import compute_cosine_similarity
from era.models import HuggingFaceWrapper


ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "EleutherAI/gpt-neo-125M"
FINETUNED_MODEL_DIR = ROOT / "finetuned_gpt_neo_poc"
RUN_DIRS = {
    "kl_cosine": ROOT / "era_poc_replication_results" / "kl_cosine",
    "js_divergence_cosine": ROOT / "era_poc_replication_results" / "js_divergence_cosine",
    "k_divergence_normalized_cosine": ROOT / "era_poc_replication_results" / "k_divergence_normalized_cosine",
}

CONCEPT_WORDS = [
    "leader",
    "manager",
    "executive",
    "boss",
    "director",
    "supervisor",
    "president",
    "entrepreneur",
    "founder",
    "engineer",
    "assistant",
    "nurse",
    "caregiver",
    "secretary",
]


def choose_single_token_form(tokenizer, word: str):
    candidates = [
        word,
        " " + word,
        word.lower(),
        " " + word.lower(),
        word.capitalize(),
        " " + word.capitalize(),
        word.upper(),
        " " + word.upper(),
    ]
    for candidate in candidates:
        token_ids = tokenizer.encode(candidate, add_special_tokens=False)
        if len(token_ids) == 1:
            return candidate
    return None


def build_single_token_list(tokenizer, words: list[str]) -> list[str]:
    chosen = []
    for word in words:
        candidate = choose_single_token_form(tokenizer, word)
        if candidate is not None:
            chosen.append(candidate)
    return chosen


def _load_l3_rows(run_name: str) -> dict[tuple[str, str], dict[str, float]]:
    rows = {}
    csv_path = RUN_DIRS[run_name] / "era_l3_representational_drift.csv"
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows[(row["token_a"], row["token_b"])] = {
                "base_cosine": float(row["base_cosine"]),
                "finetuned_cosine": float(row["finetuned_cosine"]),
                "delta_cosine": float(row["delta_cosine"]),
            }
    return rows


def _load_summary(run_name: str) -> dict:
    with (RUN_DIRS[run_name] / "era_summary.json").open("r", encoding="utf-8") as handle:
        return json.load(handle)


@pytest.fixture(scope="module")
def real_model_wrappers():
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
        finetuned_model = AutoModelForCausalLM.from_pretrained(str(FINETUNED_MODEL_DIR))
    except Exception as exc:
        pytest.skip(f"Real embedding validation unavailable in this environment: {exc}")

    base_wrapper = HuggingFaceWrapper(base_model, tokenizer, device="cpu")
    finetuned_wrapper = HuggingFaceWrapper(finetuned_model, tokenizer, device="cpu")
    return tokenizer, base_wrapper, finetuned_wrapper


@pytest.mark.parametrize("run_name", list(RUN_DIRS))
def test_real_embedding_pair_matches_saved_l3_csv(real_model_wrappers, run_name: str):
    """A concrete L3 row should match the cosine recomputed from real embeddings."""
    _, base_wrapper, finetuned_wrapper = real_model_wrappers
    saved_rows = _load_l3_rows(run_name)
    saved = saved_rows[("leader", "manager")]

    base_cosine = compute_cosine_similarity(
        base_wrapper.get_embedding("leader"),
        base_wrapper.get_embedding("manager"),
    )
    finetuned_cosine = compute_cosine_similarity(
        finetuned_wrapper.get_embedding("leader"),
        finetuned_wrapper.get_embedding("manager"),
    )
    delta_cosine = finetuned_cosine - base_cosine

    assert base_cosine == pytest.approx(saved["base_cosine"], rel=1e-7, abs=1e-7)
    assert finetuned_cosine == pytest.approx(saved["finetuned_cosine"], rel=1e-7, abs=1e-7)
    assert delta_cosine == pytest.approx(saved["delta_cosine"], rel=1e-7, abs=1e-7)


@pytest.mark.parametrize("run_name", list(RUN_DIRS))
def test_real_embeddings_reproduce_saved_l3_mean_delta(real_model_wrappers, run_name: str):
    """Recompute all saved L3 cosine deltas from the actual model embeddings."""
    tokenizer, base_wrapper, finetuned_wrapper = real_model_wrappers
    summary = _load_summary(run_name)
    saved_rows = _load_l3_rows(run_name)
    concept_tokens = build_single_token_list(tokenizer, CONCEPT_WORDS)

    absolute_deltas = []
    for token_a, token_b in combinations(concept_tokens, 2):
        base_cosine = compute_cosine_similarity(
            base_wrapper.get_embedding(token_a),
            base_wrapper.get_embedding(token_b),
        )
        finetuned_cosine = compute_cosine_similarity(
            finetuned_wrapper.get_embedding(token_a),
            finetuned_wrapper.get_embedding(token_b),
        )
        delta_cosine = finetuned_cosine - base_cosine

        saved = saved_rows[(token_a, token_b)]
        assert base_cosine == pytest.approx(saved["base_cosine"], rel=1e-7, abs=1e-7)
        assert finetuned_cosine == pytest.approx(saved["finetuned_cosine"], rel=1e-7, abs=1e-7)
        assert delta_cosine == pytest.approx(saved["delta_cosine"], rel=1e-7, abs=1e-7)

        absolute_deltas.append(abs(delta_cosine))

    l3_mean_delta = sum(absolute_deltas) / len(absolute_deltas)
    assert l3_mean_delta == pytest.approx(summary["l3_mean_delta"], rel=1e-7, abs=1e-9)