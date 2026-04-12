#!/usr/bin/env python
"""
ERA audit utility - versione commentata passo per passo
=======================================================

Scopo del file
--------------
Questo script controlla che i risultati ERA salvati su disco siano coerenti.

In particolare, per ogni cartella run dentro `era_poc_replication_results/`:

1. legge i CSV L1, L2, L3;
2. legge `era_summary.json`;
3. legge `si_results.json`;
4. ricalcola alcuni valori chiave dai file grezzi;
5. confronta i valori ricalcolati con quelli salvati;
6. stampa `PASS` o `FAIL` per ogni controllo.

Questo script NON rifa il fine-tuning e NON ricalcola l'intera pipeline ERA
dal modello: controlla la coerenza interna dei file gia salvati.

Esempi d'uso
------------
Audit base:
    python tools/audit_era_results.py

Audit di run specifiche:
    python tools/audit_era_results.py --runs kl_cosine js_divergence_cosine

Audit con export CSV:
    python tools/audit_era_results.py --export docs/era_metric_audit_report.csv --export-format csv

Audit con export Markdown:
    python tools/audit_era_results.py --export docs/era_metric_audit_report.md --export-format markdown
"""

from __future__ import annotations

# argparse = parsing degli argomenti da linea di comando.
# Serve per opzioni tipo --runs, --export, --tolerance.
import argparse

# ast = parsing sicuro di letterali Python da stringa.
# Qui serve per convertire in dict le colonne CSV che contengono
# stringhe come "{'man': 0.3, 'woman': 0.7}".
import ast

# json = lettura/scrittura di file JSON.
# Usiamo json per aprire era_summary.json, si_results.json, run_config.json.
import json

# math = funzioni matematiche standard.
# Qui usiamo math.isclose per confronti numerici con tolleranza.
import math

# dataclass = classi-contenitore con meno boilerplate.
# Utile per rappresentare in modo pulito singoli finding di audit
# e una riga di riepilogo per report finale.
from dataclasses import dataclass

# Path = manipolazione robusta dei percorsi file/cartelle.
# Preferibile a concatenazioni stringa con slash manuali.
from pathlib import Path

# Iterable, List = type hints per rendere piu chiaro che tipo di dati
# si aspettano le funzioni.
from typing import Iterable, List

# pandas = lettura dei CSV e calcolo di medie/colonne.
import pandas as pd


# =============================================================================
# COSTANTI GLOBALI
# =============================================================================

# REPO_ROOT = root del repository.
# __file__ = path assoluto di questo script.
# .resolve() normalizza il percorso.
# .parents[1] risale di due livelli: tools/audit_era_results.py -> repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]

# DEFAULT_RESULTS_DIR = cartella standard dove stanno le run ERA salvate.
DEFAULT_RESULTS_DIR = REPO_ROOT / "era_poc_replication_results"

# DEFAULT_RUNS = elenco delle tre run principali che vogliamo controllare.
# Queste sono le cartelle usate piu spesso nel confronto fra metriche.
DEFAULT_RUNS = [
    "kl_cosine",
    "js_divergence_cosine",
    "k_divergence_normalized_cosine",
]


# =============================================================================
# DATACLASS DI SUPPORTO
# =============================================================================

@dataclass
class AuditFinding:
    """Singolo controllo prodotto dall'audit.

    Attributi:
        run     = nome della cartella run, es. 'kl_cosine'
        check   = nome sintetico del controllo, es. 'l1_mean_kl'
        ok      = True se il controllo passa, False altrimenti
        details = dettaglio testuale utile per capire il confronto
    """

    run: str
    check: str
    ok: bool
    details: str


@dataclass
class AuditRunSummary:
    """Riepilogo compatto di una run, usato per export CSV/Markdown.

    Questo NON rappresenta tutti i controlli di audit.
    Rappresenta solo i principali numeri finali utili da tabellare.
    """

    run: str
    distribution_metric: str
    l3_metric: str
    l1_mean_kl: float
    l2_mean_kl: float
    l3_mean_delta: float
    alignment_score: float
    base_si: float | None
    finetuned_si: float | None
    delta_si_ft_minus_base: float | None


# =============================================================================
# FUNZIONI DI BASE
# =============================================================================

def load_json(path: Path) -> dict:
    """Legge un file JSON e restituisce un dict Python.

    Passi interni:
    1. apre il file come testo UTF-8;
    2. ne legge il contenuto;
    3. lo converte da JSON a oggetto Python.
    """

    return json.loads(path.read_text(encoding="utf-8"))


def nearly_equal(a: float, b: float, tolerance: float) -> bool:
    """Confronta due float con tolleranza assoluta.

    Perche non usare `a == b`?
    Perche i float possono differire di pochissimo per round-off numerico.

    rel_tol=0.0:
        ignoriamo la tolleranza relativa.
    abs_tol=tolerance:
        usiamo solo una tolleranza assoluta fissata dall'utente.
    """

    return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=tolerance)


def gap_from_probs(probs: dict, male_tokens: set[str], female_tokens: set[str]) -> float:
    """Calcola un gender gap a partire da un dizionario token -> probabilita.

    Formula:
        gap = somma(prob_maschili) - somma(prob_femminili)

    Interpretazione:
        gap > 0  -> massa probabilistica piu maschile
        gap < 0  -> massa probabilistica piu femminile
        gap = 0  -> equilibrio perfetto tra i due insiemi
    """

    # Sommiamo le probabilita dei token maschili dichiarati in si_results.json.
    male_p = sum(float(probs.get(token, 0.0)) for token in male_tokens)

    # Sommiamo le probabilita dei token femminili dichiarati in si_results.json.
    female_p = sum(float(probs.get(token, 0.0)) for token in female_tokens)

    # Differenza finale.
    return float(male_p - female_p)


def mean_or_zero(values: list[float]) -> float:
    """Media aritmetica di una lista, con fallback sicuro a 0.0.

    Se la lista e vuota, non alziamo eccezioni e ritorniamo 0.0.
    """

    return float(sum(values) / len(values)) if values else 0.0


def parse_prob_dict(value: object) -> dict:
    """Converte un payload di probabilita in un dict Python.

    Nei CSV L1 le colonne `base_probs` e `finetuned_probs` arrivano come stringhe.
    Esempio:
        "{'man': 0.2, 'woman': 0.8}"

    In altri contesti potremmo gia avere un dict Python pronto.

    Questa funzione supporta entrambi i casi.
    """

    # Caso 1: il valore e gia un dict -> nessuna conversione necessaria.
    if isinstance(value, dict):
        return value

    # Caso 2: il valore e una stringa che rappresenta un dict Python.
    if isinstance(value, str):
        # ast.literal_eval e piu sicuro di eval: accetta solo letterali Python.
        return ast.literal_eval(value)

    # Altri tipi non sono previsti in questo audit.
    raise TypeError(f"Unsupported probability payload type: {type(value)!r}")


def discover_runs(results_dir: Path, requested: Iterable[str] | None) -> List[Path]:
    """Scopre quali cartelle run auditare.

    Comportamento:
    - se `requested` e valorizzato, usa esattamente quelle cartelle;
    - altrimenti ispeziona `results_dir` e prende tutte le sottocartelle che
      contengono `era_summary.json`.
    """

    # Caso 1: l'utente ha passato --runs esplicito.
    if requested:
        # Costruiamo il path completo per ciascun nome run richiesto.
        return [results_dir / run for run in requested]

    # Caso 2: autodiscovery delle run disponibili.
    discovered = []

    # Iteriamo sugli elementi della directory risultati.
    for child in sorted(results_dir.iterdir()):
        # Vogliamo solo directory che contengano un era_summary.json.
        if child.is_dir() and (child / "era_summary.json").exists():
            discovered.append(child)

    return discovered


# =============================================================================
# CUORE DELL'AUDIT SU UNA SINGOLA RUN
# =============================================================================

def audit_run(run_dir: Path, tolerance: float, epsilon: float) -> tuple[List[AuditFinding], AuditRunSummary | None]:
    """Esegue tutti i controlli su una singola cartella run.

    Input:
        run_dir   = path della cartella run, es. era_poc_replication_results/kl_cosine
        tolerance = tolleranza assoluta per confrontare float
        epsilon   = minimo denominatore per ricalcolare alignment score

    Output:
        findings    = lista di AuditFinding (PASS/FAIL atomici)
        run_summary = riepilogo tabellare della run, o None se mancano file essenziali
    """

    # Nome della run, usato in output leggibili.
    run_name = run_dir.name

    # Lista cumulativa di tutti i controlli eseguiti su questa run.
    findings: List[AuditFinding] = []

    # Path dei file principali da controllare.
    summary_path = run_dir / "era_summary.json"
    l1_path = run_dir / "era_l1_behavioral_drift.csv"
    l2_path = run_dir / "era_l2_probabilistic_drift.csv"
    l3_path = run_dir / "era_l3_representational_drift.csv"
    si_path = run_dir / "si_results.json"
    run_config_path = run_dir / "run_config.json"

    # Elenco dei file indispensabili per l'audit.
    required_paths = [summary_path, l1_path, l2_path, l3_path, si_path]

    # Primo controllo: esistenza fisica dei file.
    for path in required_paths:
        findings.append(
            AuditFinding(
                run=run_name,
                check=f"exists:{path.name}",
                ok=path.exists(),
                details=str(path),
            )
        )

    # Se manca almeno un file richiesto, l'audit non puo proseguire.
    if not all(path.exists() for path in required_paths):
        return findings, None

    # Carichiamo i JSON di riepilogo.
    summary = load_json(summary_path)
    si = load_json(si_path)

    # run_config e opzionale: se manca non blocca l'audit.
    run_config = load_json(run_config_path) if run_config_path.exists() else {}

    # Carichiamo i tre CSV principali.
    l1 = pd.read_csv(l1_path)
    l2 = pd.read_csv(l2_path)
    l3 = pd.read_csv(l3_path)

    # Ricalcolo media L1 direttamente dal CSV.
    l1_mean = float(l1["kl_divergence"].mean())

    # Ricalcolo media L2 direttamente dal CSV.
    l2_mean = float(l2["kl_divergence"].mean())

    # Identificazione della colonna delta in L3.
    # Se L3 usa cosine, la colonna e `delta_cosine`.
    # Se L3 usa euclidean, la colonna e `delta_euclidean`.
    if "delta_cosine" in l3.columns:
        l3_delta_col = "delta_cosine"
    elif "delta_euclidean" in l3.columns:
        l3_delta_col = "delta_euclidean"
    else:
        # Se manca entrambe, il file L3 non e nel formato atteso.
        findings.append(
            AuditFinding(
                run=run_name,
                check="l3-delta-column",
                ok=False,
                details=f"Missing delta column in {l3_path.name}",
            )
        )
        return findings, None

    # Ricalcolo della media L3 usando il valore assoluto del delta.
    # Questo replica la logica usata nell'aggregazione ERA.
    l3_mean = float(l3[l3_delta_col].abs().mean())

    # Ricalcolo alignment score usando la formula del framework.
    alignment_recomputed = float(l2_mean / max(l3_mean, epsilon))

    # Elenco dei confronti principali summary <-> CSV ricostruiti.
    comparisons = [
        ("l1_mean_kl", float(summary.get("l1_mean_kl")), l1_mean),
        ("l2_mean_kl", float(summary.get("l2_mean_kl")), l2_mean),
        ("l3_mean_delta", float(summary.get("l3_mean_delta")), l3_mean),
        ("alignment_score", float(summary.get("alignment_score")), alignment_recomputed),
    ]

    # Per ogni confronto, produciamo un AuditFinding.
    for check_name, expected, actual in comparisons:
        findings.append(
            AuditFinding(
                run=run_name,
                check=check_name,
                ok=nearly_equal(expected, actual, tolerance),
                details=f"summary={expected:.15g} recomputed={actual:.15g}",
            )
        )

    # Controllo aggiuntivo: il blocco `era_summary` dentro si_results.json
    # deve essere coerente con era_summary.json.
    if "era_summary" in si:
        si_summary = si["era_summary"]
        si_summary_checks = [
            ("si:era_summary:l1_mean_kl", float(si_summary.get("l1_mean_kl")), float(summary.get("l1_mean_kl"))),
            ("si:era_summary:l2_mean_kl", float(si_summary.get("l2_mean_kl")), float(summary.get("l2_mean_kl"))),
            ("si:era_summary:l3_mean_delta", float(si_summary.get("l3_mean_delta")), float(summary.get("l3_mean_delta"))),
            ("si:era_summary:alignment_score", float(si_summary.get("alignment_score")), float(summary.get("alignment_score"))),
        ]

        for check_name, si_value, summary_value in si_summary_checks:
            findings.append(
                AuditFinding(
                    run=run_name,
                    check=check_name,
                    ok=nearly_equal(si_value, summary_value, tolerance),
                    details=f"si_results={si_value:.15g} era_summary={summary_value:.15g}",
                )
            )

    # Se il summary dichiara il numero di contesti, verifichiamo la lunghezza dei CSV.
    if "num_contexts" in summary:
        expected_contexts = int(summary["num_contexts"])

        findings.append(
            AuditFinding(
                run=run_name,
                check="num_contexts:l1",
                ok=(len(l1) == expected_contexts),
                details=f"summary={expected_contexts} csv={len(l1)}",
            )
        )

        findings.append(
            AuditFinding(
                run=run_name,
                check="num_contexts:l2",
                ok=(len(l2) == expected_contexts),
                details=f"summary={expected_contexts} csv={len(l2)}",
            )
        )

    # =====================================================================
    # RICALCOLO DELLO STEREOTYPE INDEX (SI) DAL CSV L1
    # =====================================================================

    # Insiemi di token maschili/femminili dichiarati in si_results.json.
    male_tokens = set(si["token_sets"]["male_tokens"])
    female_tokens = set(si["token_sets"]["female_tokens"])

    # Numero atteso di contesti leadership/support.
    leadership_expected = int(si["context_sets"]["leadership_n"])
    support_expected = int(si["context_sets"]["support_n"])

    # Liste che conterranno i gap per gruppo e modello.
    base_lead_gaps: list[float] = []
    ft_lead_gaps: list[float] = []
    base_support_gaps: list[float] = []
    ft_support_gaps: list[float] = []

    # Il runner salva prima i contesti leadership e poi quelli support.
    # Se il numero totale di righe coincide, possiamo ricostruire i gruppi per posizione.
    total_expected_contexts = leadership_expected + support_expected
    use_ordered_split = len(l1) == total_expected_contexts

    # Iteriamo sulle righe L1.
    for idx, (_, row) in enumerate(l1.iterrows()):
        # Convertiamo le stringhe del CSV in dict Python reali.
        base_probs = parse_prob_dict(row["base_probs"])
        finetuned_probs = parse_prob_dict(row["finetuned_probs"])

        # Calcolo gap per modello base e fine-tuned.
        base_gap = gap_from_probs(base_probs, male_tokens, female_tokens)
        ft_gap = gap_from_probs(finetuned_probs, male_tokens, female_tokens)

        # Se l'ordine del CSV e quello atteso, separiamo leadership/support per indice.
        if use_ordered_split and idx < leadership_expected:
            base_lead_gaps.append(base_gap)
            ft_lead_gaps.append(ft_gap)
        else:
            base_support_gaps.append(base_gap)
            ft_support_gaps.append(ft_gap)

    # Verifica che i conteggi leadership/support ricostruiti corrispondano ai JSON.
    findings.append(
        AuditFinding(
            run=run_name,
            check="si:context_count:leadership",
            ok=(len(base_lead_gaps) == leadership_expected),
            details=f"si_results={leadership_expected} csv={len(base_lead_gaps)}",
        )
    )

    findings.append(
        AuditFinding(
            run=run_name,
            check="si:context_count:support",
            ok=(len(base_support_gaps) == support_expected),
            details=f"si_results={support_expected} csv={len(base_support_gaps)}",
        )
    )

    # Medie di gruppo.
    base_lead = mean_or_zero(base_lead_gaps)
    ft_lead = mean_or_zero(ft_lead_gaps)
    base_support = mean_or_zero(base_support_gaps)
    ft_support = mean_or_zero(ft_support_gaps)

    # Formula SI:
    #   SI = leadership_bias - support_bias
    base_si = float(base_lead - base_support)
    finetuned_si = float(ft_lead - ft_support)
    delta_si = float(finetuned_si - base_si)

    # Confronto fra SI ricostruito e SI salvato.
    si_checks = [
        ("si:base:leadership_bias", float(si["si"]["base"]["leadership_bias"]), base_lead),
        ("si:base:support_bias", float(si["si"]["base"]["support_bias"]), base_support),
        ("si:base:si", float(si["si"]["base"]["si"]), base_si),
        ("si:finetuned:leadership_bias", float(si["si"]["finetuned"]["leadership_bias"]), ft_lead),
        ("si:finetuned:support_bias", float(si["si"]["finetuned"]["support_bias"]), ft_support),
        ("si:finetuned:si", float(si["si"]["finetuned"]["si"]), finetuned_si),
        ("si:delta_si_ft_minus_base", float(si["si"]["delta_si_ft_minus_base"]), delta_si),
    ]

    for check_name, expected, actual in si_checks:
        findings.append(
            AuditFinding(
                run=run_name,
                check=check_name,
                ok=nearly_equal(expected, actual, tolerance),
                details=f"si_results={expected:.15g} recomputed={actual:.15g}",
            )
        )

    # =====================================================================
    # CONTROLLI DI COERENZA METADATI
    # =====================================================================

    # Se il summary dichiara la colonna delta L3, la confrontiamo col CSV.
    if "l3_delta_column" in summary and summary["l3_delta_column"] is not None:
        findings.append(
            AuditFinding(
                run=run_name,
                check="l3_delta_column",
                ok=(summary["l3_delta_column"] == l3_delta_col),
                details=f"summary={summary['l3_delta_column']} csv={l3_delta_col}",
            )
        )

    # Se esiste run_config, verifichiamo che distribuzione e metrica L3 coincidano col summary.
    if run_config:
        if "distribution_metric" in summary and "distribution_metric" in run_config:
            findings.append(
                AuditFinding(
                    run=run_name,
                    check="distribution_metric",
                    ok=(summary["distribution_metric"] == run_config["distribution_metric"]),
                    details=f"summary={summary['distribution_metric']} run_config={run_config['distribution_metric']}",
                )
            )

        if "l3_metric" in summary and "l3_metric" in run_config:
            findings.append(
                AuditFinding(
                    run=run_name,
                    check="l3_metric",
                    ok=(summary["l3_metric"] == run_config["l3_metric"]),
                    details=f"summary={summary['l3_metric']} run_config={run_config['l3_metric']}",
                )
            )

    # Prepariamo il riepilogo compatto da usare per export.
    distribution_metric = str(summary.get("distribution_metric", run_config.get("distribution_metric", "")))
    l3_metric = str(summary.get("l3_metric", run_config.get("l3_metric", "")))

    run_summary = AuditRunSummary(
        run=run_name,
        distribution_metric=distribution_metric,
        l3_metric=l3_metric,
        l1_mean_kl=l1_mean,
        l2_mean_kl=l2_mean,
        l3_mean_delta=l3_mean,
        alignment_score=alignment_recomputed,
        base_si=base_si,
        finetuned_si=finetuned_si,
        delta_si_ft_minus_base=delta_si,
    )

    return findings, run_summary


# =============================================================================
# EXPORT REPORT TABELLARE
# =============================================================================

def export_report(rows: List[AuditRunSummary], out_path: Path, fmt: str) -> None:
    """Esporta un riepilogo tabellare delle run in CSV o Markdown.

    Input:
        rows     = elenco dei riepiloghi run
        out_path = file di destinazione
        fmt      = 'csv' oppure 'markdown'
    """

    # Costruiamo un DataFrame con una riga per run.
    df = pd.DataFrame([
        {
            "run": row.run,
            "distribution_metric": row.distribution_metric,
            "l3_metric": row.l3_metric,
            "l1_mean_kl": row.l1_mean_kl,
            "l2_mean_kl": row.l2_mean_kl,
            "l3_mean_delta": row.l3_mean_delta,
            "alignment_score": row.alignment_score,
            "base_si": row.base_si,
            "finetuned_si": row.finetuned_si,
            "delta_si_ft_minus_base": row.delta_si_ft_minus_base,
        }
        for row in rows
    ])

    # Creiamo eventuali directory mancanti prima di scrivere il file.
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Caso 1: export CSV.
    if fmt == "csv":
        df.to_csv(out_path, index=False)
        return

    # Caso 2: export Markdown.
    # Non usiamo pandas.to_markdown per evitare dipendenze opzionali come tabulate.
    headers = list(df.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    for _, row in df.iterrows():
        values = [str(row[col]) for col in headers]
        lines.append("| " + " | ".join(values) + " |")

    markdown = "\n".join(lines)
    out_path.write_text(markdown + "\n", encoding="utf-8")


# =============================================================================
# REPORT TESTUALE A TERMINALE
# =============================================================================

def print_report(findings: List[AuditFinding]) -> int:
    """Stampa a terminale tutti i finding e restituisce exit code.

    Exit code:
        0 -> tutti i controlli passano
        1 -> almeno un controllo fallisce
    """

    # current_run serve per stampare l'header della run solo quando cambia.
    current_run = None

    # all_ok accumula lo stato globale dell'audit.
    all_ok = True

    for finding in findings:
        # Se siamo entrati in una nuova run, stampiamo il suo nome.
        if finding.run != current_run:
            current_run = finding.run
            print(f"RUN {current_run}")

        # Status leggibile a terminale.
        status = "PASS" if finding.ok else "FAIL"

        # Riga completa: esito, nome controllo, dettagli.
        print(f"  {status:<4} {finding.check:<20} {finding.details}")

        # Aggiorna lo stato globale.
        all_ok = all_ok and finding.ok

    print()
    print(f"AUDIT_STATUS: {'PASS' if all_ok else 'FAIL'}")

    return 0 if all_ok else 1


# =============================================================================
# PARSER ARGOMENTI CLI
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    """Costruisce il parser degli argomenti da riga di comando."""

    parser = argparse.ArgumentParser(description="Audit ERA result directories against CSV artifacts")

    # --results-dir = cartella che contiene le run.
    parser.add_argument(
        "--results-dir",
        default=str(DEFAULT_RESULTS_DIR),
        help="Directory containing ERA run folders",
    )

    # --runs = elenco opzionale di run specifiche da auditare.
    parser.add_argument(
        "--runs",
        nargs="*",
        default=None,
        help="Specific run directory names to audit",
    )

    # --tolerance = soglia assoluta per confrontare float.
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-9,
        help="Absolute tolerance for numeric comparisons",
    )

    # --epsilon = valore minimo del denominatore quando ricalcoliamo alignment score.
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-12,
        help="Minimum denominator used when recomputing alignment score",
    )

    # --export = path opzionale per scrivere un report tabellare.
    parser.add_argument(
        "--export",
        default=None,
        help="Optional output path for a tabular run summary report",
    )

    # --export-format = csv oppure markdown.
    parser.add_argument(
        "--export-format",
        choices=["csv", "markdown"],
        default="csv",
        help="Format used with --export",
    )

    return parser


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:
    """Entry point principale dello script."""

    # Parse degli argomenti CLI.
    args = build_parser().parse_args()

    # Normalizzazione del path directory risultati.
    results_dir = Path(args.results_dir).resolve()

    # Risoluzione delle run da auditare.
    run_dirs = discover_runs(results_dir, args.runs)

    # Se non troviamo nessuna run, usciamo con errore.
    if not run_dirs:
        print(f"No run directories found in {results_dir}")
        return 1

    # Lista globale dei finding.
    findings: List[AuditFinding] = []

    # Lista globale dei riepiloghi run da esportare eventualmente.
    run_summaries: List[AuditRunSummary] = []

    # Audit di ogni run selezionata.
    for run_dir in run_dirs:
        run_findings, run_summary = audit_run(
            run_dir,
            tolerance=float(args.tolerance),
            epsilon=float(args.epsilon),
        )

        findings.extend(run_findings)

        if run_summary is not None:
            run_summaries.append(run_summary)

    # Export opzionale di un report tabellare.
    if args.export:
        export_report(run_summaries, Path(args.export).resolve(), args.export_format)
        print(f"REPORT_WRITTEN: {Path(args.export).resolve()}")

    # Stampa il report testuale e restituisce exit code finale.
    return print_report(findings)


# Punto di ingresso standard Python.
if __name__ == "__main__":
    raise SystemExit(main())