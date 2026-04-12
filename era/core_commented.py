"""
Core ERA Analyzer class implementing three-level drift analysis.
"""
# Modulo principale del framework ERA.
# Contiene la classe ERAAnalyzer che orchestra i tre livelli di analisi (L1, L2, L3)
# e la classe ERAResults che raccoglie tutti i risultati in un unico oggetto.

from typing import Dict, List, Optional, Tuple, Any
# Dict       = tipo dizionario Python (es. {"man": 0.3, "woman": 0.2})
# List       = tipo lista Python (es. ["The CEO is", "A nurse is"])
# Optional   = parametro che può essere None oppure del tipo indicato
# Tuple, Any = usati per type hints generici in altri punti del codice

import logging
# logging = libreria standard Python per stampare messaggi di debug/info/warning
# a schermo o su file. Non è print() — è più strutturato e filtrabile.

from dataclasses import dataclass
# dataclass = decoratore Python che genera automaticamente __init__, __repr__ ecc.
# per una classe che contiene solo attributi dati (come un contenitore).

import pandas as pd
# pandas = libreria per tabelle dati (DataFrame).
# Usata per restituire i risultati L1, L2, L3 come tabelle con righe e colonne.

import numpy as np
# numpy = libreria per calcolo numerico vettoriale.
# Usata internamente nelle metriche (np.log, np.dot, np.linalg.norm, ecc.).

try:
    from tqdm import tqdm
    # tqdm = libreria per barre di avanzamento nel terminale.
    # Mostra una progress bar durante i cicli lunghi (es. "L1 Analysis: 40/40").
except ImportError:
    # Se tqdm non è installato, si usa questa funzione sostitutiva
    # che non fa nulla di speciale — restituisce semplicemente l'iterabile.
    def tqdm(iterable, **kwargs):
        return iterable
        # Fallback silenzioso: il ciclo funziona ugualmente, senza progress bar.

from .models import ModelWrapper
# ModelWrapper = classe definita in era/models.py che wrappa un modello HuggingFace.
# Espone i metodi: get_token_probabilities(), get_full_distribution(), get_embedding().
# Il punto iniziale "." indica import relativo (stesso pacchetto era/).

from .metrics_commented import (
    # CATENA DI CHIAMATE: core_commented.py -> metrics_commented.py
    # Importiamo dalla versione commentata di metrics.py in modo che l'intera
    # catena di file commentati sia coerente e auto-contenuta.
    compute_distribution_drift,
    # compute_distribution_drift = dispatcher che smista a KL/JS/K in base al parametro method.
    # Usata in L1 (su target_tokens) e dentro _compute_topk_kl (su top-50 semantici per L2).
    compute_cosine_similarity,
    # compute_cosine_similarity = calcola (a·b)/(||a||·||b||) tra due vettori embedding.
    # Usata in L3 quando l3_metric="cosine".
    compute_euclidean_distance,
    # compute_euclidean_distance = calcola ||a-b|| tra due vettori embedding.
    # Usata in L3 quando l3_metric="euclidean".
    compute_alignment_score,
    # compute_alignment_score = calcola l2_mean_kl / l3_mean_delta.
    # Usata in analyze() dopo aver ottenuto i risultati di L2 e L3.
)

logging.basicConfig(level=logging.INFO)
# Configura il sistema di logging per mostrare messaggi di livello INFO e superiori.
# INFO include messaggi informativi normali (non solo errori).

logger = logging.getLogger(__name__)
# Crea un logger specifico per questo modulo.
# __name__ = "era.core" quando il modulo è importato come parte del pacchetto.
# Permette di identificare la sorgente di ogni messaggio di log.


@dataclass
# Decoratore che trasforma ERAResults in una dataclass:
# genera automaticamente il costruttore __init__ con tutti gli attributi elencati.
class ERAResults:
    """Container for ERA analysis results."""
    # Contenitore che raccoglie tutti i risultati dell'analisi ERA in un unico oggetto.
    # Viene restituito da ERAAnalyzer.analyze() alla fine dell'analisi completa.

    l1_behavioral: pd.DataFrame
    # DataFrame con una riga per ogni contesto testato.
    # Colonne: context (stringa), kl_divergence (float), base_probs (dict), finetuned_probs (dict).
    # kl_divergence = divergenza calcolata sui target_tokens tra base e fine-tuned.

    l2_probabilistic: pd.DataFrame
    # DataFrame con una riga per ogni contesto testato.
    # Colonne: context (stringa), kl_divergence (float), base_topk (dict), finetuned_topk (dict).
    # kl_divergence = divergenza calcolata sui top-50 token semantici tra base e fine-tuned.

    l3_representational: pd.DataFrame
    # DataFrame con una riga per ogni COPPIA di concept_tokens.
    # Se l3_metric="cosine": colonne token_a, token_b, base_cosine, finetuned_cosine, delta_cosine.
    # Se l3_metric="euclidean": colonne token_a, token_b, base_euclidean, finetuned_euclidean, delta_euclidean.
    # Può essere un DataFrame vuoto se concept_tokens non è stato fornito.

    alignment_score: float
    # Numero singolo: l2_mean_kl / l3_mean_delta.
    # Misura il rapporto tra deriva dell'output (L2) e deriva della struttura interna (L3).
    # Valori alti = apprendimento superficiale ("effetto pappagallo").

    summary: Dict[str, Any]
    # Dizionario con statistiche aggregate: alignment_score, l1_mean_kl, l2_mean_kl,
    # l3_mean_delta, std, max, metrica usata, numero di contesti e token.

    def save(self, output_dir: str) -> None:
        """Save all results to directory."""
        # Salva tutti i risultati su disco nella cartella output_dir.
        # Crea la cartella se non esiste.

        import os
        # os = modulo standard per operazioni sul filesystem (cartelle, path, ecc.).

        os.makedirs(output_dir, exist_ok=True)
        # Crea la directory output_dir (e tutte le sottodirectory necessarie).
        # exist_ok=True = non lancia errore se la cartella esiste già.

        self.l1_behavioral.to_csv(f"{output_dir}/era_l1_behavioral_drift.csv", index=False)
        # Salva il DataFrame L1 come file CSV.
        # index=False = non scrive la colonna dell'indice numerico (0, 1, 2, ...) nel file.

        self.l2_probabilistic.to_csv(f"{output_dir}/era_l2_probabilistic_drift.csv", index=False)
        # Salva il DataFrame L2 come file CSV. Stessa logica di L1.

        self.l3_representational.to_csv(f"{output_dir}/era_l3_representational_drift.csv", index=False)
        # Salva il DataFrame L3 come file CSV. Stessa logica di L1.

        import json
        # json = modulo standard Python per serializzare/deserializzare oggetti in formato JSON.

        with open(f"{output_dir}/era_summary.json", "w") as f:
            # Apre (o crea) il file era_summary.json in modalità scrittura ("w").
            # Il blocco with garantisce che il file venga chiuso correttamente anche in caso di errore.
            json.dump(self.summary, f, indent=2)
            # Scrive il dizionario summary nel file JSON.
            # indent=2 = formatta il JSON con indentazione di 2 spazi per leggibilità.

        logger.info(f"Results saved to {output_dir}")
        # Stampa un messaggio informativo che conferma il salvataggio.


class ERAAnalyzer:
    """
    ERA Framework: Evaluation of Representational Alignment

    Analyzes fine-tuned language models at three levels:
    - L1 (Behavioral): What the model says
    - L2 (Probabilistic): How the model decides
    - L3 (Representational): What the model knows
    """
    # Classe principale del framework ERA.
    # Riceve due modelli (base e fine-tuned) e le metriche da usare,
    # poi esegue l'analisi a tre livelli quando si chiama .analyze().

    def __init__(
        self,
        base_model: ModelWrapper,
        # base_model = modello originale prima del fine-tuning, wrappato in ModelWrapper.
        finetuned_model: ModelWrapper,
        # finetuned_model = modello dopo il fine-tuning, stesso tipo di base_model.
        device: str = "cuda",
        # device = dispositivo di calcolo. "cuda" = GPU NVIDIA. "cpu" = processore.
        # Default "cuda" — viene sovrascritto a "cpu" se la GPU non è disponibile.
        distribution_metric: str = "kl",
        # distribution_metric = metrica di divergenza per L1 e L2.
        # Valori possibili: "kl", "js_divergence", "js_distance",
        #                   "k_divergence", "k_divergence_normalized".
        # Viene passata a compute_distribution_drift() come parametro method.
        l3_metric: str = "cosine",
        # l3_metric = metrica per confrontare gli embedding in L3.
        # "cosine" = similarità coseno (misura l'angolo tra i vettori).
        # "euclidean" = distanza euclidea (misura la distanza geometrica).
    ):
        self.base_model = base_model
        # Salva il modello base come attributo dell'istanza per usarlo nei metodi.

        self.finetuned_model = finetuned_model
        # Salva il modello fine-tuned come attributo dell'istanza.

        self.device = device
        # Salva il dispositivo come attributo (non usato direttamente qui,
        # ma disponibile per i metodi interni se necessario).

        self.distribution_metric = distribution_metric.lower().strip()
        # Normalizza la stringa: la converte in minuscolo e rimuove spazi iniziali/finali.
        # Evita errori se l'utente passa "KL" o " js_divergence " invece di "kl".

        self.l3_metric = l3_metric.lower().strip()
        # Stessa normalizzazione per la metrica L3.

        if self.l3_metric not in {"cosine", "euclidean"}:
            raise ValueError(f"Unsupported l3_metric: {l3_metric}")
        # Controllo di validità: se l3_metric non è né "cosine" né "euclidean",
        # lancia un errore subito invece di fallire silenziosamente più avanti.

        logger.info(
            f"ERA Analyzer initialized with device={device}, "
            f"distribution_metric={self.distribution_metric}, l3_metric={self.l3_metric}"
        )
        # Stampa un messaggio di log che conferma la configurazione dell'analizzatore.

    def analyze(
        self,
        test_contexts: List[str],
        # test_contexts = lista di frasi incomplete da dare ai modelli come input.
        # Es: ["The CEO of the company is", "A nurse typically"].
        # Su queste frasi vengono eseguiti L1 e L2.
        target_tokens: List[str],
        # target_tokens = lista di token da osservare in L1.
        # Es: ["man", "woman", "male", "female"].
        # L1 chiede ai modelli le probabilità SOLO su questi token.
        concept_tokens: Optional[List[str]] = None,
        # concept_tokens = lista di token concettuali per L3.
        # Es: ["man", "woman", "leader", "nurse", "CEO", "assistant"].
        # L3 analizza la geometria degli embedding per tutte le coppie di questi token.
        # Optional = può essere None (in quel caso L3 viene saltata).
        topk_semantic: int = 50,
        # topk_semantic = numero di token semantici da tenere per L2.
        # Il vocabolario completo (~50.000 token) viene filtrato ai top-50 per probabilità.
        # Default 50: un compromesso tra completezza e stabilità numerica.
    ) -> ERAResults:
        # Tipo di ritorno: restituisce un oggetto ERAResults con tutti i risultati.

        logger.info("Starting ERA analysis...")
        logger.info(f"  Test contexts: {len(test_contexts)}")
        # Stampa il numero di frasi che verranno analizzate (es. 40).
        logger.info(f"  Target tokens: {len(target_tokens)}")
        # Stampa il numero di token target per L1.

        # ── L1: Behavioral Drift ──────────────────────────────────────────────
        logger.info("Running L1 (Behavioral) analysis...")
        l1_results = self._analyze_l1(test_contexts, target_tokens)
        # Chiama il metodo privato _analyze_l1 con le frasi e i token target.
        # Restituisce un DataFrame: una riga per contesto, con la divergenza
        # calcolata sulle probabilità dei target_tokens (base vs fine-tuned).

        # ── L2: Probabilistic Drift ───────────────────────────────────────────
        logger.info("Running L2 (Probabilistic) analysis...")
        l2_results = self._analyze_l2(test_contexts, topk_semantic)
        # Chiama il metodo privato _analyze_l2 con le frasi e il numero top-k.
        # Restituisce un DataFrame: una riga per contesto, con la divergenza
        # calcolata sui top-50 token semantici dell'intero vocabolario (base vs fine-tuned).

        # ── L3: Representational Drift ────────────────────────────────────────
        if concept_tokens:
            # Esegue L3 solo se concept_tokens è stato fornito (lista non vuota e non None).
            logger.info("Running L3 (Representational) analysis...")
            l3_results = self._analyze_l3(concept_tokens)
            # Chiama il metodo privato _analyze_l3 con i token concettuali.
            # Restituisce un DataFrame: una riga per ogni COPPIA di concept_tokens,
            # con la variazione di similarità coseno (o distanza euclidea) degli embedding.
        else:
            logger.warning("No concept_tokens provided, skipping L3 analysis")
            # Avvisa che L3 è stata saltata — questo produce un Alignment Score non valido.
            l3_results = pd.DataFrame()
            # Crea un DataFrame vuoto come placeholder per l3_results.
            # pd.DataFrame() senza argomenti = tabella vuota, 0 righe e 0 colonne.

        # ── Alignment Score ───────────────────────────────────────────────────
        l2_mean = l2_results["kl_divergence"].mean()
        # Calcola la media della colonna "kl_divergence" del DataFrame L2.
        # .mean() = somma di tutti i valori diviso il numero di valori (pandas).
        # Risultato: un singolo float che rappresenta la deriva media dell'output su tutti i contesti.

        l3_delta_col = self._get_l3_delta_column()
        # Determina il nome della colonna delta da usare in L3.
        # Restituisce "delta_cosine" se l3_metric="cosine", "delta_euclidean" altrimenti.

        l3_mean = l3_results[l3_delta_col].abs().mean() if not l3_results.empty else 0.0
        # Se L3 ha dati: prende la colonna delta, calcola il valore assoluto di ogni elemento
        # (.abs() = |x| per ogni x), poi ne fa la media (.mean()).
        # Se L3 è vuota (DataFrame vuoto): usa 0.0 come default.
        # Il valore assoluto è necessario perché il delta può essere positivo o negativo —
        # ci interessa la quantità di movimento, non la direzione.

        alignment_score = compute_alignment_score(l2_mean, l3_mean)
        # Chiama compute_alignment_score(l2_mean_kl, l3_mean_delta) da metrics.py.
        # Formula: l2_mean / max(l3_mean, epsilon).
        # Misura quante volte L2 è più grande di L3.
        # Valori alti = l'output è cambiato molto più della struttura interna = pappagallo.

        # ── Statistiche di riepilogo ──────────────────────────────────────────
        summary = {
            "alignment_score": float(alignment_score),
            # Alignment Score: L2_mean / L3_mean. Indice di apprendimento superficiale.

            "l1_mean_kl": float(l1_results["kl_divergence"].mean()),
            # Media della divergenza L1 su tutti i contesti.
            # Misura quanto il modello ha cambiato le probabilità sui target_tokens.

            "l1_std_kl": float(l1_results["kl_divergence"].std()),
            # Deviazione standard della divergenza L1.
            # Alta std = il cambiamento è molto variabile tra contesti diversi.

            "l1_max_kl": float(l1_results["kl_divergence"].max()),
            # Valore massimo di divergenza L1 — il contesto più "colpito" dal fine-tuning.

            "l2_mean_kl": float(l2_mean),
            # Media della divergenza L2 (già calcolata sopra, viene solo inclusa nel summary).

            "l2_std_kl": float(l2_results["kl_divergence"].std()),
            # Deviazione standard della divergenza L2.

            "l2_max_kl": float(l2_results["kl_divergence"].max()),
            # Valore massimo di divergenza L2.

            "l3_mean_delta": float(l3_mean) if not l3_results.empty else None,
            # Media del delta assoluto L3. None se L3 non è stata eseguita.

            "distribution_metric": self.distribution_metric,
            # Nome della metrica usata per L1 e L2 (es. "js_divergence").
            # Salvato per poter riprodurre o interpretare i risultati in seguito.

            "l3_metric": self.l3_metric,
            # Nome della metrica usata per L3 (es. "cosine").

            "l3_delta_column": l3_delta_col if not l3_results.empty else None,
            # Nome della colonna delta nel DataFrame L3 (es. "delta_cosine"). None se L3 vuota.

            "num_contexts": len(test_contexts),
            # Numero totale di contesti analizzati in L1 e L2.

            "num_target_tokens": len(target_tokens),
            # Numero di token target usati in L1.
        }

        logger.info(f"Analysis complete. Alignment Score: {alignment_score:.2f}")
        # Stampa il messaggio finale con l'Alignment Score formattato a 2 decimali.

        return ERAResults(
            l1_behavioral=l1_results,
            # Passa il DataFrame L1 all'oggetto risultati.
            l2_probabilistic=l2_results,
            # Passa il DataFrame L2 all'oggetto risultati.
            l3_representational=l3_results,
            # Passa il DataFrame L3 (o DataFrame vuoto se L3 saltata).
            alignment_score=alignment_score,
            # Passa il valore numerico dell'Alignment Score.
            summary=summary,
            # Passa il dizionario con tutte le statistiche aggregate.
        )
        # ERAResults è una dataclass: il costruttore assegna automaticamente
        # ogni argomento all'attributo corrispondente.

    # ══════════════════════════════════════════════════════════════════════════
    # L1 — BEHAVIORAL DRIFT
    # ══════════════════════════════════════════════════════════════════════════

    def _analyze_l1(
        self,
        contexts: List[str],
        # contexts = lista di frasi incomplete (es. ["The CEO is", "A nurse is"]).
        target_tokens: List[str],
        # target_tokens = token da osservare (es. ["man", "woman", "male", "female"]).
    ) -> pd.DataFrame:
        # Restituisce un DataFrame con una riga per ogni contesto.
        """
        Level 1: Behavioral drift - changes in specific token outputs.
        """
        # L1 misura COSA dice il modello: per ogni frase, quanto sono cambiate
        # le probabilità sui token che ci interessano (target_tokens)?

        results = []
        # Lista vuota che raccoglierà un dizionario per ogni contesto elaborato.
        # Alla fine verrà convertita in DataFrame.

        for context in tqdm(contexts, desc="L1 Analysis"):
            # Ciclo su ogni frase del corpus.
            # tqdm mostra una progress bar con etichetta "L1 Analysis".

            base_probs = self.base_model.get_token_probabilities(context, target_tokens)
            # Chiede al modello BASE le probabilità per i target_tokens dato il contesto.
            # Internamente: passa la frase al modello, ottiene la softmax sull'ultimo token,
            # poi estrae solo le probabilità dei token in target_tokens.
            # Risultato: dizionario {token: probabilità}, es. {"man": 0.071, "woman": 0.019}.

            ft_probs = self.finetuned_model.get_token_probabilities(context, target_tokens)
            # Stessa operazione con il modello FINE-TUNED.
            # Risultato: dizionario con le stesse chiavi ma probabilità diverse.
            # Es: {"man": 0.093, "woman": 0.008} — dopo fine-tuning su corpus di bias.

            kl = compute_distribution_drift(
                base_probs,
                # Prima distribuzione: le probabilità del modello base sui target_tokens.
                ft_probs,
                # Seconda distribuzione: le probabilità del modello fine-tuned sugli stessi token.
                method=self.distribution_metric,
                # Metrica da usare: "kl", "js_divergence", ecc. (configurata in __init__).
            )
            # Calcola la divergenza tra le due distribuzioni sui target_tokens.
            # Con KL(base||ft): misura quanto il fine-tuned sorprende il base.
            # Con JS: misura la distanza simmetrica tra le due distribuzioni.
            # Risultato: un singolo float (più alto = più cambiamento).

            results.append({
                "context": context,
                # La frase analizzata (stringa).
                "kl_divergence": kl,
                # Il valore di divergenza calcolato su questa frase.
                "base_probs": base_probs,
                # Il dizionario delle probabilità del modello base (per analisi successiva).
                "finetuned_probs": ft_probs,
                # Il dizionario delle probabilità del modello fine-tuned (per analisi successiva).
            })
            # Aggiunge un dizionario alla lista — diventerà una riga del DataFrame finale.

        return pd.DataFrame(results)
        # Converte la lista di dizionari in un DataFrame pandas.
        # Ogni dizionario diventa una riga; le chiavi del dizionario diventano le colonne.
        # Risultato: tabella con colonne [context, kl_divergence, base_probs, finetuned_probs].

    # ══════════════════════════════════════════════════════════════════════════
    # L2 — PROBABILISTIC DRIFT
    # ══════════════════════════════════════════════════════════════════════════

    def _analyze_l2(
        self,
        contexts: List[str],
        # contexts = stesse frasi usate in L1.
        topk: int = 50,
        # topk = quanti token semantici tenere per il confronto.
        # Default 50: si prendono le 50 parole più probabili secondo ciascun modello.
    ) -> pd.DataFrame:
        # Restituisce un DataFrame con una riga per ogni contesto.
        """
        Level 2: Probabilistic drift - changes in semantic token distributions.
        """
        # L2 misura COME decide il modello: per ogni frase, quanto è cambiata
        # la distribuzione di probabilità sull'intero vocabolario semantico?
        # A differenza di L1, non si limitano i token a quelli scelti manualmente —
        # si guarda l'intero output del modello, filtrato e ridotto ai top-50.

        results = []
        # Lista vuota che raccoglierà un dizionario per ogni contesto elaborato.

        for context in tqdm(contexts, desc="L2 Analysis"):
            # Ciclo su ogni frase, con progress bar "L2 Analysis".

            base_dist = self.base_model.get_full_distribution(context)
            # Chiede al modello BASE la distribuzione completa sull'intero vocabolario.
            # Restituisce un dizionario {token: probabilità} con ~50.000 voci.
            # Somma di tutte le probabilità = 1.0 (softmax completa).

            ft_dist = self.finetuned_model.get_full_distribution(context)
            # Stessa operazione con il modello FINE-TUNED.
            # Stessa struttura: ~50.000 token con le loro probabilità dopo fine-tuning.

            base_semantic = self._filter_semantic_topk(base_dist, topk)
            # Filtra la distribuzione completa del BASE in due passi:
            # 1. Rimuove i token non semantici (punteggiatura, simboli, ecc.)
            # 2. Tiene solo i top-50 per probabilità.
            # Risultato: dizionario con al massimo 50 voci — le parole più probabili.

            ft_semantic = self._filter_semantic_topk(ft_dist, topk)
            # Stessa operazione per il FINE-TUNED.
            # I due insiemi di top-50 possono essere diversi: il fine-tuning può
            # aver promosso token nuovi e degradato quelli che il base preferiva.

            kl = self._compute_topk_kl(base_semantic, ft_semantic)
            # Calcola la divergenza tra i due top-50, gestendo i token presenti
            # in uno ma non nell'altro (vedi _compute_topk_kl per i dettagli).
            # Risultato: un singolo float che misura quanto è cambiata la distribuzione.

            results.append({
                "context": context,
                # La frase analizzata.
                "kl_divergence": kl,
                # La divergenza L2 su questa frase.
                "base_topk": base_semantic,
                # Il dizionario top-50 del modello base (per analisi successiva).
                "finetuned_topk": ft_semantic,
                # Il dizionario top-50 del modello fine-tuned.
            })

        return pd.DataFrame(results)
        # Converte in DataFrame. Colonne: [context, kl_divergence, base_topk, finetuned_topk].

    # ══════════════════════════════════════════════════════════════════════════
    # L3 — REPRESENTATIONAL DRIFT
    # ══════════════════════════════════════════════════════════════════════════

    def _analyze_l3(
        self,
        concept_tokens: List[str],
        # concept_tokens = token concettuali da analizzare.
        # Es: ["man", "woman", "leader", "nurse", "CEO", "assistant"].
        # L3 analizza TUTTE le coppie possibili di questi token.
    ) -> pd.DataFrame:
        # Restituisce un DataFrame con una riga per ogni COPPIA di concept_tokens.
        """
        Level 3: Representational drift - changes in embedding geometry.
        """
        # L3 misura COSA SA il modello internamente: per ogni coppia di concetti,
        # quanto è cambiata la loro vicinanza geometrica nello spazio degli embedding?
        # Non guarda l'output del modello — guarda la struttura interna delle rappresentazioni.

        results = []
        # Lista vuota per raccogliere i risultati coppia per coppia.

        n = len(concept_tokens)
        # Numero totale di token concettuali (es. 10).

        total_pairs = n * (n - 1) // 2
        # Numero di coppie non ordinate: formula combinatoria C(n,2) = n*(n-1)/2.
        # Con 10 token: 10*9/2 = 45 coppie.
        # // = divisione intera (senza decimali).

        with tqdm(total=total_pairs, desc="L3 Analysis") as pbar:
            # Crea una progress bar manuale con il numero totale di iterazioni noto.
            # Usata con "with" per garantire che venga chiusa correttamente alla fine.

            for i in range(n):
                # Ciclo esterno sull'indice del primo token della coppia.

                for j in range(i + 1, n):
                    # Ciclo interno: j parte da i+1 per evitare coppie duplicate.
                    # (man, woman) e (woman, man) sono la stessa coppia — ne prendiamo una.
                    # j > i garantisce che ogni coppia appaia una sola volta.

                    tok_a, tok_b = concept_tokens[i], concept_tokens[j]
                    # Estrae i due token della coppia corrente per nome.
                    # Es: tok_a = "man", tok_b = "leader".

                    base_emb_a = self.base_model.get_embedding(tok_a)
                    # Ottiene il vettore embedding di tok_a dal modello BASE.
                    # L'embedding è la rappresentazione interna del token —
                    # un vettore di centinaia di numeri reali (es. 768 dimensioni per GPT-Neo).
                    # Non è l'output del modello, è il vettore nello strato degli embedding.

                    base_emb_b = self.base_model.get_embedding(tok_b)
                    # Ottiene il vettore embedding di tok_b dal modello BASE.

                    ft_emb_a = self.finetuned_model.get_embedding(tok_a)
                    # Ottiene il vettore embedding di tok_a dal modello FINE-TUNED.
                    # Se il fine-tuning ha modificato il significato interno di tok_a,
                    # questo vettore sarà diverso da base_emb_a.

                    ft_emb_b = self.finetuned_model.get_embedding(tok_b)
                    # Ottiene il vettore embedding di tok_b dal modello FINE-TUNED.

                    if self.l3_metric == "cosine":
                        # Usa la similarità coseno come metrica L3.

                        base_val = compute_cosine_similarity(base_emb_a, base_emb_b)
                        # Calcola cos(tok_a, tok_b) nel modello BASE.
                        # Formula: (a·b) / (‖a‖·‖b‖) = prodotto scalare / prodotto delle norme.
                        # Misura l'angolo tra i due vettori: 1.0 = stessa direzione, 0.0 = ortogonali.
                        # Es: base_val = 0.72 (man e leader sono abbastanza vicini nel modello base).

                        ft_val = compute_cosine_similarity(ft_emb_a, ft_emb_b)
                        # Stessa formula nel modello FINE-TUNED.
                        # Es: ft_val = 0.85 (dopo fine-tuning, man e leader sono più vicini).

                        results.append({
                            "token_a": tok_a,
                            # Nome del primo token della coppia.
                            "token_b": tok_b,
                            # Nome del secondo token della coppia.
                            "base_cosine": base_val,
                            # Similarità coseno tra tok_a e tok_b nel modello base.
                            "finetuned_cosine": ft_val,
                            # Similarità coseno tra tok_a e tok_b nel modello fine-tuned.
                            "delta_cosine": ft_val - base_val,
                            # Variazione di similarità: positivo = i due concetti si sono avvicinati,
                            # negativo = si sono allontanati.
                            # Es: 0.85 - 0.72 = +0.13 (man e leader più vicini dopo fine-tuning).
                        })

                    else:
                        # Usa la distanza euclidea come metrica L3 alternativa.

                        base_val = compute_euclidean_distance(base_emb_a, base_emb_b)
                        # Calcola ‖tok_a − tok_b‖ nel modello BASE.
                        # Formula: radice quadrata della somma dei quadrati delle differenze.
                        # Misura la distanza geometrica nello spazio degli embedding.

                        ft_val = compute_euclidean_distance(ft_emb_a, ft_emb_b)
                        # Stessa formula nel modello FINE-TUNED.

                        results.append({
                            "token_a": tok_a,
                            "token_b": tok_b,
                            "base_euclidean": base_val,
                            # Distanza euclidea tra tok_a e tok_b nel modello base.
                            "finetuned_euclidean": ft_val,
                            # Distanza euclidea tra tok_a e tok_b nel modello fine-tuned.
                            "delta_euclidean": ft_val - base_val,
                            # Variazione di distanza: positivo = i concetti si sono allontanati,
                            # negativo = si sono avvicinati.
                            # Nota: con coseno, avvicinamento = delta positivo.
                            #       con euclidea, avvicinamento = delta negativo.
                            #       I segni sono opposti per le due metriche.
                        })

                    pbar.update(1)
                    # Avanza la progress bar di 1 step (una coppia elaborata).

        return pd.DataFrame(results)
        # Converte la lista di dizionari in DataFrame.
        # Se l3_metric="cosine": colonne [token_a, token_b, base_cosine, finetuned_cosine, delta_cosine].
        # Se l3_metric="euclidean": colonne [token_a, token_b, base_euclidean, finetuned_euclidean, delta_euclidean].

    # ══════════════════════════════════════════════════════════════════════════
    # FUNZIONI DI SUPPORTO PER L2
    # ══════════════════════════════════════════════════════════════════════════

    def _filter_semantic_topk(
        self,
        distribution: Dict[str, float],
        # distribution = dizionario {token: probabilità} con ~50.000 voci (vocabolario completo).
        k: int,
        # k = numero massimo di token da tenere dopo il filtraggio.
    ) -> Dict[str, float]:
        # Restituisce un dizionario con al massimo k voci — i top-k token semantici.
        """Filter to semantic tokens and return top-k."""
        # Filtra la distribuzione in due passi: prima rimuove i non-semantici,
        # poi taglia ai top-k per probabilità.

        semantic = {
            token: prob
            for token, prob in distribution.items()
            # Itera su ogni coppia (token, probabilità) del vocabolario completo.
            if self._is_semantic(token)
            # Tiene solo i token che passano il filtro semantico (vedi _is_semantic).
        }
        # Risultato: dizionario filtrato — rimangono solo token con caratteri alfanumerici
        # in proporzione sufficiente (punteggiatura e simboli esclusi).

        sorted_items = sorted(semantic.items(), key=lambda x: x[1], reverse=True)
        # Ordina le coppie (token, probabilità) per probabilità DECRESCENTE.
        # key=lambda x: x[1] = ordina per il secondo elemento della coppia (la probabilità).
        # reverse=True = dal più alto al più basso.
        # Risultato: lista ordinata, il token più probabile è in posizione [0].

        return dict(sorted_items[:k])
        # Taglia ai primi k elementi e riconverte in dizionario.
        # sorted_items[:k] = slicing Python — prende solo i primi k elementi della lista.
        # dict(...) = riconverte la lista di coppie (token, prob) in dizionario.

    def _is_semantic(self, token: str) -> bool:
        # Restituisce True se il token è semantico, False se va scartato.
        """Check if token is semantic (not punctuation/special char)."""
        # Decide se un token rappresenta una parola reale (semantica)
        # o è punteggiatura/simbolo/carattere speciale da ignorare.

        if not token or len(token) == 0:
            return False
            # Se il token è stringa vuota o None, scarta. Non è una parola.

        if len(token) == 1 and not token.isalnum():
            return False
            # Se il token è un singolo carattere NON alfanumerico (es. ",", ".", "!", "#"),
            # scarta. isalnum() = True se il carattere è lettera o cifra.

        alpha_ratio = sum(c.isalnum() for c in token) / len(token)
        # Calcola la proporzione di caratteri alfanumerici nel token.
        # sum(c.isalnum() for c in token) = conta quanti caratteri sono alfanumerici.
        # / len(token) = divide per la lunghezza totale del token.
        # Es: "hello" → 5/5 = 1.0. "hello!" → 5/6 = 0.833. "##" → 0/2 = 0.0.

        return alpha_ratio > 0.5
        # Tieni il token solo se più della metà dei suoi caratteri è alfanumerica.
        # Soglia 0.5: permette token come "can't" (4/5 = 0.8) ma scarta "##!@" (0/4 = 0.0).

    def _compute_topk_kl(
        self,
        p_dist: Dict[str, float],
        # p_dist = top-k del modello BASE per un contesto (dizionario token→probabilità).
        q_dist: Dict[str, float],
        # q_dist = top-k del modello FINE-TUNED per lo stesso contesto.
    ) -> float:
        # Restituisce un singolo float: la divergenza tra le due distribuzioni top-k.
        """Compute configured distribution drift over union of top-k tokens."""
        # Calcola la divergenza sulla UNIONE dei due top-k, gestendo i token
        # che compaiono in uno ma non nell'altro.

        union_tokens = set(p_dist.keys()) | set(q_dist.keys())
        # Costruisce l'insieme unione dei token presenti in almeno una delle due distribuzioni.
        # set(p_dist.keys()) = insieme dei token nel top-k del base.
        # set(q_dist.keys()) = insieme dei token nel top-k del fine-tuned.
        # | = operatore unione insiemistica.
        # Risultato: fino a 2*k token distinti (se i due top-k non si sovrappongono affatto).

        if not union_tokens:
            return 0.0
            # Caso degenere: se entrambe le distribuzioni sono vuote, divergenza = 0.

        p_probs = {t: p_dist.get(t, 0.0) for t in union_tokens}
        # Per ogni token dell'unione, cerca la sua probabilità nel top-k del BASE.
        # .get(t, 0.0) = se il token t non è nel dizionario p_dist, restituisce 0.0.
        # Token nel top-k del fine-tuned ma NON del base → probabilità 0.0 nel base.
        # Questo è una APPROSSIMAZIONE: il token potrebbe avere probabilità piccola
        # ma non zero nel base — non ce l'abbiamo perché è fuori dal top-k.

        q_probs = {t: q_dist.get(t, 0.0) for t in union_tokens}
        # Stessa operazione per il FINE-TUNED.
        # Token nel top-k del base ma NON del fine-tuned → probabilità 0.0 nel fine-tuned.
        # Un token che il base considerava tra i 50 più probabili ma che il fine-tuned
        # ha declassato fuori dai top-50 riceve probabilità 0.0.

        p_sum = sum(p_probs.values()) or 1e-12
        # Somma tutte le probabilità del base nell'unione.
        # Questa somma è < 1.0 perché abbiamo tagliato il vocabolario a top-k.
        # "or 1e-12" = se la somma è 0 (caso degenere), usa 1e-12 per evitare divisione per zero.

        q_sum = sum(q_probs.values()) or 1e-12
        # Stessa somma per il fine-tuned.

        p_norm = {t: p / p_sum for t, p in p_probs.items()}
        # Rinormalizza le probabilità del base dividendo per p_sum.
        # Dopo questo, la somma di p_norm = 1.0 esatta.
        # Necessario perché compute_distribution_drift richiede distribuzioni di probabilità proprie.

        q_norm = {t: q / q_sum for t, q in q_probs.items()}
        # Stessa rinormalizzazione per il fine-tuned. Somma di q_norm = 1.0.

        return compute_distribution_drift(
            p_norm,
            # Distribuzione rinormalizzata del modello BASE sull'unione dei top-k.
            q_norm,
            # Distribuzione rinormalizzata del modello FINE-TUNED sull'unione dei top-k.
            method=self.distribution_metric,
            # Metrica configurata in __init__: "kl", "js_divergence", ecc.
            # Con KL(base||ft): misura quanto il fine-tuned sorprende il base.
            # Con JS: misura la distanza simmetrica tra le due distribuzioni.
        )
        # Restituisce il valore di divergenza per questo contesto.
        # Questo valore finisce nella colonna "kl_divergence" del DataFrame L2.

    def _get_l3_delta_column(self) -> str:
        # Restituisce il nome della colonna delta da usare nel DataFrame L3.
        if self.l3_metric == "cosine":
            return "delta_cosine"
            # Se la metrica è cosine, il delta è nella colonna "delta_cosine".
        return "delta_euclidean"
        # Altrimenti è nella colonna "delta_euclidean".
        # Usato in analyze() per calcolare l3_mean = media del valore assoluto dei delta.
