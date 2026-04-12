# Appunti miei per spiegare i risultati a Pietro

Questi sono appunti pratici, scritti in modo semplice, per raccontare cosa c'e nei file risultati e come leggerli senza perdersi nei tecnicismi.

## 1) Mappa veloce dei file (cosa guardare e perche)

- `era_poc_replication_results/era_summary.json`
  - Questo e il file da aprire per primo.
  - E la sintesi finale del run: L1, L2, L3 e alignment score.
  - Se devo spiegare "in 30 secondi" come e andata, parto da qui.

- `era_poc_replication_results/si_results.json`
  - Qui c'e lo Stereotype Index (SI): base vs finetuned.
  - Serve per capire se, alla fine, l'indice stereotipico e salito o sceso.

- `era_poc_replication_results/era_l1_behavioral_drift.csv`
  - Qui vedo il livello comportamentale (L1), prompt per prompt.
  - Mi fa capire in quali contesti il modello cambia davvero nel modo in cui usa token di genere.

- `era_poc_replication_results/era_l2_probabilistic_drift.csv`
  - Qui c'e il livello probabilistico (L2), sempre con dettaglio per contesto.
  - Confronta le distribuzioni top-k token tra base e finetuned.
  - In pratica: dove e quanto si spostano le probabilita di output.

- `era_poc_replication_results/era_l3_representational_drift.csv`
  - Questo e il livello rappresentazionale (L3).
  - Guarda le variazioni di similarita coseno tra coppie di concetti.
  - In parole semplici: quanto si e mossa la struttura semantica interna del modello.

- `era_poc_replication_results/kl_cosine/*`
  - Copia completa dei risultati del run con metrica KL + cosine.
  - Utile quando voglio ricostruire esattamente quel run (anche via `run_config.json`).

- `era_poc_replication_results/js_divergence_cosine/*`
  - Stessa logica, ma con JS-divergence al posto di KL (L1/L2).
  - Serve per confronto metodologico e robustezza.

- `era_poc_replication_results/comparisons/summary_comparison_kl_vs_js_divergence_cosine.json`
  - Confronto diretto numerico tra i due setup metrici.
  - Mi dice subito cosa cambia nei valori passando da KL a JS.

- `era_poc_replication_results/comparisons/comparison_notes_kl_vs_js_divergence_cosine.md`
  - Nota discorsiva di interpretazione del confronto.
  - Utile quando devo spiegare "ok, i numeri cambiano, ma il senso cambia oppure no?".

## 2) Lettura semplice dei risultati (quello che direi a voce)

Prendendo come riferimento il run principale in `era_poc_replication_results/era_summary.json`:

- L1 medio e basso (`l1_mean_kl = 0.0048`).
  - Quindi sul piano "comportamentale stretto" il drift medio non e enorme.

- L2 medio e alto (`l2_mean_kl = 2.0371`).
  - Qui invece il modello mostra uno spostamento forte nelle distribuzioni probabilistiche.
  - Cioe: cambia parecchio "come pesa" i token nei vari contesti.

- L3 e praticamente nullo (`l3_mean_delta = 0.000017`).
  - Le relazioni rappresentazionali interne (embedding-level) si muovono pochissimo.

- Alignment score molto alto (`119026`).
  - Questo e il punto chiave: tanto movimento in superficie (L2), quasi niente in profondita (L3).
  - Traduzione pratica: effetto soprattutto probabilistico/comportamentale, non vera ristrutturazione interna profonda.

Nel file `era_poc_replication_results/si_results.json`:

- SI base: `0.6268`
- SI finetuned: `0.5968`
- Delta: `-0.0300`

Quindi c'e un miglioramento lieve (calo dell'indice), ma non una correzione radicale.

## 3) Come lo direi a Pietro in modo rapido

"I file sono organizzati bene: c'e un file di sintesi (`era_summary.json`), un file indice stereotipi (`si_results.json`) e tre CSV di dettaglio per i tre livelli ERA (L1-L2-L3). Il risultato importante e che il modello cambia molto a livello di probabilita di output (L2), ma quasi zero a livello rappresentazionale interno (L3). Quindi il fine-tuning sembra aver prodotto soprattutto un effetto di superficie, mentre la struttura semantica profonda resta quasi invariata. Lo SI migliora un po', ma non abbastanza da dire che il bias sia risolto." 

## 4) Nota operativa

- La cartella `era_racial_bias_results/` al momento e vuota.
- Quindi questi commenti riguardano il blocco PoC gender in `era_poc_replication_results/`.
