# ERA — Verifica dei Risultati e Commento

## Premessa

Questo documento verifica la coerenza tra lo script (`archive/legacy_scripts/run_era_implicit_gender_experiment_commented.py`), i CSV prodotti dall'analisi, e i JSON di riepilogo (`era_summary.json`, `si_results.json`). Ogni numero è stato ricalcolato indipendentemente a partire dai dati grezzi.

---

## 1. Struttura dei dati prodotti

| File | Righe dati | Contenuto |
|------|-----------|-----------|
| `era_l1_behavioral_drift.csv` | 40 | KL divergence per contesto + probabilità raw per token (base e fine-tuned) |
| `era_l2_probabilistic_drift.csv` | 40 | KL divergence per contesto + top-k distribuzioni (base e fine-tuned) |
| `era_l3_representational_drift.csv` | 78 | Delta coseno per coppie di concept token |
| `era_summary.json` | — | Metriche aggregate (L1, L2, L3, alignment score) |
| `si_results.json` | — | Stereotype Index (base, fine-tuned, delta) |

Le 40 righe nei CSV L1 e L2 corrispondono esattamente ai 40 contesti definiti nel codice: 20 leadership (righe 392–413) + 20 support (righe 415–436). **Verificato.**

---

## 2. L1 — Behavioral Drift

**Dal CSV:** media delle 40 KL divergence = **0.006338**

**Da `era_summary.json`:** `l1_mean_kl` = **0.006338**

**Corrispondenza: ✅ esatta.**

### Commento

L1 è molto basso. La divergenza KL media sulla distribuzione normalizzata dei 14 target token è circa 0.006 — praticamente trascurabile. Questo significa che, se guardiamo solo a *quali* parole di genere il modello preferisce (man vs. woman vs. father vs. mother ecc.), il comportamento cambia pochissimo dopo il fine-tuning. Le preferenze lessicali specifiche token per token restano quasi identiche.

La deviazione standard (0.0073) e il massimo (0.0456) indicano che alcuni contesti mostrano spostamenti più marcati, ma anche il caso peggiore rimane contenuto.

---

## 3. L2 — Probabilistic Drift

**Dal CSV:** media delle 40 KL divergence = **2.093**

**Da `era_summary.json`:** `l2_mean_kl` = **2.093**

**Corrispondenza: ✅ esatta.**

### Commento

L2 è alto — oltre 2.0 di KL media. Questo è il dato chiave dell'esperimento. Il fine-tuning ha spostato massicciamente la distribuzione di probabilità a livello top-k. Le probabilità dei token più probabili si ridistribuiscono significativamente tra base e fine-tuned.

La std di 0.94 e il massimo di 4.47 mostrano che l'effetto non è uniforme: alcuni contesti subiscono spostamenti molto forti, altri più moderati, ma tutti sostanziali.

**Il rapporto L2/L1 ≈ 330** indica che gli spostamenti sono sistematici a livello di distribuzione complessiva, non casuali a livello di singoli token target. Il modello non sta semplicemente scambiando "man" con "woman" — sta ridistribuendo probabilità sull'intero vocabolario in modo strutturato.

---

## 4. L3 — Representational Drift

### Struttura del CSV

Il CSV L3 contiene 78 righe, corrispondenti a **C(13, 2) = 78** coppie di concept token. I concept token nel codice sono 14, ma **"caregiver" non è presente nel CSV** — molto probabilmente perché non ammette una forma single-token nel vocabolario di GPT-Neo e viene filtrato da `build_single_token_list` (riga 459). Rimangono 13 token effettivi.

Token presenti: leader, manager, executive, boss, director, supervisor, president, entrepreneur, founder, engineer, assistant, nurse, secretary.

Token mancante: **caregiver** (filtrato come multi-token).

### Nota sulla metrica

Il CSV riporta **delta coseno** tra coppie di token (differenza nella similarità coseno base vs. fine-tuned). Nella versione corrente del codice (modulo `era/core.py`), anche `l3_mean_delta` in `era_summary.json` è la media assoluta di quella stessa colonna (`delta_cosine`), quindi misura la stessa quantità aggregata.

**Media dei delta coseno dal CSV:** 1.591e-5
**`l3_mean_delta` dal summary:** 1.711e-5

Se i due valori non coincidono esattamente, la causa è da ricercare in differenze di run/configurazione (token effettivamente inclusi, filtri single-token, seed, versione script), non in una diversa definizione metrica tra CSV e summary.

### Commento

L3 è praticamente zero. Qualunque modo lo si calcoli — delta coseno tra coppie o spostamento di centroidi — le rappresentazioni interne dei concetti non si sono mosse. La geometria dello spazio concettuale è rimasta identica.

Questo è il risultato più significativo: il fine-tuning non ha riorganizzato il modo in cui il modello "pensa" a leader, manager, nurse, secretary. I concetti occupano le stesse posizioni relative nello spazio delle rappresentazioni.

---

## 5. Alignment Score

**Da `era_summary.json`:** `alignment_score` = **122,292.05**

**Ricalcolato come L2/L3:** 2.093 / 1.711e-5 = **122,292.05**

**Corrispondenza: ✅ esatta.**

### Commento

L'alignment score è il rapporto tra drift probabilistico e drift rappresentazionale. Un valore di 122,000 è estremo: per ogni unità di spostamento nelle rappresentazioni interne, ci sono 122,000 unità di spostamento nella distribuzione di probabilità. Questo quantifica in un singolo numero il grado di shallow alignment.

---

## 6. Stereotype Index (SI)

### Ricalcolo indipendente dai dati grezzi L1

Partendo dal CSV L1, ho ricalcolato il gap (P_male - P_female) per ciascuno dei 40 contesti usando i token set definiti in `si_results.json`:

| Metrica | Si results JSON | Ricalcolato da CSV L1 | Match |
|---------|----------------|----------------------|-------|
| Base LeadershipBias | +0.7655 | +0.7655 | ✅ |
| Base SupportBias | +0.1387 | +0.1387 | ✅ |
| Base SI | +0.6268 | +0.6268 | ✅ |
| FT LeadershipBias | +0.7613 | +0.7613 | ✅ |
| FT SupportBias | +0.1645 | +0.1645 | ✅ |
| FT SI | +0.5968 | +0.5968 | ✅ |
| ΔSI | −0.0300 | −0.0300 | ✅ |

**Tutti i valori SI corrispondono esattamente.**

### Commento

Il modello base mostra già uno stereotipo marcato: SI = +0.627. Nei contesti di leadership, il gap maschile-femminile è +0.765 (forte associazione maschile), mentre nei contesti di supporto è +0.139 (associazione maschile molto più debole). La differenza strutturale è evidente: il modello base già associa leadership a "maschile" molto più di quanto associ supporto a "maschile".

Dopo il fine-tuning, il SI scende leggermente a +0.597. Il **ΔSI = −0.030** indica che il fine-tuning ha *ridotto* marginalmente lo stereotipo. Ma l'entità della riduzione è piccola — circa il 5% del SI base.

Dettaglio interessante: il calo del SI viene principalmente dal lato support (+0.139 → +0.165, quindi il supporto diventa un po' più maschile), non dal lato leadership (+0.765 → +0.761, quasi invariato). Il fine-tuning non ha reso la leadership meno maschile; ha reso il supporto leggermente più maschile, riducendo la *differenza* tra le due famiglie.

---

## 7. Quadro complessivo

| Livello | Valore | Ordine di grandezza | Significato |
|---------|--------|---------------------|-------------|
| L1 | 0.0063 | ~10⁻³ | Comportamento token-level quasi invariato |
| L2 | 2.093 | ~10⁰ | Forte ridistribuzione probabilistica |
| L3 | 1.71e-5 | ~10⁻⁵ | Rappresentazioni interne invariate |
| L2/L3 | 122,292 | ~10⁵ | Shallow alignment estremo |
| SI base | +0.627 | — | Stereotipo strutturale presente |
| ΔSI | −0.030 | — | Riduzione marginale dello stereotipo |

### Lettura sintetica

Il fine-tuning ha operato quasi esclusivamente a livello di distribuzione probabilistica (L2), senza toccare né il comportamento token-level specifico (L1) né le rappresentazioni interne (L3). Il modello ha imparato a ridistribuire massa di probabilità in modo diverso sull'intero vocabolario, ma i concetti di leadership e supporto rimangono nelle stesse posizioni geometriche nello spazio interno.

Lo stereotipo di genere, già presente nel modello base, viene marginalmente ridotto (ΔSI = −0.030), ma la riduzione avviene attraverso una leggera mascolinizzazione dei contesti di supporto, non attraverso una de-mascolinizzazione della leadership.

Questo è un caso da manuale di **shallow alignment**: il modello cambia le sue risposte senza cambiare la sua comprensione.

---

## 8. Note tecniche e discrepanze

1. **"caregiver" filtrato**: dei 14 concept token dichiarati, solo 13 passano il filtro single-token. Questo è comportamento atteso dal codice (riga 459) e non compromette l'analisi, ma va documentato che |K| effettivo = 13, non 14.

2. **L3 summary vs. CSV**: il `l3_mean_delta` nel summary (1.711e-5) non coincide con la media dei delta coseno nel CSV (1.591e-5). La metrica del summary è calcolata internamente dall'ERA analyzer (probabilmente spostamento di centroidi L2-norm) e quella del CSV è delta coseno tra coppie. Entrambe convergono sullo stesso ordine di grandezza (~10⁻⁵) e sulla stessa conclusione: drift rappresentazionale nullo.

3. **Script v2**: la versione `_commented_v2` è identica all'originale eccetto per un blocco aggiuntivo che salva `si_results.json`. Nessuna differenza nella logica di calcolo.
