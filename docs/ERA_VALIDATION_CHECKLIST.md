# ERA Validation Checklist

Questa checklist serve a verificare, calcolo per calcolo, che:

1. la formula dichiarata sia corretta;
2. il codice implementi davvero quella formula;
3. i risultati salvati nei CSV e nei summary siano coerenti.

Usare questa checklist prima di presentare risultati a Pietro, inserirli in una nota tecnica o citarli in un paper.

## Uso Rapido

Se vuoi usare questo file in modo pratico, il flusso minimo e questo:

1. Apri un terminale nella cartella root del repository `ERA-framework`.
2. Assicurati di avere attivato la virtual environment `.venv`.
3. Esegui lo script di audit sui risultati che vuoi controllare.
4. Se il terminale stampa `AUDIT_STATUS: PASS`, i file salvati sono coerenti con i calcoli ricostruiti.
5. Se compare anche un solo `FAIL`, fermati e non usare quei numeri finche il motivo non e chiaro.

In questo repository la cartella da cui lanciare i comandi e:

`C:\Users\ACER\Desktop\ERA-framework`

Quindi i comandi vanno eseguiti da li, non da `docs/`, non da `tools/`, non da un'altra cartella.

## Prima Di Lanciare I Comandi

Verifica queste condizioni minime:

1. Sei nella root del repository `ERA-framework`.
2. La virtual environment `.venv` esiste ed e attiva.
3. Esistono gia le cartelle risultati che vuoi controllare, ad esempio:
	`era_poc_replication_results/kl_cosine`
	`era_poc_replication_results/js_divergence_cosine`
	`era_poc_replication_results/k_divergence_normalized_cosine`

Se queste cartelle non esistono, l'audit non puo verificare nulla: prima bisogna avere i risultati della run.

## Principio Di Lavoro

Per ogni metrica o aggregazione importante, validare sempre tre livelli:

1. Formula: definizione matematica esplicita.
2. Codice: riga o blocco che realizza la formula.
3. Output: numero ricostruibile dai file salvati.

Se uno dei tre livelli non coincide, il risultato non va considerato affidabile.

## L1 Behavioral Drift

Obiettivo: verificare che il drift L1 sia davvero il confronto tra le distribuzioni sui target token.

Controlli:

1. Verificare che i target token siano costruiti prima dell'analisi in [run_era_implicit_gender_experiment_commented.py](run_era_implicit_gender_experiment_commented.py).
2. Verificare che le probabilita del modello base e fine-tuned siano ottenute tramite wrapper in [era/models.py](era/models.py).
3. Verificare che il drift sia calcolato con `compute_distribution_drift(...)` in [era/core_commented.py](era/core_commented.py).
4. Verificare che `era_l1_behavioral_drift.csv` contenga una riga per ogni contesto testato.
5. Verificare che `l1_mean_kl` nel summary sia uguale alla media della colonna `kl_divergence` del CSV L1.

Domande di audit:

1. Le probabilita in `base_probs` e `finetuned_probs` sono normalizzate sul sottoinsieme dei target token?
2. La metrica dichiarata nel summary e nel run config coincide con quella usata nel codice?
3. Il numero di contesti nel summary coincide con il numero di righe del CSV L1?

## L2 Probabilistic Drift

Obiettivo: verificare che L2 confronti davvero i top-k semantici e non l'intero vocabolario grezzo.

Controlli:

1. Verificare che il wrapper produca la distribuzione completa in [era/models.py](era/models.py).
2. Verificare che `_filter_semantic_topk(...)` in [era/core_commented.py](era/core_commented.py) rimuova i token non semantici.
3. Verificare che `_compute_topk_kl(...)` in [era/core_commented.py](era/core_commented.py) lavori sull'unione dei top-k.
4. Verificare che `era_l2_probabilistic_drift.csv` abbia una riga per ogni contesto.
5. Verificare che `l2_mean_kl` nel summary sia uguale alla media della colonna `kl_divergence` del CSV L2.

Domande di audit:

1. Il valore `topk_semantic` dichiarato nel runner e quello effettivamente usato coincidono?
2. La rinormalizzazione delle distribuzioni top-k e coerente con la metrica scelta?
3. La metrica usata in L2 e la stessa riportata in `run_config.json`?

## L3 Representational Drift

Obiettivo: verificare che L3 misuri davvero variazioni geometriche negli embedding.

Controlli:

1. Verificare che i concept token siano quelli dichiarati nel runner.
2. Verificare che gli embedding siano estratti tramite `get_embedding(...)` in [era/models.py](era/models.py).
3. Verificare che la metrica L3 sia quella attesa: `delta_cosine` o `delta_euclidean` in [era/core_commented.py](era/core_commented.py).
4. Verificare che `era_l3_representational_drift.csv` contenga una riga per ogni coppia di concept token.
5. Verificare che `l3_mean_delta` nel summary sia uguale alla media del valore assoluto della colonna delta corretta.

Domande di audit:

1. Il numero di concept token effettivi coincide con quelli che sopravvivono al filtro single-token?
2. La colonna delta usata nel summary e coerente con `l3_metric`?
3. L3 cambia solo se cambiano embeddings o metrica L3, non se cambia solo KL/JS/K in L1-L2?

## Alignment Score

Obiettivo: verificare che l'Alignment Score corrisponda davvero alla formula dichiarata.

Formula:

`alignment_score = l2_mean_kl / max(l3_mean_delta, epsilon)`

Controlli:

1. Verificare la formula in [era/metrics_commented.py](era/metrics_commented.py).
2. Verificare che [era/core_commented.py](era/core_commented.py) passi a `compute_alignment_score(...)` i valori medi giusti.
3. Ricalcolare lo score a partire da `l2_mean_kl` e `l3_mean_delta` salvati nel summary.
4. Verificare che il valore ricostruito coincida con `alignment_score` entro una tolleranza numerica dichiarata.

Domande di audit:

1. Se L3 resta costante e cambia L2, lo score cambia in modo coerente?
2. Se cambia la metrica di drift L1-L2, lo score puo cambiare anche a parita di embeddings?

## Confronto Tra Metriche Di Drift

Quando si confrontano `kl`, `js_divergence` e `k_divergence_normalized`, non ci si aspetta uguaglianza numerica.

Bisogna invece verificare:

1. che ogni run sia internamente coerente con i propri CSV;
2. che la metrica dichiarata nel summary e nel run config sia quella attesa;
3. che L3 resti identica se `l3_metric` non cambia;
4. che le differenze tra L1 e L2 siano spiegabili dal cambio di formula, non da file incoerenti.

## Controlli Minimi Prima Di Condividere Un Risultato

1. Ricalcolare media L1 da `era_l1_behavioral_drift.csv`.
2. Ricalcolare media L2 da `era_l2_probabilistic_drift.csv`.
3. Ricalcolare media assoluta L3 da `era_l3_representational_drift.csv`.
4. Ricalcolare alignment score dal summary o dai CSV.
5. Verificare che `run_config.json` dichiari la metrica corretta.
6. Verificare che il numero di righe e i conteggi nel summary coincidano con i file.

## Procedura Consigliata

1. Eseguire la run.
2. Lanciare uno script di audit automatico sui risultati.
3. Se l'audit passa, usare i numeri nella narrativa.
4. Se l'audit fallisce, bloccare subito l'uso dei risultati e correggere prima il codice o la documentazione.

## Tool Di Audit Consigliato

Script da usare:

1. [tools/audit_era_results.py](tools/audit_era_results.py)

Questo script legge i file salvati dentro `era_poc_replication_results/`, ricostruisce i valori principali dai CSV e controlla che coincidano con:

1. `era_summary.json`
2. `si_results.json`
3. `run_config.json`

In altre parole, non si fida del solo summary: ricalcola i numeri dai file grezzi.

## Comandi Pronti

### 1. Audit Base

Usa questo comando se vuoi solo sapere se tutto e coerente oppure no.

Da eseguire dalla root del repository:

```bash
c:/Users/ACER/Desktop/ERA-framework/.venv/Scripts/python.exe tools/audit_era_results.py --runs kl_cosine js_divergence_cosine k_divergence_normalized_cosine
```

Cosa fa:

1. controlla le tre run `kl_cosine`, `js_divergence_cosine`, `k_divergence_normalized_cosine`
2. verifica i file `era_summary.json`, CSV L1/L2/L3 e `si_results.json`
3. stampa una riga `PASS` o `FAIL` per ogni controllo
4. chiude con `AUDIT_STATUS: PASS` oppure `AUDIT_STATUS: FAIL`

Quando usarlo:

1. prima di riportare un numero in una nota o in una mail
2. dopo aver modificato codice di metriche o aggregazione
3. quando vuoi controllare se i file salvati sono internamente coerenti

### 2. Audit Con Report CSV

Usa questo comando se vuoi produrre una tabella riassuntiva facile da aprire o allegare.

```bash
c:/Users/ACER/Desktop/ERA-framework/.venv/Scripts/python.exe tools/audit_era_results.py --runs kl_cosine js_divergence_cosine k_divergence_normalized_cosine --export docs/era_metric_audit_report.csv --export-format csv
```

Cosa produce:

1. il normale report `PASS`/`FAIL` nel terminale
2. un file CSV in [docs/era_metric_audit_report.csv](docs/era_metric_audit_report.csv)

Quando usarlo:

1. se vuoi confrontare rapidamente le tre metriche in Excel o LibreOffice
2. se vuoi allegare un file tabellare a una nota tecnica

### 3. Audit Con Report Markdown

Usa questo comando se vuoi un file leggibile direttamente in VS Code o da incollare in una documentazione Markdown.

```bash
c:/Users/ACER/Desktop/ERA-framework/.venv/Scripts/python.exe tools/audit_era_results.py --runs kl_cosine js_divergence_cosine k_divergence_normalized_cosine --export docs/era_metric_audit_report.md --export-format markdown
```

Cosa produce:

1. il normale report `PASS`/`FAIL` nel terminale
2. un file Markdown in [docs/era_metric_audit_report.md](docs/era_metric_audit_report.md)

Quando usarlo:

1. se vuoi allegare il risultato direttamente a una documentazione nel repo
2. se vuoi un riepilogo leggibile senza aprire un foglio di calcolo

## Come Leggere L'Output Del Terminale

Esempio mentale:

1. `PASS l1_mean_kl` significa che la media salvata nel summary coincide con la media ricalcolata dal CSV L1.
2. `PASS l2_mean_kl` significa che il CSV L2 conferma il summary.                                                            
3. `PASS l3_mean_delta` significa che il drift rappresentazionale medio e coerente.
4. `PASS alignment_score` significa che il rapporto tra L2 e L3 torna davvero.
5. `PASS si:base:si` o `PASS si:delta_si_ft_minus_base` significa che anche lo Stereotype Index e coerente con i dati L1 salvati.

Se vedi `FAIL`, la regola pratica e:

1. non usare quel numero finche non hai capito la causa;
2. guarda il nome del controllo fallito;
3. confronta il file citato con il blocco formula/codice corrispondente in questa checklist.

## Cosa Non Fa L'Audit

L'audit non dimostra che una formula sia teoricamente la migliore possibile.
Dimostra invece una cosa piu concreta e fondamentale:

1. che il numero riportato nel summary corrisponde davvero ai dati salvati;
2. che i CSV, il SI e i riepiloghi non si contraddicono;
3. che la narrativa numerica parte da file coerenti.

Quindi l'audit risponde a questa domanda:

`stiamo dicendo davvero quello che il codice ha calcolato?`

Non risponde, da solo, a questa altra domanda:

`questa e la scelta metodologica migliore possibile?`

## Procedura Minima Prima Di Mandare Materiale A Pietro

1. Apri terminale in `C:\Users\ACER\Desktop\ERA-framework`.
2. Lancia l'audit base.
3. Controlla che l'ultima riga sia `AUDIT_STATUS: PASS`.
4. Se ti serve allegato, genera anche CSV o Markdown.
5. Solo a quel punto usa i numeri nei documenti.

## Output Consigliati Da Allegare Alle Note Per Pietro

1. [docs/ERA_VALIDATION_CHECKLIST.md](docs/ERA_VALIDATION_CHECKLIST.md)
2. [docs/era_metric_audit_report.csv](docs/era_metric_audit_report.csv)
3. [docs/era_metric_audit_report.md](docs/era_metric_audit_report.md)