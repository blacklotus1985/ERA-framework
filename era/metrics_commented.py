"""
metrics_commented.py
====================
Versione completamente commentata di metrics.py.

Questo file contiene ESATTAMENTE lo stesso codice di metrics.py,
con commenti riga per riga che spiegano ogni operazione matematica
e ogni scelta implementativa.

Funzioni esportate:
    - compute_kl_divergence          : KL(P||Q) divergenza di Kullback-Leibler
    - compute_js_divergence          : JS(P,Q)  divergenza di Jensen-Shannon
    - compute_k_divergence           : K(P,Q)   K-divergenza di Lin 1991 (log naturale)
    - compute_k_divergence_normalized: K(P,Q)/ln(2) -> K normalizzata in [0,1]
    - compute_js_distance            : sqrt(JS) distanza di Jensen-Shannon
    - compute_distribution_drift     : dispatcher che seleziona la metrica per L1/L2
    - compute_cosine_similarity      : similarita coseno tra due vettori embedding
    - compute_euclidean_distance     : distanza euclidea tra due vettori embedding
    - compute_alignment_score        : ERA Alignment Score = L2_KL / L3_delta
    - interpret_alignment_score      : etichetta qualitativa per l'alignment score
    - compute_statistical_significance: test statistico t-test / Mann-Whitney U

Catena di chiamate:
    run_era_implicit_gender_experiment_commented.py
        -> era/core_commented.py
            -> era/metrics_commented.py  <- SEI QUI
"""

# ---------------------------------------------------------------------------
# Import delle librerie
# ---------------------------------------------------------------------------

from typing import Dict, Union   # Dict: dizionario con tipi annotati; Union: tipo OR tra due tipi
import numpy as np               # libreria NumPy per calcoli vettoriali e matematica numerica

# Tentativo di importare PyTorch (opzionale: non tutti hanno GPU/PyTorch installato)
try:
    import torch                 # libreria deep-learning; serve solo per convertire i tensor in numpy
    TORCH_AVAILABLE = True       # flag globale: True se torch e installato
except ImportError:
    TORCH_AVAILABLE = False      # flag globale: False se torch non e disponibile


# ===========================================================================
# FUNZIONE 1: compute_kl_divergence
# Calcola la divergenza di Kullback-Leibler KL(P || Q)
# ===========================================================================

def compute_kl_divergence(
    p_dist: Dict[str, float],   # distribuzione P: dizionario {token: probabilita}
    q_dist: Dict[str, float],   # distribuzione Q: dizionario {token: probabilita}
    epsilon: float = 1e-12,     # valore piccolo per evitare log(0) -- default 10^-12
) -> float:                     # restituisce un float (non-negativo)
    """
    Calcola la divergenza KL: KL(P || Q) = sum P(x) * log(P(x) / Q(x))

    FORMULA MATEMATICA:
        KL(P||Q) = sum_x  P(x) * ln(P(x) / Q(x))

    PROPRIETA:
        - Asimmetrica: KL(P||Q) != KL(Q||P) in generale
        - Non-negativa: KL(P||Q) >= 0 sempre (teorema di Gibbs)
        - Zero se e solo se P = Q
        - Illimitata superiormente (puo divergere a +inf)
        - Logaritmo NATURALE (np.log = ln), non log2

    INTERPRETAZIONE per ERA:
        P = modello BASE, Q = modello FINE-TUNED
        KL misura quanta informazione si perde se usiamo Q per codificare P.
        Valori alti -> il fine-tuning ha cambiato molto la distribuzione.

    Args:
        p_dist: Distribuzione P (modello base)
        q_dist: Distribuzione Q (modello fine-tuned)
        epsilon: Valore piccolo per evitare log(0) e divisioni per zero

    Returns:
        Valore KL >= 0

    Example:
        >>> p = {"man": 0.6, "woman": 0.4}
        >>> q = {"man": 0.5, "woman": 0.5}
        >>> kl = compute_kl_divergence(p, q)
        >>> print(f"KL divergence: {kl:.4f}")
    """
    # Guarda di non calcolare su distribuzioni vuote
    if not p_dist or not q_dist:
        return 0.0  # nessun dato -> nessuna divergenza

    # Costruisce l'insieme UNIONE di tutti i token che compaiono in P o Q
    # (es: se P ha "man","woman" e Q ha "man","person" -> all_tokens = {"man","woman","person"})
    all_tokens = set(p_dist.keys()) | set(q_dist.keys())

    # Accumulatore della sommatoria
    kl = 0.0

    # Itera su ogni token nell'unione
    for token in all_tokens:
        # Preleva la probabilita di questo token in P (0.0 se assente)
        p = p_dist.get(token, 0.0)
        # Preleva la probabilita di questo token in Q (0.0 se assente)
        q = q_dist.get(token, 0.0)

        # Protezione numerica: sostituisce 0 con epsilon per entrambe le distribuzioni
        # Questo evita log(0) = -inf e 0/0 = NaN
        p = max(p, epsilon)   # P(x) >= epsilon
        q = max(q, epsilon)   # Q(x) >= epsilon

        # Aggiunge il contributo di questo token: P(x) * ln(P(x)/Q(x))
        # np.log = logaritmo NATURALE (base e ~= 2.718)
        kl += p * np.log(p / q)

    # max(kl, 0.0): garantisce non-negativita nonostante eventuali errori float
    return max(kl, 0.0)


# ===========================================================================
# FUNZIONE 2: compute_js_divergence
# Calcola la divergenza di Jensen-Shannon JS(P, Q)
# ===========================================================================

def compute_js_divergence(
    p_dist: Dict[str, float],   # distribuzione P
    q_dist: Dict[str, float],   # distribuzione Q
    epsilon: float = 1e-12,     # protezione numerica
) -> float:
    """
    Calcola la divergenza di Jensen-Shannon.

    FORMULA MATEMATICA:
        M = 0.5 * (P + Q)                    <- distribuzione punto-medio
        JS(P,Q) = 0.5*KL(P||M) + 0.5*KL(Q||M)

    PROPRIETA (rispetto a KL):
        - SIMMETRICA: JS(P,Q) = JS(Q,P)
        - Limitata: 0 <= JS(P,Q) <= ln(2) ~= 0.693 (con logaritmo naturale)
        - Robusta: non diverge anche se P e Q non si sovrappongono

    DIFFERENZA con KL:
        KL confronta P direttamente con Q -> puo esplodere se Q(x)=0 dove P(x)>0
        JS confronta entrambe con M -> M contiene sempre la media, quindi e sempre > 0

    Args:
        p_dist: Distribuzione P
        q_dist: Distribuzione Q
        epsilon: Protezione numerica

    Returns:
        JS divergence in [0, ln(2)]
    """
    # Distribuzione vuota -> nessuna divergenza
    if not p_dist or not q_dist:
        return 0.0

    # Costruisce l'unione di tutti i token
    all_tokens = set(p_dist.keys()) | set(q_dist.keys())

    # Estrae le probabilita di P e Q per ogni token nell'unione
    # max(..., 0.0): garantisce che non ci siano probabilita negative (errori float)
    p_probs = {t: max(p_dist.get(t, 0.0), 0.0) for t in all_tokens}
    q_probs = {t: max(q_dist.get(t, 0.0), 0.0) for t in all_tokens}

    # Somma di tutte le probabilita (dovrebbe essere ~=1 se gia normalizzate)
    p_sum = sum(p_probs.values()) or epsilon   # "or epsilon": evita divisione per 0
    q_sum = sum(q_probs.values()) or epsilon

    # Rinormalizza: assicura che la somma faccia esattamente 1
    p_norm = {t: p / p_sum for t, p in p_probs.items()}
    q_norm = {t: q / q_sum for t, q in q_probs.items()}

    # Calcola la distribuzione punto-medio M = 0.5*(P+Q)
    # Per ogni token: M(x) = 0.5*P_norm(x) + 0.5*Q_norm(x)
    m_dist = {t: 0.5 * (p_norm[t] + q_norm[t]) for t in all_tokens}

    # Calcola KL(P||M) -- quanto P diverge dal punto medio
    kl_pm = compute_kl_divergence(p_norm, m_dist, epsilon=epsilon)
    # Calcola KL(Q||M) -- quanto Q diverge dal punto medio
    kl_qm = compute_kl_divergence(q_norm, m_dist, epsilon=epsilon)

    # JS = media aritmetica dei due KL -> simmetria garantita
    return float(0.5 * (kl_pm + kl_qm))


# ===========================================================================
# FUNZIONE 3: compute_k_divergence
# Calcola la K-divergenza di Lin 1991 (NON normalizzata)
# ===========================================================================

def compute_k_divergence(
    p_dist: Dict[str, float],   # distribuzione P
    q_dist: Dict[str, float],   # distribuzione Q
    epsilon: float = 1e-12,     # protezione numerica
) -> float:
    """
    Calcola la K-divergenza di Lin (1991).

    FORMULA MATEMATICA (Lin 1991, eq. 3.1):
        M = 0.5 * (P + Q)
        K(P, Q) = KL(P || M) = sum P(x) * log(P(x) / M(x))

    DIFFERENZA con JS:
        JS(P,Q) = 0.5*KL(P||M) + 0.5*KL(Q||M)   <- simmetrica, usa ENTRAMBI i KL
        K(P,Q)  =     KL(P||M)                    <- asimmetrica, usa SOLO il primo KL

    ATTENZIONE SUL LOGARITMO:
        Lin 1991 usa log2 -> K(P,Q) <= 1 direttamente (equazione 3.13 del paper)
        Questo codice usa np.log = ln (logaritmo naturale)
        Con ln, il bound e K(P,Q) <= ln(2) ~= 0.693 (non piu <= 1)
        Per ottenere la K normalizzata [0,1] come nel paper -> usa compute_k_divergence_normalized

    Args:
        p_dist: Distribuzione P (modello base)
        q_dist: Distribuzione Q (modello fine-tuned)
        epsilon: Protezione numerica

    Returns:
        K-divergenza in [0, ln(2)] con logaritmo naturale
    """
    # Distribuzione vuota -> nessuna divergenza
    if not p_dist or not q_dist:
        return 0.0

    # Unione di tutti i token
    all_tokens = set(p_dist.keys()) | set(q_dist.keys())

    # Estrae probabilita non-negative per ogni token
    p_probs = {t: max(p_dist.get(t, 0.0), 0.0) for t in all_tokens}
    q_probs = {t: max(q_dist.get(t, 0.0), 0.0) for t in all_tokens}

    # Rinormalizza
    p_sum = sum(p_probs.values()) or epsilon
    q_sum = sum(q_probs.values()) or epsilon

    p_norm = {t: p / p_sum for t, p in p_probs.items()}
    q_norm = {t: q / q_sum for t, q in q_probs.items()}

    # Calcola M = punto medio
    m_dist = {t: 0.5 * (p_norm[t] + q_norm[t]) for t in all_tokens}

    # K(P,Q) = KL(P||M) -- solo il primo dei due KL di Jensen-Shannon
    # Questo rende K asimmetrica: K(P,Q) != K(Q,P) in generale
    return float(compute_kl_divergence(p_norm, m_dist, epsilon=epsilon))


# ===========================================================================
# FUNZIONE 4: compute_k_divergence_normalized
# K-divergenza normalizzata in [0, 1] -- EQUIVALENTE a Lin 1991 con log2
# ===========================================================================

def compute_k_divergence_normalized(
    p_dist: Dict[str, float],   # distribuzione P
    q_dist: Dict[str, float],   # distribuzione Q
    epsilon: float = 1e-12,     # protezione numerica
) -> float:
    """
    Calcola la K-divergenza normalizzata in [0, 1].

    FORMULA:
        K_norm(P,Q) = K(P,Q) / ln(2)

    PERCHE dividiamo per ln(2)?
        Con logaritmo naturale (np.log), il bound superiore e K <= ln(2) ~= 0.693
        Dividendo per ln(2) mappiamo l'intervallo [0, ln(2)] -> [0, 1]

    EQUIVALENZA con Lin 1991:
        Lin usa log2 -> K_lin(P,Q) <= 1 direttamente
        Nota: log2(x) = ln(x) / ln(2)
        Quindi K_lin(P,Q) = K_naturale(P,Q) / ln(2) <- QUESTA funzione
        Questo e l'unico modo per confrontare K con i risultati del paper Lin 1991.

    Args:
        p_dist: Distribuzione P
        q_dist: Distribuzione Q
        epsilon: Protezione numerica

    Returns:
        K normalizzata in [0, 1]
    """
    # Calcola K grezza con logaritmo naturale
    k_raw = compute_k_divergence(p_dist, q_dist, epsilon=epsilon)

    # np.log(2.0) = ln(2) ~= 0.6931472...
    # Dividiamo per ottenere l'intervallo [0, 1]
    return float(k_raw / np.log(2.0))


# ===========================================================================
# FUNZIONE 5: compute_js_distance
# Distanza di Jensen-Shannon = sqrt(JS divergence)
# ===========================================================================

def compute_js_distance(
    p_dist: Dict[str, float],   # distribuzione P
    q_dist: Dict[str, float],   # distribuzione Q
    epsilon: float = 1e-12,     # protezione numerica
) -> float:
    """
    Calcola la distanza di Jensen-Shannon: sqrt(JS divergence).

    FORMULA:
        JSD(P,Q) = sqrt( JS(P,Q) )

    PERCHE la radice quadrata?
        JS(P,Q) e una *divergenza*, non una *distanza* matematica.
        sqrt(JS) soddisfa la disuguaglianza triangolare -> e una distanza metrica.
        Intervallo: JSD(P,Q) in [0, sqrt(ln(2))] ~= [0, 0.832] con logaritmo naturale.

    Args:
        p_dist: Distribuzione P
        q_dist: Distribuzione Q
        epsilon: Protezione numerica

    Returns:
        Distanza JS in [0, sqrt(ln(2))]
    """
    # Calcola prima JS divergence, poi ne prende la radice quadrata
    # max(..., 0.0): protezione numerica contro JS negativa per errori float
    return float(np.sqrt(max(compute_js_divergence(p_dist, q_dist, epsilon=epsilon), 0.0)))


# ===========================================================================
# FUNZIONE 6: compute_distribution_drift
# Dispatcher: seleziona la metrica di drift da usare per L1 e L2
# ===========================================================================

def compute_distribution_drift(
    p_dist: Dict[str, float],   # distribuzione P (base)
    q_dist: Dict[str, float],   # distribuzione Q (fine-tuned)
    method: str = "kl",         # metrica da usare (default: KL divergence)
    epsilon: float = 1e-12,     # protezione numerica
) -> float:
    """
    Calcola il drift tra due distribuzioni usando la metrica selezionata.

    RUOLO NEL FRAMEWORK ERA:
        Questa funzione e il punto di ingresso per L1 e L2.
        - In L1 (_analyze_l1): misura il drift sulle probabilita dei target_tokens
        - In L2 (_compute_topk_kl): misura il drift sulle top-50 distribuzioni semantiche

    METRICHE DISPONIBILI:
        "kl"                     -> KL(P||Q) -- asimmetrica, illimitata
        "k" / "k_divergence"     -> K(P,Q) = KL(P||M) -- asimmetrica, <= ln(2)
        "k_divergence_normalized"-> K(P,Q)/ln(2) -- normalizzata [0,1]
        "js" / "js_divergence"   -> JS(P,Q) -- simmetrica, <= ln(2)
        "js_distance"            -> sqrt(JS) -- distanza metrica

    Args:
        p_dist: Distribuzione P (modello base)
        q_dist: Distribuzione Q (modello fine-tuned)
        method: Nome della metrica (case-insensitive)
        epsilon: Protezione numerica

    Returns:
        Valore del drift >= 0

    Raises:
        ValueError: Se method non e riconosciuto
    """
    # Normalizza la stringa method: minuscolo e senza spazi bianchi
    # (permette "KL", "Kl", " kl " come valori equivalenti di input)
    method_norm = method.lower().strip()

    # Seleziona e chiama la funzione corrispondente
    if method_norm == "kl":
        # KL divergence standard
        return compute_kl_divergence(p_dist, q_dist, epsilon=epsilon)
    if method_norm in {"k", "k_divergence", "lin_k_divergence"}:
        # K-divergenza di Lin (logaritmo naturale, range [0, ln(2)])
        return compute_k_divergence(p_dist, q_dist, epsilon=epsilon)
    if method_norm in {"k_divergence_normalized", "k_normalized", "lin_k_divergence_normalized", "k_01"}:
        # K-divergenza normalizzata [0, 1] -- equivalente a Lin 1991 con log2
        return compute_k_divergence_normalized(p_dist, q_dist, epsilon=epsilon)
    if method_norm in {"js", "js_divergence", "jensen_shannon_divergence"}:
        # JS divergence
        return compute_js_divergence(p_dist, q_dist, epsilon=epsilon)
    if method_norm in {"js_distance", "jensen_shannon_distance"}:
        # JS distanza (radice quadrata della JS divergence)
        return compute_js_distance(p_dist, q_dist, epsilon=epsilon)

    # Nessuna metrica riconosciuta -> errore esplicito
    raise ValueError(f"Unknown distribution drift method: {method}")


# ===========================================================================
# FUNZIONE 7: compute_cosine_similarity
# Misura la similarita angolare tra due vettori embedding (per L3)
# ===========================================================================

def compute_cosine_similarity(
    vec_a: Union[np.ndarray, "torch.Tensor"],   # vettore A (numpy o torch)
    vec_b: Union[np.ndarray, "torch.Tensor"],   # vettore B (numpy o torch)
) -> float:
    """
    Calcola la similarita coseno tra due vettori.

    FORMULA MATEMATICA:
        cos(theta) = (A . B) / (||A|| * ||B||)

    dove:
        A . B    = prodotto scalare (dot product)
        ||A||    = norma euclidea di A = sqrt(sum a_i^2)
        theta    = angolo tra i due vettori nello spazio embedding

    INTERVALLO: [-1, 1]
        1.0  = vettori identici (stesso significato)
        0.0  = vettori ortogonali (significato irrelato)
       -1.0  = vettori opposti (significato contrario)

    RUOLO IN L3:
        Per ogni coppia di concept_tokens (A, B) calcoliamo:
            base_cosine  = cos(emb_base(A),   emb_base(B))
            ft_cosine    = cos(emb_ft(A),     emb_ft(B))
            delta_cosine = ft_cosine - base_cosine
        Un delta_cosine negativo indica che il fine-tuning ha
        allontanato i due concetti nello spazio embedding.

    Args:
        vec_a: Primo vettore (embedding del token A)
        vec_b: Secondo vettore (embedding del token B)

    Returns:
        Similarita coseno in [-1, 1]

    Example:
        >>> import numpy as np
        >>> a = np.array([1, 0, 0])
        >>> b = np.array([1, 0, 0])
        >>> cos = compute_cosine_similarity(a, b)
        >>> print(f"Cosine similarity: {cos:.4f}")  # 1.0
    """
    # Converte torch.Tensor in numpy array se PyTorch e disponibile
    # .detach(): stacca il tensore dal computation graph (no gradiente)
    # .cpu(): sposta il tensore dalla GPU alla RAM (se era su CUDA)
    # .numpy(): converte il tensore in array NumPy
    if TORCH_AVAILABLE and isinstance(vec_a, torch.Tensor):
        vec_a = vec_a.detach().cpu().numpy()
    if TORCH_AVAILABLE and isinstance(vec_b, torch.Tensor):
        vec_b = vec_b.detach().cpu().numpy()

    # .flatten(): appiattisce il vettore in 1D se ha piu dimensioni
    # (gli embedding possono avere shape (1, hidden_size) o (hidden_size,))
    vec_a = vec_a.flatten()
    vec_b = vec_b.flatten()

    # np.dot(vec_a, vec_b): prodotto scalare = sum (a_i * b_i)
    dot_product = np.dot(vec_a, vec_b)

    # np.linalg.norm: norma L2 = sqrt(sum x_i^2) = lunghezza del vettore
    norm_a = np.linalg.norm(vec_a)   # ||A||
    norm_b = np.linalg.norm(vec_b)   # ||B||

    # Protezione: se uno dei due vettori e il vettore zero -> similarita indefinita -> 0
    if norm_a == 0 or norm_b == 0:
        return 0.0

    # cos(theta) = (A.B) / (||A||*||B||)
    return float(dot_product / (norm_a * norm_b))


# ===========================================================================
# FUNZIONE 8: compute_euclidean_distance
# Misura la distanza geometrica tra due vettori embedding (alternativa a coseno per L3)
# ===========================================================================

def compute_euclidean_distance(
    vec_a: Union[np.ndarray, "torch.Tensor"],   # vettore A
    vec_b: Union[np.ndarray, "torch.Tensor"],   # vettore B
) -> float:
    """
    Calcola la distanza euclidea tra due vettori.

    FORMULA MATEMATICA:
        d(A, B) = ||A - B|| = sqrt( sum (a_i - b_i)^2 )

    DIFFERENZA con coseno:
        - Coseno: misura l'ANGOLO -> invariante alla scala
        - Euclidea: misura la DISTANZA ASSOLUTA -> sensibile alla magnitudine

    RUOLO IN L3 (alternativa al coseno):
        Se l3_metric="euclidean":
            base_euclidean  = d(emb_base(A), emb_base(B))
            ft_euclidean    = d(emb_ft(A),   emb_ft(B))
            delta_euclidean = ft_euclidean - base_euclidean
        Un delta positivo indica che il fine-tuning ha ALLONTANATO i due concetti.
        Un delta negativo indica che il fine-tuning li ha AVVICINATI.

    Args:
        vec_a: Primo vettore
        vec_b: Secondo vettore

    Returns:
        Distanza euclidea >= 0
    """
    # Conversione da torch.Tensor a numpy (stesso pattern di compute_cosine_similarity)
    if TORCH_AVAILABLE and isinstance(vec_a, torch.Tensor):
        vec_a = vec_a.detach().cpu().numpy()
    if TORCH_AVAILABLE and isinstance(vec_b, torch.Tensor):
        vec_b = vec_b.detach().cpu().numpy()

    # Appiattisce a 1D
    vec_a = vec_a.flatten()
    vec_b = vec_b.flatten()

    # vec_a - vec_b: vettore differenza -- ogni componente e (a_i - b_i)
    # np.linalg.norm(...): norma L2 del vettore differenza = sqrt(sum (a_i - b_i)^2)
    return float(np.linalg.norm(vec_a - vec_b))


# ===========================================================================
# FUNZIONE 9: compute_alignment_score
# Calcola l'ERA Alignment Score: rapporto L2/L3
# ===========================================================================

def compute_alignment_score(
    l2_mean_kl: float,      # media KL di L2 (drift probabilistico -- output)
    l3_mean_delta: float,   # media |Δcoseno| di L3 (drift rappresentazionale -- struttura)
    epsilon: float = 1e-12, # protezione contro divisione per zero
) -> float:
    """
    Calcola l'ERA Alignment Score = L2_mean_KL / L3_mean_|delta|

    FORMULA:
        AlignmentScore = l2_mean_kl / max(l3_mean_delta, epsilon)

    INTERPRETAZIONE:
        Il numeratore (L2) misura quanto e cambiato l'OUTPUT del modello:
            "Quanto sono cambiate le distribuzioni di probabilita sulle parole?"

        Il denominatore (L3) misura quanto e cambiata la STRUTTURA INTERNA:
            "Quanto si sono spostati i concetti nello spazio embedding?"

        RAPPORTO ALTO (score > 1000):
            L2 grande, L3 piccolo -> il modello dice cose molto diverse (output)
            ma non ha modificato la sua rappresentazione interna dei concetti.
            Questo e il "parrot effect": il modello impara a ripetere nuovi pattern
            senza capirli veramente.

        RAPPORTO BASSO (score < 10):
            L2 e L3 sono proporzionati -> il cambiamento nell'output riflette un
            genuino cambiamento nella comprensione dei concetti.
            Questo e il "deep learning": apprendimento autentico.

    SCALA INTERPRETATIVA:
        < 10        : Deep learning (production-ready)
        10 - 100    : Moderate learning (acceptable for research)
        100 - 1,000 : Shallow learning (prototype only)
        1,000 - 10,000: Very shallow ("parrot" effect)
        > 10,000    : Extremely shallow (DO NOT DEPLOY)

    NOTA SULLE SCALE:
        L2 (KL divergence) e tipicamente nell'ordine di 0.01 - 10
        L3 (|delta coseno|) e tipicamente nell'ordine di 0.00001 - 0.01
        Le due metriche hanno scale molto diverse -- il rapporto le confronta
        ma potrebbe essere sensibile a questa differenza di scala.

    Args:
        l2_mean_kl: Media del drift KL da L2 (tutti i contesti)
        l3_mean_delta: Media del |delta| dalle coppie di L3
        epsilon: Evita divisione per zero se L3 = 0

    Returns:
        Alignment score (float positivo)

    Example:
        >>> score = compute_alignment_score(l2_mean_kl=1.29, l3_mean_delta=0.000029)
        >>> print(f"Alignment Score: {score:.0f}")  # ~44,500
    """
    # Protezione: se L3 = 0 (nessun cambio strutturale) usa epsilon
    # Evita divisione per zero e garantisce uno score finito
    if l3_mean_delta == 0:
        l3_mean_delta = epsilon

    # Calcola il rapporto: L2 / max(L3, epsilon)
    # max(..., epsilon): ulteriore protezione contro L3 ~= 0 per errori float
    score = l2_mean_kl / max(l3_mean_delta, epsilon)

    return float(score)


# ===========================================================================
# FUNZIONE 10: interpret_alignment_score
# Converte il valore numerico in un'etichetta qualitativa
# ===========================================================================

def interpret_alignment_score(score: float) -> str:
    """
    Restituisce un'etichetta qualitativa per l'ERA Alignment Score.

    SCALA:
        < 10        : "Deep learning (production-ready)"
        10 - 99     : "Moderate learning (acceptable for research)"
        100 - 999   : "Shallow learning (prototype only)"
        1000 - 9999 : "Very shallow alignment (parrot effect)"
        >= 10000    : "Extremely shallow alignment (DO NOT DEPLOY)"

    Args:
        score: ERA Alignment Score (output di compute_alignment_score)

    Returns:
        Stringa descrittiva
    """
    # Soglie crescenti: prima condizione vera vince (if/elif)
    if score < 10:
        return "Deep learning (production-ready)"
    elif score < 100:
        return "Moderate learning (acceptable for research)"
    elif score < 1000:
        return "Shallow learning (prototype only)"
    elif score < 10000:
        return "Very shallow alignment (parrot effect)"
    else:
        # score >= 10000
        return "Extremely shallow alignment (DO NOT DEPLOY)"


# ===========================================================================
# FUNZIONE 11: compute_statistical_significance
# Test statistico per verificare se le differenze sono significative
# ===========================================================================

def compute_statistical_significance(
    values_a: np.ndarray,   # array di valori del primo gruppo (es: KL del modello base)
    values_b: np.ndarray,   # array di valori del secondo gruppo (es: KL del fine-tuned)
    test: str = "ttest",    # tipo di test: "ttest" o "mannwhitneyu"
) -> Dict[str, float]:
    """
    Calcola la significativita statistica tra due distribuzioni di valori.

    RUOLO:
        Dopo aver calcolato i drift KL per tutti i contesti, possiamo verificare
        se la differenza tra base model e fine-tuned model e statisticamente
        significativa o potrebbe essere dovuta al caso.

    TEST DISPONIBILI:

        "ttest" -> t-test indipendente (scipy.stats.ttest_ind)
            - Ipotesi nulla H0: le due medie sono uguali
            - Assume distribuzione approssimativamente normale
            - Restituisce statistica t e p-value

        "mannwhitneyu" -> Test di Mann-Whitney U (scipy.stats.mannwhitneyu)
            - Test non-parametrico: non assume normalita
            - Ipotesi nulla H0: le due distribuzioni sono identiche
            - Piu robusto per campioni piccoli o distribuzioni non-normali
            - Restituisce statistica U e p-value

    INTERPRETAZIONE del p-value:
        p < 0.05 -> differenza statisticamente significativa al 5%
        p < 0.01 -> differenza statisticamente significativa all'1%
        p >= 0.05 -> differenza non significativa (potrebbe essere casuale)

    Args:
        values_a: Array di valori del gruppo A (es: array di KL per ogni contesto)
        values_b: Array di valori del gruppo B
        test: Tipo di test statistico

    Returns:
        Dizionario con chiavi 'statistic' (float) e 'pvalue' (float)

    Raises:
        ValueError: Se test non e riconosciuto
    """
    # scipy.stats: libreria per statistiche scientifiche
    # Importata qui dentro (lazy import) per non richiedere scipy se non usata
    from scipy import stats

    if test == "ttest":
        # ttest_ind: t-test per due campioni indipendenti
        # Restituisce (statistic, pvalue) dove statistic e la t-statistic
        statistic, pvalue = stats.ttest_ind(values_a, values_b)
    elif test == "mannwhitneyu":
        # mannwhitneyu: test non-parametrico di Mann-Whitney U
        # Restituisce (statistic, pvalue) dove statistic e la U-statistic
        statistic, pvalue = stats.mannwhitneyu(values_a, values_b)
    else:
        # Test non riconosciuto -> errore esplicito
        raise ValueError(f"Unknown test: {test}")

    # Restituisce dizionario con i risultati
    return {
        "statistic": float(statistic),   # valore della statistica del test
        "pvalue": float(pvalue),          # probabilita di osservare questo risultato per caso
    }
