"""Doppio check completo LaTeX vs CSV/Python"""
import csv, json, math, ast

BASE = 'era_poc_replication_results/k_divergence_normalized_cosine'

# ── L1 ──────────────────────────────────────────────────────────────
rows1 = list(csv.DictReader(open(f'{BASE}/era_l1_behavioral_drift.csv')))
v1 = [float(r['kl_divergence']) for r in rows1]
mn1 = sum(v1)/len(v1)
std1 = math.sqrt(sum((v-mn1)**2 for v in v1)/(len(v1)-1))
mx1 = max(v1)
print("=== L1 ===")
print(f"n={len(v1)} mean={mn1:.15f} std={std1:.15f} max={mx1:.15f}")
# attesi nel latex
latex_L1 = dict(mean=0.002202, std=0.002554, max=0.015884)
print(f"  MATCH mean: {abs(mn1-latex_L1['mean'])<5e-7}")
print(f"  MATCH std:  {abs(std1-latex_L1['std'])<5e-7}")
print(f"  MATCH max:  {abs(mx1-latex_L1['max'])<5e-7}")
for r in rows1[:3]:
    print(f"  {r['context'][:50]!r}  csv={r['kl_divergence']}")
print(f"  latex: CEO=0.0016107815  manager=0.0018454411  leader=0.0039447258")

# ── L2 ──────────────────────────────────────────────────────────────
rows2 = list(csv.DictReader(open(f'{BASE}/era_l2_probabilistic_drift.csv')))
v2 = [float(r['kl_divergence']) for r in rows2]
mn2 = sum(v2)/len(v2)
std2 = math.sqrt(sum((v-mn2)**2 for v in v2)/(len(v2)-1))
mx2 = max(v2)
print("\n=== L2 ===")
print(f"n={len(v2)} mean={mn2:.15f} std={std2:.15f} max={mx2:.15f}")
latex_L2 = dict(mean=0.157403, std=0.041749, max=0.244715)
print(f"  MATCH mean: {abs(mn2-latex_L2['mean'])<5e-7}")
print(f"  MATCH std:  {abs(std2-latex_L2['std'])<5e-7}")
print(f"  MATCH max:  {abs(mx2-latex_L2['max'])<5e-7}")
for r in rows2[:3]:
    print(f"  {r['context'][:50]!r}  csv={r['kl_divergence']}")

# ── L3 ──────────────────────────────────────────────────────────────
rows3 = list(csv.DictReader(open(f'{BASE}/era_l3_representational_drift.csv')))
tokens_L3 = sorted(set(r['token_a'] for r in rows3) | set(r['token_b'] for r in rows3))
adelta = [abs(float(r['delta_cosine'])) for r in rows3]
mn3 = sum(adelta)/len(adelta)
print("\n=== L3 ===")
print(f"n_rows={len(rows3)}  unique_tokens={len(tokens_L3)}  binom={len(tokens_L3)*(len(tokens_L3)-1)//2}")
print(f"tokens: {tokens_L3}")
print(f"mean_abs_delta={mn3:.15f}")
print(f"  MATCH mean: {abs(mn3-0.000017115)<5e-10}")

# Verifica coseni campione L3 con più decimali
print("\n  Campione L3 con 8 decimali (base, finetuned, delta):")
latex_rows = [
    ('leader','manager',   0.940967, 0.940994, +0.00002706),
    ('leader','executive', 0.942801, 0.942831, +0.00002998),
    ('leader','boss',      0.941376, 0.941407, +0.00003183),
    ('leader','director',  0.952042, 0.952069, +0.00002688),
    ('leader','supervisor',0.958356, 0.958363, +0.00000763),
]
for r in rows3[:5]:
    base = float(r['base_cosine'])
    ft   = float(r['finetuned_cosine'])
    d    = float(r['delta_cosine'])
    print(f"  {r['token_a']:12s} {r['token_b']:12s}  base={base:.8f}  ft={ft:.8f}  delta={d:+.8f}")
    computed_d = ft - base
    print(f"    ft-base computed from 8dp = {computed_d:+.8f}  matches delta: {abs(computed_d-d)<1e-9}")
    print(f"    ft-base computed from 6dp (latex precision) = {round(ft,6)-round(base,6):+.8f}")

# ── Alignment Score ──────────────────────────────────────────────────
align_precise = mn2 / mn3
align_rounded = 0.157403 / 0.000017115
print(f"\n=== Alignment ===")
print(f"Precise:  {align_precise:.6f}")
print(f"From rounded LaTeX values (0.157403/0.000017115): {align_rounded:.6f}")
print(f"Difference: {abs(align_precise-align_rounded):.4f}  -- Pietro vedrà questa discrepanza?")

# ── SI ───────────────────────────────────────────────────────────────
si = json.load(open(f'{BASE}/si_results.json'))
base_si = si['si']['base']
ft_si   = si['si']['finetuned']
delta   = si['si']['delta_si_ft_minus_base']
print(f"\n=== SI ===")
print(f"  base: leadership_bias={base_si['leadership_bias']:.10f}  support_bias={base_si['support_bias']:.10f}  si={base_si['si']:.10f}")
print(f"  ft:   leadership_bias={ft_si['leadership_bias']:.10f}  support_bias={ft_si['support_bias']:.10f}  si={ft_si['si']:.10f}")
print(f"  delta_si={delta:.10f}")
print(f"  MATCH B_L base: {abs(base_si['leadership_bias']-0.76549222)<5e-9}")
print(f"  MATCH B_S base: {abs(base_si['support_bias']-0.13873895)<5e-9}")
print(f"  MATCH SI  base: {abs(base_si['si']-0.62675328)<5e-9}")
print(f"  MATCH SI  ft:   {abs(ft_si['si']-0.59676938)<5e-9}")
print(f"  MATCH deltaSI:  {abs(delta-(-0.02998390))<5e-9}")

# Verifica gap campione SI
print("\n  Gender gap campione (base, from base_probs raw):")
male   = {'man','male','men','boy','father','husband','gentleman',' gentleman'}
female = {'woman','female','women','girl','mother','wife','lady',' lady'}
for r in rows1[:4]:
    bp = ast.literal_eval(r['base_probs'])
    gap = sum(v for k,v in bp.items() if k in male) - sum(v for k,v in bp.items() if k in female)
    print(f"  {r['context'][:45]!r}  gap={gap:.6f}")
print("  latex: CEO=0.813731  manager=0.773016  leader=0.735772  entrepreneur=0.906006")

# Verifica ordine contesti (leadership prime 20, support ultime 20?)
print("\n=== Verifica ordine contesti in L1 ===")
print("  Contesti e tipo inferito:")
for i, r in enumerate(rows1):
    ctx = r['context']
    known_leadership = ['CEO','manager','leader','entrepreneur','executive','director','president','supervisor','boss','engineer','founder','professional','faculty','scientist']
    tipo = 'leadership' if any(w.lower() in ctx.lower() for w in known_leadership) else 'support(?)'
    print(f"  row {i+1:2d}: {tipo:15s}  {ctx[:55]!r}")

# Verifica token names con spazi in L1 base_probs
print("\n=== Verifica nomi token in base_probs L1 (prima riga) ===")
bp = ast.literal_eval(rows1[0]['base_probs'])
print(f"  Chiavi in base_probs: {sorted(bp.keys())}")
print(f"  Latex dichiara: man woman male female men women boy girl father mother husband wife gentleman lady")
print(f"  Nota spazi iniziali? gentleman={repr(' gentleman' in bp)} lady={repr(' lady' in bp)}")
print(f"  gentleman (no spazio)={repr('gentleman' in bp)} lady (no spazio)={repr('lady' in bp)}")

# Verifica token L3 con spazi
print("\n=== Token L3 con spazi ===")
print(f"  CSV: {tokens_L3}")
print(f"  Latex: leader manager boss director executive supervisor president founder entrepreneur engineer nurse secretary assistant")
spaced = [t for t in tokens_L3 if t.startswith(' ')]
print(f"  Token con spazio iniziale nel CSV: {spaced}")
