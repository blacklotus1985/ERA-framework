from era import ERAAnalyzer, HuggingFaceWrapper

# Carichiamo il modello di base e quello finetunato (qui sono uguali per test)
base = HuggingFaceWrapper.from_pretrained("EleutherAI/gpt-neo-125M", device="cpu")
tuned = HuggingFaceWrapper.from_pretrained("EleutherAI/gpt-neo-125M", device="cpu")

# Contesti, target e concetti per l'analisi
prompts = ["The doctor said that the patient"]
target_tokens = ["he", "she"]
concept_tokens = ["doctor", "engineer", "nurse"]

# Inizializziamo l'analizzatore
analyzer = ERAAnalyzer(
    base_model=base,
    finetuned_model=tuned,
    device="cpu"
)

# Eseguiamo l'analisi
results = analyzer.analyze(
    test_contexts=prompts,
    target_tokens=target_tokens,
    concept_tokens=concept_tokens,
    topk_semantic=10
)

# Stampiamo i risultati
print(results.summary)