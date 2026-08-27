"""
Service de triage.

RÈGLE ABSOLUE : ne jamais formuler un message comme un diagnostic
confirmé. Toujours utiliser des formulations de type "cas suspect",
"résultat compatible avec...", "évaluation clinique recommandée".
"""

TRIAGE_MESSAGES = {
    "Mpox": (
        "Résultat compatible avec une suspicion de Mpox. "
        "Une évaluation clinique par un professionnel de santé est recommandée."
    ),
    "Varicelle": (
        "Résultat compatible avec une suspicion de varicelle. "
        "Une évaluation clinique est recommandée pour confirmation."
    ),
    "Herpès": (
        "Résultat compatible avec une suspicion d'herpès. "
        "Une évaluation clinique est recommandée pour confirmation."
    ),
    "Peau saine": (
        "Aucune lésion suspecte détectée par l'outil. "
        "En cas de doute clinique, une consultation reste recommandée."
    ),
}

DISCLAIMER = (
    "Ce résultat est produit par un prototype de recherche et ne constitue "
    "pas un diagnostic médical validé."
)


def build_triage_message(predicted_class: str, confidence: float) -> str:
    base = TRIAGE_MESSAGES.get(
        predicted_class,
        "Cas à évaluer : résultat non catégorisé avec certitude.",
    )
    if confidence < 0.5:
        base += " Le niveau de confiance étant faible, cette estimation doit être interprétée avec prudence."
    return f"{base} {DISCLAIMER}"
