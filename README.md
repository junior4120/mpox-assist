# MPOX-Assist

**Solution intelligente d'aide au triage précoce des lésions suspectes de Mpox**
*IA légère, explicable et potentiellement embarquée sur smartphone pour les
structures de soins primaires.*

> ⚠️ **Ceci est un prototype de recherche.** Il ne constitue **pas** un
> dispositif médical validé et ne remplace **pas** le jugement d'un
> professionnel de santé. Toute sortie du système utilise des formulations
> prudentes ("cas suspect", "résultat compatible avec...") et jamais
> "diagnostic confirmé".

## État d'avancement (Phase 1 — MVP)

| Composant | Statut |
|---|---|
| Backend FastAPI | ✅ Fonctionnel (mode démonstration) |
| Base de données (SQLite dev / PostgreSQL prod) | ✅ Fonctionnel |
| Moteur IA — mode démonstration | ✅ Fonctionnel |
| Grad-CAM (simulation basée sur l'image réelle) | ✅ Fonctionnel |
| Triage prudent | ✅ Fonctionnel |
| Tests API (pytest) | ✅ 6/6 passent |
| Interface Web (React) | ⏳ Prochaine étape |
| Application mobile (Flutter / simulation web) | ⏳ À venir |
| Docker / docker-compose | ⏳ À venir |
| Modèle IA réel (MobileNetV3 entraîné) | ⏳ Phase 2 — nécessite un dataset validé |
| Meta-Learning | ⏳ Phase 3 — recherche |

## Installation rapide (backend)

```bash
cd backend
pip install -r requirements.txt --break-system-packages   # ou dans un venv
cp ../.env.example ../.env
```

## Lancement

```bash
cd backend
export PYTHONPATH="$(pwd):$(pwd)/.."
uvicorn app.main:app --reload --port 8000
```

Documentation interactive : http://localhost:8000/docs

## Mode démonstration vs mode réel

- `MPOX_AI_MODE=demo` (par défaut) : le parcours complet (classification,
  probabilités, Grad-CAM) est reproduit fidèlement, mais **aucun résultat
  n'est une prédiction médicale réelle** — c'est explicitement indiqué
  dans chaque réponse API (`ai_mode: "demo"`, `is_demo_data: true`).
- `MPOX_AI_MODE=real` : nécessite un modèle entraîné exporté dans
  `ml/export/` (voir `ml/inference/real_engine.py`, non encore implémenté
  — Phase 2).

## Tests

```bash
export PYTHONPATH="$(pwd)/backend:$(pwd)"
python -m pytest tests/ -v
```

## Architecture

Voir le schéma complet dans le cahier des charges du projet. Résumé :

```
Mobile / Web → API FastAPI → Moteur IA (démo ou réel) → Grad-CAM
                    ↓
               PostgreSQL / SQLite
                    ↓
            Dashboard de supervision
```

## Limitations actuelles

- Aucun modèle entraîné sur données médicales réelles n'est utilisé —
  mode démonstration uniquement.
- Aucune valeur de performance (accuracy, recall, F1, taille modèle,
  temps d'inférence mesuré sur device) n'est publiée tant qu'elle n'a
  pas été réellement mesurée.
- Interface Web, application mobile et Docker restent à construire
  (prochaines étapes du plan de développement).

## Perspectives (Phase 2 et 3)

- Entraînement d'un modèle réel (MobileNetV3 → EfficientNet) sur dataset
  validé et documenté (`data/README.md`).
- Export ONNX/TFLite avec mesures réelles de taille et temps d'inférence.
- Exploration Meta-Learning (Prototypical Networks) vs Transfer Learning
  en contexte de faible disponibilité de données.
