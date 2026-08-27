# Données — MPOX-Assist

## ⚠️ État actuel

Aucun dataset médical réel n'est inclus dans ce dépôt. Le projet fonctionne
en **mode démonstration** (`MPOX_AI_MODE=demo`) tant qu'un dataset
approprié n'a pas été identifié et validé.

## Où placer les images

```
data/
├── raw/           # images brutes, non triées
├── processed/     # images nettoyées et redimensionnées
├── train/
│   ├── mpox/
│   ├── varicelle/
│   ├── herpes/
│   └── peau_saine/
├── validation/    # même structure que train/
└── test/          # même structure que train/
```

## Format attendu

- JPEG ou PNG
- RGB
- Taille recommandée avant redimensionnement automatique : ≥ 224×224 px

## Provenance et licence

**Avant d'ajouter toute image à ce dossier**, documenter ici :

- la source exacte (ex. dataset public, institution, consentement) ;
- la licence associée (ex. CC-BY, licence de recherche spécifique) ;
- toute restriction d'usage (recherche uniquement, non commercial, etc.).

Ne jamais committer de données dont la licence ou le consentement
n'est pas clairement établi.

## Précautions

- Aucune donnée personnelle identifiable (nom, visage, numéro de dossier
  patient) ne doit être présente dans les images ou noms de fichiers.
- Utiliser un identifiant anonymisé (`patient_reference`) pour tout
  suivi, jamais une donnée réelle.
- Ce dossier est exclu de Git via `.gitignore` (seul ce README est versionné).
