"""
verify_data.py — MPOX-Assist

Rapport diagnostic sur le dataset AVANT tout entraînement/comparaison.
N'entraîne rien, ne modifie rien : lecture seule.

Vérifie :
  - nombre d'images par classe et par split
  - images corrompues / illisibles
  - doublons exacts (hash SHA-256 du contenu)
  - fuites entre splits (même fichier présent dans train/val/test)
  - quasi-doublons (perceptual hash, optionnel si `imagehash` est installé)

Usage :
    python ml/training/verify_data.py
    python ml/training/verify_data.py --data-dir data --output report.json
"""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image

try:
    import imagehash
    HAS_IMAGEHASH = True
except ImportError:
    HAS_IMAGEHASH = False


SPLITS = ["train", "validation", "test"]
CLASS_NAMES = ["autres", "mpox", "peau_saine", "varicelle"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Distance de Hamming max entre deux phash pour les considérer "quasi-identiques".
# 0 = identique visuellement. 5 est un seuil courant, raisonnablement strict.
PHASH_THRESHOLD = 5


# ============================================================
# UTILITAIRES
# ============================================================

def list_images(folder: Path):
    if not folder.exists():
        return []
    return sorted(
        p for p in folder.rglob("*")
        if p.suffix.lower() in IMAGE_EXTENSIONS
    )


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def check_image_valid(path: Path):
    """Retourne (True, None) si l'image s'ouvre correctement, sinon (False, message)."""
    try:
        with Image.open(path) as img:
            img.verify()
        # verify() invalide l'objet, on rouvre pour un vrai load (détecte plus d'erreurs)
        with Image.open(path) as img:
            img.load()
        return True, None
    except Exception as e:
        return False, str(e)


def compute_phash(path: Path):
    try:
        with Image.open(path) as img:
            return imagehash.phash(img)
    except Exception:
        return None


# ============================================================
# RAPPORT PRINCIPAL
# ============================================================

def build_report(data_dir: Path):

    report = {
        "data_dir": str(data_dir),
        "counts": {},           # counts[split][class] = n
        "totals_per_split": {},
        "totals_per_class": defaultdict(int),
        "corrupted_images": [],
        "exact_duplicates_within_split": [],
        "leakage_across_splits": [],
        "near_duplicates": [],
        "near_duplicate_check_enabled": HAS_IMAGEHASH,
        "class_folder_mismatches": [],
    }

    # hash -> liste de (split, classe, chemin)
    hash_index = defaultdict(list)
    # pour near-duplicates : phash -> liste de (split, classe, chemin)
    phash_index = []

    for split in SPLITS:
        split_dir = data_dir / split
        report["counts"][split] = {}
        split_total = 0

        if not split_dir.exists():
            report["counts"][split]["__missing_folder__"] = True
            continue

        found_classes = sorted(
            p.name for p in split_dir.iterdir() if p.is_dir()
        )
        unexpected = set(found_classes) - set(CLASS_NAMES)
        missing = set(CLASS_NAMES) - set(found_classes)
        if unexpected or missing:
            report["class_folder_mismatches"].append({
                "split": split,
                "unexpected_folders": sorted(unexpected),
                "missing_class_folders": sorted(missing),
            })

        for class_name in CLASS_NAMES:
            class_dir = split_dir / class_name
            images = list_images(class_dir)
            report["counts"][split][class_name] = len(images)
            report["totals_per_class"][class_name] += len(images)
            split_total += len(images)

            for img_path in images:
                valid, err = check_image_valid(img_path)
                if not valid:
                    report["corrupted_images"].append({
                        "path": str(img_path),
                        "split": split,
                        "class": class_name,
                        "error": err,
                    })
                    continue  # on ne hash pas une image corrompue

                file_hash = sha256_of_file(img_path)
                hash_index[file_hash].append((split, class_name, str(img_path)))

                if HAS_IMAGEHASH:
                    ph = compute_phash(img_path)
                    if ph is not None:
                        phash_index.append((split, class_name, str(img_path), ph))

        report["totals_per_split"][split] = split_total

    report["totals_per_class"] = dict(report["totals_per_class"])
    report["grand_total"] = sum(report["totals_per_split"].values())

    # -----------------------------------------
    # Doublons exacts : même hash, même split -> doublon interne
    # même hash, splits différents -> FUITE (plus grave)
    # -----------------------------------------
    for file_hash, occurrences in hash_index.items():
        if len(occurrences) < 2:
            continue

        splits_involved = {o[0] for o in occurrences}

        if len(splits_involved) > 1:
            report["leakage_across_splits"].append({
                "hash": file_hash,
                "occurrences": [
                    {"split": s, "class": c, "path": p} for s, c, p in occurrences
                ],
            })
        else:
            report["exact_duplicates_within_split"].append({
                "hash": file_hash,
                "occurrences": [
                    {"split": s, "class": c, "path": p} for s, c, p in occurrences
                ],
            })

    # -----------------------------------------
    # Quasi-doublons (perceptual hash) — comparaison naïve O(n^2), acceptable
    # pour quelques centaines/milliers d'images. À optimiser (LSH) si le
    # dataset grossit beaucoup.
    # -----------------------------------------
    if HAS_IMAGEHASH:
        n = len(phash_index)
        for i in range(n):
            split_i, class_i, path_i, hash_i = phash_index[i]
            for j in range(i + 1, n):
                split_j, class_j, path_j, hash_j = phash_index[j]
                distance = hash_i - hash_j  # distance de Hamming
                if distance <= PHASH_THRESHOLD:
                    report["near_duplicates"].append({
                        "distance": int(distance),
                        "image_a": {"split": split_i, "class": class_i, "path": path_i},
                        "image_b": {"split": split_j, "class": class_j, "path": path_j},
                        "cross_split": split_i != split_j,
                    })

    return report


# ============================================================
# AFFICHAGE CONSOLE
# ============================================================

def print_summary(report):

    print("\n================================")
    print("   RAPPORT DE VÉRIFICATION DES DONNÉES")
    print("================================\n")

    print(f"Dossier analysé : {report['data_dir']}")
    print(f"Total d'images  : {report['grand_total']}\n")

    if report["class_folder_mismatches"]:
        print("⚠ Incohérences de dossiers de classes détectées :")
        for m in report["class_folder_mismatches"]:
            print(f"  - split '{m['split']}' : inattendus={m['unexpected_folders']} "
                  f"manquants={m['missing_class_folders']}")
        print()

    print("Répartition par split / classe :")
    header = f"{'Split':<12}" + "".join(f"{c:>14}" for c in CLASS_NAMES) + f"{'Total':>10}"
    print(header)
    print("-" * len(header))
    for split in SPLITS:
        counts = report["counts"].get(split, {})
        row = f"{split:<12}"
        for c in CLASS_NAMES:
            row += f"{counts.get(c, 0):>14}"
        row += f"{report['totals_per_split'].get(split, 0):>10}"
        print(row)

    print("\nTotal par classe (tous splits confondus) :")
    for c in CLASS_NAMES:
        print(f"  {c:<12} : {report['totals_per_class'].get(c, 0)}")

    print(f"\nImages corrompues/illisibles : {len(report['corrupted_images'])}")
    for item in report["corrupted_images"][:10]:
        print(f"  - {item['path']} ({item['split']}/{item['class']}) → {item['error']}")
    if len(report["corrupted_images"]) > 10:
        print(f"  ... et {len(report['corrupted_images']) - 10} de plus (voir le JSON)")

    print(f"\nDoublons exacts DANS un même split : {len(report['exact_duplicates_within_split'])}")

    n_leaks = len(report["leakage_across_splits"])
    if n_leaks > 0:
        print(f"\n🚨 FUITES ENTRE SPLITS DÉTECTÉES : {n_leaks} fichier(s) identique(s) "
              f"présents dans plusieurs splits (train/val/test).")
        print("   → Ceci invalide potentiellement les métriques de test actuelles.")
        for leak in report["leakage_across_splits"][:5]:
            print(f"   - {[o['split'] + '/' + o['class'] for o in leak['occurrences']]}")
    else:
        print("\n✓ Aucune fuite exacte détectée entre splits.")

    if report["near_duplicate_check_enabled"]:
        cross = [d for d in report["near_duplicates"] if d["cross_split"]]
        same = [d for d in report["near_duplicates"] if not d["cross_split"]]
        print(f"\nQuasi-doublons (perceptual hash, seuil={PHASH_THRESHOLD}) :")
        print(f"  - dans le même split : {len(same)}")
        print(f"  - ENTRE splits (à vérifier visuellement, risque de fuite douce) : {len(cross)}")
    else:
        print("\n(Vérification des quasi-doublons désactivée — installe 'imagehash' "
              "avec `pip install imagehash` pour l'activer.)")

    print("\n================================\n")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Vérification du dataset MPOX-Assist")
    parser.add_argument("--data-dir", type=str, default="data",
                         help="Dossier racine contenant train/validation/test")
    parser.add_argument("--output", type=str, default="ml/models/data_verification_report.json",
                         help="Chemin de sortie du rapport JSON")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    report = build_report(data_dir)
    print_summary(report)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"Rapport complet sauvegardé dans : {output_path}")


if __name__ == "__main__":
    main()