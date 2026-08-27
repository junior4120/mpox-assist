"""
Prépare le dataset MSLD v2.0 (fold1) : copie les images depuis
data/raw_v2/Original Images/Original Images/FOLDS/fold1/{Train,Valid,Test}/<classe>
vers data/train/, data/validation/, data/test/ avec le mapping de classes
MPOX-Assist.

Usage :
    python ml/dataset/prepare_dataset.py

Mapping retenu (voir discussion) :
    Monkeypox              -> mpox
    Chickenpox              -> varicelle
    Healthy                  -> peau_saine
    Cowpox, HFMD, Measles     -> autres  (regroupées, faute de données Herpès)
"""
import argparse
import shutil
from pathlib import Path

SPLIT_FOLDER_MAP = {
    "Train": "train",
    "Valid": "validation",
    "Test": "test",
}

CLASS_MAPPING = {
    "Monkeypox": "mpox",
    "Chickenpox": "varicelle",
    "Healthy": "peau_saine",
    "Cowpox": "autres",
    "HFMD": "autres",
    "Measles": "autres",
}


def find_images(class_dir: Path):
    exts = {".jpg", ".jpeg", ".png"}
    return [p for p in class_dir.rglob("*") if p.suffix.lower() in exts]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        default="data/raw_v2/Original Images/Original Images/FOLDS/fold1",
        help="Dossier du fold source (contient Train/, Valid/, Test/)",
    )
    parser.add_argument("--dest", default="data", help="Dossier destination")
    args = parser.parse_args()

    source = Path(args.source)
    dest = Path(args.dest)

    if not source.exists():
        raise SystemExit(f"Dossier source introuvable : {source}")

    summary = {}

    for src_split, dest_split in SPLIT_FOLDER_MAP.items():
        split_dir = source / src_split
        if not split_dir.exists():
            print(f"[IGNORÉ] {split_dir} introuvable")
            continue

        for class_folder in split_dir.iterdir():
            if not class_folder.is_dir():
                continue
            target_class = CLASS_MAPPING.get(class_folder.name)
            if target_class is None:
                print(f"[IGNORÉ] Classe non mappée : {class_folder.name}")
                continue

            images = find_images(class_folder)
            out_dir = dest / dest_split / target_class
            out_dir.mkdir(parents=True, exist_ok=True)

            for f in images:
                shutil.copy(f, out_dir / f"{class_folder.name}_{f.name}")

            key = f"{dest_split}/{target_class}"
            summary[key] = summary.get(key, 0) + len(images)
            print(f"{src_split}/{class_folder.name} -> {dest_split}/{target_class} : {len(images)} images")

    print("\n--- Résumé ---")
    for k, v in sorted(summary.items()):
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()