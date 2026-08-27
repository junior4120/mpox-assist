

from pathlib import Path
import csv

import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader


# ============================================================
# CONFIGURATION
# ============================================================

TEST_DIR = Path("data/test")

MODEL_PATH = Path(
    "ml/models/mobilenetv3_mpox_best.pth"
)

OUTPUT_DIR = Path("ml/models")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ERRORS_CSV = OUTPUT_DIR / "classification_errors.csv"

IMAGE_SIZE = 224
BATCH_SIZE = 16

# IMPORTANT :
# ImageFolder trie les dossiers alphabétiquement.
CLASS_NAMES = [
    "autres",
    "mpox",
    "peau_saine",
    "varicelle",
]


# ============================================================
# TRANSFORMATION
# ============================================================

test_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("\n========================================")
print("      MPOX-ASSIST — ANALYSE ERREURS")
print("========================================")

print(f"\nDevice : {device}")


# ============================================================
# DATASET
# ============================================================

dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=test_transforms
)

print("\nClasses détectées :")
print(dataset.class_to_idx)

print(f"\nNombre d'images de test : {len(dataset)}")


# Vérification
if dataset.classes != CLASS_NAMES:
    raise ValueError(
        f"\nLes classes ne correspondent pas.\n"
        f"Détectées : {dataset.classes}\n"
        f"Attendues : {CLASS_NAMES}"
    )


loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# MODÈLE
# ============================================================

print("\nChargement du modèle...")

model = models.mobilenet_v3_small(
    weights=None
)

in_features = model.classifier[-1].in_features

model.classifier[-1] = nn.Linear(
    in_features,
    len(CLASS_NAMES)
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(device)

model.eval()

print("✓ Modèle chargé")


# ============================================================
# PRÉDICTIONS
# ============================================================

all_errors = []

total = 0
correct = 0

# Compteurs des erreurs
error_matrix = {
    true_class: {
        predicted_class: 0
        for predicted_class in CLASS_NAMES
    }
    for true_class in CLASS_NAMES
}


with torch.no_grad():

    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        predictions = torch.argmax(
            probabilities,
            dim=1
        )

        confidences = torch.max(
            probabilities,
            dim=1
        ).values

        for i in range(len(labels)):

            true_index = labels[i].item()
            predicted_index = predictions[i].item()

            true_class = CLASS_NAMES[
                true_index
            ]

            predicted_class = CLASS_NAMES[
                predicted_index
            ]

            confidence = confidences[i].item()

            image_path = dataset.samples[
                total
            ][0]

            total += 1

            if true_index == predicted_index:

                correct += 1

            else:

                error_matrix[
                    true_class
                ][
                    predicted_class
                ] += 1

                all_errors.append({
                    "image": image_path,
                    "true_class": true_class,
                    "predicted_class": predicted_class,
                    "confidence": round(
                        confidence,
                        4
                    )
                })


# ============================================================
# RÉSULTATS GÉNÉRAUX
# ============================================================

accuracy = correct / total

print("\n========================================")
print("              RÉSULTATS")
print("========================================")

print(
    f"\nImages analysées : {total}"
)

print(
    f"Images correctement classées : {correct}"
)

print(
    f"Images mal classées : {len(all_errors)}"
)

print(
    f"Accuracy : {accuracy:.4f}"
)


# ============================================================
# MATRICE DES ERREURS
# ============================================================

print("\n========================================")
print("         MATRICE DES ERREURS")
print("========================================")

print(
    "\nVraie classe → Classe prédite"
)

for true_class in CLASS_NAMES:

    print(
        f"\n[{true_class}]"
    )

    for predicted_class in CLASS_NAMES:

        if true_class == predicted_class:
            continue

        count = error_matrix[
            true_class
        ][
            predicted_class
        ]

        if count > 0:

            print(
                f"  → {predicted_class} : {count}"
            )


# ============================================================
# ERREURS MPOX
# ============================================================

print("\n========================================")
print("          ERREURS SUR MPOX")
print("========================================")

mpox_errors = [
    error
    for error in all_errors
    if error["true_class"] == "mpox"
]

if len(mpox_errors) == 0:

    print("\n✓ Aucune erreur Mpox.")

else:

    print(
        f"\nNombre d'erreurs Mpox : "
        f"{len(mpox_errors)}"
    )

    for error in mpox_errors:

        print(
            f"\nImage : {error['image']}"
        )

        print(
            f"Vraie classe : {error['true_class']}"
        )

        print(
            f"Prédiction : {error['predicted_class']}"
        )

        print(
            f"Confiance : "
            f"{error['confidence']:.2%}"
        )


# ============================================================
# ERREURS PAR CLASSE
# ============================================================

print("\n========================================")
print("       ERREURS PAR CLASSE")
print("========================================")

for true_class in CLASS_NAMES:

    total_class = sum(
        1
        for _, label in dataset.samples
        if CLASS_NAMES[label] == true_class
    )

    errors_class = sum(
        error["true_class"] == true_class
        for error in all_errors
    )

    recall = (
        (total_class - errors_class)
        / total_class
        if total_class > 0
        else 0
    )

    print(
        f"{true_class:15s} "
        f"Total={total_class:3d} "
        f"Erreurs={errors_class:3d} "
        f"Recall={recall:.2%}"
    )


# ============================================================
# SAUVEGARDE CSV
# ============================================================

with open(
    ERRORS_CSV,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "image",
            "true_class",
            "predicted_class",
            "confidence"
        ]
    )

    writer.writeheader()

    writer.writerows(
        all_errors
    )


print("\n========================================")
print("             TERMINÉ")
print("========================================")

print(
    f"\nListe des erreurs :"
    f"\n{ERRORS_CSV}"
)