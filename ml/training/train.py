import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    recall_score,
)
import matplotlib.pyplot as plt

# timm est nécessaire pour Xception (absent de torchvision)
# pip install timm
import timm


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = Path("data")

TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "validation"
TEST_DIR = DATA_DIR / "test"

OUTPUT_DIR = Path("ml/models")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 224
BATCH_SIZE = 16
NUM_EPOCHS = 15
LEARNING_RATE = 1e-4

SEED = 42

CLASS_NAMES = [
    "autres",
    "mpox",
    "peau_saine",
    "varicelle",
]

# Liste des architectures comparées.
# Ajoute/retire des clés ici pour changer le scope de la comparaison.
ARCHITECTURES = [
    "mobilenetv3",
    "densenet121",
    "xception",
]


# ============================================================
# REPRODUCTIBILITÉ
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# DEVICE
# ============================================================

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


# ============================================================
# TRANSFORMATIONS
# ============================================================
# Xception est nativement entraîné en 299x299 dans la plupart des poids
# ImageNet, mais fonctionne aussi en 224x224 sans erreur (pooling adaptatif).
# On garde IMAGE_SIZE unique pour une comparaison strictement équitable
# entre les 3 architectures (mêmes données, même résolution).

train_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15,
        saturation=0.15
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


val_test_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


# ============================================================
# DATASET
# ============================================================

def create_datasets():

    train_dataset = datasets.ImageFolder(
        TRAIN_DIR,
        transform=train_transforms
    )

    val_dataset = datasets.ImageFolder(
        VAL_DIR,
        transform=val_test_transforms
    )

    test_dataset = datasets.ImageFolder(
        TEST_DIR,
        transform=val_test_transforms
    )

    print("\nClasses détectées par PyTorch :")
    print(train_dataset.class_to_idx)

    print("\nNombre d'images :")
    print(f"Train : {len(train_dataset)}")
    print(f"Validation : {len(val_dataset)}")
    print(f"Test : {len(test_dataset)}")

    return train_dataset, val_dataset, test_dataset


# ============================================================
# DATALOADERS
# ============================================================

def create_dataloaders(
    train_dataset,
    val_dataset,
    test_dataset
):

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader, test_loader


# ============================================================
# MODÈLE — FACTORY MULTI-ARCHITECTURES
# ============================================================
# Chaque architecture est créée pré-entraînée ImageNet, backbone gelé,
# seule la tête de classification est ré-entraînée (même logique que
# le MobileNetV3 d'origine) pour que la comparaison soit équitable.

def create_model(arch, num_classes):

    if arch == "mobilenetv3":

        weights = models.MobileNet_V3_Small_Weights.DEFAULT

        model = models.mobilenet_v3_small(
            weights=weights
        )

        for param in model.features.parameters():
            param.requires_grad = False

        in_features = model.classifier[-1].in_features

        model.classifier[-1] = nn.Linear(
            in_features,
            num_classes
        )

    elif arch == "densenet121":

        weights = models.DenseNet121_Weights.DEFAULT

        model = models.densenet121(
            weights=weights
        )

        for param in model.features.parameters():
            param.requires_grad = False

        in_features = model.classifier.in_features

        model.classifier = nn.Linear(
            in_features,
            num_classes
        )

    elif arch == "xception":

        # torchvision n'a pas Xception : on passe par timm.
        model = timm.create_model(
            "xception",
            pretrained=True,
            num_classes=num_classes
        )

        # On gèle tout le backbone, puis on dégèle uniquement
        # la tête de classification finale ("fc" chez timm.Xception).
        for param in model.parameters():
            param.requires_grad = False

        for param in model.get_classifier().parameters():
            param.requires_grad = True

    else:
        raise ValueError(f"Architecture inconnue : {arch}")

    return model


def get_model_paths(arch):
    """Chemins de sortie (modèle, métriques, matrice) propres à chaque arch."""

    return {
        "model": OUTPUT_DIR / f"{arch}_mpox_best.pth",
        "metrics": OUTPUT_DIR / f"{arch}_metrics.json",
        "confusion_matrix": OUTPUT_DIR / f"{arch}_confusion_matrix.png",
    }


# ============================================================
# ÉVALUATION
# ============================================================

def evaluate(model, loader, criterion, device):

    model.eval()

    total_loss = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            total_loss += loss.item()

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

    avg_loss = total_loss / len(loader)

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    macro_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )

    macro_recall = recall_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )

    report = classification_report(
        all_labels,
        all_predictions,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0
    )

    mpox_recall = report["mpox"]["recall"]

    return (
        avg_loss,
        accuracy,
        macro_f1,
        macro_recall,
        mpox_recall,
        all_labels,
        all_predictions,
        report
    )


# ============================================================
# MATRICE DE CONFUSION
# ============================================================

def save_confusion_matrix(labels, predictions, arch, save_path):

    cm = confusion_matrix(
        labels,
        predictions
    )

    plt.figure(figsize=(7, 6))

    plt.imshow(cm)

    plt.title(
        f"Matrice de confusion - {arch} - MPOX-Assist"
    )

    plt.colorbar()

    plt.xticks(
        range(len(CLASS_NAMES)),
        CLASS_NAMES,
        rotation=45
    )

    plt.yticks(
        range(len(CLASS_NAMES)),
        CLASS_NAMES
    )

    plt.xlabel("Prédiction")
    plt.ylabel("Vérité")

    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            plt.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center"
            )

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=200
    )

    plt.close()


# ============================================================
# ENTRAÎNEMENT D'UNE ARCHITECTURE
# ============================================================

def train_one_architecture(arch, train_loader, val_loader, test_loader, device):

    print("\n================================")
    print(f"   ENTRAÎNEMENT : {arch.upper()}")
    print("================================")

    paths = get_model_paths(arch)

    model = create_model(
        arch=arch,
        num_classes=len(CLASS_NAMES)
    )

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        filter(
            lambda p: p.requires_grad,
            model.parameters()
        ),
        lr=LEARNING_RATE
    )

    best_mpox_recall = -1

    history = []

    for epoch in range(NUM_EPOCHS):

        model.train()

        running_loss = 0

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        train_loss = (
            running_loss /
            len(train_loader)
        )

        (
            val_loss,
            val_accuracy,
            val_f1,
            val_recall,
            val_mpox_recall,
            _,
            _,
            _
        ) = evaluate(
            model,
            val_loader,
            criterion,
            device
        )

        print(f"\n[{arch}] Epoch [{epoch + 1}/{NUM_EPOCHS}]")
        print(f"Train Loss      : {train_loss:.4f}")
        print(f"Val Loss        : {val_loss:.4f}")
        print(f"Val Accuracy    : {val_accuracy:.4f}")
        print(f"Val F1          : {val_f1:.4f}")
        print(f"Val Recall      : {val_recall:.4f}")
        print(f"Val Recall Mpox : {val_mpox_recall:.4f}")

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "val_f1": val_f1,
            "val_recall": val_recall,
            "val_mpox_recall": val_mpox_recall
        })

        if val_mpox_recall > best_mpox_recall:

            best_mpox_recall = val_mpox_recall

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "arch": arch,
                    "class_names": CLASS_NAMES,
                    "image_size": IMAGE_SIZE
                },
                paths["model"]
            )

            print(f"✓ [{arch}] Meilleur modèle sauvegardé.")

    # ----------------------------
    # Test final pour cette architecture
    # ----------------------------

    print(f"\n--- TEST FINAL : {arch} ---")

    checkpoint = torch.load(
        paths["model"],
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    (
        test_loss,
        test_accuracy,
        test_f1,
        test_recall,
        test_mpox_recall,
        labels,
        predictions,
        report
    ) = evaluate(
        model,
        test_loader,
        criterion,
        device
    )

    print(f"Test Accuracy    : {test_accuracy:.4f}")
    print(f"Test F1          : {test_f1:.4f}")
    print(f"Test Recall      : {test_recall:.4f}")
    print(f"Test Recall Mpox : {test_mpox_recall:.4f}")

    print("\nClassification Report :")
    print(
        classification_report(
            labels,
            predictions,
            target_names=CLASS_NAMES,
            zero_division=0
        )
    )

    save_confusion_matrix(
        labels,
        predictions,
        arch,
        paths["confusion_matrix"]
    )

    results = {
        "model": arch,
        "classes": CLASS_NAMES,
        "image_size": IMAGE_SIZE,
        "batch_size": BATCH_SIZE,
        "epochs": NUM_EPOCHS,
        "test_accuracy": test_accuracy,
        "test_f1_macro": test_f1,
        "test_recall_macro": test_recall,
        "test_recall_mpox": test_mpox_recall,
        "classification_report": report,
        "history": history
    }

    with open(
        paths["metrics"],
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            results,
            f,
            indent=4,
            ensure_ascii=False
        )

    print(f"\nModèle    : {paths['model']}")
    print(f"Métriques : {paths['metrics']}")
    print(f"Matrice   : {paths['confusion_matrix']}")

    return results


# ============================================================
# COMPARAISON DES ARCHITECTURES
# ============================================================

def save_comparison(all_results):

    comparison_path = OUTPUT_DIR / "comparison_metrics.json"
    comparison_plot_path = OUTPUT_DIR / "comparison_metrics.png"

    summary = [
        {
            "model": r["model"],
            "test_accuracy": r["test_accuracy"],
            "test_f1_macro": r["test_f1_macro"],
            "test_recall_macro": r["test_recall_macro"],
            "test_recall_mpox": r["test_recall_mpox"],
        }
        for r in all_results
    ]

    with open(
        comparison_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            summary,
            f,
            indent=4,
            ensure_ascii=False
        )

    # Graphique comparatif en barres groupées
    metrics_to_plot = [
        "test_accuracy",
        "test_f1_macro",
        "test_recall_macro",
        "test_recall_mpox",
    ]

    labels_fr = [
        "Accuracy",
        "F1 (macro)",
        "Recall (macro)",
        "Recall Mpox",
    ]

    x = np.arange(len(metrics_to_plot))
    width = 0.8 / max(len(summary), 1)

    plt.figure(figsize=(9, 6))

    for i, row in enumerate(summary):
        values = [row[m] for m in metrics_to_plot]
        plt.bar(x + i * width, values, width, label=row["model"])

    plt.xticks(
        x + width * (len(summary) - 1) / 2,
        labels_fr
    )

    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("Comparaison des architectures - MPOX-Assist")
    plt.legend()
    plt.tight_layout()

    plt.savefig(comparison_plot_path, dpi=200)
    plt.close()

    print("\n================================")
    print("     COMPARAISON FINALE")
    print("================================\n")

    header = f"{'Modèle':<15}{'Accuracy':>10}{'F1 macro':>10}{'Recall':>10}{'Recall Mpox':>13}"
    print(header)
    print("-" * len(header))

    for row in summary:
        print(
            f"{row['model']:<15}"
            f"{row['test_accuracy']:>10.4f}"
            f"{row['test_f1_macro']:>10.4f}"
            f"{row['test_recall_macro']:>10.4f}"
            f"{row['test_recall_mpox']:>13.4f}"
        )

    print(f"\nRésumé JSON : {comparison_path}")
    print(f"Graphique   : {comparison_plot_path}")


# ============================================================
# MAIN
# ============================================================

def train(architectures):

    set_seed(SEED)

    device = get_device()

    print("\n================================")
    print("      MPOX-ASSIST TRAINING")
    print("================================")

    print(f"\nDevice : {device}")
    print(f"Architectures à comparer : {architectures}")

    train_dataset, val_dataset, test_dataset = create_datasets()

    if train_dataset.classes != CLASS_NAMES:

        print("\nATTENTION !")
        print("Classes détectées :")
        print(train_dataset.classes)

        print("\nClasses attendues :")
        print(CLASS_NAMES)

        raise ValueError(
            "L'ordre des classes ne correspond pas à CLASS_NAMES."
        )

    train_loader, val_loader, test_loader = create_dataloaders(
        train_dataset,
        val_dataset,
        test_dataset
    )

    all_results = []

    for arch in architectures:

        results = train_one_architecture(
            arch,
            train_loader,
            val_loader,
            test_loader,
            device
        )

        all_results.append(results)

    if len(all_results) > 1:
        save_comparison(all_results)

    print("\n================================")
    print("         TERMINÉ")
    print("================================")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Entraînement et comparaison de modèles pour MPOX-Assist"
    )

    parser.add_argument(
        "--arch",
        nargs="+",
        choices=ARCHITECTURES,
        default=ARCHITECTURES,
        help=(
            "Architecture(s) à entraîner. Par défaut, entraîne et compare "
            f"les trois : {ARCHITECTURES}. "
            "Exemple : --arch mobilenetv3 densenet121"
        ),
    )

    args = parser.parse_args()

    train(args.arch)