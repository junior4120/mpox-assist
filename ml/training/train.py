
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


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = Path("data")

TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "validation"
TEST_DIR = DATA_DIR / "test"

OUTPUT_DIR = Path("ml/models")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = OUTPUT_DIR / "mobilenetv3_mpox_best.pth"
METRICS_PATH = OUTPUT_DIR / "metrics.json"
CONFUSION_MATRIX_PATH = OUTPUT_DIR / "confusion_matrix.png"

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
# MODÈLE
# ============================================================

def create_model(num_classes):

    weights = models.MobileNet_V3_Small_Weights.DEFAULT

    model = models.mobilenet_v3_small(
        weights=weights
    )

    # On gèle d'abord le backbone
    for param in model.features.parameters():
        param.requires_grad = False

    # Remplacement du classifieur
    in_features = model.classifier[-1].in_features

    model.classifier[-1] = nn.Linear(
        in_features,
        num_classes
    )

    return model


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

    # Recall spécifique à Mpox
    mpox_index = CLASS_NAMES.index("mpox")

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

def save_confusion_matrix(labels, predictions):

    cm = confusion_matrix(
        labels,
        predictions
    )

    plt.figure(figsize=(7, 6))

    plt.imshow(cm)

    plt.title(
        "Matrice de confusion - MPOX-Assist"
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
        CONFUSION_MATRIX_PATH,
        dpi=200
    )

    plt.close()


# ============================================================
# ENTRAÎNEMENT
# ============================================================

def train():

    set_seed(SEED)

    device = get_device()

    print("\n================================")
    print("      MPOX-ASSIST TRAINING")
    print("================================")

    print(f"\nDevice : {device}")

    # ----------------------------
    # Dataset
    # ----------------------------

    train_dataset, val_dataset, test_dataset = create_datasets()

    # Vérification des classes
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

    # ----------------------------
    # Modèle
    # ----------------------------

    model = create_model(
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

    # ----------------------------
    # Training
    # ----------------------------

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

        print(
            f"\nEpoch [{epoch + 1}/{NUM_EPOCHS}]"
        )

        print(
            f"Train Loss      : {train_loss:.4f}"
        )

        print(
            f"Val Loss        : {val_loss:.4f}"
        )

        print(
            f"Val Accuracy    : {val_accuracy:.4f}"
        )

        print(
            f"Val F1          : {val_f1:.4f}"
        )

        print(
            f"Val Recall      : {val_recall:.4f}"
        )

        print(
            f"Val Recall Mpox : {val_mpox_recall:.4f}"
        )

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "val_f1": val_f1,
            "val_recall": val_recall,
            "val_mpox_recall": val_mpox_recall
        })

        # ----------------------------
        # Sauvegarde meilleur modèle
        # ----------------------------

        if val_mpox_recall > best_mpox_recall:

            best_mpox_recall = val_mpox_recall

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": CLASS_NAMES,
                    "image_size": IMAGE_SIZE
                },
                MODEL_PATH
            )

            print(
                "✓ Meilleur modèle sauvegardé."
            )

    # ========================================================
    # TEST FINAL
    # ========================================================

    print("\n================================")
    print("        TEST FINAL")
    print("================================")

    checkpoint = torch.load(
        MODEL_PATH,
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

    print(
        f"\nTest Accuracy    : {test_accuracy:.4f}"
    )

    print(
        f"Test F1          : {test_f1:.4f}"
    )

    print(
        f"Test Recall      : {test_recall:.4f}"
    )

    print(
        f"Test Recall Mpox : {test_mpox_recall:.4f}"
    )

    print("\nClassification Report :")

    print(
        classification_report(
            labels,
            predictions,
            target_names=CLASS_NAMES,
            zero_division=0
        )
    )

    # ----------------------------
    # Matrice confusion
    # ----------------------------

    save_confusion_matrix(
        labels,
        predictions
    )

    # ----------------------------
    # Sauvegarde métriques
    # ----------------------------

    results = {
        "model": "MobileNetV3-Small",
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
        METRICS_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            results,
            f,
            indent=4,
            ensure_ascii=False
        )

    print("\n================================")
    print("         TERMINÉ")
    print("================================")

    print(
        f"\nModèle : {MODEL_PATH}"
    )

    print(
        f"Métriques : {METRICS_PATH}"
    )

    print(
        f"Matrice : {CONFUSION_MATRIX_PATH}"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    train()