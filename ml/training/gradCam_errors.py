from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

from PIL import Image, ImageDraw, ImageFont

from torchvision import models, transforms

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = Path("ml/models/mobilenetv3_mpox_best.pth")

OUTPUT_DIR = Path("ml/models/gradcam_errors")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = [
    "autres",
    "mpox",
    "peau_saine",
    "varicelle",
]

IMAGE_SIZE = 224

ERROR_IMAGES = [
    Path("data/test/mpox/Monkeypox_MKP_111_01.jpg"),
    Path("data/test/mpox/Monkeypox_MKP_28_03.jpg"),
    Path("data/test/mpox/Monkeypox_MKP_71_01.jpg"),
    Path("data/test/mpox/Monkeypox_MKP_71_02.jpg"),
    Path("data/test/mpox/Monkeypox_MKP_71_03.jpg"),
]


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("\n========================================")
print("       MPOX-ASSIST — GRAD-CAM")
print("========================================")

print(f"\nDevice : {device}")


# ============================================================
# TRANSFORMATION
# ============================================================

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


# ============================================================
# MODÈLE
# ============================================================

print("\nChargement du modèle...")

model = models.mobilenet_v3_small(weights=None)

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


# Dernière couche convolutionnelle
target_layer = model.features[-1]

print(f"\nCouche Grad-CAM : {target_layer}")


# ============================================================
# FONCTION PRINCIPALE
# ============================================================

def generate_gradcam(image_path):

    print("\n----------------------------------------")
    print(f"Image : {image_path}")

    if not image_path.exists():
        print(f"⚠ Image introuvable : {image_path}")
        return

    # --------------------------------------------------------
    # IMAGE
    # --------------------------------------------------------

    original = Image.open(image_path).convert("RGB")

    original_resized = original.resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    )

    rgb_image = (
        np.array(original_resized)
        .astype(np.float32)
        / 255.0
    )

    # --------------------------------------------------------
    # PRÉTRAITEMENT
    # --------------------------------------------------------

    input_tensor = transform(
        original
    ).unsqueeze(0).to(device)

    # --------------------------------------------------------
    # PRÉDICTION
    # --------------------------------------------------------

    with torch.no_grad():

        output = model(input_tensor)

        probabilities = torch.softmax(
            output,
            dim=1
        )

        predicted_index = torch.argmax(
            probabilities,
            dim=1
        ).item()

        confidence = probabilities[
            0,
            predicted_index
        ].item()

    predicted_class = CLASS_NAMES[
        predicted_index
    ]

    true_class = "mpox"

    print(f"Vraie classe : {true_class}")
    print(f"Prédiction    : {predicted_class}")
    print(f"Confiance     : {confidence:.2%}")

    # --------------------------------------------------------
    # GRAD-CAM
    # --------------------------------------------------------

    cam = GradCAM(
        model=model,
        target_layers=[target_layer]
    )

    targets = [
        ClassifierOutputTarget(
            predicted_index
        )
    ]

    grayscale_cam = cam(
        input_tensor=input_tensor,
        targets=targets
    )[0]

    visualization = show_cam_on_image(
        rgb_image,
        grayscale_cam,
        use_rgb=True
    )

    # --------------------------------------------------------
    # CONVERSION DES IMAGES
    # --------------------------------------------------------

    original_np = np.array(
        original_resized
    )

    cam_rgb = visualization

    original_pil = Image.fromarray(
        original_np
    )

    cam_pil = Image.fromarray(
        cam_rgb
    )

    # --------------------------------------------------------
    # CRÉATION DU PANNEAU
    # --------------------------------------------------------

    panel_width = IMAGE_SIZE * 2
    panel_height = IMAGE_SIZE + 120

    panel = Image.new(
        "RGB",
        (panel_width, panel_height),
        "white"
    )

    panel.paste(
        original_pil,
        (0, 0)
    )

    panel.paste(
        cam_pil,
        (IMAGE_SIZE, 0)
    )

    draw = ImageDraw.Draw(panel)

    # --------------------------------------------------------
    # TITRES
    # --------------------------------------------------------

    draw.text(
        (10, IMAGE_SIZE + 10),
        "Image originale",
        fill="black"
    )

    draw.text(
        (IMAGE_SIZE + 10, IMAGE_SIZE + 10),
        "Grad-CAM",
        fill="black"
    )

    # --------------------------------------------------------
    # INFORMATIONS
    # --------------------------------------------------------

    draw.text(
        (10, IMAGE_SIZE + 40),
        f"Vraie classe : {true_class}",
        fill="black"
    )

    draw.text(
        (IMAGE_SIZE + 10, IMAGE_SIZE + 40),
        f"Prédiction : {predicted_class}",
        fill="black"
    )

    draw.text(
        (10, IMAGE_SIZE + 70),
        f"Confiance : {confidence:.2%}",
        fill="black"
    )

    if predicted_class == true_class:

        result_text = "✓ Correct"

    else:

        result_text = "✗ Erreur de classification"

    draw.text(
        (IMAGE_SIZE + 10, IMAGE_SIZE + 70),
        result_text,
        fill="black"
    )

    # --------------------------------------------------------
    # SAUVEGARDE
    # --------------------------------------------------------

    output_path = (
        OUTPUT_DIR
        / f"{image_path.stem}_comparison.jpg"
    )

    panel.save(
        output_path,
        quality=95
    )

    print(
        f"✓ Comparaison sauvegardée :"
    )

    print(
        f"  {output_path}"
    )


# ============================================================
# EXÉCUTION
# ============================================================

for image_path in ERROR_IMAGES:

    generate_gradcam(image_path)


print("\n========================================")
print("             TERMINÉ")
print("========================================")

print(
    f"\nRésultats disponibles dans :"
)

print(
    OUTPUT_DIR
)