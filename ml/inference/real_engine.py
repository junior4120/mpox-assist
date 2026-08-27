"""
Moteur IA — MODE MODÈLE RÉEL.

Charge le MobileNetV3 entraîné (transfer learning) et calcule un vrai
Grad-CAM par hooks sur la dernière couche convolutionnelle, sans
bibliothèque externe.
"""
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image

from ml.inference.base_engine import BaseEngine
from app.config import REAL_MODEL_PATH

# Mapping des noms de dossiers (ordre ImageFolder) vers les noms affichés dans l'app.
FOLDER_TO_DISPLAY = {
    "mpox": "Mpox",
    "varicelle": "Varicelle",
    "peau_saine": "Peau saine",
    "autres": "Autres affections cutanées",
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class RealEngine(BaseEngine):
    def __init__(self, model_path: str = REAL_MODEL_PATH):
        self.model_path = Path(model_path)
        self.model = None
        self.class_names = None       # ex. ["autres", "mpox", "peau_saine", "varicelle"]
        self.display_names = None     # mêmes indices, noms affichés
        self.target_layer = None
        self._activations = None
        self._gradients = None
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def load_model(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Aucun modèle réel trouvé à {self.model_path}. "
                "Entraînez-le via ml/training/train.py, ou utilisez AI_MODE=demo."
            )
        checkpoint = torch.load(self.model_path, map_location="cpu", weights_only=False)
        self.class_names = checkpoint["class_names"]
        self.display_names = [FOLDER_TO_DISPLAY.get(c, c) for c in self.class_names]

        model = models.mobilenet_v3_small(weights=None)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = torch.nn.Linear(in_features, len(self.class_names))
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        self.model = model
        self.target_layer = model.features[-1]
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, inp, output):
            self._activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self._gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def _preprocess(self, image: Image.Image) -> torch.Tensor:
        return self.transform(image.convert("RGB")).unsqueeze(0)

    def get_probabilities(self, image: Image.Image) -> Dict[str, float]:
        tensor = self._preprocess(image)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1)[0]
        return {self.display_names[i]: float(probs[i]) for i in range(len(self.display_names))}

    def predict(self, image: Image.Image) -> Tuple[str, Dict[str, float]]:
        probs = self.get_probabilities(image)
        predicted_class = max(probs, key=probs.get)
        return predicted_class, probs

    def get_target_layer(self):
        return self.target_layer

    def generate_gradcam(self, image: Image.Image, predicted_class: str) -> np.ndarray:
        tensor = self._preprocess(image)
        tensor.requires_grad_(True)

        self.model.zero_grad()
        logits = self.model(tensor)
        class_idx = self.display_names.index(predicted_class)
        score = logits[0, class_idx]
        score.backward()

        activations = self._activations[0]  # (C, H, W)
        gradients = self._gradients[0]       # (C, H, W)

        weights = gradients.mean(dim=(1, 2))  # (C,)
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = F.relu(cam)
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        cam = cam.unsqueeze(0).unsqueeze(0)
        cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)
        return cam.squeeze().detach().numpy()