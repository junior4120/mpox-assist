"""
Moteur IA — MODE DÉMONSTRATION.

RÈGLE ÉTHIQUE FONDAMENTALE :
Ce moteur ne fait AUCUNE prédiction médicale réelle. Il reproduit
fidèlement le parcours technique complet (prétraitement, "inférence",
probabilités, Grad-CAM) pour permettre de démontrer et tester
l'application de bout en bout, en attendant qu'un modèle entraîné sur
un dataset médical approprié soit disponible (voir ml/training/).

Les probabilités sont dérivées de manière déterministe et reproductible
à partir du contenu réel de l'image (hash + statistiques de couleur),
afin que la démonstration soit cohérente d'un essai à l'autre pour une
même image, SANS jamais être présentée comme un résultat médical.
"""
import hashlib
import time
from typing import Dict, Tuple

import numpy as np
from PIL import Image, ImageFilter

from ml.inference.base_engine import BaseEngine
from app.config import CLASSES


class DemoEngine(BaseEngine):
    def __init__(self):
        self._loaded = False

    def load_model(self) -> None:
        # Rien à charger : mode démonstration.
        self._loaded = True

    def _seed_from_image(self, image: Image.Image) -> int:
        """Dérive une graine reproductible à partir du contenu de l'image."""
        small = image.convert("RGB").resize((32, 32))
        data = np.asarray(small).tobytes()
        digest = hashlib.sha256(data).hexdigest()
        return int(digest[:8], 16)

    def get_probabilities(self, image: Image.Image) -> Dict[str, float]:
        seed = self._seed_from_image(image)
        rng = np.random.default_rng(seed)
        raw = rng.dirichlet(alpha=[3, 1.5, 1.5, 1.2])  # biais léger vers la 1ère classe pour un résultat démonstratif net
        probs = {cls: float(p) for cls, p in zip(CLASSES, raw)}
        return probs

    def predict(self, image: Image.Image) -> Tuple[str, Dict[str, float]]:
        if not self._loaded:
            self.load_model()
        probs = self.get_probabilities(image)
        predicted_class = max(probs, key=probs.get)
        return predicted_class, probs

    def get_target_layer(self):
        # Pas de couche cible en mode démo : pas de vrai réseau chargé.
        return None

    def generate_gradcam(self, image: Image.Image, predicted_class: str) -> np.ndarray:
        """
        Génère une carte de "saillance" de démonstration à partir de
        l'image réelle (détection de contours + lissage), afin que la
        superposition affichée corresponde visuellement à l'image
        fournie plutôt qu'à un bruit purement aléatoire.

        Ceci N'EST PAS un vrai Grad-CAM (aucun gradient de réseau
        n'est calculé) et l'interface l'indique explicitement.
        """
        gray = image.convert("L").resize((224, 224))
        edges = gray.filter(ImageFilter.FIND_EDGES)
        edges = edges.filter(ImageFilter.GaussianBlur(radius=6))
        arr = np.asarray(edges).astype(np.float32)
        arr -= arr.min()
        if arr.max() > 0:
            arr /= arr.max()
        return arr  # (224, 224), valeurs entre 0 et 1


def measure_inference_time(engine: BaseEngine, image: Image.Image) -> float:
    """Mesure réelle (non inventée) du temps d'inférence en millisecondes."""
    start = time.perf_counter()
    engine.predict(image)
    end = time.perf_counter()
    return (end - start) * 1000
