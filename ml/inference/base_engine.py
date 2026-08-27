"""
Interface commune du moteur IA MPOX-Assist.

Toute implémentation (démonstration ou réelle) doit respecter cette
interface, afin que le backend puisse basculer entre les deux modes
sans changement de code applicatif.
"""
from abc import ABC, abstractmethod
from typing import Dict, Tuple
from PIL import Image
import numpy as np


class BaseEngine(ABC):
    """Contrat commun à toute implémentation du moteur IA."""

    @abstractmethod
    def load_model(self) -> None:
        """Charge le modèle en mémoire (poids, config)."""
        raise NotImplementedError

    @abstractmethod
    def predict(self, image: Image.Image) -> Tuple[str, Dict[str, float]]:
        """Retourne (classe_prédite, {classe: probabilité, ...})."""
        raise NotImplementedError

    @abstractmethod
    def get_probabilities(self, image: Image.Image) -> Dict[str, float]:
        raise NotImplementedError

    @abstractmethod
    def get_target_layer(self):
        """Retourne la couche cible utilisée par Grad-CAM (None en mode démo)."""
        raise NotImplementedError

    @abstractmethod
    def generate_gradcam(self, image: Image.Image, predicted_class: str) -> np.ndarray:
        """Retourne une heatmap Grad-CAM (H, W) normalisée entre 0 et 1."""
        raise NotImplementedError
