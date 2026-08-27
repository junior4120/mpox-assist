"""
Configuration centrale de MPOX-Assist backend.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

BASE_DIR = Path(__file__).resolve().parent.parent

# --- Mode du moteur IA ---
# "demo"  -> résultats simulés, parcours complet reproduit, AUCUNE valeur inventée
#            présentée comme mesurée.
# "real"  -> chargement d'un vrai modèle entraîné (.pt/.onnx/.tflite/.keras/.h5)
AI_MODE = os.getenv("MPOX_AI_MODE", "demo")

# --- Classes du modèle ---
CLASSES = ["Mpox", "Varicelle", "Peau saine", "Autres affections cutanées"]

# --- Chemin du modèle réel (utilisé seulement si AI_MODE == "real") ---
REAL_MODEL_PATH = os.getenv("MPOX_MODEL_PATH", str(BASE_DIR.parent / "ml" / "export" / "model.pt"))

# --- Chemin des métriques du modèle réel (utilisé seulement si AI_MODE == "real") ---
REAL_METRICS_PATH = os.getenv("MPOX_METRICS_PATH", str(BASE_DIR.parent.parent / "ml" / "models" / "metrics.json"))

# --- Base de données ---
# SQLite en dev local (zéro config), PostgreSQL via docker-compose en Phase 2.
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    f"sqlite:///{BASE_DIR / 'mpox_assist.db'}",
)

# --- Sécurité (auth simple pour le POC, pas un système multi-utilisateurs complet) ---
API_KEY = os.getenv("MPOX_API_KEY", "poc-demo-key-change-me")

# --- Upload ---
MAX_IMAGE_SIZE_MB = 8
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# --- Version du modèle affichée (mise à jour uniquement si un modèle réel est entraîné) ---
MODEL_NAME = "MobileNetV3"
MODEL_VERSION = "v0.1-demo"