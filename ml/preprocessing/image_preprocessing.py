"""
Prétraitement des images pour l'inférence.

Aucune augmentation aléatoire n'est appliquée ici (conforme à la
section 3.3 du cahier des charges) : uniquement redimensionnement,
normalisation et vérification de format.
"""
from PIL import Image, UnidentifiedImageError

TARGET_SIZE = (224, 224)
ALLOWED_FORMATS = {"JPEG", "PNG"}


class InvalidImageError(Exception):
    pass


def load_and_validate_image(file_bytes: bytes) -> Image.Image:
    import io
    try:
        image = Image.open(io.BytesIO(file_bytes))
        image.verify()
    except (UnidentifiedImageError, OSError) as e:
        raise InvalidImageError(f"Fichier image invalide : {e}")

    # Ré-ouvrir après verify() (verify() ferme le flux interne)
    image = Image.open(io.BytesIO(file_bytes))
    if image.format not in ALLOWED_FORMATS:
        raise InvalidImageError(
            f"Format non supporté ({image.format}). Formats acceptés : {ALLOWED_FORMATS}"
        )
    return image.convert("RGB")


def preprocess_image(image: Image.Image) -> Image.Image:
    """Redimensionnement + conversion, prêt pour l'inférence."""
    resized = image.resize(TARGET_SIZE, Image.BILINEAR)
    return resized
