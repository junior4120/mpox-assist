"""
Utilitaires de visualisation Grad-CAM : conversion d'une heatmap
normalisée (0-1) en image colorée, et superposition sur l'image
originale.
"""
import numpy as np
from PIL import Image
import matplotlib.cm as cm


def heatmap_to_rgb(heatmap: np.ndarray) -> Image.Image:
    """Convertit une heatmap (H, W) en image RGB colorée (colormap jet)."""
    colormap = cm.get_cmap("jet")
    colored = colormap(heatmap)[:, :, :3]  # supprime le canal alpha
    colored = (colored * 255).astype(np.uint8)
    return Image.fromarray(colored)


def overlay_heatmap(original: Image.Image, heatmap: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """Superpose la heatmap colorée sur l'image originale."""
    original_resized = original.convert("RGB").resize((heatmap.shape[1], heatmap.shape[0]))
    heatmap_rgb = heatmap_to_rgb(heatmap)

    original_arr = np.asarray(original_resized).astype(np.float32)
    heatmap_arr = np.asarray(heatmap_rgb).astype(np.float32)

    blended = (1 - alpha) * original_arr + alpha * heatmap_arr
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    return Image.fromarray(blended)


def save_gradcam_outputs(original: Image.Image, heatmap: np.ndarray, output_dir, base_name: str) -> dict:
    """
    Sauvegarde image originale (redimensionnée), heatmap seule et
    superposition. Retourne les chemins relatifs générés.
    """
    from pathlib import Path
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    heatmap_img = heatmap_to_rgb(heatmap)
    overlay_img = overlay_heatmap(original, heatmap)

    heatmap_path = output_dir / f"{base_name}_heatmap.png"
    overlay_path = output_dir / f"{base_name}_overlay.png"

    heatmap_img.save(heatmap_path)
    overlay_img.save(overlay_path)

    return {
        "heatmap_path": str(heatmap_path),
        "overlay_path": str(overlay_path),
    }
