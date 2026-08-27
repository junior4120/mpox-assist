"""
Tests de bout en bout pour l'API MPOX-Assist (mode démonstration).
Lancer avec : pytest tests/ -v (depuis backend/, avec PYTHONPATH configuré)
"""
import io
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("MPOX_AI_MODE", "demo")
os.environ.setdefault("DATABASE_URL", "sqlite:///./test_mpox_assist.db")

from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from app.main import app
from database.db import init_db

init_db()  # s'assure que les tables existent même si l'événement startup n'est pas déclenché
client = TestClient(app)
API_KEY = os.environ.get("MPOX_API_KEY", "poc-demo-key-change-me")


def make_test_image_bytes() -> bytes:
    img = Image.new("RGB", (256, 256), color=(200, 150, 130))
    draw = ImageDraw.Draw(img)
    draw.ellipse((80, 80, 176, 176), fill=(180, 80, 70))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf.read()


def test_health():
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_model_info():
    resp = client.get("/api/v1/model/info")
    assert resp.status_code == 200
    data = resp.json()
    assert data["mode"] == "demo"
    assert "Mpox" in data["classes"]


def test_predict_requires_api_key():
    files = {"file": ("test.jpg", make_test_image_bytes(), "image/jpeg")}
    resp = client.post("/api/v1/predict", files=files)
    assert resp.status_code == 422 or resp.status_code == 401


def test_predict_rejects_bad_extension():
    files = {"file": ("test.txt", b"not an image", "text/plain")}
    resp = client.post(
        "/api/v1/predict", files=files, headers={"X-API-Key": API_KEY}
    )
    assert resp.status_code == 400


def test_predict_full_pipeline():
    files = {"file": ("test.jpg", make_test_image_bytes(), "image/jpeg")}
    resp = client.post(
        "/api/v1/predict",
        files=files,
        data={"patient_reference": "TEST-PYTEST"},
        headers={"X-API-Key": API_KEY},
    )
    assert resp.status_code == 200
    data = resp.json()

    assert data["predicted_class"] in ["Mpox", "Varicelle", "Herpès", "Peau saine"]
    assert 0.0 <= data["confidence"] <= 1.0
    assert abs(sum(data["probabilities"].values()) - 1.0) < 1e-6
    assert "diagnostic médical validé" in data["triage_message"]
    assert "diagnostic confirmé" not in data["triage_message"].lower()
    assert data["ai_mode"] == "demo"

    # Vérifie que le parcours complet a bien enregistré l'analyse
    results = client.get("/api/v1/results").json()
    assert any(r["id"] == data["analysis_id"] for r in results)


def test_statistics_flags_demo_data():
    resp = client.get("/api/v1/statistics")
    assert resp.status_code == 200
    assert resp.json()["is_demo_data"] is True
